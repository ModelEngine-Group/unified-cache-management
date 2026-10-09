#!/usr/bin/env python3
"""Capture UCM/vLLM Prometheus metrics before and after one request.

The script only uses the Python standard library. It saves the complete
Prometheus snapshots, filtered UCM snapshots, per-series deltas, and histogram
mean durations for the measurement interval.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


INTERESTING_TOKENS = (
    "cache_lookup",
    "cache_load",
    "cache_posix",
    "cache_h2d",
    "connector_",
    "layerwise_",
)

KEY_COUNTERS = (
    "cache_load_backend_shards_total",
    "cache_posix_load_success_shards_total",
    "cache_load_wait_shards_total",
    "cache_load_success_shards_total",
    "cache_lookup_hit_blocks_total",
    "cache_lookup_miss_blocks_total",
)

SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?P<labels>\{.*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|"
    r"NaN|[+-]Inf)(?:\s+\d+)?$"
)


@dataclass(frozen=True)
class Sample:
    name: str
    labels: str
    value: float

    @property
    def key(self) -> str:
        return f"{self.name}{self.labels}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture /metrics around one known Cache-hit request."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/metrics",
        help="vLLM Prometheus endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--label",
        default="cache-posix",
        help="Short test label used in the output directory name",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/ucm-metrics",
        help="Parent output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait after the request before after-snapshot (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP timeout for each snapshot (default: %(default)s)",
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip()).strip("-.")
    return sanitized or "ucm"


def fetch_metrics(url: str, timeout: float) -> str:
    request = urllib.request.Request(
        url,
        headers={"Accept": "text/plain; version=0.0.4"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"cannot fetch {url}: {exc}") from exc


def is_interesting_metric(name: str) -> bool:
    if not name.startswith("ucm:"):
        return False
    return any(token in name for token in INTERESTING_TOKENS)


def filtered_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("# HELP ucm:") or line.startswith("# TYPE ucm:"):
            metric_name = line.split(maxsplit=3)[2]
            if is_interesting_metric(metric_name):
                lines.append(line)
            continue
        match = SAMPLE_RE.match(line)
        if match and is_interesting_metric(match.group("name")):
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def parse_samples(text: str) -> dict[str, Sample]:
    samples: dict[str, Sample] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if not is_interesting_metric(name):
            continue
        value = float(match.group("value"))
        sample = Sample(name, match.group("labels") or "", value)
        samples[sample.key] = sample
    return samples


def finite_delta(before: float, after: float) -> float | None:
    if not math.isfinite(before) or not math.isfinite(after):
        return None
    return after - before


def format_number(value: float) -> str:
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def matching_metric(name: str, suffix: str) -> bool:
    return name == f"ucm:{suffix}" or name.endswith(f":{suffix}")


def aggregate_counter_delta(
    counter: str,
    before: dict[str, Sample],
    after: dict[str, Sample],
) -> float:
    total = 0.0
    for key, after_sample in after.items():
        if not matching_metric(after_sample.name, counter):
            continue
        before_value = before.get(key).value if key in before else 0.0
        delta = finite_delta(before_value, after_sample.value)
        if delta is not None:
            total += delta
    return total


def write_delta_report(
    path: Path,
    before: dict[str, Sample],
    after: dict[str, Sample],
) -> None:
    report: list[str] = []
    report.append("# UCM metrics delta report")
    report.append("")
    report.append("## Key counter deltas aggregated across workers")
    for counter in KEY_COUNTERS:
        delta = aggregate_counter_delta(counter, before, after)
        report.append(f"ucm:{counter} {format_number(delta)}")

    report.append("")
    report.append("## Per-series non-zero deltas (histogram buckets omitted)")
    emitted = 0
    for key in sorted(after):
        after_sample = after[key]
        if after_sample.name.endswith("_bucket"):
            continue
        before_value = before.get(key).value if key in before else 0.0
        delta = finite_delta(before_value, after_sample.value)
        if delta is None or delta == 0:
            continue
        report.append(f"{key} {format_number(delta)}")
        emitted += 1
    if emitted == 0:
        report.append("(no non-zero deltas found)")

    report.append("")
    report.append("## Histogram interval means by worker (delta_sum / delta_count)")
    emitted = 0
    for key in sorted(after):
        count_sample = after[key]
        if not count_sample.name.endswith("_count"):
            continue
        base_name = count_sample.name[: -len("_count")]
        sum_key = f"{base_name}_sum{count_sample.labels}"
        if sum_key not in after:
            continue
        before_count = before.get(key).value if key in before else 0.0
        before_sum = before.get(sum_key).value if sum_key in before else 0.0
        count_delta = finite_delta(before_count, count_sample.value)
        sum_delta = finite_delta(before_sum, after[sum_key].value)
        if count_delta is None or sum_delta is None or count_delta <= 0:
            continue
        mean = sum_delta / count_delta
        report.append(
            f"{base_name}{count_sample.labels} count_delta={format_number(count_delta)} "
            f"mean={format_number(mean)}"
        )
        emitted += 1
    if emitted == 0:
        report.append("(no histogram observations found in the interval)")

    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def save_snapshot(
    run_dir: Path,
    phase: str,
    url: str,
    timeout: float,
) -> tuple[str, dict[str, Sample]]:
    text = fetch_metrics(url, timeout)
    full_path = run_dir / f"{phase}.prom"
    filtered_path = run_dir / f"{phase}_filtered.prom"
    full_path.write_text(text, encoding="utf-8")
    filtered = filtered_text(text)
    filtered_path.write_text(filtered, encoding="utf-8")
    samples = parse_samples(text)
    print(f"Saved {phase}: {full_path}")
    print(f"Saved filtered {phase}: {filtered_path} ({len(samples)} series)")
    return text, samples


def print_missing_metrics_hint(text: str) -> None:
    if any(line.startswith("ucm:") for line in text.splitlines()):
        return
    print(
        "WARNING: no 'ucm:' metrics were found. Ensure enable_metrics: true, "
        "then send at least one inference request.",
        file=sys.stderr,
    )


def main() -> int:
    args = parse_args()
    if args.settle_seconds < 0 or args.timeout_seconds <= 0:
        print("settle time must be >= 0 and timeout must be > 0", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{sanitize_label(args.label)}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Metrics URL: {args.url}")
    print(f"Output directory: {run_dir}")
    try:
        before_text, before = save_snapshot(
            run_dir, "before", args.url, args.timeout_seconds
        )
        print_missing_metrics_hint(before_text)

        print()
        print("Now send exactly ONE warmed-up 64K Cache-hit request in another terminal.")
        input("After the request has completed, press Enter here to continue... ")

        if args.settle_seconds:
            print(f"Waiting {args.settle_seconds:g}s for metrics synchronization...")
            time.sleep(args.settle_seconds)

        after_text, after = save_snapshot(
            run_dir, "after", args.url, args.timeout_seconds
        )
        print_missing_metrics_hint(after_text)
    except (RuntimeError, OSError, EOFError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Partial output, if any, is in: {run_dir}", file=sys.stderr)
        return 1

    report_path = run_dir / "delta_report.txt"
    write_delta_report(report_path, before, after)
    print(f"Saved delta report: {report_path}")
    print()
    print("Please send these three files for analysis:")
    print(f"  {run_dir / 'before_filtered.prom'}")
    print(f"  {run_dir / 'after_filtered.prom'}")
    print(f"  {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
