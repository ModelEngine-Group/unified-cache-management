#!/usr/bin/env python3
"""Capture UCM/vLLM Prometheus metrics around one known request window.

Standard library only. It saves the complete and filtered Prometheus snapshots
plus a focused delta report, so answering "did this request touch the backend?"
does not require reading hundreds of series.

Usage:

    # interactive: send the request yourself, press Enter when it is done
    python3 scripts/collect_ucm_metrics.py --label cache-posix

    # scripted: wait a fixed time instead of prompting
    python3 scripts/collect_ucm_metrics.py --label cache-posix --wait-seconds 3

    # also dump every series change, not just the focused families
    python3 scripts/collect_ucm_metrics.py --label cache-posix --full
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

# Metric families worth keeping in the filtered snapshot. Keep this broad: the
# focused report is parsed from the unfiltered text, but the *_filtered.prom
# files are what a human greps, and anything missing here is invisible there.
INTERESTING_TOKENS = (
    "cache_lookup",
    "cache_load",
    "cache_dump",
    "cache_shard",
    "cache_h2d",
    "cache_d2h",
    "cache_backend",
    "cache_posix",
    "posix_",
    "connector_",
    "layerwise_",
)

# Counters that answer "what did the request actually do". A cache hit that
# still reports cache_load_backend_shards_total > 0 was served from posix.
FOCUS_COUNTERS = (
    "cache_load_shards_total",
    "cache_load_success_shards_total",
    "cache_load_backend_shards_total",
    "cache_load_wait_shards_total",
    "cache_load_failed_shards_total",
    "cache_posix_load_success_shards_total",
    "cache_lookup_hit_blocks_total",
    "cache_lookup_miss_blocks_total",
    "posix_lookup_query_blocks_total",
    "posix_lookup_hit_blocks_total",
    "posix_handle_cache_hit_total",
    "posix_handle_cache_miss_total",
    "posix_handle_cache_evict_total",
    "posix_handle_cache_bypass_total",
    "posix_healthy_count_total",
    "posix_unhealthy_count_total",
)

# Histograms that answer "where did the time go", in milliseconds: the whole
# cache task first, then its decomposition, then the posix store side.
FOCUS_HISTOGRAMS = (
    "cache_lookup_duration_ms",
    "cache_load_duration_ms",
    "cache_dump_duration_ms",
    "cache_lookup_backend_duration_ms",
    "cache_load_queue_wait_duration_ms",
    "cache_shard_backend_wait_ms",
    "cache_load_backend_submit_duration_ms",
    "cache_h2d_submit_ms",
    "cache_h2d_sync_ms",
    "cache_dump_queue_wait_duration_ms",
    "cache_dump_mkbuf_duration_ms",
    "cache_d2h_duration_ms",
    "cache_dump_backend_submit_duration_ms",
    "cache_dump_backend_wait_duration_ms",
    "posix_load_task_duration_ms",
    "posix_load_queue_wait_duration_ms",
)

# Gauges: a delta over the window is meaningless, so the report prints the
# value at the end of the window (and the start, when it moved).
FOCUS_GAUGES = (
    "posix_store_health",
    "posix_store_usage_ratio",
    "posix_store_used_bytes",
    "posix_store_capacity_bytes",
    "posix_gc_running",
    "cache_load_bandwidth_gbps",
    "cache_dump_bandwidth_gbps",
)

MAX_OTHER_COUNTERS = 40

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
        description="Capture /metrics around one known UCM cache request.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        default=1.0,
        help=(
            "Seconds to wait after the request before the after-snapshot. UCM "
            "metrics are published by a periodic loop, so a short settle is "
            "needed; keep it small so periodic work (health probe ~10-20s, GC "
            "sample ~30s) cannot leak into the window. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=None,
        help=(
            "Scripted mode: sleep this long after the before-snapshot instead "
            "of waiting for Enter. Send the request in another terminal while "
            "this sleeps."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP timeout for each snapshot (default: %(default)s)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also write every per-series non-zero delta, not just the focus lists",
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
            if is_interesting_metric(metric_name) and not metric_name.endswith(
                "_bucket"
            ):
                lines.append(line)
            continue
        match = SAMPLE_RE.match(line)
        if match and is_interesting_metric(match.group("name")):
            if match.group("name").endswith("_bucket"):
                continue
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
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.6g}"
    return f"{value:.3f}"


def matching_metric(name: str, suffix: str) -> bool:
    return name == f"ucm:{suffix}" or name.endswith(f":{suffix}")


def counter_delta(
    counter: str,
    before: dict[str, Sample],
    after: dict[str, Sample],
) -> float:
    """Sum a counter's delta across every labelled series (all workers)."""
    total = 0.0
    for key, after_sample in after.items():
        if not matching_metric(after_sample.name, counter):
            continue
        before_value = before.get(key).value if key in before else 0.0
        delta = finite_delta(before_value, after_sample.value)
        if delta is not None:
            total += delta
    return total


def histogram_interval(
    base: str,
    before: dict[str, Sample],
    after: dict[str, Sample],
) -> tuple[float, float] | None:
    """Aggregate a histogram's (count, sum) deltas across all series."""
    total_count = 0.0
    total_sum = 0.0
    seen = False
    for key, count_sample in after.items():
        if not matching_metric(count_sample.name, f"{base}_count"):
            continue
        sum_key = f"ucm:{base}_sum{count_sample.labels}"
        if sum_key not in after:
            # Metrics may be exported under a nested namespace; match on suffix.
            sum_key = next(
                (
                    k
                    for k, s in after.items()
                    if s.labels == count_sample.labels
                    and matching_metric(s.name, f"{base}_sum")
                ),
                None,
            )
        if sum_key is None:
            continue
        before_count = before.get(key).value if key in before else 0.0
        before_sum = before.get(sum_key).value if sum_key in before else 0.0
        count_delta = finite_delta(before_count, count_sample.value)
        sum_delta = finite_delta(before_sum, after[sum_key].value)
        if count_delta is None or sum_delta is None:
            continue
        seen = True
        total_count += count_delta
        total_sum += sum_delta
    return (total_count, total_sum) if seen else None


def gauge_values(name: str, samples: dict[str, Sample]) -> list[Sample]:
    return [s for s in samples.values() if matching_metric(s.name, name)]


def write_delta_report(
    path: Path,
    before: dict[str, Sample],
    after: dict[str, Sample],
    window_seconds: float,
    full: bool,
) -> None:
    report: list[str] = []
    report.append("# UCM metrics delta report")
    report.append("")
    report.append(f"window: {window_seconds:.1f}s between the two snapshots")
    report.append("counters are deltas over the window; histograms are the")
    report.append("interval mean (sum_delta / count_delta) over the same window.")
    report.append("")

    report.append("## 1. Counters: what the request actually did")
    for counter in FOCUS_COUNTERS:
        delta = counter_delta(counter, before, after)
        marker = ""
        if counter == "cache_load_backend_shards_total" and delta > 0:
            marker = "   <-- served by the backend, not by the buffer"
        elif counter == "cache_lookup_miss_blocks_total" and delta > 0:
            marker = "   <-- buffer did not hold every queried block"
        report.append(f"{counter:<44} {format_number(delta)}{marker}")

    report.append("")
    report.append("## 2. Histograms: where the time went (ms)")
    for base in FOCUS_HISTOGRAMS:
        stats = histogram_interval(base, before, after)
        if stats is None:
            report.append(f"{base:<44} (no observations)")
            continue
        count_delta, sum_delta = stats
        mean = sum_delta / count_delta if count_delta else 0.0
        report.append(
            f"{base:<44} n={format_number(count_delta)} "
            f"mean={format_number(mean)} total={format_number(sum_delta)}"
        )

    report.append("")
    report.append("## 3. Gauges: state at the end of the window")
    emitted = 0
    for gauge in FOCUS_GAUGES:
        for sample in gauge_values(gauge, after):
            before_value = (
                before.get(sample.key).value if sample.key in before else None
            )
            change = ""
            if before_value is not None and before_value != sample.value:
                change = f" (was {format_number(before_value)})"
            report.append(f"{sample.key} {format_number(sample.value)}{change}")
            emitted += 1
    if emitted == 0:
        report.append("(none reported)")

    report.append("")
    if full:
        report.append("## 4. Every per-series non-zero delta (buckets omitted)")
        emitted = 0
        for key in sorted(after):
            sample = after[key]
            if sample.name.endswith("_bucket"):
                continue
            before_value = before.get(key).value if key in before else 0.0
            delta = finite_delta(before_value, sample.value)
            if delta is None or delta == 0:
                continue
            report.append(f"{key} {format_number(delta)}")
            emitted += 1
        if emitted == 0:
            report.append("(no non-zero deltas found)")
    else:
        report.append("## 4. Other counters that moved")
        others: list[tuple[str, float]] = []
        for key in sorted(after):
            sample = after[key]
            if sample.name.endswith(("_bucket", "_count", "_sum")):
                continue
            if any(matching_metric(sample.name, c) for c in FOCUS_COUNTERS):
                continue
            if any(matching_metric(sample.name, g) for g in FOCUS_GAUGES):
                continue
            before_value = before.get(key).value if key in before else 0.0
            delta = finite_delta(before_value, sample.value)
            if delta is None or delta == 0:
                continue
            others.append((key, delta))
        others.sort(key=lambda item: abs(item[1]), reverse=True)
        if not others:
            report.append("(nothing else moved)")
        for key, delta in others[:MAX_OTHER_COUNTERS]:
            report.append(f"{key} {format_number(delta)}")
        if len(others) > MAX_OTHER_COUNTERS:
            report.append(
                f"... {len(others) - MAX_OTHER_COUNTERS} more (use --full for all)"
            )

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


def wait_for_request(args: argparse.Namespace) -> None:
    if args.wait_seconds is None:
        print()
        print("Now send the request in another terminal (one request per run).")
        input("After the request has completed, press Enter here to continue... ")
        return
    print()
    print(
        f"Send the request now; continuing automatically in {args.wait_seconds:g}s..."
    )
    time.sleep(args.wait_seconds)


def main() -> int:
    args = parse_args()
    if args.settle_seconds < 0 or args.timeout_seconds <= 0:
        print("settle time must be >= 0 and timeout must be > 0", file=sys.stderr)
        return 2
    if args.wait_seconds is not None and args.wait_seconds < 0:
        print("wait time must be >= 0", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{sanitize_label(args.label)}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Metrics URL: {args.url}")
    print(f"Output directory: {run_dir}")
    try:
        before_ts = time.time()
        before_text, before = save_snapshot(
            run_dir, "before", args.url, args.timeout_seconds
        )
        print_missing_metrics_hint(before_text)

        wait_for_request(args)

        if args.settle_seconds:
            print(f"Waiting {args.settle_seconds:g}s for metrics synchronization...")
            time.sleep(args.settle_seconds)

        after_text, after = save_snapshot(
            run_dir, "after", args.url, args.timeout_seconds
        )
        after_ts = time.time()
        print_missing_metrics_hint(after_text)
    except (RuntimeError, OSError, EOFError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Partial output, if any, is in: {run_dir}", file=sys.stderr)
        return 1

    report_path = run_dir / "delta_report.txt"
    write_delta_report(report_path, before, after, after_ts - before_ts, args.full)
    print(f"Saved delta report: {report_path}")
    print()
    print("Sections 1-3 are the focused view; raw series are in:")
    print(f"  {run_dir / 'before_filtered.prom'}")
    print(f"  {run_dir / 'after_filtered.prom'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
