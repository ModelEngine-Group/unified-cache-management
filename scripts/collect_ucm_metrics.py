#!/usr/bin/env python3
"""Capture and summarize UCM/vLLM metrics around one request window.

Standard library only. It saves the complete and filtered Prometheus snapshots
plus per-rank Connector, CacheStore, and Posix tables. Histogram averages and
p90/p99 values cover only observations added inside the capture window.

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
import csv
import json
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
    "save_duration",
    "save_completion_wait_duration",
    "load_bytes_total",
    "save_bytes_total",
)

CONNECTOR_METHODS = (
    "get_block_size",
    "get_kv_connector_stats",
    "get_num_new_matched_tokens",
    "update_state_after_alloc",
    "register_kv_caches",
    "build_connector_meta",
    "bind_connector_metadata",
    "handle_preemptions",
    "has_connector_metadata",
    "start_load_kv",
    "wait_for_layer_load",
    "save_kv_layer",
    "wait_for_save",
    "request_finished_all_groups",
    "request_finished",
    "get_finished",
    "build_connector_worker_meta",
    "update_connector_output",
    "clear_connector_metadata",
    "get_block_ids_with_load_errors",
)

CONNECTOR_HISTOGRAMS = (
    ("save_duration", "ms"),
    ("save_completion_wait_duration", "ms"),
    *((f"connector_{method}_duration_ms", "ms") for method in CONNECTOR_METHODS),
)

# Keep the unit alongside each Histogram because bandwidth is a Histogram too,
# not a Gauge. Every result is calculated independently for each worker rank.
CACHESTORE_HISTOGRAMS = (
    ("cache_lookup_duration_ms", "ms"),
    ("cache_lookup_backend_duration_ms", "ms"),
    ("cache_load_duration_ms", "ms"),
    ("cache_load_queue_wait_duration_ms", "ms"),
    ("cache_load_backend_submit_duration_ms", "ms"),
    ("cache_shard_backend_wait_ms", "ms"),
    ("cache_h2d_submit_ms", "ms"),
    ("cache_h2d_sync_ms", "ms"),
    ("cache_dump_duration_ms", "ms"),
    ("cache_dump_queue_wait_duration_ms", "ms"),
    ("cache_dump_prereq_wait_ms", "ms"),
    ("cache_dump_mkbuf_duration_ms", "ms"),
    ("cache_d2h_duration_ms", "ms"),
    ("cache_dump_backend_submit_duration_ms", "ms"),
    ("cache_dump_backend_wait_duration_ms", "ms"),
    ("cache_load_bandwidth_gbps", "GB/s"),
    ("cache_dump_bandwidth_gbps", "GB/s"),
)

POSIX_HISTOGRAMS = (
    ("posix_load_task_duration_ms", "ms"),
    ("posix_load_queue_wait_duration_ms", "ms"),
    ("posix_dump_task_duration_ms", "ms"),
    ("posix_dump_queue_wait_duration_ms", "ms"),
    ("posix_s2h_bandwidth_gbps", "GB/s"),
    ("posix_h2s_bandwidth_gbps", "GB/s"),
)

REPORT_PAGES = (
    ("connector", "Connector", CONNECTOR_HISTOGRAMS),
    ("cachestore", "CacheStore", CACHESTORE_HISTOGRAMS),
    ("posix", "Posix", POSIX_HISTOGRAMS),
)

HISTOGRAM_QUANTILES = (0.9, 0.99)

PAGE_THROUGHPUTS = {
    "connector": (
        ("load_bytes_total", "connector_load_throughput_gbps"),
        ("save_bytes_total", "connector_save_throughput_gbps"),
    ),
    "cachestore": (
        ("cache_load_bytes_total", "cache_load_throughput_gbps"),
        ("cache_dump_bytes_total", "cache_dump_throughput_gbps"),
    ),
    "posix": (
        ("posix_s2h_bytes_total", "posix_load_throughput_gbps"),
        ("posix_h2s_bytes_total", "posix_dump_throughput_gbps"),
    ),
}

SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?P<labels>\{.*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|"
    r"NaN|[+-]Inf)(?:\s+\d+)?$"
)


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    @property
    def key(self) -> str:
        return f"{self.name}{format_labels(self.labels)}"

    def label(self, name: str) -> str | None:
        return dict(self.labels).get(name)


@dataclass(frozen=True)
class HistogramStats:
    count: float
    total: float
    quantiles: dict[float, float]

    @property
    def average(self) -> float | None:
        return self.total / self.count if self.count > 0 else None


@dataclass(frozen=True, order=True)
class RankGroup:
    model_name: str
    engine: str
    worker_rank: str


@dataclass(frozen=True)
class ReportRow:
    group: RankGroup
    metric: str
    count: float | None
    average: float | None
    p90: float | None
    p99: float | None
    unit: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture /metrics and report per-rank UCM distributions.",
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
        help="Append every per-series non-zero delta after the three report pages",
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


def parse_labels(text: str) -> tuple[tuple[str, str], ...]:
    """Parse and canonicalize a Prometheus label set, including escapes."""
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    labels: dict[str, str] = {}
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos] in " ,":
            pos += 1
        if pos >= len(text):
            break
        key_start = pos
        while pos < len(text) and text[pos] != "=":
            pos += 1
        if pos >= len(text):
            break
        key = text[key_start:pos].strip()
        pos += 1
        if pos >= len(text) or text[pos] != '"':
            break
        pos += 1
        value: list[str] = []
        while pos < len(text):
            char = text[pos]
            pos += 1
            if char == "\\" and pos < len(text):
                escaped = text[pos]
                pos += 1
                value.append("\n" if escaped == "n" else escaped)
                continue
            if char == '"':
                break
            value.append(char)
        labels[key] = "".join(value)
        while pos < len(text) and text[pos] != ",":
            pos += 1
        if pos < len(text):
            pos += 1
    return tuple(sorted(labels.items()))


def format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    parts: list[str] = []
    for key, value in labels:
        escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        parts.append(f'{key}="{escaped}"')
    return "{" + ",".join(parts) + "}"


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
        sample = Sample(name, parse_labels(match.group("labels") or ""), value)
        samples[sample.key] = sample
    return samples


def finite_delta(before: float, after: float) -> float | None:
    if not math.isfinite(before) or not math.isfinite(after):
        return None
    return after - before


def counter_increase(before: float, after: float) -> float | None:
    """Return a monotonic counter increase, treating a decrease as a reset."""
    if not math.isfinite(before) or not math.isfinite(after):
        return None
    return after - before if after >= before else max(after, 0.0)


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


def rank_group(sample: Sample, aggregate_ranks: bool = False) -> RankGroup:
    labels = dict(sample.labels)
    worker_rank = (
        "ALL"
        if aggregate_ranks
        else labels.get(
            "worker_rank", labels.get("worker_id", labels.get("rank", "unknown"))
        )
    )
    return RankGroup(
        model_name=labels.get("model_name", ""),
        engine=labels.get("engine", ""),
        worker_rank=worker_rank,
    )


def counter_delta_by_rank(
    counter: str,
    before: dict[str, Sample],
    after: dict[str, Sample],
    aggregate_ranks: bool = False,
) -> dict[RankGroup, float]:
    totals: dict[RankGroup, float] = {}
    for key, after_sample in after.items():
        if not matching_metric(after_sample.name, counter):
            continue
        before_value = before.get(key).value if key in before else 0.0
        delta = counter_increase(before_value, after_sample.value)
        if delta is None:
            continue
        group = rank_group(after_sample, aggregate_ranks)
        totals[group] = totals.get(group, 0.0) + delta
    return totals


def histogram_quantile(
    quantile: float,
    cumulative_buckets: dict[float, float],
    count: float,
) -> float | None:
    """Prometheus-compatible linear interpolation within histogram buckets."""
    if not cumulative_buckets or count <= 0:
        return None
    rank = quantile * count
    previous_bound: float | None = None
    previous_count = 0.0
    for upper_bound, cumulative_count in sorted(cumulative_buckets.items()):
        # Counter resets and scrape races can make adjacent deltas slightly
        # non-monotonic. Clamp them before interpolation.
        cumulative_count = max(cumulative_count, previous_count)
        if cumulative_count < rank:
            previous_bound = upper_bound
            previous_count = cumulative_count
            continue
        if math.isinf(upper_bound):
            return previous_bound
        lower_bound = (
            0.0
            if previous_bound is None and upper_bound > 0
            else upper_bound if previous_bound is None else previous_bound
        )
        bucket_count = cumulative_count - previous_count
        if bucket_count <= 0:
            return upper_bound
        return lower_bound + (upper_bound - lower_bound) * (
            (rank - previous_count) / bucket_count
        )
    finite_bounds = [bound for bound in cumulative_buckets if math.isfinite(bound)]
    return max(finite_bounds) if finite_bounds else None


def histogram_intervals_by_rank(
    base: str,
    before: dict[str, Sample],
    after: dict[str, Sample],
    aggregate_ranks: bool = False,
) -> dict[RankGroup, HistogramStats]:
    """Calculate interval histogram statistics per rank or across all ranks."""
    buckets: dict[RankGroup, dict[float, float]] = {}
    for key, bucket_sample in after.items():
        if not matching_metric(bucket_sample.name, f"{base}_bucket"):
            continue
        le_text = bucket_sample.label("le")
        if le_text is None:
            continue
        before_value = before.get(key).value if key in before else 0.0
        delta = counter_increase(before_value, bucket_sample.value)
        if delta is None:
            continue
        bound = math.inf if le_text in {"+Inf", "Inf"} else float(le_text)
        group = rank_group(bucket_sample, aggregate_ranks)
        group_buckets = buckets.setdefault(group, {})
        group_buckets[bound] = group_buckets.get(bound, 0.0) + delta

    counts = counter_delta_by_rank(f"{base}_count", before, after, aggregate_ranks)
    totals = counter_delta_by_rank(f"{base}_sum", before, after, aggregate_ranks)
    groups = set(buckets) | set(counts) | set(totals)
    results: dict[RankGroup, HistogramStats] = {}
    for group in groups:
        group_buckets = buckets.get(group, {})
        count = counts.get(group, 0.0) or group_buckets.get(math.inf, 0.0)
        if count <= 0:
            continue
        quantiles: dict[float, float] = {}
        for quantile in HISTOGRAM_QUANTILES:
            value = histogram_quantile(quantile, group_buckets, count)
            if value is not None:
                quantiles[quantile] = value
        results[group] = HistogramStats(
            count=count,
            total=totals.get(group, 0.0),
            quantiles=quantiles,
        )
    return results


def rank_sort_key(group: RankGroup) -> tuple[str, str, int, int | str]:
    try:
        return (group.model_name, group.engine, 0, int(group.worker_rank))
    except ValueError:
        return (group.model_name, group.engine, 1, group.worker_rank)


def build_page_rows(
    page_key: str,
    histograms: tuple[tuple[str, str], ...],
    before: dict[str, Sample],
    after: dict[str, Sample],
    window_seconds: float,
    aggregate_ranks: bool,
) -> list[ReportRow]:
    rows: list[ReportRow] = []
    for metric, unit in histograms:
        stats_by_rank = histogram_intervals_by_rank(
            metric, before, after, aggregate_ranks
        )
        for group in sorted(stats_by_rank, key=rank_sort_key):
            stats = stats_by_rank[group]
            rows.append(
                ReportRow(
                    group=group,
                    metric=metric,
                    count=stats.count,
                    average=stats.average,
                    p90=stats.quantiles.get(0.9),
                    p99=stats.quantiles.get(0.99),
                    unit=unit,
                )
            )
    if window_seconds > 0:
        for counter, output_name in PAGE_THROUGHPUTS[page_key]:
            for group, byte_delta in sorted(
                counter_delta_by_rank(counter, before, after, aggregate_ranks).items(),
                key=lambda item: rank_sort_key(item[0]),
            ):
                rows.append(
                    ReportRow(
                        group=group,
                        metric=output_name,
                        count=None,
                        average=byte_delta / window_seconds / 1e9,
                        p90=None,
                        p99=None,
                        unit="GB/s",
                    )
                )
    return rows


def format_optional(value: float | None) -> str:
    return "-" if value is None else format_number(value)


def render_rows_table(rows: list[ReportRow]) -> list[str]:
    lines: list[str] = []
    lines.append(
        "| Model | Engine | Rank | Metric | Samples | Avg | P90 | P99 | Unit |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---|")
    if not rows:
        lines.append("| - | - | - | No observations | - | - | - | - | - |")
        return lines
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.group.model_name or "-",
                    row.group.engine or "-",
                    row.group.worker_rank,
                    row.metric,
                    format_optional(row.count),
                    format_optional(row.average),
                    format_optional(row.p90),
                    format_optional(row.p99),
                    row.unit,
                )
            )
            + " |"
        )
    return lines


def render_page_table(
    title: str,
    all_rank_rows: list[ReportRow],
    per_rank_rows: list[ReportRow],
) -> list[str]:
    lines = [f"## {title}", "", "### All ranks", ""]
    lines.extend(render_rows_table(all_rank_rows))
    lines.extend(("", "### Per-rank detail", ""))
    lines.extend(render_rows_table(per_rank_rows))
    return lines


def write_report_csv(
    path: Path,
    pages: dict[str, tuple[list[ReportRow], list[ReportRow]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            (
                "page",
                "scope",
                "model_name",
                "engine",
                "worker_rank",
                "metric",
                "samples",
                "avg",
                "p90",
                "p99",
                "unit",
            )
        )
        for page_key, (all_rank_rows, per_rank_rows) in pages.items():
            for scope, rows in (
                ("all_ranks", all_rank_rows),
                ("per_rank", per_rank_rows),
            ):
                for row in rows:
                    writer.writerow(
                        (
                            page_key,
                            scope,
                            row.group.model_name,
                            row.group.engine,
                            row.group.worker_rank,
                            row.metric,
                            row.count,
                            row.average,
                            row.p90,
                            row.p99,
                            row.unit,
                        )
                    )


def row_as_dict(row: ReportRow) -> dict[str, object]:
    return {
        "model_name": row.group.model_name,
        "engine": row.group.engine,
        "worker_rank": row.group.worker_rank,
        "metric": row.metric,
        "samples": row.count,
        "avg": row.average,
        "p90": row.p90,
        "p99": row.p99,
        "unit": row.unit,
    }


def write_delta_report(
    path: Path,
    before: dict[str, Sample],
    after: dict[str, Sample],
    window_seconds: float,
    full: bool,
) -> dict[str, object]:
    pages: dict[str, tuple[list[ReportRow], list[ReportRow]]] = {}
    report = [
        "# UCM per-rank metrics report",
        "",
        f"Capture window: {window_seconds:.3f}s",
        "",
        "Histogram avg/p90/p99 values use only observations added during this window.",
        "All-ranks rows merge buckets before calculating p90/p99; they do not average rank quantiles.",
        "Per-rank rows stay separate by model, engine, and worker rank.",
        "",
    ]
    for page_index, (page_key, title, histograms) in enumerate(REPORT_PAGES):
        all_rank_rows = build_page_rows(
            page_key,
            histograms,
            before,
            after,
            window_seconds,
            aggregate_ranks=True,
        )
        per_rank_rows = build_page_rows(
            page_key,
            histograms,
            before,
            after,
            window_seconds,
            aggregate_ranks=False,
        )
        pages[page_key] = (all_rank_rows, per_rank_rows)
        if page_index:
            report.extend(("", "---", ""))
        report.extend(render_page_table(title, all_rank_rows, per_rank_rows))

    if full:
        report.extend(("", "---", "", "## All non-zero series deltas", ""))
        report.append("```text")
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
        report.append("```")

    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    write_report_csv(path.parent / "metrics_report.csv", pages)
    return {
        "schema_version": 2,
        "window_seconds": window_seconds,
        "pages": {
            page_key: {
                "all_ranks": [row_as_dict(row) for row in all_rank_rows],
                "per_rank": [row_as_dict(row) for row in per_rank_rows],
            }
            for page_key, (all_rank_rows, per_rank_rows) in pages.items()
        },
    }


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

    report_path = run_dir / "metrics_report.md"
    summary = write_delta_report(
        report_path, before, after, after_ts - before_ts, args.full
    )
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved metrics report: {report_path}")
    print(f"Saved machine-readable summary: {summary_path}")
    print(f"Saved combined CSV: {run_dir / 'metrics_report.csv'}")
    print()
    print("Raw series are in:")
    print(f"  {run_dir / 'before_filtered.prom'}")
    print(f"  {run_dir / 'after_filtered.prom'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
