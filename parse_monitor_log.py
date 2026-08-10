#!/usr/bin/env python3
"""Parse vLLM inference duration monitor logs and summarize layer timings.

Usage:
    python parse_monitor_log.py vllm_server.log
    python parse_monitor_log.py vllm_server.log --forward-id 0
    python parse_monitor_log.py vllm_server.log --all
    python parse_monitor_log.py vllm_server.log --dp-rank 1
    python parse_monitor_log.py vllm_server.log --csv timings.csv
    python parse_monitor_log.py vllm_server.log --output summary.log
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def _extract_int(line, field):
    match = re.search(rf"(?:^|[, ]){re.escape(field)}=(-?\d+)", line)
    return int(match.group(1)) if match else None


def _new_step(dp_rank, scheduler_iteration_id):
    return {
        "dp_rank": dp_rank,
        "scheduler_iteration_id": scheduler_iteration_id,
        "forward_id": None,
        "fake_hit": 0,
        "workers": 0,
        "scheduler": None,
        "forward": None,
        "layers": {},
    }


def _step_for_line(steps, line):
    dp_rank = _extract_int(line, "dp_rank")
    scheduler_iteration_id = _extract_int(line, "scheduler_iteration_id")
    legacy_step_id = _extract_int(line, "step_id")
    if scheduler_iteration_id is None:
        scheduler_iteration_id = legacy_step_id
    if dp_rank is None or scheduler_iteration_id is None:
        return None
    step = steps.setdefault(
        (dp_rank, scheduler_iteration_id),
        _new_step(dp_rank, scheduler_iteration_id),
    )
    forward_id = _extract_int(line, "forward_id")
    if forward_id is None:
        forward_id = legacy_step_id
    if forward_id is not None:
        step["forward_id"] = forward_id
    return step


def _read_log_lines(log_path, start_offset=None, end_offset=None):
    if start_offset is None and end_offset is None:
        with open(log_path, encoding="utf-8", errors="replace") as log_file:
            yield from log_file
        return

    with open(log_path, "rb") as log_file:
        log_file.seek(start_offset or 0)
        if end_offset is None:
            data = log_file.read()
        else:
            data = log_file.read(max(end_offset - (start_offset or 0), 0))
    yield from data.decode("utf-8", errors="replace").splitlines()


def parse_log(log_path, start_offset=None, end_offset=None):
    steps = {}
    scheduler_pattern = re.compile(
        r"Inference duration scheduler stats:.*"
        r"scheduled_reqs=(\d+), new_reqs=(\d+), scheduled_tokens=(\d+)"
    )
    aggregate_pattern = re.compile(
        r"Inference duration aggregate:.*scope=([^,]+), "
        r"(?:fake_hit=(\d+), )?count=(\d+), avg_ms=([\d.]+), "
        r"min_ms=([\d.]+), max_ms=([\d.]+)"
    )

    for line in _read_log_lines(log_path, start_offset, end_offset):
        match = scheduler_pattern.search(line)
        if match:
            step = _step_for_line(steps, line)
            if step is not None:
                step["scheduler"] = {
                    "scheduled_reqs": int(match.group(1)),
                    "new_reqs": int(match.group(2)),
                    "scheduled_tokens": int(match.group(3)),
                }
            continue

        match = aggregate_pattern.search(line)
        if not match:
            continue
        step = _step_for_line(steps, line)
        if step is None:
            continue
        scope = match.group(1)
        fake_hit = match.group(2)
        stats = {
            "count": int(match.group(3)),
            "avg_ms": float(match.group(4)),
            "min_ms": float(match.group(5)),
            "max_ms": float(match.group(6)),
        }
        if fake_hit is not None:
            step["fake_hit"] = int(fake_hit)
        workers = _extract_int(line, "workers")
        if workers is not None:
            step["workers"] = max(step["workers"], workers)
        if scope == "forward":
            step["forward"] = stats
        elif scope.startswith("block_layer:"):
            step["layers"][scope.removeprefix("block_layer:")] = stats

    return [
        steps[key]
        for key in sorted(steps)
        if steps[key]["forward"] is not None or steps[key]["layers"]
    ]


def _layer_sort_key(layer):
    return (0, int(layer)) if layer.isdigit() else (1, layer)


def print_step(step, output=print):
    output(
        "=== Inference Duration Summary "
        f"(dp_rank {step['dp_rank']}, forward {step['forward_id']}, "
        f"scheduler iteration {step['scheduler_iteration_id']}) ==="
    )
    output(
        f"workers={step['workers']}, layers={len(step['layers'])}, "
        f"fake_hit={step['fake_hit']}"
    )
    if step["forward"] is not None:
        stats = step["forward"]
        output(
            f"forward: avg={stats['avg_ms']:.3f}ms, "
            f"min={stats['min_ms']:.3f}ms, max={stats['max_ms']:.3f}ms"
        )
    if step["scheduler"] is not None:
        scheduler = step["scheduler"]
        output(
            f"scheduled_reqs={scheduler['scheduled_reqs']}, "
            f"new_reqs={scheduler['new_reqs']}, "
            f"scheduled_tokens={scheduler['scheduled_tokens']}"
        )
    output()

    header = (
        f"{'layer':>8} | {'avg_ms':>10} | {'min_ms':>10} | "
        f"{'max_ms':>10} | {'hit_tokens':>10}"
    )
    output(header)
    output("-" * len(header))
    for layer in sorted(step["layers"], key=_layer_sort_key):
        stats = step["layers"][layer]
        output(
            f"{layer:>8} | {stats['avg_ms']:>10.3f} | "
            f"{stats['min_ms']:>10.3f} | {stats['max_ms']:>10.3f} | "
            f"{step['fake_hit']:>10}"
        )
    output("=== End Summary ===")
    output()


def _merge_stats(stats_list):
    if not stats_list:
        return None
    count = sum(stats["count"] for stats in stats_list)
    if count == 0:
        return None
    return {
        "count": count,
        "avg_ms": sum(stats["avg_ms"] * stats["count"] for stats in stats_list) / count,
        "min_ms": min(stats["min_ms"] for stats in stats_list),
        "max_ms": max(stats["max_ms"] for stats in stats_list),
    }


def print_bench(bench_id, steps, output=print):
    output(f"=== Inference Duration Bench Summary ({bench_id}) ===")
    if not steps:
        output("No forward data found for this bench.")
        output("=== End Bench Summary ===")
        output()
        return

    dp_ranks = sorted({step["dp_rank"] for step in steps})
    workers_by_dp = {}
    for step in steps:
        workers_by_dp[step["dp_rank"]] = max(
            workers_by_dp.get(step["dp_rank"], 0), step["workers"]
        )
    fake_hit = sum(step["fake_hit"] for step in steps)
    output(
        f"forwards={len(steps)}, dp_ranks={dp_ranks}, "
        f"workers={sum(workers_by_dp.values())}, fake_hit={fake_hit}"
    )

    forward_stats = _merge_stats(
        [step["forward"] for step in steps if step["forward"] is not None]
    )
    if forward_stats is not None:
        output(
            f"forward: avg={forward_stats['avg_ms']:.3f}ms, "
            f"min={forward_stats['min_ms']:.3f}ms, "
            f"max={forward_stats['max_ms']:.3f}ms"
        )

    schedulers = [step["scheduler"] for step in steps if step["scheduler"] is not None]
    if schedulers:
        output(
            f"scheduled_reqs={sum(item['scheduled_reqs'] for item in schedulers)}, "
            f"new_reqs={sum(item['new_reqs'] for item in schedulers)}, "
            f"scheduled_tokens={sum(item['scheduled_tokens'] for item in schedulers)}"
        )
    output()

    layers = sorted(
        {layer for step in steps for layer in step["layers"]},
        key=_layer_sort_key,
    )
    header = (
        f"{'layer':>8} | {'avg_ms':>10} | {'min_ms':>10} | "
        f"{'max_ms':>10} | {'hit_tokens':>10}"
    )
    output(header)
    output("-" * len(header))
    for layer in layers:
        layer_stats = _merge_stats(
            [step["layers"][layer] for step in steps if layer in step["layers"]]
        )
        if layer_stats is None:
            continue
        layer_hits = sum(step["fake_hit"] for step in steps if layer in step["layers"])
        output(
            f"{layer:>8} | {layer_stats['avg_ms']:>10.3f} | "
            f"{layer_stats['min_ms']:>10.3f} | "
            f"{layer_stats['max_ms']:>10.3f} | {layer_hits:>10}"
        )
    output("=== End Bench Summary ===")
    output()


def bench_index_path(log_path):
    resolved_log = Path(log_path).resolve()
    return resolved_log.with_name(resolved_log.name + ".bench_runs.jsonl")


def load_bench_runs(log_path):
    index_path = bench_index_path(log_path)
    if not index_path.exists():
        return []
    runs = []
    with index_path.open(encoding="utf-8") as index_file:
        for line_number, line in enumerate(index_file, 1):
            if not line.strip():
                continue
            try:
                run = json.loads(line)
                run["start_offset"] = int(run["start_offset"])
                run["end_offset"] = int(run["end_offset"])
                runs.append(run)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                print(
                    f"Ignoring invalid bench index line {line_number}: {error}",
                    file=sys.stderr,
                )
    return runs


def export_csv(steps, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "dp_rank",
                "scheduler_iteration_id",
                "forward_id",
                "workers",
                "scheduled_reqs",
                "new_reqs",
                "scheduled_tokens",
                "fake_hit",
                "forward_avg_ms",
                "layer",
                "layer_avg_ms",
                "layer_min_ms",
                "layer_max_ms",
            ]
        )
        for step in steps:
            scheduler = step["scheduler"] or {}
            forward_avg = (
                step["forward"]["avg_ms"] if step["forward"] is not None else ""
            )
            for layer in sorted(step["layers"], key=_layer_sort_key):
                stats = step["layers"][layer]
                writer.writerow(
                    [
                        step["dp_rank"],
                        step["scheduler_iteration_id"],
                        step["forward_id"],
                        step["workers"],
                        scheduler.get("scheduled_reqs", ""),
                        scheduler.get("new_reqs", ""),
                        scheduler.get("scheduled_tokens", ""),
                        step["fake_hit"],
                        forward_avg,
                        layer,
                        stats["avg_ms"],
                        stats["min_ms"],
                        stats["max_ms"],
                    ]
                )
    print(f"CSV exported to {csv_path} ({len(steps)} forwards)")


def main():
    parser = argparse.ArgumentParser(
        description="Parse vLLM inference duration monitor logs"
    )
    parser.add_argument("log_file", help="Path to vLLM log file")
    parser.add_argument(
        "--forward-id",
        "--step",
        dest="forward_id",
        type=int,
        default=None,
        help="Show forward ID; --step is an alias",
    )
    parser.add_argument("--all", action="store_true", help="Show all forwards")
    parser.add_argument(
        "--all-benches",
        action="store_true",
        help="Show one aggregate summary for every recorded bench",
    )
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=None,
        help="Only show data for this DP rank",
    )
    parser.add_argument("--csv", default=None, help="Export all forwards to CSV")
    parser.add_argument(
        "--output",
        default=None,
        help="Output destination: '-' for terminal (default), or a file path",
    )
    args = parser.parse_args()

    output_file = sys.stdout
    if args.output and args.output != "-":
        output_file = open(args.output, "a", encoding="utf-8")

    def output(message=""):
        print(message, file=output_file)
        output_file.flush()

    if args.all_benches:
        runs = load_bench_runs(args.log_file)
        if not runs:
            output("No completed bench records found.")
        for run in runs:
            steps = parse_log(
                args.log_file,
                run["start_offset"],
                run["end_offset"],
            )
            if args.dp_rank is not None:
                steps = [step for step in steps if step["dp_rank"] == args.dp_rank]
            print_bench(run["bench_id"], steps, output)
        if output_file is not sys.stdout:
            output_file.close()
        return

    steps = parse_log(args.log_file)
    if args.dp_rank is not None:
        steps = [step for step in steps if step["dp_rank"] == args.dp_rank]

    if not steps:
        output("No forward data found in log.")
    elif args.csv:
        export_csv(steps, args.csv)
    elif args.all:
        for step in steps:
            print_step(step, output)
    elif args.forward_id is not None:
        selected = [step for step in steps if step["forward_id"] == args.forward_id]
        if not selected:
            output(f"Forward ID {args.forward_id} not found.")
        for step in selected:
            print_step(step, output)
    else:
        latest_by_dp = {}
        for step in steps:
            latest_by_dp[step["dp_rank"]] = step
        for step in latest_by_dp.values():
            print_step(step, output)

    if output_file is not sys.stdout:
        output_file.close()


if __name__ == "__main__":
    main()
