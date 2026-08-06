#!/usr/bin/env python3
"""Parse vLLM inference duration monitor logs and output a summary table.

New-format logs are grouped by the explicit (dp_rank, step_id) pair. Legacy
logs without these fields still use scheduler-line ordering as a fallback.

Usage:
    python parse_monitor_log.py vllm_server.log             # latest step per DP
    python parse_monitor_log.py vllm_server.log --step 0    # step 0 for all DPs
    python parse_monitor_log.py vllm_server.log --all       # show all steps
    python parse_monitor_log.py vllm_server.log --dp-rank 1 # latest step for DP 1
    python parse_monitor_log.py vllm_server.log --csv out.csv  # export CSV
    python parse_monitor_log.py vllm_server.log --output summary.log  # write to file
"""

import argparse
import re
import sys
from collections import defaultdict


def _extract_int(line, field):
    match = re.search(rf"(?:^|[, ]){re.escape(field)}=(-?\d+)", line)
    return int(match.group(1)) if match else None


def _new_step(dp_rank=None, step_id=None):
    return {
        "bandwidth": [],
        "forward": None,
        "scheduler": None,
        "dp_rank": dp_rank,
        "step_id": step_id,
    }


def parse_log(log_path):
    kv_sizes = {}
    identified_steps = {}
    legacy_steps = []
    current = _new_step()

    bw_pattern = re.compile(
        r"KV bandwidth:.*?layer_idx=(\S+) \(compute\) -> "
        r"layer_idx=(\S+) \(load\), cur_kv_bytes_per_token=(\d+), "
        r"next_kv_bytes_per_token=(\d+), "
        r"fake_hit=(\d+), kv_total=([\d.]+) MB, "
        r"layer_avg_ms=([\d.]+), required_bandwidth=([\d.]+) GB/s"
    )
    kv_pattern = re.compile(
        r"KV cache total: layer_idx=(\d+), total_bytes_per_token=(\d+)"
    )
    fwd_pattern = re.compile(
        r"Inference duration aggregate:.*scope=forward.*"
        r"avg_ms=([\d.]+).*min_ms=([\d.]+).*max_ms=([\d.]+)"
    )
    sched_pattern = re.compile(
        r"Inference duration scheduler stats:.*"
        r"scheduled_reqs=(\d+).*scheduled_tokens=(\d+)"
    )

    def identified_step(line):
        dp_rank = _extract_int(line, "dp_rank")
        step_id = _extract_int(line, "step_id")
        if dp_rank is None or step_id is None:
            return None
        return identified_steps.setdefault(
            (dp_rank, step_id), _new_step(dp_rank, step_id)
        )

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = kv_pattern.search(line)
            if m:
                kv_sizes[int(m.group(1))] = int(m.group(2))
                continue

            m = sched_pattern.search(line)
            if m:
                step = identified_step(line)
                scheduler = (int(m.group(1)), int(m.group(2)))
                if step is not None:
                    step["scheduler"] = scheduler
                else:
                    if current["bandwidth"] or current["forward"]:
                        legacy_steps.append(current)
                    current = _new_step()
                    current["scheduler"] = scheduler
                continue

            m = fwd_pattern.search(line)
            if m:
                step = identified_step(line) or current
                step["forward"] = (
                    float(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                )
                continue

            m = bw_pattern.search(line)
            if m:
                step = identified_step(line) or current
                step["bandwidth"].append({
                    "compute_layer": m.group(1),
                    "load_layer": m.group(2),
                    "cur_kv_bytes": int(m.group(3)),
                    "next_kv_bytes": int(m.group(4)),
                    "fake_hit": int(m.group(5)),
                    "kv_total": float(m.group(6)),
                    "block_ms": float(m.group(7)),
                    "bandwidth": float(m.group(8)),
                    "worker_rank": _extract_int(line, "worker_rank"),
                })

    if current["bandwidth"] or current["forward"]:
        legacy_steps.append(current)

    if identified_steps:
        steps = [
            identified_steps[key]
            for key in sorted(identified_steps)
            if identified_steps[key]["bandwidth"]
            or identified_steps[key]["forward"]
        ]
    else:
        steps = legacy_steps

    return kv_sizes, steps


def group_bandwidth(step):
    grouped = defaultdict(list)
    for entry in step["bandwidth"]:
        grouped[entry["compute_layer"]].append(entry)

    # A worker should emit one line per layer and step. If duplicated log lines
    # are present, keep the last one instead of counting one worker twice.
    for layer, entries in list(grouped.items()):
        if any(entry["worker_rank"] is not None for entry in entries):
            by_worker = {}
            without_rank = []
            for entry in entries:
                worker_rank = entry["worker_rank"]
                if worker_rank is None:
                    without_rank.append(entry)
                else:
                    by_worker[worker_rank] = entry
            grouped[layer] = list(by_worker.values()) + without_rank
    return grouped


def print_step(kv_sizes, step, step_idx, output=print):
    bw_data = group_bandwidth(step)

    if not bw_data:
        return

    fake_hit = next(iter(bw_data.values()))[0]["fake_hit"]
    worker_ranks = {
        entry["worker_rank"]
        for entries in bw_data.values()
        for entry in entries
        if entry["worker_rank"] is not None
    }
    workers = len(worker_ranks) or len(next(iter(bw_data.values())))

    if step["dp_rank"] is not None:
        label = f"dp_rank {step['dp_rank']}, step {step['step_id']}"
    else:
        label = f"legacy step {step_idx}"
    output(f"=== Inference Duration Summary ({label}) ===")
    output(f"workers={workers}, layers={len(kv_sizes)}, fake_hit={fake_hit}")
    if step["forward"]:
        output(f"forward: avg={step['forward'][0]:.3f}ms, "
               f"min={step['forward'][1]:.3f}ms, "
               f"max={step['forward'][2]:.3f}ms")
    if step["scheduler"]:
        output(f"scheduled_reqs={step['scheduler'][0]}, "
               f"scheduled_tokens={step['scheduler'][1]}")
    output()

    header = (f"{'layer':>5} | {'layer_avg_ms':>12} | {'cur_kv_bytes/tok':>16} | "
              f"{'-> load':>8} | {'load_kv_bytes/tok':>17} | "
              f"{'kv_total_MB':>11} | {'bandwidth_GBps':>14}")
    output(header)
    output("-" * len(header))

    max_bw = 0
    max_bw_layer = ""
    min_bw = float("inf")
    min_bw_layer = ""

    for compute_layer in sorted(bw_data.keys(),
                                key=lambda x: int(x) if x.isdigit() else 9999):
        entries = bw_data[compute_layer]
        avg_layer = sum(e["block_ms"] for e in entries) / len(entries)
        avg_bw = sum(e["bandwidth"] for e in entries) / len(entries)
        load_layer = entries[0]["load_layer"]
        cur_kv = entries[0]["cur_kv_bytes"]
        next_kv = entries[0]["next_kv_bytes"]
        kv_total = entries[0]["kv_total"]

        output(f"{compute_layer:>5} | {avg_layer:>12.3f} | {cur_kv:>16} | "
               f"{load_layer:>8} | {next_kv:>17} | "
               f"{kv_total:>11.2f} | {avg_bw:>14.2f}")

        if compute_layer.isdigit() and int(compute_layer) > 0:
            if avg_bw > max_bw:
                max_bw = avg_bw
                max_bw_layer = compute_layer
            if avg_bw < min_bw and avg_bw > 0:
                min_bw = avg_bw
                min_bw_layer = compute_layer

    output()
    output(f"max_bandwidth: {max_bw:.2f} GB/s (layer {max_bw_layer})")
    output(f"min_bandwidth: {min_bw:.2f} GB/s (layer {min_bw_layer})")
    output(f"=== End Summary ===")
    output()


def export_csv(kv_sizes, steps, csv_path):
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dp_rank", "step", "fake_hit", "workers", "scheduled_reqs",
            "scheduled_tokens", "forward_avg_ms",
            "layer", "layer_avg_ms", "cur_kv_bytes_per_token",
            "load_layer", "load_kv_bytes_per_token",
            "kv_total_MB", "bandwidth_GBps",
        ])
        for step_idx, step in enumerate(steps):
            bw_data = group_bandwidth(step)
            if not bw_data:
                continue
            fake_hit = next(iter(bw_data.values()))[0]["fake_hit"]
            worker_ranks = {
                entry["worker_rank"]
                for entries in bw_data.values()
                for entry in entries
                if entry["worker_rank"] is not None
            }
            workers = len(worker_ranks) or len(next(iter(bw_data.values())))
            fwd_avg = step["forward"][0] if step["forward"] else ""
            sched = step["scheduler"] or (0, 0)
            for compute_layer in sorted(bw_data.keys(),
                                        key=lambda x: int(x) if x.isdigit() else 9999):
                entries = bw_data[compute_layer]
                avg_block = sum(e["block_ms"] for e in entries) / len(entries)
                avg_bw = sum(e["bandwidth"] for e in entries) / len(entries)
                writer.writerow([
                    step["dp_rank"] if step["dp_rank"] is not None else "",
                    step["step_id"] if step["step_id"] is not None else step_idx,
                    fake_hit, workers,
                    sched[0], sched[1], fwd_avg,
                    compute_layer, f"{avg_block:.3f}",
                    entries[0]["cur_kv_bytes"],
                    entries[0]["load_layer"],
                    entries[0]["next_kv_bytes"],
                    f"{entries[0]['kv_total']:.2f}",
                    f"{avg_bw:.2f}",
                ])
    print(f"CSV exported to {csv_path} ({len(steps)} steps)")


def main():
    parser = argparse.ArgumentParser(
        description="Parse vLLM inference duration monitor logs"
    )
    parser.add_argument("log_file", help="Path to vLLM log file")
    parser.add_argument("--step", type=int, default=None,
                        help="Show step ID (new logs) or 0-based index (legacy logs)")
    parser.add_argument("--all", action="store_true",
                        help="Show all steps")
    parser.add_argument("--dp-rank", type=int, default=None,
                        help="Only show data for this DP rank")
    parser.add_argument("--csv", default=None,
                        help="Export all steps to CSV")
    parser.add_argument("--output", default=None,
                        help="Output destination: '-' for terminal (default), "
                             "or file path to write summary tables")
    args = parser.parse_args()

    out = sys.stdout
    if args.output and args.output != "-":
        out = open(args.output, "a")

    def output(msg=""):
        print(msg, file=out)
        out.flush()

    kv_sizes, steps = parse_log(args.log_file)

    if args.dp_rank is not None:
        steps = [step for step in steps if step["dp_rank"] == args.dp_rank]

    if not steps:
        output("No step data found in log.")
        if out is not sys.stdout:
            out.close()
        return

    output(f"Found {len(steps)} step(s) in log.\n")

    if args.csv:
        export_csv(kv_sizes, steps, args.csv)
        if out is not sys.stdout:
            out.close()
        return

    identified = any(step["step_id"] is not None for step in steps)

    if args.all:
        for i, step in enumerate(steps):
            print_step(kv_sizes, step, i, output)
    elif args.step is not None:
        if identified:
            selected = [step for step in steps if step["step_id"] == args.step]
            if not selected:
                output(f"Step ID {args.step} not found.")
            for step in selected:
                print_step(kv_sizes, step, args.step, output)
        elif args.step < 0 or args.step >= len(steps):
            output(f"Step {args.step} not found (0~{len(steps)-1})")
        else:
            print_step(kv_sizes, steps[args.step], args.step, output)
    elif identified:
        latest_by_dp = {}
        for step in steps:
            latest_by_dp[step["dp_rank"]] = step
        for step in latest_by_dp.values():
            print_step(kv_sizes, step, step["step_id"], output)
    else:
        print_step(kv_sizes, steps[-1], len(steps) - 1, output)

    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
