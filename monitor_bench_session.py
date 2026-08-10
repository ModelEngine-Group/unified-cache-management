#!/usr/bin/env python3
"""Record one benchmark's server-log range and print its timing summary."""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from parse_monitor_log import bench_index_path, parse_log, print_bench


def _now():
    return datetime.now(timezone.utc).isoformat()


def _active_path(log_path):
    return log_path.with_name(log_path.name + ".bench_active.json")


def _start(log_path):
    if not log_path.is_file():
        raise RuntimeError(f"Server log does not exist: {log_path}")

    active_path = _active_path(log_path)
    bench_id = datetime.now().strftime("bench_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    state = {
        "bench_id": bench_id,
        "log_path": str(log_path),
        "start_offset": log_path.stat().st_size,
        "start_time": _now(),
    }
    try:
        with active_path.open("x", encoding="utf-8") as active_file:
            json.dump(state, active_file, ensure_ascii=False, indent=2)
            active_file.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            "A bench session is already active. Stop it before starting another."
        ) from error
    print(f"Started {bench_id}")


def _stop(log_path):
    active_path = _active_path(log_path)
    if not active_path.is_file():
        raise RuntimeError("No active bench session found.")
    if not log_path.is_file():
        raise RuntimeError(f"Server log does not exist: {log_path}")

    with active_path.open(encoding="utf-8") as active_file:
        state = json.load(active_file)
    if Path(state["log_path"]).resolve() != log_path:
        raise RuntimeError("The active bench belongs to a different server log.")

    end_offset = log_path.stat().st_size
    start_offset = int(state["start_offset"])
    if end_offset < start_offset:
        raise RuntimeError("Server log was truncated or rotated during the bench.")

    run = {
        **state,
        "end_offset": end_offset,
        "end_time": _now(),
    }
    index_path = bench_index_path(log_path)
    with index_path.open("a", encoding="utf-8") as index_file:
        index_file.write(json.dumps(run, ensure_ascii=False) + "\n")
        index_file.flush()
        os.fsync(index_file.fileno())
    active_path.unlink()

    print(f"Stopped {run['bench_id']}")
    steps = parse_log(log_path, start_offset, end_offset)
    print_bench(run["bench_id"], steps)


def main():
    parser = argparse.ArgumentParser(
        description="Mark one vLLM bench range in a server log"
    )
    parser.add_argument("action", choices=("start", "stop"))
    parser.add_argument("log_file", help="Path to the vLLM server log")
    args = parser.parse_args()

    log_path = Path(args.log_file).resolve()
    try:
        if args.action == "start":
            _start(log_path)
        else:
            _stop(log_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
