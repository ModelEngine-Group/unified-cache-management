"""Join Early stage records without inferring missing timing or splitting joint loads.

Input logs must belong to one container invocation. Cross-process durations are
omitted unless --clock-proof records the same boot ID and time namespace for
the client, scheduler, broker and every worker. Pin time includes allocation,
grouping and contention; native Wait time is completion reaping time.
"""

import argparse
import ast
import json
from pathlib import Path
import re


def records(text):
    for line in text.splitlines():
        if "UCM_EARLY " not in line:
            continue
        body = line.split("UCM_EARLY ", 1)[1]
        row = dict(re.findall(r"(\w+)=([^\s]+)", body))
        row["event"] = row.get("stage", body.split()[0])
        pid = re.search(r"\[(\d+),\d+\]", line) or re.search(r"pid=(\d+)", line)
        if pid:
            row["pid"] = pid.group(1)
        stats = re.search(r"stats=(\{.*?\})", body)
        if stats:
            row["stats"] = ast.literal_eval(stats.group(1))
        yield row


def elapsed(start, end):
    if start is None or end is None or str(start) == "None" or str(end) == "None":
        return None
    delta = int(end) - int(start)
    return delta / 1e6 if delta >= 0 else None


def same_clock(proof):
    rows = proof.get("processes", [])
    roles = {row.get("role") for row in rows}
    expected = proof.get("expected_counts", {})
    return (proof.get("complete") is True and
            {"client", "scheduler", "broker", "worker"} <= roles and
            {"client", "scheduler", "broker", "worker"} <= expected.keys() and
            all(count > 0 and sum(row.get("role") == role for row in rows) == count
                for role, count in expected.items()) and
            len({row.get("boot_id") for row in rows}) == 1 and
            len({row.get("time_namespace") for row in rows}) == 1 and
            all(row.get("boot_id") and row.get("time_namespace") for row in rows))


def summarize(rows, cross_process=False):
    jobs, associations, stages, joint_loads, dispatches = {}, {}, [], [], []
    seen = set()
    for row in rows:
        # A source may contain both live and final snapshots of the same line.
        signature = json.dumps(row, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        if row.get("stage"):
            stages.append(row)
            continue
        key = row.get("key")
        if key is None:
            continue
        job = jobs.setdefault(key, {"events": {}, "ranks": {}})
        if "request" in row:
            job["request"] = row["request"]
        if "dp" in row:
            job["dp"] = row["dp"]
        if "rank" not in row:
            job["events"][row["event"]] = row
            continue
        rank = job["ranks"].setdefault(row["rank"], {"events": {}, "batches": {}})
        if "batch" not in row:
            rank["events"][row["event"]] = row
            continue
        batch = rank["batches"].setdefault(row["batch"], {"events": {}, "stages": {}})
        batch["events"][row["event"]] = row
        if row["event"] == "batch_reserved" and "pid" in row:
            associations[(row["pid"], row["reservation"])] = (key, row["rank"], row["batch"])

    unmatched = []
    for row in stages:
        if row["stage"] == "load_dispatch":
            dispatches.append(row)  # actual native Get count, by PID/task
            continue
        if row["stage"] == "load_handles":
            handles = row.get("load_reservations", "").split(",")
            members = [associations[(row.get("pid"), handle)] for handle in handles
                       if (row.get("pid"), handle) in associations]
            joint_loads.append(dict(row, members=members,
                                    unprepared_shards=int(row["shards"]) - int(row["handles"])))
            continue
        member = associations.get((row.get("pid"), row.get("reservation")))
        if member is None:
            unmatched.append(row)
            continue
        key, rank, batch = member
        jobs[key]["ranks"][rank]["batches"][batch]["stages"][row["stage"]] = row

    for job in jobs.values():
        events = job["events"]
        submitted = events.get("submitted", {}).get("ns")
        ready = events.get("ready", {}).get("ns")
        handoff = events.get("handoff", {})
        job["submitted_to_all_ready_ms"] = elapsed(submitted, ready) if cross_process else None
        job["ready_to_native_scheduled_ms"] = elapsed(ready, handoff.get("native_scheduled_ns")) if cross_process else None
        for rank in job["ranks"].values():
            e = rank["events"]
            rank["lookup_queue_ms"] = elapsed(e.get("lookup_enqueued", {}).get("ns"), e.get("lookup_start", {}).get("ns"))
            rank["rank_to_all_ready_ms"] = elapsed(e.get("rank_ready", {}).get("ns"), ready) if cross_process else None
            rank["submitted_to_enqueued_ms"] = elapsed(submitted, e.get("lookup_enqueued", {}).get("ns")) if cross_process else None
            first_io = []
            last_reserve = []
            for batch in rank["batches"].values():
                be, bs = batch["events"], batch["stages"]
                reservation = be.get("batch_reserved", {})
                batch["data_queue_ms"] = elapsed(reservation.get("end_ns"), be.get("batch_start", {}).get("ns"))
                # Python publication may wait for the worker condition after
                # Reserve returned. Use the correlated C++ endpoint instead.
                if "end_ns" in bs.get("reserve", {}):
                    last_reserve.append(int(bs["reserve"]["end_ns"]))
                io = int(bs.get("prefetch", {}).get("first_io_ns", 0))
                if io:
                    first_io.append(io)
            # This proves submission order, not physical SSD utilization or an
            # uninterrupted I/O interval between a batch's first/last transfer.
            rank["first_io_lead_over_reserve_done_ms"] = max(0, (max(last_reserve) - min(first_io)) / 1e6) if first_io and last_reserve else None
    return dict(cross_process_clock_verified=cross_process, jobs=jobs,
                joint_loads=joint_loads, native_load_dispatches=dispatches,
                unmatched_stages=unmatched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--clock-proof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not path.is_absolute() for path in [*args.logs, args.output] + ([args.clock_proof] if args.clock_proof else [])):
        parser.error("all paths must be absolute")
    proof = json.loads(args.clock_proof.read_text()) if args.clock_proof else {}
    report = summarize((row for path in args.logs for row in records(path.read_text())), same_clock(proof))
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
