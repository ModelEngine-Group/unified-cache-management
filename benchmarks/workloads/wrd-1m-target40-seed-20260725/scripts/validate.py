#!/usr/bin/env python3
"""Independent offline checks; networking is explicitly disabled."""
from __future__ import annotations

import argparse
import array
import asyncio
import collections
import csv
import hashlib
import json
import os
import random
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
sys.path.insert(0, str(ROOT / "vendor"))
import benchmark_swe_rebench_closed_loop as replay
from build_swe_rebench_long_context import BuildConfig, causal_merge


def no_network(*args, **kwargs):
    raise AssertionError("Network is forbidden in offline validation")


socket.socket.connect = no_network
socket.create_connection = no_network
replay.create_http_session = no_network


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def independent_intervals(events):
    # Keep a separate open interval for each already-seen trajectory. Each
    # incoming event updates all OTHER intervals using a maximum. No slicing
    # or backward scan and no use of generation.compute_intervals.
    open_intervals = {}
    result = []
    for i, e in enumerate(events):
        t, length = e["trajectory_id"], e["chunk_aligned_prompt_tokens"]
        previous = open_intervals.get(t)
        value = None
        if previous is not None:
            value = {
                "ordinal": i, "trajectory_id": t, "phase": e["phase"],
                "current_prompt_tokens": length,
                "previous_prompt_tokens": previous["event"]["chunk_aligned_prompt_tokens"],
                "gap_requests": i - previous["ordinal"] - 1,
                "gap_virtual_time": round(e["virtual_ready_time"] - previous["event"]["virtual_ready_time"], 9),
                "intervening_sessions": len(previous["maxima"]),
                "intervening_query_tokens": previous["query_sum"],
                "intervening_distinct_prefix_tokens": sum(previous["maxima"].values()),
                "reuse_upper_bound": min(length, previous["event"]["chunk_aligned_prompt_tokens"]) / length,
            }
        result.append(value)
        open_intervals[t] = {"event": e, "ordinal": i, "maxima": {}, "query_sum": 0}
        for other, state in open_intervals.items():
            if other != t:
                state["maxima"][t] = max(state["maxima"].get(t, 0), length)
                state["query_sum"] += length
    return result


def validate_pool(events, profiles, w):
    activation = profiles.copy()
    random.Random(20260725).shuffle(activation)
    active = {p["trajectory_id"]: 0 for p in activation[:w]}
    by_id = {p["trajectory_id"]: p for p in profiles}
    activated = min(w, len(profiles))
    for e in events:
        t = e["trajectory_id"]
        assert t in active
        assert e["active_sessions_at_emit"] == len(active) <= w
        assert e["unactivated_sessions_remaining"] == len(profiles) - activated
        assert e["step_index"] == active[t]
        active[t] += 1
        if active[t] == len(by_id[t]["steps"]):
            del active[t]
            if activated < len(profiles):
                active[activation[activated]["trajectory_id"]] = 0
                activated += 1
    assert not active and activated == 100


def serializable_order(events):
    return "".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in events).encode()


async def validate_dispatch(events):
    # Exercise the actual dispatcher with a CPU-only fake completion callback.
    # No HTTP transport is created. This checks strict order and dependencies.
    submitted, completed = [], set()

    async def send(e, mark_submitted):
        assert e["depends_on"] is None or e["depends_on"] in completed
        submitted.append(e["request_id"])
        mark_submitted()
        await asyncio.sleep(0)
        completed.add(e["request_id"])
        return {}

    report = await replay.dispatch_closed_loop(events, send, client_concurrency=32)
    assert submitted == [e["request_id"] for e in events]
    assert len(completed) == 3379 and not any(r.error for r in report.records)
    return {"requests": len(submitted), "client_concurrency": 32,
            "mock_peak_inflight": report.observed_max_inflight,
            "meaning": "CPU-only dependency/order validation; not a timing or effective-concurrency measurement"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--verify-content", action="store_true")
    parser.add_argument("--check-originals", action="store_true",
                        help="Also check the original host paths, when available")
    args = parser.parse_args()
    generated = args.generated.resolve()
    summary = load(generated / "generation.json")
    ds = bundle_path(summary["dataset"])
    config = load(ds / "replay-inputs.json")
    for key in ["runner", "order", "manifest", "profile", "parquet", "tools", "tokenizer"]:
        config[key] = str(bundle_path(config[key]))
    w = config["session_pool"]
    baseline = ROOT / "inputs/baseline"
    stem = "causal_w32_noprimer.seed-20260725"
    old = rows(baseline / f"{stem}.order.jsonl")
    profiles = rows(config["profile"])
    old_manifest = load(baseline / f"{stem}.manifest.json")
    locked = load(ROOT / "inputs/SOURCE-LOCK.json")
    for entry in locked:
        assert sha(bundle_path(entry["snapshot_path"])) == entry["sha256"]
        if args.check_originals:
            assert sha(entry["original_path"]) == entry["sha256"], entry["original_path"]
    events, manifest = replay.validate_closed_loop_workload(
        Path(config["order"]), Path(config["manifest"]), Path(config["profile"]),
        seed=config["seed"], session_pool=w, prompt_salt_prefix=config["prompt_salt_prefix"])
    replay.validate_concurrency(config["max_concurrency"])
    assert config["max_concurrency"] == 32 and len(events) == 3379
    assert Path(config["profile"]).read_bytes() == (baseline / f"{stem}.profiles.jsonl").read_bytes()
    changed_fields = {"ordinal", "virtual_ready_time", "active_sessions_at_emit", "unactivated_sessions_remaining"}
    old_map = {e["request_id"]: e for e in old}
    assert len(old_map) == len({e["request_id"] for e in events}) == len(events)
    assert set(old_map) == {e["request_id"] for e in events}
    previous = {}
    changed_counts = collections.Counter()
    for i, e in enumerate(events):
        before = old_map[e["request_id"]]
        assert set(e) == set(before)
        for key in e:
            if e[key] != before[key]:
                assert key in changed_fields, (e["request_id"], key)
                changed_counts[key] += 1
        assert e["ordinal"] == i
        assert e["depends_on"] == previous.get(e["trajectory_id"])
        assert e["chunk_aligned_prompt_tokens"] == e["prompt_tokens"] // 256 * 256
        previous[e["trajectory_id"]] = e["request_id"]
    assert len(previous) == 100
    old_sessions = collections.defaultdict(list)
    new_sessions = collections.defaultdict(list)
    for e in old: old_sessions[e["trajectory_id"]].append(e["request_id"])
    for e in events: new_sessions[e["trajectory_id"]].append(e["request_id"])
    assert old_sessions == new_sessions
    for name in ["virtual_delay", "prompt_salt_label", "primer_included", "seed", "algorithm"]:
        assert manifest["interleaving"][name] == old_manifest["interleaving"][name]
    for key, val in old_manifest["tokenization"].items():
        if key != "tokenizer": assert manifest["tokenization"][key] == val
    validate_pool(events, profiles, w)
    assert serializable_order(causal_merge(profiles, BuildConfig(active_sessions=w))) == Path(config["order"]).read_bytes()
    assert serializable_order(causal_merge(profiles, BuildConfig(active_sessions=32))) == (baseline / f"{stem}.order.jsonl").read_bytes()

    for label, es, interval_file in [("w32", old, baseline / f"{stem}.intervals.jsonl"),
                                    (f"w{w}", events, ds / f"causal_w{w}_noprimer.seed-20260725.intervals.jsonl")]:
        independent = independent_intervals(es)
        assert [r for r in independent if r is not None] == rows(interval_file)
        per_request = load(generated / "analysis" / f"{label}-per-request-wrd.json")
        assert len(per_request) == 3379
        for actual, expected in zip(per_request, independent):
            assert actual["wrd_tokens"] == (None if expected is None else expected["intervening_distinct_prefix_tokens"])
        assert sum(r is None for r in independent) == 100
    # Independent NumPy quantiles and streaming WRD verify EVERY search candidate.
    import numpy as np
    with (generated / "analysis/search-all-w.csv").open() as f:
        search = list(csv.DictReader(f))
    for record in search:
        cw = int(record["W"])
        ce = causal_merge(profiles, BuildConfig(active_sessions=cw))
        distances = independent_intervals(ce)
        validate_pool(ce, profiles, cw)
        for stage in ["all", "full_pool", "tail"]:
            select = [i for i, e in enumerate(ce) if stage == "all" or (e["active_sessions_at_emit"] == cw) == (stage == "full_pool")]
            vals = [distances[i]["intervening_distinct_prefix_tokens"] for i in select if distances[i] is not None]
            assert len(select) == int(record[f"{stage}_requests"])
            assert len(vals) == int(record[f"{stage}_finite_count"])
            assert len(select) - len(vals) == int(record[f"{stage}_first_visits"])
            assert sum(v <= 1_000_000 for v in vals) == int(record[f"{stage}_count_le_1000000"])
            assert sum(v == 0 for v in vals) == int(record[f"{stage}_zero_count"])
            if vals:
                assert abs(float(np.mean(vals)) - float(record[f"{stage}_mean_tokens"])) < 1e-6
                for q in [50, 90, 95, 99]:
                    assert abs(float(np.percentile(vals, q)) - float(record[f"{stage}_p{q}_tokens"])) < 1e-6
    result = {
        "success": True, "requests": 3379, "first_visits": 100, "finite_wrd": 3279,
        "selected_W": w, "only_changed_order_fields": dict(changed_counts),
        "input_originals_unchanged": True if args.check_originals else None,
        "source_snapshots_verified": True, "profiles_byte_identical": True,
        "all_request_metadata_unchanged": True, "session_sequence_dependencies_and_pool_verified": True,
        "independent_streaming_wrd_matches_all_intervals": True,
        "all_100_candidates_independently_verified": True,
        "regeneration_byte_identical": True, "original_replayer_parser_passed": True,
        "network_forbidden": True, "http_requests_sent": 0,
        "mock_dispatch": asyncio.run(validate_dispatch(events)),
        "content_tokenization": "not requested",
    }
    if args.verify_content:
        print("Loading all 100 original prompt sources using the unchanged replay tokenizer path", flush=True)
        sources, source_summary = replay.load_prompt_sources(
            events, parquet_path=Path(config["parquet"]), tools_path=Path(config["tools"]),
            tokenizer_path=Path(config["tokenizer"]), prompt_salt_prefix=config["prompt_salt_prefix"])
        hashes = []
        for e in events:
            old_e = old_map[e["request_id"]]
            new_tokens = sources[e["trajectory_id"]].prompt_for(e["request_index"])
            old_tokens = sources[old_e["trajectory_id"]].prompt_for(old_e["request_index"])
            assert new_tokens == old_tokens
            assert len(new_tokens) - len(sources[e["trajectory_id"]].salt_ids) == e["prompt_tokens"]
            token_array = array.array("I", new_tokens)
            if sys.byteorder != "little": token_array.byteswap()
            hashes.append({"request_id": e["request_id"], "tokens_with_salt": len(new_tokens),
                           "token_ids_u32le_sha256": hashlib.sha256(token_array.tobytes()).hexdigest()})
        result["content_tokenization"] = {"all_3379_prompts_equal": True, "source_summary": source_summary,
                                           "hash_encoding": "token IDs as little-endian unsigned 32-bit integers"}
        (generated / "validation").mkdir(exist_ok=True)
        (generated / "validation/prompt-token-hashes.jsonl").write_text("".join(json.dumps(h, sort_keys=True) + "\n" for h in hashes))
    (generated / "validation").mkdir(exist_ok=True)
    (generated / "validation/validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
