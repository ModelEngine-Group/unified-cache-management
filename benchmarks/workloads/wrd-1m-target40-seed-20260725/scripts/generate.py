#!/usr/bin/env python3
"""Change only W using the byte-reproduced original causal_merge function."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor"))
from build_swe_rebench_long_context import BuildConfig, causal_merge

STEM = "causal_w32_noprimer.seed-20260725"
BASE = ROOT / "inputs/baseline"
SEED = 20260725
THRESHOLD = 1_000_000
EXPECTED_ORDER = "9cf0b970efa0bc4bb1c755f0ab0cf6d8dfea67eb216c894603064f280eb99392"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def jsonl_bytes(rows):
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, p):
    x = (len(values) - 1) * p
    a = int(x)
    return values[a] + (values[min(a + 1, len(values) - 1)] - values[a]) * (x - a)


def stats(values):
    values = sorted(values)
    return {
        "finite_count": len(values),
        "count_le_1000000": sum(v <= THRESHOLD for v in values),
        "cdf_1000000": sum(v <= THRESHOLD for v in values) / len(values) if values else None,
        "zero_count": values.count(0),
        "mean_tokens": sum(values) / len(values) if values else None,
        "min_tokens": values[0] if values else None,
        "max_tokens": values[-1] if values else None,
        **{f"p{p}_tokens": quantile(values, p / 100) if values else None for p in (50, 90, 95, 99)},
    }


def compute_intervals(events, w):
    # Deliberately uses the literal definition, including a maximum rather than
    # the last observation. Validation uses a separate streaming implementation.
    last = {}
    intervals, requests = [], []
    for i, e in enumerate(events):
        t = e["trajectory_id"]
        previous = last.get(t)
        distance = None
        if previous is not None:
            middle = events[previous + 1:i]
            footprint = {}
            for m in middle:
                mt = m["trajectory_id"]
                footprint[mt] = max(footprint.get(mt, 0), m["chunk_aligned_prompt_tokens"])
            distance = sum(footprint.values())
            current = e["chunk_aligned_prompt_tokens"]
            prior = events[previous]["chunk_aligned_prompt_tokens"]
            intervals.append({
                "current_prompt_tokens": current,
                "gap_requests": i - previous - 1,
                "gap_virtual_time": round(e["virtual_ready_time"] - events[previous]["virtual_ready_time"], 9),
                "intervening_distinct_prefix_tokens": distance,
                "intervening_query_tokens": sum(m["chunk_aligned_prompt_tokens"] for m in middle),
                "intervening_sessions": len(footprint),
                "ordinal": i, "phase": e["phase"], "previous_prompt_tokens": prior,
                "reuse_upper_bound": min(prior, current) / current,
                "trajectory_id": t,
            })
        requests.append({
            "ordinal": i, "request_id": e["request_id"], "trajectory_id": t,
            "request_index": e["request_index"], "depends_on": e["depends_on"],
            "previous_ordinal": previous, "first_visit": previous is None,
            "prompt_tokens": e["prompt_tokens"],
            "chunk_aligned_prompt_tokens": e["chunk_aligned_prompt_tokens"],
            "wrd_tokens": distance,
            "le_1000000": None if distance is None else distance <= THRESHOLD,
            "active_sessions_at_emit": e["active_sessions_at_emit"],
            "unactivated_sessions_remaining": e["unactivated_sessions_remaining"],
            "stage": "full_pool" if e["active_sessions_at_emit"] == w else "tail",
        })
        last[t] = i
    return intervals, requests


def summarize(requests, w):
    result = {"W": w, "requests": len(requests), "first_visits": sum(r["first_visit"] for r in requests)}
    for name in ("all", "full_pool", "tail"):
        rows = requests if name == "all" else [r for r in requests if r["stage"] == name]
        values = [r["wrd_tokens"] for r in rows if r["wrd_tokens"] is not None]
        result[f"{name}_requests"] = len(rows)
        result[f"{name}_first_visits"] = sum(r["first_visit"] for r in rows)
        result.update({f"{name}_{k}": v for k, v in stats(values).items()})
    result["tail_start_ordinal"] = next((r["ordinal"] for r in requests if r["stage"] == "tail"), None)
    result["tail_finite_share"] = result["tail_finite_count"] / result["all_finite_count"]
    result["tail_effect_pp"] = (result["all_cdf_1000000"] - result["full_pool_cdf_1000000"]) * 100
    result["difference_from_target_pp"] = (result["all_cdf_1000000"] - .4) * 100
    return result


def file_record(path):
    return {"path": str(path.resolve()), "sha256": sha(path), "size_bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    profiles = read_jsonl(BASE / f"{STEM}.profiles.jsonl")
    baseline = read_jsonl(BASE / f"{STEM}.order.jsonl")
    original_manifest = json.loads((BASE / f"{STEM}.manifest.json").read_text())
    identity = json.loads((BASE / "BUNDLE-IDENTITY.json").read_text())
    assert sha(BASE / f"{STEM}.order.jsonl") == EXPECTED_ORDER
    for ext, key in [("profiles.jsonl", "profiles_sha256"), ("intervals.jsonl", "intervals_sha256"),
                     ("manifest.json", "manifest_file_sha256"), ("tier-model.jsonl", "tier_model_sha256")]:
        assert sha(BASE / f"{STEM}.{ext}") == identity["integrity"][key]
    assert causal_merge(profiles, BuildConfig(active_sessions=32)) == baseline
    assert hashlib.sha256(jsonl_bytes(baseline)).hexdigest() == EXPECTED_ORDER
    bi, br = compute_intervals(baseline, 32)
    assert bi == read_jsonl(BASE / f"{STEM}.intervals.jsonl")
    baseline_summary = summarize(br, 32)
    for key, expected in [("all_p50_tokens", 2002944), ("all_p90_tokens", 2620160), ("all_p99_tokens", 2934220.8)]:
        assert abs(baseline_summary[key] - expected) < 1e-6

    results = []
    for w in range(1, 101):
        events = causal_merge(profiles, BuildConfig(active_sessions=w))
        _, rows = compute_intervals(events, w)
        row = summarize(rows, w)
        assert (row["requests"], row["first_visits"], row["all_finite_count"]) == (3379, 100, 3279)
        results.append(row)
    best = min(results, key=lambda r: (abs(r["difference_from_target_pp"]), r["W"]))
    w = best["W"]
    events = causal_merge(profiles, BuildConfig(active_sessions=w))
    intervals, rows = compute_intervals(events, w)
    assert events == causal_merge(profiles, BuildConfig(active_sessions=w))
    out.mkdir(parents=True)
    analysis = out / "analysis"
    analysis.mkdir()
    dataset = out / f"closed-loop-w{w}-64k128k"
    dataset.mkdir()
    stem = f"causal_w{w}_noprimer.seed-{SEED}"
    paths = {k: dataset / f"{stem}.{suffix}" for k, suffix in (
        ("order", "order.jsonl"), ("profiles", "profiles.jsonl"),
        ("intervals", "intervals.jsonl"), ("manifest", "manifest.json"))}
    paths["order"].write_bytes(jsonl_bytes(events))
    paths["intervals"].write_bytes(jsonl_bytes(intervals))
    shutil.copyfile(BASE / f"{STEM}.profiles.jsonl", paths["profiles"])
    write_csv(analysis / "search-all-w.csv", results)
    for label, records in [("w32", br), (f"w{w}", rows)]:
        write_csv(analysis / f"{label}-per-request-wrd.csv", records)
        write_json(analysis / f"{label}-per-request-wrd.json", records)
    nonmonotone = [{"from_W": a["W"], "to_W": b["W"], "increase_pp": 100 * (b["all_cdf_1000000"] - a["all_cdf_1000000"])}
                   for a, b in zip(results, results[1:]) if b["all_cdf_1000000"] > a["all_cdf_1000000"]]
    comparison = {
        "baseline": baseline_summary, "selected": best,
        "target_cdf": .4, "target_interval": [.38, .42],
        "within_target": .38 <= best["all_cdf_1000000"] <= .42,
        "selection": "minimum absolute full-data CDF error; ties use smaller integer W",
        "nonmonotone_adjacent_steps": nonmonotone,
        "neighbors": [r for r in results if abs(r["W"] - w) <= 1],
    }
    write_json(analysis / "comparison.json", comparison)
    manifest = {
        "schema_version": 1,
        "interleaving": {**original_manifest["interleaving"], "active_sessions": w},
        "profile_summary": original_manifest["profile_summary"],
        "tokenization": {**original_manifest["tokenization"], "tokenizer": str(ROOT / "inputs/tokenizer")},
        "source": {
            "repo_id": original_manifest["source"]["repo_id"],
            "revision": original_manifest["source"]["revision"],
            "parquet": str(ROOT / "inputs/source-metadata/trajectories.selected100.sparse-index.parquet"),
            "parquet_sha256": sha(ROOT / "inputs/source-metadata/trajectories.selected100.sparse-index.parquet"),
            "tools": str(ROOT / "inputs/source-metadata/tools.json"),
            "reused_profiles": file_record(BASE / f"{STEM}.profiles.jsonl"),
            "baseline_manifest": file_record(BASE / f"{STEM}.manifest.json"),
            "source_lock": str(ROOT / "inputs/SOURCE-LOCK.json"),
        },
        "outputs": {k: file_record(p) for k, p in paths.items() if k != "manifest"},
        "interval_analysis": {
            "definition": "sum of maximum aligned unsalted prompt tokens per intervening trajectory, between consecutive visits to the same trajectory, excluding endpoints",
            "cdf_weight": "one per finite reuse; first visits excluded; real zero included",
            "quantiles": "linear interpolation at (n-1)*p",
            "full_pool_definition": "current event active_sessions_at_emit == W",
            "tail_definition": "current event active_sessions_at_emit < W",
            "summary": best,
        },
        "active_session_sensitivity_sweep": file_record(analysis / "search-all-w.csv"),
        "tier_model": {"included": False, "reason": "No tier prediction was computed; original W32 model is retained only as a provenance input."},
        "interpretation_limits": {
            "wrd": "Global submission-order session-prefix footprint proxy; not DP block/shard reuse distance, cache hit rate, or Early speedup.",
            "concurrency": "C32 unchanged; W is an active-session pool and reduced W can reduce effective runnable concurrency.",
            "virtual_time": "Synthetic ready times only determine order; replay waits for dependencies and has no arrival schedule.",
        },
        "generator": {**file_record(ROOT / "vendor/build_swe_rebench_long_context.py"), "entrypoint": "causal_merge(profiles, BuildConfig(active_sessions=W))", "baseline_order_byte_reproduced": True},
    }
    write_json(paths["manifest"], manifest)
    # The old W32 salt is intentionally unchanged: renaming it would change tokens.
    bundle = {
        "schema_version": 1, "purpose": "Closed-loop C32 replay with WRD-targeted fixed W",
        "client_protocol": {**identity["client_protocol"], "active_session_pool": w, "client_concurrency": 32},
        "workload": identity["workload"],
        "wrd": best,
        "integrity": {
            "order_sha256": sha(paths["order"]), "profiles_sha256": sha(paths["profiles"]),
            "intervals_sha256": sha(paths["intervals"]), "manifest_file_sha256": sha(paths["manifest"]),
            "compact_parquet_sha256": manifest["source"]["parquet_sha256"],
            "tokenizer_json_sha256": sha(ROOT / "inputs/tokenizer/tokenizer.json"),
            "chat_template_sha256": sha(ROOT / "inputs/tokenizer/chat_template.jinja"),
            "full_submit_order_sha256": hashlib.sha256(json.dumps(list(range(len(events))), separators=(",", ":")).encode()).hexdigest(),
            "full_submit_order_sha256_definition": "Inherited replay metric: compact JSON list of ordinals. Does NOT distinguish reordering request identities.",
            "request_id_sequence_sha256": hashlib.sha256("\n".join(e["request_id"] for e in events).encode()).hexdigest(),
            "request_id_sequence_sha256_definition": "UTF-8 request IDs joined by LF without terminal LF",
        },
        "paths": {k: str(p) for k, p in paths.items()},
    }
    write_json(dataset / "BUNDLE-IDENTITY.json", bundle)
    replay = {
        "runner": str(ROOT / "vendor/benchmark_swe_rebench_closed_loop.py"),
        "order": str(paths["order"]), "manifest": str(paths["manifest"]), "profile": str(paths["profiles"]),
        "parquet": manifest["source"]["parquet"], "tools": manifest["source"]["tools"],
        "tokenizer": manifest["tokenization"]["tokenizer"],
        "max_concurrency": 32, "session_pool": w, "seed": SEED,
        "prompt_salt_prefix": manifest["interleaving"]["prompt_salt_label"],
        "max_model_len": 200000, "max_output_tokens": 1, "request_timeout": 3600,
        "dependency_policy": "wait_for_depends_on_completion", "send_order": "strict_manifest_order",
    }
    write_json(dataset / "replay-inputs.json", replay)
    write_json(out / "generation.json", {
        "dataset": str(dataset), "selected_W": w, "baseline_summary": baseline_summary,
        "selected_summary": best, "baseline_exact_order_sha256": EXPECTED_ORDER,
        "baseline_intervals_exact_match": True, "repeat_generation_identical": True,
        "tier_model_generated": False, "http_requests_sent": 0,
    })
    print(json.dumps({"dataset": str(dataset), "W": w, "cdf": best["all_cdf_1000000"], "difference_pp": best["difference_from_target_pp"]}, indent=2))


if __name__ == "__main__":
    main()
