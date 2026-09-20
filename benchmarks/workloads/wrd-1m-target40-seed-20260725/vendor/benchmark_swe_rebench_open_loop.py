#!/usr/bin/env python3
"""Persisted open-loop replay for causal SWE-rebench workloads.

The module intentionally keeps schedule generation and dispatch independent
from the HTTP implementation so the timing policy can be tested without a
server or GPU.  It supports exact uniform spacing and fixed-seed Poisson
arrivals; both are generated once and then replayed byte-for-byte by A/B runs.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import copy
import hashlib
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


DEFAULT_RATE = 2.14
DEFAULT_SEED = 20260725
HTTP_READ_BUFSIZE = 8 * 1024 * 1024
DEFAULT_PARQUET = Path(
    "/data/datasets/glm51-vllm-lmcache/raw/"
    "swe_rebench_openhands_trajectories/trajectories.parquet"
)
DEFAULT_TOOLS = Path(
    "/data/datasets/glm51-vllm-lmcache/raw/"
    "swe_rebench_openhands_trajectories/tools.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "minimum": round(min(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "p50": round(percentile(values, 0.50), 6),
        "p95": round(percentile(values, 0.95), 6),
        "p99": round(percentile(values, 0.99), 6),
        "maximum": round(max(values), 6),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_order(events: Sequence[Mapping[str, Any]]) -> None:
    if not events:
        raise ValueError("request order is empty")
    if [event.get("ordinal") for event in events] != list(range(len(events))):
        raise ValueError("request ordinals must be contiguous and ordered")
    seen: set[str] = set()
    previous: dict[str, str] = {}
    for event in events:
        request_id = str(event["request_id"])
        trajectory_id = str(event["trajectory_id"])
        if request_id in seen:
            raise ValueError(f"duplicate request ID: {request_id}")
        expected = previous.get(trajectory_id)
        if event.get("depends_on") != expected:
            raise ValueError(
                f"dependency mismatch at ordinal {event['ordinal']}: "
                f"expected {expected!r}, found {event.get('depends_on')!r}"
            )
        seen.add(request_id)
        previous[trajectory_id] = request_id


def generate_uniform_schedule(
    events: Sequence[Mapping[str, Any]],
    *,
    offered_rate: float = DEFAULT_RATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate exact deterministic uniform nominal arrivals.

    Event ``i`` arrives at ``(i + 1) / offered_rate``.  Starting after one
    interval keeps the existing ``request_count / nominal_horizon`` rate
    definition exact while every adjacent arrival is separated by ``1/lambda``.
    """
    validate_order(events)
    if not math.isfinite(offered_rate) or offered_rate <= 0:
        raise ValueError("offered_rate must be finite and positive")
    interval = 1.0 / offered_rate
    schedule: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        elapsed = (index + 1) * interval
        schedule.append(
            {
                "ordinal": int(event["ordinal"]),
                "request_id": str(event["request_id"]),
                "nominal_arrival_s": round(elapsed, 12),
                "interarrival_s": round(interval, 12),
            }
        )
    return schedule, {
        "generator": "deterministic_uniform_spacing",
        "arrival_process": "deterministic_uniform",
        "configured_offered_rate_rps": offered_rate,
        "nominal_interarrival_s": interval,
        "jitter_s": 0.0,
        "event_count": len(events),
        "nominal_horizon_s": len(events) * interval,
        "nominal_realized_rate_rps": offered_rate,
        "nominal_rate_relative_error": 0.0,
        "interarrival_s": distribution([interval] * len(events)),
    }


def generate_poisson_schedule(
    events: Sequence[Mapping[str, Any]],
    *,
    offered_rate: float = DEFAULT_RATE,
    seed: int = DEFAULT_SEED,
    burstiness: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate an immutable seeded Poisson schedule without rate normalization.

    ``burstiness=1`` denotes exponential inter-arrivals (coefficient of
    variation one).  The sampled horizon is deliberately not rescaled to the
    configured rate: the manifest records the naturally realized nominal rate.
    """
    validate_order(events)
    if not math.isfinite(offered_rate) or offered_rate <= 0:
        raise ValueError("offered_rate must be finite and positive")
    if not math.isfinite(burstiness) or not math.isclose(
        burstiness, 1.0, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("Poisson schedule requires burstiness=1")
    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    interarrivals: list[float] = []
    elapsed = 0.0
    for event in events:
        uniform = rng.random()
        while uniform == 0.0:
            uniform = rng.random()
        interval = max(round(-math.log1p(-uniform) / offered_rate, 12), 1e-12)
        elapsed = round(elapsed + interval, 12)
        interarrivals.append(interval)
        schedule.append(
            {
                "ordinal": int(event["ordinal"]),
                "request_id": str(event["request_id"]),
                "nominal_arrival_s": elapsed,
                "interarrival_s": interval,
            }
        )
    realized_rate = len(events) / elapsed
    return schedule, {
        "generator": "seeded_poisson_exponential_v1",
        "arrival_process": "poisson",
        "configured_offered_rate_rps": offered_rate,
        "schedule_seed": int(seed),
        "burstiness": 1.0,
        "rng_algorithm": "python_random_mt19937_inverse_exponential_v1",
        "event_count": len(events),
        "nominal_horizon_s": elapsed,
        "nominal_realized_rate_rps": realized_rate,
        "nominal_rate_relative_error": abs(realized_rate - offered_rate) / offered_rate,
        "interarrival_s": distribution(interarrivals),
    }


def write_schedule(
    *,
    order_path: Path,
    schedule_path: Path,
    manifest_path: Path,
    offered_rate: float = DEFAULT_RATE,
    arrival_process: str = "deterministic_uniform",
    seed: int = DEFAULT_SEED,
    burstiness: float = 1.0,
) -> dict[str, Any]:
    """Persist a schedule and manifest, refusing to overwrite either file."""
    if schedule_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite schedule artifacts: {schedule_path}, "
            f"{manifest_path}"
        )
    events = load_jsonl(order_path)
    if arrival_process == "deterministic_uniform":
        schedule, metadata = generate_uniform_schedule(
            events,
            offered_rate=offered_rate,
        )
    elif arrival_process == "poisson":
        schedule, metadata = generate_poisson_schedule(
            events,
            offered_rate=offered_rate,
            seed=seed,
            burstiness=burstiness,
        )
    else:
        raise ValueError(f"unsupported arrival process: {arrival_process}")
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with schedule_path.open("x", encoding="utf-8") as handle:
        for row in schedule:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        **metadata,
        "order": {
            "path": str(order_path.resolve()),
            "sha256": sha256_file(order_path),
        },
        "schedule": {
            "path": str(schedule_path.resolve()),
            "sha256": sha256_file(schedule_path),
        },
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def load_schedule(
    *,
    order: Sequence[Mapping[str, Any]],
    order_path: Path,
    schedule_path: Path,
    manifest_path: Path,
    expected_offered_rate: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["order"]["sha256"] != sha256_file(order_path):
        raise ValueError("order SHA-256 does not match schedule manifest")
    if manifest["schedule"]["sha256"] != sha256_file(schedule_path):
        raise ValueError("schedule SHA-256 does not match schedule manifest")
    if expected_offered_rate is not None and not math.isclose(
        float(manifest["configured_offered_rate_rps"]),
        expected_offered_rate,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("configured offered rate does not match schedule manifest")
    schedule = load_jsonl(schedule_path)
    if len(schedule) != len(order):
        raise ValueError("schedule and order event counts differ")
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported schedule manifest schema version")
    configured_rate = float(manifest["configured_offered_rate_rps"])
    if int(manifest.get("event_count", -1)) != len(order):
        raise ValueError("schedule manifest event count mismatch")
    process = manifest.get("arrival_process")
    if process == "deterministic_uniform":
        if manifest.get("generator") != "deterministic_uniform_spacing":
            raise ValueError("uniform schedule manifest generator mismatch")
        interval = float(manifest.get("nominal_interarrival_s", math.nan))
        if not math.isfinite(interval) or not math.isclose(
            interval, 1.0 / configured_rate, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                "schedule manifest nominal interval is not 1 / offered rate"
            )
        if float(manifest.get("jitter_s", math.nan)) != 0.0:
            raise ValueError("deterministic schedule must have zero jitter")
        expected_schedule, expected_metadata = generate_uniform_schedule(
            order, offered_rate=configured_rate
        )
    elif process == "poisson":
        if manifest.get("generator") != "seeded_poisson_exponential_v1":
            raise ValueError("Poisson schedule manifest generator mismatch")
        if manifest.get("rng_algorithm") != (
            "python_random_mt19937_inverse_exponential_v1"
        ):
            raise ValueError("Poisson schedule RNG algorithm mismatch")
        schedule_seed = int(manifest.get("schedule_seed", -1))
        burstiness = float(manifest.get("burstiness", math.nan))
        if not math.isclose(burstiness, 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Poisson schedule requires burstiness=1")
        expected_schedule, expected_metadata = generate_poisson_schedule(
            order,
            offered_rate=configured_rate,
            seed=schedule_seed,
            burstiness=burstiness,
        )
    else:
        raise ValueError(f"unsupported schedule arrival process: {process}")
    expected_horizon = float(expected_metadata["nominal_horizon_s"])
    if not math.isclose(
        float(manifest.get("nominal_horizon_s", math.nan)),
        expected_horizon,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("schedule manifest nominal horizon mismatch")
    if not math.isclose(
        float(manifest.get("nominal_realized_rate_rps", math.nan)),
        float(expected_metadata["nominal_realized_rate_rps"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("schedule manifest nominal realized rate mismatch")
    if not math.isclose(
        float(manifest.get("nominal_rate_relative_error", math.nan)),
        float(expected_metadata["nominal_rate_relative_error"]),
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("schedule manifest nominal rate error mismatch")
    previous_arrival = -math.inf
    for index, (event, arrival, expected) in enumerate(
        zip(order, schedule, expected_schedule, strict=True)
    ):
        if arrival.get("ordinal") != event.get("ordinal"):
            raise ValueError("schedule ordinal does not match order")
        if str(arrival.get("request_id")) != str(event.get("request_id")):
            raise ValueError("schedule request ID does not match order")
        nominal = float(arrival["nominal_arrival_s"])
        if not math.isfinite(nominal) or nominal < previous_arrival:
            raise ValueError("schedule arrivals must be finite and nondecreasing")
        if not math.isclose(
            nominal,
            float(expected["nominal_arrival_s"]),
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"schedule nominal arrival mismatch at ordinal {index}")
        if not math.isclose(
            float(arrival["interarrival_s"]),
            float(expected["interarrival_s"]),
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"schedule interarrival mismatch at ordinal {index}")
        previous_arrival = nominal
    return schedule, manifest


def _manifest_canonical_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = copy.deepcopy(dict(manifest))
    canonical.setdefault("integrity", {}).pop("manifest_canonical_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_workload_artifacts(
    order_path: Path,
    manifest_path: Path,
    profile_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order = load_jsonl(order_path)
    validate_order(order)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity = manifest.get("integrity", {})
    if integrity.get("order_canonical_sha256") != sha256_file(order_path):
        raise ValueError("workload manifest/order SHA-256 mismatch")
    if integrity.get("profiles_canonical_sha256") != sha256_file(profile_path):
        raise ValueError("workload manifest/profile SHA-256 mismatch")
    expected_manifest_hash = integrity.get("manifest_canonical_sha256")
    if expected_manifest_hash and expected_manifest_hash != _manifest_canonical_sha256(
        manifest
    ):
        raise ValueError("workload manifest canonical SHA-256 mismatch")
    workload = manifest.get("workload", {})
    if int(workload.get("requests", -1)) != len(order):
        raise ValueError("workload manifest request count mismatch")
    return order, manifest


def normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = str(message["role"])
    normalized: dict[str, Any] = {
        "role": role,
        "content": message.get("content") or "",
    }
    if role == "assistant" and message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            function = call["function"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": arguments}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            calls.append(
                {
                    "id": call.get("id"),
                    "type": call.get("type") or "function",
                    "function": {
                        "name": function["name"],
                        "arguments": arguments,
                    },
                }
            )
        normalized["tool_calls"] = calls
    if role == "tool":
        normalized["name"] = message.get("name")
        normalized["tool_call_id"] = message.get("tool_call_id")
    return normalized


def subsequence_ends(values: Sequence[int], marker: Sequence[int]) -> list[int]:
    return [
        start + len(marker)
        for start in range(len(values) - len(marker) + 1)
        if values[start : start + len(marker)] == marker
    ]


@dataclass
class PromptSource:
    token_ids: array.array
    ends: dict[int, int]
    salt_ids: list[int]

    def prompt_for(self, request_index: int) -> list[int]:
        return [*self.salt_ids, *self.token_ids[: self.ends[request_index]]]


def _load_selected_parquet_rows(
    parquet: Any, selected_source_rows: set[int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    global_start = 0
    for row_group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(row_group).num_rows
        group_end = global_start + count
        wanted = {
            value for value in selected_source_rows if global_start <= value < group_end
        }
        if wanted:
            offset = 0
            for batch in parquet.iter_batches(
                batch_size=32,
                columns=["trajectory_id", "trajectory"],
                row_groups=[row_group],
            ):
                for row in batch.to_pylist():
                    source_row = global_start + offset
                    offset += 1
                    if source_row in wanted:
                        selected.append({**row, "source_row": source_row})
        global_start = group_end
    if len(selected) != len(selected_source_rows):
        raise RuntimeError(
            f"loaded {len(selected)} source rows, expected {len(selected_source_rows)}"
        )
    return selected


def load_prompt_sources(
    order: Sequence[Mapping[str, Any]],
    *,
    parquet_path: Path,
    tools_path: Path,
    tokenizer_path: Path,
    prompt_salt_prefix: str,
) -> tuple[dict[str, PromptSource], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError("runner requires pyarrow and transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    marker = tokenizer.encode("<|assistant|></think>", add_special_tokens=False)
    by_trajectory: dict[str, list[Mapping[str, Any]]] = {}
    for event in order:
        by_trajectory.setdefault(str(event["trajectory_id"]), []).append(event)
    rows = _load_selected_parquet_rows(
        pq.ParquetFile(parquet_path), {int(event["source_row"]) for event in order}
    )
    sources: dict[str, PromptSource] = {}
    body_lengths: list[int] = []
    salted_lengths: list[int] = []
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        messages = [normalize_message(message) for message in row["trajectory"]]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids: Any = (
            tokenized["input_ids"]
            if hasattr(tokenized, "keys") and "input_ids" in tokenized
            else tokenized
        )
        if hasattr(full_ids, "tolist"):
            full_ids = full_ids.tolist()
        ends = subsequence_ends(full_ids, marker)
        selected_ends: dict[int, int] = {}
        salt = (
            f"<|workload_namespace|>{prompt_salt_prefix}|"
            f"trajectory={trajectory_id}\n"
        )
        salt_ids = tokenizer.encode(salt, add_special_tokens=False)
        for event in by_trajectory[trajectory_id]:
            request_index = int(event["request_index"])
            if request_index >= len(ends):
                raise ValueError(
                    f"assistant boundary missing for {event['request_id']}"
                )
            body = int(ends[request_index])
            expected = int(event.get("body_prompt_tokens", event["prompt_tokens"]))
            if body != expected:
                raise ValueError(
                    f"body token mismatch for {event['request_id']}: {body} != {expected}"
                )
            expected_salted = int(
                event.get("salted_prompt_tokens", body + len(salt_ids))
            )
            if body + len(salt_ids) != expected_salted:
                raise ValueError(f"salted token mismatch for {event['request_id']}")
            selected_ends[request_index] = body
            body_lengths.append(body)
            salted_lengths.append(expected_salted)
        sources[trajectory_id] = PromptSource(
            token_ids=array.array("I", full_ids),
            ends=selected_ends,
            salt_ids=salt_ids,
        )
    if set(sources) != set(by_trajectory):
        raise ValueError("tokenized trajectory set does not match order")
    return sources, {
        "trajectory_count": len(sources),
        "body_prompt_tokens": distribution(body_lengths),
        "salted_prompt_tokens": distribution(salted_lengths),
        "tokenizer": str(tokenizer_path.resolve()),
        "prompt_salt_scope": "trajectory",
    }


@dataclass(frozen=True)
class DispatchTiming:
    ordinal: int
    nominal_arrival_s: float
    dependency_ready_s: float
    dispatch_started_s: float


@dataclass
class DispatchRecord:
    ordinal: int
    request_id: str
    trajectory_id: str
    depends_on: str | None
    nominal_arrival_s: float
    arrived_s: float | None = None
    arrival_lag_ms: float | None = None
    dependency_ready_s: float | None = None
    actual_submit_s: float | None = None
    schedule_lag_ms: float | None = None
    dependency_deferral_ms: float | None = None
    completion_s: float | None = None
    total_ms: float | None = None
    inflight_after_submit: int | None = None
    inflight_before_completion: int | None = None
    backlog_at_submit: int | None = None
    backlog_after_completion: int | None = None
    submission_sequence: int | None = None
    completion_sequence: int | None = None
    response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DispatchReport:
    records: list[DispatchRecord]
    state_timeline: list[dict[str, Any]]
    safety_cap: int
    cap_triggered: bool
    cap_trigger: dict[str, Any] | None
    origin: float
    wall_seconds: float


class SafetyCapExceeded(RuntimeError):
    def __init__(self, message: str, report: DispatchReport) -> None:
        super().__init__(message)
        self.report = report


Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
SendEvent = Callable[
    [Mapping[str, Any], DispatchTiming, Callable[[float | None], None]],
    Awaitable[Mapping[str, Any] | None],
]
RecordCallback = Callable[[DispatchRecord], None]


async def dispatch_open_loop(
    events: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    send_event: SendEvent,
    *,
    safety_cap: int,
    clock: Clock = time.perf_counter,
    sleep: Sleep = asyncio.sleep,
    record_callback: RecordCallback | None = None,
    progress_interval_s: float | None = None,
) -> DispatchReport:
    """Dispatch arrivals by nominal time, waiting only on direct dependencies."""
    validate_order(events)
    if safety_cap < 1:
        raise ValueError("safety_cap must be positive")
    if progress_interval_s is not None and progress_interval_s <= 0:
        raise ValueError("progress_interval_s must be positive")
    if len(events) != len(schedule):
        raise ValueError("schedule and event counts differ")
    for event, arrival in zip(events, schedule, strict=True):
        if int(event["ordinal"]) != int(arrival["ordinal"]):
            raise ValueError("schedule ordinal does not match event")
        if str(event["request_id"]) != str(arrival["request_id"]):
            raise ValueError("schedule request ID does not match event")

    origin = clock()
    loop = asyncio.get_running_loop()
    completion = {str(event["request_id"]): loop.create_future() for event in events}
    records = [
        DispatchRecord(
            ordinal=int(event["ordinal"]),
            request_id=str(event["request_id"]),
            trajectory_id=str(event["trajectory_id"]),
            depends_on=(
                None if event.get("depends_on") is None else str(event["depends_on"])
            ),
            nominal_arrival_s=float(arrival["nominal_arrival_s"]),
        )
        for event, arrival in zip(events, schedule, strict=True)
    ]
    state = {
        "arrived": 0,
        "submitted": 0,
        "dependency_waiting": 0,
        "outstanding": 0,
        "in_flight": 0,
        "completed": 0,
    }
    timeline: list[dict[str, Any]] = []
    submit_ordinals: list[int] = []
    completion_counter = 0
    cap_trigger: dict[str, Any] | None = None

    def relative_now() -> float:
        return max(clock() - origin, 0.0)

    def snapshot(reason: str, ordinal: int) -> None:
        timeline.append(
            {
                "time_s": round(relative_now(), 9),
                "reason": reason,
                "ordinal": ordinal,
                **state,
                "submit_pending": state["outstanding"] - state["in_flight"],
                "backlog": state["arrived"] - state["submitted"],
                "system_outstanding": state["arrived"] - state["completed"],
            }
        )

    def emit_progress(stage: str) -> None:
        value = {
            "stage": stage,
            "elapsed_s": round(relative_now(), 3),
            **state,
            "backlog": state["arrived"] - state["submitted"],
            "system_outstanding": state["arrived"] - state["completed"],
            "safety_cap": safety_cap,
        }
        print(
            "OPEN_LOOP_PROGRESS "
            + json.dumps(value, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    progress_stop = asyncio.Event()

    async def progress_loop() -> None:
        assert progress_interval_s is not None
        while not progress_stop.is_set():
            try:
                await asyncio.wait_for(
                    progress_stop.wait(), timeout=progress_interval_s
                )
            except TimeoutError:
                snapshot("progress", -1)
                emit_progress("running")

    async def handle_one(
        event: Mapping[str, Any],
        arrival: Mapping[str, Any],
        record: DispatchRecord,
    ) -> None:
        nonlocal completion_counter, cap_trigger
        nominal = float(arrival["nominal_arrival_s"])
        delay = origin + nominal - clock()
        if delay > 0:
            await sleep(delay)
        record.arrived_s = relative_now()
        record.arrival_lag_ms = max((record.arrived_s - nominal) * 1000, 0.0)
        state["arrived"] += 1
        snapshot("arrival", record.ordinal)

        dependency = record.depends_on
        if dependency is not None and not completion[dependency].done():
            state["dependency_waiting"] += 1
            snapshot("dependency_wait", record.ordinal)
            await completion[dependency]
            state["dependency_waiting"] -= 1
        record.dependency_ready_s = relative_now()
        record.dependency_deferral_ms = max(
            (record.dependency_ready_s - (record.arrived_s or nominal)) * 1000,
            0.0,
        )
        snapshot("dependency_ready", record.ordinal)

        if state["outstanding"] >= safety_cap:
            cap_trigger = {
                "ordinal": record.ordinal,
                "request_id": record.request_id,
                "time_s": round(relative_now(), 9),
                "outstanding": state["outstanding"],
                "safety_cap": safety_cap,
            }
            snapshot("safety_cap", record.ordinal)
            raise RuntimeError(
                f"safety cap {safety_cap} reached before {record.request_id}; "
                "open-loop replay aborted instead of throttling"
            )
        state["outstanding"] += 1
        dispatch_started = relative_now()
        submitted = False
        submitted_signal = asyncio.Event()
        submit_lock = threading.Lock()
        submit_requested = False

        def mark_submitted(at: float | None = None) -> None:
            """Thread-safe bridge for socket-send callbacks."""
            nonlocal submit_requested
            with submit_lock:
                if submit_requested:
                    raise RuntimeError(
                        f"submit boundary repeated for {record.request_id}"
                    )
                submit_requested = True

            def register() -> None:
                nonlocal submitted
                submitted = True
                absolute = clock() if at is None else at
                record.actual_submit_s = max(absolute - origin, 0.0)
                record.schedule_lag_ms = max(
                    (record.actual_submit_s - nominal) * 1000, 0.0
                )
                record.submission_sequence = len(submit_ordinals)
                submit_ordinals.append(record.ordinal)
                state["submitted"] += 1
                state["in_flight"] += 1
                record.inflight_after_submit = state["in_flight"]
                record.backlog_at_submit = state["arrived"] - state["submitted"]
                snapshot("submit", record.ordinal)
                submitted_signal.set()

            loop.call_soon_threadsafe(register)

        try:
            timing = DispatchTiming(
                ordinal=record.ordinal,
                nominal_arrival_s=nominal,
                dependency_ready_s=record.dependency_ready_s,
                dispatch_started_s=dispatch_started,
            )
            response = await send_event(event, timing, mark_submitted)
            # A callback from an executor thread may be queued immediately
            # before send_event returns. Yield once so registration is visible.
            if submit_requested and not submitted_signal.is_set():
                await submitted_signal.wait()
            if not submitted:
                raise RuntimeError(
                    f"send_event returned without submit boundary: {record.request_id}"
                )
            if response is not None:
                record.response = dict(response)
        except asyncio.CancelledError:
            record.error = "CancelledError: open-loop dispatcher aborted"
            raise
        except Exception as error:  # per-request errors remain auditable
            record.error = f"{type(error).__name__}: {error}"
        finally:
            completed_at = relative_now()
            record.completion_s = completed_at
            record.completion_sequence = completion_counter
            completion_counter += 1
            if submitted:
                record.total_ms = max(
                    (completed_at - (record.actual_submit_s or 0.0)) * 1000, 0.0
                )
                record.inflight_before_completion = state["in_flight"]
                state["in_flight"] -= 1
            state["outstanding"] -= 1
            state["completed"] += 1
            record.backlog_after_completion = state["arrived"] - state["submitted"]
            snapshot("completion", record.ordinal)
            future = completion[record.request_id]
            if not future.done():
                future.set_result(completed_at)
            if record_callback is not None:
                record_callback(record)

    tasks = [
        asyncio.create_task(handle_one(event, arrival, record))
        for event, arrival, record in zip(events, schedule, records, strict=True)
    ]
    progress_task = (
        asyncio.create_task(progress_loop())
        if progress_interval_s is not None
        else None
    )
    fatal: BaseException | None = None
    try:
        await asyncio.gather(*tasks)
    except BaseException as error:
        fatal = error
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        progress_stop.set()
        if progress_task is not None:
            await progress_task
            emit_progress("finished")

    report = DispatchReport(
        records=records,
        state_timeline=timeline,
        safety_cap=safety_cap,
        cap_triggered=cap_trigger is not None,
        cap_trigger=cap_trigger,
        origin=origin,
        wall_seconds=max(clock() - origin, 0.0),
    )
    if fatal is not None:
        if cap_trigger is not None:
            raise SafetyCapExceeded(str(fatal), report) from fatal
        raise fatal
    return report


def summarize_dispatch(
    report: DispatchReport,
    *,
    configured_offered_rate: float,
    schedule_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    submitted = [r for r in report.records if r.actual_submit_s is not None]
    completed = [r for r in report.records if r.completion_s is not None]
    successes = [r for r in completed if r.error is None]
    nominal_horizon = max(r.nominal_arrival_s for r in report.records)
    last_submit = max((r.actual_submit_s or 0.0) for r in submitted) if submitted else 0
    last_completion = (
        max((r.completion_s or 0.0) for r in completed) if completed else 0
    )
    lag_ms = [
        float(r.schedule_lag_ms) for r in submitted if r.schedule_lag_ms is not None
    ]
    arrival_lag_ms = [
        float(r.arrival_lag_ms) for r in report.records if r.arrival_lag_ms is not None
    ]
    deferral_ms = [
        float(r.dependency_deferral_ms)
        for r in report.records
        if r.dependency_deferral_ms is not None
    ]
    in_flight = [int(row["in_flight"]) for row in report.state_timeline]
    backlog = [int(row["backlog"]) for row in report.state_timeline]
    submit_order = [
        r.ordinal
        for r in sorted(submitted, key=lambda item: int(item.submission_sequence or 0))
    ]
    submit_hash = hashlib.sha256(
        json.dumps(submit_order, separators=(",", ":")).encode()
    ).hexdigest()
    halfway = nominal_horizon / 2
    three_quarter = nominal_horizon * 0.75
    submitted_by_nominal_end = sum(
        r.actual_submit_s is not None and r.actual_submit_s <= nominal_horizon
        for r in report.records
    )
    submitted_by_half = sum(
        r.actual_submit_s is not None and r.actual_submit_s <= halfway
        for r in report.records
    )
    submitted_by_three_quarter = sum(
        r.actual_submit_s is not None and r.actual_submit_s <= three_quarter
        for r in report.records
    )
    completed_by_half = sum(
        r.completion_s is not None and r.completion_s <= halfway for r in report.records
    )
    completed_by_three_quarter = sum(
        r.completion_s is not None and r.completion_s <= three_quarter
        for r in report.records
    )
    nominal_arrived_by_half = sum(
        r.nominal_arrival_s <= halfway for r in report.records
    )
    nominal_arrived_by_three_quarter = sum(
        r.nominal_arrival_s <= three_quarter for r in report.records
    )
    # Nominal arrivals are offered independently of event-loop wake-up lag.
    # Therefore this definition also exposes client-side scheduling stalls.
    backlog_at_half = nominal_arrived_by_half - submitted_by_half
    backlog_at_three_quarter = (
        nominal_arrived_by_three_quarter - submitted_by_three_quarter
    )
    backlog_at_nominal_end = len(report.records) - submitted_by_nominal_end
    backlog_second_half_growth = backlog_at_nominal_end - backlog_at_half
    backlog_second_half_slope = (
        backlog_second_half_growth / (nominal_horizon - halfway)
        if nominal_horizon > halfway
        else math.nan
    )
    completed_by_nominal_end = sum(
        r.completion_s is not None and r.completion_s <= nominal_horizon
        for r in report.records
    )
    system_outstanding_at_half = nominal_arrived_by_half - completed_by_half
    system_outstanding_at_three_quarter = (
        nominal_arrived_by_three_quarter - completed_by_three_quarter
    )
    system_outstanding_at_nominal_end = len(report.records) - completed_by_nominal_end
    system_outstanding_second_half_growth = (
        system_outstanding_at_nominal_end - system_outstanding_at_half
    )
    system_outstanding_second_half_slope = (
        system_outstanding_second_half_growth / (nominal_horizon - halfway)
        if nominal_horizon > halfway
        else math.nan
    )
    offered_window_submit_rate = submitted_by_nominal_end / nominal_horizon
    offered_window_completion_rate = completed_by_nominal_end / nominal_horizon
    level_threshold = max(8, math.ceil(0.005 * len(report.records)))
    growth_threshold = max(4, math.ceil(0.002 * len(report.records)))
    client_sustained_growth = (
        backlog_second_half_growth >= growth_threshold
        and backlog_at_three_quarter > backlog_at_half
        and backlog_at_nominal_end > backlog_at_three_quarter
    )
    completion_sustained_growth = (
        system_outstanding_second_half_growth >= growth_threshold
        and system_outstanding_at_three_quarter > system_outstanding_at_half
        and system_outstanding_at_nominal_end > system_outstanding_at_three_quarter
    )
    client_overload = (
        backlog_at_nominal_end >= level_threshold and client_sustained_growth
    )
    completion_overload = (
        system_outstanding_at_nominal_end >= level_threshold
        and completion_sustained_growth
    )
    overload = client_overload or completion_overload
    actual_submit_rate = len(submitted) / last_submit if last_submit > 0 else math.nan
    completion_rate = (
        len(successes) / last_completion if last_completion > 0 else math.nan
    )
    submit_rate_error = (
        abs(offered_window_submit_rate - configured_offered_rate)
        / configured_offered_rate
    )
    valid_steady_state = (
        not report.cap_triggered
        and not overload
        and len(submitted) == len(report.records)
        and not any(r.error for r in report.records)
        and submit_rate_error <= 0.05
    )
    return {
        "configured_offered_rate_rps": configured_offered_rate,
        "offered_rate_rps": configured_offered_rate,
        "nominal_realized_rate_rps": len(report.records) / nominal_horizon,
        "actual_submit_rate_rps": actual_submit_rate,
        "actual_submit_rate_during_nominal_window_rps": offered_window_submit_rate,
        "completion_throughput_rps": completion_rate,
        "completion_rate_during_nominal_window_rps": offered_window_completion_rate,
        "request_count": len(report.records),
        "submitted_requests": len(submitted),
        "successful_requests": len(successes),
        "errors": len(completed) - len(successes),
        "wall_seconds": report.wall_seconds,
        "schedule_lag_ms": distribution(lag_ms),
        "arrival_lag_ms": distribution(arrival_lag_ms),
        "dependency_deferral_ms": distribution(deferral_ms),
        "in_flight": distribution(in_flight),
        "backlog": distribution(backlog),
        "max_in_flight": max(in_flight, default=0),
        "max_backlog": max(backlog, default=0),
        "final_backlog": backlog[-1] if backlog else 0,
        "backlog_at_nominal_end": backlog_at_nominal_end,
        "backlog_at_nominal_half": backlog_at_half,
        "backlog_at_nominal_three_quarter": backlog_at_three_quarter,
        "backlog_second_half_growth": backlog_second_half_growth,
        "backlog_second_half_slope_rps": backlog_second_half_slope,
        "system_outstanding_at_nominal_half": system_outstanding_at_half,
        "system_outstanding_at_nominal_three_quarter": (
            system_outstanding_at_three_quarter
        ),
        "system_outstanding_at_nominal_end": system_outstanding_at_nominal_end,
        "system_outstanding_second_half_growth": system_outstanding_second_half_growth,
        "system_outstanding_second_half_slope_rps": (
            system_outstanding_second_half_slope
        ),
        "submitted_by_nominal_end": submitted_by_nominal_end,
        "completed_by_nominal_half": completed_by_half,
        "completed_by_nominal_end": completed_by_nominal_end,
        "nominal_horizon_s": nominal_horizon,
        "whole_run_last_submit_s": last_submit,
        "whole_run_last_completion_s": last_completion,
        "safety_cap": report.safety_cap,
        "cap_triggered": report.cap_triggered,
        "safety_cap_triggered": report.cap_triggered,
        "cap_trigger": report.cap_trigger,
        "overload": overload,
        "client_submission_overload": client_overload,
        "client_sustained_backlog_growth": client_sustained_growth,
        "completion_overload": completion_overload,
        "completion_sustained_outstanding_growth": completion_sustained_growth,
        "valid_steady_state_rate_experiment": valid_steady_state,
        "actual_submit_order_sha256": submit_hash,
        "schedule_sha256": schedule_manifest["schedule"]["sha256"],
    }


def request_summary(
    records: Sequence[DispatchRecord], wall_seconds: float
) -> dict[str, Any]:
    successful = [
        record
        for record in records
        if record.error is None
        and record.actual_submit_s is not None
        and record.completion_s is not None
    ]
    ttft_ms = [
        float(record.response["ttft_ms"])
        for record in successful
        if record.response.get("ttft_ms") is not None
    ]
    prompt_tokens = sum(
        int(record.response.get("prompt_tokens", 0)) for record in successful
    )
    output_tokens = sum(
        int(record.response.get("output_tokens", 0)) for record in successful
    )
    return {
        "requests": len(records),
        "successful_requests": len(successful),
        "errors": len(records) - len(successful),
        "wall_seconds": round(wall_seconds, 6),
        "completion_throughput_rps": (
            len(successful) / wall_seconds if wall_seconds > 0 else math.nan
        ),
        "request_throughput_rps": (
            len(successful) / wall_seconds if wall_seconds > 0 else math.nan
        ),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_mean_ms": statistics.fmean(ttft_ms) if ttft_ms else math.nan,
        "ttft_p50_ms": percentile(ttft_ms, 0.50),
        "ttft_p95_ms": percentile(ttft_ms, 0.95),
        "ttft_p99_ms": percentile(ttft_ms, 0.99),
    }


def _serialized_record(record: DispatchRecord) -> dict[str, Any]:
    value = asdict(record)
    response = value.pop("response")
    value.update(response)
    return value


class AtomicJsonlTrace:
    """Write a new trace through a private file and atomically publish it."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.temporary = target.with_suffix(target.suffix + ".tmp")
        if self.target.exists() or self.temporary.exists():
            raise FileExistsError(
                f"refusing to overwrite trace artifacts: {self.target}, "
                f"{self.temporary}"
            )
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("x", encoding="utf-8")
        self._closed = False

    def append(self, record: DispatchRecord) -> None:
        if self._closed:
            raise RuntimeError("trace writer is already closed")
        self._handle.write(
            json.dumps(
                _serialized_record(record),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        # target did not exist at construction and each run owns a fresh path.
        self.temporary.replace(self.target)
        self._closed = True


async def _health_check(session: Any, base_url: str) -> None:
    async with session.get(f"{base_url.rstrip('/')}/health") as response:
        if response.status != 200:
            body = (await response.text())[:1000]
            raise RuntimeError(
                f"health endpoint returned HTTP {response.status}: {body}"
            )


def create_http_session(
    aiohttp_module: Any,
    *,
    timeout_seconds: float,
    trace_configs: Sequence[Any] = (),
) -> Any:
    """Create the unbounded client with room for 128k-token SSE lines."""
    timeout = aiohttp_module.ClientTimeout(total=timeout_seconds)
    connector = aiohttp_module.TCPConnector(
        limit=0,
        limit_per_host=0,
        force_close=True,
    )
    return aiohttp_module.ClientSession(
        timeout=timeout,
        connector=connector,
        trace_configs=list(trace_configs),
        # return_token_ids can put every prompt token ID in one SSE JSON line.
        read_bufsize=HTTP_READ_BUFSIZE,
    )


async def consume_completion_sse(
    content: Any,
    *,
    clock: Clock,
) -> tuple[float | None, int, int]:
    """Consume a vLLM completion SSE stream and return timing plus usage."""
    first_token_at: float | None = None
    prompt_tokens = 0
    output_tokens = 0
    while not content.at_eof():
        raw_line = await content.readline()
        if not raw_line:
            break
        line = raw_line.decode(errors="replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        item = json.loads(line[6:])
        usage = item.get("usage")
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
            output_tokens = int(usage.get("completion_tokens", output_tokens))
        choices = item.get("choices") or []
        token_ids = choices[0].get("token_ids") or [] if choices else []
        text_value = choices[0].get("text") if choices else None
        if first_token_at is None and (token_ids or text_value):
            first_token_at = clock()
    return first_token_at, prompt_tokens, output_tokens


async def replay(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    if args.trace_output == args.output:
        raise ValueError("--trace-output and --output must be different paths")
    if args.max_output_tokens != 1:
        raise ValueError("this workload requires --max-output-tokens=1")
    if args.max_model_len != 200_000:
        raise ValueError("this experiment requires --max-model-len=200000")
    if args.safety_cap < 1:
        raise ValueError("--safety-cap must be positive")
    if args.request_limit is not None and args.request_limit < 1:
        raise ValueError("--request-limit must be positive")

    full_order, workload_manifest = validate_workload_artifacts(
        args.order, args.manifest, args.profile
    )
    full_schedule, schedule_manifest = load_schedule(
        order=full_order,
        order_path=args.order,
        schedule_path=args.schedule,
        manifest_path=args.schedule_manifest,
        expected_offered_rate=args.expected_offered_rate,
    )
    arrival_process = schedule_manifest.get("arrival_process")
    if arrival_process == "deterministic_uniform":
        if not math.isclose(
            float(schedule_manifest["nominal_interarrival_s"]),
            1.0 / args.expected_offered_rate,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("uniform replay interval is not 1 / offered rate")
        if float(schedule_manifest.get("jitter_s", math.nan)) != 0.0:
            raise ValueError("uniform replay requires zero arrival jitter")
    elif arrival_process == "poisson":
        if int(schedule_manifest.get("schedule_seed", -1)) != args.seed:
            raise ValueError("Poisson schedule seed does not match runner seed")
        if not math.isclose(
            float(schedule_manifest.get("burstiness", math.nan)),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Poisson replay requires burstiness=1")
    else:
        raise ValueError(f"unsupported formal arrival process: {arrival_process}")
    workload = workload_manifest["workload"]
    if int(workload["seed"]) != args.seed:
        raise ValueError("runner seed does not match workload seed")
    if int(workload["active_session_pool"]) != args.session_pool:
        raise ValueError("--session-pool does not match workload manifest")
    expected_salt = str(workload["prompt_salt_label"])
    if args.prompt_salt_prefix != expected_salt:
        raise ValueError("prompt salt prefix does not match workload manifest")
    limit = len(full_order) if args.request_limit is None else args.request_limit
    if limit > len(full_order):
        raise ValueError("--request-limit exceeds workload request count")
    order = full_order[:limit]
    schedule = full_schedule[:limit]

    if args.server_pid_file is not None:
        pid = int(args.server_pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)

    sources, prompt_summary = load_prompt_sources(
        order,
        parquet_path=args.parquet,
        tools_path=args.tools,
        tokenizer_path=args.tokenizer,
        prompt_salt_prefix=args.prompt_salt_prefix,
    )
    request_namespace = sha256_file(args.order)[:16]

    try:
        import aiohttp
    except ModuleNotFoundError as error:
        raise RuntimeError("runner requires aiohttp") from error

    loop = asyncio.get_running_loop()
    trace = aiohttp.TraceConfig()

    async def on_chunk_sent(_session: Any, trace_context: Any, _params: Any) -> None:
        context = trace_context.trace_request_ctx
        if not context["submit_marked"]:
            context["submit_marked"] = True
            context["submitted_at"] = loop.time()
            context["mark_submitted"](context["submitted_at"])

    trace.on_request_chunk_sent.append(on_chunk_sent)
    base_url = args.base_url.rstrip("/")

    async with create_http_session(
        aiohttp,
        timeout_seconds=args.request_timeout,
        trace_configs=[trace],
    ) as session:
        await _health_check(session, base_url)

        async def send_event(
            event: Mapping[str, Any],
            _timing: DispatchTiming,
            mark_submitted: Callable[[float | None], None],
        ) -> Mapping[str, Any]:
            prompt_ids = sources[str(event["trajectory_id"])].prompt_for(
                int(event["request_index"])
            )
            if len(prompt_ids) > args.max_model_len:
                raise ValueError(
                    f"salted prompt exceeds max model length: {event['request_id']}"
                )
            api_request_id = f"swe-open-{request_namespace}-{int(event['ordinal'])}"
            payload = await asyncio.to_thread(
                json.dumps,
                {
                    "model": args.model,
                    "prompt": prompt_ids,
                    "max_tokens": args.max_output_tokens,
                    "ignore_eos": True,
                    "temperature": 0,
                    "seed": args.seed,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "return_token_ids": True,
                    "request_id": api_request_id,
                },
                separators=(",", ":"),
            )
            trace_context: dict[str, Any] = {
                "mark_submitted": mark_submitted,
                "submit_marked": False,
                "submitted_at": None,
            }
            async with session.post(
                f"{base_url}/v1/completions",
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                trace_request_ctx=trace_context,
            ) as response:
                if response.status >= 400:
                    body = (await response.text())[:2000]
                    raise RuntimeError(f"HTTP {response.status}: {body}")
                first_token_at, prompt_tokens, output_tokens = (
                    await consume_completion_sse(
                        response.content,
                        clock=loop.time,
                    )
                )
            completed_at = loop.time()
            submitted_at = trace_context["submitted_at"]
            if submitted_at is None:
                raise RuntimeError("aiohttp did not expose the request send boundary")
            if first_token_at is None:
                if output_tokens > 0:
                    first_token_at = completed_at
                else:
                    raise RuntimeError(
                        f"no streamed token returned for {event['request_id']}"
                    )
            expected_tokens = len(prompt_ids)
            if prompt_tokens != expected_tokens:
                raise ValueError(
                    f"server prompt token mismatch for {event['request_id']}: "
                    f"{prompt_tokens} != {expected_tokens}"
                )
            return {
                "expected_body_prompt_tokens": int(
                    event.get("body_prompt_tokens", event["prompt_tokens"])
                ),
                "expected_salted_prompt_tokens": expected_tokens,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": max((first_token_at - submitted_at) * 1000, 0.0),
            }

        cap_error: SafetyCapExceeded | None = None
        trace_writer = AtomicJsonlTrace(args.trace_output)
        try:
            try:
                report = await dispatch_open_loop(
                    order,
                    schedule,
                    send_event,
                    safety_cap=args.safety_cap,
                    clock=loop.time,
                    record_callback=trace_writer.append,
                    progress_interval_s=10.0,
                )
            except SafetyCapExceeded as error:
                cap_error = error
                report = error.report
        finally:
            trace_writer.close()

    rate_compliance = summarize_dispatch(
        report,
        configured_offered_rate=args.expected_offered_rate,
        schedule_manifest=schedule_manifest,
    )
    summary = request_summary(report.records, report.wall_seconds)
    result = {
        "schema_version": 2,
        "status": "aborted_safety_cap" if cap_error else "completed",
        "run_id": args.run_id,
        "model": args.model,
        "order": str(args.order.resolve()),
        "manifest": str(args.manifest.resolve()),
        "profile": str(args.profile.resolve()),
        "schedule": str(args.schedule.resolve()),
        "schedule_manifest": str(args.schedule_manifest.resolve()),
        "order_sha256": sha256_file(args.order),
        "manifest_sha256": sha256_file(args.manifest),
        "profile_sha256": sha256_file(args.profile),
        "schedule_sha256": sha256_file(args.schedule),
        "schedule_manifest_sha256": sha256_file(args.schedule_manifest),
        "trace_output": str(args.trace_output.resolve()),
        "trace_sha256": sha256_file(args.trace_output),
        "actual_submit_order_sha256": rate_compliance["actual_submit_order_sha256"],
        "seed": args.seed,
        "session_pool": args.session_pool,
        "session_pool_is_client_concurrency": False,
        "request_limit": args.request_limit,
        "prompt_preparation": prompt_summary,
        "summary": summary,
        "rate_compliance": rate_compliance,
        "state_timeline": report.state_timeline,
        "records": [_serialized_record(record) for record in report.records],
    }
    if cap_error is not None:
        result["fatal_error"] = f"{type(cap_error).__name__}: {cap_error}"
    return result, 2 if cap_error else (1 if summary["errors"] else 0)


def _schedule_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an immutable uniform or fixed-seed Poisson schedule"
    )
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--offered-rate", type=float, default=DEFAULT_RATE)
    parser.add_argument(
        "--arrival-process",
        choices=("deterministic_uniform", "poisson"),
        default="deterministic_uniform",
    )
    parser.add_argument("--schedule-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--burstiness", type=float, default=1.0)
    return parser


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--tools", type=Path, default=DEFAULT_TOOLS)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--server-pid-file", type=Path)
    parser.add_argument("--max-model-len", type=int, default=200_000)
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--session-pool", type=int, default=32)
    parser.add_argument("--prompt-salt-prefix", required=True)
    parser.add_argument("--expected-offered-rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--safety-cap", type=int, default=128)
    parser.add_argument("--fail-on-safety-cap", action="store_true")
    parser.add_argument("--request-limit", type=int)
    return parser


def _parse_args() -> argparse.Namespace:
    if len(sys.argv) > 1 and sys.argv[1] in {"schedule", "generate-schedule"}:
        args = _schedule_parser().parse_args(sys.argv[2:])
        args.command = "schedule"
        return args
    args = _run_parser().parse_args()
    args.command = "run"
    return args


def main() -> int:
    args = _parse_args()
    if args.command == "schedule":
        manifest = write_schedule(
            order_path=args.order,
            schedule_path=args.schedule,
            manifest_path=args.schedule_manifest,
            offered_rate=args.offered_rate,
            arrival_process=args.arrival_process,
            seed=args.schedule_seed,
            burstiness=args.burstiness,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    result, exit_code = asyncio.run(replay(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary result: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["rate_compliance"], indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
