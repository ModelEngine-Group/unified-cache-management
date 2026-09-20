#!/usr/bin/env python3
"""Completion-driven closed-loop replay for causal SWE-rebench workloads.

This runner reproduces the client semantics used by the earlier vLLM +
LMCache C16/C32 campaign while remaining backend-agnostic.  It sends requests
to an OpenAI-compatible vLLM endpoint, so the server under test can be UCM on
CUDA or vLLM-Ascend + UCM.  There is deliberately no arrival schedule or
offered-rate parameter: a completion frees one of the fixed client slots.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from benchmark_swe_rebench_open_loop import (
    consume_completion_sse,
    create_http_session,
    distribution,
    load_jsonl,
    load_prompt_sources,
    percentile,
    sha256_file,
    validate_order,
)


ALLOWED_CONCURRENCY = (16, 32)
DEFAULT_SEED = 20260725
SendEvent = Callable[
    [Mapping[str, Any], Callable[[float | None], None]],
    Awaitable[Mapping[str, Any] | None],
]


@dataclass
class ClosedLoopRecord:
    """Auditable timing state for one closed-loop request."""

    ordinal: int
    request_id: str
    trajectory_id: str
    depends_on: str | None
    dispatch_considered_s: float | None = None
    dependency_ready_s: float | None = None
    actual_submit_s: float | None = None
    completion_s: float | None = None
    dependency_wait_ms: float | None = None
    client_admission_wait_ms: float | None = None
    total_ms: float | None = None
    inflight_after_submit: int | None = None
    inflight_before_completion: int | None = None
    submission_sequence: int | None = None
    completion_sequence: int | None = None
    response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ClosedLoopReport:
    """Dispatcher result independent of the HTTP transport."""

    records: list[ClosedLoopRecord]
    state_timeline: list[dict[str, Any]]
    client_concurrency: int
    observed_max_inflight: int
    wall_seconds: float


class AtomicClosedLoopTrace:
    """Publish a new JSONL request trace atomically."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.temporary = target.with_suffix(target.suffix + ".tmp")
        if self.target.exists() or self.temporary.exists():
            raise FileExistsError(f"refusing to overwrite trace: {self.target}")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("x", encoding="utf-8")
        self._closed = False

    def append(self, record: ClosedLoopRecord) -> None:
        """Append one completed record."""
        if self._closed:
            raise RuntimeError("trace is already closed")
        self._handle.write(
            json.dumps(
                serialize_record(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    def close(self) -> None:
        """Flush, fsync, and atomically publish the trace."""
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self.temporary.replace(self.target)
        self._closed = True


def validate_concurrency(value: int) -> None:
    """Accept only the two concurrency points in the inherited campaign."""
    if value not in ALLOWED_CONCURRENCY:
        raise ValueError(
            f"client concurrency must be one of {ALLOWED_CONCURRENCY}, got {value}"
        )


def validate_closed_loop_workload(
    order_path: Path,
    manifest_path: Path,
    profile_path: Path,
    *,
    seed: int,
    session_pool: int,
    prompt_salt_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the exact timestamp-free W32 workload and its hashes."""
    events = load_jsonl(order_path)
    validate_order(events)
    if any("arrival_time_s" in event for event in events):
        raise ValueError("closed-loop order must not contain arrival timestamps")
    if any(str(event.get("phase")) != "measured" for event in events):
        raise ValueError("the no-primer campaign accepts measured requests only")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    expected_order_hash = outputs.get("order", {}).get("sha256")
    expected_profile_hash = outputs.get("profiles", {}).get("sha256")
    if expected_order_hash != sha256_file(order_path):
        raise ValueError("closed-loop order SHA-256 does not match manifest")
    if expected_profile_hash != sha256_file(profile_path):
        raise ValueError("closed-loop profile SHA-256 does not match manifest")

    interleaving = manifest.get("interleaving", {})
    if int(interleaving.get("seed", -1)) != seed:
        raise ValueError("runner seed does not match workload manifest")
    if int(interleaving.get("active_sessions", -1)) != session_pool:
        raise ValueError("session pool does not match workload manifest")
    if str(interleaving.get("prompt_salt_label")) != prompt_salt_prefix:
        raise ValueError("prompt salt does not match workload manifest")
    if bool(interleaving.get("primer_included", True)):
        raise ValueError("closed-loop campaign requires no-primer workload")

    profile_summary = manifest.get("profile_summary", {})
    if int(profile_summary.get("measured_requests", -1)) != len(events):
        raise ValueError("manifest request count does not match order")
    if int(profile_summary.get("trajectories", -1)) != len(
        {str(event["trajectory_id"]) for event in events}
    ):
        raise ValueError("manifest trajectory count does not match order")
    return events, manifest


async def dispatch_closed_loop(
    events: Sequence[Mapping[str, Any]],
    send_event: SendEvent,
    *,
    client_concurrency: int,
    clock: Callable[[], float] = time.perf_counter,
    record_callback: Callable[[ClosedLoopRecord], None] | None = None,
) -> ClosedLoopReport:
    """Replay strict manifest order with causal and closed-loop bounds.

    The behavior intentionally matches the historical runner: dispatch order
    is the order file, a request waits for its direct dependency, and a new
    request is created only when fewer than ``client_concurrency`` prior tasks
    remain incomplete.  Thus C16/C32 is a completion-driven outstanding limit,
    not W, ``max_num_seqs``, an offered rate, or a nominal schedule.

    Preserving exact send order also preserves the historical global
    head-of-line behavior when the next manifest entry depends on an unfinished
    request.  That property is recorded rather than hidden.
    """
    validate_order(events)
    validate_concurrency(client_concurrency)
    loop = asyncio.get_running_loop()
    origin = clock()
    completion = {
        str(event["request_id"]): asyncio.Event() for event in events
    }
    records = [
        ClosedLoopRecord(
            ordinal=int(event["ordinal"]),
            request_id=str(event["request_id"]),
            trajectory_id=str(event["trajectory_id"]),
            depends_on=(
                None if event.get("depends_on") is None else str(event["depends_on"])
            ),
        )
        for event in events
    ]
    in_flight_tasks: set[asyncio.Task[None]] = set()
    state_timeline: list[dict[str, Any]] = []
    submit_ordinals: list[int] = []
    completion_sequence = 0
    observed_max_inflight = 0
    network_inflight = 0

    def relative_now() -> float:
        return max(clock() - origin, 0.0)

    def snapshot(reason: str, record: ClosedLoopRecord) -> None:
        state_timeline.append(
            {
                "time_s": round(relative_now(), 9),
                "reason": reason,
                "ordinal": record.ordinal,
                "inflight": network_inflight,
                "active_tasks": len(in_flight_tasks),
                "completed": completion_sequence,
            }
        )

    async def reap_finished(wait_for_one: bool) -> None:
        nonlocal in_flight_tasks
        if not in_flight_tasks:
            return
        done = {task for task in in_flight_tasks if task.done()}
        if not done and wait_for_one:
            done, pending = await asyncio.wait(
                in_flight_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            in_flight_tasks = set(pending)
        else:
            in_flight_tasks -= done
        if done:
            await asyncio.gather(*done)

    async def run_one(
        event: Mapping[str, Any],
        record: ClosedLoopRecord,
        submitted_signal: asyncio.Event,
    ) -> None:
        nonlocal completion_sequence, observed_max_inflight, network_inflight
        submitted = False

        def mark_submitted(at: float | None = None) -> None:
            def register() -> None:
                nonlocal submitted, observed_max_inflight, network_inflight
                if submitted:
                    raise RuntimeError(
                        f"submit boundary repeated for {record.request_id}"
                    )
                submitted = True
                absolute = clock() if at is None else at
                record.actual_submit_s = max(absolute - origin, 0.0)
                record.submission_sequence = len(submit_ordinals)
                submit_ordinals.append(record.ordinal)
                network_inflight += 1
                observed_max_inflight = max(observed_max_inflight, network_inflight)
                record.inflight_after_submit = network_inflight
                snapshot("submit", record)
                submitted_signal.set()

            loop.call_soon_threadsafe(register)

        try:
            response = await send_event(event, mark_submitted)
            if not submitted_signal.is_set():
                await asyncio.sleep(0)
            if not submitted_signal.is_set():
                raise RuntimeError("send completed without an HTTP send boundary")
            if response is not None:
                record.response = dict(response)
        except asyncio.CancelledError:
            record.error = "CancelledError: closed-loop replay aborted"
            raise
        except Exception as error:  # noqa: BLE001 - preserve per-request failure
            record.error = f"{type(error).__name__}: {error}"
        finally:
            now = relative_now()
            record.completion_s = now
            if submitted and record.actual_submit_s is not None:
                record.inflight_before_completion = network_inflight
                network_inflight -= 1
                record.total_ms = max((now - record.actual_submit_s) * 1000, 0.0)
            record.completion_sequence = completion_sequence
            completion_sequence += 1
            completion[record.request_id].set()
            snapshot("complete", record)
            if record_callback is not None:
                record_callback(record)
            # A transport failure before the callback must not deadlock the
            # dispatcher.  It remains an explicit request error and invalidates
            # a formal run.
            submitted_signal.set()

    for event, record in zip(events, records, strict=True):
        record.dispatch_considered_s = relative_now()
        dependency_started = clock()
        if record.depends_on is not None:
            await completion[record.depends_on].wait()
        record.dependency_ready_s = relative_now()
        record.dependency_wait_ms = max((clock() - dependency_started) * 1000, 0.0)
        snapshot("dependency_ready", record)

        admission_started = clock()
        await reap_finished(wait_for_one=False)
        while len(in_flight_tasks) >= client_concurrency:
            await reap_finished(wait_for_one=True)
        record.client_admission_wait_ms = max(
            (clock() - admission_started) * 1000, 0.0
        )

        submitted_signal = asyncio.Event()
        task = asyncio.create_task(run_one(event, record, submitted_signal))
        in_flight_tasks.add(task)
        # The historical runner serialized the actual socket-send boundaries,
        # not response processing.  This keeps the client send order identical
        # to the manifest even with many outstanding requests.
        await submitted_signal.wait()

    if in_flight_tasks:
        await asyncio.gather(*in_flight_tasks)
    wall_seconds = relative_now()
    return ClosedLoopReport(
        records=records,
        state_timeline=state_timeline,
        client_concurrency=client_concurrency,
        observed_max_inflight=observed_max_inflight,
        wall_seconds=wall_seconds,
    )


def serialize_record(record: ClosedLoopRecord) -> dict[str, Any]:
    """Flatten transport response metrics into the request record."""
    value = asdict(record)
    response = value.pop("response")
    value.update(response)
    return value


def summarize(report: ClosedLoopReport) -> dict[str, Any]:
    """Summarize one closed-loop run without inventing offered-rate metrics."""
    successful = [
        record
        for record in report.records
        if record.error is None
        and record.actual_submit_s is not None
        and record.completion_s is not None
    ]
    ttfts = [
        float(record.response["ttft_ms"])
        for record in successful
        if record.response.get("ttft_ms") is not None
    ]
    submitted_records = [
        record for record in report.records if record.submission_sequence is not None
    ]
    submit_order = [
        record.ordinal
        for record in sorted(
            submitted_records,
            key=lambda item: int(item.submission_sequence or 0),
        )
    ]
    submit_hash = hashlib.sha256(
        json.dumps(submit_order, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "requests": len(report.records),
        "successful_requests": len(successful),
        "errors": len(report.records) - len(successful),
        "wall_seconds": round(report.wall_seconds, 6),
        "completion_throughput_rps": (
            len(successful) / report.wall_seconds
            if report.wall_seconds > 0
            else math.nan
        ),
        "ttft_mean_ms": statistics.fmean(ttfts) if ttfts else math.nan,
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "prompt_tokens": sum(
            int(record.response.get("prompt_tokens", 0)) for record in successful
        ),
        "output_tokens": sum(
            int(record.response.get("output_tokens", 0)) for record in successful
        ),
        "client_concurrency": report.client_concurrency,
        "observed_max_inflight": report.observed_max_inflight,
        "dependency_wait_ms": distribution(
            [float(record.dependency_wait_ms or 0.0) for record in report.records]
        ),
        "client_admission_wait_ms": distribution(
            [
                float(record.client_admission_wait_ms or 0.0)
                for record in report.records
            ]
        ),
        "actual_submit_order_sha256": submit_hash,
        "arrival_schedule": None,
        "offered_rate_rps": None,
    }


async def replay(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Validate inputs, run one excluded warmup, then replay once."""
    validate_concurrency(args.max_concurrency)
    if args.max_output_tokens != 1:
        raise ValueError("closed-loop campaign requires one output token")
    if args.max_model_len != 200_000:
        raise ValueError("closed-loop campaign requires max_model_len=200000")
    if args.output.exists() or args.trace_output.exists():
        raise FileExistsError("refusing to overwrite result or trace output")
    if args.request_limit is not None and args.request_limit < 1:
        raise ValueError("--request-limit must be positive")

    full_order, manifest = validate_closed_loop_workload(
        args.order,
        args.manifest,
        args.profile,
        seed=args.seed,
        session_pool=args.session_pool,
        prompt_salt_prefix=args.prompt_salt_prefix,
    )
    limit = len(full_order) if args.request_limit is None else args.request_limit
    if limit > len(full_order):
        raise ValueError("--request-limit exceeds workload request count")
    order = full_order[:limit]
    sources, prompt_summary = load_prompt_sources(
        order,
        parquet_path=args.parquet,
        tools_path=args.tools,
        tokenizer_path=args.tokenizer,
        prompt_salt_prefix=args.prompt_salt_prefix,
    )

    try:
        import aiohttp
    except ModuleNotFoundError as error:
        raise RuntimeError("runner requires aiohttp") from error

    loop = asyncio.get_running_loop()
    trace_config = aiohttp.TraceConfig()

    async def on_chunk_sent(_session: Any, trace_context: Any, _params: Any) -> None:
        context = trace_context.trace_request_ctx
        if not context["submit_marked"]:
            context["submit_marked"] = True
            context["submitted_at"] = loop.time()
            context["mark_submitted"](context["submitted_at"])

    trace_config.on_request_chunk_sent.append(on_chunk_sent)
    base_url = args.base_url.rstrip("/")
    namespace = sha256_file(args.order)[:16]

    async with create_http_session(
        aiohttp,
        timeout_seconds=args.request_timeout,
        trace_configs=[trace_config],
    ) as session:
        async with session.get(f"{base_url}/health") as response:
            if response.status != 200:
                raise RuntimeError(f"health endpoint returned HTTP {response.status}")

        async def completion_request(
            *,
            prompt: str | list[int],
            request_id: str,
            mark_submitted: Callable[[float | None], None],
        ) -> dict[str, Any]:
            payload = await asyncio.to_thread(
                json.dumps,
                {
                    "model": args.model,
                    "prompt": prompt,
                    "max_tokens": args.max_output_tokens,
                    "ignore_eos": True,
                    "temperature": 0,
                    "seed": args.seed,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "return_token_ids": True,
                    "request_id": request_id,
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
                first, prompt_tokens, output_tokens = await consume_completion_sse(
                    response.content,
                    clock=loop.time,
                )
            completed = loop.time()
            submitted = trace_context["submitted_at"]
            if submitted is None:
                raise RuntimeError("aiohttp did not expose HTTP send boundary")
            if first is None:
                if output_tokens > 0:
                    first = completed
                else:
                    raise RuntimeError(f"no token returned for {request_id}")
            return {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": max((first - submitted) * 1000, 0.0),
            }

        warmup_submit: float | None = None

        def mark_warmup_submitted(at: float | None = None) -> None:
            nonlocal warmup_submit
            warmup_submit = loop.time() if at is None else at

        warmup_started = loop.time()
        warmup_response = await completion_request(
            prompt=(
                "Warm up this inference service and answer briefly. "
                "This request is excluded from closed-loop metrics."
            ),
            request_id=f"swe-closed-warmup-{args.run_id}",
            mark_submitted=mark_warmup_submitted,
        )
        warmup_completed = loop.time()
        warmup = {
            **warmup_response,
            "included_in_metrics": False,
            "total_ms": max(
                (warmup_completed - (warmup_submit or warmup_started)) * 1000,
                0.0,
            ),
        }

        async def send_event(
            event: Mapping[str, Any],
            mark_submitted: Callable[[float | None], None],
        ) -> Mapping[str, Any]:
            prompt_ids = sources[str(event["trajectory_id"])].prompt_for(
                int(event["request_index"])
            )
            if len(prompt_ids) > args.max_model_len:
                raise ValueError(
                    f"salted prompt exceeds max model length: {event['request_id']}"
                )
            response = await completion_request(
                prompt=prompt_ids,
                request_id=(
                    f"swe-closed-{namespace}-{args.max_concurrency}-"
                    f"{int(event['ordinal'])}"
                ),
                mark_submitted=mark_submitted,
            )
            if int(response["prompt_tokens"]) != len(prompt_ids):
                raise ValueError(
                    f"server prompt token mismatch for {event['request_id']}: "
                    f"{response['prompt_tokens']} != {len(prompt_ids)}"
                )
            return {
                **response,
                "expected_body_prompt_tokens": int(event["prompt_tokens"]),
                "expected_salted_prompt_tokens": len(prompt_ids),
            }

        trace_writer = AtomicClosedLoopTrace(args.trace_output)
        try:
            report = await dispatch_closed_loop(
                order,
                send_event,
                client_concurrency=args.max_concurrency,
                clock=loop.time,
                record_callback=trace_writer.append,
            )
        finally:
            trace_writer.close()

    summary = summarize(report)
    expected_ordinals = list(range(len(order)))
    submitted_ordinals = [
        record.ordinal
        for record in sorted(
            (record for record in report.records if record.submission_sequence is not None),
            key=lambda item: int(item.submission_sequence or 0),
        )
    ]
    previous_submission: dict[str, int] = {}
    same_trajectory_gaps: list[float] = []
    for record in sorted(
        (record for record in report.records if record.submission_sequence is not None),
        key=lambda item: int(item.submission_sequence or 0),
    ):
        assert record.submission_sequence is not None
        previous = previous_submission.get(record.trajectory_id)
        if previous is not None:
            same_trajectory_gaps.append(record.submission_sequence - previous - 1)
        previous_submission[record.trajectory_id] = record.submission_sequence
    result = {
        "schema_version": 1,
        "status": "completed" if summary["errors"] == 0 else "completed_with_errors",
        "run_id": args.run_id,
        "model": args.model,
        "protocol": {
            "name": "closed_loop_completion_driven",
            "client_concurrency": args.max_concurrency,
            "active_session_pool": args.session_pool,
            "active_session_pool_is_client_concurrency": False,
            "arrival_schedule": None,
            "offered_rate_rps": None,
            "send_order": "strict_manifest_order",
            "dependency_policy": "wait_for_depends_on_completion",
            "global_hol_compatibility": "preserved_from_historical_runner",
        },
        "order": str(args.order.resolve()),
        "manifest": str(args.manifest.resolve()),
        "profile": str(args.profile.resolve()),
        "order_sha256": sha256_file(args.order),
        "manifest_sha256": sha256_file(args.manifest),
        "profile_sha256": sha256_file(args.profile),
        "trace_output": str(args.trace_output.resolve()),
        "trace_sha256": sha256_file(args.trace_output),
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "request_timeout_s": args.request_timeout,
        "prompt_salt_prefix": args.prompt_salt_prefix,
        "request_limit": args.request_limit,
        "workload_profile_summary": manifest.get("profile_summary"),
        "prompt_preparation": prompt_summary,
        "warmup": warmup,
        "execution_order": {
            "client_send_order_matches_manifest": submitted_ordinals
            == expected_ordinals,
            "actual_submit_order_sha256": summary[
                "actual_submit_order_sha256"
            ],
            "same_trajectory_intervening_requests": distribution(
                same_trajectory_gaps
            ),
            "manifest_same_trajectory_intervening_requests": manifest.get(
                "interval_analysis", {}
            ).get("all_measured", {}).get("gap_requests"),
        },
        "summary": summary,
        "state_timeline": report.state_timeline,
        "records": [serialize_record(record) for record in report.records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return result, 0 if summary["errors"] == 0 else 1


def parse_args() -> argparse.Namespace:
    """Parse the closed-loop-only CLI; no schedule option exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=200000)
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--session-pool", type=int, default=32)
    parser.add_argument("--prompt-salt-prefix", required=True)
    parser.add_argument("--request-limit", type=int)
    return parser.parse_args()


def main() -> int:
    """Run one closed-loop replay and print its compact summary."""
    args = parse_args()
    result, status = asyncio.run(replay(args))
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Results: {args.output.resolve()}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
