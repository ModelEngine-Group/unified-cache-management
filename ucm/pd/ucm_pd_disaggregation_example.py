# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
# UCM modifications to vLLM Ascend's load_balance_proxy_server_example.py (8cd3c3762).
# License text: https://github.com/vllm-project/vllm-ascend/blob/8cd3c3762/LICENSE
#
# Tutorial: UCM P/D Disaggregation Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python ucm_pd_disaggregation_example.py \
#     --host 0.0.0.0 --port 9000 --workers 2 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
# The default handoff protocol is Mooncake. Add --ucm-pd for UCM shared storage.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import base64
import csv
import functools
import glob
import hashlib
import heapq
import ipaddress
import json
import logging
import logging.config
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from calendar import timegm
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from logging.handlers import QueueHandler, QueueListener, WatchedFileHandler
from multiprocessing.managers import BaseManager
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, cast

# ---- UCM extensions: protocol, observations and offline CLI ----

# Engine log events shared by the live collector and the offline CSV comparison.

MOONCAKE_PREFILL = re.compile(r"Delaying free of \d+ blocks for request (?P<rid>\S+)")
SCHEDULER_RECV = re.compile(r"Finished recving KV transfer for request (?P<rid>\S+)")
UCM_PREFILL = re.compile(
    r"ucm_pd_prefill_done request_id=(?P<rid>\S+) ts_ns=(?P<ns>\d+)"
)
UCM_RANK = re.compile(
    r"ucm_pd_sync_load_done request_id=(?P<rid>\S+) rank=(?P<rank>\d+) ts_ns=(?P<ns>\d+)"
)
EPOCH = re.compile(r"(?<![\w.])\d{10}\.\d{3,9}(?!\d)")
FAILED_LOAD = re.compile(
    r"Failed to (?:transfer KV cache for|reformat KV cache after all pulls for) request (\S+?): "
    r"|(?:Request|request) (\S+) (?:aborted\.|failed to load, skip caching|loaded an incomplete UCM block"
    r"|async (?:load task failed|FAWA \S+ load failed)|submit (?:FAWA )?load task error)"
)


@dataclass(frozen=True)
class HandoffEvent:
    engine_id: str
    ns: (
        int | None
    )  # A failed or aborted load invalidates any completion for this sample.
    rank: int | None


def parse_handoff_event(line, pattern, exact_ids=False):
    """Return the normalized engine request key and event, or None for unrelated lines."""
    asctime = None
    if line.lstrip().startswith("{"):
        # Avoid JSON-decoding unrelated scheduler DEBUG traffic in the live reader.
        if not pattern.search(line) and not FAILED_LOAD.search(line):
            return None
        try:
            record = json.loads(line)
        except ValueError:
            return None
        line = record.get("message", "")
        if not isinstance(line, str):
            return None
        asctime = record.get("asctime")
    match = pattern.search(line)
    rank = None
    if match:
        fields = match.groupdict()
        rid = fields["rid"]
        if "ns" in fields:
            ns = int(fields["ns"])
        elif isinstance(asctime, str):
            # vLLM's built-in JSON formatter emits local time with milliseconds;
            # the documented engine launch sets TZ=UTC on both hosts.
            stamp = datetime.strptime(asctime, "%Y-%m-%d %H:%M:%S,%f")
            ns = timegm(stamp.timetuple()) * 1_000_000_000 + stamp.microsecond * 1000
        else:
            stamp = EPOCH.search(line, 0, match.start())
            if stamp is None:
                raise ValueError(
                    "missing fractional Unix timestamp; see the engine logging options in ucm_pd_disaggregation_example.md"
                )
            ns = int(Decimal(stamp.group()) * 1_000_000_000)
        if "rank" in fields:
            rank = int(fields["rank"])
    elif pattern in (SCHEDULER_RECV, UCM_RANK) and (failed := FAILED_LOAD.search(line)):
        rid = failed.group(1) or failed.group(2)
        ns = None
    else:
        return None
    key = rid if exact_ids else re.sub(r"-[0-9a-f]{8}$", "", rid)
    return key, HandoffEvent(rid, ns, rank)


def read_handoff_events(paths, pattern, exact_ids):
    events = defaultdict(list)
    for path in paths:
        with path.open() as log:
            for lineno, line in enumerate(log, 1):
                try:
                    parsed = parse_handoff_event(line, pattern, exact_ids)
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
                if parsed:
                    key, event = parsed
                    events[key].append(event)
    return events


def handoff_duration(starts, ends, ranks=None):
    if any(e.ns is None for e in ends):
        return None, "failed_or_aborted"
    if len(starts) > 1 or len({e.engine_id for e in ends}) > 1:
        return None, "ambiguous_attempts"
    if not starts or not ends:
        return None, "missing_prefill" if not starts else "missing_decode"
    if len({e.rank for e in ends}) != len(ends):
        return None, "repeated_completion"
    if ranks is not None and {e.rank for e in ends} != ranks:
        return None, "rank_set_mismatch"
    elapsed = max(e.ns for e in ends) - starts[0].ns
    return (elapsed / 1_000_000, "observed") if elapsed >= 0 else (None, "clock_order")


def compare_handoff(argv: list[str]) -> None:
    """Compare final Mooncake and UCM engine logs by request ID and write CSV to stdout."""
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} compare-handoff",
        description=compare_handoff.__doc__,
        epilog="Supply each log once. For synchronous UCM loads, list all expected ranks with --ucm-sync-ranks.",
    )
    for option in ("mooncake-p", "mooncake-d", "ucm-p", "ucm-d"):
        parser.add_argument(f"--{option}", nargs="+", required=True, type=Path)
    parser.add_argument("--ucm-sync-ranks", type=int, nargs="+", metavar="RANK")
    parser.add_argument(
        "--exact-ids",
        action="store_true",
        help="Use when engine request-ID randomization is disabled.",
    )
    args = parser.parse_args(argv)
    ranks = set(args.ucm_sync_ranks) if args.ucm_sync_ranks else None
    try:
        mp = read_handoff_events(args.mooncake_p, MOONCAKE_PREFILL, args.exact_ids)
        md = read_handoff_events(args.mooncake_d, SCHEDULER_RECV, args.exact_ids)
        up = read_handoff_events(args.ucm_p, UCM_PREFILL, args.exact_ids)
        ud = read_handoff_events(
            args.ucm_d, UCM_RANK if ranks else SCHEDULER_RECV, args.exact_ids
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    ids = mp.keys() | md.keys() | up.keys() | ud.keys()
    if not ids:
        parser.error("no handoff events found in the supplied logs")
    writer = csv.writer(sys.stdout)
    ucm_column = "ucm_sync_rank_ms" if ranks else "ucm_scheduler_ms"
    writer.writerow(
        [
            "request_id",
            "mooncake_scheduler_ms",
            ucm_column,
            "ucm_minus_mooncake_ms",
            "mooncake_status",
            "ucm_status",
        ]
    )
    for rid in sorted(ids):
        moon, moon_status = handoff_duration(mp[rid], md[rid])
        ucm, ucm_status = handoff_duration(up[rid], ud[rid], ranks)
        diff = ucm - moon if moon is not None and ucm is not None else None
        writer.writerow(
            [
                rid,
                *[f"{ms:.3f}" if ms is not None else "" for ms in (moon, ucm, diff)],
                moon_status,
                ucm_status,
            ]
        )


# Dispatch before importing the HTTP stack so offline analysis needs only Python.
if __name__ == "__main__" and sys.argv[1:2] == ["compare-handoff"]:
    raise SystemExit(compare_handoff(sys.argv[2:]))


# Online request timing and shared-log collection.

# A stalled filesystem or a lost worker must not grow the serving process indefinitely.
MAX_PENDING_HANDOFFS = 65536


def parse_pd_options(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.description = (
        "Serve P/D requests using Mooncake or UCM shared-store handoff."
    )
    parser.epilog = (
        "Run '%(prog)s compare-handoff --help' for offline comparison of engine logs."
    )
    group = parser.add_argument_group("P/D handoff and timing")
    group.add_argument(
        "--ucm-pd",
        action="store_true",
        help="Use UCM shared-store handoff. By default, forward Mooncake's prefill metadata.",
    )
    group.add_argument(
        "--proxy-log",
        metavar="PATH",
        help="Also write proxy logs to a file that reopens after rotation.",
    )
    group.add_argument(
        "--timing-backend",
        choices=["mooncake", "ucm-async", "ucm-sync"],
        help="Enable per-request summaries from shared engine logs.",
    )
    group.add_argument("--timing-p-logs", nargs="+", metavar="GLOB")
    group.add_argument("--timing-d-logs", nargs="+", metavar="GLOB")
    group.add_argument("--timing-sync-ranks", nargs="+", type=int, metavar="RANK")
    group.add_argument(
        "--timing-exact-ids",
        action="store_true",
        help="Engine request-ID randomization is disabled.",
    )
    group.add_argument(
        "--timing-log-wait",
        type=float,
        default=30,
        metavar="SECONDS",
        help="Maximum wait for late engine logs after a request finishes.",
    )
    args = parser.parse_args()
    if not args.timing_backend:
        if (
            args.timing_p_logs
            or args.timing_d_logs
            or args.timing_sync_ranks
            or args.timing_exact_ids
        ):
            parser.error("select --timing-backend to enable shared-log timing")
        return args
    if args.ucm_pd and args.timing_backend == "mooncake":
        parser.error("--ucm-pd requires ucm-async or ucm-sync when collecting timings")
    if not args.ucm_pd and args.timing_backend != "mooncake":
        parser.error("add --ucm-pd to use UCM handoff with UCM timings")
    if not args.timing_p_logs or not args.timing_d_logs:
        parser.error("--timing-backend needs --timing-p-logs and --timing-d-logs")
    if (args.timing_backend == "ucm-sync") != bool(args.timing_sync_ranks):
        parser.error("supply --timing-sync-ranks for ucm-sync only")
    if not math.isfinite(args.timing_log_wait) or args.timing_log_wait <= 0:
        parser.error("--timing-log-wait must be positive and finite")
    return args


def single_sequence(api, body):
    if body.get("n", 1) != 1 or body.get("best_of", 1) not in (None, 1):
        return False
    if api == "/chat/completions":
        messages = body.get("messages", [])
        return not messages or not isinstance(messages[0], list)
    prompt = body.get("prompt")
    return (
        not isinstance(prompt, list) or len(prompt) <= 1 or isinstance(prompt[0], int)
    )


class RequestTiming:
    """Worker-local observations. All times are monotonic seconds; no log file access."""

    def __init__(self, api, body, received_at):
        self.rid = body["request_id"]
        self.key = (
            ("chatcmpl-" + self.rid)
            if api == "/chat/completions"
            else ("cmpl-" + self.rid + "-0")
        )
        self.single = single_sequence(api, body)
        self.stream = bool(body.get("stream"))
        self.received = received_at
        self.p_seconds = self.d_seconds = 0.0
        self.first_p = self.first_d_send = self.first_token = self.last_token = None
        self.first_source = None
        self.prompt_tokens = self.cached_prompt_tokens = self.usage_tokens = None
        self.id_count = 0
        self.saw_ids = self.missing_ids = False
        self.recomputes = self.retries = 0
        self.recompute_signals = 0
        self.interrupted = False
        self.terminal = False
        self.prefiller = self.decoder = None

    def prefill_response(self, payload, at):
        if self.first_p is None:
            self.first_p = at
            usage = payload.get("usage") or {}
            self.prompt_tokens = usage.get("prompt_tokens")
            self.cached_prompt_tokens = (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            )

    def observe(self, payload, at, source=None):
        if not isinstance(payload, dict):
            return
        if payload.get("error"):
            self.interrupted = True
        usage = payload.get("usage") or {}
        if self.prompt_tokens is None:
            self.prompt_tokens = usage.get("prompt_tokens")
        if isinstance(usage.get("completion_tokens"), int):
            self.usage_tokens = usage["completion_tokens"]
        elif source:
            self.usage_tokens = (
                None  # A previous cumulative count predates this output.
            )
        for choice in payload.get("choices") or ():
            if choice.get("stop_reason") == "recomputed":
                self.recompute_signals += 1
                continue
            if choice.get("finish_reason") is not None:
                self.terminal = True
            ids = choice.get("token_ids")
            if isinstance(ids, list):
                self.saw_ids = True
                self.id_count += len(ids)
            elif source:
                self.missing_ids = True
        if source:
            if self.first_token is None:
                self.first_token = at
                self.first_source = source
            self.last_token = at

    def summary(self, status):
        finished_at = time.perf_counter()
        elapsed = finished_at - self.received
        if status == "ok" and self.recompute_signals > self.recomputes:
            status = "unresolved_recompute"
        if status == "ok" and (self.interrupted or self.stream and not self.terminal):
            status = "interrupted"
        tokens = self.usage_tokens
        token_source = "usage" if tokens is not None else None
        if (
            self.saw_ids
            and not self.missing_ids
            and (tokens is None or tokens < self.id_count)
        ):
            tokens, token_source = self.id_count, "token_ids"
        if self.recomputes or status != "ok":
            tokens, token_source = (
                None,
                None,
            )  # Last-attempt usage cannot represent the logical output.
        first = self.first_token
        tpot = None
        if (
            self.single
            and first is not None
            and self.last_token > first
            and tokens is not None
            and tokens > 1
        ):
            tpot = (self.last_token - first) * 1000 / (tokens - 1)
        return {
            "request_id": self.rid,
            "proxy_start_ns": round(self.received * 1_000_000_000),
            "proxy_end_ns": round(finished_at * 1_000_000_000),
            "status": status,
            "prefiller": self.prefiller,
            "decoder": self.decoder,
            "prompt_tokens": self.prompt_tokens,
            "prefill_cached_tokens": self.cached_prompt_tokens,
            "output_tokens": tokens,
            "count_source": token_source,
            "sequence_scope": "single" if self.single else "multi",
            "first_token_source": self.first_source,
            "ttft_ms": (first - self.received) * 1000 if first is not None else None,
            "p_response_to_first_token_ms": (
                (first - self.first_p) * 1000 if first is not None else None
            ),
            "d_send_to_first_token_ms": (
                (first - self.first_d_send) * 1000 if first is not None else None
            ),
            "p_response_to_d_send_ms": (
                (self.first_d_send - self.first_p) * 1000
                if self.first_d_send is not None
                else None
            ),
            "generation_ms": (
                (self.last_token - first) * 1000
                if first is not None and not self.recomputes and status == "ok"
                else None
            ),
            "tpot_ms": tpot,
            "output_tps": (
                tokens / elapsed if tokens is not None and elapsed > 0 else None
            ),
            "decode_tps": 1000 / tpot if tpot and tpot > 0 else None,
            "prefill_http_ms": self.p_seconds * 1000 if self.prefiller else None,
            "decode_http_ms": (
                self.d_seconds * 1000 if self.first_d_send is not None else None
            ),
            "e2e_ms": elapsed * 1000,
            "recomputes": self.recomputes,
            "http_retries": self.retries,
        }


class LogTail:
    """Follow one active filename, draining a renamed inode before switching."""

    def __init__(self, path, history=False):
        self.path = path
        self.file = path.open("rb")
        if history:
            self.file.seek(0, os.SEEK_END)
        self.partial = b""

    def read(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            stat = None
        current = os.fstat(self.file.fileno())
        if current.st_size < self.file.tell():
            self.file.seek(0)
            self.partial = b""
        data = self.file.read(256 * 1024)
        if (
            not data
            and stat
            and (stat.st_dev, stat.st_ino) != (current.st_dev, current.st_ino)
        ):
            replacement = self.path.open("rb")
            self.file.close()
            self.file = replacement
            self.partial = b""
            data = self.file.read(256 * 1024)
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        return lines, len(data) == 256 * 1024, stat is None and not data


@dataclass
class PendingHandoff:
    starts: list = field(default_factory=list)
    ends: list = field(default_factory=list)
    reports: list = field(default_factory=list)
    active: int = 0
    reused: bool = False


class HandoffCollector:
    """Owned by the process serving SharedProxyScheduler. Only its thread touches shared files."""

    def __init__(self, args, logger):
        self.logger = logger
        self.backend = args.timing_backend
        self.ranks = set(args.timing_sync_ranks or []) or None
        self.wait_seconds = args.timing_log_wait
        self.exact_ids = args.timing_exact_ids
        self.sources = [
            (
                args.timing_p_logs,
                True,
                MOONCAKE_PREFILL if self.backend == "mooncake" else UCM_PREFILL,
            ),
            (
                args.timing_d_logs,
                False,
                UCM_RANK if self.ranks else SCHEDULER_RECV,
            ),
        ]
        self.inbox = Queue(maxsize=MAX_PENDING_HANDOFFS * 2)
        self.overflow = threading.Event()
        self.pending = {}
        self.tails = {}
        self.warned = set()
        self.stop = threading.Event()
        # Establish initial offsets before admitting traffic; later-created files start at byte zero.
        self.discover(history=True)
        self.thread = threading.Thread(
            target=self.run, name="pd-handoff-logs", daemon=True
        )
        self.thread.start()

    def register(self, key):
        self._enqueue(key, None)

    def finish(self, key, summary):
        self._enqueue(key, summary)

    def _enqueue(self, key, summary):
        try:
            self.inbox.put_nowait((key, summary))
        except Full:
            self.overflow.set()

    def warn_once(self, key, message):
        if key not in self.warned:
            self.warned.add(key)
            self.logger.warning("P/D timing log %s: %s", key, message)

    def discover(self, history=False):
        for patterns, is_p, pattern in self.sources:
            for wildcard in patterns:
                for name in glob.iglob(wildcard):
                    path = Path(name)
                    if (path, is_p) not in self.tails:
                        try:
                            self.tails[path, is_p] = (LogTail(path, history), pattern)
                        except OSError as exc:
                            self.warn_once(path, exc)

    def drain_inbox(self):
        if self.overflow.is_set():
            self.overflow.clear()
            self.logger.warning(
                "P/D timing queue overflow: observations dropped; in-flight joins are ambiguous"
            )
            for slot in self.pending.values():
                slot.reused = True
        while True:
            try:
                key, report = self.inbox.get_nowait()
            except Empty:
                return
            slot = self.pending.get(key)
            if slot is None:
                if len(self.pending) >= MAX_PENDING_HANDOFFS:
                    oldest = next(iter(self.pending))
                    del self.pending[oldest]
                    self.warn_once(
                        "capacity",
                        "evicting oldest request; verify shared log reader keeps up",
                    )
                slot = self.pending[key] = PendingHandoff()
            if report is None:
                slot.reused |= bool(slot.active or slot.reports)
                slot.active += 1
            else:
                if not slot.active:
                    slot.reused = True
                slot.active = max(0, slot.active - 1)
                slot.reports.append((report, time.monotonic() + self.wait_seconds))

    def poll(self):
        self.discover()
        self.drain_inbox()
        backlog = False
        for (path, is_p), (tail, pattern) in list(self.tails.items()):
            try:
                lines, more, gone = tail.read()
                backlog |= more
                if gone:
                    tail.file.close()
                    del self.tails[path, is_p]
            except OSError as exc:
                self.warn_once(path, exc)
                continue
            # Registration is enqueued before the P HTTP call. A new request
            # could have arrived while this thread was reading a shared file.
            self.drain_inbox()
            for raw in lines:
                try:
                    event = parse_handoff_event(
                        raw.decode(errors="replace"), pattern, self.exact_ids
                    )
                except ValueError as exc:
                    self.warn_once(path, exc)
                    continue
                if event and (slot := self.pending.get(event[0])):
                    (slot.starts if is_p else slot.ends).append(event[1])
        return backlog

    def flush(self, closing=False):
        now = time.monotonic()
        for key, slot in list(self.pending.items()):
            if not slot.reports:
                continue
            remaining = []
            for metrics, deadline in slot.reports:
                ms, status = handoff_duration(slot.starts, slot.ends, self.ranks)
                if slot.reused or metrics["recomputes"] or metrics["http_retries"]:
                    ms, status = None, "ambiguous_attempts"
                elif metrics["status"] != "ok":
                    ms, status = None, metrics["status"]
                elif metrics["sequence_scope"] != "single":
                    ms, status = None, "multi_output"
                if (
                    status in ("missing_prefill", "missing_decode", "rank_set_mismatch")
                    and now < deadline
                    and not closing
                ):
                    remaining.append((metrics, deadline))
                    continue
                values = {
                    "request_id": metrics["request_id"],
                    "backend": self.backend,
                    "kv_handoff_ms": ms,
                    "kv_status": status,
                    **metrics,
                }
                fields = " ".join(
                    f"{name}={format_field(value)}" for name, value in values.items()
                )
                self.logger.info("pd_request_timing %s", fields)
            slot.reports = remaining
            if not slot.active and not remaining:
                del self.pending[key]

    def run(self):
        try:
            while not self.stop.is_set():
                more = self.poll()
                self.flush()
                if not more:
                    self.stop.wait(0.2)
            self.poll()
            self.flush(closing=True)
        finally:
            for tail, _ in self.tails.values():
                tail.file.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def format_field(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, str) and any(c.isspace() for c in value):
        return json.dumps(value)
    return str(value)


def build_decode_request(req_data: dict, prefill_reply: dict, *, ucm_pd: bool) -> dict:
    """Build this attempt's D payload. Empty P metadata falls back to the client's original parameters."""
    payload = req_data.copy()
    handoff = prefill_reply.get("kv_transfer_params")
    if handoff:
        payload["kv_transfer_params"] = handoff
    if ucm_pd:
        payload["kv_transfer_params"] = {
            **(payload.get("kv_transfer_params") or {}),
            "do_remote_prefill": True,
        }
    if not payload.get("stream", False):
        payload.pop("stream_options", None)
    return payload


async def prefill_for_decode(
    client: "httpx.AsyncClient",
    prefiller_key: str,
    api: str,
    req_data: dict,
    request_id: str,
    args: argparse.Namespace,
    timing: RequestTiming | None,
) -> tuple[dict, int | None, float]:
    """Return the new D payload, P cached-token count and P response time for this attempt."""
    if args.ucm_pd:
        req_data["request_id"] = request_id
    sent_at = time.perf_counter()
    if timing is not None:
        timing.prefiller = prefiller_key
    try:
        response = await send_request_to_service(
            client,
            api,
            req_data,
            request_id,
            max_retries=args.max_retries,
            base_delay=args.retry_delay,
            timing=timing,
        )
        received_at = time.perf_counter()
        logger.info(
            "PD prefill_response request_id=%s prefiller=%s", request_id, prefiller_key
        )
        reply = response.json()
        decode_request = build_decode_request(req_data, reply, ucm_pd=args.ucm_pd)
        cached_tokens = extract_cached_tokens(reply)
    except (Exception, asyncio.CancelledError):
        if timing is not None:
            timing.p_seconds += time.perf_counter() - sent_at
        raise
    if timing is not None:
        timing.p_seconds += received_at - sent_at
        timing.prefill_response(reply, received_at)
    return decode_request, cached_tokens, received_at


_SSE_EVENT_END = re.compile(rb"\r?\n\r?\n")


def _output_source(message: dict | None, echo: bool) -> str | None:
    """Identify generated output in an OpenAI-compatible streaming message."""
    if not isinstance(message, dict):
        return None
    for choice in message.get("choices") or ():
        if choice.get("stop_reason") == "recomputed":
            continue
        if choice.get("token_ids"):
            return "token_ids"
        if (
            echo
        ):  # Echoed prompt text is indistinguishable from generated text without output IDs.
            continue
        delta = choice.get("delta") or {}
        for key in ("content", "reasoning", "reasoning_content"):
            if delta.get(key):
                return key
        if choice.get("text"):
            return "text"
        for tool in delta.get("tool_calls") or ():
            function = tool.get("function") or {}
            if function.get("name") or function.get("arguments"):
                return "tool_call"
    return None


@dataclass
class DecodeEvent:
    """A complete D response or SSE frame, with its original bytes and arrival time."""

    raw: bytes
    received_at: float
    is_sse: bool
    message: dict | None = None
    done: bool = False

    @classmethod
    def parse(cls, raw: bytes, received_at: float, is_sse: bool):
        data = raw
        if is_sse:
            data = b"\n".join(
                line[5:] for line in raw.splitlines() if line.startswith(b"data:")
            ).strip()
            if not data or data == b"[DONE]":
                return cls(raw, received_at, True, done=data == b"[DONE]")
        try:
            message = json.loads(data)
        except ValueError:
            message = None
        return cls(
            raw, received_at, is_sse, message if isinstance(message, dict) else None
        )

    def rewritten(self) -> bytes:
        """Replace the JSON data, preserving any SSE comments, id and event fields."""
        data = json.dumps(self.message, ensure_ascii=False).encode("utf-8")
        if not self.is_sse:
            return data
        lines = []
        for line in self.raw.splitlines(keepends=True):
            if not line.startswith(b"data:"):
                lines.append(line)
            elif data is not None:
                lines.append(b"data: " + data + line[len(line.rstrip(b"\r\n")) :])
                data = None
        return b"".join(lines)


async def decode_response_events(first, chunks) -> AsyncIterator[DecodeEvent]:
    """Frame one HTTP attempt. All events completed by one read carry that read's timestamp."""
    pending = bytearray()
    item = first
    while item is not None:
        chunk, received_at, is_sse = item
        search_from = max(0, len(pending) - 3)
        pending.extend(chunk)
        if is_sse:
            consumed = 0
            while end := _SSE_EVENT_END.search(pending, search_from):
                raw = bytes(pending[consumed : end.end()])
                consumed = search_from = end.end()
                yield DecodeEvent.parse(raw, received_at, True)
            if consumed:
                del pending[:consumed]
        item = await anext(chunks, None)
    if pending:
        if is_sse:
            # Preserve a truncated tail on the wire without treating it as a dispatched SSE event.
            yield DecodeEvent(bytes(pending), received_at, True)
        else:
            yield DecodeEvent.parse(bytes(pending), received_at, False)


@dataclass
class DecodeObserver:
    """Observe complete D events across all attempts of one logical request."""

    request_id: str
    streaming: bool
    prefill_response_at: float | None
    echo: bool
    timing: RequestTiming | None
    logged_first_token: bool = False

    def observe(self, event: DecodeEvent) -> None:
        if event.message is None:
            if event.done and self.timing is not None:
                self.timing.terminal = True
            return
        source = None
        if self.streaming and event.is_sse and self.prefill_response_at is not None:
            if self.timing is not None or not self.logged_first_token:
                source = _output_source(event.message, self.echo)
            if source and not self.logged_first_token:
                self.logged_first_token = True
                logger.info(
                    "PD decode_first_token request_id=%s source=%s p_response_to_first_token_ms=%.2f",
                    self.request_id,
                    source,
                    (event.received_at - self.prefill_response_at) * 1000,
                )
        if self.timing is not None:
            self.timing.observe(event.message, event.received_at, source)


class DecodeAttempt:
    """Measure one D HTTP attempt up to its last byte, or its end if no byte arrived."""

    def __init__(self, request_id, decoder, attempt, prefill_response_at, timing):
        self.request_id = request_id
        self.decoder = decoder
        self.attempt = attempt
        self.prefill_response_at = prefill_response_at
        self.timing = timing
        self.last_byte_at = None
        self.sent_at = time.perf_counter()
        if timing is not None:
            if attempt > 1:
                timing.retries += 1
            if timing.first_d_send is None:
                timing.first_d_send = self.sent_at

    def received_bytes(self) -> float:
        received_at = time.perf_counter()
        if self.last_byte_at is None and self.prefill_response_at is not None:
            logger.info(
                "PD decode_first_bytes request_id=%s decoder=%s attempt=%d "
                "p_response_to_d_send_ms=%.2f d_send_to_first_bytes_ms=%.2f",
                self.request_id,
                self.decoder,
                self.attempt,
                (self.sent_at - self.prefill_response_at) * 1000,
                (received_at - self.sent_at) * 1000,
            )
        if self.timing is not None:
            previous = (
                self.last_byte_at if self.last_byte_at is not None else self.sent_at
            )
            self.timing.d_seconds += received_at - previous
        self.last_byte_at = received_at
        return received_at

    def finish(self) -> None:
        if self.timing is not None and self.last_byte_at is None:
            self.timing.d_seconds += time.perf_counter() - self.sent_at


class _FileLogListener(QueueListener):
    def enqueue_sentinel(self):
        # Shutdown may wait for disk. The request path always uses put_nowait.
        self.queue.put(self._sentinel)


class ProxyFileHandler(QueueHandler):
    """Keep rotation checks and file writes off the HTTP and scheduler threads."""

    def __init__(self, path: str):
        self.sink = WatchedFileHandler(path, encoding="utf-8")
        super().__init__(Queue(maxsize=65536))
        self.dropped = 0
        self.listener = _FileLogListener(self.queue, self.sink)
        self.listener.start()

    def enqueue(self, record):
        if self.dropped:
            record.msg = f"pd_proxy_log_gap dropped={self.dropped}\n{record.msg}"
        try:
            self.queue.put_nowait(record)
        except Full:
            self.dropped += 1
        else:
            self.dropped = 0

    def close(self):
        if self.listener is not None:
            if self.dropped:
                self.queue.put(
                    logging.makeLogRecord(
                        {"msg": f"pd_proxy_log_gap dropped={self.dropped}"}
                    )
                )
            self.listener.stop()
            self.listener = None
            self.sink.close()
        super().close()


def proxy_log_handlers(path: str | None):
    if not path:
        return None
    from uvicorn.config import LOGGING_CONFIG

    # A later uvicorn dictConfig would close the parent process's file listener.
    logging.config.dictConfig(LOGGING_CONFIG)
    return [logging.StreamHandler(), ProxyFileHandler(path)]


# ---- vLLM Ascend proxy: scheduling, HTTP forwarding and process lifecycle ----
# Preserve upstream line wrapping so subsequent Ascend changes can be compared directly.
# fmt: off

import httpx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, Response, StreamingResponse  # noqa: E402

logger = logging.getLogger(__name__)

try:
    import uvloop  # type: ignore[import-not-found]

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_key: str
    prefiller_score: float
    decoder_key: str
    decoder_score: float
    decoder_host: str
    decoder_port: int
    prefiller_cached_tokens: int | None = None
    decoder_released: bool = False
    prefill_response_at: float | None = None
    timing: RequestTiming | None = None
    decode_request: dict = field(default_factory=dict, repr=False)


TAINT_PRIORITY = 1e15


def extract_cached_tokens(response_json: dict) -> int | None:
    usage = response_json.get("usage") or {}
    prompt_tokens_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_tokens_details.get("cached_tokens")
    return cached_tokens if isinstance(cached_tokens, int) else 0


def update_cached_tokens_in_chunk(chunk_json: dict, cached_tokens: int | None) -> bool:
    if cached_tokens is None:
        return False
    usage = chunk_json.get("usage")
    if not isinstance(usage, dict):
        return False
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_tokens_details, dict):
        prompt_tokens_details = {}
    usage["prompt_tokens_details"] = prompt_tokens_details
    prompt_tokens_details["cached_tokens"] = cached_tokens
    return True


global_args: argparse.Namespace | None = None
shared_scheduler: "SharedProxyScheduler | None" = None
runtime: "WorkerRuntime | None" = None


@dataclass
class BackendServer:
    host: str
    port: int
    ordinal: int
    active_tokens: float = 0.0
    active_kv_cache: float = 0.0
    heap_seq: int = 0


@dataclass
class RolePools:
    """Per-role scheduling state: live servers, priority heap, and drain-isolated keys."""

    servers: dict[str, BackendServer] = field(default_factory=dict)
    heap: list[tuple[float, int, int, str]] = field(default_factory=list)
    tainted: set[str] = field(default_factory=set)


def setup_logging(log_level: str, proxy_log: str | None = None) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
        handlers=proxy_log_handlers(proxy_log),
    )
    logger.setLevel(getattr(logging, log_level.upper()))


def conversation_affinity_key(req_data: dict) -> str | None:
    """Identify a conversation without hashing its growing message history."""
    user = req_data.get("user")
    if isinstance(user, str) and user:
        identity = [req_data.get("model"), "session", user]
    else:
        messages = req_data.get("messages") or []
        first_user_index = next((i for i, message in enumerate(messages) if message.get("role") == "user"), None)
        if first_user_index is None:
            return None
        instructions = [m for m in messages[:first_user_index] if m.get("role") in ("system", "developer")]
        identity = [req_data.get("model"), instructions, messages[first_user_index]]
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def next_req_id() -> str:
    return str(uuid.uuid4())


def calculate_prefill_score(request_length: int) -> float:
    length_score = request_length / 4.0
    return length_score * 0.0345 + 120.0745


def calculate_decode_score(request_length: int) -> float:
    return request_length


def normalize_host(host: str) -> str:
    return host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")


def server_key(host: str, port: int) -> str:
    return f"{normalize_host(host)}:{int(port)}"


def build_server_url(host: str, port: int) -> str:
    url = f"http://{host}:{port}"
    try:
        ip = ipaddress.ip_address(host)
        if isinstance(ip, ipaddress.IPv6Address):
            url = f"http://[{host}]:{port}"
    except Exception:
        pass
    return url


def build_base_url(host: str, port: int) -> str:
    return f"{build_server_url(host, port)}/v1"


class SharedProxyScheduler:
    """Centralized mutable scheduling state shared by all uvicorn workers.

    Uses lazy-deletion min-heap: on priority change, push a new entry and
    bump the server's ``heap_seq`` counter; stale entries (whose seq does
    not match) are skipped on pop.
    """

    def __init__(self, prefiller_instances, decoder_instances):
        self._lock = threading.RLock()
        self.request_num = 0
        self.waiting_nodes: dict[str, tuple[str, tuple[str, int], int]] = {}
        self._pools: dict[ServerRole, RolePools] = {
            ServerRole.PREFILL: RolePools(),
            ServerRole.DECODE: RolePools(),
        }
        self._ordinal = 0
        self._timing_collector = None

        for host, port in prefiller_instances:
            self._add_server_no_lock(ServerRole.PREFILL, host, port)
        for host, port in decoder_instances:
            self._add_server_no_lock(ServerRole.DECODE, host, port)

    def _pool(self, role: ServerRole) -> RolePools:
        return self._pools[role]

    @property
    def prefillers(self) -> dict[str, BackendServer]:
        return self._pool(ServerRole.PREFILL).servers

    @property
    def decoders(self) -> dict[str, BackendServer]:
        return self._pool(ServerRole.DECODE).servers

    def _next_ordinal(self) -> int:
        ordinal = self._ordinal
        self._ordinal += 1
        return ordinal

    def _priority(self, role: ServerRole, entry: BackendServer, key: str) -> float:
        if key in self._pool(role).tainted:
            return TAINT_PRIORITY
        if role is ServerRole.PREFILL:
            return entry.active_tokens + entry.active_kv_cache * 0.3
        return entry.active_tokens

    def _push_heap(self, role: ServerRole, key: str) -> None:
        pool = self._pool(role)
        entry = pool.servers[key]
        entry.heap_seq += 1
        heapq.heappush(pool.heap, (self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key))
        if len(pool.heap) > 2 * len(pool.servers):
            self._reset_heap(role)

    def _pop_valid(self, role: ServerRole) -> str:
        pool = self._pool(role)
        while pool.heap:
            _, _, seq, key = heapq.heappop(pool.heap)
            if key not in pool.servers:
                continue
            entry = pool.servers[key]
            if entry.heap_seq == seq:
                return key
        raise RuntimeError(f"No available {role.value} servers")

    def _reset_heap(self, role: ServerRole, *, bump_seq: bool = False) -> None:
        pool = self._pool(role)
        heap = []
        for key, entry in pool.servers.items():
            if bump_seq:
                entry.heap_seq += 1
            heap.append((self._priority(role, entry, key), entry.ordinal, entry.heap_seq, key))
        heapq.heapify(heap)
        pool.heap = heap

    def _add_server_no_lock(self, role: ServerRole, host: str, port: int) -> bool:
        key = server_key(host, port)
        pool = self._pool(role)
        if key in pool.servers:
            return False
        pool.servers[key] = BackendServer(host, int(port), self._next_ordinal())
        self._push_heap(role, key)
        return True

    def get_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {
                "prefill_instances": [
                    {"host": e.host, "port": e.port}
                    for _, e in sorted(self.prefillers.items(), key=lambda item: item[1].ordinal)
                ],
                "decode_instances": [
                    {"host": e.host, "port": e.port}
                    for _, e in sorted(self.decoders.items(), key=lambda item: item[1].ordinal)
                ],
            }

    def log_status(self, msg: str) -> None:
        snapshot = self.get_snapshot()
        logger.info(
            "%s prefill=%s decode=%s",
            msg,
            [f"{s['host']}:{s['port']}" for s in snapshot["prefill_instances"]],
            [f"{s['host']}:{s['port']}" for s in snapshot["decode_instances"]],
        )

    def healthcheck(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "prefill_instances": len(self.prefillers),
                "decode_instances": len(self.decoders),
                "request_num": self.request_num,
            }

    def _pick_server(
        self,
        role: ServerRole,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
        affinity_key: str | None = None,
    ) -> dict[str, Any]:
        if affinity_key and role is ServerRole.PREFILL:
            pool = self._pool(role)
            candidates = pool.servers.keys() - pool.tainted
            if not candidates:
                raise RuntimeError("No available prefill servers for session affinity")
            # Rendezvous hashing is stable across workers/restarts and only
            # remaps conversations whose chosen endpoint is removed.
            key = max(
                candidates,
                key=lambda k: (hashlib.sha256((affinity_key + "\0" + k).encode("utf-8")).digest(), k),
            )
            logger.debug("Prefill affinity key=%s backend=%s", affinity_key[:16], key)
        else:
            key = self._pop_valid(role)
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens += load
        if kv_cache:
            entry.active_kv_cache += load
        self._push_heap(role, key)
        return {"key": key, "host": entry.host, "port": entry.port}

    def _release_load(
        self,
        role: ServerRole,
        key: str | None,
        load: float,
        *,
        active_tokens: bool = False,
        kv_cache: bool = False,
    ) -> None:
        if not key or key not in self._pool(role).servers:
            return
        entry = self._pool(role).servers[key]
        if active_tokens:
            entry.active_tokens -= load
        if kv_cache:
            entry.active_kv_cache = max(0.0, entry.active_kv_cache - load)
        self._push_heap(role, key)

    def begin_request(self, load: float, affinity_key: str | None = None, timing_key: str | None = None) -> dict[str, Any]:
        """Pick a prefiller, reserve KV pressure, and count this as an active request."""
        with self._lock:
            picked = self._pick_server(ServerRole.PREFILL, load, kv_cache=True, affinity_key=affinity_key)
            self.request_num += 1
            if timing_key is not None:
                self._timing_collector.register(timing_key)
            return picked

    def reserve_prefill_kv(self, load: float, affinity_key: str | None = None) -> dict[str, Any]:
        """Pick a prefiller for recompute without bumping the active request count."""
        with self._lock:
            return self._pick_server(ServerRole.PREFILL, load, kv_cache=True, affinity_key=affinity_key)

    def pick_decoder(self, load: float) -> dict[str, Any]:
        with self._lock:
            return self._pick_server(ServerRole.DECODE, load, active_tokens=True)

    def release_prefill_kv(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.PREFILL, key, load, kv_cache=True)

    def release_decoder(self, key: str, load: float) -> None:
        with self._lock:
            self._release_load(ServerRole.DECODE, key, load, active_tokens=True)

    def finish_request(
        self,
        prefiller_key: str | None,
        prefiller_load: float,
        decoder_key: str | None,
        decoder_load: float,
        release_prefill_kv: bool,
        timing_report: tuple | None = None,
    ) -> None:
        with self._lock:
            if release_prefill_kv:
                self._release_load(ServerRole.PREFILL, prefiller_key, prefiller_load, kv_cache=True)
            self._release_load(ServerRole.DECODE, decoder_key, decoder_load, active_tokens=True)
            self.request_num = max(0, self.request_num - 1)
            if timing_report is not None:
                self._timing_collector.finish(*timing_report)

    def get_waiting_nodes(self) -> dict[str, tuple[str, tuple[str, int], int]]:
        with self._lock:
            return dict(self.waiting_nodes)

    def add_instances(self, role: ServerRole, instances: list[tuple[str, int]]) -> list[str]:
        waiting_nodes: list[str] = []
        with self._lock:
            servers = self._pool(role).servers
            for host, port in instances:
                key = server_key(host, port)
                if key in servers or key in self.waiting_nodes:
                    continue
                self.waiting_nodes[key] = (role.value, (host, int(port)), 0)
                waiting_nodes.append(f"{host}:{port}")
        return waiting_nodes

    def mark_waiting_retry(self, key: str, retry_count: int) -> None:
        with self._lock:
            if key not in self.waiting_nodes:
                return
            instance_type, server, _ = self.waiting_nodes[key]
            self.waiting_nodes[key] = (instance_type, server, retry_count)

    def activate_waiting_instance(self, role: ServerRole, host: str, port: int) -> None:
        with self._lock:
            key = server_key(host, port)
            self.waiting_nodes.pop(key, None)
            pool = self._pool(role)
            if key in pool.tainted:
                pool.tainted.discard(key)
                self._push_heap(role, key)
                return
            if self._add_server_no_lock(role, host, port):
                self.log_status(f"Add {role.value} instance: {host}:{port}.")

    def drop_waiting_instance(self, key: str) -> None:
        with self._lock:
            self.waiting_nodes.pop(key, None)

    def remove_instances(self, role: ServerRole, instances: list[tuple[str, int]]) -> bool:
        if not instances:
            return False
        keys = {server_key(host, port) for host, port in instances}
        with self._lock:
            pool = self._pool(role)
            if self.request_num > 0:
                pool.tainted.update(keys)
                self._reset_heap(role, bump_seq=True)
                logger.warning("Start to taint %s instances %s.", role.value, sorted(keys))
                return True

            removed = False
            for key in keys:
                removed = pool.servers.pop(key, None) is not None or removed
                self.waiting_nodes.pop(key, None)
            pool.tainted.difference_update(keys)
            if removed:
                self._reset_heap(role, bump_seq=True)
                self.log_status(f"Remove {role.value} instances: {sorted(keys)}.")
            return False

    def finalize_tainted_instances(self) -> None:
        with self._lock:
            if self.request_num != 0:
                return
            for role in ServerRole:
                pool = self._pool(role)
                if not pool.tainted:
                    continue
                keys = list(pool.tainted)
                for key in keys:
                    pool.servers.pop(key, None)
                pool.tainted.clear()
                self._reset_heap(role, bump_seq=True)
                self.log_status(f"Remove {role.value} instances after drain: {keys}.")


class SchedulerManager(BaseManager):
    """Multiprocessing RPC bridge; body is empty but required by BaseManager."""


def _shared_scheduler_proxy() -> "SharedProxyScheduler":
    if shared_scheduler is None:
        raise RuntimeError("shared scheduler is not initialized")
    return shared_scheduler


SchedulerManager.register("get_scheduler", callable=_shared_scheduler_proxy)


class WorkerRuntime:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        self._clients: dict[ServerRole, dict[str, httpx.AsyncClient]] = {
            ServerRole.PREFILL: {},
            ServerRole.DECODE: {},
        }
        self._async_lock = asyncio.Lock()

    async def schedule(self, method: str, /, *args, **kwargs) -> Any:
        async with self._async_lock:
            return getattr(self.scheduler, method)(*args, **kwargs)

    async def get_client(self, role: ServerRole, key: str) -> httpx.AsyncClient:
        clients = self._clients[role]
        if key not in clients:
            await self.sync_clients()
        return clients[key]

    async def sync_clients(self) -> None:
        snapshot = self.scheduler.get_snapshot()
        role_targets = {
            ServerRole.PREFILL: {
                server_key(s["host"], s["port"]): (s["host"], s["port"]) for s in snapshot["prefill_instances"]
            },
            ServerRole.DECODE: {
                server_key(s["host"], s["port"]): (s["host"], s["port"]) for s in snapshot["decode_instances"]
            },
        }
        for role, targets in role_targets.items():
            await self._sync_clients(role, targets)

    async def _sync_clients(self, role: ServerRole, targets: dict[str, tuple[str, int]]) -> None:
        clients = self._clients[role]
        for key in [key for key in clients if key not in targets]:
            await clients.pop(key).aclose()
        for key, (host, port) in targets.items():
            if key in clients:
                continue
            clients[key] = httpx.AsyncClient(
                timeout=None,
                base_url=build_base_url(host, port),
                limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
            )

    async def close(self) -> None:
        for role in ServerRole:
            for client in list(self._clients[role].values()):
                await client.aclose()
            self._clients[role].clear()


def get_runtime() -> WorkerRuntime:
    if runtime is None:
        raise RuntimeError("worker runtime is not initialized")
    return runtime


class NodeListener:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            args = get_global_args()
            for key, (instance_type, server, retries) in list(self.scheduler.get_waiting_nodes().items()):
                host, port = server
                is_valid = asyncio.run(self.check_instance_status(host, port))
                print(f"Checking instance {key}...")
                retries += 1
                if is_valid:
                    self.scheduler.activate_waiting_instance(ServerRole(instance_type), host, port)
                elif retries >= args.max_waiting_retries:
                    print(f"Instance {key} was not added to the proxy.")
                    self.scheduler.drop_waiting_instance(key)
                else:
                    self.scheduler.mark_waiting_retry(key, retries)

            self.scheduler.finalize_tainted_instances()
            time.sleep(args.waiting_retry_interval)

    @staticmethod
    async def check_instance_status(host: str, port: int) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            async with httpx.AsyncClient(timeout=5.0, base_url=build_base_url(host, port)) as client:
                response = await client.get(endpoint, headers=headers)
                response.raise_for_status()
                return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def manager_config_path(proxy_port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"vllm_lb_proxy_manager_{proxy_port}.json"


def write_manager_config(proxy_port: int, host: str, manager_port: int, authkey: bytes) -> None:
    manager_config_path(proxy_port).write_text(
        json.dumps(
            {
                "host": host,
                "port": manager_port,
                "authkey": base64.b64encode(authkey).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )


def read_manager_config(proxy_port: int) -> dict[str, Any]:
    path = manager_config_path(proxy_port)
    if not path.is_file():
        raise RuntimeError(
            f"Manager config not found at {path}. "
            "Start the proxy from __main__ with --workers > 1 before worker processes connect."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_manager_config(proxy_port: int) -> None:
    manager_config_path(proxy_port).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument(
        "--enable-prefill-affinity",
        action="store_true",
        help="Route conversations to stable prefill endpoints to reuse prefix cache; decode remains load balanced.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes. Scheduling state is shared across workers.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for the proxy server.",
    )
    args = parse_pd_options(parser)
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def get_global_args() -> argparse.Namespace:
    global global_args
    if global_args is None:
        global_args = parse_args()
    return global_args


def connect_shared_scheduler(proxy_port: int):
    manager_cfg = read_manager_config(proxy_port)
    manager = SchedulerManager(
        address=(manager_cfg["host"], manager_cfg["port"]),
        authkey=base64.b64decode(manager_cfg["authkey"]),
    )
    manager.connect()
    return manager.get_scheduler()  # type: ignore[attr-defined]


def bootstrap_parent_process(args: argparse.Namespace) -> None:
    """Initialize cross-worker shared state in the parent process before uvicorn spawns workers."""
    global shared_scheduler
    if args.workers <= 1:
        return

    _ensure_scheduler(args)

    authkey = os.urandom(16)
    manager = SchedulerManager(address=("127.0.0.1", 0), authkey=authkey)
    server = manager.get_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.address)
    write_manager_config(args.port, host, port, authkey)


def _ensure_scheduler(args) -> SharedProxyScheduler:
    global shared_scheduler
    if shared_scheduler is not None:
        return shared_scheduler
    shared_scheduler = SharedProxyScheduler(args.prefiller_instances, args.decoder_instances)
    if args.timing_backend:
        shared_scheduler._timing_collector = HandoffCollector(args, logger)
    NodeListener(shared_scheduler)
    return shared_scheduler


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime
    args = get_global_args()
    if args.workers > 1:
        scheduler = connect_shared_scheduler(args.port)
    else:
        scheduler = _ensure_scheduler(args)
    runtime = WorkerRuntime(scheduler)
    await runtime.sync_clients()
    snapshot = scheduler.get_snapshot()
    logger.info(
        "Initialized %s prefill clients and %s decode clients in worker %s.",
        len(snapshot["prefill_instances"]),
        len(snapshot["decode_instances"]),
        os.getpid(),
    )
    try:
        yield
    finally:
        await runtime.close()
        if args.workers == 1 and scheduler._timing_collector is not None:
            await asyncio.to_thread(scheduler._timing_collector.close)
        runtime = None


app = FastAPI(lifespan=lifespan)


def create_app():
    args = get_global_args()
    setup_logging(args.log_level, args.proxy_log)
    return app


async def listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def auth_headers(request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }


def build_prefill_request(req_data: dict) -> dict:
    payload = req_data.copy()
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["min_tokens"] = 1
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = 1
    payload.pop("stream_options", None)
    return payload


async def send_request_to_service(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
    timing=None,
):
    req_data = build_prefill_request(req_data)
    headers = auth_headers(request_id)
    max_attempts = max(1, max_retries)
    for attempt in range(1, max_attempts + 1):
        if timing is not None and attempt > 1:
            timing.retries += 1
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 or attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
        except httpx.RequestError as exc:
            if attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, exc)
        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))


async def stream_service_response(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
    *,
    prefill_response_at: float | None = None,
    timing=None,
) -> AsyncIterator[tuple[bytes, float, bool]]:
    """Yield raw body chunks, their monotonic arrival times and the response's SSE mode."""
    headers = auth_headers(request_id)
    max_attempts = max(1, max_retries)
    for attempt in range(1, max_attempts + 1):
        observation = DecodeAttempt(request_id, client.base_url, attempt, prefill_response_at, timing)
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                if response.status_code >= 400:
                    # Buffer the upstream error body so callers can forward the real
                    # cause (e.g. vLLM's "maximum context length" message) instead of
                    # dropping it into an empty 200. The body is unreadable once this
                    # streaming context exits, so read it before raising.
                    await response.aread()
                    response.raise_for_status()
                media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
                is_sse = media_type == "text/event-stream" if media_type else bool(req_data.get("stream"))
                async for chunk in response.aiter_bytes():
                    yield chunk, observation.received_bytes(), is_sse
                return
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 or attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        except httpx.RequestError as exc:
            if observation.last_byte_at is not None:
                if timing is not None:
                    timing.interrupted = True
                logger.error("Streaming to client interrupted after response started: %s", exc)
                return
            if attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        except Exception as exc:
            if observation.last_byte_at is not None:
                if timing is not None:
                    timing.interrupted = True
                logger.error("Streaming to client interrupted after response started: %s", exc)
                return
            if attempt == max_attempts:
                raise
            logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, exc)
        finally:
            observation.finish()
        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))


async def _abort_prefill_selection(
    runtime: WorkerRuntime,
    prefiller_key: str,
    prefiller_score: float,
    *,
    is_initial_request: bool,
    timing=None,
) -> None:
    if is_initial_request:
        report = (timing.key, timing.summary("upstream_error")) if timing is not None else None
        await runtime.schedule("finish_request", prefiller_key, prefiller_score, None, 0.0, release_prefill_kv=True,
                               timing_report=report)
    else:
        await runtime.schedule("release_prefill_kv", prefiller_key, prefiller_score)


async def _finish_instance(runtime: WorkerRuntime, info: InstanceInfo, *, release_prefill_kv: bool, status="upstream_error") -> None:
    await runtime.schedule(
        "finish_request",
        info.prefiller_key,
        info.prefiller_score,
        None if info.decoder_released else info.decoder_key,
        info.decoder_score,
        release_prefill_kv,
        timing_report=(info.timing.key, info.timing.summary(status)) if info.timing is not None else None,
    )
    info.decoder_released = True


async def _release_decoder_once(runtime: WorkerRuntime, info: InstanceInfo) -> None:
    if info.decoder_released:
        return
    await runtime.schedule("release_decoder", info.decoder_key, info.decoder_score)
    info.decoder_released = True


async def assign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    *,
    is_initial_request: bool,
    timing=None,
) -> InstanceInfo:
    runtime = get_runtime()
    args = get_global_args()
    prefiller_score = calculate_prefill_score(request_length)
    decoder_score = calculate_decode_score(request_length)
    request_id = req_data.get("request_id") or next_req_id()
    pick_prefill = "begin_request" if is_initial_request else "reserve_prefill_kv"
    timing_args = {"timing_key": timing.key} if timing is not None and is_initial_request else {}
    if args.enable_prefill_affinity:
        prefiller = await runtime.schedule(pick_prefill, prefiller_score, conversation_affinity_key(req_data), **timing_args)
    else:
        prefiller = await runtime.schedule(pick_prefill, prefiller_score, **timing_args)
    prefiller_key = prefiller["key"]

    try:
        decode_request, prefiller_cached_tokens, prefill_response_at = await prefill_for_decode(
            await runtime.get_client(ServerRole.PREFILL, prefiller_key),
            prefiller_key, api, req_data, request_id, args, timing,
        )
    except (Exception, asyncio.CancelledError):
        await _abort_prefill_selection(runtime, prefiller_key, prefiller_score, is_initial_request=is_initial_request, timing=timing)
        raise

    decoder_key = None
    try:
        decoder = await runtime.schedule("pick_decoder", decoder_score)
        decoder_key = decoder["key"]
        if timing is not None:
            timing.decoder = decoder_key
        prefiller_client = await runtime.get_client(ServerRole.PREFILL, prefiller_key)
        decoder_client = await runtime.get_client(ServerRole.DECODE, decoder_key)
    except (Exception, asyncio.CancelledError):
        await _abort_prefill_selection(runtime, prefiller_key, prefiller_score, is_initial_request=is_initial_request, timing=timing)
        if decoder_key is not None:
            await runtime.schedule("release_decoder", decoder_key, decoder_score)
        raise

    logger.debug("Using %s %s", prefiller_client.base_url, decoder_client.base_url)
    return InstanceInfo(
        request_id=request_id,
        prefiller_key=prefiller_key,
        prefiller_score=prefiller_score,
        decoder_key=decoder_key,
        decoder_score=decoder_score,
        decoder_host=decoder["host"],
        decoder_port=decoder["port"],
        prefiller_cached_tokens=prefiller_cached_tokens,
        prefill_response_at=prefill_response_at,
        timing=timing,
        decode_request=decode_request,
    )


async def reassign_instances(
    api: str,
    req_data: Any,
    request_length: int,
    previous_instance: InstanceInfo,
    *,
    previous_prefiller_kv_released: bool,
) -> InstanceInfo:
    runtime = get_runtime()
    if not previous_prefiller_kv_released:
        await runtime.schedule("release_prefill_kv", previous_instance.prefiller_key, previous_instance.prefiller_score)
    await _release_decoder_once(runtime, previous_instance)
    req_data["request_id"] = previous_instance.request_id  # Keep the logical ID across recompute attempts.
    extra = {}
    if previous_instance.timing is not None:
        previous_instance.timing.recomputes += 1
        extra["timing"] = previous_instance.timing
    return await assign_instances(api, req_data, request_length, is_initial_request=False, **extra)


async def handle_completions_impl(api: str, request: Request):
    received_at = time.perf_counter()
    runtime = get_runtime()
    args = get_global_args()
    request_released = False
    try:
        req_data = await request.json()
        if args.enable_prefill_affinity and (session_id := request.headers.get("X-Session-Id")):
            req_data["user"] = session_id
        if request_id := request.headers.get("X-Request-Id"):
            req_data["request_id"] = request_id
        req_body = await request.body()
        request_length = len(req_body)
        extra = {}
        if getattr(args, "timing_backend", None):
            req_data["request_id"] = req_data.get("request_id") or next_req_id()
            extra["timing"] = RequestTiming(api, req_data, received_at)
        instance_info = await assign_instances(api, req_data, request_length, is_initial_request=True, **extra)
        stream_flag = bool(req_data.get("stream", False))
        observer = DecodeObserver(
            instance_info.request_id, stream_flag, instance_info.prefill_response_at,
            bool(req_data.get("echo")), instance_info.timing,
        )
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def open_decode(info):
            client = await runtime.get_client(ServerRole.DECODE, info.decoder_key)
            chunks = stream_service_response(
                client, api, info.decode_request, request_id=info.request_id,
                max_retries=args.max_retries, base_delay=args.retry_delay,
                prefill_response_at=info.prefill_response_at, timing=info.timing,
            )
            # Read the first bytes before committing the HTTP status to the client.
            return chunks, await anext(chunks, None)

        try:
            preopened_gen, preopened_first = await open_decode(instance_info)
        except asyncio.CancelledError:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True, status="cancelled")
            request_released = True
            raise
        except httpx.HTTPStatusError as exc:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
            return Response(
                content=exc.response.content, status_code=exc.response.status_code, media_type="application/json"
            )
        except httpx.RequestError as exc:
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Decode backend request to {api} failed: {exc}",
                        "type": "upstream_error",
                        "code": "decode_backend_unavailable",
                    }
                },
            )

        async def generate_stream():
            nonlocal instance_info
            nonlocal request_released
            generated_token = ""
            released_kv = False
            retry_count = 0
            completion_tokens = 0
            reported_prefiller_cached_tokens = instance_info.prefiller_cached_tokens
            status = "interrupted"
            chunks, first = preopened_gen, preopened_first
            try:
                while True:
                    async with aclosing(chunks):
                        if first is not None:
                            await runtime.schedule(
                                "release_prefill_kv", instance_info.prefiller_key, instance_info.prefiller_score
                            )
                            released_kv = True
                        async for event in decode_response_events(first, chunks):
                            observer.observe(event)
                            message = event.message
                            if message is None:
                                yield event.raw
                                continue

                            choices = message.get("choices") or []
                            rewrite = (not choices or not stream_flag) and update_cached_tokens_in_chunk(
                                message, reported_prefiller_cached_tokens
                            )
                            if choices:
                                choice = choices[0]
                                delta = choice.get("delta") or {}
                                reply = choice.get("message") or {}
                                generated_token += (
                                    delta.get("content") or reply.get("content") or choice.get("text") or ""
                                )
                                usage = message.get("usage") or {}
                                completion_tokens += 1 if stream_flag else usage.get("completion_tokens", 0)
                                if choice.get("stop_reason") == "recomputed":
                                    break
                                if retry_count and not stream_flag:
                                    if chat_flag:
                                        choice["message"]["content"] = generated_token
                                    else:
                                        choice["text"] = generated_token
                                    rewrite = True
                            yield event.rewritten() if rewrite else event.raw
                        else:
                            status = "ok"
                            return

                    # The old D stream is closed before starting another P/D attempt.
                    retry_count += 1
                    if chat_flag:
                        messages[0]["content"] = (
                            origin_prompt + ([{"type": "text", "text": generated_token}] if generated_token else [])
                            if isinstance(origin_prompt, list) else (origin_prompt or "") + generated_token
                        )
                    else:
                        req_data["prompt"] = origin_prompt + generated_token
                    req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                    instance_info = await reassign_instances(
                        api, req_data, len(json.dumps(req_data).encode("utf-8")), instance_info,
                        previous_prefiller_kv_released=released_kv,
                    )
                    released_kv = False
                    chunks, first = await open_decode(instance_info)
            except asyncio.CancelledError:
                status = "cancelled"
                logger.warning(
                    "Streaming from decoder %s:%s was cancelled; releasing request %s resources",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    instance_info.request_id,
                )
                raise
            except Exception as exc:
                logger.error(
                    "Error during streaming from decoder %s:%s: %s while handling request %s; releasing prefiller KV",
                    instance_info.decoder_host,
                    instance_info.decoder_port,
                    exc,
                    instance_info.request_id,
                )
            finally:
                await _finish_instance(runtime, instance_info, release_prefill_kv=not released_kv, status=status)
                released_kv = True
                request_released = True

        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print("".join(traceback.format_exception(*exc_info)))
        if not request_released and "instance_info" in locals():
            await _finish_instance(runtime, instance_info, release_prefill_kv=True)
            request_released = True
        if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
            try:
                err_body = e.response.json()
            except Exception:
                err_body = {"error": {"message": e.response.text or "upstream error"}}
            return JSONResponse(err_body, status_code=e.response.status_code)
        raise


async def adjust_instances_impl(adjust_mode: str, request: Request):
    req_data = await request.json()
    instance_type = req_data.get("type", "")
    instances = req_data.get("instances", [])
    if isinstance(instances, str):
        instances = [instances]
    parsed_instances = parse_server_addresses(instances)
    all_msg = f"{adjust_mode} {instance_type} instances: {[f'{host}:{port}' for host, port in parsed_instances]}."

    try:
        role = ServerRole(instance_type)
    except ValueError:
        return {
            "error": (
                f"Instance type {instance_type!r} is not supported. "
                f"Only '{ServerRole.PREFILL.value}' and '{ServerRole.DECODE.value}' are allowed."
            )
        }

    scheduler = get_runtime().scheduler

    if adjust_mode == "add":
        waiting_nodes = scheduler.add_instances(role, parsed_instances)
        if waiting_nodes:
            all_msg = f"Instances {waiting_nodes} are waiting to be added."
    elif adjust_mode == "remove":
        need_waiting = scheduler.remove_instances(role, parsed_instances)
        if need_waiting:
            all_msg = (
                f"Instances {[f'{host}:{port}' for host, port in parsed_instances]} "
                "are isolated and waiting to be removed."
            )

    snapshot = scheduler.get_snapshot()
    return {
        "message": all_msg,
        "current_prefill_instances": [f"{server['host']}:{server['port']}" for server in snapshot["prefill_instances"]],
        "current_decode_instances": [f"{server['host']}:{server['port']}" for server in snapshot["decode_instances"]],
    }


def parse_server_addresses(instances: list[str]) -> list[tuple[str, int]]:
    return [(host, int(port)) for host, port in (instance.split(":") for instance in instances)]


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await handle_completions_impl("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await handle_completions_impl("/chat/completions", request)


async def handle_models_impl():
    runtime = get_runtime()
    request_id = next_req_id()
    await runtime.sync_clients()
    snapshot = runtime.scheduler.get_snapshot()
    candidates = [(ServerRole.PREFILL, s) for s in snapshot["prefill_instances"]] + [
        (ServerRole.DECODE, s) for s in snapshot["decode_instances"]
    ]
    if not candidates:
        return JSONResponse(status_code=503, content={"error": "No available backend instances"})
    failures: list[str] = []
    for role, server in candidates:
        host, port = server["host"], server["port"]
        key = server_key(host, port)
        try:
            client = await runtime.get_client(role, key)
            response = await client.get("/models", headers=auth_headers(request_id), timeout=3.0)
            response.raise_for_status()
            return Response(content=response.content, media_type="application/json")
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning("GET /models from %s:%s failed: %s", host, port, exc)
            failures.append(f"{host}:{port}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "All backend instances failed to serve /v1/models",
            "failed": failures,
        },
    )


@app.get("/v1/models")
async def handle_models():
    return await handle_models_impl()


@app.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    params = dict(request.query_params)
    runtime = get_runtime()
    await runtime.sync_clients()
    snapshot = runtime.scheduler.get_snapshot()
    backend_instances = [(ServerRole.PREFILL, server) for server in snapshot["prefill_instances"]] + [
        (ServerRole.DECODE, server) for server in snapshot["decode_instances"]
    ]
    failures: list[str] = []
    for role, server in backend_instances:
        base_url = build_server_url(server["host"], server["port"])
        try:
            client = await runtime.get_client(role, server_key(server["host"], server["port"]))
            resp = await client.post(f"{base_url}/reset_prefix_cache", params=params)
            resp.raise_for_status()
        except Exception as e:
            logger.error("reset_prefix_cache failed for %s: %s", base_url, e)
            failures.append(base_url)
    if failures:
        return JSONResponse(status_code=500, content={"failed": failures})
    return Response(status_code=200)


@app.get("/healthcheck")
async def healthcheck():
    return get_runtime().scheduler.healthcheck()


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await adjust_instances_impl("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await adjust_instances_impl("remove", request)


if __name__ == "__main__":
    global_args = parse_args()
    setup_logging(global_args.log_level, global_args.proxy_log)
    bootstrap_parent_process(global_args)
    import uvicorn

    module_name = Path(__file__).stem
    try:
        uvicorn.run(
            f"{module_name}:create_app",
            host=global_args.host,
            port=global_args.port,
            workers=global_args.workers,
            log_config=None if global_args.proxy_log else uvicorn.config.LOGGING_CONFIG,
            factory=True,
            app_dir=str(Path(__file__).resolve().parent),
        )
    finally:
        cleanup_manager_config(global_args.port)
        if shared_scheduler is not None and shared_scheduler._timing_collector is not None:
            shared_scheduler._timing_collector.close()
# fmt: on
