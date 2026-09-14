"""Early-only request state and worker queues; native scheduling/load stay native."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import os
import threading
import time
import uuid
import weakref

from .early_control import Control

log = logging.getLogger(__name__)


@dataclass
class Job:
    request: object
    view: object
    identity: tuple
    key: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "PENDING"
    blocks: list | None = None
    hit: int = 0
    dispatched: bool = False


class EarlyScheduler:
    def __init__(self, connector, control_factory=Control):
        self.connector = connector
        self.lock = threading.RLock()
        self.jobs = {}
        self.by_key = {}
        self.seen = weakref.WeakSet()
        self.traced_requests = weakref.WeakSet()
        self.closed = False
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-early-hash"
        )
        parallel = connector._vllm_config.parallel_config
        # Dense engines reset communication DP rank to zero. The native index
        # retains the actual engine identity for both dense and MoE models.
        self.dp_rank = parallel.data_parallel_index
        broker = dict(tp_size=connector.tp_size) if self.dp_rank == 0 else None
        self.control = control_factory(
            f"ipc:///dev/shm/ucm_early_{connector.unique_id}.sock",
            f"s{self.dp_rank}",
            self.receive,
            broker,
        )

    def identity(self, request):
        # ConstantList is stable for the ordinary immutable waiting request.
        # Do not scan tokens on retry. Streaming/mutation falls back to native.
        return (
            request.num_tokens,
            self.connector.hash_block_size,
            self.connector.cp_world_size,
            getattr(request, "cache_salt", None),
            id(getattr(request, "lora_request", None)),
            id(getattr(request, "mm_features", None)),
            id(getattr(request, "prompt_embeds", None)),
            getattr(request, "num_preemptions", 0),
        )

    def submit(self, request):
        with self.lock:
            if self.closed or request in self.seen:
                return
            self.seen.add(request)
            if getattr(request, "resumable", False):
                return
            previous = self.jobs.get(request.request_id)
            if previous is not None:
                self._cancel(previous)
            job = Job(request, request.all_token_ids, self.identity(request))
            self.jobs[request.request_id] = job
            self.by_key[job.key] = job
            self.pool.submit(self._prepare, job)

    def _prepare(self, job):
        try:
            c = self.connector
            blocks = c.request_block_hasher(job.request)
            with self.lock:
                if job.state != "PENDING" or not self.valid(job, job.request):
                    return
                job.blocks = blocks
                self.trace_request(job.request, blocks)
                submitted_ns = time.monotonic_ns()
                self.control.send(
                    {
                        "op": "job",
                        "key": job.key,
                        "dp": self.dp_rank,
                        "blocks": blocks,
                        "request": job.request.request_id,
                        "submitted_ns": submitted_ns,
                    }
                )
            log.info(
                "UCM_EARLY submitted key=%s request=%s blocks=%s ns=%s dp=%s",
                job.key,
                job.request.request_id,
                len(blocks),
                submitted_ns,
                self.dp_rank,
            )
        except Exception:
            log.exception("UCM_EARLY prepare failed key=%s", job.key)
            with self.lock:
                if job.state == "PENDING":
                    job.state = "FAILED"

    def trace_request(self, request, blocks, *, enabled=None):
        with self.lock:
            if request in self.traced_requests:
                return
            self.traced_requests.add(request)
            if enabled is None:
                enabled = self.connector.enable_record_traces
            if (
                not enabled
                or request.request_id in self.connector.requests_meta
                or not blocks
            ):
                return
            log.info(
                "timestamp: %s, input_length: %s, output_length: %s, "
                "ucm_block_ids: %s",
                time.perf_counter(),
                request.num_tokens,
                request.max_tokens,
                [block_id.hex() for block_id in blocks],
            )

    def valid(self, job, request):
        return (
            job.request is request
            and job.view is request.all_token_ids
            and job.identity == self.identity(request)
        )

    def probe(self, request):
        with self.lock:
            job = self.jobs.get(request.request_id)
            if job is None:
                return None
            if not self.valid(job, request):
                self._cancel(job)
                self.jobs.pop(request.request_id, None)
                return None
            return job

    def receive(self, message):
        with self.lock:
            if message["op"] == "lost":
                self.closed = True
                for job in self.jobs.values():
                    job.state = "FAILED"
                return
            job = self.by_key.get(message.get("key"))
            if job is None or job.state not in ("PENDING", "READY"):
                return
            job.hit = message.get("hit", 0)
            job.state = message["state"]
            log.info("UCM_EARLY scheduler_result key=%s dp=%s state=%s ns=%s",
                     job.key, self.dp_rank, job.state, time.monotonic_ns())

    def _cancel(self, job):
        job.state = "CANCELLED"
        self.by_key.pop(job.key, None)
        self.control.send({"op": "cancel", "key": job.key})

    def scheduled(self, request_meta):
        keys = {}
        with self.lock:
            for request_id, meta in request_meta.items():
                job = self.jobs.get(request_id)
                if job is None:
                    continue
                keys[request_id] = job.key
                if not job.dispatched:
                    job.dispatched = True
                    # Read the native event without consuming or changing it.
                    scheduled_ns = next((int(event.timestamp * 1e9)
                        for event in getattr(job.request, "events", ())
                        if getattr(event.type, "name", None) == "SCHEDULED"), None)
                    log.info("UCM_EARLY handoff key=%s request=%s dp=%s ns=%s native_scheduled_ns=%s",
                             job.key, request_id, self.dp_rank, time.monotonic_ns(), scheduled_ns)
                    if meta.load_block_ids[0]:
                        # Ordered before finish/cancel on this control connection.
                        self.control.send(
                            {"op": "handoff", "key": job.key, "request": request_id}
                        )
                    else:
                        self.control.send({"op": "finish", "key": job.key})
        return keys

    def finish(self, request, cancelled=False):
        request_id = request if isinstance(request, str) else request.request_id
        with self.lock:
            job = self.jobs.get(request_id)
            if job is None or (
                not isinstance(request, str) and job.request is not request
            ):
                return
            self.jobs.pop(request_id)
            self.by_key.pop(job.key, None)
            job.state = "CANCELLED"
            self.control.send(
                {"op": "cancel" if cancelled else "finish", "key": job.key}
            )
            log.info(
                "UCM_EARLY terminal key=%s request=%s cancelled=%s",
                job.key,
                request_id,
                cancelled,
            )

    def close(self):
        with self.lock:
            if self.closed:
                # lost transport still requires hash executor and socket cleanup
                pass
            self.closed = True
            for job in list(self.jobs.values()):
                self._cancel(job)
            self.jobs.clear()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.control.close()


@dataclass(eq=False)
class WorkerJob:
    key: str
    blocks: list
    entered: float = field(default_factory=time.monotonic)
    batches: list = field(default_factory=list)
    reserve_done: bool = False
    cancelled: bool = False
    ready: bool = False
    loading: bool = False


@dataclass(eq=False)
class WorkerBatch:
    index: int
    begin: int
    reservation: dict
    entered: float = field(default_factory=time.monotonic)
    done: bool = False
    started: bool = False
    can_submit: bool = True
    inflight_bytes: int = 0
    stats: dict = field(default_factory=dict)


class EarlyWorker:
    def __init__(self, connector, control_factory=Control):
        self.connector, self.store = connector, connector.store
        self.condition = threading.Condition()
        self.jobs = {}
        self.preparing = []
        self.selected = []
        self.request_keys = {}
        self.closed = False
        self.lookup_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="ucm-early-lookup"
        )
        dp_rank = connector._vllm_config.parallel_config.data_parallel_index
        self.control = control_factory(
            f"ipc:///dev/shm/ucm_early_{connector.unique_id}.sock",
            f"w{dp_rank}.{connector.tp_rank % connector.tp_size}",
            self.receive,
        )
        self.data_thread = threading.Thread(
            target=self._data, name="ucm-early-data", daemon=True
        )
        self.data_thread.start()

    def receive(self, message):
        op = message["op"]
        if op == "lost":
            with self.condition:
                self.closed = True
                for job in list(self.jobs.values()):
                    if not (job.loading or job.ready):
                        self._release(job)
                self.condition.notify_all()
        elif op == "reserve":
            with self.condition:
                if self.closed:
                    return
                key = message["key"]
                if key in self.jobs:
                    return
                job = WorkerJob(key, message["blocks"])
                log.info("UCM_EARLY lookup_enqueued key=%s rank=%s ns=%s submitted_ns=%s",
                         key, self.connector.tp_rank, time.monotonic_ns(), message.get("submitted_ns"))
                self.jobs[key] = job
                if message.get("request"):
                    self.request_keys[message["request"]] = job.key
                self.lookup_pool.submit(self._reserve, job)
        elif op == "handoff":
            with self.condition:
                for job in self.jobs.values():
                    if job.key == message["key"] and not job.cancelled:
                        job.loading = True
                        self.request_keys[message["request"]] = job.key
        elif op == "release":
            with self.condition:
                key = message["key"]
                job = self.jobs.get(key)
                if job:
                    # Native load/finish owns release once consumption may start.
                    if not (
                        job.loading or (job.ready and message.get("reason") == "failed")
                    ):
                        self.jobs.pop(key)
                        self._release(job)

    def _release(self, job):
        job.cancelled = True
        for batch in job.batches:
            self.store.release_reservation(batch.reservation["handle"])
        self.condition.notify_all()

    def _reserve(self, job):
        started = time.monotonic()
        c = self.connector
        log.info("UCM_EARLY lookup_start key=%s rank=%s ns=%s queue_ms=%.3f",
                 job.key, c.tp_rank, time.monotonic_ns(), (started - job.entered) * 1000)
        try:
            blocks = job.blocks
            if not c.is_mla and c.tp_rank % c.tp_size:
                blocks = [c.request_hasher(block) for block in blocks]
            begin, candidate_count = 0, 32
            while begin < len(blocks):
                with self.condition:
                    if job.cancelled or self.closed:
                        return
                candidates = blocks[begin:begin + candidate_count]
                reserving = time.monotonic_ns()
                reservation = self.store.reserve_prefix(candidates)
                batch = WorkerBatch(len(job.batches), begin, reservation,
                                    done=reservation["remaining_l2_work"] == 0)
                with self.condition:
                    if job.cancelled or self.closed:
                        self.store.release_reservation(reservation["handle"])
                        return
                    job.batches.append(batch)
                    log.info(
                        "UCM_EARLY batch_reserved key=%s rank=%s batch=%s begin=%s reservation=%s "
                        "start_ns=%s end_ns=%s candidates=%s hit=%s remaining_l2_work=%s pid=%s",
                        job.key, c.tp_rank, batch.index, begin, reservation["handle"],
                        reserving, time.monotonic_ns(), len(candidates),
                        reservation["total_hit_blocks"], reservation["remaining_l2_work"], os.getpid(),
                    )
                    if not batch.done:
                        # Publication precedes localization/lease of the next
                        # batch. Its handles and lease stay in this reservation.
                        self.preparing.append((job, batch))
                        self.condition.notify_all()
                hit = reservation["total_hit_blocks"]
                if not 0 <= hit <= len(candidates):
                    raise RuntimeError("invalid reserved prefix length")
                if hit < len(candidates):
                    break
                begin += hit
                candidate_count = min(candidate_count * 2, 256)
            with self.condition:
                job.reserve_done = True
                if not job.cancelled and all(batch.done for batch in job.batches):
                    self._answer(job)
            log.info("UCM_EARLY lookup_done key=%s rank=%s ns=%s batches=%s elapsed_ms=%.3f",
                     job.key, c.tp_rank, time.monotonic_ns(), len(job.batches),
                     (time.monotonic() - started) * 1000)
        except Exception as error:
            self._answer(job, str(error))

    @staticmethod
    def ratio(batch, now):
        # Shards not READY at reserve, including peer-owned in-flight work.
        return 1 + (now - batch.entered) / max(batch.reservation["remaining_l2_work"], 1)

    def _choose(self, round_keys):
        # Condition held; one pending batch per request participates in Stan
        # HRRN selection. A request gets one quantum before any repeat while
        # another eligible request has not had a turn.
        self.preparing = [item for item in self.preparing
                          if not item[0].cancelled and not item[1].done and item[1].can_submit]
        heads = {}
        for job, batch in self.preparing:
            current = heads.get(job.key)
            if current is None or batch.index < current[1].index:
                heads[job.key] = (job, batch)
        round_keys.intersection_update(heads)
        if not round_keys:
            round_keys.update(heads)
        eligible = [heads[key] for key in round_keys]
        now = time.monotonic()
        eligible.sort(key=lambda item: self.ratio(item[1], now), reverse=True)
        self.selected = eligible[:max(1, len(eligible) // 4)]
        chosen = {batch for _, batch in self.selected}
        self.preparing = [item for item in self.preparing if item[1] not in chosen]

    def _step(self, job, batch, available):
        # All backend calls are outside the worker condition. A Step either
        # advances bounded work or checks completion; it never waits on healthy
        # unfinished native I/O. Error returns have already drained accepted I/O.
        capability = batch.reservation["batch"]
        try:
            if not batch.started:
                batch.started = True
                log.info("UCM_EARLY batch_start key=%s rank=%s batch=%s reservation=%s ns=%s queue_ms=%.3f",
                         job.key, self.connector.tp_rank, batch.index, batch.reservation["handle"],
                         time.monotonic_ns(), (time.monotonic() - batch.entered) * 1000)
            stats = capability.step(available)
            with self.condition:
                batch.stats = stats
                batch.inflight_bytes = stats["inflight_bytes"]
                batch.can_submit = stats["can_submit"]
                # Conservative shard work: peer completions not yet observed
                # may remain charged. This never substitutes total hit length.
                batch.reservation["remaining_l2_work"] = stats["remaining_l2_work"]
                if stats["done"]:
                    batch.done = True
                    log.info("UCM_EARLY batch_done key=%s rank=%s batch=%s reservation=%s ns=%s "
                             "protected_claim_bytes=%s protection_limit_bytes=%s stats=%s",
                             job.key, self.connector.tp_rank, batch.index, batch.reservation["handle"],
                             time.monotonic_ns(), capability.protected_bytes,
                             capability.protection_limit_bytes, stats)
                    if job.reserve_done and all(part.done for part in job.batches):
                        self._answer(job)
                return stats["advanced"]
        except Exception as error:
            with self.condition:
                batch.done, batch.can_submit, batch.inflight_bytes = True, False, 0
                self._answer(job, str(error))
            return True

    def _data(self):
        active, round_keys = [], set()
        window, peak = 0, 0
        while True:
            advanced = False
            for job, batch in active:
                if not batch.done:
                    advanced = self._step(job, batch, 0) or advanced
            active = [item for item in active if not item[1].done]
            inflight = sum(batch.inflight_bytes for _, batch in active)
            with self.condition:
                if self.closed:
                    self.preparing.clear()
                    self.selected.clear()
                    if not active:
                        return
                else:
                    self.selected = [item for item in self.selected
                                     if not item[0].cancelled and not item[1].done and item[1].can_submit]
                    if not self.selected:
                        self._choose(round_keys)
                    if self.selected:
                        job, batch = self.selected[0]
                        capability = batch.reservation["batch"]
                        window = min(window or capability.window_bytes, capability.window_bytes)
                        if window - inflight >= capability.shard_bytes:
                            self.selected.pop(0)
                            round_keys.discard(job.key)
                        else:
                            job = batch = None
                    else:
                        job = batch = None
                if self.closed:
                    job = batch = None
            if batch is not None:
                if not batch.started:
                    active.append((job, batch))
                advanced = self._step(job, batch, window - inflight) or advanced
                peak = max(peak, sum(part.inflight_bytes for _, part in active))
                if peak > window:
                    self._answer(job, "Early exceeded native-derived byte window")
                with self.condition:
                    if not job.cancelled and not batch.done and batch.can_submit:
                        self.preparing.append((job, batch))
                continue
            with self.condition:
                if not active and not self.preparing and not self.selected:
                    log.info("UCM_EARLY data_idle rank=%s ns=%s inflight_peak_bytes=%s window_bytes=%s",
                             self.connector.tp_rank, time.monotonic_ns(), peak, window)
                    self.condition.wait_for(lambda: self.closed or self.preparing or self.selected)
                elif not advanced:
                    # Native Check offers no completion callback. Match the
                    # existing bounded readiness cadence, with one data thread.
                    self.condition.wait(timeout=.001)

    def _answer(self, job, error=None):
        with self.condition:
            if job.cancelled or job.ready:
                return
            if error is not None:
                self._release(job)
                log.warning(
                    "UCM_EARLY failed key=%s rank=%s reason=%s",
                    job.key,
                    self.connector.tp_rank,
                    error,
                )
            else:
                job.ready = True
                fast = not any(batch.started for batch in job.batches)
                log.info("UCM_EARLY rank_ready key=%s rank=%s ns=%s total_hit_blocks=%s fast_path=%s",
                         job.key, self.connector.tp_rank, time.monotonic_ns(),
                         sum(batch.reservation["total_hit_blocks"] for batch in job.batches), fast)
                if fast:
                    log.info("UCM_EARLY fast_path key=%s rank=%s data_queue_wait=0 l2_read_shards=0",
                             job.key, self.connector.tp_rank)
            self.control.send(
                {
                    "op": "answer",
                    "key": job.key,
                    "state": "FAILED" if error else "READY",
                    "hit": sum(batch.reservation["total_hit_blocks"] for batch in job.batches),
                    "reason": error,
                }
            )

    def bind(self, metadata):
        with self.condition:
            self.request_keys.update(getattr(metadata, "early_keys", {}))
            loading = {
                self.request_keys.get(request_id)
                for request_id, meta in metadata.request_meta.items()
                if meta.load_block_ids[0]
            }
            handles = []
            for job in self.jobs.values():
                if job.key in loading and not job.cancelled:
                    job.loading = True
                    if job.ready:
                        handles.extend(batch.reservation["handle"] for batch in job.batches)
            self.store.set_load_reservations(handles)

    def loaded(self, request_ids, *, reason="handoff"):
        with self.condition:
            keys = {self.request_keys.get(request_id) for request_id in request_ids}
            for key, job in list(self.jobs.items()):
                if job.key in keys:
                    self._release(job)
                    del self.jobs[key]
                    log.info(
                        "UCM_EARLY released key=%s rank=%s reason=%s ns=%s",
                        job.key,
                        self.connector.tp_rank,
                        reason,
                        time.monotonic_ns(),
                    )

    def finished(self, request_ids):
        with self.condition:
            self.loaded(request_ids, reason="finish")
            for request_id in request_ids:
                self.request_keys.pop(request_id, None)

    def close(self):
        self.finished(set(self.request_keys))
        with self.condition:
            self.closed = True
            for job in list(self.jobs.values()):
                self._release(job)
            self.jobs.clear()
            self.condition.notify_all()
        self.lookup_pool.shutdown(wait=True, cancel_futures=True)
        self.data_thread.join()  # native I/O timeout bounds an in-flight transfer
        self.control.close()
