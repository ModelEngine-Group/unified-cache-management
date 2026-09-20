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
    owner: str | None = None
    attempt: str | None = None
    grant: int = 0
    expected: tuple | None = None
    capacity_truncated: bool = False
    ready_received_ns: int | None = None
    first_pending_ns: int | None = None
    pending_probes: int = 0
    match_calls: int = 0
    match_ns: int = 0
    hbm_tokens: int | None = None
    external_tokens: int | None = None


class EarlyScheduler:
    def __init__(self, connector, control_factory=Control):
        self.connector = connector
        self.lock = threading.RLock()
        self.jobs = {}
        self.by_key = {}
        self.seen = weakref.WeakSet()
        self.traced_requests = weakref.WeakSet()
        self.closed = False
        self.fatal = None
        self.sequence = 0
        self.owner_domain_active = False
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-early-hash"
        )
        parallel = connector._vllm_config.parallel_config
        # Dense engines reset communication DP rank to zero. The native index
        # retains the actual engine identity for both dense and MoE models.
        self.dp_rank = parallel.data_parallel_index
        broker = dict(tp_size=connector.tp_size, dp_size=getattr(parallel, "data_parallel_size", 1)) if self.dp_rank == 0 else None
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
            self.check()
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
                self.sequence += 1
                self.control.send(
                    {
                        "op": "job",
                        "key": job.key,
                        "dp": self.dp_rank,
                        "blocks": blocks,
                        "request": job.request.request_id,
                        "submitted_ns": submitted_ns,
                        "sequence": self.sequence,
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
            self.check()
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
            if message["op"] == "domain_active":
                self.owner_domain_active = True
                return
            if message["op"] == "domain_lost":
                self.fatal = message["reason"]
                return
            if message["op"] == "lost":
                if self.owner_domain_active or any(job.owner for job in self.jobs.values()):
                    self.fatal = "scheduler control lost with owner generations"
                self.closed = True
                for job in self.jobs.values():
                    job.state = "FAILED"
                return
            job = self.by_key.get(message.get("key"))
            if job is None or job.state not in ("PENDING", "READY"):
                return
            if message["op"] == "owner_assigned":
                job.owner, job.attempt, job.grant = message["owner"], message["attempt"], message["grant"]
                return
            job.hit = message.get("hit", 0)
            job.capacity_truncated = message.get("capacity_truncated", False)
            job.state = message["state"]
            if job.state == "READY":
                job.ready_received_ns = time.monotonic_ns()
            job.owner = message.get("owner")
            job.attempt = message.get("attempt")
            job.grant = message.get("grant", 0)
            log.info("UCM_EARLY scheduler_result key=%s dp=%s state=%s ns=%s",
                     job.key, self.dp_rank, job.state, time.monotonic_ns())

    def _cancel(self, job):
        job.state = "CANCELLED"
        self.by_key.pop(job.key, None)
        self.control.send({"op": "cancel", "key": job.key})

    def cancel_unused(self, job):
        # Keep identity/hash state until request finish: an allocation retry
        # must not hash again or revive an already cancelled background job.
        with self.lock:
            if self.jobs.get(job.request.request_id) is job and job.state != "CANCELLED":
                self._cancel(job)

    def scheduled(self, request_meta, scheduled_tokens=None):
        keys = {}
        with self.lock:
            # Native HBM-only requests may have no external dispatch metadata.
            # Observe their actual budget charge from SchedulerOutput as well.
            for request_id, count in (scheduled_tokens or {}).items():
                job = self.jobs.get(request_id)
                if job is not None and job.state == "CANCELLED" and not job.dispatched:
                    job.dispatched = True
                    log.info(
                        "UCM_EARLY hbm_scheduled key=%s request=%s dp=%s "
                        "request_tokens=%s hbm_tokens=%s external_tokens=%s "
                        "scheduled_tokens=%s ns=%s",
                        job.key, request_id, self.dp_rank, job.request.num_tokens,
                        job.hbm_tokens, job.external_tokens, count, time.monotonic_ns(),
                    )
            for request_id, meta in request_meta.items():
                job = self.jobs.get(request_id)
                if job is None or job.state == "CANCELLED":
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
                    log.info(
                        "UCM_EARLY schedule_match key=%s request=%s dp=%s owner=%s "
                        "ready_received_ns=%s first_pending_ns=%s pending_probes=%s "
                        "match_calls=%s match_ms=%.3f hbm_tokens=%s external_tokens=%s "
                        "request_tokens=%s scheduled_tokens=%s "
                        "scheduled_ns=%s handoff_transport=%s",
                        job.key, request_id, self.dp_rank, job.owner,
                        job.ready_received_ns, job.first_pending_ns, job.pending_probes,
                        job.match_calls, job.match_ns / 1e6, job.hbm_tokens,
                        job.external_tokens, job.request.num_tokens,
                        None if scheduled_tokens is None else scheduled_tokens.get(request_id),
                        scheduled_ns,
                        "metadata" if job.owner else "control",
                    )
                    # Owner tickets already travel in native step metadata. The
                    # broker ignores owner handoff; it must not release pins.
                    if not job.owner:
                        if meta.load_block_ids[0]:
                            # Keep legacy handoff ordered before finish/cancel.
                            self.control.send(
                                {"op": "handoff", "key": job.key, "request": request_id}
                            )
                        else:
                            self.control.send({"op": "finish", "key": job.key})
        return keys

    def check(self):
        if getattr(self, "fatal", None):
            raise RuntimeError("Early shared domain fenced; restart with fresh native L1: " + self.fatal)

    def owner_metadata(self, keys):
        with self.lock:
            tickets = {}
            for request_id, key in keys.items():
                job = self.by_key.get(key)
                if job is not None and job.owner and job.state == "READY":
                    if job.expected is None:
                        raise RuntimeError("owner result lacks native consumption plan")
                    tickets[request_id] = dict(key=key, attempt=job.attempt, owner=job.owner,
                                              grant=job.grant, expected=job.expected,
                                              protected_hit=job.hit, capacity_truncated=job.capacity_truncated)
            return tickets

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
    owner: str | None = None
    attempt: str | None = None
    grant: int = 0
    capacity_end: int | None = None
    truncate_reason: str | None = None
    lookup_finished: bool = False
    request_id: str | None = None
    deferred: bool = False
    lookup_cursor: int = 0
    lookup_count: int = 32
    lookup_paused: bool = False


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
    discarded: bool = False
    eligibility_requested: bool = False
    eligible: bool = False
    activating: bool = False
    activated: bool = False
    waiting_capacity: bool = False
    wait_epoch: int = 0
    wait_started: float | None = None



class EarlyWorker:
    def __init__(self, connector, control_factory=Control):
        self.connector, self.store = connector, connector.store
        self.condition = threading.Condition()
        self.release_serial = 0
        self.jobs = {}
        self.preparing = []
        self.selected = []
        self.request_keys = {}
        self.closed = False
        self.lookup_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ucm-early-lookup"
        )
        # Keep four background execution slots in total. Activation must not
        # sit behind whole-prefix queries or the first-read pipeline is lost.
        self.activation_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ucm-early-activate"
        )
        dp_rank = connector._vllm_config.parallel_config.data_parallel_index
        from .early_owner import worker_capability
        from .early_owner_worker import OwnerWorker
        peer = f"w{dp_rank}.{connector.tp_rank % connector.tp_size}"
        capability = worker_capability(connector)
        self.owner_protocol = OwnerWorker(self, capability, peer)
        self.io_limit = 0
        log.info("UCM_EARLY owner_capability peer=%s capability=%s", peer, capability)
        self.control = control_factory(
            f"ipc:///dev/shm/ucm_early_{connector.unique_id}.sock",
            peer,
            self.receive,
            hello={"capability": capability},
        )
        self.data_thread = threading.Thread(
            target=self._data, name="ucm-early-data", daemon=True
        )
        self.data_thread.start()
        self.release_thread = None
        if capability.get("ready") and callable(getattr(type(self.store), "wait_release", None)):
            self.release_thread = threading.Thread(target=self._release_events,
                name="ucm-early-release", daemon=True)
            self.release_thread.start()

    def _release_events(self):
        try:
            epoch = self.store.release_epoch()
            while not self.closed and not self.owner_protocol.fatal:
                current = self.store.wait_release(epoch, 250)
                if current == epoch:
                    continue  # shutdown heartbeat only; no capacity retry
                epoch = current
                with self.condition:
                    self.release_serial += 1
                    self.condition.notify_all()
        except Exception as error:
            self.owner_protocol.poison(f"release event unavailable: {error}")

    def _request_execution(self, job, batch):
        if batch.eligibility_requested:
            return
        batch.eligibility_requested = True
        batch.can_submit = False
        self.control.send(dict(op="execution_request", key=job.key, attempt=job.attempt,
                               batch=batch.index, begin=batch.begin,
                               count=batch.reservation["total_hit_blocks"]))

    def _activate(self, job, batch):
        try:
            started_ns = time.monotonic_ns()
            stats = batch.reservation["batch"].activate()
            log.info("UCM_EARLY batch_activated key=%s attempt=%s rank=%s batch=%s reservation=%s "
                     "pin_start_ns=%s pin_end_ns=%s hit=%s remaining_l2_work=%s protected_bytes=%s",
                     job.key, job.attempt, self.connector.tp_rank, batch.index,
                     batch.reservation["handle"], started_ns, time.monotonic_ns(),
                     stats["total_hit_blocks"], stats["remaining_l2_work"],
                     batch.reservation["batch"].protected_bytes)
            with self.condition:
                batch.activating = False
                if job.cancelled or batch.discarded:
                    self.store.release_reservation(batch.reservation["handle"])
                    batch.done = True
                    return
                batch.activated = True
                old = batch.reservation["total_hit_blocks"]
                batch.reservation.update(total_hit_blocks=stats["total_hit_blocks"],
                                         remaining_l2_work=stats["remaining_l2_work"])
                if stats["total_hit_blocks"] < old:
                    # Actual L1-only data disappeared; distinguish from NoSpace.
                    log.info("UCM_EARLY revalidation_miss key=%s batch=%s lost_blocks=%s",
                             job.key, batch.index, old - stats["total_hit_blocks"])
                    self._truncate(job, batch.begin + stats["total_hit_blocks"], reason="revalidation_miss")
                batch.done = stats["done"]
                batch.can_submit = not batch.done
                if not batch.done:
                    self.preparing.append((job, batch))
                self._maybe_answer(job)
                self.condition.notify_all()
        except Exception as error:
            with self.condition:
                batch.activating = False
                batch.done = True
                self._answer(job, str(error))

    def receive(self, message):
        if self.owner_protocol.receive(message):
            return
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
        if job.lookup_paused:
            job.lookup_paused = False
            job.lookup_finished = True
        for batch in job.batches:
            self.store.release_reservation(batch.reservation["handle"])
        self.condition.notify_all()

    def _reserve(self, job):
        started = time.monotonic()
        c = self.connector
        log.info("UCM_EARLY lookup_start key=%s rank=%s ns=%s queue_ms=%.3f",
                 job.key, c.tp_rank, time.monotonic_ns(), (started - job.entered) * 1000)
        paused = False
        try:
            blocks = job.blocks
            if not c.is_mla and c.tp_rank % c.tp_size:
                blocks = [c.request_hasher(block) for block in blocks]
            begin, candidate_count = job.lookup_cursor, job.lookup_count
            while begin < len(blocks):
                with self.condition:
                    if job.cancelled or self.closed or self.owner_protocol.fatal:
                        return
                    if job.capacity_end is not None:
                        break
                    if job.deferred and any(b.waiting_capacity for b in job.batches):
                        paused = job.lookup_paused = True
                        return
                candidates = blocks[begin:begin + candidate_count]
                reserving = time.monotonic_ns()
                reservation = (self.store.reserve_prefix(candidates, deferred=True) if job.deferred
                               else self.store.reserve_prefix(candidates))
                hit = reservation["total_hit_blocks"]
                if not 0 <= hit <= len(candidates):
                    self.store.release_reservation(reservation["handle"])
                    raise RuntimeError("invalid reserved prefix length")
                batch = WorkerBatch(len(job.batches), begin, reservation,
                                    done=not job.deferred and reservation["remaining_l2_work"] == 0)
                with self.condition:
                    if self.owner_protocol.fatal:
                        job.batches.append(batch)  # quarantine newly acquired pins too
                        return
                    if job.cancelled or self.closed or job.capacity_end is not None:
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
                    if job.deferred and reservation["remaining_l2_work"] == 0:
                        # L1 fast path acquires execution eligibility and pins
                        # outside the SSD data queue, including L1-only blocks.
                        self._request_execution(job, batch)
                    elif not batch.done:
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
                job.lookup_cursor, job.lookup_count = begin, candidate_count
            with self.condition:
                job.reserve_done = True
                if job.deferred:
                    self.control.send(dict(op="lookup_complete", key=job.key, attempt=job.attempt,
                                           hit=sum(b.reservation["total_hit_blocks"] for b in job.batches)))
            log.info("UCM_EARLY lookup_done key=%s rank=%s ns=%s batches=%s elapsed_ms=%.3f",
                     job.key, c.tp_rank, time.monotonic_ns(), len(job.batches),
                     (time.monotonic() - started) * 1000)
        except Exception as error:
            self._answer(job, str(error))
        finally:
            with self.condition:
                job.lookup_finished = not paused
                if job.capacity_end is not None:
                    job.reserve_done = True
                self._maybe_answer(job)
                self.condition.notify_all()

    def _truncate(self, job, end, reason="capacity"):
        # Exceptional capacity or an actually missing revalidated L1-only tail.
        # Normal resource waits never call this method.
        job.truncate_reason = reason
        job.capacity_end = end if job.capacity_end is None else min(end, job.capacity_end)
        for part in job.batches:
            if part.begin < job.capacity_end or part.discarded:
                continue
            part.discarded, part.can_submit = True, False
            if part.activating:
                self.store.release_reservation(part.reservation["handle"])
            elif part.started and not part.done:
                part.reservation["batch"].discard()  # only stops submission, no wait
            else:
                part.done = True
                self.store.release_reservation(part.reservation["handle"])
        self.condition.notify_all()

    @staticmethod
    def _hit(job):
        # Batches are appended in prefix order. Final lengths replace lookup
        # lengths only after completion; a partial block is never counted.
        hit = 0
        for part in job.batches:
            if part.discarded:
                break
            if part.begin != hit:
                raise RuntimeError("non-contiguous completed prefix")
            hit += part.reservation["total_hit_blocks"]
        return hit

    def _maybe_answer(self, job):
        if (job.reserve_done and job.lookup_finished
                and all(part.done for part in job.batches)):
            self._answer(job)

    @staticmethod
    def ratio(batch, now):
        # Shards not READY at reserve, including peer-owned in-flight work.
        return 1 + (now - batch.entered) / max(batch.reservation["remaining_l2_work"], 1)

    def _choose(self, round_keys):
        # Condition held; one pending batch per request participates in Stan
        # HRRN selection. A request gets one quantum before any repeat while
        # another eligible request has not had a turn.
        self.preparing = [item for item in self.preparing
                          if not item[0].cancelled and not item[1].done and not item[1].discarded and item[1].can_submit]
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
                waiting = stats.get("waiting_capacity", False)
                if waiting and not batch.waiting_capacity:
                    batch.wait_started = time.monotonic()
                    log.info("UCM_EARLY capacity_wait key=%s attempt=%s batch=%s reservation=%s epoch=%s ns=%s",
                             job.key, job.attempt, batch.index, batch.reservation["handle"],
                             stats["wait_epoch"], time.monotonic_ns())
                elif batch.waiting_capacity and not waiting:
                    log.info("UCM_EARLY capacity_resumed key=%s attempt=%s batch=%s wait_ms=%.3f ns=%s",
                             job.key, job.attempt, batch.index,
                             (time.monotonic() - batch.wait_started) * 1000, time.monotonic_ns())
                was_waiting = batch.waiting_capacity
                batch.waiting_capacity = waiting
                batch.wait_epoch = stats.get("wait_epoch", 0)
                batch.inflight_bytes = stats["inflight_bytes"]
                batch.can_submit = stats["can_submit"] and not batch.discarded
                if was_waiting and not waiting and batch.can_submit:
                    if (job, batch) not in self.preparing:
                        self.preparing.append((job, batch))
                    if job.lookup_paused and not job.cancelled:
                        job.lookup_paused = False
                        self.lookup_pool.submit(self._reserve, job)
                # Conservative shard work: peer completions not yet observed
                # may remain charged. This never substitutes total hit length.
                batch.reservation["remaining_l2_work"] = stats["remaining_l2_work"]
                if stats["done"]:
                    batch.done = True
                    if batch.discarded:
                        self.store.release_reservation(batch.reservation["handle"])
                    else:
                        hit = stats["total_hit_blocks"]
                        if type(hit) is not int or not 0 <= hit <= batch.reservation["total_hit_blocks"]:
                            raise RuntimeError("invalid completed prefix length")
                        batch.reservation["total_hit_blocks"] = hit
                        if stats["capacity_truncated"]:
                            self._truncate(job, batch.begin + hit)
                    log.info("UCM_EARLY batch_done key=%s rank=%s batch=%s reservation=%s ns=%s "
                             "protected_reference_bytes=%s physical_reservable_bytes=%s stats=%s",
                             job.key, self.connector.tp_rank, batch.index, batch.reservation["handle"],
                             time.monotonic_ns(), capability.protected_bytes,
                             capability.protection_limit_bytes, stats)
                    self._maybe_answer(job)
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
            if self.owner_protocol.fatal:
                return  # Quarantine; native references are not death-recoverable.
            self.owner_protocol.tick()
            with self.condition:
                observed_release = self.release_serial
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
                                     if not item[0].cancelled and not item[1].done and not item[1].discarded and item[1].can_submit]
                    if not self.selected:
                        self._choose(round_keys)
                    if self.selected:
                        job, batch = self.selected[0]
                        capability = batch.reservation["batch"]
                        window = min(window or capability.window_bytes, capability.window_bytes)
                        if self.io_limit:
                            window = min(window, self.io_limit)
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
                if job.deferred and not batch.activated:
                    with self.condition:
                        self._request_execution(job, batch)
                    continue
                if not batch.started:
                    active.append((job, batch))
                advanced = self._step(job, batch, window - inflight) or advanced
                peak = max(peak, sum(part.inflight_bytes for _, part in active))
                if peak > window:
                    self._answer(job, "Early exceeded native-derived byte window")
                with self.condition:
                    if not job.cancelled and not batch.discarded and not batch.done and batch.can_submit:
                        self.preparing.append((job, batch))
                continue
            with self.condition:
                if not active and not self.preparing and not self.selected:
                    log.debug("UCM_EARLY data_idle rank=%s ns=%s inflight_peak_bytes=%s window_bytes=%s",
                             self.connector.tp_rank, time.monotonic_ns(), peak, window)
                    self.condition.wait(timeout=.1 if self.owner_protocol.release_requested else 2)
                elif not advanced:
                    # Native Check offers no completion callback. Match the
                    # existing bounded readiness cadence, with one data thread.
                    only_resource_wait = active and all(
                        b.waiting_capacity and not b.inflight_bytes for _, b in active)
                    if self.release_serial == observed_release:
                        self.condition.wait(timeout=2 if only_resource_wait else .001)

    def _answer(self, job, error=None):
        with self.condition:
            if self.owner_protocol.fatal:
                return
            if job.owner:
                self.owner_protocol.answer(job, error)
                return
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
                hit = self._hit(job)
                fast = hit > 0 and not any(batch.started for batch in job.batches)
                log.info("UCM_EARLY rank_ready key=%s rank=%s ns=%s total_hit_blocks=%s fast_path=%s",
                         job.key, self.connector.tp_rank, time.monotonic_ns(),
                         self._hit(job), fast)
                if fast:
                    log.info("UCM_EARLY fast_path key=%s rank=%s data_queue_wait=0 l2_read_shards=0",
                             job.key, self.connector.tp_rank)
            self.control.send(
                {
                    "op": "answer",
                    "key": job.key,
                    "state": "FAILED" if error else "READY",
                    "hit": self._hit(job),
                    "reason": error,
                }
            )

    def bind(self, metadata):
        self.owner_protocol.bind(metadata)
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
                        handles.extend(batch.reservation["handle"] for batch in job.batches if not batch.discarded)
            self.store.set_load_reservations(handles)

    def loaded(self, request_ids, *, reason="handoff"):
        with self.condition:
            keys = {self.request_keys.get(request_id) for request_id in request_ids}
            for key, job in list(self.jobs.items()):
                if job.key in keys and not job.owner:
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
        if self.owner_protocol.fatal:
            return
        with self.condition:
            # Native worker callbacks carry reusable request IDs, not a
            # generation. Only the broker's generation-scoped retire can stop
            # owner consumption; an old finish must not stop a new request.
            self.loaded(request_ids, reason="finish")
            for request_id in request_ids:
                job = self.jobs.get(self.request_keys.get(request_id))
                if job is None or not job.owner:
                    self.request_keys.pop(request_id, None)

    def close(self):
        if any(job.owner for job in self.jobs.values()):
            self.owner_protocol.shutdown()
        if self.owner_protocol.fatal:
            from .early_owner_worker import quarantine
            quarantine.append(self)
            self.closed = True
            with self.condition:
                self.condition.notify_all()
            self.lookup_pool.shutdown(wait=False, cancel_futures=True)
            self.activation_pool.shutdown(wait=False, cancel_futures=True)
            self.control.close()
            return
        self.finished(set(self.request_keys))
        with self.condition:
            self.closed = True
            for job in list(self.jobs.values()):
                self._release(job)
            self.jobs.clear()
            self.condition.notify_all()
        self.lookup_pool.shutdown(wait=True, cancel_futures=True)
        self.activation_pool.shutdown(wait=True, cancel_futures=True)
        self.data_thread.join()  # native I/O timeout bounds an in-flight transfer
        if self.release_thread is not None:
            self.release_thread.join()
        self.control.close()
