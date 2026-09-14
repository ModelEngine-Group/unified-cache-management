"""Early-only request state and worker queues; native scheduling/load stay native."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
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
                self.control.send(
                    {
                        "op": "job",
                        "key": job.key,
                        "dp": self.dp_rank,
                        "blocks": blocks,
                        "request": job.request.request_id,
                    }
                )
            log.info(
                "UCM_EARLY submitted key=%s request=%s blocks=%s",
                job.key,
                job.request.request_id,
                len(blocks),
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


@dataclass
class WorkerJob:
    key: str
    blocks: list
    entered: float = field(default_factory=time.monotonic)
    batches: list = field(default_factory=list)
    reserve_done: bool = False
    cancelled: bool = False
    ready: bool = False
    loading: bool = False


@dataclass
class WorkerBatch:
    index: int
    begin: int
    reservation: dict
    entered: float = field(default_factory=time.monotonic)
    done: bool = False


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
                        "start_ns=%s end_ns=%s candidates=%s hit=%s remaining_l2_work=%s",
                        job.key, c.tp_rank, batch.index, begin, reservation["handle"],
                        reserving, time.monotonic_ns(), len(candidates),
                        reservation["total_hit_blocks"], reservation["remaining_l2_work"],
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

    def _data(self):
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.closed or self.preparing or self.selected
                )
                if self.closed:
                    return
                if not self.selected:
                    self.preparing = [item for item in self.preparing if not item[0].cancelled]
                    now = time.monotonic()
                    self.preparing.sort(key=lambda item: self.ratio(item[1], now), reverse=True)
                    count = max(1, len(self.preparing) // 4)
                    self.selected, self.preparing = self.preparing[:count], self.preparing[count:]
                if not self.selected:
                    continue
                job, batch = self.selected.pop(0)
                if job.cancelled:
                    continue
            try:
                log.info("UCM_EARLY batch_start key=%s rank=%s batch=%s reservation=%s ns=%s queue_ms=%.3f",
                         job.key, self.connector.tp_rank, batch.index, batch.reservation["handle"],
                         time.monotonic_ns(), (time.monotonic() - batch.entered) * 1000)
                stats = self.store.prefetch_reserved(batch.reservation["handle"])
                with self.condition:
                    batch.done = True
                    log.info("UCM_EARLY batch_done key=%s rank=%s batch=%s reservation=%s ns=%s stats=%s",
                             job.key, self.connector.tp_rank, batch.index, batch.reservation["handle"],
                             time.monotonic_ns(), stats)
                    if job.reserve_done and all(part.done for part in job.batches):
                        self._answer(job)
            except Exception as error:
                self._answer(job, str(error))

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
                fast = all(batch.reservation["remaining_l2_work"] == 0 for batch in job.batches)
                log.info("UCM_EARLY rank_ready key=%s rank=%s ns=%s total_hit_blocks=%s fast_path=%s",
                         job.key, self.connector.tp_rank, time.monotonic_ns(),
                         sum(batch.reservation["total_hit_blocks"] for batch in job.batches), fast)
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

    def loaded(self, request_ids):
        with self.condition:
            keys = {self.request_keys.get(request_id) for request_id in request_ids}
            for key, job in list(self.jobs.items()):
                if job.key in keys:
                    self._release(job)
                    del self.jobs[key]
                    log.info(
                        "UCM_EARLY released key=%s rank=%s reason=handoff ns=%s",
                        job.key,
                        self.connector.tp_rank,
                        time.monotonic_ns(),
                    )

    def finished(self, request_ids):
        with self.condition:
            self.loaded(request_ids)
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
