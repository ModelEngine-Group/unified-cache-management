"""SGLang direct linker for Store V2.

Lookup, layerwise Get and Put for known pages share a Store request route.
Supports replicated DSA and DeepSeek V4 device pools.
"""

import hashlib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from functools import wraps
from itertools import count
from threading import Lock
from time import perf_counter, sleep, thread_time
from uuid import uuid4

import torch.distributed as dist
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.utils import get_device_module

from ucm.integration.sglang.patch.request_hooks import install_request_hooks
from ucm.logger import init_logger

from .device_layout import DeviceLayout, fingerprint
from .runtime import Runtime, proxy

logger = init_logger(__name__)


def _log_duration(name, threshold_ms=20):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args[0]._log_performance:
                return func(*args, **kwargs)
            start = perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (perf_counter() - start) * 1e3
                if duration_ms >= threshold_ms:
                    logger.info(
                        "SGLang UCM V2 slow operation=%s duration_ms=%.3f",
                        name,
                        duration_ms,
                    )

        return wrapper

    return decorate


class _MeasuredLayerWiseLoadCounter(LayerWiseLoadCounter):
    def __init__(self, num_layers, *, log_performance=True):
        super().__init__(num_layers)
        self._log_performance = log_performance
        self._measured_layers = num_layers
        self._wait_totals = {}

    def wait_until(self, threshold):
        if not self._log_performance:
            return super().wait_until(threshold)
        # The model/KV pool calls the counter directly, outside linker methods.
        index = self.consumer_index
        start = perf_counter()
        try:
            return super().wait_until(threshold)
        finally:
            duration_ms = (perf_counter() - start) * 1e3
            total, peak, calls = self._wait_totals.get(index, (0.0, 0.0, 0))
            self._wait_totals[index] = (
                total + duration_ms,
                max(peak, duration_ms),
                calls + 1,
            )
            if threshold == self._measured_layers - 1:
                self._log_wait_total(index)

    def _log_wait_total(self, index):
        values = self._wait_totals.pop(index, None)
        if values is not None:
            total, peak, calls = values
            if total >= 20:
                logger.info(
                    "SGLang UCM V2 load waits: batch=%s calls=%d "
                    "total_ms=%.3f max_ms=%.3f",
                    index,
                    calls,
                    total,
                    peak,
                )

    def finish(self, index):
        try:
            return super().finish(index)
        finally:
            self._log_wait_total(index)

    def reset(self):
        self._wait_totals.clear()
        return super().reset()


class _OffloadResult(Future):
    def __init__(self):
        super().__init__()
        self.reservations = []


class UcmDirectLinker(UnifiedCacheLinker):
    load_mode = "layer_wise"
    prefetch_policy = "timeout"

    def __init__(
        self, server_args, params, *, components, extra_config=None, forward_stream=None
    ):
        self.forward_stream = forward_stream
        self.page_size = params.page_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        self._require_single_host(params)
        pool_group = resolve_hybrid_device_pool_group(
            kvcache=params.token_to_kv_pool_allocator.get_kvcache(),
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.layout = DeviceLayout(
            pool_group, "sglang", get_model().model_path, tp_group=tp_group
        )
        self.pools = {entry.name: entry for entry in self.layout.entries}
        self._coordinated_puts = self.layout.has_shared_shards
        worker_index, worker_number = 0, 1
        if tp_group is not None and dist.is_initialized():
            worker_index = dist.get_rank(tp_group)
            worker_number = dist.get_world_size(tp_group)
        self._dump_rank, self._dump_size = 0, 1
        if tp_group is not None and dist.is_initialized():
            if pool_group.rank_replicated:
                self._dump_rank = dist.get_rank(tp_group)
                self._dump_size = dist.get_world_size(tp_group)
        extra_config = extra_config or {}
        # The assembler has no capability query for prerequisite events.
        self._forward_event_to_store = extra_config.get("forward_event_to_store", False)
        # Cache domains are scoped to a DP replica and shared by its TP ranks.
        parallel = get_parallel()
        tp_rank = getattr(parallel, "tp_rank", None)
        self._log_performance = (worker_index if tp_rank is None else tp_rank) == 0
        unique_id = fingerprint(
            (
                self.layout.identity,
                getattr(parallel, "dp_rank", None) or 0,
                getattr(parallel, "attn_dp_rank", None) or 0,
                self._dump_size,
                params.pp_rank,
                params.pp_size,
                params.attn_cp_rank,
                params.attn_cp_size,
                self.layout.section_sizes,
            )
        )[:32]
        self.runtime = Runtime(
            extra_config,
            unique_id=unique_id,
            block_size=max(self.layout.section_sizes.values()),
            device_id=self.layout.device_id,
            worker_index=worker_index,
            worker_number=worker_number,
            dump_partitions=self._dump_size,
            coordinated_puts=self._coordinated_puts,
        )
        install_request_hooks(type(self), coordinated_puts=self._coordinated_puts)
        # CP uses separate cache domains. Within an attention-TP cache domain,
        # a committed Mamba object already includes every rank's state shard.
        self._lookup_owner = not (
            params.attn_cp_size == 1 and worker_number > 1 and worker_index != 0
        )
        self._registered_requests = set()
        self._request_pages = {}
        self._page_requests = {}
        self._pending_puts = {}
        self._finished_requests = set()
        self._scheduler_lock = Lock()
        self.layer_done_counter = _MeasuredLayerWiseLoadCounter(
            self.layout.num_layers, log_performance=self._log_performance
        )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self._pending_loads = {}
        self._load_batches = {}
        self._offload_results = deque()
        self._uncommitted_offloads = deque()
        self._jobs = {}
        self._poll_queued = False
        self._store_offloads = set()
        self._store_offload_error = None
        self._store_offload_lock = Lock()
        self._store_offload_queued = False
        self._closing = False
        self._closed = False
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-v2-load"
        )
        self._dump_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-v2-dump"
        )
        self._job_ids = count(1)
        self._io_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ucm-v2-io",
            initializer=get_device_module().set_device,
            initargs=(self.layout.device_id,),
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-v2-control"
        )
        logger.info(
            "SGLang UCM V2 linker initialized: store_id=%s dp_rank=%s attn_dp_rank=%s "
            "tp_rank=%s pp_rank=%s attn_cp_rank=%s page_size=%s layers=%s "
            "section_bytes=%s dump_partitions=%s coordinated_puts=%s",
            unique_id,
            getattr(parallel, "dp_rank", None) or 0,
            getattr(parallel, "attn_dp_rank", None) or 0,
            getattr(parallel, "tp_rank", None),
            params.pp_rank,
            params.attn_cp_rank,
            self.page_size,
            self.layout.num_layers,
            self.layout.section_sizes,
            self._dump_size,
            self._coordinated_puts,
        )

    @staticmethod
    def _require_single_host(params):
        """Check the worker axes of one DP replica before building its layout."""
        if not dist.is_initialized():
            return
        # SGLang scopes tp_cache_group to attention TP when attention DP is
        # enabled. Do not substitute the world group: independent DP replicas
        # have separate cache domains and may live on different hosts.
        groups = {}
        for name, attribute in (
            ("cache TP", "tp_cache_group"),
            ("attention TP", "attn_tp_cache_group"),
            ("attention CP", "attn_cp_cache_group"),
            ("PP", "pp_cache_group"),
        ):
            group = getattr(params, attribute, None)
            if group is not None and dist.get_world_size(group) > 1:
                groups.setdefault(group, []).append(name)
        if not groups:
            return

        host_id = None
        try:
            with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
                host_id = stream.read().strip() or None
        except OSError:
            host_id = None
        # Carry the earlier axes' observations into the later axes, so a
        # mismatch discovered in one TP slice reaches the whole replica.
        host_ids = {host_id}
        checked_axes = []
        for group, names in groups.items():
            peers = [None] * dist.get_world_size(group)
            dist.all_gather_object(peers, host_ids, group=group)
            host_ids = set().union(*peers)
            checked_axes.extend(names)
        # Finish every collective before rejecting the topology; an early
        # exception could leave peers blocked in a later cache-group check.
        if None in host_ids or len(host_ids) != 1:
            raise ValueError(
                "SGLang UCM Store V2 requires all workers of each DP replica "
                "on one host with a readable boot ID: Cache v2 uses host-local "
                "shared memory. Cross-node TP, PP, and attention CP within a "
                "DP replica are not supported. "
                f"Checked cache axes: {', '.join(checked_axes)}; "
                f"observed hosts: {len(host_ids - {None})}; "
                f"missing host identity: {None in host_ids}. "
                "Independent DP replicas may run on different hosts."
            )

    def lookup(self, rid, transfers):
        """Intersect restorable boundaries across the requested sections."""
        start = perf_counter()
        store_ms = 0.0
        store_calls = 0

        def query(method, keys):
            nonlocal store_ms, store_calls
            query_start = perf_counter()
            store_calls += 1
            try:
                return method(rid, keys)
            finally:
                store_ms += (perf_counter() - query_start) * 1e3

        self._check_open()
        expanded = self.layout.expand(transfers)
        if not expanded:
            return []
        _, fa_transfers = expanded[0]
        page_keys = fa_transfers[0].keys
        candidates = list(range(1, len(page_keys) + 1))
        with self._scheduler_lock:
            self._associate_pages(rid, page_keys)
            if not self._lookup_owner:
                # SGLang intersects boundaries across ranks: all boundaries
                # are neutral. Keep local request registration for Get/Put.
                return candidates
            for section, section_transfers in expanded:
                block_ids = section.block_ids(page_keys[: candidates[-1]])
                keys = b"".join(block_ids)
                transfer = section_transfers[0]
                if transfer.hit_policy != "trailing_pages":
                    index, _prefetchable = query(self.runtime.lookup_on_prefix, keys)
                    candidates = [end for end in candidates if end <= index + 1]
                elif len(transfer.keys) == 1:
                    # State snapshots may have sparse boundaries. Enumerate all
                    # hits, not just the latest: another section/rank may need
                    # an earlier boundary when SGLang intersects the results.
                    boundaries = set()
                    remaining = len(block_ids)
                    while remaining:
                        index, _prefetchable = query(
                            self.runtime.lookup_on_reverse,
                            b"".join(block_ids[:remaining]),
                        )
                        if index < 0:
                            break
                        boundaries.add(index + 1)
                        remaining = index
                    candidates = [end for end in candidates if end in boundaries]
                else:
                    # A multi-page tail requires every page in its window;
                    # a rightmost-hit index alone cannot establish that.
                    hits, _prefetchable = query(self.runtime.lookup, keys)
                    prefix = [0]
                    for hit in hits:
                        prefix.append(prefix[-1] + int(not hit))
                    window = len(transfer.keys)
                    candidates = [
                        end
                        for end in candidates
                        if end < len(prefix)
                        and prefix[end] == prefix[max(0, end - window)]
                    ]
                if not candidates:
                    break
        total_ms = (perf_counter() - start) * 1e3
        if self._log_performance and total_ms >= 10:
            logger.info(
                "SGLang UCM V2 lookup: rid=%s requested_pages=%d "
                "restorable_prefix_pages=%d candidates=%s total_ms=%.3f "
                "store_ms=%.3f other_ms=%.3f store_calls=%d",
                rid,
                len(page_keys),
                candidates[-1] if candidates else 0,
                candidates,
                total_ms,
                store_ms,
                max(0.0, total_ms - store_ms),
                store_calls,
            )
        return candidates

    def associate_pages(self, rid, keys):
        with self._scheduler_lock:
            self._associate_pages(rid, keys)

    def _associate_pages(self, rid, keys):
        if rid not in self._registered_requests:
            context = proxy().RequestContext()
            context.request_id = rid
            self.runtime.on_new_request(context)
            self._registered_requests.add(rid)
        request_pages = self._request_pages.setdefault(rid, set())
        for key in keys:
            if key not in request_pages:
                request_pages.add(key)
                self._page_requests.setdefault(key, set()).add(rid)

    def finish_request(self, rid):
        """Release the route after the caller is done with this Store request."""
        with self._scheduler_lock:
            for key in self._request_pages.pop(rid, ()):
                owners = self._page_requests[key]
                owners.remove(rid)
                if not owners:
                    del self._page_requests[key]
            if self._pending_puts.get(rid):
                self._finished_requests.add(rid)
            else:
                self._close_request(rid)

    def _close_request(self, rid):
        if rid in self._registered_requests:
            self.runtime.on_request_finished(rid)
            self._registered_requests.remove(rid)
        self._finished_requests.discard(rid)

    def _release_put_request(self, rid):
        remaining = self._pending_puts[rid] - 1
        if remaining:
            self._pending_puts[rid] = remaining
        else:
            del self._pending_puts[rid]
            if rid in self._finished_requests:
                self._close_request(rid)

    @_log_duration("load")
    def load(self, rid, transfers):
        self._check_open()
        expanded = self.layout.expand(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded or rid not in self._registered_requests:
            return False
        if rid in self._pending_loads:
            raise RuntimeError(f"UCM load for {rid} is already queued")
        self._pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid):
        if self._pending_loads.pop(rid, None) is None:
            return False
        self.finish_request(rid)
        return True

    @_log_duration("start_layer_wise_loading")
    def start_layer_wise_loading(self):
        if not self._pending_loads:
            return -1
        events = self._record_transfer_events()
        index = self.layer_done_counter.update_producer()
        pending, self._pending_loads = self._pending_loads, {}
        self._load_batches[index] = self._load_executor.submit(
            self._load_worker, index, pending, events, perf_counter()
        )
        return index

    def _complete_get(self, rid, token):
        with self._scheduler_lock:
            self.runtime.complete_get(rid, token)

    def _record_transfer_events(self):
        record_ready = getattr(self, "record_transfer_ready", None)
        if record_ready is not None:
            return record_ready().events

        device_module = get_device_module()
        indices_ready = device_module.Event()
        indices_ready.record()
        events = [indices_ready]
        if self.forward_stream is not None:
            forward_done = device_module.Event()
            forward_done.record(self.forward_stream)
            events.append(forward_done)
        return events

    def _load_worker(self, index, pending, events, queued_at=None):
        start = perf_counter()
        queue_ms = (start - queued_at) * 1e3 if queued_at is not None else 0.0
        event_ms = prepare_ms = forward_ms = layers_ms = 0.0
        metrics = dict(
            payload_ms=0.0,
            payload_cpu_ms=0.0,
            addr_ms=0.0,
            tolist_ms=0.0,
            pack_ms=0.0,
            native_pack_ms=0.0,
            native_submit_ms=0.0,
            native_timed_jobs=0,
            io_queue_ms=0.0,
            submit_ms=0.0,
            job_latency_sum_ms=0.0,
            publish_lag_ms=0.0,
            bytes=0,
            segments=0,
            jobs=0,
        )
        try:
            with ExitStack() as requests:
                indices_ready, *forward_events = events
                event_start = perf_counter()
                indices_ready.synchronize()
                event_ms = (perf_counter() - event_start) * 1e3
                prepare_start = perf_counter()
                context = proxy().IoContext()
                prepared = []
                for rid, sections in pending.items():
                    for section, transfers in sections:
                        pages = section.prepare(transfers)
                        block_ids = section.block_ids(pages.keys)
                        with self._scheduler_lock:
                            token = self.runtime.prepare_get(rid, b"".join(block_ids))
                        requests.callback(self._complete_get, rid, token)
                        if tuple(key.to_bytes() for key in token.keys) != block_ids:
                            raise RuntimeError(
                                f"Store Get token does not match the requested pages for {rid}"
                            )
                        prepared.append((pages, token))

                prepare_ms = (perf_counter() - prepare_start) * 1e3
                forward_start = perf_counter()
                self._apply_forward_dependency(context, forward_events)
                forward_ms = (perf_counter() - forward_start) * 1e3
                layers_start = perf_counter()
                try:
                    self._load_layers(index, prepared, context, metrics)
                finally:
                    layers_ms = (perf_counter() - layers_start) * 1e3
        except Exception as error:
            logger.exception("UCM V2 load batch %s failed", index)
            self.layer_done_counter.fail(index, error)
            return False
        finally:
            total_ms = queue_ms + (perf_counter() - start) * 1e3
            if self._log_performance and total_ms >= 20:
                logger.info(
                    "SGLang UCM V2 load: batch=%s requests=%d total_ms=%.3f "
                    "queue_ms=%.3f event_ms=%.3f prepare_ms=%.3f "
                    "forward_ms=%.3f layers_ms=%.3f "
                    "payload_ms=%.3f payload_cpu_ms=%.3f "
                    "addr_ms=%.3f tolist_ms=%.3f pack_ms=%.3f "
                    "io_queue_ms=%.3f submit_ms=%.3f "
                    "native_pack_ms=%.3f native_submit_ms=%.3f native_timed_jobs=%d "
                    "job_latency_sum_ms=%.3f publish_lag_ms=%.3f "
                    "bytes=%d segments=%d jobs=%d payload_mode=%s",
                    index,
                    len(pending),
                    total_ms,
                    queue_ms,
                    event_ms,
                    prepare_ms,
                    forward_ms,
                    layers_ms,
                    metrics["payload_ms"],
                    metrics["payload_cpu_ms"],
                    metrics["addr_ms"],
                    metrics["tolist_ms"],
                    metrics["pack_ms"],
                    metrics["io_queue_ms"],
                    metrics["submit_ms"],
                    metrics["native_pack_ms"],
                    metrics["native_submit_ms"],
                    metrics["native_timed_jobs"],
                    metrics["job_latency_sum_ms"],
                    metrics["publish_lag_ms"],
                    metrics["bytes"],
                    metrics["segments"],
                    metrics["jobs"],
                    "arrays" if getattr(self.runtime, "supports_get_arrays", False) else "objects",
                )
        return True

    def _apply_forward_dependency(self, context, forward_events):
        if not forward_events:
            return
        forward_done = forward_events[0]
        if self._forward_event_to_store:
            context.prerequisite_handle = int(
                getattr(forward_done, "cuda_event", 0) or 0
            )
        if not context.prerequisite_handle:
            forward_done.synchronize()

    def _load_layers(self, index, prepared, context, metrics):
        layer_jobs = []
        next_complete = 0
        use_arrays = getattr(self.runtime, "supports_get_arrays", False)
        try:
            for layer in range(self.layout.num_layers):
                payload_timings = metrics if self._log_performance else None
                payload_start = perf_counter()
                cpu_start = thread_time() if payload_timings is not None else 0.0
                submissions = []
                for pages, token in prepared:
                    build_payloads = (
                        pages.payload_arrays if use_arrays else pages.payloads
                    )
                    payloads = (
                        build_payloads(layer, timings=payload_timings)
                        if payload_timings is not None
                        else build_payloads(layer)
                    )
                    if payloads:
                        submissions.append((payloads, token))
                if payload_timings is not None:
                    metrics["payload_cpu_ms"] += (thread_time() - cpu_start) * 1e3
                metrics["payload_ms"] += (perf_counter() - payload_start) * 1e3
                # All pages of a section/layer share the same fragment geometry.
                # Count once per submission rather than traversing every page again.
                for payloads, _ in submissions:
                    count = len(payloads[0]) if use_arrays else len(payloads)
                    sizes = payloads[1] if use_arrays else payloads[0].sizes
                    metrics["bytes"] += count * sum(sizes)
                    metrics["segments"] += count * len(sizes)
                jobs = []
                layer_jobs.append(jobs)
                # Wait only for submission. Store I/O continues on the backend.
                self._io_executor.submit(
                    self._submit_load_layer,
                    submissions,
                    context,
                    jobs,
                    metrics,
                    perf_counter(),
                    use_arrays,
                ).result()
                while next_complete <= layer and all(
                    job.done() for job in layer_jobs[next_complete]
                ):
                    self._publish_load_layer(index, next_complete, layer_jobs, metrics)
                    next_complete += 1

            for layer in range(next_complete, len(layer_jobs)):
                self._publish_load_layer(index, layer, layer_jobs, metrics)
        finally:
            # Keep the Get window and the destination pages alive until all
            # accepted jobs finish, including after partial submission failure.
            for jobs in layer_jobs:
                for job in jobs:
                    job.result()

    def _submit_load_layer(
        self, submissions, context, jobs, metrics, queued_at, use_arrays=False
    ):
        metrics["io_queue_ms"] += (perf_counter() - queued_at) * 1e3
        for payloads, token in submissions:
            result = _OffloadResult()
            result.load_metrics = metrics
            result.submitted_at = perf_counter()
            try:
                submit = (
                    self.runtime.submit_get_arrays
                    if use_arrays
                    else self.runtime.submit_get
                )
                native_timings = self._submit_job(submit, result, payloads, token, context)
            finally:
                metrics["submit_ms"] += (perf_counter() - result.submitted_at) * 1e3
            metrics["jobs"] += 1
            jobs.append(result)
            # Older array bindings return None. Count measured jobs so their
            # missing measurements cannot be mistaken for zero native cost.
            if use_arrays and self._log_performance and native_timings is not None:
                native_pack_ms, native_submit_ms = native_timings
                metrics["native_pack_ms"] = metrics.get("native_pack_ms", 0.0) + native_pack_ms
                metrics["native_submit_ms"] = metrics.get("native_submit_ms", 0.0) + native_submit_ms
                metrics["native_timed_jobs"] = metrics.get("native_timed_jobs", 0) + 1

    def _publish_load_layer(self, index, layer, layer_jobs, metrics):
        results = [job.result() for job in layer_jobs[layer]]
        if not all(results):
            raise RuntimeError(f"Store Get failed for layer {layer}")
        if layer_jobs[layer]:
            metrics["publish_lag_ms"] += (
                perf_counter() - max(job.observed_at for job in layer_jobs[layer])
            ) * 1e3
        self.layer_done_counter.complete(index, layer)

    @_log_duration("finish_layer_wise_loading")
    def finish_layer_wise_loading(self, counter_index):
        if counter_index < 0:
            return True
        landed = self._load_batches.pop(counter_index).result()
        return self.layer_done_counter.finish(counter_index) and landed

    def prefetch(self, transfers):
        """Optional linker device-prefetch path: Get all layers into private GPU
        slots and resolve a Future after writes stop. This is distinct from
        Store Prefetch, which promotes objects within the storage hierarchy.
        """
        raise NotImplementedError

    @_log_duration("offload")
    def offload(self, transfers, session_id=None):
        self._check_open()
        expanded = self.layout.expand(transfers, allow_partial=True)
        if not expanded:
            return False
        tail_owner = self._tail_owner(expanded)
        expanded = [
            (section, selected)
            for section, transfers in expanded
            if (selected := self._partition_offload(section, transfers, tail_owner))
        ]
        if not expanded:
            # Every rank acknowledges the node, even when its page slice is empty.
            result = Future()
            result.set_result(True)
            self._offload_results.append(result)
            return True
        events = self._record_transfer_events()
        result = _OffloadResult()
        start = perf_counter()
        result.timings = {}

        def log_completion(future):
            duration_ms = (perf_counter() - start) * 1e3
            if self._log_performance and duration_ms >= 20:
                logger.info(
                    "SGLang UCM V2 offload: session=%s total_ms=%.3f "
                    "queue_ms=%.3f event_ms=%.3f prepare_ms=%.3f "
                    "forward_ms=%.3f submit_to_complete_ms=%.3f success=%s",
                    session_id,
                    duration_ms,
                    result.timings.get("queue_ms", 0.0),
                    result.timings.get("event_ms", 0.0),
                    result.timings.get("prepare_ms", 0.0),
                    result.timings.get("forward_ms", 0.0),
                    (perf_counter() - result.timings["submit_start"]) * 1e3
                    if "submit_start" in result.timings
                    else 0.0,
                    not future.cancelled()
                    and future.exception() is None
                    and bool(future.result()),
                )

        result.add_done_callback(log_completion)
        pages_by_request = {}
        with self._scheduler_lock:
            keys = (key for _, transfers in expanded for key in transfers[0].keys)
            for index, key in enumerate(keys):
                for rid in sorted(self._page_requests.get(key, ())):
                    pages_by_request.setdefault(rid, []).append(index)
            for rid in pages_by_request:
                self._pending_puts[rid] = self._pending_puts.get(rid, 0) + 1
        try:
            self._dump_executor.submit(
                self._submit_offloads,
                expanded,
                events,
                result,
                pages_by_request,
                perf_counter(),
            )
        except Exception:
            with self._scheduler_lock:
                for rid in pages_by_request:
                    self._release_put_request(rid)
            raise
        self._offload_results.append(result)
        return True

    def _tail_owner(self, sections):
        """Pick the rank that keeps trailing-window sections whole.

        Full-attention pages split across ranks, but a trailing window
        (SWA/state) only exists at a node boundary and must stay together, so
        one rank owns all of it. The choice is independent of local device
        addresses: it hashes the node's tail key, preferring a non-empty FA
        tail key (visible identically on every rank) and falling back to the
        trailing section's own tail key.
        """
        if self._dump_size == 1:
            return 0
        fa = next(
            (t[0] for s, t in sections if s.name == "FA" and t and t[0].keys),
            None,
        )
        key_transfer = (
            fa
            if fa is not None
            else next((t[0] for _, t in sections if t and t[0].keys), None)
        )
        if key_transfer is None:
            return 0
        digest = hashlib.sha256(key_transfer.keys[-1].encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % self._dump_size

    def _partition_offload(self, section, transfers, tail_owner):
        if self._dump_size == 1:
            return transfers
        if section.name != "FA":
            # Trailing-window sections are indivisible: keep them whole on the
            # tail owner and drop them everywhere else.
            return transfers if self._dump_rank == tail_owner else ()
        pages = len(transfers[0].keys)
        start = pages * self._dump_rank // self._dump_size
        end = pages * (self._dump_rank + 1) // self._dump_size
        if start == end:
            return ()

        views = {}
        selected = []
        for entry, transfer in zip(section.entries, transfers):
            indices = transfer.host_indices
            view_key = (id(indices), entry.page_size)
            if view_key not in views:
                views[view_key] = indices.reshape(-1)[
                    start * entry.page_size : end * entry.page_size
                ]
            selected.append(
                replace(
                    transfer,
                    keys=transfer.keys[start:end],
                    host_indices=views[view_key],
                )
            )
        return tuple(selected)

    def _submit_offloads(self, sections, events, result, pages_by_request, queued_at=None):
        start = perf_counter()
        timings = getattr(result, "timings", {})
        timings["queue_ms"] = (start - queued_at) * 1e3 if queued_at is not None else 0.0
        try:
            indices_ready, *forward_events = events
            indices_ready.synchronize()
            timings["event_ms"] = (perf_counter() - start) * 1e3
            prepare_start = perf_counter()
            prepared, block_ids, payloads = [], [], []
            for section, transfers in sections:
                pages = section.prepare(transfers)
                prepared.append(pages)
                block_ids.extend(section.block_ids(pages.keys))
                payloads.extend(pages.payloads())
            context = proxy().IoContext()
            timings["prepare_ms"] = (perf_counter() - prepare_start) * 1e3
            forward_start = perf_counter()
            self._apply_forward_dependency(context, forward_events)
            timings["forward_ms"] = (perf_counter() - forward_start) * 1e3
            timings["submit_start"] = perf_counter()
        except Exception:
            logger.exception("UCM V2 offload preparation failed")
            with self._scheduler_lock:
                for rid in pages_by_request:
                    self._release_put_request(rid)
            result.set_result(False)
            return

        covered = {index for indices in pages_by_request.values() for index in indices}
        writes = list(pages_by_request.items())
        unmatched = [index for index in range(len(block_ids)) if index not in covered]
        if unmatched:
            writes.append((None, unmatched))
        remaining = len(writes)
        success = True
        completion_lock = Lock()

        def finish_node(write):
            nonlocal remaining, success
            with completion_lock:
                success &= write.result()
                remaining -= 1
                if remaining == 0:
                    result.set_result(success)

        for rid, indices in writes:
            write = Future()
            write.add_done_callback(finish_node)
            self._submit_offload(
                rid,
                tuple(block_ids[i] for i in indices),
                [payloads[i] for i in indices],
                context,
                (prepared, events),
                write,
                result.reservations if self._coordinated_puts else None,
            )

    def _submit_offload(
        self, rid, block_ids, payloads, context, resources, result, reservations
    ):
        temporary = rid is None
        if temporary:
            rid = f"ucm-offload:{uuid4().hex}"
        try:
            if temporary:
                request = proxy().RequestContext()
                request.request_id = rid
                with self._scheduler_lock:
                    self.runtime.on_new_request(request)
            try:
                with self._scheduler_lock:
                    token = self.runtime.prepare_put(rid, b"".join(block_ids))
                try:
                    if tuple(key.to_bytes() for key in token.keys) != block_ids:
                        raise RuntimeError(
                            "Store Put token does not match the node pages"
                        )
                    completion = Future()
                    completion.add_done_callback(
                        lambda done: self._control_executor.submit(
                            self._finish_offload,
                            rid,
                            token,
                            result,
                            done.result(),
                            temporary,
                            reservations,
                        )
                    )
                    # Keep the pages and events in the in-flight record.
                    self._io_executor.submit(
                        self._submit_job,
                        self.runtime.submit_put,
                        completion,
                        payloads,
                        token,
                        context,
                        resources=resources,
                    ).result()
                except Exception:
                    with self._scheduler_lock:
                        self.runtime.complete_put(rid, token, False)
                    raise
            except Exception:
                with self._scheduler_lock:
                    if temporary:
                        self.runtime.on_request_finished(rid)
                raise
        except Exception:
            logger.exception("UCM V2 offload submission failed for %s", rid)
            if not temporary:
                with self._scheduler_lock:
                    self._release_put_request(rid)
            result.set_result(False)
            return

    def _finish_offload(self, rid, token, result, success, temporary, reservations):
        if reservations is not None:
            reservations.append((rid, token, temporary))
            result.set_result(success)
            return
        result.set_result(self._complete_put(rid, token, temporary, success))

    def _complete_put(self, rid, token, temporary, success):
        try:
            with self._scheduler_lock:
                try:
                    self.runtime.complete_put(rid, token, success)
                finally:
                    if temporary:
                        self.runtime.on_request_finished(rid)
                    else:
                        self._release_put_request(rid)
        except Exception:
            logger.exception("UCM V2 offload cleanup failed for %s", rid)
            return False
        # Commit may wake the topology's drain loop, but its plans do not
        # belong to this node and cannot change the node's Put result.
        self._request_store_offload()
        return success

    def _request_store_offload(self):
        with self._store_offload_lock:
            if self._closing or self._store_offload_queued:
                return
            self._store_offload_queued = True
            self._control_executor.submit(self._run_store_offloads)

    def _run_store_offloads(self):
        with self._store_offload_lock:
            self._store_offload_queued = False
        try:
            self._drive_store_offloads()
        except Exception:
            # A scheduler tick can retry Prepare; never replay an issued plan.
            logger.exception("UCM V2 Store offload drain failed")

    def _drive_store_offloads(self):
        """Run on the control executor; allow only one minted batch at a time."""
        if self._store_offloads:
            return tuple(self._store_offloads)
        if self._store_offload_error is not None:
            raise RuntimeError(
                "UCM V2 Store offload receipt failed"
            ) from self._store_offload_error
        with self._scheduler_lock:
            plans = self.runtime.prepare_offload()
        if not plans:
            return None
        for plan in plans:
            completion = Future()
            receipt = Future()
            self._store_offloads.add(receipt)
            completion.add_done_callback(
                lambda done, plan=plan, receipt=receipt: self._control_executor.submit(
                    self._finish_store_offload, plan, done.result(), receipt
                )
            )
            self._io_executor.submit(self._submit_store_offload, plan, completion)
        return tuple(self._store_offloads)

    def _submit_store_offload(self, plan, completion):
        try:
            self._submit_job(self.runtime.submit_offload, completion, plan)
        except Exception:
            logger.exception("UCM V2 Store offload submission failed")
            completion.set_result(False)

    def _finish_store_offload(self, plan, success, receipt):
        if not success:
            logger.error("UCM V2 Store offload failed; plan will not be replayed")
        try:
            with self._scheduler_lock:
                self.runtime.complete_offload(plan)
        except Exception as error:
            # Do not destroy the Store as if its outstanding pins were released.
            self._store_offload_error = error
            logger.exception("UCM V2 Store offload receipt failed")
        finally:
            self._store_offloads.remove(receipt)
            receipt.set_result(None)
        if not self._store_offloads and self._store_offload_error is None:
            self._request_store_offload()

    def _flush_store_offloads(self):
        # Producers must be quiescent. Keep the executor alive for polling and
        # receipts until both accepted jobs and committed pending plans drain.
        while True:
            pending = self._control_executor.submit(self._drive_store_offloads).result()
            if pending is None:
                return
            for completion in pending:
                completion.result()

    def _submit_job(self, submit, completion, *args, resources=None):
        job_id = next(self._job_ids)
        submission_result = submit(job_id, *args)
        self._jobs[job_id] = (completion, resources)
        self._schedule_poll()
        return submission_result

    def _schedule_poll(self):
        if not self._poll_queued:
            self._poll_queued = True
            self._io_executor.submit(self._poll_jobs)

    def _poll_jobs(self):
        self._poll_queued = False
        if not self._jobs:
            return
        try:
            results = self.runtime.get_finished()
            for result in results:
                completion, resources = self._jobs.pop(result.job_id)
                self._measure_load_completion(completion)
                completion.set_result(result.success)
            if not results:
                sleep(0.001)
        except Exception:
            logger.exception("UCM V2 completion polling failed; draining accepted jobs")
            for job_id in tuple(self._jobs):
                try:
                    self.runtime.wait(job_id)
                    success = True
                except Exception:
                    success = False
                completion, resources = self._jobs.pop(job_id)
                self._measure_load_completion(completion)
                completion.set_result(success)
        if self._jobs:
            self._schedule_poll()

    @staticmethod
    def _measure_load_completion(completion):
        if not hasattr(completion, "load_metrics"):
            return
        completion.observed_at = perf_counter()
        # Includes native submission, transfer and polling delay. This is NOT
        # a CUDA-event measurement; overlapping job latencies are not additive
        # with the batch wall time.
        completion.load_metrics["job_latency_sum_ms"] += (
            completion.observed_at - completion.submitted_at
        ) * 1e3

    def num_completed_offloads(self):
        self._request_store_offload()
        completed = 0
        for result in self._offload_results:
            if not result.done():
                break
            completed += 1
        return completed

    def pop_completed_offload(self):
        result = self._offload_results.popleft()
        success = result.result()
        if self._coordinated_puts:
            self._uncommitted_offloads.append(result)
        return success

    def commit_completed_puts(self, successes):
        results = []
        for success in successes:
            result = self._uncommitted_offloads.popleft()
            committed = success
            for rid, token, temporary in result.reservations:
                committed &= self._complete_put(rid, token, temporary, success)
            results.append(committed)
        return results

    def reset(self):
        for index in tuple(self._load_batches):
            self.finish_layer_wise_loading(index)
        while self._offload_results:
            self.pop_completed_offload()
        # Reset has no group verdict. Abort unpublished reservations.
        self.commit_completed_puts([False] * len(self._uncommitted_offloads))
        self._pending_loads.clear()
        self.layer_done_counter.reset()
        for rid in tuple(self._registered_requests):
            self.finish_request(rid)
        self._flush_store_offloads()

    def _check_open(self):
        if self._closing:
            raise RuntimeError("UCM V2 linker is closing")

    def close(self):
        if self._closed:
            return
        with self._store_offload_lock:
            self._closing = True
        self.reset()
        self._load_executor.shutdown()
        self._dump_executor.shutdown()
        self._io_executor.shutdown()
        self._control_executor.shutdown()
        self._closed = True
