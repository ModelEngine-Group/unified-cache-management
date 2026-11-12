"""SGLang direct linker for Store V2.

Lookup and layerwise Get share one Store request registration.
The first target is GLM-5.2 DSA (KV + indexer in one FA object per logical page).
"""

import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from itertools import count
from threading import Lock
from time import sleep
from uuid import uuid4

import torch.distributed as dist
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
)
from sglang.srt.runtime_context import get_model

from .device_layout import DsaLayout
from .runtime import Runtime, proxy


logger = logging.getLogger(__name__)


class UcmDirectLinker(UnifiedCacheLinker):
    load_mode = "layer_wise"
    prefetch_policy = "timeout"

    def __init__(
        self, server_args, params, *, components, extra_config=None, forward_stream=None
    ):
        self.forward_stream = forward_stream
        self.page_size = params.page_size
        pool_group = resolve_hybrid_device_pool_group(
            kvcache=params.token_to_kv_pool_allocator.get_kvcache(),
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.layout = DsaLayout(pool_group, "sglang", get_model().model_path)
        self._dump_rank, self._dump_size = 0, 1
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if pool_group.rank_replicated and tp_group is not None and dist.is_initialized():
            self._dump_rank = dist.get_rank(tp_group)
            self._dump_size = dist.get_world_size(tp_group)
        extra_config = extra_config or {}
        # The assembler has no capability query for prerequisite events.
        self._forward_event_to_store = extra_config.get("forward_event_to_store", False)
        self.runtime = Runtime(extra_config)
        self._registered_requests = set()
        self._scheduler_lock = Lock()
        self.layer_done_counter = LayerWiseLoadCounter(self.layout.num_layers)
        self._pending_loads = {}
        self._load_batches = {}
        self._offload_results = deque()
        self._jobs = {}
        self._poll_queued = False
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-v2-load"
        )
        self._dump_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-v2-dump"
        )
        self._job_ids = count(1)
        self._io_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-v2-io"
        )

    def lookup(self, rid, transfers):
        """Return restorable boundaries for joint KV/indexer objects."""
        expanded = self.layout.expand(transfers)
        if not expanded:
            return []
        keys = b"".join(self.layout.block_ids(expanded[0].keys))
        with self._scheduler_lock:
            if rid not in self._registered_requests:
                context = proxy().RequestContext()
                context.request_id = rid
                self.runtime.on_new_request(context)
                self._registered_requests.add(rid)
            hits, _prefetchable = self.runtime.lookup(rid, keys)

        prefix_pages = next((i for i, hit in enumerate(hits) if not hit), len(hits))
        return list(range(1, prefix_pages + 1))

    def finish_request(self, rid):
        """Release the route after the caller is done with this Store request."""
        with self._scheduler_lock:
            if rid in self._registered_requests:
                self.runtime.on_request_finished(rid)
                self._registered_requests.remove(rid)

    def load(self, rid, transfers):
        expanded = self.layout.expand(transfers)
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

    def start_layer_wise_loading(self):
        if not self._pending_loads:
            return -1
        ready = self.record_transfer_ready()
        index = self.layer_done_counter.update_producer()
        pending, self._pending_loads = self._pending_loads, {}
        self._load_batches[index] = self._load_executor.submit(
            self._load_worker, index, pending, ready
        )
        return index

    def _complete_get(self, rid):
        with self._scheduler_lock:
            self.runtime.complete_get(rid)

    def _load_worker(self, index, pending, ready):
        try:
            with ExitStack() as requests:
                for rid in pending:
                    requests.callback(self.finish_request, rid)
                indices_ready, *forward_events = ready.events
                indices_ready.synchronize()
                context = proxy().IoContext()
                prepared = []
                for rid, transfers in pending.items():
                    pages = self.layout.prepare(transfers)
                    block_ids = self.layout.block_ids(pages.keys)
                    with self._scheduler_lock:
                        token = self.runtime.prepare_get(rid, b"".join(block_ids))
                    requests.callback(self._complete_get, rid)
                    if tuple(key.to_bytes() for key in token.keys) != block_ids:
                        raise RuntimeError(
                            f"Store Get token does not match the requested pages for {rid}"
                        )
                    prepared.append((pages, token))

                self._apply_forward_dependency(context, forward_events)
                self._load_layers(index, prepared, context)
        except Exception as error:
            logger.exception("UCM V2 load batch %s failed", index)
            self.layer_done_counter.fail(index, error)
            return False
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

    def _load_layers(self, index, prepared, context):
        layer_jobs = []
        next_complete = 0
        try:
            for layer in range(self.layout.num_layers):
                submissions = [
                    (pages.payloads(layer), token) for pages, token in prepared
                ]
                jobs = []
                layer_jobs.append(jobs)
                # Wait only for submission. Store I/O continues on the backend.
                self._io_executor.submit(
                    self._submit_load_layer, submissions, context, jobs
                ).result()
                while next_complete <= layer and all(
                    job.done() for job in layer_jobs[next_complete]
                ):
                    self._publish_load_layer(index, next_complete, layer_jobs)
                    next_complete += 1

            for layer in range(next_complete, len(layer_jobs)):
                self._publish_load_layer(index, layer, layer_jobs)
        finally:
            # Keep the Get window and the destination pages alive until all
            # accepted jobs finish, including after partial submission failure.
            for jobs in layer_jobs:
                for job in jobs:
                    job.result()

    def _submit_load_layer(self, submissions, context, jobs):
        for payloads, token in submissions:
            result = Future()
            self._submit_job(self.runtime.submit_get, payloads, token, context, result)
            jobs.append(result)

    def _publish_load_layer(self, index, layer, layer_jobs):
        results = [job.result() for job in layer_jobs[layer]]
        if not all(results):
            raise RuntimeError(f"Store Get failed for layer {layer}")
        self.layer_done_counter.complete(index, layer)

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

    def offload(self, transfers, session_id=None):
        expanded = self.layout.expand(transfers)
        if not expanded:
            return False
        expanded = self._partition_offload(expanded)
        if not expanded:
            # Every rank acknowledges the node, even when its page slice is empty.
            result = Future()
            result.set_result(True)
            self._offload_results.append(result)
            return True
        ready = self.record_transfer_ready()
        result = Future()
        self._dump_executor.submit(self._submit_offload, expanded, ready, result)
        self._offload_results.append(result)
        return True

    def _partition_offload(self, transfers):
        if self._dump_size == 1:
            return transfers
        pages = len(transfers[0].keys)
        start = pages * self._dump_rank // self._dump_size
        end = pages * (self._dump_rank + 1) // self._dump_size
        if start == end:
            return ()

        views = {}
        selected = []
        for entry, transfer in zip(self.layout.entries, transfers):
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

    def _submit_offload(self, transfers, ready, result):
        rid = f"ucm-offload:{uuid4().hex}"
        try:
            indices_ready, *forward_events = ready.events
            indices_ready.synchronize()
            pages = self.layout.prepare(transfers)
            block_ids = self.layout.block_ids(pages.keys)
            payloads = pages.payloads()
            context = proxy().IoContext()
            self._apply_forward_dependency(context, forward_events)

            request = proxy().RequestContext()
            request.request_id = rid
            with self._scheduler_lock:
                self.runtime.on_new_request(request)
            try:
                with self._scheduler_lock:
                    token = self.runtime.prepare_put(rid, b"".join(block_ids))
                try:
                    if tuple(key.to_bytes() for key in token.keys) != block_ids:
                        raise RuntimeError("Store Put token does not match the node pages")
                    completion = Future()
                    completion.add_done_callback(
                        lambda done: self._finish_offload(rid, result, done.result())
                    )
                    # Keep the pages and events in the in-flight record.
                    self._io_executor.submit(
                        self._submit_job,
                        self.runtime.submit_put,
                        payloads,
                        token,
                        context,
                        completion,
                        (pages, ready),
                    ).result()
                except Exception:
                    with self._scheduler_lock:
                        self.runtime.complete_put(rid, False)
                    raise
            except Exception:
                with self._scheduler_lock:
                    self.runtime.on_request_finished(rid)
                raise
        except Exception:
            logger.exception("UCM V2 offload submission failed for %s", rid)
            result.set_result(False)
            return

    def _finish_offload(self, rid, result, success):
        try:
            with self._scheduler_lock:
                try:
                    self.runtime.complete_put(rid, success)
                finally:
                    self.runtime.on_request_finished(rid)
        except Exception:
            logger.exception("UCM V2 offload cleanup failed for %s", rid)
            success = False
        result.set_result(success)

    def _submit_job(self, submit, payloads, token, context, completion, resources=None):
        job_id = next(self._job_ids)
        submit(job_id, payloads, token, context)
        self._jobs[job_id] = (completion, resources)
        self._schedule_poll()

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
                completion.set_result(success)
        if self._jobs:
            self._schedule_poll()

    def num_completed_offloads(self):
        completed = 0
        for result in self._offload_results:
            if not result.done():
                break
            completed += 1
        return completed

    def pop_completed_offload(self):
        return self._offload_results.popleft().result()

    def reset(self):
        for index in tuple(self._load_batches):
            self.finish_layer_wise_loading(index)
        while self._offload_results:
            self.pop_completed_offload()
        self._pending_loads.clear()
        self.layer_done_counter.reset()
        for rid in tuple(self._registered_requests):
            self.finish_request(rid)

    def close(self):
        self.reset()
        self._load_executor.shutdown()
        self._dump_executor.shutdown()
        self._io_executor.shutdown()
