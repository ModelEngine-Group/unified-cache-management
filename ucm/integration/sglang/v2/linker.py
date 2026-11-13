"""SGLang direct linker for Store V2.

Lookup, layerwise Get and Put for known pages share a Store request route.
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
from .request_hooks import install_request_hooks
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
        self.runtime = Runtime(extra_config, dump_partitions=self._dump_size)
        install_request_hooks(type(self))
        self._registered_requests = set()
        self._request_pages = {}
        self._page_requests = {}
        self._pending_puts = {}
        self._finished_requests = set()
        self._scheduler_lock = Lock()
        self.layer_done_counter = LayerWiseLoadCounter(self.layout.num_layers)
        self._pending_loads = {}
        self._load_batches = {}
        self._offload_results = deque()
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
            max_workers=1, thread_name_prefix="ucm-v2-io"
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ucm-v2-control"
        )

    def lookup(self, rid, transfers):
        """Return restorable boundaries for joint KV/indexer objects."""
        self._check_open()
        expanded = self.layout.expand(transfers)
        if not expanded:
            return []
        page_keys = expanded[0].keys
        keys = b"".join(self.layout.block_ids(page_keys))
        with self._scheduler_lock:
            self._associate_pages(rid, page_keys)
            hits, _prefetchable = self.runtime.lookup(rid, keys)

        prefix_pages = next((i for i, hit in enumerate(hits) if not hit), len(hits))
        return list(range(1, prefix_pages + 1))

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

    def load(self, rid, transfers):
        self._check_open()
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

    def _complete_get(self, rid, token):
        with self._scheduler_lock:
            self.runtime.complete_get(rid, token)

    def _load_worker(self, index, pending, ready):
        try:
            with ExitStack() as requests:
                indices_ready, *forward_events = ready.events
                indices_ready.synchronize()
                context = proxy().IoContext()
                prepared = []
                for rid, transfers in pending.items():
                    pages = self.layout.prepare(transfers)
                    block_ids = self.layout.block_ids(pages.keys)
                    with self._scheduler_lock:
                        token = self.runtime.prepare_get(rid, b"".join(block_ids))
                    requests.callback(self._complete_get, rid, token)
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
            self._submit_job(self.runtime.submit_get, result, payloads, token, context)
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
        self._check_open()
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
        pages_by_request = {}
        with self._scheduler_lock:
            for index, key in enumerate(expanded[0].keys):
                for rid in sorted(self._page_requests.get(key, ())):
                    pages_by_request.setdefault(rid, []).append(index)
            for rid in pages_by_request:
                self._pending_puts[rid] = self._pending_puts.get(rid, 0) + 1
        try:
            self._dump_executor.submit(
                self._submit_offloads, expanded, ready, result, pages_by_request
            )
        except Exception:
            with self._scheduler_lock:
                for rid in pages_by_request:
                    self._release_put_request(rid)
            raise
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

    def _submit_offloads(self, transfers, ready, result, pages_by_request):
        try:
            indices_ready, *forward_events = ready.events
            indices_ready.synchronize()
            pages = self.layout.prepare(transfers)
            block_ids = self.layout.block_ids(pages.keys)
            payloads = pages.payloads()
            context = proxy().IoContext()
            self._apply_forward_dependency(context, forward_events)
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
                (pages, ready),
                write,
            )

    def _submit_offload(self, rid, block_ids, payloads, context, resources, result):
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
                        raise RuntimeError("Store Put token does not match the node pages")
                    completion = Future()
                    completion.add_done_callback(
                        lambda done: self._control_executor.submit(
                            self._finish_offload,
                            rid,
                            token,
                            result,
                            done.result(),
                            temporary,
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

    def _finish_offload(self, rid, token, result, success, temporary):
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
            result.set_result(False)
            return
        result.set_result(success)
        # Commit may wake the topology's drain loop, but its plans do not
        # belong to this node and cannot change the node's Put result.
        self._request_store_offload()

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
        submit(job_id, *args)
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
        self._request_store_offload()
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
