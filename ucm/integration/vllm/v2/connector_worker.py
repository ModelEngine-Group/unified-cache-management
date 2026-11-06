"""Worker KV registration, transfer construction and IO lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)

from ucm.integration.vllm.device import create_device
from ucm.logger import init_logger

from .connector_scheduler import RequestMetadata, UCMConnectorMetadata, UCMDispatchPlan
from .kv_cache_layout import UCMKVCacheLayout
from .runtime import ParallelRank, UCMStorageHandshake
from .store_client import (
    PreparedJob,
    StoreIOError,
    StoreWorkerClient,
    TransferBatch,
    TransferTask,
)

if TYPE_CHECKING:
    import torch
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from .runtime import UCMRuntimeContext

logger = init_logger(__name__)


@dataclass(frozen=True)
class JobResult:
    """This rank's terminal result for a Scheduler-assigned parent job."""

    job_id: int
    rank: ParallelRank
    success: bool


@dataclass
class UCMWorkerMetadata(KVConnectorWorkerMetadata):
    failed_load_request_ids: set[str] = field(default_factory=set)
    job_results: set[JobResult] = field(default_factory=set)

    def mark_load_failed(self, request_id: str) -> None:
        self.failed_load_request_ids.add(request_id)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "UCMWorkerMetadata":
        self.failed_load_request_ids.update(other.failed_load_request_ids)
        self.job_results.update(other.job_results)
        return self


class UCMConnectorWorker:
    """Own local KV registration and all per-step transfer lifetimes."""

    def __init__(
        self,
        context: "UCMRuntimeContext",
        kv_cache_config: KVCacheConfig,
        *,
        num_hidden_layers: int,
    ) -> None:
        self.context = context
        self.spec = context.spec
        self.parallel_topology = context.parallel_topology
        self.use_layerwise = context.use_layerwise
        self._kv_cache_config = kv_cache_config
        self._num_hidden_layers = num_hidden_layers
        if context.rank is None:
            raise ValueError("Worker context requires parallel rank coordinates")
        self._parallel_rank = context.rank
        self._store_worker = StoreWorkerClient(context.launch_config)
        self._device = create_device()
        self._layer_load_tasks: dict[
            int | None, list[tuple[RequestMetadata, TransferTask]]
        ] = {}
        self._dump_tasks: list[TransferTask] = []
        self._dump_submitted_layer_ids: set[int] = set()
        self._failed_load_request_ids: set[str] = set()
        self._failed_dump_jobs: set[int] = set()
        # Keep completed tasks here until the step's parent results are reported.
        self._transfer_tasks_by_job: dict[int, list[TransferTask]] = {}
        self.layout: UCMKVCacheLayout | None = None
        self._completion_metadata = UCMWorkerMetadata()
        self._invalid_vllm_block_ids = set()

    def check_io_drained(self) -> None:
        if self._layer_load_tasks or self._dump_tasks:
            raise RuntimeError("UCM transfers still own KV pointers")

    def _record_transfer_event(self) -> int:
        if self._device is None:
            return 0
        event_handle = self._device.get_event_handle()
        if not event_handle:
            # A failed event recording must not drop the stream dependency.
            self._device.synchronize()
        return event_handle

    def _release_transfer_event(self, task: TransferTask) -> None:
        if task.success is not None and task.prerequisite_handle:
            self._device.destroy_event_handle(task.prerequisite_handle)
            task.prerequisite_handle = 0

    def _wait_transfer(self, task: TransferTask) -> None:
        try:
            self._store_worker.wait(task)
        finally:
            # Unknown IO state still owns the event, even if wait raised.
            self._release_transfer_event(task)

    def shutdown(self) -> None:
        first_error = None
        for layer_tasks in self._layer_load_tasks.values():
            for _, task in layer_tasks:
                try:
                    self._wait_transfer(task)
                except Exception as exc:
                    first_error = first_error or exc
        for task in self._dump_tasks:
            try:
                self._wait_transfer(task)
            except Exception as exc:
                first_error = first_error or exc
        try:
            self._store_worker.close()
        except Exception as exc:
            first_error = first_error or exc
        else:
            # close drains the backend; keep descriptors if it cannot finish.
            self._layer_load_tasks.clear()
            self._dump_tasks.clear()
        finally:
            for job_tasks in self._transfer_tasks_by_job.values():
                for task in job_tasks:
                    self._release_transfer_event(task)
        if first_error is None:
            self._transfer_tasks_by_job.clear()
        if first_error is not None:
            raise first_error

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        self.layout = UCMKVCacheLayout(
            self.spec,
            kv_caches,
            kv_cache_config=self._kv_cache_config,
            num_hidden_layers=self._num_hidden_layers,
        )
        self.layout.build_store_layouts(self.parallel_topology, self._parallel_rank)

    def get_handshake_metadata(self) -> UCMStorageHandshake | None:
        # PCP ranks share a stage's policies. Report one representative per TP rank
        # so CUDA's (pp_rank, tp_rank) handshake keys remain unique.
        if self._parallel_rank.pcp_rank != 0:
            return None
        if self.layout is None:
            raise RuntimeError("Register KV caches before the storage handshake")
        return UCMStorageHandshake(
            rank=self._parallel_rank,
            store_sizes_by_policy=dict(self.layout.store_sizes_by_policy),
        )

    def _mark_load_failed(self, request_metadata: RequestMetadata) -> None:
        request_id = request_metadata.request_id
        self._failed_load_request_ids.add(request_id)
        self._completion_metadata.mark_load_failed(request_id)
        # Match the existing load_failure_patch's group 0 block table.
        self._invalid_vllm_block_ids.update(
            int(block_id)
            for load_plan in request_metadata.load_plans
            for group_id, group_block_ids in chain(
                zip(load_plan.group_ids, load_plan.vllm_block_ids_by_group),
                zip(load_plan.group_ids, load_plan.state_checkpoint_block_ids_by_group),
            )
            if group_id == 0
            for block_id in group_block_ids
            if block_id != 0
        )

    def begin_step(self, connector_metadata: UCMConnectorMetadata) -> None:
        # No work from a previous step may outlive its KV block ownership.
        # Binding always precedes forward; start_load_kv may run afterwards
        # in 0.30 when this step has no synchronous loads.
        self.check_io_drained()
        self._dump_submitted_layer_ids.clear()
        self._failed_load_request_ids.clear()
        self._failed_dump_jobs.clear()
        self._transfer_tasks_by_job.clear()
        for request_metadata in connector_metadata.requests.values():
            if request_metadata.prepare_get_failed:
                self._mark_load_failed(request_metadata)

    def _report_job_result(self, job_id: int, success: bool) -> None:
        job_tasks = self._transfer_tasks_by_job.get(job_id, ())
        self._completion_metadata.job_results.add(
            JobResult(
                job_id,
                self._parallel_rank,
                success and all(task.success is True for task in job_tasks),
            )
        )

    def _execute_prefetch(self, prefetch_job: PreparedJob) -> None:
        success = True
        # All workers in this DP replica share shm, so rank (0, 0, 0) alone
        # submits the prefetch. Other ranks acknowledge the no-op so the
        # scheduler can collect a result from every rank for this job.
        if self._parallel_rank == ParallelRank(tp_rank=0, pp_rank=0, pcp_rank=0):
            try:
                prefetch_task = self._store_worker.submit_prefetch(prefetch_job)
                self._transfer_tasks_by_job[prefetch_job.job_id] = [prefetch_task]
                self._store_worker.wait(prefetch_task)
            except StoreIOError:
                logger.warning(
                    "UCM prefetch failed; recomputing this request", exc_info=True
                )
                success = False
        self._report_job_result(prefetch_job.job_id, success)

    def _submit_load(
        self,
        request_metadata: RequestMetadata,
        load_plan: UCMDispatchPlan,
        layer_id: int | None = None,
    ) -> None:
        transfer_batches = self._build_transfer_batches(
            load_plan, is_dump=False, layer_id=layer_id
        )
        if not transfer_batches:
            return
        load_job = load_plan.prepared_job
        # MRV2 may still be zeroing blocks or copying Mamba state on the
        # compute stream. Get must wait before writing those device ranges.
        event_handle = self._record_transfer_event()
        try:
            load_task = self._store_worker.submit_get(
                load_job,
                transfer_batches,
                is_layerwise=layer_id is not None,
                prerequisite_handle=event_handle,
            )
        except Exception:
            # A rejected submission starts no IO and has no event consumer.
            if event_handle:
                self._device.destroy_event_handle(event_handle)
            raise
        self._transfer_tasks_by_job.setdefault(load_job.job_id, []).append(load_task)
        self._layer_load_tasks.setdefault(layer_id, []).append(
            (request_metadata, load_task)
        )
        if layer_id is None:
            self._wait_layer_load(None)

    def start_load_kv(
        self,
        connector_metadata: UCMConnectorMetadata,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        for request_metadata in connector_metadata.requests.values():
            if request_metadata.prefetch_job is not None:
                self._execute_prefetch(request_metadata.prefetch_job)
            if request_metadata.request_id in self._failed_load_request_ids:
                continue
            try:
                for load_plan in request_metadata.load_plans:
                    if self.use_layerwise:
                        for layer_id in self.layout.layer_ids:
                            self._submit_load(request_metadata, load_plan, layer_id)
                    else:
                        self._submit_load(request_metadata, load_plan)
            except StoreIOError:
                self._mark_load_failed(request_metadata)
        if self.use_layerwise:
            state_layer_ids = {
                layer_spec.layer_index
                for state_group in self.spec.state_groups
                if state_group.group_id in self.layout.group_layouts
                for layer_spec in self.layout.group_layouts[
                    state_group.group_id
                ].layer_specs
            }
            for layer_id in state_layer_ids:
                self._wait_layer_load(layer_id)

    def _wait_layer_load(self, layer_id: int | None) -> None:
        unfinished_tasks = []
        fatal_error = None
        for request_metadata, task in self._layer_load_tasks.get(layer_id, ()):
            try:
                self._wait_transfer(task)
            except StoreIOError:
                self._mark_load_failed(request_metadata)
            except Exception as exc:
                # Unknown completion state: retain the task/descriptors and fail.
                unfinished_tasks.append((request_metadata, task))
                fatal_error = fatal_error or exc
        if unfinished_tasks:
            self._layer_load_tasks[layer_id] = unfinished_tasks
        else:
            self._layer_load_tasks.pop(layer_id, None)
        if fatal_error is not None:
            raise fatal_error

    def _drain_dump_tasks(self) -> None:
        unfinished_tasks = []
        fatal_error = None
        for task in self._dump_tasks:
            try:
                self._wait_transfer(task)
            except StoreIOError:
                self._failed_dump_jobs.add(task.job_id)
            except Exception as exc:
                unfinished_tasks.append(task)
                fatal_error = fatal_error or exc
        self._dump_tasks = unfinished_tasks
        if fatal_error is not None:
            raise fatal_error

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise:
            return
        assert self.layout is not None
        layer_id = self.layout.layer_id_by_name.get(layer_name)
        if layer_id is not None:
            # An attention hook also waits for indexer caches of the same layer.
            self._wait_layer_load(layer_id)

    def _dump_layer(
        self, connector_metadata: UCMConnectorMetadata, layer_id: int
    ) -> None:
        # MTP/speculative decoding can revisit a layer in this step. Only
        # the first visit submits the Scheduler's dump plans for that layer.
        if layer_id in self._dump_submitted_layer_ids:
            return
        for request_metadata in connector_metadata.requests.values():
            if request_metadata.request_id in self._failed_load_request_ids:
                continue
            for dump_plan in request_metadata.dump_plans:
                dump_job = dump_plan.prepared_job
                if dump_job is None or dump_job.job_id in self._failed_dump_jobs:
                    continue
                transfer_batches = self._build_transfer_batches(
                    dump_plan, is_dump=True, layer_id=layer_id
                )
                if not transfer_batches:
                    continue
                try:
                    self._submit_dump(dump_job, transfer_batches, is_layerwise=True)
                except StoreIOError:
                    self._failed_dump_jobs.add(dump_job.job_id)
        self._dump_submitted_layer_ids.add(layer_id)

    def _submit_dump(
        self,
        dump_job: PreparedJob,
        transfer_batches: tuple[TransferBatch, ...],
        *,
        is_layerwise: bool,
    ) -> None:
        # Record after the compute stream has submitted the KV/state writes.
        # Store waits for this event before reading the dump source.
        event_handle = self._record_transfer_event()
        try:
            dump_task = self._store_worker.submit_put(
                dump_job,
                transfer_batches,
                is_layerwise=is_layerwise,
                prerequisite_handle=event_handle,
            )
        except Exception:
            if event_handle:
                self._device.destroy_event_handle(event_handle)
            raise
        self._transfer_tasks_by_job.setdefault(dump_job.job_id, []).append(dump_task)
        self._dump_tasks.append(dump_task)

    def save_kv_layer(
        self,
        connector_metadata: UCMConnectorMetadata,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if not self.use_layerwise:
            return
        layer_id = self.layout.layer_id_by_name.get(layer_name)
        if layer_id is not None:
            self._dump_layer(connector_metadata, layer_id)

    def wait_for_save(self, connector_metadata: UCMConnectorMetadata) -> None:
        for layer_id in tuple(self._layer_load_tasks):
            self._wait_layer_load(layer_id)
        if self.use_layerwise:
            try:
                for layer_id in self.layout.layer_ids:
                    self._dump_layer(connector_metadata, layer_id)
            finally:
                self._drain_dump_tasks()
        else:
            for request_metadata in connector_metadata.requests.values():
                if request_metadata.request_id in self._failed_load_request_ids:
                    continue
                for dump_plan in request_metadata.dump_plans:
                    dump_job = dump_plan.prepared_job
                    if dump_job is None:
                        continue
                    transfer_batches = self._build_transfer_batches(
                        dump_plan, is_dump=True
                    )
                    if not transfer_batches:
                        continue
                    try:
                        # One bulk submission for this plan's Prepare on this rank.
                        self._submit_dump(
                            dump_job, transfer_batches, is_layerwise=False
                        )
                    except StoreIOError:
                        self._failed_dump_jobs.add(dump_job.job_id)
                        logger.warning(
                            "UCM dump submission failed; skipping this job",
                            exc_info=True,
                        )
                self._drain_dump_tasks()
        for request_metadata in connector_metadata.requests.values():
            load_succeeded = (
                request_metadata.request_id not in self._failed_load_request_ids
            )
            # Empty ranks acknowledge each prepared job after local IO drains.
            for dispatch_plan in (
                request_metadata.load_plans + request_metadata.dump_plans
            ):
                prepared_job = dispatch_plan.prepared_job
                if prepared_job is None:
                    continue
                self._report_job_result(
                    prepared_job.job_id,
                    load_succeeded
                    and prepared_job.job_id not in self._failed_dump_jobs,
                )

    def poll_finished(self) -> None:
        # wait() and get_finished() consume the same native completion queue.
        # StoreWorkerClient caches consumed terminal status, so a later layer wait
        # never waits for a second notification of that job.
        self._store_worker.poll_finished()
        for job_tasks in self._transfer_tasks_by_job.values():
            for task in job_tasks:
                self._release_transfer_event(task)

    def build_connector_worker_meta(self) -> UCMWorkerMetadata | None:
        if (
            not self._completion_metadata.failed_load_request_ids
            and not self._completion_metadata.job_results
        ):
            return None
        worker_metadata = self._completion_metadata
        self._completion_metadata = UCMWorkerMetadata()
        return worker_metadata

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_vllm_block_ids = self._invalid_vllm_block_ids
        self._invalid_vllm_block_ids = set()
        return invalid_vllm_block_ids

    def _build_transfer_batches(
        self,
        dispatch_plan: UCMDispatchPlan,
        *,
        is_dump: bool,
        layer_id: int | None = None,
    ) -> tuple[TransferBatch, ...]:
        """Resolve one prepared plan into this rank's device-to-store ranges."""
        # Scheduler emits keys only for stages whose handshake declared this
        # policy; that declaration comes from this stage's store layouts.
        stage_ucm_keys = dispatch_plan.ucm_keys_by_pp_rank.get(
            self._parallel_rank.pp_rank
        )
        if stage_ucm_keys is None:
            return ()
        transfer_batches = []
        chunk_indices = dispatch_plan.token_starts // self.spec.chunk_size
        # A State plan restores the same source ranges to running and checkpoint
        # targets in one Submit. Both targets stay under this plan's single job.
        for group_id, group_block_ids in chain(
            zip(dispatch_plan.group_ids, dispatch_plan.vllm_block_ids_by_group),
            zip(
                dispatch_plan.group_ids,
                dispatch_plan.state_checkpoint_block_ids_by_group,
            ),
        ):
            # PP stages may have no layers in a participating native group.
            store_layout = self.layout.store_layouts.get(group_id)
            if store_layout is None:
                continue
            segment_mask = store_layout.group_layout.segment_mask(layer_id)
            if segment_mask is not None and not segment_mask.any():
                continue
            selected_chunk_rows = (
                store_layout.select_writer_chunk_rows(chunk_indices)
                if is_dump
                else np.arange(len(chunk_indices))
            )
            if not len(selected_chunk_rows):
                continue
            selected_block_ids = group_block_ids.reshape(len(chunk_indices), -1)[
                selected_chunk_rows
            ].reshape(-1)
            store_offsets, device_ptrs, byte_sizes = (
                store_layout.resolve_transfer_ranges(
                    selected_block_ids,
                    dispatch_plan.token_starts[selected_chunk_rows],
                    segment_mask=segment_mask,
                )
            )
            store_offsets = np.broadcast_to(
                store_offsets[0]
                + np.uint64(self.layout.store_offsets_by_group[group_id]),
                store_offsets.shape,
            )
            transfer_batches.append(
                TransferBatch(
                    ucm_keys=tuple(
                        stage_ucm_keys[int(row)] for row in selected_chunk_rows
                    ),
                    device_ptrs=device_ptrs,
                    byte_sizes=byte_sizes,
                    store_offsets=store_offsets,
                )
            )
        return tuple(transfer_batches)
