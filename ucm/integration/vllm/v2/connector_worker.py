"""Worker KV registration, transfer construction and IO lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

from ucm.logger import init_logger

from .connector_scheduler import RequestMetadata, UCMConnectorMetadata
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
        self.fingerprint = context.fingerprint
        self._kv_cache_config = kv_cache_config
        self._num_hidden_layers = num_hidden_layers
        if context.rank is None:
            raise ValueError("Worker context requires parallel rank coordinates")
        self._parallel_rank = context.rank
        self._store_worker = StoreWorkerClient(context.launch_config)
        self._layer_load_tasks: dict[int | None, list[tuple[str, TransferTask]]] = {}
        self._dump_tasks: list[TransferTask] = []
        self._dump_submitted_layer_ids: set[int] = set()
        self._failed_load_request_ids: set[str] = set()
        self._failed_dump_jobs: set[int] = set()
        self._sub_job_ids_by_job: dict[int, int] = {}
        self._step_io_completed = False
        self.layout: UCMKVCacheLayout | None = None
        self._completion_metadata = UCMWorkerMetadata()
        self._invalid_vllm_block_ids = set()
        self._connector_metadata = None

    def has_connector_metadata(self) -> bool:
        return self._connector_metadata is not None

    def _get_connector_metadata(self) -> UCMConnectorMetadata:
        if self._connector_metadata is None:
            raise RuntimeError("No bound UCM metadata")
        return self._connector_metadata

    def clear_connector_metadata(self) -> None:
        if self._layer_load_tasks or self._dump_tasks:
            raise RuntimeError("Cannot clear metadata while transfers own KV pointers")
        self._connector_metadata = None

    def finalize_no_forward(self) -> None:
        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if metadata.no_forward:
                self.wait_for_save()

    def shutdown(self) -> None:
        first_error = None
        for layer_tasks in self._layer_load_tasks.values():
            for _, task in layer_tasks:
                try:
                    self._store_worker.wait(task)
                except Exception as exc:
                    first_error = first_error or exc
        for task in self._dump_tasks:
            try:
                self._store_worker.wait(task)
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

    def get_handshake_metadata(self) -> UCMStorageHandshake:
        if self.layout is None:
            raise RuntimeError("Register KV caches before the storage handshake")
        return UCMStorageHandshake(
            rank=self._parallel_rank,
            fingerprint=self.fingerprint,
            stored_policies=tuple(self.layout.store_sizes_by_policy),
        )

    def _mark_load_failed(self, request_metadata: RequestMetadata) -> None:
        request_id = request_metadata.request_id
        self._failed_load_request_ids.add(request_id)
        self._completion_metadata.mark_load_failed(request_id)
        self._invalid_vllm_block_ids.update(
            int(block_id)
            for load_plan in request_metadata.load_plans
            for group_block_ids in load_plan.vllm_block_ids_by_group
            for block_id in group_block_ids
        )

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # No work from a previous step may outlive its KV block ownership.
        # Binding always precedes forward; start_load_kv may run afterwards
        # in 0.30 when this step has no synchronous loads.
        if self._layer_load_tasks or self._dump_tasks:
            raise RuntimeError("Previous UCM transfers have not been drained")
        self._dump_submitted_layer_ids.clear()
        self._failed_load_request_ids.clear()
        self._failed_dump_jobs.clear()
        self._sub_job_ids_by_job.clear()
        self._step_io_completed = False
        self._connector_metadata = connector_metadata

    def _next_sub_job_id(self, job_id: int) -> int:
        sub_job_id = self._sub_job_ids_by_job.get(job_id, 0) + 1
        self._sub_job_ids_by_job[job_id] = sub_job_id
        return sub_job_id

    def _execute_prefetch(self, prefetch_job: PreparedJob) -> None:
        success = True
        # One shared Store prefetch per job. Other ranks acknowledge the no-op.
        if self._parallel_rank == ParallelRank():
            try:
                prefetch_task = self._store_worker.submit_prefetch(prefetch_job)
                self._store_worker.wait(prefetch_task)
            except StoreIOError:
                success = False
        self._completion_metadata.job_results.add(
            JobResult(prefetch_job.job_id, self._parallel_rank, success)
        )

    def _submit_load(
        self, request_metadata: RequestMetadata, layer_id: int | None = None
    ) -> None:
        transfer_batches = self._build_transfer_batches(
            request_metadata, is_dump=False, layer_id=layer_id
        )
        if not transfer_batches:
            return
        load_job = request_metadata.load_job
        sub_job_id = (
            self._next_sub_job_id(load_job.job_id) if layer_id is not None else None
        )
        load_task = self._store_worker.submit_get(
            load_job, transfer_batches, sub_job_id
        )
        self._layer_load_tasks.setdefault(layer_id, []).append(
            (request_metadata.request_id, load_task)
        )
        if layer_id is None:
            self._wait_layer_load(None)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self.has_connector_metadata():
            return
        connector_metadata = self._get_connector_metadata()
        for request_metadata in connector_metadata.requests.values():
            if request_metadata.prefetch_job is not None:
                self._execute_prefetch(request_metadata.prefetch_job)
            if request_metadata.prepare_get_failed:
                self._mark_load_failed(request_metadata)
                continue
            if request_metadata.load_job is None:
                continue
            try:
                if self.use_layerwise:
                    for layer_id in self.layout.layer_ids:
                        self._submit_load(request_metadata, layer_id)
                else:
                    self._submit_load(request_metadata)
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
        metadata = self._get_connector_metadata()
        unfinished_tasks = []
        fatal_error = None
        for request_id, task in self._layer_load_tasks.get(layer_id, ()):
            try:
                self._store_worker.wait(task)
            except StoreIOError:
                self._mark_load_failed(metadata.requests[request_id])
            except Exception as exc:
                # Unknown completion state: retain the task/descriptors and fail.
                unfinished_tasks.append((request_id, task))
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
                self._store_worker.wait(task)
            except StoreIOError:
                self._failed_dump_jobs.add(task.job_id)
            except Exception as exc:
                unfinished_tasks.append(task)
                fatal_error = fatal_error or exc
        self._dump_tasks = unfinished_tasks
        if fatal_error is not None:
            raise fatal_error

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise or not self.has_connector_metadata():
            return
        assert self.layout is not None
        layer_id = self.layout.layer_id_by_name.get(layer_name)
        if layer_id is not None:
            # An attention hook also waits for indexer caches of the same layer.
            self._wait_layer_load(layer_id)

    def _dump_layer(self, layer_id: int) -> None:
        if layer_id in self._dump_submitted_layer_ids:
            return
        for request_metadata in self._get_connector_metadata().requests.values():
            dump_job = request_metadata.dump_job
            if (
                dump_job is None
                or request_metadata.request_id in self._failed_load_request_ids
                or dump_job.job_id in self._failed_dump_jobs
            ):
                continue
            transfer_batches = self._build_transfer_batches(
                request_metadata, is_dump=True, layer_id=layer_id
            )
            if not transfer_batches:
                continue
            try:
                dump_task = self._store_worker.submit_put(
                    dump_job, transfer_batches, self._next_sub_job_id(dump_job.job_id)
                )
                self._dump_tasks.append(dump_task)
            except StoreIOError:
                self._failed_dump_jobs.add(dump_job.job_id)
        self._dump_submitted_layer_ids.add(layer_id)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if not self.use_layerwise or not self.has_connector_metadata():
            return
        if self._step_io_completed:
            return
        layer_id = self.layout.layer_id_by_name.get(layer_name)
        if layer_id is not None:
            self._dump_layer(layer_id)

    def wait_for_save(self) -> None:
        if not self.has_connector_metadata() or self._step_io_completed:
            return
        connector_metadata = self._get_connector_metadata()
        for layer_id in tuple(self._layer_load_tasks):
            self._wait_layer_load(layer_id)
        if self.use_layerwise:
            try:
                for layer_id in self.layout.layer_ids:
                    self._dump_layer(layer_id)
            finally:
                self._drain_dump_tasks()
        else:
            for request_metadata in connector_metadata.requests.values():
                dump_job = request_metadata.dump_job
                if (
                    dump_job is None
                    or request_metadata.request_id in self._failed_load_request_ids
                ):
                    continue
                transfer_batches = self._build_transfer_batches(
                    request_metadata, is_dump=True
                )
                if not transfer_batches:
                    continue
                try:
                    # One native submission per Prepare on this rank in bulk mode.
                    dump_task = self._store_worker.submit_put(
                        dump_job, transfer_batches
                    )
                    self._dump_tasks.append(dump_task)
                except StoreIOError:
                    self._failed_dump_jobs.add(dump_job.job_id)
                    logger.warning(
                        "UCM dump submission failed; skipping this job", exc_info=True
                    )
                self._drain_dump_tasks()
        for request_metadata in connector_metadata.requests.values():
            load_succeeded = (
                request_metadata.request_id not in self._failed_load_request_ids
            )
            if request_metadata.load_job is not None:
                self._completion_metadata.job_results.add(
                    JobResult(
                        request_metadata.load_job.job_id,
                        self._parallel_rank,
                        load_succeeded,
                    )
                )
            if request_metadata.dump_job is not None:
                self._completion_metadata.job_results.add(
                    JobResult(
                        request_metadata.dump_job.job_id,
                        self._parallel_rank,
                        load_succeeded
                        and request_metadata.dump_job.job_id
                        not in self._failed_dump_jobs,
                    )
                )
        # Empty ranks acknowledge the parent too, after all local IO is terminal.
        self._step_io_completed = True

    def poll_finished(self) -> None:
        # wait() and get_finished() consume the same native completion queue.
        # StoreWorkerClient caches consumed terminal status, so a later layer wait
        # never waits for a second notification of that job.
        self._store_worker.poll_finished()

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

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        # wait_for_save drains all layerwise work before returning to the
        # engine; bulk IO completes synchronously. No task spans scheduler steps.
        return None

    def _build_transfer_batches(
        self,
        request_metadata: RequestMetadata,
        *,
        is_dump: bool,
        layer_id: int | None = None,
    ) -> tuple[TransferBatch, ...]:
        """Resolve Scheduler plans into this rank's device-to-object ranges."""
        dispatch_plans = (
            request_metadata.dump_plans if is_dump else request_metadata.load_plans
        )
        transfer_batches = []
        for dispatch_plan in dispatch_plans:
            chunk_indices = dispatch_plan.token_starts // self.spec.chunk_size
            for group_id, group_block_ids in zip(
                dispatch_plan.group_ids, dispatch_plan.vllm_block_ids_by_group
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
                stage_object_keys = dispatch_plan.object_keys_by_pp_rank[
                    self._parallel_rank.pp_rank
                ]
                transfer_batches.append(
                    TransferBatch(
                        object_keys=tuple(
                            stage_object_keys[int(row)] for row in selected_chunk_rows
                        ),
                        device_ptrs=device_ptrs,
                        byte_sizes=byte_sizes,
                        store_offsets=store_offsets,
                        store_sizes=(
                            np.full(
                                len(selected_chunk_rows),
                                self.layout.store_sizes_by_policy[
                                    dispatch_plan.restore_policy
                                ],
                                dtype=np.uint64,
                            )
                            if is_dump
                            else None
                        ),
                    )
                )
        return tuple(transfer_batches)
