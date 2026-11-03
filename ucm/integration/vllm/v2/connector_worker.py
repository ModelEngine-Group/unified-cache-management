"""Worker KV registration, transfer construction and IO lifecycle."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

from ucm.logger import init_logger

from .connector_scheduler import UCMConnectorMetadata
from .kv_cache_layout import UCMKVCacheLayout
from .kv_cache_spec import RestorePolicy, groups_by_restore_policy
from .runtime import (
    ParallelRank,
    UCMStorageHandshake,
    make_ucm_chunk_ids,
)
from .store_client import PreparedJob, StoreClient, StoreIOError, Transfer, TransferTask

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
    load_failed_reqs: set[str] = field(default_factory=set)
    job_results: set[JobResult] = field(default_factory=set)

    def mark_failed(self, request_id: str) -> None:
        self.load_failed_reqs.add(request_id)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "UCMWorkerMetadata":
        if not isinstance(other, UCMWorkerMetadata):
            raise TypeError(f"Cannot aggregate {type(other).__name__}")
        self.load_failed_reqs.update(other.load_failed_reqs)
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
        self._store = StoreClient(context.launch_config)
        self._layer_load_tasks: dict[int, list[tuple[str, TransferTask]]] = {}
        self._dump_tasks: list[TransferTask] = []
        self._dump_submitted_layers = set()
        self._failed_load_reqs = set()
        self._failed_dump_jobs: set[int] = set()
        self._sub_job_ids: dict[int, int] = {}
        self._step_save_finalized = False
        self.layout: UCMKVCacheLayout | None = None
        self.transfer_builder: UCMTransferBuilder | None = None
        self._worker_metadata = UCMWorkerMetadata()
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
                    self._store.wait(task)
                except Exception as exc:
                    first_error = first_error or exc
        for task in self._dump_tasks:
            try:
                self._store.wait(task)
            except Exception as exc:
                first_error = first_error or exc
        try:
            self._store.close()
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
        self.transfer_builder = UCMTransferBuilder(self.layout, self._parallel_rank)

    def get_handshake_metadata(self) -> UCMStorageHandshake:
        if self.transfer_builder is None:
            raise RuntimeError("Register KV caches before the storage handshake")
        return UCMStorageHandshake(
            rank=self._parallel_rank,
            fingerprint=self.fingerprint,
            stored_policies=tuple(
                policy
                for policy, groups in self.transfer_builder.groups_by_policy.items()
                if groups
            ),
        )

    def _mark_load_failed(self, request_id: str, request: Any) -> None:
        self._failed_load_reqs.add(request_id)
        self._worker_metadata.mark_failed(request_id)
        self._invalid_vllm_block_ids.update(
            int(vllm_block_id)
            for plan in request.load_plans
            for vllm_block_ids in plan.vllm_block_ids_by_group
            for vllm_block_id in vllm_block_ids.tolist()
        )

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # No work from a previous step may outlive its KV block ownership.
        # Binding always precedes forward; start_load_kv may run afterwards
        # in 0.30 when this step has no synchronous loads.
        if self._layer_load_tasks or self._dump_tasks:
            raise RuntimeError("Previous UCM transfers have not been drained")
        self._dump_submitted_layers.clear()
        self._failed_load_reqs.clear()
        self._failed_dump_jobs.clear()
        self._sub_job_ids.clear()
        self._step_save_finalized = False
        self._connector_metadata = connector_metadata

    def _submit_job(
        self,
        operation: str,
        job: PreparedJob,
        transfers: tuple[Transfer, ...],
        *,
        layerwise: bool,
    ) -> TransferTask | None:
        if not transfers:
            return None
        sub_job_id = None
        if layerwise:
            sub_job_id = self._sub_job_ids.get(job.job_id, 0) + 1
            self._sub_job_ids[job.job_id] = sub_job_id
        return self._store.submit(operation, job, transfers, sub_job_id)

    @staticmethod
    def _transfers_by_request(
        transfers: tuple[Transfer, ...]
    ) -> dict[str, list[Transfer]]:
        by_request = defaultdict(list)
        for transfer in transfers:
            by_request[transfer.request_id].append(transfer)
        return by_request

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        assert self.layout is not None and self.transfer_builder is not None
        for request_id, request in metadata.requests.items():
            if request.prefetch_job is not None:
                success = True
                if self._parallel_rank == ParallelRank():
                    try:
                        task = self._store.submit_prefetch(request.prefetch_job)
                        self._store.wait(task)
                    except StoreIOError:
                        success = False
                self._worker_metadata.job_results.add(
                    JobResult(request.prefetch_job.job_id, self._parallel_rank, success)
                )
            if request.load_error:
                self._mark_load_failed(request_id, request)
                continue
            request_metadata = UCMConnectorMetadata(requests={request_id: request})
            try:
                if self.use_layerwise:
                    for layer_id in self.layout.layer_names_by_id:
                        transfers = self.transfer_builder.build_load_transfers(
                            request_metadata, layer_id=layer_id
                        )
                        task = self._submit_job(
                            "get", request.load_job, transfers, layerwise=True
                        )
                        if task is not None:
                            self._layer_load_tasks.setdefault(layer_id, []).append(
                                (request_id, task)
                            )
                else:
                    transfers = self.transfer_builder.build_load_transfers(
                        request_metadata
                    )
                    task = self._submit_job(
                        "get", request.load_job, transfers, layerwise=False
                    )
                    if task is not None:
                        self._layer_load_tasks.setdefault(-1, []).append(
                            (request_id, task)
                        )
                        self._wait_layer_load(-1)
            except StoreIOError:
                self._mark_load_failed(request_id, request)
        if self.use_layerwise:
            state_layer_ids = {
                layer.layer_index
                for group in self.spec.state_groups
                if group.group_id in self.layout.group_layouts
                for layer in self.layout.group_layouts[group.group_id].layer_specs
            }
            for layer_id in state_layer_ids:
                self._wait_layer_load(layer_id)

    def _wait_layer_load(self, layer_id: int) -> None:
        metadata = self._get_connector_metadata()
        pending = []
        fatal = None
        for request_id, task in self._layer_load_tasks.get(layer_id, ()):
            try:
                self._store.wait(task)
            except StoreIOError:
                self._mark_load_failed(request_id, metadata.requests[request_id])
            except Exception as exc:
                # Unknown completion state: retain the task/descriptors and fail.
                pending.append((request_id, task))
                fatal = fatal or exc
        if pending:
            self._layer_load_tasks[layer_id] = pending
        else:
            self._layer_load_tasks.pop(layer_id, None)
        if fatal is not None:
            raise fatal

    def _drain_dump_tasks(self) -> None:
        pending = []
        fatal = None
        for task in self._dump_tasks:
            try:
                self._store.wait(task)
            except StoreIOError:
                self._failed_dump_jobs.add(task.job_id)
            except Exception as exc:
                pending.append(task)
                fatal = fatal or exc
        self._dump_tasks = pending
        if fatal is not None:
            raise fatal

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise or not self.has_connector_metadata():
            return
        assert self.layout is not None
        layer_id = self.layout.layer_id_by_name.get(layer_name)
        if layer_id is not None:
            # An attention hook also waits for indexer caches of the same layer.
            self._wait_layer_load(layer_id)

    def _dump_metadata(self) -> UCMConnectorMetadata:
        metadata = self._get_connector_metadata()
        return UCMConnectorMetadata(
            requests={
                request_id: request
                for request_id, request in metadata.requests.items()
                if request_id not in self._failed_load_reqs
            }
        )

    def _save_layer(self, layer_name: str, metadata: UCMConnectorMetadata) -> None:
        assert self.transfer_builder is not None
        if layer_name in self._dump_submitted_layers:
            return
        transfers = self.transfer_builder.build_dump_transfers(
            metadata, layer_name=layer_name
        )
        for request_id, batches in self._transfers_by_request(transfers).items():
            job = metadata.requests[request_id].dump_job
            if job.job_id in self._failed_dump_jobs:
                continue
            try:
                task = self._submit_job("put", job, tuple(batches), layerwise=True)
                if task is not None:
                    self._dump_tasks.append(task)
            except StoreIOError:
                self._failed_dump_jobs.add(job.job_id)
        self._dump_submitted_layers.add(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if not self.use_layerwise or not self.has_connector_metadata():
            return
        assert self.layout is not None
        if self._step_save_finalized or layer_name not in self.layout.layer_id_by_name:
            return
        # Save this cache only: another cache with the same layer index may
        # still be computing. Its own hook or the end-of-forward fallback saves it.
        self._save_layer(layer_name, self._dump_metadata())

    def wait_for_save(self) -> None:
        if not self.has_connector_metadata() or self._step_save_finalized:
            return
        assert self.layout is not None and self.transfer_builder is not None
        bound = self._get_connector_metadata()
        for layer_id in tuple(self._layer_load_tasks):
            self._wait_layer_load(layer_id)
        if self.use_layerwise:
            try:
                for layer_name in self.layout.layer_id_by_name:
                    self._save_layer(layer_name, self._dump_metadata())
            finally:
                self._drain_dump_tasks()
        else:
            transfers = self.transfer_builder.build_dump_transfers(
                self._dump_metadata()
            )
            for request_id, batches in self._transfers_by_request(transfers).items():
                job = bound.requests[request_id].dump_job
                try:
                    # All groups/chunks for this Prepare form one native bulk
                    # submission, whose id is exactly the Scheduler job_id.
                    task = self._store.submit("put", job, tuple(batches))
                    self._dump_tasks.append(task)
                    try:
                        self._store.wait(task)
                    except StoreIOError:
                        self._dump_tasks.pop()
                        raise
                    else:
                        self._dump_tasks.pop()
                except StoreIOError:
                    self._failed_dump_jobs.add(job.job_id)
                    logger.warning("UCM dump failed; skipping this job", exc_info=True)
        for request in bound.requests.values():
            load_ok = request.request_id not in self._failed_load_reqs
            if request.load_job is not None:
                self._worker_metadata.job_results.add(
                    JobResult(request.load_job.job_id, self._parallel_rank, load_ok)
                )
            if request.dump_job is not None:
                self._worker_metadata.job_results.add(
                    JobResult(
                        request.dump_job.job_id,
                        self._parallel_rank,
                        load_ok
                        and request.dump_job.job_id not in self._failed_dump_jobs,
                    )
                )
        # No-op ranks still report the parent job. Completion is emitted only
        # after every submitted subjob is terminal and its event is consumed.
        self._step_save_finalized = True

    def poll_finished(self) -> None:
        # wait() and get_finished() consume the same native completion queue.
        # StoreClient caches consumed terminal status, so a later layer wait
        # never waits for a second notification of that job.
        self._store.poll_finished()

    def build_connector_worker_meta(self) -> UCMWorkerMetadata | None:
        if (
            not self._worker_metadata.load_failed_reqs
            and not self._worker_metadata.job_results
        ):
            return None
        worker_metadata = self._worker_metadata
        self._worker_metadata = UCMWorkerMetadata()
        return worker_metadata

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_vllm_block_ids = self._invalid_vllm_block_ids
        self._invalid_vllm_block_ids = set()
        return invalid_vllm_block_ids

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        # wait_for_save drains all layerwise work before returning to the
        # engine; bulk IO completes synchronously. No task spans scheduler steps.
        return None


class UCMTransferBuilder:
    def __init__(
        self, layout: UCMKVCacheLayout, rank: ParallelRank = ParallelRank()
    ) -> None:
        self.layout = layout
        self.rank = rank
        self.groups_by_policy = {
            policy: tuple(
                layout.store_layouts[group.group_id]
                for group in groups
                if group.group_id in layout.store_layouts
            )
            for policy, groups in groups_by_restore_policy(layout.spec).items()
        }
        self.group_byte_offsets: dict[int, int] = {}
        self.record_nbytes: dict[RestorePolicy, int] = {}
        for policy, stores in self.groups_by_policy.items():
            offset = 0
            for store in stores:
                self.group_byte_offsets[store.group_id] = offset
                offset += store.whole_chunk_nbytes
            self.record_nbytes[policy] = offset

    def build_load_transfers(
        self,
        metadata: UCMConnectorMetadata,
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
    ) -> tuple[Transfer, ...]:
        return self._build(metadata, False, layer_name, layer_id)

    def build_dump_transfers(
        self,
        metadata: UCMConnectorMetadata,
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
    ) -> tuple[Transfer, ...]:
        return self._build(metadata, True, layer_name, layer_id)

    def _build(
        self,
        metadata: UCMConnectorMetadata,
        dump: bool,
        layer_name: str | None,
        layer_id: int | None,
    ) -> tuple[Transfer, ...]:
        if layer_name is not None and layer_id is not None:
            raise ValueError("Specify either layer_name or layer_id")
        names = (
            self.layout.layer_names_by_id[layer_id]
            if layer_id is not None
            else None if layer_name is None else frozenset((layer_name,))
        )
        transfers = []
        for request in metadata.requests.values():
            for plan in request.dump_plans if dump else request.load_plans:
                positions = {gid: i for i, gid in enumerate(plan.group_ids)}
                if len(positions) != len(plan.group_ids) or len(positions) != len(
                    plan.vllm_block_ids_by_group
                ):
                    raise ValueError("Plan must name each native group exactly once")
                indices = plan.token_starts // self.layout.spec.chunk_size
                for store in self.groups_by_policy[plan.restore_policy]:
                    gid = store.group_id
                    base = self.group_byte_offsets[gid]
                    hbm = store.group_layout
                    mask = (
                        None if names is None else hbm.segment_mask(layer_names=names)
                    )
                    if mask is not None and not mask.any():
                        continue
                    rows = (
                        store.writer_rows(indices) if dump else np.arange(len(indices))
                    )
                    if not len(rows):
                        continue
                    blocks = (
                        np.asarray(
                            plan.vllm_block_ids_by_group[positions[gid]],
                            dtype=np.uint64,
                        )
                        .reshape(len(indices), -1)[rows]
                        .reshape(-1)
                    )
                    offsets, ptrs, sizes = store.resolve_matrices(
                        blocks, plan.token_starts[rows], segment_mask=mask
                    )
                    offsets = np.broadcast_to(
                        offsets[0] + np.uint64(base), offsets.shape
                    )
                    ids = make_ucm_chunk_ids(
                        tuple(plan.common_ucm_chunk_ids[row] for row in rows),
                        plan.restore_policy,
                        self.rank.pp_rank,
                    )
                    transfers.append(
                        (
                            request.request_id,
                            Transfer(
                                request.request_id,
                                ids,
                                indices[rows],
                                ptrs,
                                sizes,
                                offsets,
                                (
                                    np.full(
                                        len(rows),
                                        self.record_nbytes[plan.restore_policy],
                                        dtype=np.uint64,
                                    )
                                    if dump
                                    else None
                                ),
                            ),
                        )
                    )
        combined = {}
        for request_id, transfer in transfers:
            key = (request_id, transfer.ucm_chunk_ids, tuple(transfer.chunk_indices))
            if key not in combined:
                combined[key] = transfer
                continue
            previous = combined[key]

            def join(left, right):
                return np.broadcast_to(
                    np.concatenate((left[0], right[0])),
                    (len(left), left.shape[1] + right.shape[1]),
                )

            combined[key] = Transfer(
                transfer.request_id,
                transfer.ucm_chunk_ids,
                transfer.chunk_indices,
                np.concatenate((previous.ptrs, transfer.ptrs), axis=1),
                join(previous.sizes, transfer.sizes),
                join(previous.ucm_chunk_offsets, transfer.ucm_chunk_offsets),
                transfer.record_nbytes,
            )
        return tuple(combined.values())
