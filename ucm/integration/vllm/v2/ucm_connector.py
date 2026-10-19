"""Single vLLM facade for the isolated connector-v2 implementation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)

from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.utils import Config

from .factory import create_proxy
from .layout import (
    UCMKVCacheLayout,
    UCMKVCacheSpec,
    parse_kv_cache_config,
)
from .parallel import ParallelLayout
from .ucm_block_id import UCM_BLOCK_ID_VERSION, writer_rows
from .ucm_kv_cache import UCMTransferBuilder
from .ucm_lookup import UCMLookup
from .ucm_proxy import (
    UCMProxyError,
    UCMProxyTask,
    UCMProxyTransfer,
)
from .ucm_scheduler import UCMConnectorMetadata, UCMDispatcher
from .ucm_sharding import (
    ParallelRank,
    UCMRecordClass,
    UCMSharding,
    groups_by_attention_type,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


@dataclass(frozen=True)
class UCMRuntimeContext:
    role: KVConnectorRole
    device_type: str
    engine_id: str | None
    rank: int | None
    world_size: int

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        *,
        rank: int | None = None,
    ) -> "UCMRuntimeContext":
        parallel = vllm_config.parallel_config
        return cls(
            role=role,
            device_type=vllm_config.device_config.device_type,
            engine_id=vllm_config.instance_id,
            rank=rank,
            world_size=parallel.world_size,
        )


@dataclass
class UCMWorkerMetadata(KVConnectorWorkerMetadata):
    load_failed_reqs: set[str] = field(default_factory=set)

    def mark_failed(self, request_id: str) -> None:
        self.load_failed_reqs.add(request_id)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "UCMWorkerMetadata":
        if not isinstance(other, UCMWorkerMetadata):
            raise TypeError(f"Cannot aggregate {type(other).__name__}")
        self.load_failed_reqs.update(other.load_failed_reqs)
        return self


def _load_launch_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Resolve kv_connector_extra_config through the shared ucm.utils.Config.

    The UCM_CONFIG_FILE yaml and terminal-input forms resolve exactly like
    every other UCM connector, including the v2 Store/Proxy configuration.
    """
    return dict(Config(vllm_config.kv_transfer_config).get_config())


def _worker_rank() -> int:
    from vllm.distributed.parallel_state import get_world_group

    return get_world_group().rank


def _worker_parallel_ranks(layout: ParallelLayout) -> tuple[int, int, int]:
    """This worker's (tp_rank, pp_rank, pcp_rank), read from vLLM's groups."""
    from vllm.distributed.parallel_state import (
        get_pcp_group,
        get_pp_group,
        get_tp_group,
    )

    pcp_rank = get_pcp_group().rank_in_group if layout.pcp > 1 else 0
    return get_tp_group().rank_in_group, get_pp_group().rank_in_group, pcp_rank


def _jsonable(value: Any) -> Any:
    """Best-effort JSON rendering for raw KVCacheConfig payloads."""

    import enum as _enum
    from dataclasses import fields as _fields
    from dataclasses import is_dataclass as _is_dataclass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, _enum.Enum):
        return str(value)
    if _is_dataclass(value):
        return {
            dataclass_field.name: _jsonable(getattr(value, dataclass_field.name))
            for dataclass_field in _fields(value)
        }
    if isinstance(value, dict):
        return {
            str(dict_key): _jsonable(dict_value)
            for dict_key, dict_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(element) for element in value]
    if type(value).__module__.split(".")[0] == "torch":
        return str(value)
    # Plain objects (e.g. slot wrappers): expose their attributes so
    # SimpleNamespace-style entries serialize like their dataclass twins.
    slot_names = getattr(value, "__slots__", None)
    if slot_names:
        return {
            slot_name: _jsonable(getattr(value, slot_name))
            for slot_name in slot_names
            if hasattr(value, slot_name)
        }
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict) and attributes:
        return {
            str(attribute_name): _jsonable(attribute_value)
            for attribute_name, attribute_value in attributes.items()
        }
    return repr(value)


def _dump_raw_kv_cache_config(kv_cache_config: Any, rank: int | None) -> None:
    """Write the raw KVCacheConfig vLLM handed over to a JSON file.

    Enabled by ``UCM_V2_DUMP_CONFIG=<path>``; a ``%d`` in the path receives
    the worker rank so tensor-parallel shards can be captured side by side.
    """

    import json

    path_template = os.environ.get("UCM_V2_DUMP_CONFIG", "")
    if not path_template:
        return
    path = Path(
        path_template % rank
        if "%d" in path_template or "%s" in path_template
        else path_template
    )

    group_payloads = []
    for group in kv_cache_config.kv_cache_groups:
        group_payloads.append(
            {
                "layer_names": group.layer_names,
                "is_eagle_group": group.is_eagle_group,
                "kv_cache_spec": _jsonable(group.kv_cache_spec),
            }
        )
    tensor_payloads = [_jsonable(tensor) for tensor in kv_cache_config.kv_cache_tensors]
    payload = {
        "num_blocks": kv_cache_config.num_blocks,
        "kv_cache_tensors": tensor_payloads,
        "kv_cache_groups": group_payloads,
        "prefix_cache_retention_interval": kv_cache_config.prefix_cache_retention_interval,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2, ensure_ascii=False)
    print(
        f"[ucm-v2] raw KVCacheConfig dumped to {path} "
        f"(num_blocks={payload['num_blocks']}, "
        f"tensors={len(tensor_payloads)}, "
        f"groups={len(group_payloads)})",
        flush=True,
    )


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    """One v2 lifecycle facade for grouped caches."""

    # This worker's rank.
    _parallel_rank: ParallelRank = (0, 0, 0)
    # (replica index, replica count) of this rank for each record class.
    # A missing record class means one replica.
    _replica_index_and_count_by_class: Mapping[UCMRecordClass, tuple[int, int]] = {}

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        # Resolve the YAML form as well, before vLLM selects its graph mode.
        launch_config = Config(
            SimpleNamespace(kv_connector_extra_config=extra_config)
        ).get_config()
        return bool(launch_config.get("use_layerwise", False))

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        launch_config = _load_launch_config(vllm_config)
        # vLLM's resolved scheduler granularity, not the raw CacheConfig
        # page size: multiple groups hash the LCM of group block sizes (the
        # resolve call is monkey-patched by vllm-ascend inside its process).
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        scheduler_block_size, _ = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        ucm_cache_block_size = launch_config.get("ucm_cache_block_size")
        if ucm_cache_block_size is not None:
            ucm_cache_block_size = int(ucm_cache_block_size)
        worker_rank = None if role == KVConnectorRole.SCHEDULER else _worker_rank()
        _dump_raw_kv_cache_config(kv_cache_config, worker_rank)
        self.parallel_layout = ParallelLayout.from_config(vllm_config)

        self.context = UCMRuntimeContext.from_vllm_config(
            vllm_config, role, rank=worker_rank
        )
        text_config = vllm_config.model_config.hf_text_config
        self._num_hidden_layers = text_config.num_hidden_layers
        self.spec: UCMKVCacheSpec = parse_kv_cache_config(
            kv_cache_config,
            scheduler_block_size=scheduler_block_size,
            hf_text_config=text_config,
            ucm_cache_block_size=ucm_cache_block_size,
            device_type=self.context.device_type,
        )
        self.parallel_layout.apply_context_parallel(self.spec)
        # Raises at startup if the topology does not fit in UCM block ids.
        self._sharding = UCMSharding(self.spec, self.parallel_layout)
        if role == KVConnectorRole.WORKER:
            self._parallel_rank = _worker_parallel_ranks(self.parallel_layout)
            self._replica_index_and_count_by_class = {
                record_class: self._sharding.replica_index_and_count(
                    record_class, self._parallel_rank
                )
                for attention_type, _ in groups_by_attention_type(self.spec)
                for record_class in self._sharding.record_classes(attention_type)
            }
        model_dtype = str(vllm_config.model_config.dtype).rsplit(".", 1)[-1]
        namespace_schema = {
            "model": vllm_config.model_config.hf_text_config.to_dict(),
            "kv_dtype": vllm_config.cache_config.cache_dtype,
            "mamba_dtype": vllm_config.cache_config.mamba_cache_dtype,
            "mamba_ssm_dtype": vllm_config.cache_config.mamba_ssm_cache_dtype,
            "kv_cache_layout": kv_cache_config.kv_cache_layout,
            "ascend_gqa_head_first": (
                envs.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND")
                if self.context.device_type == "npu"
                else None
            ),
        }
        schema_digest = hashlib.sha256(
            json.dumps(namespace_schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        # The Store only sees UCM block ids, so everything that changes the
        # stored data (model, layout, block sizes, topology, layerwise) and
        # the id version goes into the hash seed.
        namespace = (
            f"{self.context.device_type}-{model_dtype}"
            f"-{schema_digest}-b{self.spec.scheduler_block_size}"
            f"-c{self.spec.ucm_cache_block_size}"
            f"-{self.parallel_layout.namespace}-r{UCM_BLOCK_ID_VERSION}"
        )
        self.use_layerwise = bool(launch_config.get("use_layerwise", False))
        if self.use_layerwise:
            # Layerwise copies payloads only; bulk Block First records may
            # additionally include slot padding.
            namespace += "-layerwise"
        hasher = RequestHasher(vllm_config, 0)
        base_seed = hasher(("UCM_HASH_SEED", namespace))
        self._proxy = create_proxy(
            launch_config, record_token_span=self.spec.ucm_cache_block_size
        )
        self._layer_load_tasks: dict[int, list[tuple[str, UCMProxyTask]]] = {}
        self._dump_tasks: list[UCMProxyTask] = []
        self._saved_layer_names: set[str] = set()
        self._failed_load_reqs: set[str] = set()
        self._save_error: UCMProxyError | None = None
        self._save_complete = False
        self.layout: UCMKVCacheLayout | None = None
        self.transfer_builder: UCMTransferBuilder | None = None
        self._worker_metadata = UCMWorkerMetadata()
        self._invalid_vllm_block_ids: set[int] = set()
        self.dispatcher = (
            UCMDispatcher(
                self.spec,
                UCMLookup(self.spec, self._proxy, self._sharding),
                # Plans carry common UCM block ids. Each worker builds the
                # local UCM block ids of its own shards when building transfers.
                hasher,
                base_seed,
                load_threshold_tokens=int(
                    launch_config.get("load_tokens_threshold", 0)
                ),
                # CUDA V1 and CUDA/Ascend V2 runners copy Mamba state into the running
                # block before connector loads, so loads must fill it too. Only
                # Ascend's V1 runner copies after loads, from the boundary block.
                need_load_to_running_state=(
                    self.context.device_type != "npu" or vllm_config.use_v2_model_runner
                ),
            )
            if role == KVConnectorRole.SCHEDULER
            else None
        )

    def shutdown(self) -> None:
        first_error = None
        for layer_tasks in self._layer_load_tasks.values():
            for _, task in layer_tasks:
                try:
                    self._proxy.wait(task)
                except Exception as exc:
                    first_error = first_error or exc
        for task in self._dump_tasks:
            try:
                self._proxy.wait(task)
            except Exception as exc:
                first_error = first_error or exc
        self._layer_load_tasks.clear()
        self._dump_tasks.clear()
        try:
            self._proxy.close()
        except Exception as exc:
            first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.dispatcher is not None
        lookup_result = self.dispatcher.lookup(request, num_computed_tokens)
        return lookup_result.external_hit_tokens, False

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> UCMConnectorMetadata:
        assert self.dispatcher is not None
        return self.dispatcher.build_from_scheduler_output(scheduler_output)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        # 0.30 supplies authoritative tables in kv_connector_block_state.
        return None

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        if self.context.role != KVConnectorRole.WORKER:
            raise RuntimeError("KV cache registration is only available on worker")
        self.layout = UCMKVCacheLayout(
            self.spec,
            kv_caches,
            kv_cache_config=self._kv_cache_config,
            num_hidden_layers=self._num_hidden_layers,
            use_layerwise=self.use_layerwise,
        )
        self.transfer_builder = UCMTransferBuilder(
            self.layout, self._sharding, self._parallel_rank
        )
        self._proxy.register_tensors(kv_caches)

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
        self._saved_layer_names.clear()
        self._failed_load_reqs.clear()
        self._save_error = None
        self._save_complete = False
        super().bind_connector_metadata(connector_metadata)

    def _owned_rows(
        self, record_class: UCMRecordClass, common_ucm_block_ids: Sequence[bytes]
    ) -> np.ndarray:
        """Rows of a plan that this rank writes for one record class.

        Replicas hold the same data, so they split the writes by common UCM
        block id and each block is stored once. Every replica still loads all blocks.
        """
        replica_index, replica_count = self._replica_index_and_count_by_class.get(
            record_class, (0, 1)
        )
        if replica_count == 1:
            return np.arange(len(common_ucm_block_ids), dtype=np.int64)
        else:
            return writer_rows(common_ucm_block_ids, replica_index, replica_count)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        assert self.layout is not None and self.transfer_builder is not None
        for request_id, request in metadata.requests.items():
            request_metadata = UCMConnectorMetadata(requests={request_id: request})
            try:
                if self.use_layerwise:
                    for layer_id in self.layout.layer_names_by_id:
                        for batch in self.transfer_builder.build_load_transfers(
                            request_metadata, layer_id=layer_id
                        ):
                            task = self._proxy.enqueue("load", batch)
                            self._layer_load_tasks.setdefault(layer_id, []).append(
                                (request_id, task)
                            )
                else:
                    for batch in self.transfer_builder.build_load_transfers(
                        request_metadata
                    ):
                        self._proxy.submit("load", batch)
            except UCMProxyError:
                self._mark_load_failed(request_id, request)
        if self.use_layerwise:
            # State layerwise hooks are not wired in this connector yet. Wait
            # before forward, including attention caches sharing a layer index.
            state_layer_ids = {
                layer.layer_index
                for group_layout in self.layout.group_layouts.values()
                if group_layout.is_state_snapshot
                for layer in group_layout.layers
            }
            for layer_id in state_layer_ids:
                self._wait_layer_load(layer_id)

    def _wait_layer_load(self, layer_id: int) -> None:
        metadata = self._get_connector_metadata()
        for request_id, task in self._layer_load_tasks.pop(layer_id, ()):
            try:
                self._proxy.wait(task)
            except UCMProxyError:
                self._mark_load_failed(request_id, metadata.requests[request_id])

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
        assert self.layout is not None and self.transfer_builder is not None
        if layer_name in self._saved_layer_names:
            return
        if self._save_error is not None:
            raise self._save_error
        try:
            for batch in self.transfer_builder.build_dump_transfers(
                metadata, layer_name=layer_name, row_selector=self._owned_rows
            ):
                self._dump_tasks.append(self._proxy.enqueue("dump", batch))
        except UCMProxyError as exc:
            self._save_error = exc
            raise
        self._saved_layer_names.add(layer_name)

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
        if self._save_complete or layer_name not in self.layout.layer_id_by_name:
            return
        # Save this cache only: another cache with the same layer index may
        # still be computing. Its own hook or the end-of-forward fallback saves it.
        self._save_layer(layer_name, self._dump_metadata())

    def _dump_empty_shards(
        self, metadata: UCMConnectorMetadata
    ) -> list[UCMProxyTransfer]:
        assert self.transfer_builder is not None
        empty_markers = list(
            self.transfer_builder.build_empty_markers(metadata, self._owned_rows)
        )
        for transfer in empty_markers:
            self._proxy.submit("dump", transfer)
        return empty_markers

    def wait_for_save(self) -> None:
        if not self.has_connector_metadata() or self._save_complete:
            return
        assert self.layout is not None and self.transfer_builder is not None
        metadata = self._dump_metadata()

        def abort_pending(*, failed_only: bool = False) -> None:
            abort_requests = {
                request_id: request
                for request_id, request in self._get_connector_metadata().requests.items()
                if request_id in self._failed_load_reqs or not failed_only
            }
            for ucm_block_ids, block_indices in self.transfer_builder.dump_targets(
                UCMConnectorMetadata(requests=abort_requests), self._owned_rows
            ):
                self._proxy.abort(ucm_block_ids, block_indices)

        try:
            if not self.use_layerwise:
                dump_batches = self.transfer_builder.build_dump_transfers(
                    metadata, row_selector=self._owned_rows
                )
                for batch in dump_batches:
                    self._proxy.submit("dump", batch)
                empty_markers = self._dump_empty_shards(metadata)
                for transfer in (*dump_batches, *empty_markers):
                    self._proxy.commit(transfer.ucm_block_ids, transfer.block_indices)
            else:
                # Drain loads whose hooks were skipped before deciding which saves are valid.
                for layer_id in tuple(self._layer_load_tasks):
                    self._wait_layer_load(layer_id)
                metadata = self._dump_metadata()
                try:
                    for layer_name in self.layout.layer_id_by_name:
                        self._save_layer(layer_name, metadata)
                except UCMProxyError as exc:
                    self._save_error = exc
                # A later submission failure must not strand earlier source pointers.
                for task in self._dump_tasks:
                    try:
                        self._proxy.wait(task)
                    except UCMProxyError as exc:
                        self._save_error = exc
                self._dump_tasks.clear()
                if self._save_error is not None:
                    raise self._save_error
                self._dump_empty_shards(metadata)
                for ucm_block_ids, block_indices in self.transfer_builder.dump_targets(
                    metadata, self._owned_rows
                ):
                    self._proxy.commit(ucm_block_ids, block_indices)
            if self._failed_load_reqs:
                abort_pending(failed_only=True)
        except UCMProxyError:
            abort_pending()
            raise
        self._save_complete = True

    def build_connector_worker_meta(self) -> UCMWorkerMetadata | None:
        if not self._worker_metadata.load_failed_reqs:
            return None
        worker_metadata = self._worker_metadata
        self._worker_metadata = UCMWorkerMetadata()
        return worker_metadata

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        # The no-forward path skips wait_for_save, but can still carry an
        # exact boundary hand-off. Forward/draft finalization keeps the
        # engine's own wait_for_save ordering.
        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if metadata.no_forward and any(
                request.dump_plans for request in metadata.requests.values()
            ):
                self.wait_for_save()
        return super().get_transfer_results(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_vllm_block_ids = self._invalid_vllm_block_ids
        self._invalid_vllm_block_ids = set()
        return invalid_vllm_block_ids

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        assert self.dispatcher is not None
        worker_metadata = connector_output.kv_connector_worker_meta
        if worker_metadata is None:
            return
        for request_id in worker_metadata.load_failed_reqs:
            self.dispatcher.requests.pop(request_id, None)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        # wait_for_save drains all layerwise work before returning to the
        # engine; bulk IO completes synchronously. No task spans scheduler steps.
        return None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


__all__ = [
    "UCMConnector",
    "UCMConnectorMetadata",
    "UCMRuntimeContext",
    "UCMWorkerMetadata",
]
