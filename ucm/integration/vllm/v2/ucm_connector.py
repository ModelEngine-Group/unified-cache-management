"""Single vLLM facade for the isolated connector-v2 implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)

from .ucm_kv_cache import UCMKVCacheLayout, UCMKVCacheSpec, parse_kv_cache_config
from .ucm_proxy import (
    SimpleFileUCMProxy,
    UCMProxyAdapter,
    UCMProxyError,
)
from .ucm_scheduler import (
    RequestHasher,
    UCMConnectorMetadata,
    UCMDispatcher,
    UCMLookupCoordinator,
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
        world_size = int(getattr(parallel, "world_size", 0) or 0)
        if world_size <= 0:
            world_size = (
                int(getattr(parallel, "tensor_parallel_size", 1))
                * int(getattr(parallel, "pipeline_parallel_size", 1))
                * int(getattr(parallel, "data_parallel_size", 1))
            )
        return cls(
            role=role,
            device_type=str(
                getattr(
                    getattr(vllm_config, "device_config", None),
                    "device_type",
                    None,
                )
                or "unknown"
            ).lower(),
            engine_id=getattr(vllm_config, "instance_id", None),
            rank=rank,
            world_size=world_size,
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
    kv_transfer_config = vllm_config.kv_transfer_config
    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if not extra_config:
        return {}
    if "UCM_CONFIG_FILE" not in extra_config:
        return dict(extra_config)

    import yaml

    path = Path(str(extra_config["UCM_CONFIG_FILE"]))
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"UCM config file must contain a mapping: {path}")
    return config


def _storage_root(launch_config: dict[str, Any]) -> Path:
    explicit = launch_config.get("v2_storage_path")
    if explicit:
        return Path(str(explicit))

    connectors = launch_config.get("ucm_connectors")
    if (
        not isinstance(connectors, (list, tuple))
        or len(connectors) != 1
        or not isinstance(connectors[0], dict)
    ):
        raise ValueError(
            "connector v2 requires exactly one ucm_connectors entry or "
            "v2_storage_path"
        )
    store_config = connectors[0].get("ucm_connector_config") or {}
    if not isinstance(store_config, dict):
        raise ValueError("ucm_connector_config must be a mapping")
    backends = store_config.get("storage_backends")
    if isinstance(backends, (list, tuple)):
        backends = backends[0] if backends else None
    elif isinstance(backends, str) and os.pathsep in backends:
        backends = backends.split(os.pathsep, 1)[0]
    if not backends:
        raise ValueError("connector v2 requires storage_backends or v2_storage_path")
    return Path(str(backends))


def _worker_rank(vllm_config: "VllmConfig") -> int:
    configured = getattr(vllm_config.parallel_config, "rank", None)
    if configured is not None:
        return int(configured)
    from vllm.distributed.parallel_state import get_world_group

    return int(get_world_group().rank)


def _jsonable(value: Any) -> Any:
    """Best-effort JSON rendering for raw KVCacheConfig payloads."""

    import enum as _enum
    from dataclasses import fields as _fields, is_dataclass as _is_dataclass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, _enum.Enum):
        return str(value)
    if _is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in _fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if type(value).__module__.split(".")[0] == "torch":
        return str(value)
    # Plain objects (e.g. slot wrappers): expose their attributes so
    # SimpleNamespace-style entries serialize like their dataclass twins.
    slots = getattr(value, "__slots__", None)
    if slots:
        return {
            name: _jsonable(getattr(value, name))
            for name in slots
            if hasattr(value, name)
        }
    value_vars = getattr(value, "__dict__", None)
    if isinstance(value_vars, dict) and value_vars:
        return {str(key): _jsonable(item) for key, item in value_vars.items()}
    return repr(value)


def _dump_raw_kv_cache_config(kv_cache_config: Any, rank: int | None) -> None:
    """Write the raw KVCacheConfig vLLM handed over to a JSON file.

    Enabled by ``UCM_V2_DUMP_CONFIG=<path>``; a ``%d`` in the path receives
    the worker rank so tensor-parallel shards can be captured side by side.
    """

    import json

    raw_path = os.environ.get("UCM_V2_DUMP_CONFIG", "")
    if not raw_path:
        return
    path = Path(raw_path % rank if "%d" in raw_path or "%s" in raw_path else raw_path)

    groups = []
    for group in getattr(kv_cache_config, "kv_cache_groups", ()) or ():
        groups.append(
            {
                "layer_names": list(getattr(group, "layer_names", ()) or ()),
                "is_eagle_group": bool(getattr(group, "is_eagle_group", False)),
                "kv_cache_spec": _jsonable(group.kv_cache_spec),
            }
        )
    tensors = [
        _jsonable(tensor)
        for tensor in getattr(kv_cache_config, "kv_cache_tensors", ()) or ()
    ]
    payload = {
        "num_blocks": int(getattr(kv_cache_config, "num_blocks", 0)),
        "kv_cache_tensors": tensors,
        "kv_cache_groups": groups,
        "prefix_cache_retention_interval": int(
            getattr(kv_cache_config, "prefix_cache_retention_interval", 0) or 0
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    print(
        f"[ucm-v2] raw KVCacheConfig dumped to {path} "
        f"(num_blocks={payload['num_blocks']}, "
        f"tensors={len(tensors)}, "
        f"groups={len(groups)})",
        flush=True,
    )


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    """One v2 lifecycle facade for grouped caches and the DSV4 policy."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        launch_config = _load_launch_config(vllm_config)
        scheduler_block_size = int(vllm_config.cache_config.block_size)
        chunk_size = launch_config.get("chunk_size")
        if chunk_size is not None:
            chunk_size = int(chunk_size)
        is_scheduler = role == KVConnectorRole.SCHEDULER
        rank = None if is_scheduler else _worker_rank(vllm_config)
        _dump_raw_kv_cache_config(kv_cache_config, rank)
        hasher = RequestHasher(vllm_config, 0)
        base_seed = hasher("UCM_HASH_SEED")

        self.context = UCMRuntimeContext.from_vllm_config(vllm_config, role, rank=rank)
        self.spec: UCMKVCacheSpec = parse_kv_cache_config(
            kv_cache_config,
            scheduler_block_size=scheduler_block_size,
            chunk_size=chunk_size,
            device_type=self.context.device_type,
        )
        dtype = str(vllm_config.model_config.dtype).rsplit(".", 1)[-1]
        policy = "dsv4" if self.spec.is_dsv4 else "grouped"
        namespace = (
            f"{self.context.device_type}-{dtype}-{policy}"
            f"-b{self.spec.scheduler_block_size}-c{self.spec.chunk_size}"
        )
        root = _storage_root(launch_config) / ".ucm-v2" / namespace
        self._proxy = UCMProxyAdapter(SimpleFileUCMProxy(root))
        self.layout: UCMKVCacheLayout | None = None
        self._worker_metadata = UCMWorkerMetadata()
        self._invalid_block_ids: set[int] = set()
        self.dispatcher = UCMDispatcher(self.spec) if is_scheduler else None
        self.lookup_coordinator = (
            UCMLookupCoordinator(
                self.spec,
                self._proxy,
                # Scheduler always creates logical rank-0 keys. Worker-side
                # rank scoping is a separate physical-data policy.
                RequestHasher(vllm_config, 0),
                base_seed,
                load_threshold_tokens=int(
                    launch_config.get("load_tokens_threshold", 0)
                ),
            )
            if is_scheduler
            else None
        )

    def get_block_size(self) -> int:
        return self.spec.scheduler_block_size

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if self.lookup_coordinator is None or self.dispatcher is None:
            raise RuntimeError("lookup is only available on the scheduler role")
        result = self.lookup_coordinator.lookup(request, num_computed_tokens)
        self.dispatcher.record_lookup(request, num_computed_tokens, result)
        return result.external_hit_tokens, False

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> UCMConnectorMetadata:
        if self.dispatcher is None:
            raise RuntimeError("dispatch is only available on the scheduler role")
        return self.dispatcher.build_from_scheduler_output(scheduler_output)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        # vLLM 0.26 SchedulerOutput is the single source of truth for complete
        # new/resumed block tables and cached-request deltas.
        return None

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        if self.context.role != KVConnectorRole.WORKER:
            raise RuntimeError("KV cache registration is only available on worker")
        self.layout = UCMKVCacheLayout(self.spec, kv_caches)
        self._proxy.register_tensors(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMConnectorMetadata):
            raise TypeError(f"Unexpected connector metadata: {type(metadata).__name__}")
        if self.layout is None:
            raise RuntimeError("register_kv_caches must run before loading")
        for request_id, request in metadata.requests.items():
            request_metadata = UCMConnectorMetadata(requests={request_id: request})
            batch = self.layout.build_load_batches(request_metadata)
            try:
                self._proxy.load(
                    batch.block_ids, batch.offsets, batch.ptrs, batch.sizes
                )
            except UCMProxyError:
                self._worker_metadata.mark_failed(request_id)
                self._invalid_block_ids.update(
                    block_id
                    for plan in request.load_plans
                    for group in plan.vllm_blocks
                    for block_id in group.block_ids
                )
        # Synchronous load errors are returned through vLLM's invalid-block and
        # worker-metadata channels; aborting here would bypass those channels.

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMConnectorMetadata):
            raise TypeError(f"Unexpected connector metadata: {type(metadata).__name__}")
        if self.layout is None:
            raise RuntimeError("register_kv_caches must run before saving")
        batch = self.layout.build_dump_batches(metadata)
        self._proxy.dump(batch.block_ids, batch.offsets, batch.ptrs, batch.sizes)

    def build_connector_worker_meta(self) -> UCMWorkerMetadata | None:
        if not self._worker_metadata.load_failed_reqs:
            return None
        result = self._worker_metadata
        self._worker_metadata = UCMWorkerMetadata()
        return result

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        if self.dispatcher is None:
            return
        metadata = getattr(connector_output, "kv_connector_worker_meta", None)
        if not isinstance(metadata, UCMWorkerMetadata):
            return
        for request_id in metadata.load_failed_reqs:
            self.dispatcher.requests.pop(request_id, None)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        # Bulk v2 I/O is synchronous, so no transfer owns preempted blocks.
        return None

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

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
