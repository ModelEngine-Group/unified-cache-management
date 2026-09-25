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
    UCMProxyTask,
)
from .ucm_scheduler import (
    RequestHasher,
    UCMConnectorMetadata,
    UCMDispatcher,
)
from ucm.utils import Config

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
    """Resolve kv_connector_extra_config through the shared ucm.utils.Config.

    The UCM_CONFIG_FILE yaml and terminal-input forms resolve exactly like
    every other UCM connector; v2 adds no launch keys of its own.
    """
    return dict(Config(vllm_config.kv_transfer_config).get_config())


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
    """One v2 lifecycle facade for grouped caches."""

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
        rank = None if role == KVConnectorRole.SCHEDULER else _worker_rank(vllm_config)
        _dump_raw_kv_cache_config(kv_cache_config, rank)
        hasher = RequestHasher(vllm_config, 0)
        base_seed = hasher("UCM_HASH_SEED")

        self.context = UCMRuntimeContext.from_vllm_config(vllm_config, role, rank=rank)
        # The scheduler's representative group spec can hide DSV4's per-layer
        # C4/C128 ratios. Read the same model metadata on both sides. Ratios of
        # zero denote uncompressed layers, as in vLLM's DSV4 attention module.
        text_config = getattr(vllm_config.model_config, "hf_text_config", None)
        compress_ratios = getattr(text_config, "compress_ratios", ()) or ()
        num_layers = int(
            getattr(text_config, "num_hidden_layers", len(compress_ratios))
        )
        attention_tokens_per_state = {
            index: max(1, int(ratio))
            for index, ratio in enumerate(compress_ratios)
            if index < num_layers
        }
        self.spec: UCMKVCacheSpec = parse_kv_cache_config(
            kv_cache_config,
            scheduler_block_size=scheduler_block_size,
            ucm_cache_block_size=ucm_cache_block_size,
            device_type=self.context.device_type,
            attention_tokens_per_state=attention_tokens_per_state,
            num_hidden_layers=num_layers,
        )
        dtype = str(vllm_config.model_config.dtype).rsplit(".", 1)[-1]
        # -r2: block-major record layout and ndarray plans (records dumped
        # by earlier revisions are not byte-compatible).
        namespace = (
            f"{self.context.device_type}-{dtype}"
            f"-b{self.spec.scheduler_block_size}"
            f"-c{self.spec.ucm_cache_block_size}-r2"
        )
        self.use_layerwise = bool(launch_config.get("use_layerwise", False))
        if self.use_layerwise:
            # Layerwise copies payloads only; bulk Block First records may
            # additionally include slot padding. Keep file caches separate.
            namespace += "-layerwise"
        root = _storage_root(launch_config) / ".ucm-v2" / namespace
        self._proxy = UCMProxyAdapter(SimpleFileUCMProxy(root))
        self._layer_load_tasks: dict[int, list[tuple[str, UCMProxyTask]]] = {}
        self._dump_tasks: list[UCMProxyTask] = []
        self._saved_layer_names: set[str] = set()
        self._failed_load_reqs: set[str] = set()
        self._save_error: UCMProxyError | None = None
        self._save_complete = False
        self.layout: UCMKVCacheLayout | None = None
        self._worker_metadata = UCMWorkerMetadata()
        self._invalid_block_ids: set[int] = set()
        self.dispatcher = (
            UCMDispatcher(
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
            if role == KVConnectorRole.SCHEDULER
            else None
        )

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.dispatcher is not None
        result = self.dispatcher.lookup(request, num_computed_tokens)
        return result.external_hit_tokens, False

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
        # vLLM 0.26 SchedulerOutput is the single source of truth for complete
        # new/resumed block tables and cached-request deltas.
        return None

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]) -> None:
        if self.context.role != KVConnectorRole.WORKER:
            raise RuntimeError("KV cache registration is only available on worker")
        self.layout = UCMKVCacheLayout(self.spec, kv_caches)
        self._proxy.register_tensors(kv_caches)

    def _mark_load_failed(self, request_id: str, request: Any) -> None:
        self._failed_load_reqs.add(request_id)
        self._worker_metadata.mark_failed(request_id)
        self._invalid_block_ids.update(
            int(block_id)
            for plan in request.load_plans
            for blocks in plan.windows
            for block_id in blocks.tolist()
        )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        # No work from a previous step may outlive its KV block ownership.
        if self._layer_load_tasks or self._dump_tasks:
            raise RuntimeError("Previous UCM transfers have not been drained")
        self._saved_layer_names.clear()
        self._failed_load_reqs.clear()
        self._save_error = None
        self._save_complete = False
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        assert self.layout is not None
        for request_id, request in metadata.requests.items():
            request_metadata = UCMConnectorMetadata(requests={request_id: request})
            try:
                if self.use_layerwise:
                    for layer_id in self.layout.layer_names_by_id:
                        tasks = self._layer_load_tasks.setdefault(layer_id, [])
                        for batch in self.layout.build_load_transfers(
                            request_metadata, layer_id=layer_id
                        ):
                            tasks.append(
                                (request_id, self._proxy.enqueue("load", batch))
                            )
                else:
                    for batch in self.layout.build_load_transfers(request_metadata):
                        self._proxy.submit("load", batch)
            except UCMProxyError:
                self._mark_load_failed(request_id, request)
        if self.use_layerwise:
            # State layers have no Attention/MLA hook. They must be ready before
            # forward, even when they share a model layer index with attention.
            state_layer_ids = {
                layer.layer_index
                for group in self.spec.groups
                if group.is_state_snapshot
                for layer in group.layers
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
                key: value
                for key, value in metadata.requests.items()
                if key not in self._failed_load_reqs
            }
        )

    def _save_layer(self, layer_name: str, metadata: UCMConnectorMetadata) -> None:
        assert self.layout is not None
        if layer_name in self._saved_layer_names:
            return
        if self._save_error is not None:
            raise self._save_error
        try:
            for batch in self.layout.build_dump_transfers(
                metadata, layer_name=layer_name
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

    def wait_for_save(self) -> None:
        if not self.has_connector_metadata():
            return
        assert self.layout is not None
        if not self.use_layerwise:
            metadata = self._get_connector_metadata()
            assert isinstance(metadata, UCMConnectorMetadata)
            batches = self.layout.build_dump_transfers(metadata)
            for batch in batches:
                self._proxy.submit("dump", batch)
            for batch in batches:
                self._proxy.commit(batch.keys)
            return
        if self._save_complete:
            return
        # Drain loads whose hooks were skipped (e.g. no-forward execution).
        for layer_id in tuple(self._layer_load_tasks):
            self._wait_layer_load(layer_id)
        metadata = self._dump_metadata()
        try:
            for layer_name in self.layout.layer_id_by_name:
                self._save_layer(layer_name, metadata)
        except UCMProxyError as exc:
            self._save_error = exc
        # Always drain submitted writes, including after a later submission fails.
        for task in self._dump_tasks:
            try:
                self._proxy.wait(task)
            except UCMProxyError as exc:
                self._save_error = exc
        self._dump_tasks.clear()
        if self._save_error is not None:
            raise self._save_error
        # Keys come from plans: no full-layer pointer matrix or uniqueness scan.
        for request in metadata.requests.values():
            for plan in request.dump_plans:
                self._proxy.commit(plan.keys)
        self._save_complete = True

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
        assert self.dispatcher is not None
        metadata = getattr(connector_output, "kv_connector_worker_meta", None)
        if not isinstance(metadata, UCMWorkerMetadata):
            return
        for request_id in metadata.load_failed_reqs:
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
