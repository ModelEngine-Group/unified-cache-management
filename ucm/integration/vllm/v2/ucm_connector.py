"""Isolated v2 connector facade.

The production connector registration deliberately continues to point at the
legacy module.  This facade wires the v2 components for contract and DT work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .ucm_kv_cache import UCMKVCacheLayout, UCMKVCacheSpec, parse_kv_cache_config
from .ucm_proxy import UCMProxy, UCMProxyAdapter, UCMProxyBatch, UCMProxyError
from .ucm_scheduler import (
    RequestHasher,
    UCMConnectorMetadata,
    UCMDispatcher,
    UCMLookupCoordinator,
)


@dataclass(frozen=True)
class UCMRuntimeContext:
    role: Literal["scheduler", "worker"]
    device: str | None
    engine_id: str | None
    rank: int | None
    world_size: int

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: Any,
        role: Literal["scheduler", "worker"],
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
            device=getattr(vllm_config, "device", None),
            engine_id=getattr(vllm_config, "instance_id", None),
            rank=rank,
            world_size=world_size,
        )


@dataclass
class UCMWorkerMetadata:
    load_failed_reqs: set[str] = field(default_factory=set)

    def mark_failed(self, request_id: str) -> None:
        self.load_failed_reqs.add(request_id)

    def aggregate(self, other: Any) -> "UCMWorkerMetadata":
        if not isinstance(other, UCMWorkerMetadata):
            raise TypeError(f"Cannot aggregate {type(other).__name__}")
        self.load_failed_reqs.update(other.load_failed_reqs)
        return self


class UCMConnector:
    """One v2 lifecycle facade shared by direct, hybrid, and DSV4 policies."""

    def __init__(
        self,
        *,
        vllm_config: Any,
        kv_cache_config: Any,
        proxy: UCMProxy,
        role: Literal["scheduler", "worker"],
        scheduler_block_size: int,
        base_seed: bytes,
        rank: int | None = None,
        chunk_size: int | None = None,
        load_threshold_tokens: int = 0,
    ) -> None:
        self.context = UCMRuntimeContext.from_vllm_config(
            vllm_config, role, rank=rank
        )
        self.spec: UCMKVCacheSpec = parse_kv_cache_config(
            kv_cache_config,
            scheduler_block_size=scheduler_block_size,
            chunk_size=chunk_size,
        )
        self.proxy = UCMProxyAdapter(proxy)
        self.metadata: UCMConnectorMetadata | None = None
        self.layout: UCMKVCacheLayout | None = None
        self.worker_metadata = UCMWorkerMetadata()
        self._invalid_block_ids: set[int] = set()
        self.dispatcher = UCMDispatcher(self.spec) if role == "scheduler" else None
        self.lookup_coordinator = (
            UCMLookupCoordinator(
                self.spec,
                self.proxy,
                # Scheduler always creates logical rank-0 keys. Worker-side
                # rank scoping is a separate physical-data policy.
                RequestHasher(vllm_config, 0),
                base_seed,
                load_threshold_tokens=load_threshold_tokens,
            )
            if role == "scheduler"
            else None
        )

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        if self.lookup_coordinator is None or self.dispatcher is None:
            raise RuntimeError("lookup is only available on the scheduler role")
        result = self.lookup_coordinator.lookup(request, num_computed_tokens)
        self.dispatcher.record_lookup(request, num_computed_tokens, result)
        return result.external_hit_tokens, False

    def build_connector_meta(self, scheduler_output: Any) -> UCMConnectorMetadata:
        if self.dispatcher is None:
            raise RuntimeError("dispatch is only available on the scheduler role")
        return self.dispatcher.build_from_scheduler_output(scheduler_output)

    def register_kv_caches(self, kv_caches: Any, *, num_blocks: int) -> None:
        if self.context.role != "worker":
            raise RuntimeError("KV cache registration is only available on worker")
        self.layout = UCMKVCacheLayout(self.spec, kv_caches, num_blocks=num_blocks)
        self.proxy.register_tensors(kv_caches)

    def bind_connector_metadata(self, metadata: UCMConnectorMetadata) -> None:
        self.metadata = metadata

    def clear_connector_metadata(self) -> None:
        self.metadata = None

    def has_connector_metadata(self) -> bool:
        return self.metadata is not None

    @staticmethod
    def _call_proxy(adapter_method: Any, batch: UCMProxyBatch) -> None:
        adapter_method(batch.block_ids, batch.offsets, batch.ptrs, batch.sizes)

    def start_load_kv(self, _forward_context: Any = None) -> None:
        if self.metadata is None:
            return
        if self.layout is None:
            raise RuntimeError("register_kv_caches must run before loading")
        batch = self.layout.build_load_batches(self.metadata)
        try:
            self._call_proxy(self.proxy.load, batch)
        except UCMProxyError:
            self.worker_metadata.load_failed_reqs.update(self.metadata.requests)
            self._invalid_block_ids.update(
                block_id
                for request in self.metadata.requests.values()
                for plan in request.load_plans
                for group in plan.vllm_blocks
                for block_id in group.block_ids
            )
            raise

    def wait_for_layer_load(self, _layer_name: str) -> None:
        return None

    def save_kv_layer(self, _layer_name: str, *_args: Any, **_kwargs: Any) -> None:
        return None

    def wait_for_save(self) -> None:
        if self.metadata is None:
            return
        if self.layout is None:
            raise RuntimeError("register_kv_caches must run before saving")
        batch = self.layout.build_dump_batches(self.metadata)
        self._call_proxy(self.proxy.dump, batch)

    def build_connector_worker_meta(self) -> UCMWorkerMetadata | None:
        if not self.worker_metadata.load_failed_reqs:
            return None
        result = self.worker_metadata
        self.worker_metadata = UCMWorkerMetadata()
        return result

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    def update_connector_output(self, connector_output: Any) -> None:
        if self.dispatcher is None:
            return
        metadata = getattr(connector_output, "kv_connector_worker_meta", None)
        if not isinstance(metadata, UCMWorkerMetadata):
            return
        for request_id in metadata.load_failed_reqs:
            self.dispatcher.requests.pop(request_id, None)


__all__ = [
    "UCMConnector",
    "UCMConnectorMetadata",
    "UCMRuntimeContext",
    "UCMWorkerMetadata",
]
