"""Layout entry point: metadata, HBM views and storage templates.

One entry point, :func:`build_group_layouts`: walk every KV group of the
parsed spec and hand the runtime views to :class:`KVCacheGroupLayout`,
which flattens them into addressing columns. Bind the worker's native
per-layer specs and placement here; the shared group policy carries no
physical declarations. Transient and empty PP groups keep their native
group IDs but do not compile a physical layout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from .group import BlockAccess, BlockFirstView, KVCacheGroupLayout, TensorDescriptor
from .kv_cache import (
    UCMLayerSpec,
    UCMKVCacheGroupInfo,
    UCMKVCacheLayout,
    UCMKVCacheSpec,
    _concrete_specs,
    _layer_index,
    parse_kv_cache_config,
)
from .store_layout import GroupStoreLayout
from .view import ComponentView, LayerView, MemorySegment

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec


def _parse_descriptors(
    kv_cache_tensors: Sequence[object],
) -> dict[str, tuple[TensorDescriptor, int]]:
    """Index the worker's native placement declarations by registered owner."""
    declared_at = {}
    for entry in kv_cache_tensors:
        descriptor = TensorDescriptor(
            layers=tuple(entry.layers),
            offset=entry.offset,
            layer_stride=entry.layer_stride,
            block_stride=entry.block_stride,
        )
        for position, name in enumerate(descriptor.layers):
            declared_at[name] = (descriptor, position)
    return declared_at


def _storage_block_size(spec: "KVCacheSpec", device_type: str) -> int:
    """Physical attention entries per block, including native DCP replicas."""
    storage_block_size = int(spec.block_size // spec.tokens_per_state)
    if device_type == "npu":
        from vllm_ascend.core.kv_cache_interface import (
            AscendSFAIndexerCacheSpec,
            AscendSlidingWindowMLASpec,
        )

        if isinstance(spec, AscendSlidingWindowMLASpec):
            storage_block_size = spec.storage_block_size
        elif isinstance(spec, AscendSFAIndexerCacheSpec):
            storage_block_size *= spec.sfa_dcp_replicated_indexer_size
    return storage_block_size


def _attention_view_order(
    spec: "KVCacheSpec", device_type: str, layer_name: str
) -> Literal["BHNC", "BNHC"]:
    """Match the native worker's view ABI; physical ordering remains in strides."""
    if device_type != "npu":
        return "BHNC"
    from vllm import envs
    from vllm.v1.kv_cache_interface import (
        HiddenStateCacheSpec,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
    )

    # Ascend cache-only views keep upstream BHNC. MLA/SFA backends publish
    # BNHC independently of the ordinary GQA backend's layout setting.
    if isinstance(spec, HiddenStateCacheSpec) or "cache_only_layers" in layer_name:
        return "BHNC"
    if isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
        return "BNHC"
    return "BHNC" if envs.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND") else "BNHC"


def build_group_layouts(
    spec: "UCMKVCacheSpec",
    kv_cache_config: "KVCacheConfig",
    kv_caches: Mapping[str, "torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]"],
    *,
    num_hidden_layers: int,
) -> dict[int, KVCacheGroupLayout]:
    """Bind local native declarations and real views to each persistent group."""
    block_first_layout = False
    if spec.device_type in ("cuda", "cpu"):
        from vllm.v1.kv_cache_layout import KVCacheLayout

        native_layout = KVCacheLayout[kv_cache_config.kv_cache_layout]
        block_first_layout = (
            native_layout.is_block_outermost and native_layout.is_block_compact
        )

    declared_at = _parse_descriptors(kv_cache_config.kv_cache_tensors)
    layouts = {}
    for group in spec.groups:
        native_group = kv_cache_config.kv_cache_groups[group.group_id]
        if group.is_transient or not native_group.layer_names:
            continue
        layers = []
        for name, native_spec in _concrete_specs(native_group):
            storage_block_size = (
                1  # State is one indivisible checkpoint, not a token-row array.
                if group.is_state_snapshot
                else _storage_block_size(native_spec, spec.device_type)
            )
            descriptor, position = declared_at.get(name, (None, 0))
            layers.append(
                UCMLayerSpec(
                    layer_name=name,
                    layer_index=_layer_index(name, spec.device_type, num_hidden_layers),
                    kv_cache_spec=native_spec,
                    storage_block_size=storage_block_size,
                    num_blocks=kv_cache_config.num_blocks,
                    descriptor=descriptor,
                    descriptor_position=position,
                    attention_view_order=_attention_view_order(
                        native_spec, spec.device_type, name
                    ),
                )
            )
        layouts[group.group_id] = KVCacheGroupLayout(
            group,
            kv_caches,
            layers=layers,
            device_type=spec.device_type,
            block_first_layout=block_first_layout,
        )
    return layouts


__all__ = [
    "BlockAccess",
    "BlockFirstView",
    "ComponentView",
    "GroupStoreLayout",
    "KVCacheGroupLayout",
    "LayerView",
    "MemorySegment",
    "TensorDescriptor",
    "UCMLayerSpec",
    "UCMKVCacheGroupInfo",
    "UCMKVCacheLayout",
    "UCMKVCacheSpec",
    "build_group_layouts",
    "parse_kv_cache_config",
]
