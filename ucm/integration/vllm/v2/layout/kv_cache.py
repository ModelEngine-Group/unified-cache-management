"""KV-cache metadata and the layout aggregate, independent of dispatch plans.

Parse shared persistence policy from KVCacheConfig; worker registration binds
HBM group layouts and compiles each group's storage template. No request keys,
connector metadata or proxy transfers are interpreted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from .store_layout import GroupStoreLayout
from .view import LAYOUT_DEBUG, layout_debug

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheSpec

    from .group import KVCacheGroupLayout, TensorDescriptor


_SLIDING_KINDS = frozenset(
    (KVCacheSpecKind.SLIDING_WINDOW, KVCacheSpecKind.SLIDING_WINDOW_MLA)
)


@dataclass(frozen=True)
class UCMLayerSpec:
    """Worker declaration; storage_block_size counts cache entries (one for State)."""

    layer_name: str
    layer_index: int
    kv_cache_spec: "KVCacheSpec"
    storage_block_size: int
    num_blocks: int
    # The native placement declaration and this layer's position in it.
    descriptor: "TensorDescriptor | None" = None
    descriptor_position: int = 0
    # The worker backend's logical attention axes, before view normalization.
    attention_view_order: Literal["BHNC", "BNHC"] = "BHNC"


@dataclass(frozen=True)
class UCMKVCacheGroupSpec:
    """Shared persistence policy; tokens_per_block is tokens per block ID."""

    group_id: int
    tokens_per_block: int
    kinds: frozenset[KVCacheSpecKind]
    tail_tokens: int | None = None
    is_eagle_group: bool = False
    # Native transfer/cacheability flags select hash-addressed store groups.
    is_persistent: bool = True

    @property
    def is_attention(self) -> bool:
        return self.is_persistent and KVCacheSpecKind.MAMBA not in self.kinds

    @property
    def is_sliding_window(self) -> bool:
        return not self.kinds.isdisjoint(_SLIDING_KINDS)

    @property
    def is_state_snapshot(self) -> bool:
        return KVCacheSpecKind.MAMBA in self.kinds


@dataclass(frozen=True)
class UCMKVCacheSpec:
    """UCM policy sizes, all in tokens.

    scheduler_block_size is what vLLM's
    ``resolve_kv_cache_block_sizes`` reports for this engine -- the
    token-alignment invariant of the resident KV pool (single group:
    ``cache_config.block_size``; multiple groups: LCM).  tokens_per_chunk
    counts tokens per record and per shared hash-chain key, defining the
    restore boundaries. An explicit size must be a multiple of
    scheduler_block_size; State defaults to that common alignment, while
    attention-only models default to their smallest FA block span.
    """

    groups: tuple[UCMKVCacheGroupSpec, ...]
    scheduler_block_size: int
    tokens_per_chunk: int
    device_type: str
    num_attn_module: int = 1
    # Original native group IDs index this shared scheduler/worker contract.
    blocks_per_chunk: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "blocks_per_chunk",
            tuple(
                _group_blocks_per_chunk(group, self.tokens_per_chunk)
                for group in self.groups
            ),
        )

    @property
    def state_groups(self) -> tuple[UCMKVCacheGroupSpec, ...]:
        return tuple(group for group in self.groups if group.is_state_snapshot)

    @property
    def fa_groups(self) -> tuple[UCMKVCacheGroupSpec, ...]:
        return tuple(
            group
            for group in self.groups
            if group.is_attention and not group.is_sliding_window
        )

    @property
    def wa_groups(self) -> tuple[UCMKVCacheGroupSpec, ...]:
        return tuple(group for group in self.groups if group.is_sliding_window)


def _group_blocks_per_chunk(group: UCMKVCacheGroupSpec, tokens_per_chunk: int) -> int:
    """vLLM blocks one ucm key's window spans (0 = the group stores nothing).

    Non-aligned tails retain their partial first block. A State snapshot is
    one indivisible checkpoint page; full attention spans the chunk's tokens,
    using several native blocks or a subrange of the containing block.
    """

    token_block = group.tokens_per_block
    if not group.is_persistent:
        return 0
    if group.is_state_snapshot:
        return 1
    if group.is_sliding_window:
        tail = group.tail_tokens or 0
        return (tail + token_block - 1) // token_block if tail > 0 else 0
    return (tokens_per_chunk + token_block - 1) // token_block


def _concrete_specs(
    group: "KVCacheGroupSpec",
) -> tuple[tuple[str, "KVCacheSpec"], ...]:
    """Expand a native group into ordered (layer_name, per-layer spec) pairs."""
    group_spec = group.kv_cache_spec
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return tuple(
            (name, group_spec.kv_cache_specs[name]) for name in group.layer_names
        )
    return tuple((name, group_spec) for name in group.layer_names)


def _layer_index(
    name: str, device_type: str, num_hidden_layers: int, num_attn_module: int = 1
) -> int:
    """Ascend MTP uses local IDs; upstream uses global IDs."""
    index = extract_layer_index(name, num_attn_module)
    if device_type.lower() == "npu" and "mtp" in name.split("."):
        if index < num_hidden_layers:
            index += num_hidden_layers
    return index


def _member_specs(group_spec: "KVCacheSpec") -> tuple["KVCacheSpec", ...]:
    """Declared members remain available even on an empty PP worker group."""
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return tuple(group_spec.kv_cache_specs.values())
    return (group_spec,)


def _classify(specs: Sequence["KVCacheSpec"]) -> frozenset[KVCacheSpecKind]:
    spec_kinds = tuple(get_kv_cache_spec_kind(spec) for spec in specs)
    unknown = tuple(
        type(spec).__qualname__
        for spec, kind in zip(specs, spec_kinds)
        if kind
        not in {
            KVCacheSpecKind.FULL_ATTENTION,
            KVCacheSpecKind.MLA_ATTENTION,
            KVCacheSpecKind.SINK_FULL_ATTENTION,
            KVCacheSpecKind.SLIDING_WINDOW,
            KVCacheSpecKind.SLIDING_WINDOW_MLA,
            KVCacheSpecKind.MAMBA,
        }
    )
    if unknown:
        raise TypeError(f"Unsupported KV cache spec types: {sorted(set(unknown))}")

    kinds = frozenset(spec_kinds)
    if KVCacheSpecKind.MAMBA in kinds and len(kinds) != 1:
        raise TypeError(f"Mamba and attention specs cannot share a KV group: {kinds}")
    return kinds


def _sliding_tail_tokens(
    group: "KVCacheGroupSpec",
    compressor_ratios: Mapping[int, int],
    device_type: str,
    num_hidden_layers: int,
    num_attn_module: int,
) -> int:
    """Resolve logical history retention; only legacy compressors subtract a pool."""
    # Empty PP groups keep the global UniformType map. A representative
    # sliding spec without owner names cannot identify compressor tails.
    pairs = _concrete_specs(group)
    if not pairs and isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        pairs = tuple(group.kv_cache_spec.kv_cache_specs.items())
    if not pairs:
        raise NotImplementedError("Empty PP sliding group has no owner tail semantics")
    tails = set()
    for name, spec in pairs:
        tail = spec.sliding_window
        if name.endswith(".compressor.state_cache"):
            layer_index = _layer_index(
                name, device_type, num_hidden_layers, num_attn_module
            )
            tail -= compressor_ratios[layer_index]
        if tail < 0:
            raise ValueError(f"Negative sliding tail for {name}: {tail}")
        tails.add(tail)
    if len(tails) != 1:
        raise ValueError(f"Group has inconsistent tail sizes {sorted(tails)}")
    return tails.pop()


def _select_tokens_per_chunk(
    groups: Sequence[UCMKVCacheGroupSpec],
    scheduler_block_size: int,
    requested: int | None,
) -> int:
    state_groups = tuple(group for group in groups if group.is_state_snapshot)
    fa_groups = tuple(
        group for group in groups if group.is_attention and not group.is_sliding_window
    )
    if state_groups and not fa_groups:
        raise ValueError("State snapshots require a full-attention prefix")
    if requested is not None:
        if requested <= 0 or requested % scheduler_block_size:
            raise ValueError(
                "tokens_per_chunk must be a positive multiple of scheduler_block_size"
            )
        tokens_per_chunk = requested
    elif state_groups:
        # Complete shared restore boundaries, even if FA/State spans differ.
        tokens_per_chunk = scheduler_block_size
    else:
        tokens_per_chunk = min(
            (group.tokens_per_block for group in fa_groups),
            default=scheduler_block_size,
        )

    for group in groups:
        if not group.is_persistent:
            continue
        if group.is_state_snapshot or (group.is_sliding_window and group.tail_tokens):
            if tokens_per_chunk % group.tokens_per_block:
                raise ValueError(
                    f"Group {group.group_id} block span {group.tokens_per_block} "
                    f"must divide tokens_per_chunk {tokens_per_chunk}"
                )
        elif not group.is_sliding_window:
            if (
                tokens_per_chunk % group.tokens_per_block
                and group.tokens_per_block % tokens_per_chunk
            ):
                raise ValueError(
                    f"FA group {group.group_id} block span {group.tokens_per_block} "
                    f"and tokens_per_chunk {tokens_per_chunk} must divide each other"
                )
    return tokens_per_chunk


def parse_kv_cache_config(
    kv_cache_config: "KVCacheConfig",
    *,
    scheduler_block_size: int,
    num_hidden_layers: int,
    num_attn_module: int = 1,
    tokens_per_chunk: int | None = None,
    device_type: str = "npu",
    compressor_tokens_per_state: Mapping[int, int] | None = None,
) -> UCMKVCacheSpec:
    """Compile group persistence and restore policy on scheduler and worker.

    No physical row counts, placement descriptors or layer layouts are parsed.
    Compressor ratios only determine the legacy DSV4 history tail. Native group
    positions remain block-table IDs, including excluded and empty PP groups.
    Native prefix_cacheable_group_ids selects the groups UCM stores; scratch
    reconstruction and its resume boundaries belong to the engine.
    """
    device_type = device_type.lower()
    compressor_ratios = compressor_tokens_per_state or {}
    raw_groups = kv_cache_config.kv_cache_groups
    if not raw_groups:
        raise ValueError("kv_cache_config.kv_cache_groups must not be empty")
    groups = []
    persistent_ids = set(kv_cache_config.prefix_cacheable_group_ids)
    for group_id, group in enumerate(raw_groups):
        is_persistent = group_id in persistent_ids
        specs = _member_specs(group.kv_cache_spec) if is_persistent else ()
        kinds = _classify(specs)
        if KVCacheSpecKind.MAMBA in kinds and any(
            spec.mamba_cache_mode != "align" for spec in specs
        ):
            raise ValueError("connector v2 requires mamba_cache_mode='align'")
        sliding = not kinds.isdisjoint(_SLIDING_KINDS)
        if not is_persistent:
            tail_tokens = 0
        elif sliding:
            tail_tokens = _sliding_tail_tokens(
                group,
                compressor_ratios,
                device_type,
                num_hidden_layers,
                num_attn_module,
            )
        else:
            tail_tokens = None
        groups.append(
            UCMKVCacheGroupSpec(
                group_id=group_id,
                tokens_per_block=group.kv_cache_spec.block_size,
                kinds=kinds,
                tail_tokens=tail_tokens,
                is_eagle_group=group.is_eagle_group,
                is_persistent=is_persistent,
            )
        )
    tokens_per_chunk = _select_tokens_per_chunk(
        groups, scheduler_block_size, tokens_per_chunk
    )
    result = UCMKVCacheSpec(
        tuple(groups),
        scheduler_block_size,
        tokens_per_chunk,
        device_type,
        num_attn_module,
    )
    if LAYOUT_DEBUG:
        for group in result.groups:
            layout_debug(
                f"policy group={group.group_id} "
                f"kinds={sorted(kind.value for kind in group.kinds)} "
                f"persistent={group.is_persistent} token_block={group.tokens_per_block} "
                f"tail={group.tail_tokens} blocks_per_chunk={result.blocks_per_chunk[group.group_id]}"
            )
        layout_debug(
            f"spec scheduler_block={scheduler_block_size} "
            f"tokens_per_chunk={tokens_per_chunk} device={device_type}"
        )
    return result


class UCMKVCacheLayout:
    """Shared policy, HBM layouts and storage templates compiled at registration."""

    def __init__(
        self,
        spec: UCMKVCacheSpec,
        kv_caches: Mapping[
            str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
        ],
        *,
        kv_cache_config: "KVCacheConfig",
        num_hidden_layers: int,
        use_layerwise: bool = False,
    ) -> None:
        from . import build_group_layouts

        self.spec = spec
        self.group_layouts: Mapping[int, "KVCacheGroupLayout"] = build_group_layouts(
            spec, kv_cache_config, kv_caches, num_hidden_layers=num_hidden_layers
        )
        self.store_layouts = {
            group.group_id: GroupStoreLayout.build(
                spec,
                self.group_layouts[group.group_id],
                use_layerwise=use_layerwise,
            )
            for group in spec.groups
            if group.group_id in self.group_layouts
        }
        # A callback may name only attention, while indexer and other caches
        # of the same model layer have distinct registered names/groups.
        self.layer_id_by_name = {
            layer.layer_name: layer.layer_index
            for layout in self.group_layouts.values()
            for layer in layout.layer_specs
        }
        names_by_id: dict[int, list[str]] = {}
        for name, layer_id in self.layer_id_by_name.items():
            names_by_id.setdefault(layer_id, []).append(name)
        self.layer_names_by_id = {
            layer_id: frozenset(names) for layer_id, names in names_by_id.items()
        }
