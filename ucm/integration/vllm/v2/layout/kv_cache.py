"""KV-cache metadata and the layout aggregate, independent of dispatch plans.

Parse shared persistence policy from KVCacheConfig; worker registration binds
HBM group layouts and compiles each group's storage template. No request keys,
connector metadata or proxy transfers are interpreted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KpoolTailSpec,
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
class UCMKVCacheGroupInfo:
    """Shared persistence policy; token_block_size is tokens per block ID."""

    group_id: int
    token_block_size: int
    kinds: frozenset[KVCacheSpecKind]
    tail_tokens: int | None = None
    # vLLM blocks one ucm key's window spans (0 = the group stores
    # nothing); the parser stamps it once the cache block size is known.
    tail_blocks: int = 0
    is_eagle_group: bool = False
    # Scratch remains in HBM; UCM reconstructs it at complete-pool boundaries.
    is_transient: bool = False

    @property
    def is_attention(self) -> bool:
        return not self.is_transient and KVCacheSpecKind.MAMBA not in self.kinds

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
    ``cache_config.block_size``; multiple groups: LCM).  ucm_cache_block_size
    counts tokens per record and per shared hash-chain key, defining the
    restore boundaries. An explicit size must be a multiple of
    scheduler_block_size; State defaults to that common alignment, while
    attention-only models default to their smallest FA block span.
    """

    groups: tuple[UCMKVCacheGroupInfo, ...]
    scheduler_block_size: int
    ucm_cache_block_size: int
    device_type: str

    @property
    def state_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_state_snapshot)

    @property
    def fa_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(
            group
            for group in self.groups
            if group.is_attention and not group.is_sliding_window
        )

    @property
    def wa_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_sliding_window)


def _group_tail_blocks(group: UCMKVCacheGroupInfo, ucm_block_size: int) -> int:
    """vLLM blocks one ucm key's window spans (0 = the group stores nothing).

    HMA's ``tail_blocks`` with the one generalization v2 needs: a tail
    that does not divide the block still keeps its partial head block
    (ceil, not HMA's floor).  A state snapshot is one indivisible
    checkpoint page; a full-attention key spans the whole ucm block
    (several blocks when the UCM block is larger than a native block,
    the containing block otherwise).
    """

    token_block = group.token_block_size
    if group.is_transient:
        return 0
    if group.is_state_snapshot:
        return 1
    if group.is_sliding_window:
        tail = group.tail_tokens or 0
        return (tail + token_block - 1) // token_block if tail > 0 else 0
    return (ucm_block_size + token_block - 1) // token_block


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


def _layer_index(name: str, device_type: str, num_hidden_layers: int) -> int:
    """Ascend MTP uses local IDs; upstream uses global IDs."""
    index = extract_layer_index(name)
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
        if kind == KVCacheSpecKind.UNKNOWN
    )
    if unknown:
        raise TypeError(f"Unsupported KV cache spec types: {sorted(set(unknown))}")

    kinds = frozenset(spec_kinds)
    if KVCacheSpecKind.MAMBA in kinds and len(kinds) != 1:
        raise TypeError(f"Mamba and attention specs cannot share a KV group: {kinds}")
    return kinds


def _transient_alignment(
    specs: Sequence["KVCacheSpec"],
    model_type: str,
    indexer_ratio: int | None,
    device_type: str,
) -> int | None:
    """Return the restore alignment for omitted scratch; native models own pairing."""
    if all(isinstance(spec, CircularBufferSpec) for spec in specs):
        if model_type not in ("qwen4_exp", "qwen4_exp_text"):
            raise NotImplementedError(
                f"No UCM restore policy for CircularBufferSpec in {model_type}"
            )
        return indexer_ratio
    if model_type in ("glm5_next", "glm5_next_text"):
        tail_types = (KpoolTailSpec,)
        if device_type == "npu":
            from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec

            tail_types = (KpoolTailSpec, AscendIndexerKPoolTailSpec)
        if all(isinstance(spec, tail_types) for spec in specs):
            return indexer_ratio
    return None


def _sliding_tail_tokens(
    group: "KVCacheGroupSpec",
    compressor_ratios: Mapping[int, int],
    device_type: str,
    num_hidden_layers: int,
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
            layer_index = _layer_index(name, device_type, num_hidden_layers)
            tail -= compressor_ratios[layer_index]
        if tail < 0:
            raise ValueError(f"Negative sliding tail for {name}: {tail}")
        tails.add(tail)
    if len(tails) != 1:
        raise ValueError(f"Group has inconsistent tail sizes {sorted(tails)}")
    return tails.pop()


def _select_cache_block_size(
    groups: Sequence[UCMKVCacheGroupInfo],
    scheduler_block_size: int,
    requested: int | None,
    transient_alignments: Mapping[int, int],
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
                "ucm_cache_block_size must be a positive multiple of scheduler_block_size"
            )
        ucm_block_size = requested
    elif state_groups:
        # Complete shared restore boundaries, even if FA/State spans differ.
        ucm_block_size = scheduler_block_size
    else:
        ucm_block_size = min(
            (group.token_block_size for group in fa_groups),
            default=scheduler_block_size,
        )

    for group in groups:
        if group.is_transient:
            alignment = transient_alignments[group.group_id]
            if ucm_block_size % alignment:
                raise ValueError(
                    f"Transient group {group.group_id} requires complete pools "
                    f"of {alignment} tokens"
                )
        elif group.is_state_snapshot or (group.is_sliding_window and group.tail_tokens):
            if ucm_block_size % group.token_block_size:
                raise ValueError(
                    f"Group {group.group_id} block span {group.token_block_size} "
                    f"must divide UCM block size {ucm_block_size}"
                )
        elif not group.is_sliding_window:
            if (
                ucm_block_size % group.token_block_size
                and group.token_block_size % ucm_block_size
            ):
                raise ValueError(
                    f"FA group {group.group_id} block span {group.token_block_size} "
                    f"and UCM block size {ucm_block_size} must divide each other"
                )
    return ucm_block_size


def parse_kv_cache_config(
    kv_cache_config: "KVCacheConfig",
    *,
    scheduler_block_size: int,
    num_hidden_layers: int,
    indexer_tokens_per_state: int | None,
    ucm_cache_block_size: int | None = None,
    device_type: str = "npu",
    compressor_tokens_per_state: Mapping[int, int] | None = None,
    model_type: str = "",
) -> UCMKVCacheSpec:
    """Compile group persistence and restore policy on scheduler and worker.

    No physical row counts, placement descriptors or layer layouts are parsed.
    Compressor ratios only determine the legacy DSV4 history tail. Native group
    positions remain block-table IDs, including transient and empty PP groups.
    """
    device_type = device_type.lower()
    compressor_ratios = compressor_tokens_per_state or {}
    raw_groups = kv_cache_config.kv_cache_groups
    if not raw_groups:
        raise ValueError("kv_cache_config.kv_cache_groups must not be empty")
    groups = []
    # group_id -> compression-group token span; UCM restores at complete pools.
    transient_alignments = {}
    for group_id, group in enumerate(raw_groups):
        specs = _member_specs(group.kv_cache_spec)
        alignment = _transient_alignment(
            specs, model_type, indexer_tokens_per_state, device_type
        )
        if alignment is not None:
            transient_alignments[group_id] = alignment
        # A recognized scratch ring has no FA/WA/State persistence kind.
        kinds = (
            frozenset()
            if alignment is not None and isinstance(specs[0], CircularBufferSpec)
            else _classify(specs)
        )
        if KVCacheSpecKind.MAMBA in kinds and any(
            spec.mamba_cache_mode != "align" for spec in specs
        ):
            raise ValueError("connector v2 requires mamba_cache_mode='align'")
        transient = alignment is not None
        sliding = not kinds.isdisjoint(_SLIDING_KINDS)
        if transient:
            tail_tokens = 0
        elif sliding:
            tail_tokens = _sliding_tail_tokens(
                group, compressor_ratios, device_type, num_hidden_layers
            )
        else:
            tail_tokens = None
        groups.append(
            UCMKVCacheGroupInfo(
                group_id=group_id,
                token_block_size=group.kv_cache_spec.block_size,
                kinds=kinds,
                tail_tokens=tail_tokens,
                is_eagle_group=group.is_eagle_group,
                is_transient=transient,
            )
        )
    ucm_block_size = _select_cache_block_size(
        groups, scheduler_block_size, ucm_cache_block_size, transient_alignments
    )
    result = UCMKVCacheSpec(
        tuple(
            replace(group, tail_blocks=_group_tail_blocks(group, ucm_block_size))
            for group in groups
        ),
        scheduler_block_size,
        ucm_block_size,
        device_type,
    )
    if LAYOUT_DEBUG:
        for group in result.groups:
            layout_debug(
                f"policy group={group.group_id} "
                f"kinds={sorted(kind.value for kind in group.kinds)} "
                f"transient={group.is_transient} token_block={group.token_block_size} "
                f"tail={group.tail_tokens} tail_blocks={group.tail_blocks}"
            )
        layout_debug(
            f"spec scheduler_block={scheduler_block_size} "
            f"ucm_cache_block={ucm_block_size} device={device_type}"
        )
    return result


class UCMKVCacheLayout:
    """Shared policy, HBM layouts and storage templates compiled at registration."""

    def __init__(
        self,
        spec: UCMKVCacheSpec,
        kv_caches: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
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
                group, self.group_layouts[group.group_id], spec.ucm_cache_block_size,
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
            for layer in layout.layers
        }
        names_by_id: dict[int, list[str]] = {}
        for name, layer_id in self.layer_id_by_name.items():
            names_by_id.setdefault(layer_id, []).append(name)
        self.layer_names_by_id = {
            layer_id: frozenset(names) for layer_id, names in names_by_id.items()
        }

