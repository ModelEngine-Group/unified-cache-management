"""KV-cache metadata and the layout aggregate, independent of dispatch plans.

Parse shared persistence policy from KVCacheConfig; worker registration binds
HBM group layouts and compiles each group's storage template. No UCM chunk ids,
connector metadata or proxy transfers are interpreted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import lcm
from typing import TYPE_CHECKING, Any, Literal

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from ucm.logger import init_logger

from .group import GroupLayoutMetadata
from .store_layout import GroupStoreLayout

logger = init_logger(__name__)

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheSpec

    from ..runtime import ParallelRank, ParallelTopology
    from .group import KVCacheGroupLayout, TensorDescriptor
    from .store_layout import PPStageMetadata


class RestorePolicy(Enum):
    """How a stored group must be restored, derived from its native spec."""

    FA = "FA"  # full-attention groups
    WA = "WA"  # sliding-window groups that store a window tail
    STATE = "State"  # mamba state snapshot groups

    @property
    def tag(self) -> bytes:
        return _RESTORE_POLICY_TAGS[self]


_RESTORE_POLICY_TAGS = {
    RestorePolicy.FA: b"\x00",
    RestorePolicy.WA: b"\x01",
    RestorePolicy.STATE: b"\x02",
}


_SLIDING_KINDS = frozenset(
    (KVCacheSpecKind.SLIDING_WINDOW, KVCacheSpecKind.SLIDING_WINDOW_MLA)
)
_SUPPORTED_KINDS = frozenset(
    (
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.MLA_ATTENTION,
        KVCacheSpecKind.SINK_FULL_ATTENTION,
        KVCacheSpecKind.SLIDING_WINDOW,
        KVCacheSpecKind.SLIDING_WINDOW_MLA,
        KVCacheSpecKind.MAMBA,
    )
)
_MLA_KINDS = frozenset(
    (KVCacheSpecKind.MLA_ATTENTION, KVCacheSpecKind.SLIDING_WINDOW_MLA)
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
    """Native local block capacity and its effective logical token span."""

    group_id: int
    block_size: int
    kinds: frozenset[KVCacheSpecKind]
    # SWA history length; None for groups without tail semantics.
    tail_tokens: int | None = None
    is_eagle_group: bool = False
    # Only groups vLLM lists in prefix_cacheable_group_ids are stored.
    is_persistent: bool = True
    # True when this group actually uses DCP: one block then covers
    # block_size * dcp_size logical tokens, following the scheduler helper.
    dcp_sharded: bool = True
    dcp_size: int = 1

    @property
    def block_token_span(self) -> int:
        return self.block_size * (self.dcp_size if self.dcp_sharded else 1)

    @property
    def is_attention(self) -> bool:
        return self.is_persistent and KVCacheSpecKind.MAMBA not in self.kinds

    @property
    def is_sliding_window(self) -> bool:
        return not self.kinds.isdisjoint(_SLIDING_KINDS)

    @property
    def is_state_snapshot(self) -> bool:
        return KVCacheSpecKind.MAMBA in self.kinds

    @property
    def restore_policy(self) -> RestorePolicy | None:
        """Normalize native kinds into UCM restore semantics."""
        if not self.is_persistent:
            return None
        if self.is_state_snapshot:
            return RestorePolicy.STATE
        if self.is_sliding_window:
            return RestorePolicy.WA if (self.tail_tokens or 0) > 0 else None
        return RestorePolicy.FA


@dataclass(frozen=True)
class UCMKVCacheSpec:
    """Stored groups and the global logical token span of one UCM object.

    Default chunk_size is the LCM of stored groups' effective block spans.
    Explicit non-DCP FA slices may divide a block and must align to entries.
    Engine scheduler granularity is a parser diagnostic, not stored metadata.
    """

    groups: tuple[UCMKVCacheGroupSpec, ...]
    chunk_size: int
    device_type: str
    # Attention modules per decoder layer, for layer ids (Longcat has 2).
    num_attn_module: int = 1
    # vLLM blocks one UCM chunk's window spans in each group, indexed by the
    # native group ID (0 = the group stores nothing). Scheduler and worker
    # share it.
    blocks_per_chunk: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "blocks_per_chunk",
            tuple(
                _group_blocks_per_chunk(group, self.chunk_size) for group in self.groups
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


def _group_blocks_per_chunk(group: UCMKVCacheGroupSpec, chunk_size: int) -> int:
    """vLLM blocks one UCM chunk's window spans (0 = the group stores nothing).

    HMA's ``tail_blocks`` with the one generalization v2 needs: a tail
    that does not divide the block still keeps its partial head block
    (ceil, not HMA's floor).  A state snapshot is one indivisible
    checkpoint page; a full-attention UCM chunk spans every vLLM block it
    covers, or one slice of a vLLM block when it is smaller.
    """

    tokens_per_block = group.block_token_span
    if group.is_state_snapshot:
        return 1
    elif group.is_sliding_window:
        tail_tokens = group.tail_tokens or 0
        return (
            (tail_tokens + tokens_per_block - 1) // tokens_per_block
            if tail_tokens > 0
            else 0
        )
    elif group.is_persistent:
        return (chunk_size + tokens_per_block - 1) // tokens_per_block
    else:
        # A group that is not stored has no kind and spans no blocks.
        return 0


def _concrete_specs(
    group: "KVCacheGroupSpec",
) -> tuple[tuple[str, "KVCacheSpec"], ...]:
    """Expand a native group into ordered (layer_name, per-layer spec) pairs.

    vLLM keeps only this worker's layers in a UniformType map, but an empty
    PP group keeps the map of all stages.
    """
    group_spec = group.kv_cache_spec
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return tuple(group_spec.kv_cache_specs.items())
    else:
        return tuple((layer_name, group_spec) for layer_name in group.layer_names)


def _layer_index(
    layer_name: str, device_type: str, num_hidden_layers: int, num_attn_module: int = 1
) -> int:
    """Ascend MTP uses local IDs; upstream uses global IDs."""
    layer_index = extract_layer_index(layer_name, num_attn_module)
    if device_type.lower() == "npu" and "mtp" in layer_name.split("."):
        if layer_index < num_hidden_layers:
            layer_index += num_hidden_layers
    return layer_index


def _member_specs(group_spec: "KVCacheSpec") -> tuple["KVCacheSpec", ...]:
    """Declared members remain available even on an empty PP worker group."""
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return tuple(group_spec.kv_cache_specs.values())
    return (group_spec,)


def _spec_tp_shards(spec: KVCacheSpec, tp_size: int) -> int:
    """Resolve the native TP declaration to an actual shard count."""
    if (
        getattr(spec, "tp_replicated", False)
        or get_kv_cache_spec_kind(spec) in _MLA_KINDS
    ):
        return 1
    max_tp_shards = getattr(spec, "max_tp_shards", None)
    return tp_size if max_tp_shards is None else min(max_tp_shards, tp_size)


def _classify(specs: Sequence["KVCacheSpec"]) -> frozenset[KVCacheSpecKind]:
    spec_kinds = tuple(get_kv_cache_spec_kind(spec) for spec in specs)
    unknown_spec_types = tuple(
        type(spec).__qualname__
        for spec, kind in zip(specs, spec_kinds)
        if kind not in _SUPPORTED_KINDS
    )
    if unknown_spec_types:
        raise TypeError(
            f"Unsupported KV cache spec types: {sorted(set(unknown_spec_types))}"
        )

    kinds = frozenset(spec_kinds)
    if KVCacheSpecKind.MAMBA in kinds and len(kinds) != 1:
        raise TypeError(f"Mamba and attention specs cannot share a KV group: {kinds}")
    return kinds


def _sliding_tail_tokens(
    group: "KVCacheGroupSpec",
    hf_text_config: Any,
    device_type: str,
    num_attn_module: int,
) -> int:
    """Resolve logical history retention; only legacy compressors subtract a pool."""
    # The parser has already rejected empty persistent sliding groups.
    layer_specs = _concrete_specs(group)
    tail_sizes = set()
    for layer_name, spec in layer_specs:
        tail_tokens = spec.sliding_window
        if layer_name.endswith(".compressor.state_cache"):
            # DSV4: a restore boundary closes every compression group, so only
            # the overlap of the next group (window - ratio) is read again.
            layer_index = _layer_index(
                layer_name,
                device_type,
                hf_text_config.num_hidden_layers,
                num_attn_module,
            )
            tail_tokens -= max(1, int(hf_text_config.compress_ratios[layer_index]))
        if tail_tokens < 0:
            raise ValueError(f"Negative sliding tail for {layer_name}: {tail_tokens}")
        tail_sizes.add(tail_tokens)
    if len(tail_sizes) != 1:
        raise ValueError(f"Group has inconsistent tail sizes {sorted(tail_sizes)}")
    return tail_sizes.pop()


def _select_chunk_size(
    groups: Sequence[UCMKVCacheGroupSpec],
    scheduler_block_size: int,
    requested_chunk_size: int | None,
    tokens_per_state: Mapping[int, int],
    dcp_size: int,
) -> int:
    stored_groups = tuple(group for group in groups if group.restore_policy is not None)
    if requested_chunk_size is None:
        # Choose once, after resolving DCP spans. Scratch/zero-tail groups
        # do not constrain an object that never stores them.
        chunk_size = (
            lcm(*(group.block_token_span for group in stored_groups))
            if stored_groups
            else scheduler_block_size
        )
    else:
        chunk_size = requested_chunk_size
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
    for group in stored_groups:
        span = group.block_token_span
        if (
            dcp_size == 1
            and group.restore_policy is RestorePolicy.FA
            and chunk_size < span
        ):
            if span % chunk_size or chunk_size % tokens_per_state[group.group_id]:
                raise ValueError(
                    f"FA group {group.group_id}: chunk_size {chunk_size} "
                    f"must divide its block span {span} and be a multiple of "
                    f"its tokens_per_state={tokens_per_state[group.group_id]}"
                )
        elif chunk_size % span:
            if dcp_size > 1:
                raise ValueError(
                    f"CP UCM chunks must contain whole distributed KV pages: group {group.group_id} span {span} must divide chunk_size {chunk_size}"
                )
            raise ValueError(
                f"Group {group.group_id} block span {span} must divide chunk_size {chunk_size}"
            )
        if group.restore_policy is RestorePolicy.WA and chunk_size < group.tail_tokens:
            raise ValueError(
                f"WA group {group.group_id}: chunk_size {chunk_size} "
                f"must cover its tail_tokens {group.tail_tokens}"
            )
    return chunk_size


def parse_kv_cache_config(
    kv_cache_config: "KVCacheConfig",
    *,
    scheduler_block_size: int,
    hf_text_config: Any,
    chunk_size: int | None = None,
    dcp_size: int = 1,
    device_type: str = "npu",
    num_attn_module: int = 1,
) -> UCMKVCacheSpec:
    """Compile group persistence and restore policy on scheduler and worker.

    No physical row counts, placement descriptors or layer layouts are parsed.
    hf_text_config supplies the DSV4 compressor ratios the native specs do not
    carry. vLLM's prefix_cacheable_group_ids selects the groups UCM stores;
    rebuilding scratch caches belongs to the engine. Native group positions
    remain block-table IDs, including non-persistent and empty PP groups.
    """
    from vllm.v1.core.kv_cache_utils import dcp_world_size_for_kv_cache_spec

    device_type = device_type.lower()
    native_groups = kv_cache_config.kv_cache_groups
    if not native_groups:
        raise ValueError("kv_cache_config.kv_cache_groups must not be empty")
    group_specs: list[UCMKVCacheGroupSpec] = []
    # Groups vLLM does not list as prefix-cacheable (scratch rings, replayed
    # windows) are not stored.
    stored_group_ids = set(kv_cache_config.prefix_cacheable_group_ids)
    # Tokens one KV row covers in compressed attention (vLLM's
    # tokens_per_state, 1 when not compressed). Not about Mamba state.
    # A slice of a vLLM block must hold whole rows.
    tokens_per_state: dict[int, int] = {}
    for group_id, native_group in enumerate(native_groups):
        member_specs = _member_specs(native_group.kv_cache_spec)
        tokens_per_state[group_id] = lcm(
            *(
                max(int(getattr(spec, "tokens_per_state", 1)), 1)
                for spec in member_specs
            )
        )
        is_persistent = group_id in stored_group_ids
        # Groups that are not stored get no kind.
        if is_persistent:
            spec_kinds = _classify(member_specs)
        else:
            spec_kinds = frozenset()
        if KVCacheSpecKind.MAMBA in spec_kinds and any(
            spec.mamba_cache_mode != "align" for spec in member_specs
        ):
            raise ValueError("connector v2 requires mamba_cache_mode='align'")
        is_sliding_window = not spec_kinds.isdisjoint(_SLIDING_KINDS)
        if is_sliding_window:
            if not native_group.layer_names:
                raise NotImplementedError(
                    f"Persistent SWA group {group_id} has no local PP layers; "
                    "empty persistent SWA groups are not supported"
                )
            tail_tokens = _sliding_tail_tokens(
                native_group, hf_text_config, device_type, num_attn_module
            )
        else:
            tail_tokens = None
        group_dcp_size = dcp_world_size_for_kv_cache_spec(
            native_group.kv_cache_spec, dcp_size
        )
        group_specs.append(
            UCMKVCacheGroupSpec(
                group_id=group_id,
                block_size=native_group.kv_cache_spec.block_size,
                kinds=spec_kinds,
                tail_tokens=tail_tokens,
                is_eagle_group=native_group.is_eagle_group,
                is_persistent=is_persistent,
                dcp_sharded=is_persistent and group_dcp_size > 1,
                dcp_size=group_dcp_size if is_persistent else 1,
            )
        )
    selected_chunk_size = _select_chunk_size(
        group_specs,
        scheduler_block_size,
        chunk_size,
        tokens_per_state,
        dcp_size,
    )
    spec = UCMKVCacheSpec(
        tuple(group_specs),
        selected_chunk_size,
        device_type,
        num_attn_module,
    )
    for group_spec, blocks_per_chunk in zip(spec.groups, spec.blocks_per_chunk):
        logger.info(
            "UCM group=%d kinds=%s persistent=%s block_size=%d block_token_span=%d "
            "tail=%s blocks_per_chunk=%d",
            group_spec.group_id,
            sorted(kind.value for kind in group_spec.kinds),
            group_spec.is_persistent,
            group_spec.block_size,
            group_spec.block_token_span,
            group_spec.tail_tokens,
            blocks_per_chunk,
        )
    logger.info(
        "UCM spec initialized: scheduler_block_size=%d chunk_size=%d device=%s",
        scheduler_block_size,
        selected_chunk_size,
        device_type,
    )
    return spec


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
        # A callback may name only attention, while indexer and other caches
        # of the same model layer have distinct registered names/groups.
        self.layer_id_by_name = {
            layer.layer_name: layer.layer_index
            for group_layout in self.group_layouts.values()
            for layer in group_layout.layer_specs
        }
        layer_name_lists: dict[int, list[str]] = {}
        for layer_name, layer_id in self.layer_id_by_name.items():
            layer_name_lists.setdefault(layer_id, []).append(layer_name)
        self.layer_names_by_id = {
            layer_id: frozenset(layer_names)
            for layer_id, layer_names in layer_name_lists.items()
        }

    def export_layout_metadata(self, tp_size: int) -> dict[int, GroupLayoutMetadata]:
        metadata = {}
        for group_spec in self.spec.groups:
            if not group_spec.is_persistent:
                continue
            group_layout = self.group_layouts.get(group_spec.group_id)
            metadata[group_spec.group_id] = (
                group_layout.export_metadata(tp_size)
                if group_layout is not None
                else GroupLayoutMetadata((), (), True, 1)
            )
        return metadata

    def build_store_layouts(
        self,
        topology: ParallelTopology,
        rank: ParallelRank,
        pp_metadata: Sequence[PPStageMetadata],
    ) -> None:
        expected = {group.group_id for group in self.spec.groups if group.is_persistent}
        if any(set(groups) != expected for _, groups, _ in pp_metadata):
            raise ValueError("PP metadata disagree on native group IDs")
        self.store_layouts = {
            group_id: GroupStoreLayout.build(
                self.spec,
                group_id,
                self.group_layouts.get(group_id),
                pp_metadata,
                topology,
                rank,
            )
            for group_id in sorted(expected)
        }


def groups_by_restore_policy(
    spec: UCMKVCacheSpec,
) -> dict[RestorePolicy, tuple[UCMKVCacheGroupSpec, ...]]:
    """Restore policies in dispatch order, each with the groups it stores.

    FA holds the full-attention groups; WA the sliding groups that
    re-store a window tail (tail 0 groups store nothing); State the
    mamba snapshot groups. Absent restore policies are omitted, and
    dispatch plans index groups in this order.
    """

    return {
        restore_policy: groups
        for restore_policy in RestorePolicy
        if (
            groups := tuple(
                group for group in spec.groups if group.restore_policy is restore_policy
            )
        )
    }
