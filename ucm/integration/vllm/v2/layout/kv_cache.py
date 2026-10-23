"""KV-cache metadata and the layout aggregate, independent of dispatch plans.

Parse shared persistence policy from KVCacheConfig; worker registration binds
HBM group layouts and compiles each group's storage template. No UCM chunk ids,
connector metadata or proxy transfers are interpreted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import lcm
from typing import TYPE_CHECKING, Any, Literal

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from ucm.logger import init_logger

from .store_layout import GroupStoreLayout
from .view import LAYOUT_DEBUG, layout_debug

logger = init_logger(__name__)

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheSpec

    from .group import KVCacheGroupLayout, TensorDescriptor


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
    """Shared persistence policy; tokens_per_block is tokens per block ID."""

    group_id: int
    tokens_per_block: int
    kinds: frozenset[KVCacheSpecKind]
    tail_tokens: int | None = None
    is_eagle_group: bool = False
    # Only groups vLLM lists in prefix_cacheable_group_ids are stored.
    is_persistent: bool = True
    # Number of different shards of this group's data across TP ranks.
    # None means every TP rank holds different data. MLA and replicated
    # Mamba state have 1.
    tp_shards: int | None = None
    # Same as vLLM's dcp_sharded: tokens are spread over DCP ranks, so one
    # block covers dcp * tokens_per_block tokens.
    dcp_sharded: bool = True

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
    counts tokens per record (one UCM chunk id), defining the
    restore boundaries. An explicit size must be a multiple of
    scheduler_block_size; State defaults to that common alignment, while
    attention-only models default to their smallest FA block span.
    """

    groups: tuple[UCMKVCacheGroupSpec, ...]
    scheduler_block_size: int
    tokens_per_chunk: int
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
    """vLLM blocks one UCM chunk's window spans (0 = the group stores nothing).

    HMA's ``tail_blocks`` with the one generalization v2 needs: a tail
    that does not divide the block still keeps its partial head block
    (ceil, not HMA's floor).  A state snapshot is one indivisible
    checkpoint page; a full-attention UCM chunk spans every vLLM block it
    covers, or one slice of a vLLM block when it is smaller.
    """

    tokens_per_block = group.tokens_per_block
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
        return (tokens_per_chunk + tokens_per_block - 1) // tokens_per_block
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


def _group_sharding(specs: Sequence["KVCacheSpec"]) -> tuple[int | None, bool]:
    """Read (tp_shards, dcp_sharded) from the native specs.

    If any spec in a group is dcp_sharded, the group is sharded for correctness,
    Sideeffect is the specs which are not dcp_sharded are replicated

    Specs that split TP differently cannot share one shard layout (2 and 3
    shards on TP6 put different heads on ranks 2 and 3), so the group falls
    back to None: every TP rank stores its own copy.
    The scheduler only sees one spec of a group, so it takes this result
    from the workers' handshake.
    """
    tp_shards = {_spec_tp_shards(spec) for spec in specs}
    # vLLM 0.30.0 has no dcp_sharded. It splits every group's tokens over DCP
    # ranks, and caches that are not split cannot run with DCP, so a missing
    # flag means sharded.
    return (
        tp_shards.pop() if len(tp_shards) == 1 else None,
        any(getattr(spec, "dcp_sharded", True) for spec in specs),
    )


def _spec_tp_shards(spec: "KVCacheSpec") -> int | None:
    """Number of different shards across TP, or None if every rank differs."""
    # for mamba spec, check tp_replicated, True means the mamba state is shared between tp
    if getattr(spec, "tp_replicated", False):
        # Mamba state that every TP rank holds in full.
        return 1
    elif get_kv_cache_spec_kind(spec) in _MLA_KINDS:
        # vLLM 0.30.0 and Ascend MLA specs do not have max_tp_shards.
        max_tp_shards = getattr(spec, "max_tp_shards", None)
        assert max_tp_shards in (None, 1), (
            f"MLA cache split across TP is not supported: "
            f"max_tp_shards={max_tp_shards}"
        )
        return 1
    else:
        # vLLM 0.30.0 has no max_tp_shards. None means every TP rank stores
        # its own copy, which is always correct.
        return getattr(spec, "max_tp_shards", None)


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
    # A sliding spec without layer names cannot identify compressor tails.
    layer_specs = _concrete_specs(group)
    if not layer_specs:
        raise NotImplementedError("Empty PP sliding group has no owner tail semantics")
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


def _select_tokens_per_chunk(
    groups: Sequence[UCMKVCacheGroupSpec],
    scheduler_block_size: int,
    requested_tokens_per_chunk: int | None,
    tokens_per_state: Mapping[int, int],
) -> int:
    if requested_tokens_per_chunk is None and any(
        group.is_state_snapshot for group in groups
    ):
        # State needs restore boundaries that every group shares.
        tokens_per_chunk = scheduler_block_size
    elif requested_tokens_per_chunk is None:
        # Attention only: the smallest FA block span.
        tokens_per_chunk = min(
            (
                group.tokens_per_block
                for group in groups
                if group.is_attention and not group.is_sliding_window
            ),
            default=scheduler_block_size,
        )
    elif requested_tokens_per_chunk > 0:
        tokens_per_chunk = requested_tokens_per_chunk
        logger.info(
            "Use tokens_per_chunk=%d from the connector config instead of "
            "scheduler_block_size=%d.",
            requested_tokens_per_chunk,
            scheduler_block_size,
        )
    else:
        raise ValueError("tokens_per_chunk must be positive")

    for group in groups:
        is_full_attention = group.is_attention and not group.is_sliding_window
        if is_full_attention and tokens_per_chunk < group.tokens_per_block:
            # Each UCM chunk stores one slice of a vLLM block. Slices must
            # tile the block and hold whole compressed KV rows.
            if (
                group.tokens_per_block % tokens_per_chunk
                or tokens_per_chunk % tokens_per_state[group.group_id]
            ):
                raise ValueError(
                    f"FA group {group.group_id}: tokens_per_chunk {tokens_per_chunk} "
                    f"must divide its block span {group.tokens_per_block} and be a "
                    f"multiple of its tokens_per_state="
                    f"{tokens_per_state[group.group_id]}"
                )
        elif not group.is_persistent or (
            group.is_sliding_window and not group.tail_tokens
        ):
            # A group that is not stored or a sliding group with no tail stores nothing,
            # so it puts no limit on tokens_per_chunk.
            continue
        else:
            # FA, State and sliding groups with a tail: a UCM chunk holds
            # whole vLLM blocks of this group.
            if tokens_per_chunk % group.tokens_per_block:
                raise ValueError(
                    f"Group {group.group_id} block span {group.tokens_per_block} "
                    f"must divide tokens_per_chunk {tokens_per_chunk}"
                )
    return tokens_per_chunk


def parse_kv_cache_config(
    kv_cache_config: "KVCacheConfig",
    *,
    scheduler_block_size: int,
    hf_text_config: Any,
    tokens_per_chunk: int | None = None,
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
    device_type = device_type.lower()
    native_groups = kv_cache_config.kv_cache_groups
    if not native_groups:
        raise ValueError("kv_cache_config.kv_cache_groups must not be empty")
    group_infos: list[UCMKVCacheGroupSpec] = []
    # Groups vLLM does not list as prefix-cacheable (scratch rings, replayed
    # windows) are not stored.
    stored_group_ids = set(kv_cache_config.prefix_cacheable_group_ids)
    # Tokens one KV row covers in compressed attention (vLLM's
    # tokens_per_state, 1 when not compressed). Not about Mamba state.
    # A slice of a vLLM block must hold whole rows.
    tokens_per_state: dict[int, int] = {}
    uses_prefix_replay = False
    for group_id, native_group in enumerate(native_groups):
        member_specs = _member_specs(native_group.kv_cache_spec)
        tokens_per_state[group_id] = lcm(
            *(
                max(int(getattr(spec, "tokens_per_state", 1)), 1)
                for spec in member_specs
            )
        )
        uses_prefix_replay = uses_prefix_replay or any(
            getattr(spec, "prefix_replay_tokens", 0) > 0 for spec in member_specs
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
            tail_tokens = _sliding_tail_tokens(
                native_group, hf_text_config, device_type, num_attn_module
            )
        elif is_persistent:
            tail_tokens = None
        else:
            tail_tokens = 0
        tp_shards, dcp_sharded = _group_sharding(member_specs)
        group_infos.append(
            UCMKVCacheGroupSpec(
                group_id=group_id,
                tokens_per_block=native_group.kv_cache_spec.block_size,
                kinds=spec_kinds,
                tail_tokens=tail_tokens,
                is_eagle_group=native_group.is_eagle_group,
                is_persistent=is_persistent,
                tp_shards=tp_shards,
                dcp_sharded=dcp_sharded,
            )
        )
    selected_tokens_per_chunk = _select_tokens_per_chunk(
        group_infos,
        scheduler_block_size,
        tokens_per_chunk,
        tokens_per_state,
    )
    if uses_prefix_replay and selected_tokens_per_chunk < scheduler_block_size:
        logger.warning(
            "tokens_per_chunk=%d is smaller than scheduler_block_size=%d, but "
            "vLLM SWA bounded replay rounds external hits down to "
            "scheduler_block_size, so the smaller UCM chunks give no finer hits.",
            selected_tokens_per_chunk,
            scheduler_block_size,
        )
    spec = UCMKVCacheSpec(
        tuple(group_infos),
        scheduler_block_size,
        selected_tokens_per_chunk,
        device_type,
        num_attn_module,
    )
    if LAYOUT_DEBUG:
        for group_info, blocks_per_chunk in zip(spec.groups, spec.blocks_per_chunk):
            layout_debug(
                f"policy group={group_info.group_id} "
                f"kinds={sorted(kind.value for kind in group_info.kinds)} "
                f"persistent={group_info.is_persistent} "
                f"token_block={group_info.tokens_per_block} "
                f"tail={group_info.tail_tokens} blocks_per_chunk={blocks_per_chunk}"
            )
        layout_debug(
            f"spec scheduler_block={scheduler_block_size} "
            f"ucm_cache_block={selected_tokens_per_chunk} device={device_type}"
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
