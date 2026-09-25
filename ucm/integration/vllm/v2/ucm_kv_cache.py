"""Semantic KV-cache description and the dump/load batch orchestrator for connector v2.

Physical placement -- where every layer's blocks and states sit -- lives in
``.layout``.  This module keeps the semantic layer (groups, cache kinds,
block sizing) and walks dispatch plans over the layout model to produce
proxy batches with deterministic record offsets.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import chain, repeat
from typing import TYPE_CHECKING, Literal

import numpy as np
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

from .layout import build_group_layouts
from .layout.group import TensorDescriptor
from .layout.view import LAYOUT_DEBUG, layout_debug
from .record_layout import GroupRecordLayout
from .ucm_proxy import KVCacheValue, UCMProxyBatch, UCMProxyTransfer

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )

    from .layout.group import KVCacheGroupLayout
    from .ucm_scheduler import UCMConnectorMetadata, UCMGroupDispatchPlan


_SLIDING_KINDS = frozenset(
    (KVCacheSpecKind.SLIDING_WINDOW, KVCacheSpecKind.SLIDING_WINDOW_MLA)
)


@dataclass(frozen=True)
class UCMLayerSpec:
    """One registered layer; storage_block_size counts stored states, not tokens."""

    layer_name: str
    layer_index: int
    kv_cache_spec: "KVCacheSpec"
    storage_block_size: int
    num_blocks: int
    # 0.29 placement: the declaration covering this layer and the
    # layer's position in it; None on 0.26 (per-tensor overlay).
    descriptor: "TensorDescriptor | None" = None
    descriptor_position: int = 0


@dataclass(frozen=True)
class UCMKVCacheGroupInfo:
    """One native KV group; token_block_size is tokens covered by one block ID."""

    group_id: int
    layers: tuple[UCMLayerSpec, ...]
    group_spec: "KVCacheSpec"
    token_block_size: int
    kinds: frozenset[KVCacheSpecKind]
    tail_tokens: int | None = None
    # vLLM blocks one ucm key's window spans (0 = the group stores
    # nothing); the parser stamps it once the cache block size is known.
    tail_blocks: int = 0
    is_eagle_group: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def is_attention(self) -> bool:
        return KVCacheSpecKind.MAMBA not in self.kinds

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
    is both the record unit and the shared hash chain's key granularity:
    the smallest token_block_size across the full-attention groups
    (token_block_size already folds compression), or the scheduler block
    when the model has no full-attention group.
    """

    groups: tuple[UCMKVCacheGroupInfo, ...]
    scheduler_block_size: int
    ucm_cache_block_size: int
    device_type: str

    @property
    def alignment_block_size(self) -> int:
        """The multi-group boundary every external store aligns to.

        The model-check harness trims its source prompt to a multiple of
        this (lcm with the native group block sizes), so a dump always
        covers complete cache blocks of every group.  The cache_block_size
        is that boundary: FA chains hash at it and every group's window
        within it is whole or a measured fraction.
        """

        return self.ucm_cache_block_size

    @property
    def attn_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_attention)

    @property
    def state_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_state_snapshot)

    @property
    def sw_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_sliding_window)

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

    def dispatch_routes(
        self,
    ) -> tuple[
        tuple[Literal["FA", "WA", "State"], tuple["UCMKVCacheGroupInfo", ...]], ...
    ]:
        """The routing table every dump/load works over: key kind -> groups.

        FA holds the full-attention groups; WA the sliding groups that
        re-store a window tail (tail 0 groups store nothing); State the
        mamba snapshot groups. Empty kinds are absent, and
        ``group_ucm_block_ids`` / dispatch plans index these in order.
        """

        routes: list[
            tuple[Literal["FA", "WA", "State"], tuple["UCMKVCacheGroupInfo", ...]]
        ] = []
        if self.fa_groups:
            routes.append(("FA", self.fa_groups))
        wa_stored = tuple(
            group for group in self.wa_groups if (group.tail_tokens or 0) > 0
        )
        if wa_stored:
            routes.append(("WA", wa_stored))
        if self.state_groups:
            routes.append(("State", self.state_groups))
        return tuple(routes)

    @property
    def layer_to_group(self) -> Mapping[str, int]:
        return {
            layer.layer_name: group.group_id
            for group in self.groups
            for layer in group.layers
        }


def _group_tail_blocks(group: UCMKVCacheGroupInfo, ucm_block_size: int) -> int:
    """vLLM blocks one ucm key's window spans (0 = the group stores nothing).

    HMA's ``tail_blocks`` with the one generalization v2 needs: a tail
    that does not divide the block still keeps its partial head block
    (ceil, not HMA's floor).  A state snapshot is one indivisible
    checkpoint page; a full-attention key spans the whole ucm block
    (several blocks when the unit is the larger side, the containing
    block otherwise).
    """

    token_block = group.token_block_size
    if group.is_state_snapshot:
        return 1
    if group.is_sliding_window:
        tail = group.tail_tokens or 0
        return (tail + token_block - 1) // token_block if tail > 0 else 0
    return (ucm_block_size + token_block - 1) // token_block


def _concrete_specs(
    group: "KVCacheGroupSpec",
) -> tuple[tuple[str, "KVCacheSpec"], ...]:
    group_spec = group.kv_cache_spec
    nested = getattr(group_spec, "kv_cache_specs", None)
    names = tuple(getattr(group, "layer_names", ()))
    if nested:
        missing = [name for name in names if name not in nested]
        if missing:
            raise ValueError(f"KV cache group is missing specs for layers {missing}")
        return tuple((name, nested[name]) for name in names)
    return tuple((name, group_spec) for name in names)


def _spec_tokens_per_state(spec: "KVCacheSpec") -> int:
    """Compression ratio of one stored state (DSV4 C4A = 4).

    Ascend 0.26 names the field ``compress_ratio``; vLLM 0.29 renamed it to
    ``tokens_per_state`` (vLLM #51718) with identical semantics for DSV4
    (int > 1 compresses multiple tokens into one stored state). Mamba
    specs carry a -1 sentinel on 0.29 which callers treat as no
    compression; sliding and state groups carry 1.
    """
    value = getattr(spec, "compress_ratio", None)
    if value is None:
        value = getattr(spec, "tokens_per_state", 1)
    if value is None:
        return 1
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _layer_index(name: str, device_type: str, num_hidden_layers: int | None) -> int:
    """Ascend MTP uses local IDs; official 0.29 already uses global IDs."""
    index = extract_layer_index(name)
    if device_type.lower() == "npu" and "mtp" in name.split("."):
        if num_hidden_layers is None or num_hidden_layers <= 0:
            raise ValueError("MTP layer IDs require model num_hidden_layers")
        if index < num_hidden_layers:
            index += num_hidden_layers
    return index


def _classify(
    group: "KVCacheGroupSpec",
    concrete: Sequence[tuple[str, "KVCacheSpec"]],
) -> frozenset[KVCacheSpecKind]:
    specs = (
        tuple(spec for _, spec in concrete)
        or tuple(getattr(group.kv_cache_spec, "kv_cache_specs", {}).values())
        or (group.kv_cache_spec,)
    )
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


def _parse_descriptors(
    kv_cache_tensors: "Sequence[object]",
) -> dict[str, tuple["TensorDescriptor", int]]:
    """Mirror vLLM 0.29 kv_cache_tensors; undeclared configs yield {}."""

    declared_at: dict[str, tuple["TensorDescriptor", int]] = {}
    for entry in kv_cache_tensors:
        layers = tuple(str(name) for name in getattr(entry, "layers", ()) or ())
        if not layers:
            raise ValueError("A declared KV cache tensor covers no layers")
        descriptor = TensorDescriptor(
            layers=layers,
            offset=int(getattr(entry, "offset", 0)),
            layer_stride=int(getattr(entry, "layer_stride")),
            block_stride=int(getattr(entry, "block_stride")),
        )
        if (
            descriptor.offset < 0
            or descriptor.layer_stride < 0
            or descriptor.block_stride <= 0
        ):
            raise ValueError(f"Invalid declared placement: {descriptor}")
        for position, name in enumerate(layers):
            if name in declared_at:
                raise ValueError(
                    f"Layer {name} is covered by more than one declared tensor"
                )
            declared_at[name] = (descriptor, position)
    return declared_at


def parse_kv_cache_config(
    kv_cache_config: "KVCacheConfig",
    *,
    scheduler_block_size: int,
    ucm_cache_block_size: int | None = None,
    device_type: str = "npu",
    attention_tokens_per_state: Mapping[int, int] | None = None,
    num_hidden_layers: int | None = None,
) -> UCMKVCacheSpec:
    """Describe logical groups and per-layer storage from KVCacheConfig.

    vLLM 0.29's scheduler replaces a UniformTypeKVCacheSpecs map with one
    representative spec. For DSV4 that loses C4/C128 per-layer ratios.
    The connector supplies attention_tokens_per_state from the model config
    on both scheduler and worker; direct callers with full specs can omit it.
    This mapping applies to full attention, never compressor state tensors.
    """

    attention_tokens_per_state = attention_tokens_per_state or {}

    # 0.29 kv_cache_tensors carry a layers placement per tensor; 0.26
    # entries (size + shared_by) do not, so only the declared ones parse.
    kv_cache_tensors = tuple(getattr(kv_cache_config, "kv_cache_tensors", ()) or ())
    declared = tuple(
        tensor
        for tensor in kv_cache_tensors
        if getattr(tensor, "layers", None) is not None
    )
    declared_at = _parse_descriptors(declared)
    raw_groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()))
    if not raw_groups:
        raise ValueError("kv_cache_config.kv_cache_groups must not be empty")

    classified: list[
        tuple[
            "KVCacheGroupSpec",
            tuple[tuple[str, "KVCacheSpec"], ...],
            frozenset[KVCacheSpecKind],
        ]
    ] = []
    layer_indices: dict[str, int] = {}
    for raw_group in raw_groups:
        concrete = _concrete_specs(raw_group)
        layer_indices.update(
            (name, _layer_index(name, device_type, num_hidden_layers))
            for name, _ in concrete
        )
        kinds = _classify(raw_group, concrete)
        if KVCacheSpecKind.MAMBA in kinds:
            check_specs = (
                concrete
                or tuple(getattr(raw_group.kv_cache_spec, "kv_cache_specs", {}).items())
                or (("", raw_group.kv_cache_spec),)
            )
            modes = {
                str(getattr(spec, "mamba_cache_mode", None)) for _, spec in check_specs
            }
            if modes != {"align"}:
                raise ValueError(
                    "connector v2 supports Mamba state only with "
                    f"mamba_cache_mode='align', got {sorted(modes)}"
                )
            block_sizes = {int(getattr(spec, "block_size")) for _, spec in check_specs}
            if block_sizes != {scheduler_block_size}:
                raise ValueError(
                    "Mamba align block size must equal cache_config.block_size="
                    f"{scheduler_block_size}, got {sorted(block_sizes)}"
                )
        classified.append((raw_group, concrete, kinds))

    device_type = str(device_type).lower()
    if len(raw_groups) != 1 and ucm_cache_block_size is not None:
        raise ValueError(
            "custom ucm_cache_block_size is supported only for a single KV group"
        )

    # Compression detection drives only the per-layer ratio recovery in
    # the group loop below (0.29's representative spec loses per-layer
    # ratios); the cache block itself comes from the FA groups' token
    # spans, which already fold compression.
    has_compression = any(
        max((_spec_tokens_per_state(spec) for _, spec in concrete), default=1) > 1
        for _raw_group, concrete, kinds in classified
        if KVCacheSpecKind.MAMBA not in kinds and kinds.isdisjoint(_SLIDING_KINDS)
    )

    groups: list[UCMKVCacheGroupInfo] = []
    fa_token_blocks: list[int] = []
    attention_tokens_per_state_by_layer: dict[int, int] = {}
    if has_compression:
        for _, concrete, kinds in classified:
            if KVCacheSpecKind.MAMBA in kinds or not kinds.isdisjoint(_SLIDING_KINDS):
                continue
            for name, concrete_spec in concrete:
                attention_tokens_per_state_by_layer[layer_indices[name]] = (
                    attention_tokens_per_state.get(
                        layer_indices[name], _spec_tokens_per_state(concrete_spec)
                    )
                )
    num_blocks = int(getattr(kv_cache_config, "num_blocks", 0))
    for group_id, (raw_group, concrete, kinds) in enumerate(classified):
        representative = (
            concrete[0][1]
            if concrete
            else next(
                iter(getattr(raw_group.kv_cache_spec, "kv_cache_specs", {}).values()),
                raw_group.kv_cache_spec,
            )
        )
        physical_block_size = int(getattr(raw_group.kv_cache_spec, "block_size"))
        if KVCacheSpecKind.MAMBA in kinds:
            # Mamba state blocks follow the scheduler block (align check
            # above); they never fold tokens.
            token_block_size = physical_block_size
        else:
            # Ascend 0.26 attention specs report the storage span and
            # carry compress_ratio: the token span is the product.  vLLM
            # 0.29 reports token spans directly, so its spec block is
            # already the token span.
            ascend_ratio = getattr(representative, "compress_ratio", None)
            token_block_size = (
                physical_block_size * int(ascend_ratio)
                if ascend_ratio
                else physical_block_size
            )
            if kinds.isdisjoint(_SLIDING_KINDS):
                fa_token_blocks.append(token_block_size)
        layers: list[UCMLayerSpec] = []
        for index, (name, spec) in enumerate(concrete):
            # Normalize to the number of stored states one group block
            # spans.  Ascend 0.26 reports the C4 storage span as block_size
            # directly; vLLM 0.29 reports the logical span and the storage
            # axis is block_size // tokens_per_state (DSV4 C4A: 256/4=64
            # states; C128A: 256/128=2; uncompressed specs keep the block).
            logical = int(getattr(spec, "block_size"))
            ratio = _spec_tokens_per_state(spec)
            if has_compression and kinds.isdisjoint(_SLIDING_KINDS):
                ratio = attention_tokens_per_state_by_layer[layer_indices[name]]
            if device_type == "npu":
                # Ascend's replicated DCP indexer allocates consecutive kernel
                # rows per logical block, as declared by its native cache spec.
                storage_block_size = logical * int(
                    getattr(spec, "sfa_dcp_replicated_indexer_size", 1)
                )
            elif ratio > 1 and logical % ratio == 0:
                storage_block_size = logical // ratio
            else:
                storage_block_size = logical
            layers.append(
                UCMLayerSpec(
                    name,
                    layer_indices[name],
                    spec,
                    storage_block_size,
                    num_blocks,
                    *(declared_at.get(name, (None, 0))),
                )
            )
        tail_tokens: int | None = None
        if not kinds.isdisjoint(_SLIDING_KINDS):
            # What a sliding group re-stores at each hash boundary -- not
            # necessarily the whole window: swa_cache keeps the full
            # window, a compressor state cache keeps window minus its
            # layer's compression ratio, and window == ratio leaves
            # nothing to store (tail 0 groups join no chain).
            tails: set[int] = set()
            tail_specs = concrete or tuple(
                getattr(raw_group.kv_cache_spec, "kv_cache_specs", {}).items()
            )
            for name, concrete_spec in tail_specs:
                window = int(getattr(concrete_spec, "sliding_window"))
                if name.lower().endswith("swa_cache"):
                    tail = window
                else:
                    layer_index = layer_indices.get(name)
                    if layer_index is None:
                        layer_index = _layer_index(name, device_type, num_hidden_layers)
                    ratio = attention_tokens_per_state_by_layer.get(
                        layer_index, attention_tokens_per_state.get(layer_index)
                    )
                    if ratio is None:
                        raise ValueError(
                            "Cannot find matching full-attention compression ratio "
                            f"for sliding layer {layer_index}"
                        )
                    tail = window - ratio
                if tail < 0:
                    raise ValueError(f"Negative sliding tail for {name}: {tail}")
                tails.add(tail)
            if not tail_specs:
                raise NotImplementedError(
                    "An empty PP sliding group needs global per-layer tail semantics; "
                    "this engine projection does not provide enough information"
                )
            if len(tails) != 1:
                raise ValueError(
                    f"Group {group_id} has inconsistent tail sizes {sorted(tails)}"
                )
            tail_tokens = tails.pop()
        groups.append(
            UCMKVCacheGroupInfo(
                group_id=group_id,
                layers=tuple(layers),
                group_spec=raw_group.kv_cache_spec,
                token_block_size=token_block_size,
                kinds=kinds,
                tail_tokens=tail_tokens,
                is_eagle_group=any(
                    "eagle" in layer.layer_name.lower() for layer in layers
                ),
            )
        )

    state_groups = tuple(group for group in groups if group.is_state_snapshot)
    if state_groups:
        if not any(group.is_attention for group in groups):
            raise ValueError(
                "State-only KV cache groups are unsupported: mamba snapshots "
                "restore behind a full-attention prefix"
            )
        mismatched_groups = {
            group.group_id: group.token_block_size
            for group in groups
            if group.token_block_size != scheduler_block_size
        }
        if mismatched_groups:
            raise ValueError(
                "Mamba align requires every KV group block size to equal "
                f"cache_config.block_size={scheduler_block_size}, got "
                f"{mismatched_groups}"
            )
    if len(groups) == 1 and ucm_cache_block_size is not None:
        selected_block = ucm_cache_block_size
        if (
            selected_block < scheduler_block_size
            or selected_block % scheduler_block_size
        ):
            raise ValueError(
                "ucm_cache_block_size must be a positive multiple of "
                "scheduler_block_size"
            )
    elif fa_token_blocks:
        selected_block = min(fa_token_blocks)
    else:
        selected_block = scheduler_block_size

    # Flat plan windows need a uniform per-key block count on FA chains:
    # every FA group's token span must divide, or be divided by, the
    # cache block -- otherwise a key straddles blocks and the window
    # shape varies per key.
    for group in groups:
        if not (group.is_attention and not group.is_sliding_window):
            continue
        if selected_block % group.token_block_size and (
            group.token_block_size % selected_block
        ):
            raise ValueError(
                f"Full-attention group {group.group_id} token_block_size "
                f"{group.token_block_size} neither divides nor is a "
                f"multiple of ucm_cache_block_size {selected_block}"
            )

    # Sliding tails keep a static window shape: the boundary must land
    # on the group's block grid, so its token block must divide the
    # cache block.
    for group in groups:
        if not group.is_sliding_window or not (group.tail_tokens or 0):
            continue
        if selected_block % group.token_block_size:
            raise ValueError(
                f"Sliding-window group {group.group_id} token_block_size "
                f"{group.token_block_size} must divide "
                f"ucm_cache_block_size {selected_block}"
            )

    groups = [
        replace(group, tail_blocks=_group_tail_blocks(group, selected_block))
        for group in groups
    ]

    if LAYOUT_DEBUG:
        for group in groups:
            kind_names = ",".join(sorted(kind.value for kind in group.kinds))
            layout_debug(
                f"spec group={group.group_id} layers={group.num_layers} "
                f"kinds={{{kind_names}}} token_block={group.token_block_size} "
                f"tail={group.tail_tokens} tail_blocks={group.tail_blocks}"
            )
        layout_debug(
            f"spec scheduler_block={scheduler_block_size} "
            f"ucm_cache_block={selected_block} "
            f"device={device_type}"
        )

    return UCMKVCacheSpec(
        groups=tuple(groups),
        scheduler_block_size=scheduler_block_size,
        ucm_cache_block_size=selected_block,
        device_type=device_type,
    )


class UCMKVCacheLayout:
    """Ragged, per-layer physical layout with deterministic record offsets."""

    def __init__(
        self,
        spec: UCMKVCacheSpec,
        kv_caches: Mapping[str, KVCacheValue],
    ) -> None:
        self.spec = spec
        self.group_layouts: Mapping[int, "KVCacheGroupLayout"] = build_group_layouts(
            spec, kv_caches
        )
        self.record_layouts = {
            group.group_id: GroupRecordLayout.build(
                group, self.group_layouts[group.group_id], spec.ucm_cache_block_size
            )
            for group in spec.groups
            if group.layers
        }
        # Per-kind participating groups, in dispatch_routes() order -- the
        # plan's windows array is positional over this order.
        self._routes_by_kind: Mapping[str, tuple["UCMKVCacheGroupInfo", ...]] = {
            kind: groups for kind, groups in spec.dispatch_routes()
        }
        # A callback may name only attention, while indexer and other caches
        # of the same model layer have distinct registered names/groups.
        self.layer_id_by_name = {
            layer.layer_name: layer.layer_index
            for group in spec.groups
            for layer in group.layers
        }
        names_by_id: dict[int, list[str]] = {}
        for name, layer_id in self.layer_id_by_name.items():
            names_by_id.setdefault(layer_id, []).append(name)
        self.layer_names_by_id = {
            layer_id: frozenset(names) for layer_id, names in names_by_id.items()
        }

    def _selected_layer_names(
        self, layer_name: str | None, layer_id: int | None
    ) -> frozenset[str] | None:
        if layer_name is not None and layer_id is not None:
            raise ValueError("Specify either layer_name or layer_id")
        if layer_id is not None:
            return self.layer_names_by_id[layer_id]
        return None if layer_name is None else frozenset((layer_name,))

    def build_load_transfers(
        self,
        metadata: "UCMConnectorMetadata",
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
    ) -> tuple[UCMProxyTransfer, ...]:
        return self._build_transfers(
            (p for r in metadata.requests.values() for p in r.load_plans),
            self._selected_layer_names(layer_name, layer_id),
        )

    def build_dump_transfers(
        self,
        metadata: "UCMConnectorMetadata",
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
    ) -> tuple[UCMProxyTransfer, ...]:
        return self._build_transfers(
            (p for r in metadata.requests.values() for p in r.dump_plans),
            self._selected_layer_names(layer_name, layer_id),
        )

    def _build_transfers(
        self,
        plans: Iterable["UCMGroupDispatchPlan"],
        layer_names: frozenset[str] | None,
    ) -> tuple[UCMProxyTransfer, ...]:
        transfers = []
        for plan in plans:
            groups = [
                g
                for g in self._iter_plan_segments(plan, layer_names, matrices=True)
                if g[2].shape[1]
            ]
            if not groups:
                continue
            ptrs = (
                groups[0][2]
                if len(groups) == 1
                else np.concatenate([g[2] for g in groups], axis=1)
            )
            if len(groups) == 1:
                offsets, sizes = groups[0][1], groups[0][3]
            else:
                offsets = np.broadcast_to(
                    np.concatenate([g[1][0] for g in groups]), ptrs.shape
                )
                sizes = np.broadcast_to(
                    np.concatenate([g[3][0] for g in groups]), ptrs.shape
                )
            # Each plan belongs to one request's hash chain, so keys are unique.
            # Keep requests separate: shared-prefix keys may have distinct targets.
            transfers.append(UCMProxyTransfer(plan.keys, ptrs, sizes, offsets))
        return tuple(transfers)

    def build_load_batches(
        self,
        metadata: "UCMConnectorMetadata",
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
    ) -> UCMProxyBatch:
        """Select one cache name or all names of one model layer across groups."""
        plans = (
            plan
            for request_meta in metadata.requests.values()
            for plan in request_meta.load_plans
        )
        return self._build_batches(
            plans, self._selected_layer_names(layer_name, layer_id)
        )

    def build_dump_batches(
        self,
        metadata: "UCMConnectorMetadata",
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
    ) -> UCMProxyBatch:
        """Select memory ranges; a filtered batch may not be a complete record.

        Layerwise saving needs a backend that accumulates partial records and
        publishes only when complete. SimpleFileUCMProxy supports this through
        repeated dump calls followed by commit(keys).
        """
        plans = (
            plan
            for request_meta in metadata.requests.values()
            for plan in request_meta.dump_plans
        )
        return self._build_batches(
            plans, self._selected_layer_names(layer_name, layer_id)
        )

    def _build_batches(
        self,
        plans: Iterable["UCMGroupDispatchPlan"],
        layer_names: frozenset[str] | None,
    ) -> UCMProxyBatch:
        keys: list[bytes] = []
        offsets: list[np.ndarray] = []
        ptrs: list[np.ndarray] = []
        sizes: list[np.ndarray] = []
        for plan in plans:
            for (
                group_keys,
                group_offsets,
                group_ptrs,
                group_sizes,
            ) in self._iter_plan_segments(plan, layer_names):
                keys.extend(group_keys)
                offsets.append(group_offsets)
                ptrs.append(group_ptrs)
                sizes.append(group_sizes)
        if not keys:
            return UCMProxyBatch(
                (),
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.uint64),
            )
        return UCMProxyBatch(
            tuple(keys),
            offsets[0] if len(offsets) == 1 else np.concatenate(offsets),
            ptrs[0] if len(ptrs) == 1 else np.concatenate(ptrs),
            sizes[0] if len(sizes) == 1 else np.concatenate(sizes),
        )

    def _iter_plan_segments(
        self,
        plan: "UCMGroupDispatchPlan",
        layer_names: frozenset[str] | None,
        *,
        matrices: bool = False,
    ) -> Iterator[tuple[Sequence[bytes], np.ndarray, np.ndarray, np.ndarray]]:
        """One plan's records as per-group arrays, straight off templates.

        The scheduler sends keys and window block ids; each group's
        static template (compiled at layout init -- the offsets/sizes
        grids of one key's window, block-major: block 0's views, block
        1's views, ...) places them inside the key's record, and group
        g's contribution starts at group_record_offset.  All keys of
        a plan resolve in one vectorized pass per group; only the block
        ids (and full-attention sub-span heads) vary per call.  Block
        First groups with whole unfiltered windows take the
        one-span-per-block fast path, paddings riding inside.
        """

        if not plan.keys:
            return
        physical_groups = self._routes_by_kind[plan.hash_group]
        if len(plan.windows) != len(physical_groups):
            raise ValueError(
                f"{plan.hash_group} plan carries {len(plan.windows)} windows "
                f"for {len(physical_groups)} groups"
            )
        if LAYOUT_DEBUG:
            layout_debug(
                f"plan hash_group={plan.hash_group} "
                f"tokens=[{plan.token_start},{plan.token_end}) "
                f"keys={len(plan.keys)} "
                f"groups={[group.group_id for group in physical_groups]}"
            )
        key_count = len(plan.keys)
        group_record_offset = 0
        for group, blocks in zip(physical_groups, plan.windows):
            if not group.layers:
                continue
            group_layout = self.group_layouts[group.group_id]
            # Normalize at the pickle boundary: int64 ids mixed into the
            # uint64 stride arithmetic below would silently promote.
            blocks = np.asarray(blocks, dtype=np.uint64)
            record_layout = self.record_layouts[group.group_id]
            per_key = record_layout.blocks_per_key
            total = key_count * per_key
            if len(blocks) != total:
                raise ValueError(
                    f"Plan window for group {group.group_id} carries "
                    f"{len(blocks)} blocks for {key_count} keys x "
                    f"{per_key} blocks each"
                )
            mask = (
                None
                if layer_names is None
                else group_layout.segment_mask(layer_names=layer_names)
            )
            token_offsets = None
            if record_layout.dynamic_token_offsets:
                first_key = plan.token_start // self.spec.ucm_cache_block_size
                token_offsets = (
                    np.arange(first_key, first_key + key_count, dtype=np.uint64)
                    * self.spec.ucm_cache_block_size
                    % group_layout.token_block_size
                )
            if matrices:
                offsets, ptrs, sizes = record_layout.resolve_matrices(
                    blocks, key_count, token_offsets=token_offsets, segment_mask=mask
                )
                if group_record_offset:
                    offsets = np.broadcast_to(
                        offsets[0] + group_record_offset, offsets.shape
                    )
                yield plan.keys, offsets, ptrs, sizes
                group_record_offset += record_layout.record_bytes
                continue
            offsets, ptrs, sizes, entries_per_key = record_layout.resolve(
                blocks, key_count, token_offsets=token_offsets, segment_mask=mask
            )
            # resolve returns fresh arrays, so this cannot modify a template
            # or a previously submitted batch.
            if group_record_offset:
                offsets += group_record_offset
            if LAYOUT_DEBUG:
                layout_debug(
                    f"record group={group.group_id} keys={key_count} "
                    f"blocks={total} entries={len(ptrs)} "
                    f"record_size={group_record_offset + record_layout.record_bytes}"
                )
            yield (
                (
                    list(plan.keys)
                    if entries_per_key == 1
                    else list(
                        chain.from_iterable(
                            repeat(key, entries_per_key) for key in plan.keys
                        )
                    )
                ),
                offsets,
                ptrs,
                sizes,
            )
            group_record_offset += record_layout.record_bytes
