"""Semantic KV-cache description and the dump/load batch orchestrator for connector v2.

Physical placement -- where every layer's blocks and states sit -- lives in
``.layout``.  This module keeps the semantic layer (groups, cache kinds,
block sizing, the DSV4 policy) and walks dispatch plans over the layout
model to produce proxy batches with deterministic record offsets.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

from .layout import build_group_layouts
from .layout.geometry import LAYOUT_DEBUG, layout_debug
from .ucm_proxy import KVCacheValue, UCMProxyBatch

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )

    from .layout.group import GroupLayout
    from .ucm_scheduler import UCMConnectorMetadata, UCMGroupDispatchPlan


_SLIDING_KINDS = frozenset(
    (KVCacheSpecKind.SLIDING_WINDOW, KVCacheSpecKind.SLIDING_WINDOW_MLA)
)


@dataclass(frozen=True)
class UCMLayerSpec:
    layer_name: str
    layer_index: int
    kv_cache_spec: "KVCacheSpec"
    storage_block_size: int


@dataclass(frozen=True)
class UCMKVCacheGroupInfo:
    group_id: int
    layers: tuple[UCMLayerSpec, ...]
    group_spec: "KVCacheSpec"
    token_block_size: int
    hash_block_size: int
    kinds: frozenset[KVCacheSpecKind]
    is_c4a: bool = False
    tail_tokens: int | None = None
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
    groups: tuple[UCMKVCacheGroupInfo, ...]
    scheduler_block_size: int
    alignment_block_size: int
    chunk_size: int
    device_type: str
    is_dsv4: bool

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

    @property
    def c4a_group(self) -> UCMKVCacheGroupInfo | None:
        matches = tuple(group for group in self.groups if group.is_c4a)
        if len(matches) > 1:
            raise ValueError("More than one C4A group was classified")
        return matches[0] if matches else None

    @property
    def layer_to_group(self) -> Mapping[str, int]:
        return {
            layer.layer_name: group.group_id
            for group in self.groups
            for layer in group.layers
        }


def _layer_index(layer_name: str, fallback: int) -> int:
    match = re.search(r"(?:layers|layer)\.(\d+)", layer_name)
    return int(match.group(1)) if match else fallback


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


def _spec_compress_ratio(spec: "KVCacheSpec") -> int:
    """Compression ratio of one stored state (DSV4 C4A = 4).

    Ascend 0.26 names the field ``compress_ratio``; vLLM 0.29 renamed it to
    ``tokens_per_state`` (vLLM #51718) with identical semantics for DSV4
    (int > 1 compresses multiple tokens into one stored state). Mamba specs
    carry a -1 sentinel on 0.29 which is not meaningful as a ratio; callers
    only read this on attention/MLA specs.
    """
    value = getattr(spec, "compress_ratio", None)
    if value is None:
        value = getattr(spec, "tokens_per_state", 1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _classify(
    group: "KVCacheGroupSpec",
    concrete: Sequence[tuple[str, "KVCacheSpec"]],
) -> tuple[frozenset[KVCacheSpecKind], bool]:
    specs = tuple(spec for _, spec in concrete) or (group.kv_cache_spec,)
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
    is_c4a = any(
        _spec_compress_ratio(spec) == 4
        and kind == KVCacheSpecKind.MLA_ATTENTION
        for spec, kind in zip(specs, spec_kinds)
    )
    return kinds, is_c4a


def parse_kv_cache_config(
    kv_cache_config: "KVCacheConfig",
    *,
    scheduler_block_size: int,
    chunk_size: int | None = None,
    device_type: str = "npu",
) -> UCMKVCacheSpec:
    """Describe logical groups and per-layer storage from KVCacheConfig."""

    raw_groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()))
    if not raw_groups:
        raise ValueError("kv_cache_config.kv_cache_groups must not be empty")
    if scheduler_block_size <= 0:
        raise ValueError("scheduler_block_size must be positive")

    classified: list[
        tuple[
            "KVCacheGroupSpec",
            tuple[tuple[str, "KVCacheSpec"], ...],
            frozenset[KVCacheSpecKind],
            bool,
        ]
    ] = []
    dsv4 = False
    for raw_group in raw_groups:
        concrete = _concrete_specs(raw_group)
        kinds, is_c4a = _classify(raw_group, concrete)
        if KVCacheSpecKind.MAMBA in kinds:
            modes = {
                str(getattr(spec, "mamba_cache_mode", None)) for _, spec in concrete
            }
            if modes != {"align"}:
                raise ValueError(
                    "connector v2 supports Mamba state only with "
                    f"mamba_cache_mode='align', got {sorted(modes)}"
                )
            block_sizes = {int(getattr(spec, "block_size")) for _, spec in concrete}
            if block_sizes != {scheduler_block_size}:
                raise ValueError(
                    "Mamba align block size must equal cache_config.block_size="
                    f"{scheduler_block_size}, got {sorted(block_sizes)}"
                )
        classified.append((raw_group, concrete, kinds, is_c4a))
        if KVCacheSpecKind.SLIDING_WINDOW_MLA in kinds:
            dsv4 = True

    device_type = str(device_type).lower()
    if len(raw_groups) != 1 and chunk_size is not None:
        raise ValueError("custom chunk_size is supported only for a single KV group")

    c4_sizes: set[int] = set()
    if dsv4:
        for raw_group, _, _, is_c4a in classified:
            if is_c4a:
                c4_sizes.add(int(getattr(raw_group.kv_cache_spec, "block_size")))
        if len(c4_sizes) != 1:
            raise ValueError(
                "DeepSeek V4 requires exactly one C4A block size, got "
                f"{sorted(c4_sizes)}"
            )
        c4_size = c4_sizes.pop()
        # Ascend 0.26 reports the C4 storage span as block_size; vLLM 0.29
        # reports the logical span and derives the storage axis from
        # tokens_per_state.
        canonical_size = c4_size * 4 if device_type == "npu" else scheduler_block_size
    else:
        canonical_size = scheduler_block_size

    groups: list[UCMKVCacheGroupInfo] = []
    attention_compress_ratio_by_layer: dict[int, int] = {}
    if dsv4:
        for _, concrete, kinds, _ in classified:
            if KVCacheSpecKind.MAMBA in kinds or not kinds.isdisjoint(_SLIDING_KINDS):
                continue
            for fallback, (name, concrete_spec) in enumerate(concrete):
                attention_compress_ratio_by_layer[_layer_index(name, fallback)] = (
                    _spec_compress_ratio(concrete_spec)
                )
    for group_id, (raw_group, concrete, kinds, is_c4a) in enumerate(classified):
        representative = concrete[0][1] if concrete else raw_group.kv_cache_spec
        physical_block_size = int(getattr(raw_group.kv_cache_spec, "block_size"))
        compress_ratio = _spec_compress_ratio(representative)
        token_block_size = (
            physical_block_size * compress_ratio
            if dsv4 and device_type == "npu"
            else physical_block_size
        )
        hash_block_size = canonical_size if dsv4 else physical_block_size
        layers: list[UCMLayerSpec] = []
        for index, (name, spec) in enumerate(concrete):
            # Normalize to the number of stored states one manager block
            # spans.  Ascend 0.26 reports the C4 storage span as block_size
            # directly; vLLM 0.29 reports the logical span and the storage
            # axis is block_size // tokens_per_state (DSV4 C4A: 256/4=64
            # states; C128A: 256/128=2; uncompressed specs keep the block).
            logical = int(getattr(spec, "block_size"))
            ratio = _spec_compress_ratio(spec)
            if device_type == "npu":
                storage_block_size = logical
            elif ratio > 1 and logical % ratio == 0:
                storage_block_size = logical // ratio
            else:
                storage_block_size = logical
            layers.append(
                UCMLayerSpec(
                    name,
                    _layer_index(name, index),
                    spec,
                    storage_block_size,
                )
            )
        tail_tokens: int | None = None
        if dsv4 and not kinds.isdisjoint(_SLIDING_KINDS):
            tails: set[int] = set()
            for fallback, (name, concrete_spec) in enumerate(concrete):
                window = int(getattr(concrete_spec, "sliding_window"))
                if name.lower().endswith("swa_cache"):
                    tail = window
                else:
                    layer_index = _layer_index(name, fallback)
                    if layer_index not in attention_compress_ratio_by_layer:
                        raise ValueError(
                            "Cannot find matching full-attention compression ratio "
                            f"for DSV4 layer {layer_index}"
                        )
                    tail = window - attention_compress_ratio_by_layer[layer_index]
                if tail < 0:
                    raise ValueError(f"Negative DSV4 tail for {name}: {tail}")
                tails.add(tail)
            if len(tails) != 1:
                raise ValueError(
                    f"DSV4 group {group_id} has inconsistent tail sizes {sorted(tails)}"
                )
            tail_tokens = tails.pop()
        groups.append(
            UCMKVCacheGroupInfo(
                group_id=group_id,
                layers=tuple(layers),
                group_spec=raw_group.kv_cache_spec,
                token_block_size=token_block_size,
                hash_block_size=hash_block_size,
                kinds=kinds,
                is_c4a=is_c4a,
                tail_tokens=tail_tokens,
                is_eagle_group=any(
                    "eagle" in layer.layer_name.lower() for layer in layers
                ),
            )
        )

    state_groups = tuple(group for group in groups if group.is_state_snapshot)
    if state_groups:
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
    if dsv4:
        selected_chunk = canonical_size
        alignment = canonical_size
    elif len(groups) == 1:
        selected_chunk = chunk_size or scheduler_block_size
        if (
            selected_chunk < scheduler_block_size
            or selected_chunk % scheduler_block_size
        ):
            raise ValueError(
                "chunk_size must be a positive multiple of scheduler_block_size"
            )
        alignment = scheduler_block_size
    elif state_groups:
        selected_chunk = scheduler_block_size
        # In Mamba align mode vLLM makes the state checkpoint block equal to
        # the final attention/cache block after platform block-size alignment.
        alignment = scheduler_block_size
    else:
        selected_chunk = scheduler_block_size
        alignment = math.lcm(*(group.token_block_size for group in groups))

    if dsv4:
        fa_group_ids = {
            group.group_id
            for group in groups
            if group.is_attention and not group.is_sliding_window
        }
        wa_group_ids = {group.group_id for group in groups if group.is_sliding_window}
        all_group_ids = {group.group_id for group in groups}
        if (
            not fa_group_ids
            or not wa_group_ids
            or fa_group_ids | wa_group_ids != all_group_ids
        ):
            raise ValueError(
                "DeepSeek V4 groups must partition into full-attention FA "
                "and sliding/state WA groups"
            )

    if LAYOUT_DEBUG:
        for group in groups:
            kind_names = ",".join(sorted(kind.value for kind in group.kinds))
            layout_debug(
                f"spec group={group.group_id} layers={group.num_layers} "
                f"kinds={{{kind_names}}} token_block={group.token_block_size} "
                f"hash_block={group.hash_block_size} tail={group.tail_tokens}"
            )
        layout_debug(
            f"spec scheduler_block={scheduler_block_size} "
            f"alignment={alignment} chunk={selected_chunk} "
            f"device={device_type} dsv4={dsv4}"
        )

    return UCMKVCacheSpec(
        groups=tuple(groups),
        scheduler_block_size=scheduler_block_size,
        alignment_block_size=alignment,
        chunk_size=selected_chunk,
        device_type=device_type,
        is_dsv4=dsv4,
    )


class UCMKVCacheLayout:
    """Ragged, per-layer physical layout with deterministic record offsets."""

    def __init__(
        self,
        spec: UCMKVCacheSpec,
        kv_caches: Mapping[str, KVCacheValue],
        *,
        num_blocks: int,
        kv_cache_tensors: Sequence[object] = (),
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.spec = spec
        self.num_blocks = num_blocks
        self.group_layouts: Mapping[int, "GroupLayout"] = build_group_layouts(
            spec,
            kv_caches,
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
        )

    def build_load_batches(
        self, metadata: "UCMConnectorMetadata", layer_name: str | None = None
    ) -> UCMProxyBatch:
        plans = (
            plan
            for request_meta in metadata.requests.values()
            for plan in request_meta.load_plans
        )
        return self._build_batches(plans, layer_name)

    def build_dump_batches(
        self, metadata: "UCMConnectorMetadata", layer_name: str | None = None
    ) -> UCMProxyBatch:
        plans = (
            plan
            for request_meta in metadata.requests.values()
            for plan in request_meta.dump_plans
        )
        return self._build_batches(plans, layer_name)

    def _build_batches(
        self,
        plans: Iterable["UCMGroupDispatchPlan"],
        layer_name: str | None,
    ) -> UCMProxyBatch:
        keys: list[bytes] = []
        offsets: list[int] = []
        ptrs: list[int] = []
        sizes: list[int] = []
        for plan in plans:
            for key, offset, ptr, size in self._iter_plan_segments(plan, layer_name):
                keys.append(key)
                offsets.append(offset)
                ptrs.append(ptr)
                sizes.append(size)
        return UCMProxyBatch(tuple(keys), tuple(offsets), tuple(ptrs), tuple(sizes))

    def _iter_plan_segments(
        self,
        plan: "UCMGroupDispatchPlan",
        layer_name: str | None,
    ) -> Iterator[tuple[bytes, int, int, int]]:
        """Scheduler shell around the layout model's group addressing.

        This layer owns the dispatch semantics -- key splitting, block-map
        resolution, WA tail windows, state-block selection, record offsets
        across groups -- and delegates the byte-span arithmetic to each
        group's precomputed ``GroupLayout``.
        """

        if not plan.keys:
            return
        token_count = plan.token_end - plan.token_start
        if token_count <= 0 or token_count % len(plan.keys):
            raise ValueError("Dispatch token range must divide evenly across keys")
        key_tokens = token_count // len(plan.keys)
        block_maps = {
            selection.group_id: {
                selection.start_block_index + index: block_id
                for index, block_id in enumerate(selection.block_ids)
            }
            for selection in plan.vllm_blocks
        }
        if LAYOUT_DEBUG:
            for group_id in sorted(block_maps):
                block_map = block_maps[group_id]
                pairs = sorted(block_map.items())
                shown = ",".join(f"{k}->{v}" for k, v in pairs[:8])
                more = "" if len(pairs) <= 8 else ",..."
                layout_debug(
                    f"plan hash_group={plan.hash_group} group={group_id} "
                    f"tokens=[{plan.token_start},{plan.token_end}) "
                    f"keys={len(plan.keys)} vllm_blocks {{{shown}{more}}}"
                )
        for key_index, key in enumerate(plan.keys):
            record_offset = 0
            key_start = plan.token_start + key_index * key_tokens
            key_end = key_start + key_tokens
            for group_id in sorted(block_maps):
                group_info = self.spec.groups[group_id]
                group_key_start = key_start
                if plan.hash_group == "WA":
                    if not group_info.tail_tokens:
                        continue
                    group_key_start = max(key_end - group_info.tail_tokens, 0)
                group_layout = self.group_layouts[group_id]
                if group_info.is_state_snapshot:
                    # A state snapshot lives in the last block of its range.
                    spans = group_layout.state_plan_range(
                        block_maps[group_id], key_end
                    )
                else:
                    spans = group_layout.plan_spans(
                        block_maps[group_id], group_key_start, key_end
                    )
                entries, record_offset = group_layout.emit_record(
                    spans, record_offset, layer_name
                )
                for ptr, size, offset in entries:
                    yield key, offset, ptr, size
            if LAYOUT_DEBUG:
                layout_debug(
                    f"record key={key.hex()[:16]}... tokens="
                    f"[{key_start},{key_end}) groups={sorted(block_maps)} "
                    f"record_size={record_offset}"
                )
