"""Semantic KV-cache description and the dump/load batch orchestrator for connector v2.

Physical placement -- where every layer's blocks and states sit -- lives in
``.layout``.  This module keeps the semantic layer (groups, cache kinds,
block sizing, the DSV4 policy) and walks dispatch plans over the layout
model to produce proxy batches with deterministic record offsets.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

from .layout import build_group_layouts
from .layout.group import TensorDescriptor
from .layout.view import LAYOUT_DEBUG, layout_debug
from .ucm_proxy import KVCacheValue, UCMProxyBatch

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
    """UCM policy sizes, all in tokens.

    scheduler_block_size is the historical name for CacheConfig.block_size,
    not vLLM's resolved scheduler granularity. chunk_size is UCM's record/hash
    unit; neither adds a new native block-allocation setting.
    """

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
    (int > 1 compresses multiple tokens into one stored state). Mamba specs
    carry a -1 sentinel on 0.29 which is not meaningful as a ratio; callers
    only read this on attention/MLA specs.
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
    attention_tokens_per_state: Mapping[int, int],
    layer_indices: Mapping[str, int],
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
        attention_tokens_per_state.get(
            layer_indices[name], _spec_tokens_per_state(spec)
        )
        == 4
        and kind == KVCacheSpecKind.MLA_ATTENTION
        for (name, spec), kind in zip(concrete, spec_kinds)
    )
    return kinds, is_c4a


def _parse_descriptors(
    kv_cache_tensors: "Sequence[object]",
) -> tuple[tuple["TensorDescriptor", ...], dict[str, tuple["TensorDescriptor", int]]]:
    """Mirror vLLM 0.29 kv_cache_tensors; undeclared configs yield ()."""

    descriptors: list["TensorDescriptor"] = []
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
        descriptors.append(descriptor)
    return tuple(descriptors), declared_at


def parse_kv_cache_config(
    kv_cache_config: "KVCacheConfig",
    *,
    scheduler_block_size: int,
    chunk_size: int | None = None,
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

    source = os.environ.get("UCM_V2_DESCRIPTOR_SOURCE", "auto").strip().lower()
    source = source or "auto"
    if source not in ("auto", "declared"):
        raise ValueError(f"Unknown UCM_V2_DESCRIPTOR_SOURCE={source!r}")
    kv_cache_tensors = tuple(getattr(kv_cache_config, "kv_cache_tensors", ()) or ())
    has_declarations = bool(kv_cache_tensors) and all(
        getattr(tensor, "layers", None) is not None for tensor in kv_cache_tensors
    )
    if source == "declared" and not has_declarations:
        raise ValueError("UCM_V2_DESCRIPTOR_SOURCE=declared requires declared tensors")
    _, declared_at = _parse_descriptors(kv_cache_tensors if has_declarations else ())
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
    layer_indices: dict[str, int] = {}
    for raw_group in raw_groups:
        concrete = _concrete_specs(raw_group)
        layer_indices.update(
            (name, _layer_index(name, device_type, num_hidden_layers))
            for name, _ in concrete
        )
        kinds, is_c4a = _classify(
            raw_group, concrete, attention_tokens_per_state, layer_indices
        )
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
    attention_tokens_per_state_by_layer: dict[int, int] = {}
    if dsv4:
        for _, concrete, kinds, _ in classified:
            if KVCacheSpecKind.MAMBA in kinds or not kinds.isdisjoint(_SLIDING_KINDS):
                continue
            for name, concrete_spec in concrete:
                attention_tokens_per_state_by_layer[layer_indices[name]] = (
                    attention_tokens_per_state.get(
                        layer_indices[name], _spec_tokens_per_state(concrete_spec)
                    )
                )
    num_blocks = int(getattr(kv_cache_config, "num_blocks", 0))
    for group_id, (raw_group, concrete, kinds, is_c4a) in enumerate(classified):
        representative = concrete[0][1] if concrete else raw_group.kv_cache_spec
        physical_block_size = int(getattr(raw_group.kv_cache_spec, "block_size"))
        compress_ratio = _spec_tokens_per_state(representative)
        token_block_size = (
            physical_block_size * compress_ratio
            if dsv4 and device_type == "npu"
            else physical_block_size
        )
        hash_block_size = canonical_size if dsv4 else physical_block_size
        layers: list[UCMLayerSpec] = []
        for index, (name, spec) in enumerate(concrete):
            # Normalize to the number of stored states one group block
            # spans.  Ascend 0.26 reports the C4 storage span as block_size
            # directly; vLLM 0.29 reports the logical span and the storage
            # axis is block_size // tokens_per_state (DSV4 C4A: 256/4=64
            # states; C128A: 256/128=2; uncompressed specs keep the block).
            logical = int(getattr(spec, "block_size"))
            ratio = _spec_tokens_per_state(spec)
            if dsv4 and kinds.isdisjoint(_SLIDING_KINDS):
                ratio = attention_tokens_per_state_by_layer[layer_indices[name]]
            if device_type == "npu":
                storage_block_size = logical
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
        if dsv4 and not kinds.isdisjoint(_SLIDING_KINDS):
            tails: set[int] = set()
            for name, concrete_spec in concrete:
                window = int(getattr(concrete_spec, "sliding_window"))
                if name.lower().endswith("swa_cache"):
                    tail = window
                else:
                    layer_index = layer_indices[name]
                    if layer_index not in attention_tokens_per_state_by_layer:
                        raise ValueError(
                            "Cannot find matching full-attention compression ratio "
                            f"for DSV4 layer {layer_index}"
                        )
                    tail = window - attention_tokens_per_state_by_layer[layer_index]
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


class _GroupWindows(NamedTuple):
    """One key's hash window over one group, as per-block windows.

    ``block_ids`` are physical ids carrying the windows in logical order;
    ``whole`` marks a window that covers every block entirely.
    """

    block_ids: list[int]
    local_starts: list[int]
    local_ends: list[int]
    whole: bool


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
        publishes only when complete. SimpleFileUCMProxy.dump requires the
        whole record and must not be called once per filtered layer batch.
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
        offsets: list[int] = []
        ptrs: list[int] = []
        sizes: list[int] = []
        for plan in plans:
            for key, group_ptrs, group_sizes, group_offsets in self._iter_plan_segments(
                plan, layer_names
            ):
                keys.extend((key,) * len(group_ptrs))
                ptrs.extend(group_ptrs)
                sizes.extend(group_sizes)
                offsets.extend(group_offsets)
        return UCMProxyBatch(tuple(keys), tuple(offsets), tuple(ptrs), tuple(sizes))

    def _iter_plan_segments(
        self,
        plan: "UCMGroupDispatchPlan",
        layer_names: frozenset[str] | None,
    ) -> Iterator[tuple[bytes, list[int], list[int], list[int]]]:
        """Scheduler shell around the group layouts' column addressing.

        This layer owns the dispatch semantics -- key splitting, block-map
        resolution, WA tail windows, state-block selection -- and the
        per-hash-block record ledger.  Layouts only answer (ptr, size)
        grids; this shell places them in one key's record: each group's
        contribution starts at the running block_offset, and the next
        group continues after it.
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
            block_offset = 0
            key_start = plan.token_start + key_index * key_tokens
            key_end = key_start + key_tokens
            for group_id in sorted(block_maps):
                group_info = self.spec.groups[group_id]
                if plan.hash_group == "WA" and not group_info.tail_tokens:
                    continue
                group_layout = self.group_layouts[group_id]
                windows = self._key_windows(
                    plan, group_info, group_layout, block_maps[group_id],
                    key_start, key_end,
                )
                mask = (
                    None
                    if layer_names is None
                    else group_layout.view_mask(layer_names=layer_names)
                )
                ptrs, sizes, offsets, record_bytes = self._group_record(
                    group_layout, windows, mask
                )
                if ptrs:
                    yield (
                        key,
                        ptrs,
                        sizes,
                        [offset + block_offset for offset in offsets],
                    )
                block_offset += record_bytes
            if LAYOUT_DEBUG:
                layout_debug(
                    f"record key={key.hex()[:16]}... tokens="
                    f"[{key_start},{key_end}) groups={sorted(block_maps)} "
                    f"record_size={block_offset}"
                )

    def _key_windows(
        self,
        plan: "UCMGroupDispatchPlan",
        group_info: UCMKVCacheGroupInfo,
        group_layout: "KVCacheGroupLayout",
        block_map: Mapping[int, int],
        key_start: int,
        key_end: int,
    ) -> "_GroupWindows":
        """One key's hash window over one group, as per-block windows.

        Logical block ordinals carry the record; physical ids only
        address.  A state snapshot contributes the last complete block
        as one indivisible page; a WA group stores only its tail window.
        """

        token_block = group_layout.token_block_size
        if group_info.is_state_snapshot:
            logical = max((key_end - 1) // token_block, 0)
            return _GroupWindows([block_map[logical]], [0], [token_block], True)
        window_start = key_start
        if plan.hash_group == "WA":
            window_start = max(key_end - (group_info.tail_tokens or 0), 0)
        first = window_start // token_block
        stop = (key_end - 1) // token_block + 1
        block_ids = [block_map[logical] for logical in range(first, stop)]
        starts = [
            max(window_start - (first + ordinal) * token_block, 0)
            for ordinal in range(stop - first)
        ]
        ends = [
            min(key_end - (first + ordinal) * token_block, token_block)
            for ordinal in range(stop - first)
        ]
        whole = all(
            start == 0 and end == token_block for start, end in zip(starts, ends)
        )
        return _GroupWindows(block_ids, starts, ends, whole)

    def _group_record(
        self,
        group_layout: "KVCacheGroupLayout",
        windows: "_GroupWindows",
        mask: "np.ndarray | None",
    ) -> tuple[list[int], list[int], list[int], int]:
        """Place one group's windows in the key's record.

        Three ledgers, one per group shape:

        - Block First, whole and unfiltered: the block slot is one IO
          span, paddings riding inside (``block_first_segments``).
        - Block First, layered or sub-block: the slot-image ledger --
          offsets are the descriptor anchors, so a layered load lands
          exactly where the span's bytes were dumped.
        - Everything else: layer-major per hash window.  View L's slot
          starts after every earlier view's window bytes: with four
          vLLM blocks per hash and per-block layer size ``s_l``, layer
          0 lands at 0, layer 1 at ``4 * s_0``, layer 2 at
          ``4 * (s_0 + s_1)``, ...  Unselected views still occupy their
          bytes, so a layerwise batch addresses the very same slots.

        Returns flattened (ptrs, sizes, offsets) and the group's record
        size -- the block_offset the next group continues from.
        """

        span = group_layout.block_first if windows.whole else None
        if span is not None and mask is None:
            ptrs, sizes = group_layout.block_first_segments(windows.block_ids)
            offsets = (
                np.arange(len(windows.block_ids), dtype=np.int64)
                * group_layout.record_size
            )
            return (
                ptrs.tolist(),
                sizes.tolist(),
                offsets.tolist(),
                len(windows.block_ids) * group_layout.record_size,
            )
        ptrs, sizes = group_layout.extract_segments(
            windows.block_ids, windows.local_starts, windows.local_ends
        )
        if span is not None:
            # Slot-image ledger: block-major rows at their anchors.
            offsets = (
                np.arange(len(windows.block_ids), dtype=np.int64)[:, None]
                * group_layout.record_size
                + group_layout.record_slots
            )
            record_bytes = len(windows.block_ids) * group_layout.record_size
        else:
            # Layer-major ledger: view slots by window bytes, blocks
            # packed inside each view's slot.
            view_bytes = sizes.sum(axis=0)
            view_slots = np.cumsum(view_bytes) - view_bytes
            within_view = np.cumsum(sizes, axis=0) - sizes
            offsets = view_slots + within_view
            record_bytes = int(view_bytes.sum())
        if mask is not None:
            ptrs = ptrs[:, mask]
            sizes = sizes[:, mask]
            offsets = offsets[:, mask]
        # Slot-image records read block-major; layer-major records read
        # view-major (each view's blocks back to back).
        if span is not None:
            return (
                ptrs.reshape(-1).tolist(),
                sizes.reshape(-1).tolist(),
                offsets.reshape(-1).tolist(),
                record_bytes,
            )
        return (
            ptrs.T.reshape(-1).tolist(),
            sizes.T.reshape(-1).tolist(),
            offsets.T.reshape(-1).tolist(),
            record_bytes,
        )
