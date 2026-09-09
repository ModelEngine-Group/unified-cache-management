"""Read-only KV-cache description and ragged runtime layout for connector v2."""

from __future__ import annotations

import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

from .ucm_proxy import KVCacheValue, UCMProxyBatch

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )

    from .ucm_scheduler import UCMConnectorMetadata, UCMGroupDispatchPlan


_SLIDING_KINDS = frozenset(
    (KVCacheSpecKind.SLIDING_WINDOW, KVCacheSpecKind.SLIDING_WINDOW_MLA)
)

# Debug trace for the v2 layout: set UCM_V2_LAYOUT_DEBUG=1 to log, on stderr,
# what UCMKVCacheLayout registers per layer and which pointer each allocated
# vLLM block resolves to.  Zero cost when disabled.
_LAYOUT_DEBUG = os.environ.get("UCM_V2_LAYOUT_DEBUG", "0") not in ("", "0")


def _layout_debug(message: str) -> None:
    if _LAYOUT_DEBUG:
        print(f"[ucm-v2-layout] {message}", file=sys.stderr, flush=True)


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
        getattr(spec, "compress_ratio", 1) == 4
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
        # Ascend 0.26 reports the C4 storage span as block_size. Official
        # vLLM 0.27 reports the logical span and exposes storage_block_size
        # separately on each concrete spec.
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
                attention_compress_ratio_by_layer[_layer_index(name, fallback)] = int(
                    getattr(concrete_spec, "compress_ratio", 1)
                )
    for group_id, (raw_group, concrete, kinds, is_c4a) in enumerate(classified):
        representative = concrete[0][1] if concrete else raw_group.kv_cache_spec
        physical_block_size = int(getattr(raw_group.kv_cache_spec, "block_size"))
        compress_ratio = int(getattr(representative, "compress_ratio", 1))
        token_block_size = (
            physical_block_size * compress_ratio
            if dsv4 and device_type == "npu"
            else physical_block_size
        )
        hash_block_size = canonical_size if dsv4 else physical_block_size
        layers: list[UCMLayerSpec] = []
        for index, (name, spec) in enumerate(concrete):
            storage_block_size = (
                getattr(spec, "storage_block_size", None)
                if device_type != "npu"
                else getattr(spec, "block_size", None)
            )
            layers.append(
                UCMLayerSpec(
                    name,
                    _layer_index(name, index),
                    spec,
                    int(storage_block_size or physical_block_size),
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

    if _LAYOUT_DEBUG:
        for group in groups:
            kind_names = ",".join(sorted(kind.value for kind in group.kinds))
            _layout_debug(
                f"spec group={group.group_id} layers={group.num_layers} "
                f"kinds={{{kind_names}}} token_block={group.token_block_size} "
                f"hash_block={group.hash_block_size} tail={group.tail_tokens}"
            )
        _layout_debug(
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


@dataclass(frozen=True)
class UCMTensorViewLayout:
    base_ptr: int
    row_stride_bytes: int
    token_stride_bytes: int
    tokens_per_row: int
    rows_per_vllm_block: int
    bytes_per_token: int
    row_payload_bytes: int
    buffer_size_bytes: int


@dataclass(frozen=True)
class UCMBlockIORegion:
    """One naturally contiguous IO region for each vLLM physical block."""

    base_ptr: int
    block_stride_bytes: int
    block_payload_bytes: int
    buffer_size_bytes: int


@dataclass(frozen=True)
class UCMLayerKVCacheLayout:
    layer_name: str
    layer_index: int
    group_id: int
    views: tuple[UCMTensorViewLayout, ...]
    block_regions: tuple[UCMBlockIORegion, ...]


@dataclass(frozen=True)
class UCMGroupKVCacheLayout:
    group_id: int
    layers: tuple[UCMLayerKVCacheLayout, ...]


def _tensor_views(tensor: KVCacheValue) -> tuple["torch.Tensor", ...]:
    if isinstance(tensor, (tuple, list)):
        if not tensor:
            raise ValueError("KV cache component tuple must not be empty")
        return tuple(tensor)
    return (tensor,)


def _row_payload_bytes(
    shape: tuple[int, ...], strides: tuple[int, ...], element_size: int
) -> int:
    """Return one row's payload and reject non-dense trailing dimensions.

    The verified Ascend layouts may pad between rows, but each component payload
    after dimension 0 is dense.  Copying ``stride(0)`` bytes would incorrectly
    include another component (notably Kimi's shared Attention/Mamba page).
    """

    expected_stride = 1
    for size, stride in zip(reversed(shape[1:]), reversed(strides[1:])):
        if stride != expected_stride:
            raise ValueError(
                "KV tensor trailing dimensions must be dense; "
                f"shape={shape}, strides={strides}"
            )
        expected_stride *= size
    return expected_stride * element_size


def _view_layout(
    tensor: "torch.Tensor",
    expected_block_size: int,
    *,
    num_blocks: int,
    state_snapshot: bool = False,
) -> UCMTensorViewLayout:
    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) < 2 or len(shape) > 4:
        raise ValueError(
            "KV component views must be 2-D, 3-D, or 4-D, " f"got shape={shape}"
        )
    if any(value <= 0 for value in shape):
        raise ValueError(f"KV tensor dimensions must be positive, got shape={shape}")
    if shape[0] % num_blocks:
        raise ValueError(
            f"KV tensor first dimension {shape[0]} is not divisible by "
            f"num_blocks={num_blocks}"
        )
    element_size = int(tensor.element_size())
    strides = tuple(int(tensor.stride(index)) for index in range(len(shape)))
    row_stride = strides[0] * element_size
    row_payload = _row_payload_bytes(shape, strides, element_size)
    if row_stride < row_payload:
        raise ValueError(
            f"KV tensor row stride {row_stride} is smaller than payload {row_payload}"
        )
    rows_per_block = shape[0] // num_blocks
    if state_snapshot:
        return UCMTensorViewLayout(
            base_ptr=int(tensor.data_ptr()),
            row_stride_bytes=row_stride,
            token_stride_bytes=row_payload,
            tokens_per_row=1,
            rows_per_vllm_block=rows_per_block,
            bytes_per_token=row_payload,
            row_payload_bytes=row_payload,
            buffer_size_bytes=(shape[0] - 1) * row_stride + row_payload,
        )

    # All currently verified attention/cache component views use dimension 1
    # as the storage-token axis. Kimi MLA is the important non-trivial case:
    # (num_blocks * 6, 128, 1, width) represents one logical 768-token block.
    tokens_per_row = shape[1]
    if rows_per_block * tokens_per_row != expected_block_size:
        raise ValueError(
            "KV tensor does not match the concrete cache spec: "
            f"rows_per_vllm_block={rows_per_block}, tokens_per_row={tokens_per_row}, "
            f"expected physical block_size={expected_block_size}"
        )
    token_stride = strides[1] * element_size
    bytes_per_token = token_stride
    if tokens_per_row * bytes_per_token != row_payload:
        raise ValueError(
            "KV tensor token axis does not cover a dense row payload: "
            f"shape={shape}, strides={strides}"
        )
    return UCMTensorViewLayout(
        base_ptr=int(tensor.data_ptr()),
        row_stride_bytes=row_stride,
        token_stride_bytes=token_stride,
        tokens_per_row=tokens_per_row,
        rows_per_vllm_block=rows_per_block,
        bytes_per_token=bytes_per_token,
        row_payload_bytes=row_payload,
        buffer_size_bytes=(shape[0] - 1) * row_stride + row_payload,
    )


def _block_region_from_view(
    view: UCMTensorViewLayout,
) -> UCMBlockIORegion | None:
    """Return a full-block region when all of its dimension-0 slices are dense."""

    if view.rows_per_vllm_block > 1 and view.row_stride_bytes != view.row_payload_bytes:
        return None
    block_stride = view.rows_per_vllm_block * view.row_stride_bytes
    block_payload = view.rows_per_vllm_block * view.row_payload_bytes
    return UCMBlockIORegion(
        base_ptr=view.base_ptr,
        block_stride_bytes=block_stride,
        block_payload_bytes=block_payload,
        buffer_size_bytes=view.buffer_size_bytes,
    )


def _regions_from_views(
    views: Sequence[UCMTensorViewLayout],
) -> tuple[UCMBlockIORegion, ...]:
    regions = tuple(_block_region_from_view(view) for view in views)
    if any(region is None for region in regions):
        return ()
    return tuple(region for region in regions if region is not None)


def _raw_block_region(
    tensor: "torch.Tensor",
    *,
    num_blocks: int,
    payload_bytes: int | None = None,
) -> UCMBlockIORegion:
    """Describe one block-major tensor as one native IO page per block."""

    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) < 2 or shape[0] != num_blocks:
        raise ValueError(
            "Block-major KV tensor must have num_blocks on dimension 0, "
            f"got shape={shape}, num_blocks={num_blocks}"
        )
    element_size = int(tensor.element_size())
    strides = tuple(int(tensor.stride(index)) for index in range(len(shape)))
    block_stride = strides[0] * element_size
    dense_payload = _row_payload_bytes(shape, strides, element_size)
    if block_stride < dense_payload:
        raise ValueError(
            f"KV block stride {block_stride} is smaller than payload {dense_payload}"
        )
    block_payload = dense_payload if payload_bytes is None else int(payload_bytes)
    if block_payload <= 0 or block_payload > dense_payload:
        raise ValueError(
            f"KV block payload {block_payload} is outside dense page size {dense_payload}"
        )
    return UCMBlockIORegion(
        base_ptr=int(tensor.data_ptr()),
        block_stride_bytes=block_stride,
        block_payload_bytes=block_payload,
        buffer_size_bytes=(num_blocks - 1) * block_stride + block_payload,
    )


def _attention_layouts(
    value: KVCacheValue,
    layer: UCMLayerSpec,
    *,
    num_blocks: int,
) -> tuple[tuple[UCMTensorViewLayout, ...], tuple[UCMBlockIORegion, ...]]:
    """Resolve actual component containers without imposing a platform policy."""

    if isinstance(value, (tuple, list)):
        raw_views = _tensor_views(value)
        views = tuple(
            _view_layout(
                view,
                layer.storage_block_size,
                num_blocks=num_blocks,
            )
            for view in raw_views
        )
        return views, _regions_from_views(views)

    shape = tuple(int(item) for item in value.shape)
    if len(shape) != 5:
        views = (
            _view_layout(
                value,
                layer.storage_block_size,
                num_blocks=num_blocks,
            ),
        )
        return views, _regions_from_views(views)

    if (
        shape[0] != num_blocks
        or shape[1] != 2
        or not callable(getattr(value, "unbind", None))
    ):
        raise ValueError(
            "Verified combined K/V tensors require block-major shape "
            f"(num_blocks, 2, ...), got shape={shape}, num_blocks={num_blocks}"
        )
    component_tensors = tuple(value.unbind(1))
    if len(component_tensors) != 2:
        raise ValueError(f"Combined KV tensor did not produce K/V views: {shape}")
    views = tuple(
        _view_layout(
            component,
            layer.storage_block_size,
            num_blocks=num_blocks,
        )
        for component in component_tensors
    )
    # The actual runtime value is one block-major tensor.  Keep K/V together
    # as its native page for full-block IO; the component views remain only for
    # exact sub-block addressing.
    return views, (_raw_block_region(value, num_blocks=num_blocks),)


def _dtype_size(dtype: "torch.dtype") -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    import importlib

    torch = importlib.import_module("torch")
    return int(torch.empty((), dtype=dtype).element_size())


def _state_layouts(
    value: KVCacheValue,
    layer: UCMLayerSpec,
    *,
    num_blocks: int,
) -> tuple[tuple[UCMTensorViewLayout, ...], tuple[UCMBlockIORegion, ...]]:
    """Resolve an explicit component tuple or one combined raw state page."""

    raw_views = _tensor_views(value)
    expected_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in (getattr(layer.kv_cache_spec, "shapes", None) or ())
    )
    actual_shapes = tuple(
        tuple(int(item) for item in view.shape[1:]) for view in raw_views
    )
    if not expected_shapes:
        views = tuple(
            _view_layout(
                view,
                layer.storage_block_size,
                num_blocks=num_blocks,
                state_snapshot=True,
            )
            for view in raw_views
        )
        return views, _regions_from_views(views)
    if expected_shapes and actual_shapes == expected_shapes:
        views = tuple(
            _view_layout(
                view,
                layer.storage_block_size,
                num_blocks=num_blocks,
                state_snapshot=True,
            )
            for view in raw_views
        )
        return views, _regions_from_views(views)

    if len(raw_views) != 1 or not expected_shapes:
        raise ValueError(
            f"State components for {layer.layer_name} do not match spec shapes: "
            f"{actual_shapes} != {expected_shapes}"
        )

    raw = raw_views[0]
    shape = tuple(int(item) for item in raw.shape)
    strides = tuple(int(raw.stride(index)) for index in range(len(shape)))
    element_size = int(raw.element_size())
    if shape[0] != num_blocks or element_size != 1:
        raise ValueError(
            "Combined state backing must be one byte page per block: "
            f"shape={shape}, element_size={element_size}, num_blocks={num_blocks}"
        )
    row_stride = strides[0] * element_size
    row_payload = _row_payload_bytes(shape, strides, element_size)
    page_size = int(getattr(layer.kv_cache_spec, "page_size_bytes", row_stride))
    if row_payload != row_stride or row_stride != page_size:
        raise ValueError(
            "Combined state backing must be a dense padded page: "
            f"shape={shape}, strides={strides}, page_size={page_size}"
        )

    dtypes = tuple(getattr(layer.kv_cache_spec, "dtypes", ()) or ())
    if len(dtypes) != len(expected_shapes):
        raise ValueError(
            f"State spec for {layer.layer_name} must provide one dtype per shape"
        )
    offset = 0
    layouts: list[UCMTensorViewLayout] = []
    for component_shape, dtype in zip(expected_shapes, dtypes, strict=True):
        component_size = math.prod(component_shape) * _dtype_size(dtype)
        if offset + component_size > row_stride:
            raise ValueError(
                f"State components exceed padded page for {layer.layer_name}"
            )
        layouts.append(
            UCMTensorViewLayout(
                base_ptr=int(raw.data_ptr()) + offset,
                row_stride_bytes=row_stride,
                token_stride_bytes=component_size,
                tokens_per_row=1,
                rows_per_vllm_block=1,
                bytes_per_token=component_size,
                row_payload_bytes=component_size,
                buffer_size_bytes=(num_blocks - 1) * row_stride + component_size,
            )
        )
        offset += component_size
    # Keep the meaningful conv/SSM payload together because the actual runtime
    # value is one block-major page.  Page padding remains outside the record.
    region = _raw_block_region(raw, num_blocks=num_blocks, payload_bytes=offset)
    return tuple(layouts), (region,)


class UCMKVCacheLayout:
    """Ragged, per-layer physical layout with deterministic record offsets."""

    def __init__(
        self,
        spec: UCMKVCacheSpec,
        kv_caches: Mapping[str, KVCacheValue],
        *,
        num_blocks: int,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.spec = spec
        self.num_blocks = num_blocks
        groups: dict[int, UCMGroupKVCacheLayout] = {}
        layers: dict[str, UCMLayerKVCacheLayout] = {}
        for group in spec.groups:
            group_layers: list[UCMLayerKVCacheLayout] = []
            for layer in sorted(
                group.layers,
                key=lambda item: (item.layer_index, item.layer_name),
            ):
                if layer.layer_name not in kv_caches:
                    raise ValueError(f"Missing KV cache tensor for {layer.layer_name}")
                if group.is_state_snapshot:
                    parsed_views, block_regions = _state_layouts(
                        kv_caches[layer.layer_name],
                        layer,
                        num_blocks=num_blocks,
                    )
                else:
                    parsed_views, block_regions = _attention_layouts(
                        kv_caches[layer.layer_name],
                        layer,
                        num_blocks=num_blocks,
                    )
                if group.is_state_snapshot and any(
                    view.rows_per_vllm_block != 1 for view in parsed_views
                ):
                    raise ValueError(
                        "State components require exactly one "
                        f"physical row per vLLM block for {layer.layer_name}"
                    )
                if _LAYOUT_DEBUG:
                    for view_index, view in enumerate(parsed_views):
                        _layout_debug(
                            f"register num_blocks={num_blocks} "
                            f"group={group.group_id} layer={layer.layer_name} "
                            f"view={view_index} "
                            f"state={int(group.is_state_snapshot)} "
                            f"base_ptr={view.base_ptr:#x} "
                            f"rows_per_block={view.rows_per_vllm_block} "
                            f"tokens_per_row={view.tokens_per_row} "
                            f"row_stride={view.row_stride_bytes} "
                            f"payload={view.row_payload_bytes} "
                            f"bytes_per_token={view.bytes_per_token} "
                            f"buffer={view.buffer_size_bytes}"
                        )
                    for region_index, region in enumerate(block_regions):
                        _layout_debug(
                            f"io-region group={group.group_id} "
                            f"layer={layer.layer_name} region={region_index} "
                            f"base_ptr={region.base_ptr:#x} "
                            f"block_stride={region.block_stride_bytes} "
                            f"block_payload={region.block_payload_bytes} "
                            f"buffer={region.buffer_size_bytes}"
                        )
                item = UCMLayerKVCacheLayout(
                    layer.layer_name,
                    layer.layer_index,
                    group.group_id,
                    parsed_views,
                    block_regions,
                )
                group_layers.append(item)
                layers[layer.layer_name] = item
            groups[group.group_id] = UCMGroupKVCacheLayout(
                group.group_id, tuple(group_layers)
            )
        self.groups = groups
        self.layers = layers

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
            self._append_plan(plan, layer_name, keys, offsets, ptrs, sizes)
        return UCMProxyBatch(tuple(keys), tuple(offsets), tuple(ptrs), tuple(sizes))

    def _append_plan(
        self,
        plan: "UCMGroupDispatchPlan",
        layer_name: str | None,
        keys: list[bytes],
        offsets: list[int],
        ptrs: list[int],
        sizes: list[int],
    ) -> None:
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
        if _LAYOUT_DEBUG:
            for group_id in sorted(block_maps):
                block_map = block_maps[group_id]
                pairs = sorted(block_map.items())
                shown = ",".join(f"{k}->{v}" for k, v in pairs[:8])
                more = "" if len(pairs) <= 8 else ",..."
                _layout_debug(
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
                group_layout = self.groups[group_id]
                group_key_start = key_start
                if plan.hash_group == "WA":
                    if not group_info.tail_tokens:
                        continue
                    group_key_start = max(key_end - group_info.tail_tokens, 0)
                for layer in group_layout.layers:
                    segments = self._segments_for_block_regions(
                        group_info,
                        layer.block_regions,
                        block_maps[group_id],
                        group_key_start,
                        key_end,
                        layer_name=layer.layer_name,
                    )
                    if segments is None:
                        segments = tuple(
                            segment
                            for view in layer.views
                            for segment in self._segments_for_view(
                                group_info,
                                view,
                                block_maps[group_id],
                                group_key_start,
                                key_end,
                                layer_name=layer.layer_name,
                            )
                        )
                    for ptr, size in segments:
                        if layer_name is None or layer.layer_name == layer_name:
                            keys.append(key)
                            offsets.append(record_offset)
                            ptrs.append(ptr)
                            sizes.append(size)
                        record_offset += size
            if _LAYOUT_DEBUG:
                _layout_debug(
                    f"record key={key.hex()[:16]}... tokens="
                    f"[{key_start},{key_end}) groups={sorted(block_maps)} "
                    f"record_size={record_offset}"
                )

    def _segments_for_block_regions(
        self,
        group: UCMKVCacheGroupInfo,
        regions: Sequence[UCMBlockIORegion],
        block_map: Mapping[int, int],
        token_start: int,
        token_end: int,
        *,
        layer_name: str = "",
    ) -> tuple[tuple[int, int], ...] | None:
        if not regions:
            return None
        logical_blocks: tuple[int, ...]
        if group.is_state_snapshot:
            logical_blocks = (max((token_end - 1) // group.token_block_size, 0),)
        else:
            first = token_start // group.token_block_size
            last = (token_end - 1) // group.token_block_size
            logical_blocks = tuple(range(first, last + 1))
            if any(
                max(token_start, block * group.token_block_size)
                != block * group.token_block_size
                or min(token_end, (block + 1) * group.token_block_size)
                != (block + 1) * group.token_block_size
                for block in logical_blocks
            ):
                return None

        result: list[tuple[int, int]] = []
        # Preserve the component structure exposed by the runtime value.  A
        # combined block-major tensor has one region; an Ascend tuple/list has
        # one region per explicit K/V, index/scale, or conv/SSM component.
        for region in regions:
            for logical_block in logical_blocks:
                if logical_block not in block_map:
                    raise ValueError(
                        f"Missing vLLM block {logical_block} for group {group.group_id}"
                    )
                block_id = block_map[logical_block]
                if block_id < 0 or block_id >= self.num_blocks:
                    raise ValueError(
                        f"vLLM block ID {block_id} is outside [0, {self.num_blocks})"
                    )
                ptr = region.base_ptr + block_id * region.block_stride_bytes
                size = region.block_payload_bytes
                if ptr + size > region.base_ptr + region.buffer_size_bytes:
                    raise ValueError(
                        "KV cache IO region exceeds registered tensor buffer"
                    )
                if _LAYOUT_DEBUG:
                    _layout_debug(
                        f"io-segment group={group.group_id} layer={layer_name} "
                        f"block={block_id} ptr={ptr:#x} size={size}"
                    )
                result.append((ptr, size))
        return tuple(result)

    def _segments_for_view(
        self,
        group: UCMKVCacheGroupInfo,
        view: UCMTensorViewLayout,
        block_map: Mapping[int, int],
        token_start: int,
        token_end: int,
        *,
        layer_name: str = "",
    ) -> tuple[tuple[int, int], ...]:
        if group.is_state_snapshot:
            logical_block = max((token_end - 1) // group.token_block_size, 0)
            if logical_block not in block_map:
                raise ValueError(
                    f"Missing state checkpoint block {logical_block} "
                    f"for group {group.group_id}"
                )
            block_id = block_map[logical_block]
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(
                    f"vLLM block ID {block_id} is outside [0, {self.num_blocks})"
                )
            if _LAYOUT_DEBUG:
                _layout_debug(
                    f"state-segment group={group.group_id} "
                    f"layer={layer_name} block={block_id} "
                    f"rows={view.rows_per_vllm_block} "
                    f"ptr={view.base_ptr + block_id * view.rows_per_vllm_block * view.row_stride_bytes:#x} "
                    f"view_off={block_id * view.rows_per_vllm_block * view.row_stride_bytes} "
                    f"size={view.rows_per_vllm_block * view.row_payload_bytes}"
                )
            state_segments: list[tuple[int, int]] = []
            first_row = block_id * view.rows_per_vllm_block
            for row in range(view.rows_per_vllm_block):
                ptr = view.base_ptr + (first_row + row) * view.row_stride_bytes
                size = view.row_payload_bytes
                if ptr + size > view.base_ptr + view.buffer_size_bytes:
                    raise ValueError(
                        "KV cache state segment exceeds registered tensor buffer"
                    )
                state_segments.append((ptr, size))
            return tuple(state_segments)
        result: list[tuple[int, int]] = []
        previous_block_id: int | None = None

        def append_segment(ptr: int, size: int, block_id: int) -> None:
            nonlocal previous_block_id
            if not size:
                return
            if (
                result
                and previous_block_id == block_id
                and result[-1][0] + result[-1][1] == ptr
            ):
                previous_ptr, previous_size = result[-1]
                result[-1] = (previous_ptr, previous_size + size)
            else:
                result.append((ptr, size))
            previous_block_id = block_id

        first = token_start // group.token_block_size
        last = (token_end - 1) // group.token_block_size
        for logical_block in range(first, last + 1):
            if logical_block not in block_map:
                raise ValueError(
                    f"Missing vLLM block {logical_block} for group {group.group_id}"
                )
            block_id = block_map[logical_block]
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(
                    f"vLLM block ID {block_id} is outside [0, {self.num_blocks})"
                )
            logical_begin = max(token_start, logical_block * group.token_block_size)
            logical_end = min(token_end, (logical_block + 1) * group.token_block_size)
            storage_tokens = view.rows_per_vllm_block * view.tokens_per_row
            numerator_begin = (
                logical_begin - logical_block * group.token_block_size
            ) * storage_tokens
            numerator_end = (
                logical_end - logical_block * group.token_block_size
            ) * storage_tokens
            if (
                numerator_begin % group.token_block_size
                or numerator_end % group.token_block_size
            ):
                raise ValueError(
                    "Logical token range cannot be represented exactly by tensor layout"
                )
            physical_begin = numerator_begin // group.token_block_size
            physical_end = numerator_end // group.token_block_size
            while physical_begin < physical_end:
                row_in_block, token_in_row = divmod(physical_begin, view.tokens_per_row)
                row_end = min(
                    physical_end,
                    (row_in_block + 1) * view.tokens_per_row,
                )
                row_index = block_id * view.rows_per_vllm_block + row_in_block
                ptr = (
                    view.base_ptr
                    + row_index * view.row_stride_bytes
                    + token_in_row * view.token_stride_bytes
                )
                size = (row_end - physical_begin) * view.bytes_per_token
                if ptr + size > view.base_ptr + view.buffer_size_bytes:
                    raise ValueError(
                        "KV cache segment exceeds registered tensor buffer"
                    )
                if _LAYOUT_DEBUG:
                    _layout_debug(
                        f"segment group={group.group_id} layer={layer_name} "
                        f"block={block_id} row={row_in_block}/"
                        f"{view.rows_per_vllm_block} "
                        f"tokens=[{logical_begin},{logical_end}) "
                        f"ptr={ptr:#x} view_off={ptr - view.base_ptr} size={size}"
                    )
                append_segment(ptr, size, block_id)
                physical_begin = row_end
        return tuple(result)
