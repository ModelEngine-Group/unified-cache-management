"""Runtime-view geometry shared by the declared and inferred layout modes.

Both modes must answer the same physical questions -- where a layer's
components live, how kernel rows tile a block, what one block's contiguous
IO region is -- and both answer them from the tensors that
``register_kv_caches`` hands over.  The modes differ only in where the base
addressing comes from (official declarations vs. synthesis), which this
module deliberately does not know about.

The semantic layer translates every spec spelling before it reaches here:
``layer.storage_block_size`` is always the number of stored states one
manager block spans (Ascend 0.26 reports it as ``block_size``, vLLM 0.29
derives it as ``block_size // tokens_per_state``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .model import BlockRegion, ComponentSlot

if TYPE_CHECKING:
    import torch

    from ..ucm_kv_cache import UCMLayerSpec
    from ..ucm_proxy import KVCacheValue


def component_tensors(value: "KVCacheValue") -> tuple["torch.Tensor", ...]:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("KV cache component tuple must not be empty")
        return tuple(value)
    return (value,)


def row_payload_bytes(
    shape: tuple[int, ...], strides: tuple[int, ...], element_size: int
) -> int:
    """Return one row's payload and reject non-dense trailing dimensions.

    The verified Ascend layouts may pad between rows, but each component payload
    after dimension 0 is dense.  Copying ``stride(0)`` bytes would incorrectly
    include another component (notably Kimi's shared Attention/Mamba page).

    vLLM 0.29 views permute the trailing dims per the resolved KVCacheLayout
    (e.g. LBNHC orders memory [B, N, H, C] while the view exposes [B, H, N, C]),
    so a dense permutation of dims 1.. is accepted as well: sorted by stride,
    consecutive dims must tile exactly (still rejects padding / foreign bytes).
    """

    expected_stride = 1
    for size, stride in zip(reversed(shape[1:]), reversed(strides[1:])):
        if stride != expected_stride:
            break
        expected_stride *= size
    else:
        return expected_stride * element_size

    pairs = sorted(zip(strides[1:], shape[1:]))
    expected_stride = 1
    for stride, size in pairs:
        if stride != expected_stride:
            raise ValueError(
                "KV tensor trailing dimensions must be dense (C-order or a "
                f"dense permutation); shape={shape}, strides={strides}"
            )
        expected_stride *= size
    return expected_stride * element_size


def component(
    tensor: "torch.Tensor",
    expected_block_size: int,
    *,
    num_blocks: int,
    state_snapshot: bool = False,
) -> ComponentSlot:
    """Derive one component's placement from its runtime view."""

    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) < 2 or len(shape) > 4:
        raise ValueError(
            "KV component views must be 2-D, 3-D, or 4-D, "
            f"got shape={shape}"
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
    payload = row_payload_bytes(shape, strides, element_size)
    if row_stride < payload:
        raise ValueError(
            f"KV tensor row stride {row_stride} is smaller than payload {payload}"
        )
    rows_per_block = shape[0] // num_blocks
    states_per_row: int
    bytes_per_state: int
    if state_snapshot:
        states_per_row = 1
        bytes_per_state = payload
    else:
        # Ascend 0.26 attention views keep the token axis on dimension 1 with
        # C-order trailing dims (e.g. Kimi MLA stores one logical block as six
        # kernel rows). vLLM 0.29 exposes uniform [B, H, N, C] views whose dims
        # are permuted by the resolved KVCacheLayout, and whose N axis counts
        # *stored states*, not raw tokens: DSV4's C4A cache stores one 584B
        # state per 4 tokens (256-token block -> 64 states), the C128A variant
        # and its indexer store 2/64 states per 256-token block. Try the dim-1
        # reading first, then derive the row geometry arithmetically: the
        # row payload must tile the expected block exactly.
        states_per_row = shape[1]
        bytes_per_state = strides[1] * element_size
        if (
            rows_per_block * states_per_row != expected_block_size
            or states_per_row * bytes_per_state != payload
        ):
            states_per_row = expected_block_size // rows_per_block
            if (
                expected_block_size % rows_per_block
                or states_per_row <= 0
                or payload % states_per_row
            ):
                raise ValueError(
                    "KV tensor does not match a dense row-payload tiling of "
                    "the block: shape="
                    f"{shape}, strides={strides}, "
                    f"rows_per_block={rows_per_block}, "
                    f"expected_block_size={expected_block_size}, "
                    f"row_payload={payload}"
                )
            bytes_per_state = payload // states_per_row
    return ComponentSlot(
        base_ptr=int(tensor.data_ptr()),
        block_stride=rows_per_block * row_stride,
        row_stride_bytes=row_stride,
        rows_per_block=rows_per_block,
        states_per_row=states_per_row,
        bytes_per_state=bytes_per_state,
        buffer_size_bytes=(shape[0] - 1) * row_stride + payload,
    )


def dense_region(component_slot: ComponentSlot) -> BlockRegion | None:
    """Return a full-block region when all of its dimension-0 slices are dense."""

    if (
        component_slot.rows_per_block > 1
        and component_slot.row_stride_bytes != component_slot.row_payload_bytes
    ):
        return None
    return BlockRegion(
        base_ptr=component_slot.base_ptr,
        block_stride=component_slot.rows_per_block * component_slot.row_stride_bytes,
        payload_bytes=(
            component_slot.rows_per_block * component_slot.row_payload_bytes
        ),
        buffer_size_bytes=component_slot.buffer_size_bytes,
    )


def dense_regions(
    components: Sequence[ComponentSlot],
) -> tuple[BlockRegion, ...]:
    regions = tuple(dense_region(component_slot) for component_slot in components)
    if any(region is None for region in regions):
        return ()
    return tuple(region for region in regions if region is not None)


def dtype_size(dtype: "torch.dtype") -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    import importlib

    torch = importlib.import_module("torch")
    return int(torch.empty((), dtype=dtype).element_size())


def layer_structures(
    value: "KVCacheValue",
    layer: "UCMLayerSpec",
    *,
    num_blocks: int,
    state_snapshot: bool,
) -> tuple[tuple[ComponentSlot, ...], tuple[BlockRegion, ...]]:
    """Resolve one layer's components and whole-block IO regions."""

    if state_snapshot:
        return state_structures(value, layer, num_blocks=num_blocks)
    return attention_structures(value, layer, num_blocks=num_blocks)


def attention_structures(
    value: "KVCacheValue",
    layer: "UCMLayerSpec",
    *,
    num_blocks: int,
) -> tuple[tuple[ComponentSlot, ...], tuple[BlockRegion, ...]]:
    """Resolve actual component containers without imposing a platform policy.

    Ascend 0.26 hands over explicit K/V (and index/scale) tuples; vLLM 0.29
    hands over one packed view per layer.  Both become one ComponentSlot per
    tensor; higher-rank views are rejected by :func:`component`.
    """

    tensors = component_tensors(value)
    components = tuple(
        component(tensor, layer.storage_block_size, num_blocks=num_blocks)
        for tensor in tensors
    )
    return components, dense_regions(components)


def state_structures(
    value: "KVCacheValue",
    layer: "UCMLayerSpec",
    *,
    num_blocks: int,
) -> tuple[tuple[ComponentSlot, ...], tuple[BlockRegion, ...]]:
    """Resolve an explicit component tuple or one combined raw state page."""

    tensors = component_tensors(value)
    expected_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in (getattr(layer.kv_cache_spec, "shapes", None) or ())
    )
    actual_shapes = tuple(
        tuple(int(item) for item in tensor.shape[1:]) for tensor in tensors
    )
    if not expected_shapes or actual_shapes == expected_shapes:
        components = tuple(
            component(
                tensor,
                layer.storage_block_size,
                num_blocks=num_blocks,
                state_snapshot=True,
            )
            for tensor in tensors
        )
        return components, dense_regions(components)

    if len(tensors) != 1 or not expected_shapes:
        raise ValueError(
            f"State components for {layer.layer_name} do not match spec shapes: "
            f"{actual_shapes} != {expected_shapes}"
        )

    raw = tensors[0]
    shape = tuple(int(item) for item in raw.shape)
    strides = tuple(int(raw.stride(index)) for index in range(len(shape)))
    element_size = int(raw.element_size())
    if shape[0] != num_blocks or element_size != 1:
        raise ValueError(
            "Combined state backing must be one byte page per block: "
            f"shape={shape}, element_size={element_size}, num_blocks={num_blocks}"
        )
    page_stride = strides[0] * element_size
    payload = row_payload_bytes(shape, strides, element_size)
    page_size = int(getattr(layer.kv_cache_spec, "page_size_bytes", page_stride))
    # Ascend 0.26 exposes C = the full padded page (payload == page_stride
    # == page_size, padding at the page tail). vLLM 0.29 exposes C = the dense
    # state content only, with the page padding between blocks
    # (payload <= page_stride == page_size). Components are carved from the
    # front of each page in both forms; the page padding stays outside the
    # record either way.
    if page_stride != page_size or payload > page_stride:
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
    components: list[ComponentSlot] = []
    for component_shape, dtype in zip(expected_shapes, dtypes, strict=True):
        component_size = math.prod(component_shape) * dtype_size(dtype)
        if offset + component_size > page_stride:
            raise ValueError(
                f"State components exceed padded page for {layer.layer_name}"
            )
        components.append(
            ComponentSlot(
                base_ptr=int(raw.data_ptr()) + offset,
                block_stride=page_stride,
                row_stride_bytes=page_stride,
                rows_per_block=1,
                states_per_row=1,
                bytes_per_state=component_size,
                buffer_size_bytes=(num_blocks - 1) * page_stride + component_size,
            )
        )
        offset += component_size
    # Keep the meaningful conv/SSM payload together because the actual runtime
    # value is one block-major page.  Page padding remains outside the record.
    region = BlockRegion(
        base_ptr=int(raw.data_ptr()),
        block_stride=page_stride,
        payload_bytes=offset,
        buffer_size_bytes=(num_blocks - 1) * page_stride + offset,
    )
    return tuple(components), (region,)
