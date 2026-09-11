"""View geometry: read addressing facts straight off the runtime views.

The views vLLM hands over at ``register_kv_caches`` are the addressing
source of truth -- each view's data_ptr/shape/stride already encodes
where its layer's blocks sit (0.26: one allocation per view; 0.29: one
packed backing, views strided per the declarations).  This module turns
a view into a :class:`ComponentSlot`; the spec contributes the only two
facts views cannot express: the token->state compression ratio and the
state-page component layout.
"""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING

# Debug trace for the v2 layout: set UCM_V2_LAYOUT_DEBUG=1 to log, on
# stderr, the group layout each group compiled to and which pointer each
# dispatched vLLM block resolves to.  Zero cost when disabled.
LAYOUT_DEBUG = os.environ.get("UCM_V2_LAYOUT_DEBUG", "0") not in ("", "0")


def layout_debug(message: str) -> None:
    if LAYOUT_DEBUG:
        print(f"[ucm-v2-layout] {message}", file=sys.stderr, flush=True)


if TYPE_CHECKING:
    import torch

    from ..ucm_kv_cache import UCMLayerSpec
    from ..ucm_proxy import KVCacheValue


class ComponentSlot:
    """One component's geometry, read straight off its runtime view.

    ``base_ptr`` is the view's block-0 address; block ``b`` starts at
    ``base_ptr + b * block_stride`` and holds ``payload_bytes`` of
    content (``states_per_row`` states of ``bytes_per_state`` bytes per
    kernel row, ``rows_per_block`` rows).
    """

    __slots__ = (
        "base_ptr",
        "block_stride",
        "row_stride_bytes",
        "rows_per_block",
        "states_per_row",
        "bytes_per_state",
    )

    def __init__(
        self,
        base_ptr: int,
        block_stride: int,
        row_stride_bytes: int,
        rows_per_block: int,
        states_per_row: int,
        bytes_per_state: int,
    ) -> None:
        self.base_ptr = base_ptr
        self.block_stride = block_stride
        self.row_stride_bytes = row_stride_bytes
        self.rows_per_block = rows_per_block
        self.states_per_row = states_per_row
        self.bytes_per_state = bytes_per_state

    @property
    def states_per_block(self) -> int:
        return self.rows_per_block * self.states_per_row

    @property
    def payload_bytes(self) -> int:
        return self.rows_per_block * self.states_per_row * self.bytes_per_state

    def __repr__(self) -> str:
        return (
            f"ComponentSlot(base_ptr={self.base_ptr:#x}, "
            f"block_stride={self.block_stride}, "
            f"rows_per_block={self.rows_per_block}, "
            f"states_per_row={self.states_per_row}, "
            f"bytes_per_state={self.bytes_per_state})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComponentSlot):
            return NotImplemented
        return (
            self.base_ptr == other.base_ptr
            and self.block_stride == other.block_stride
            and self.row_stride_bytes == other.row_stride_bytes
            and self.rows_per_block == other.rows_per_block
            and self.states_per_row == other.states_per_row
            and self.bytes_per_state == other.bytes_per_state
        )


def row_payload_bytes(
    shape: tuple[int, ...], strides: tuple[int, ...], element_size: int
) -> int:
    """One row's payload; rejects non-dense trailing dimensions.

    The verified layouts may pad between rows, but each component's
    payload after dimension 0 is dense (possibly a dense permutation of
    dims 1.., as the vLLM 0.29 [B, H, N, C] views over [B, N, H, C]
    memory).
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
    if shape[0] % num_blocks:
        raise ValueError(
            f"KV tensor first dimension {shape[0]} is not divisible by "
            f"num_blocks={num_blocks}"
        )
    element_size = int(tensor.element_size())
    strides = tuple(int(tensor.stride(index)) for index in range(len(shape)))
    row_stride = strides[0] * element_size
    payload = row_payload_bytes(shape, strides, element_size)
    rows_per_block = shape[0] // num_blocks
    if rows_per_block > 1 and row_stride != payload:
        # Align-family layouts (mamba/state models, Kimi MLA) store a
        # block's kernel rows densely by design; a padded multi-row view
        # is a layout we have never seen and cannot address correctly
        # with a single span, so fail fast instead of copying garbage.
        raise ValueError(
            "Padded multi-row blocks are not a supported layout "
            f"(shape={shape}, strides={strides}, row_stride={row_stride}, "
            f"row_payload={payload})"
        )
    states_per_row: int
    bytes_per_state: int
    if state_snapshot:
        # A state snapshot (mamba/SSM page) has no per-token axis: the
        # whole row payload is one indivisible record.
        states_per_row = 1
        bytes_per_state = payload
    else:
        # Map the spec's block size onto the view's rows.  A block spans
        # exactly expected_block_size stored states -- one per token, or
        # one per tokens_per_state on compressed caches (DSV4's C4A:
        # 256-token block = 64 states of 584B).  States never straddle
        # kernel rows, so each row holds expected_block_size //
        # rows_per_block states, each payload // states_per_row bytes.
        # One derivation covers the Ascend 0.26 token-axis dialect
        # (Kimi MLA: one logical block as dense kernel rows) and the
        # vLLM 0.29 permuted [B, H, N, C] views (N counts stored states)
        # identically.
        states_per_row, remainder = divmod(
            expected_block_size, rows_per_block
        )
        if remainder or states_per_row <= 0 or payload % states_per_row:
            raise ValueError(
                "KV tensor does not match a dense row-payload tiling of "
                f"the block: shape={shape}, strides={strides}, "
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
    )


def component_tensors(value: "KVCacheValue") -> tuple["torch.Tensor", ...]:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("KV cache component tuple must not be empty")
        return tuple(value)
    return (value,)


def layer_structures(
    value: "KVCacheValue",
    layer: "UCMLayerSpec",
    *,
    num_blocks: int,
    state_snapshot: bool,
) -> tuple[ComponentSlot, ...]:
    """Resolve one layer's components (attention or state snapshot)."""

    if state_snapshot:
        return state_structures(value, layer, num_blocks=num_blocks)
    tensors = component_tensors(value)
    return tuple(
        component(tensor, layer.storage_block_size, num_blocks=num_blocks)
        for tensor in tensors
    )


def state_structures(
    value: "KVCacheValue",
    layer: "UCMLayerSpec",
    *,
    num_blocks: int,
) -> tuple[ComponentSlot, ...]:
    """Resolve an explicit component tuple or one combined raw state page.

    The combined page stays a single component: whole-block state IO is
    a byte copy, so the conv/SSM split the spec describes adds no
    addressing information.  Ascend 0.26 exposes C = the full padded
    page (payload == page_stride == page_size, padding at the tail);
    vLLM 0.29 exposes C = the dense state content only (payload <=
    page_stride == page_size, padding between blocks).
    """

    tensors = component_tensors(value)
    expected_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in (getattr(layer.kv_cache_spec, "shapes", None) or ())
    )
    actual_shapes = tuple(
        tuple(int(item) for item in tensor.shape[1:]) for tensor in tensors
    )
    if not expected_shapes or actual_shapes == expected_shapes:
        return tuple(
            component(
                tensor,
                layer.storage_block_size,
                num_blocks=num_blocks,
                state_snapshot=True,
            )
            for tensor in tensors
        )

    if len(tensors) != 1:
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
    content = sum(
        math.prod(component_shape) * dtype_size(dtype)
        for component_shape, dtype in zip(expected_shapes, dtypes, strict=True)
    )
    if content > payload:
        raise ValueError(
            f"State components ({content}B) exceed the page content "
            f"({payload}B) for {layer.layer_name}"
        )
    return (
        ComponentSlot(
            base_ptr=int(raw.data_ptr()),
            block_stride=page_stride,
            row_stride_bytes=page_stride,
            rows_per_block=1,
            states_per_row=1,
            bytes_per_state=content,
        ),
    )


def dtype_size(dtype: "torch.dtype") -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    import importlib

    torch = importlib.import_module("torch")
    return int(torch.empty((), dtype=dtype).element_size())
