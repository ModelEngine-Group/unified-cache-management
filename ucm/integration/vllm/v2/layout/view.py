"""Compile registered logical views into contiguous source-memory segments.

LayerView groups a layer name's components; components can share physical
storage. Backend axis conventions and runtime shape/stride define addressing.
The spec supplies logical block sizes, compression ratios and state-page
semantics. No KV storage is allocated or copied here.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class MemorySegment:
    """One contiguous token/state segment within each logical block.

    ``base_ptr`` is this segment's block-0 address; block ``b`` starts at
    ``base_ptr + b * block_stride_bytes`` and holds ``payload_bytes`` of
    content as ``states_per_block`` states of ``bytes_per_state`` bytes,
    evenly spaced (dense-row views; the row geometry is a derivation
    detail, consumed inside :func:`build_tensor_view` only).
    """

    base_ptr: int
    block_stride_bytes: int
    states_per_block: int
    bytes_per_state: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class ComponentView:
    """One registered tensor view, potentially split into head segments.

    No storage is allocated. Segment bases include the view's storage offset
    already, because they are derived from tensor.data_ptr().
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    segments: tuple[MemorySegment, ...]


@dataclass(frozen=True, slots=True)
class LayerView:
    """Logical layer-name view; components may alias the same storage."""

    layer_name: str
    layer_id: int
    components: tuple[ComponentView, ...]

    @property
    def segments(self) -> tuple[MemorySegment, ...]:
        return tuple(s for component in self.components for s in component.segments)


def build_layer_view(
    value: "torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]",
    layer: "UCMLayerSpec",
    *,
    state_snapshot: bool,
    device_type: str,
) -> LayerView:
    """Normalize the two supported runtime ABIs, not dimension-size guesses.

    Official 0.29 attention views are logically BHNC regardless of physical
    stride order. Ascend 0.26 attention views are BNHC. MLA may expose BNC;
    either ABI may tile a logical block with multiple kernel rows. State
    components retain their explicit spec/page interpretation.
    """
    tensors = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if state_snapshot:
        segments = _state_tensor_views(tensors, layer)
        return LayerView(
            layer.layer_name,
            layer.layer_index,
            tuple(
                ComponentView(
                    tuple(t.shape),
                    tuple(t.stride(i) for i in range(len(t.shape))),
                    (s,),
                )
                for t, s in zip(tensors, segments, strict=True)
            ),
        )
    components = []
    for tensor in tensors:
        shape = tuple(int(x) for x in tensor.shape)
        strides = tuple(int(tensor.stride(i)) for i in range(len(shape)))
        if len(shape) == 4:
            token_axis, head_axis = (1, 2) if device_type == "npu" else (2, 1)
            segments = _attention_segments(tensor, layer, token_axis, head_axis)
        else:
            segments = (build_tensor_view(tensor, layer),)
        components.append(ComponentView(shape, strides, segments))
    return LayerView(layer.layer_name, layer.layer_index, tuple(components))


def _attention_segments(
    tensor: "torch.Tensor",
    layer: "UCMLayerSpec",
    token_axis: int,
    head_axis: int,
) -> tuple[MemorySegment, ...]:
    """Keep NHC contiguous; split HNC into independently addressable heads.

    Head fragments keep the original whole-page byte order for compact HNC.
    Cross-block/cross-layer head strides (LHBNC/BHLNC) use the same formula.
    Multiple kernel rows with separated heads require a ragged range mapping
    and are rejected rather than silently copied as contiguous token data.
    """
    shape = tuple(int(x) for x in tensor.shape)
    strides = tuple(int(tensor.stride(i)) for i in range(4))
    if layer.num_blocks <= 0 or shape[0] % layer.num_blocks:
        raise ValueError("Attention view does not tile logical blocks")
    rows = shape[0] // layer.num_blocks
    states, heads, channels = shape[token_axis], shape[head_axis], shape[3]
    if rows <= 0 or rows * states != layer.storage_block_size:
        raise ValueError("Attention token axis disagrees with storage_block_size")
    if strides[3] != 1 or any(s <= 0 for s in strides):
        raise ValueError(
            "Attention components require positive strides and dense channels"
        )
    # NHC (or a singleton head) keeps one segment for the entire token range.
    if strides[token_axis] == heads * channels and (
        heads == 1 or strides[head_axis] == channels
    ):
        return (build_tensor_view(tensor, layer),)
    if strides[token_axis] != channels or strides[head_axis] < states * channels:
        raise ValueError("Unsupported attention token/head strides")
    if rows != 1:
        raise ValueError(
            "Multi-row head-separated attention requires a ragged range mapping"
        )
    element = int(tensor.element_size())
    return tuple(
        MemorySegment(
            base_ptr=int(tensor.data_ptr()) + head * strides[head_axis] * element,
            block_stride_bytes=strides[0] * element,
            states_per_block=states,
            bytes_per_state=channels * element,
            payload_bytes=states * channels * element,
        )
        for head in range(heads)
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


def build_tensor_view(
    tensor: "torch.Tensor",
    layer: "UCMLayerSpec",
    state_snapshot: bool = False,
) -> MemorySegment:
    """Derive one component's placement from its runtime view.

    The layer spec carries the two facts the view cannot express: the
    number of stored states one block spans (storage_block_size) and how
    many blocks the view's dim 0 tiles (num_blocks).
    """

    expected_block_size = layer.storage_block_size
    num_blocks = layer.num_blocks

    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) < 2 or len(shape) > 4:
        raise ValueError(
            "KV component views must be 2-D, 3-D, or 4-D, " f"got shape={shape}"
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
    states_per_block: int
    bytes_per_state: int
    if state_snapshot:
        # A state snapshot (mamba/SSM page) has no per-token axis: each
        # row is one indivisible record.
        states_per_block = rows_per_block
        bytes_per_state = payload
    else:
        # Map the spec's block size onto the view's rows.  A block spans
        # exactly expected_block_size stored states -- one per token, or
        # one per tokens_per_state on compressed caches (DSV4's C4A:
        # 256-token block = 64 states of 584B).  States never straddle
        # kernel rows, so each row holds expected_block_size //
        # rows_per_block states, each payload // states_per_row bytes.
        # This derivation covers the Ascend 0.26 token-axis dialect
        # (Kimi MLA: one logical block as dense kernel rows) and the
        # vLLM 0.29 permuted [B, H, N, C] views (N counts stored states)
        # identically for whole blocks. Partial-token IO requires states
        # to be contiguous in memory (NHC, or H=1 as in DSV4). Head-separated
        # HNC is split by _attention_segments before reaching this helper.
        states_per_row, remainder = divmod(expected_block_size, rows_per_block)
        if remainder or states_per_row <= 0 or payload % states_per_row:
            raise ValueError(
                "KV tensor does not match a dense row-payload tiling of "
                f"the block: shape={shape}, strides={strides}, "
                f"rows_per_block={rows_per_block}, "
                f"expected_block_size={expected_block_size}, "
                f"row_payload={payload}"
            )
        states_per_block = expected_block_size
        bytes_per_state = payload // states_per_row
    return MemorySegment(
        base_ptr=int(tensor.data_ptr()),
        block_stride_bytes=rows_per_block * row_stride,
        states_per_block=states_per_block,
        bytes_per_state=bytes_per_state,
        payload_bytes=rows_per_block * payload,
    )


def _state_tensor_views(
    tensors: tuple["torch.Tensor", ...],
    layer: "UCMLayerSpec",
) -> tuple[MemorySegment, ...]:
    """Resolve an explicit component tuple or one combined raw state page.

    The combined page stays a single component: whole-block state IO is
    a byte copy, so the conv/SSM split the spec describes adds no
    addressing information.  Ascend 0.26 exposes C = the full padded
    page (payload == page_stride == page_size, padding at the tail);
    vLLM 0.29 exposes C = the dense state content only (payload <=
    page_stride == page_size, padding between blocks).
    """

    expected_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in (getattr(layer.kv_cache_spec, "shapes", None) or ())
    )
    actual_shapes = tuple(
        tuple(int(item) for item in tensor.shape[1:]) for tensor in tensors
    )
    if not expected_shapes or actual_shapes == expected_shapes:
        return tuple(
            build_tensor_view(tensor, layer, state_snapshot=True) for tensor in tensors
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
    num_blocks = layer.num_blocks
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
        # torch.dtype (and the test doubles) carry itemsize directly.
        math.prod(component_shape) * dtype.itemsize
        for component_shape, dtype in zip(expected_shapes, dtypes, strict=True)
    )
    if content > payload:
        raise ValueError(
            f"State components ({content}B) exceed the page content "
            f"({payload}B) for {layer.layer_name}"
        )
    return (
        MemorySegment(
            base_ptr=int(raw.data_ptr()),
            block_stride_bytes=page_stride,
            states_per_block=1,
            bytes_per_state=content,
            payload_bytes=content,
        ),
    )
