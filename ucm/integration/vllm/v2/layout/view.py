"""Compile registered logical views into contiguous source-memory segments.

LayerView groups a layer name's components; components can share physical
storage. Backend axis conventions and runtime shape/stride define addressing.
The spec supplies logical block sizes, compression ratios and state-page
semantics. No KV storage is allocated or copied here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .kv_cache import UCMLayerSpec


@dataclass(frozen=True, slots=True)
class MemorySegment:
    """One contiguous token/state segment within each logical block.

    ``base_ptr`` is this segment's block-0 address; block ``b`` starts at
    ``base_ptr + b * block_stride_bytes`` and holds ``payload_bytes`` of
    content as ``states_per_block`` states of ``bytes_per_state`` bytes.
    The group's partial-token policy determines whether token offsets
    can use that byte ratio. Compact multi-head HNC pages retain their
    physical head-major order and require whole-block IO. ``storage_base_ptr``
    identifies the backing for registration-time range merging only.
    """

    base_ptr: int
    block_stride_bytes: int
    states_per_block: int
    bytes_per_state: int
    payload_bytes: int
    storage_base_ptr: int


@dataclass(frozen=True, slots=True)
class ComponentView:
    """One registered tensor view, potentially split into head segments.

    No storage is allocated. Segment bases include the view's storage offset
    already, because they are derived from tensor.data_ptr().
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    segments: tuple[MemorySegment, ...]
    supports_partial_tokens: bool


@dataclass(frozen=True, slots=True)
class LayerView:
    """One layer name's original components and final unique IO segments."""

    layer_name: str
    layer_id: int
    components: tuple[ComponentView, ...]

    segments: tuple[MemorySegment, ...]
    supports_partial_tokens: bool


def _merge_segments(
    segments: Sequence[MemorySegment],
) -> tuple[tuple[MemorySegment, ...], bool]:
    """Union overlapping/adjacent spans of one layer name's registered views.

    Backings keep first-component order; offsets within a backing use physical
    order. Joined ranges are indivisible, so token slicing is disabled.
    """
    by_backing: dict[tuple[int, int], list[MemorySegment]] = {}
    for segment in segments:
        identity = (segment.storage_base_ptr, segment.block_stride_bytes)
        by_backing.setdefault(identity, []).append(segment)

    merged: list[MemorySegment] = []
    whole_block_only = False
    for entries in by_backing.values():
        spans: list[MemorySegment] = []
        for segment in sorted(entries, key=lambda item: item.base_ptr):
            if spans:
                previous = spans[-1]
                previous_end = previous.base_ptr + previous.payload_bytes
                if segment.base_ptr <= previous_end:
                    end = max(previous_end, segment.base_ptr + segment.payload_bytes)
                    size = end - previous.base_ptr
                    spans[-1] = replace(
                        previous,
                        payload_bytes=size,
                        states_per_block=1,
                        bytes_per_state=size,
                    )
                    whole_block_only = True
                    continue
            spans.append(segment)
        merged.extend(spans)
    return tuple(merged), whole_block_only


def build_layer_view(
    kv_cache: "torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]",
    layer: "UCMLayerSpec",
    *,
    state_snapshot: bool,
    device_type: str,
) -> LayerView:
    """Normalize registered views to BHNC without moving cache bytes.

    The worker supplies BHNC or BNHC explicitly; physical ordering stays
    in the tensor strides. A block may span multiple kernel rows.
    CUDA State is already a BHNC byte page; Ascend State
    components are exposed as BHNC byte pages with their original strides.
    """
    tensors = tuple(kv_cache) if isinstance(kv_cache, (tuple, list)) else (kv_cache,)
    if (
        not state_snapshot
        and device_type == "npu"
        and len(tensors) == 1
        and tensors[0].ndim == 5
    ):
        combined = tensors[0]
        if combined.shape[0] != 2:
            raise ValueError("Ascend combined KV views must have a leading K/V axis")
        tensors = tuple(combined.unbind(0))
    components = []
    for tensor in tensors:
        # Empty components (e.g. MLA head_size_v=0) contribute no IO bytes.
        if tensor.numel() == 0:
            continue
        if state_snapshot and device_type == "npu":
            tensor = _as_byte_page(tensor)
        elif (
            not state_snapshot
            and tensor.ndim == 4
            and layer.attention_view_order == "BNHC"
        ):
            tensor = tensor.permute(0, 2, 1, 3)
        shape = tuple(tensor.shape)
        if len(shape) != 4:
            raise ValueError(
                f"Layer {layer.layer_name} requires a normalized BHNC view, "
                f"got shape={shape}"
            )
        strides = tuple(tensor.stride())
        components.append(_build_bhnc_view(tensor, layer, shape, strides))
    segments, whole_block_only = _merge_segments(
        tuple(segment for component in components for segment in component.segments)
    )
    return LayerView(
        layer_name=layer.layer_name,
        layer_id=layer.layer_index,
        components=tuple(components),
        segments=segments,
        supports_partial_tokens=not whole_block_only
        and all(component.supports_partial_tokens for component in components),
    )


def _as_byte_page(tensor: "torch.Tensor") -> "torch.Tensor":
    """Expose a dense per-block payload as [B,1,1,bytes], without copying.

    Component storage offsets and padded/cross-layer block strides stay
    intact. Shape/stride supply the content size; no State spec is unpacked.
    """
    import torch

    element_size = int(tensor.element_size())
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    row_payload = row_payload_bytes(shape, strides, element_size)
    # Flatten the first dense physical payload before changing dtype. A dense
    # permutation need not have stride(-1)==1, which dtype view() requires.
    payload_view = torch.as_strided(
        tensor,
        size=(row_payload // element_size,),
        stride=(1,),
        storage_offset=tensor.storage_offset(),
    )
    byte_view = payload_view.view(torch.uint8)
    return torch.as_strided(
        byte_view,
        size=(shape[0], 1, 1, row_payload),
        stride=(strides[0] * element_size, row_payload, row_payload, 1),
        storage_offset=byte_view.storage_offset(),
    )


def _build_bhnc_view(
    tensor: "torch.Tensor",
    layer: "UCMLayerSpec",
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> ComponentView:
    """Keep compact pages together; split only physically separated heads.

    Physical NHC and singleton-head views support partial-token IO.
    Multi-head HNC requires whole blocks, including the separated-head
    layouts LHBNC/BHLNC. Compact HNC stays one segment without repacking.
    Tiled separated-head views contribute one span per kernel row/head.
    LayerView initialization deduplicates and merges spans of this layer name.
    """
    if shape[0] % layer.num_blocks:
        raise ValueError("BHNC view does not tile logical blocks")
    rows_per_block = shape[0] // layer.num_blocks
    heads, states, channels = shape[1:]
    if rows_per_block * states != layer.storage_block_size:
        raise ValueError("BHNC state axis disagrees with storage_block_size")
    if strides[3] != 1:
        raise ValueError("BHNC components require dense channels")
    element_size = int(tensor.element_size())
    base_ptr = int(tensor.data_ptr())
    storage_base_ptr = int(tensor.untyped_storage().data_ptr())
    token_contiguous = strides[2] == heads * channels and (
        heads == 1 or strides[1] == channels
    )
    if not token_contiguous and (
        strides[2] != channels or strides[1] < states * channels
    ):
        raise ValueError("Unsupported BHNC state/head strides")
    # Physical NHC or compact HNC occupies one span for a complete block.
    if token_contiguous or strides[1] == states * channels:
        row_payload = states * heads * channels * element_size
        row_stride = strides[0] * element_size
        if rows_per_block > 1 and row_stride != row_payload:
            raise ValueError("Multi-row BHNC blocks require contiguous kernel rows")
        return ComponentView(
            shape=shape,
            strides=strides,
            segments=(
                MemorySegment(
                    base_ptr=base_ptr,
                    block_stride_bytes=rows_per_block * row_stride,
                    states_per_block=rows_per_block * states,
                    bytes_per_state=heads * channels * element_size,
                    payload_bytes=rows_per_block * row_payload,
                    storage_base_ptr=storage_base_ptr,
                ),
            ),
            supports_partial_tokens=token_contiguous,
        )
    fragments = (
        MemorySegment(
            base_ptr=base_ptr + (row * strides[0] + head * strides[1]) * element_size,
            block_stride_bytes=rows_per_block * strides[0] * element_size,
            states_per_block=states,
            bytes_per_state=channels * element_size,
            payload_bytes=states * channels * element_size,
            storage_base_ptr=storage_base_ptr,
        )
        for row in range(rows_per_block)
        for head in range(heads)
    )
    segments = tuple(sorted(fragments, key=lambda segment: segment.base_ptr))
    return ComponentView(shape, strides, segments, supports_partial_tokens=False)


def row_payload_bytes(
    shape: tuple[int, ...], strides: tuple[int, ...], element_size: int
) -> int:
    """Dense per-block content size for Ascend byte-page normalization.

    Block strides may include padding or other layers; trailing dimensions
    must form one contiguous payload, allowing dense axis permutations.
    """

    # Singleton axes carry no address displacement, so their strides do
    # not constrain density (e.g. an H=1 view over a shared allocation).
    stride_size_pairs = sorted(
        (axis_stride, axis_size)
        for axis_stride, axis_size in zip(strides[1:], shape[1:])
        if axis_size != 1
    )
    expected_stride = 1
    for axis_stride, axis_size in stride_size_pairs:
        if axis_stride != expected_stride:
            raise ValueError(
                "KV tensor trailing dimensions must be dense (C-order or a "
                f"dense permutation); shape={shape}, strides={strides}"
            )
        expected_stride *= axis_size
    return expected_stride * element_size
