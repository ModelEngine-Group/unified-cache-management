"""Compile registered logical views into contiguous source-memory segments.

LayerView groups a layer name's components; components can share physical
storage. Backend axis conventions and runtime shape/stride define addressing.
The spec supplies logical block sizes, compression ratios and state-page
semantics. No KV storage is allocated or copied here.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
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

    from .kv_cache import UCMLayerSpec


@dataclass(frozen=True, slots=True)
class MemorySegment:
    """One contiguous token/state segment within each logical block.

    ``base_ptr`` is this segment's block-0 address; block ``b`` starts at
    ``base_ptr + b * block_stride_bytes`` and holds ``payload_bytes`` of
    content as ``states_per_block`` states of ``bytes_per_state`` bytes.
    The group's partial-token policy determines whether token offsets
    can use that byte ratio. Compact multi-head HNC pages retain their
    physical head-major order and require whole-block IO.
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
    supports_partial_tokens: bool


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
    """Normalize registered views to BHNC without moving cache bytes.

    The worker supplies BHNC or BNHC explicitly; physical ordering stays
    in the tensor strides. A block may span multiple kernel rows.
    CUDA State is already a BHNC byte page; Ascend State
    components are exposed as BHNC byte pages with their original strides.
    """
    tensors = tuple(value) if isinstance(value, (tuple, list)) else (value,)
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
    return LayerView(layer.layer_name, layer.layer_index, tuple(components))


def _as_byte_page(tensor: "torch.Tensor") -> "torch.Tensor":
    """Expose a dense per-block payload as [B,1,1,bytes], without copying.

    Component storage offsets and padded/cross-layer block strides stay
    intact. Shape/stride supply the content size; no State spec is unpacked.
    """
    import torch

    element_size = int(tensor.element_size())
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    payload = row_payload_bytes(shape, strides, element_size)
    byte_view = tensor.view(torch.uint8)
    return torch.as_strided(
        byte_view,
        size=(shape[0], 1, 1, payload),
        stride=(strides[0] * element_size, payload, payload, 1),
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
    Tiled separated-head views contribute one span per kernel row/head;
    adjacent spans are merged in physical byte order at initialization.
    """
    if shape[0] % layer.num_blocks:
        raise ValueError("BHNC view does not tile logical blocks")
    rows = shape[0] // layer.num_blocks
    heads, states, channels = shape[1:]
    if rows * states != layer.storage_block_size:
        raise ValueError("BHNC state axis disagrees with storage_block_size")
    if strides[3] != 1:
        raise ValueError("BHNC components require dense channels")
    element = int(tensor.element_size())
    base_ptr = int(tensor.data_ptr())
    token_contiguous = strides[2] == heads * channels and (
        heads == 1 or strides[1] == channels
    )
    if not token_contiguous and (
        strides[2] != channels or strides[1] < states * channels
    ):
        raise ValueError("Unsupported BHNC state/head strides")
    # Physical NHC or compact HNC occupies one span for a complete block.
    if token_contiguous or strides[1] == states * channels:
        row_payload = states * heads * channels * element
        row_stride = strides[0] * element
        if rows > 1 and row_stride != row_payload:
            raise ValueError("Multi-row BHNC blocks require contiguous kernel rows")
        return ComponentView(
            shape=shape,
            strides=strides,
            segments=(
                MemorySegment(
                    base_ptr=base_ptr,
                    block_stride_bytes=rows * row_stride,
                    states_per_block=rows * states,
                    bytes_per_state=heads * channels * element,
                    payload_bytes=rows * row_payload,
                ),
            ),
            supports_partial_tokens=token_contiguous,
        )
    fragments = (
        MemorySegment(
            base_ptr=base_ptr + (row * strides[0] + head * strides[1]) * element,
            block_stride_bytes=rows * strides[0] * element,
            states_per_block=states,
            bytes_per_state=channels * element,
            payload_bytes=states * channels * element,
        )
        for row in range(rows)
        for head in range(heads)
    )
    segments: list[MemorySegment] = []
    for fragment in sorted(fragments, key=lambda item: item.base_ptr):
        if segments and (
            segments[-1].base_ptr + segments[-1].payload_bytes == fragment.base_ptr
        ):
            previous = segments[-1]
            # All fragments have the same block stride and bytes/state.
            segments[-1] = replace(
                previous,
                states_per_block=previous.states_per_block + fragment.states_per_block,
                payload_bytes=previous.payload_bytes + fragment.payload_bytes,
            )
        else:
            segments.append(fragment)
    return ComponentView(shape, strides, tuple(segments), supports_partial_tokens=False)


def row_payload_bytes(
    shape: tuple[int, ...], strides: tuple[int, ...], element_size: int
) -> int:
    """Dense per-block content size for Ascend byte-page normalization.

    Block strides may include padding or other layers; trailing dimensions
    must form one contiguous payload, allowing dense axis permutations.
    """

    # Singleton axes carry no address displacement, so their strides do
    # not constrain density (e.g. an H=1 view over a shared allocation).
    pairs = sorted(
        (stride, size) for stride, size in zip(strides[1:], shape[1:]) if size != 1
    )
    expected_stride = 1
    for stride, size in pairs:
        if stride != expected_stride:
            raise ValueError(
                "KV tensor trailing dimensions must be dense (C-order or a "
                f"dense permutation); shape={shape}, strides={strides}"
            )
        expected_stride *= size
    return expected_stride * element_size
