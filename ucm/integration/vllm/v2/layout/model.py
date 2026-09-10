"""Version-neutral layout IR and segment addressing for connector v2.

The IR speaks the vLLM 0.29 layout vocabulary: one backing allocation per
worker, ``KVCacheTensor``-style descriptors that place every layer's blocks
with ``offset + L * layer_stride + b * block_stride``, and per-component
page geometry read from the runtime views.  vLLM versions without the
declaration API are translated into the same IR by ``inferred.py``; nothing
below the ``build_layout_model`` entry point knows which mode produced a
model.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Debug trace for the v2 layout: set UCM_V2_LAYOUT_DEBUG=1 to log, on stderr,
# what the layout model registered per layer and which pointer each dispatched
# vLLM block resolves to.  Zero cost when disabled.
LAYOUT_DEBUG = os.environ.get("UCM_V2_LAYOUT_DEBUG", "0") not in ("", "0")


def layout_debug(message: str) -> None:
    if LAYOUT_DEBUG:
        print(f"[ucm-v2-layout] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class BackingAllocation:
    """One physical allocation holding KV cache bytes.

    vLLM 0.29 allocates a single backing per worker and carves the per-layer
    views out of it; older versions allocate one tensor per (set of) layer(s),
    which the inferred mode reports as separate backings.
    """

    backing_id: int
    base_ptr: int
    size_bytes: int


@dataclass(frozen=True)
class TensorDescriptor:
    """Placement of a layer set inside one backing (vLLM 0.29 KVCacheTensor).

    Layer ``L`` (its position in ``layers``) addresses block ``b`` at
    ``offset + L * layer_stride + b * block_stride`` bytes from the backing
    base.  On interleaved models (DSV4) ``block_stride`` is the shared
    manager-slot stride and differs from any single page size; on
    layer-contiguous models (GLM) it equals the page stride.  The inferred
    mode synthesizes one descriptor per runtime view base with
    ``offset = layer_stride = 0``; layers sharing a tensor (the legacy
    ``shared_by`` overlay) then share the descriptor naturally.
    """

    descriptor_id: int
    backing_id: int
    layers: tuple[str, ...]
    offset: int
    layer_stride: int
    block_stride: int


@dataclass(frozen=True)
class ComponentSlot:
    """One addressable component (K, V, latent, conv/SSM state, ...) of a layer.

    ``base_ptr`` is pre-resolved to the component's block-0 / row-0 address,
    so the hot path is ``base_ptr + block_id * block_stride`` plus in-page row
    arithmetic.  A kernel row stores ``states_per_row`` states of
    ``bytes_per_state`` bytes; a stored state may cover several logical tokens
    (DSV4's C4A cache stores one state per four tokens).
    """

    base_ptr: int
    block_stride: int
    row_stride_bytes: int
    rows_per_block: int
    states_per_row: int
    bytes_per_state: int
    buffer_size_bytes: int

    @property
    def states_per_block(self) -> int:
        return self.rows_per_block * self.states_per_row

    @property
    def row_payload_bytes(self) -> int:
        return self.states_per_row * self.bytes_per_state

    @property
    def payload_bytes(self) -> int:
        """Content bytes of one block; padding keeps this <= block_stride."""

        return self.rows_per_block * self.row_payload_bytes


@dataclass(frozen=True)
class BlockRegion:
    """One naturally contiguous IO region per vLLM physical block.

    A combined block-major K/V tensor or a shared state page contributes one
    region spanning all of its components; page and row padding stay outside
    ``payload_bytes``.
    """

    base_ptr: int
    block_stride: int
    payload_bytes: int
    buffer_size_bytes: int


@dataclass(frozen=True)
class LayerSlot:
    """Per-layer resolved placement.

    ``components`` serve token-level addressing (partial-block IO) and
    ``regions`` whole-block IO; a layer whose views have non-dense kernel rows
    has no regions and always walks its components.
    """

    layer_name: str
    layer_index: int
    group_id: int
    components: tuple[ComponentSlot, ...]
    regions: tuple[BlockRegion, ...]


def coalesce_segments(
    segments: Sequence[tuple[int, int, str | None]],
) -> tuple[tuple[int, int, frozenset[str]], ...]:
    """Drop duplicate IO ranges and merge adjacent ones, per record.

    ``segments`` are ``(ptr, size, layer_name)`` triples in dispatch order.
    Overlay layouts expose the same physical bytes through several layers
    (Kimi's shared Attention/Mamba page arrives once per group), and
    consecutive blocks are contiguous whenever content fills the block
    stride; transferring each byte range once keeps the transfer count
    proportional to the data rather than to the layer count.  Duplicate
    ranges read the same bytes, so dropping them keeps the record
    byte-identical for merged ranges and content-identical for
    deduplicated ones; owners accumulate so per-layer batches still find
    their segments.
    """

    ranges: list[list[int]] = []
    owners: list[set[str]] = []
    index: dict[tuple[int, int], int] = {}
    for ptr, size, layer in segments:
        if size <= 0:
            continue
        key = (ptr, size)
        position = index.get(key)
        if position is not None:
            if layer is not None:
                owners[position].add(layer)
            continue
        if ranges and ranges[-1][0] + ranges[-1][1] == ptr:
            ranges[-1][1] += size
            position = len(ranges) - 1
        else:
            ranges.append([ptr, size])
            owners.append(set())
            position = len(ranges) - 1
        if layer is not None:
            owners[position].add(layer)
        index[key] = position
    return tuple(
        (ptr, size, frozenset(layer_names))
        for (ptr, size), layer_names in zip(
            (tuple(range_) for range_ in ranges), owners
        )
    )


class LayoutModel:
    """Resolved per-layer placement plus ragged block/token addressing."""

    def __init__(
        self,
        *,
        backings: Sequence[BackingAllocation],
        descriptors: Sequence[TensorDescriptor],
        groups: Mapping[int, tuple[LayerSlot, ...]],
        num_blocks: int,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.backings = tuple(backings)
        self.descriptors = tuple(descriptors)
        self.groups = dict(groups)
        self.slots = {
            slot.layer_name: slot
            for group_slots in self.groups.values()
            for slot in group_slots
        }
        self.num_blocks = num_blocks

    def log_registration(
        self, group_id: int, slot: LayerSlot, *, state: bool
    ) -> None:
        if not LAYOUT_DEBUG:
            return
        for view_index, component in enumerate(slot.components):
            layout_debug(
                f"register num_blocks={self.num_blocks} "
                f"group={group_id} layer={slot.layer_name} "
                f"view={view_index} state={int(state)} "
                f"base_ptr={component.base_ptr:#x} "
                f"rows_per_block={component.rows_per_block} "
                f"states_per_row={component.states_per_row} "
                f"row_stride={component.row_stride_bytes} "
                f"payload={component.row_payload_bytes} "
                f"bytes_per_state={component.bytes_per_state} "
                f"buffer={component.buffer_size_bytes}"
            )
        for region_index, region in enumerate(slot.regions):
            layout_debug(
                f"io-region group={group_id} layer={slot.layer_name} "
                f"region={region_index} base_ptr={region.base_ptr:#x} "
                f"block_stride={region.block_stride} "
                f"block_payload={region.payload_bytes} "
                f"buffer={region.buffer_size_bytes}"
            )

    def resolve_block_id(
        self, logical_block: int, block_map: Mapping[int, int]
    ) -> int:
        if logical_block not in block_map:
            raise ValueError(f"Missing vLLM block {logical_block} in the plan")
        block_id = block_map[logical_block]
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(
                f"vLLM block ID {block_id} is outside [0, {self.num_blocks})"
            )
        return block_id

    def layer_segments(
        self,
        slot: LayerSlot,
        *,
        block_map: Mapping[int, int],
        token_start: int,
        token_end: int,
        token_block_size: int,
        state: bool,
    ) -> tuple[tuple[int, int], ...]:
        """Byte segments covering ``[token_start, token_end)`` for one layer.

        Whole blocks go through the layer's IO regions; partial coverage (or
        non-dense rows) falls back to walking each component's kernel rows.
        """

        segments = self._region_segments(
            slot,
            block_map=block_map,
            token_start=token_start,
            token_end=token_end,
            token_block_size=token_block_size,
            state=state,
        )
        if segments is None:
            segments = tuple(
                segment
                for component in slot.components
                for segment in self._component_segments(
                    slot,
                    component,
                    block_map=block_map,
                    token_start=token_start,
                    token_end=token_end,
                    token_block_size=token_block_size,
                    state=state,
                )
            )
        return segments

    def _region_segments(
        self,
        slot: LayerSlot,
        *,
        block_map: Mapping[int, int],
        token_start: int,
        token_end: int,
        token_block_size: int,
        state: bool,
    ) -> tuple[tuple[int, int], ...] | None:
        """One segment per (region, block) when the range covers whole blocks."""

        if not slot.regions:
            return None
        logical_blocks: tuple[int, ...]
        if state:
            # A state snapshot lives in the last block of its token range.
            logical_blocks = (max((token_end - 1) // token_block_size, 0),)
        else:
            first = token_start // token_block_size
            last = (token_end - 1) // token_block_size
            logical_blocks = tuple(range(first, last + 1))
            if any(
                max(token_start, block * token_block_size)
                != block * token_block_size
                or min(token_end, (block + 1) * token_block_size)
                != (block + 1) * token_block_size
                for block in logical_blocks
            ):
                return None

        result: list[tuple[int, int]] = []
        # Preserve the component structure exposed by the runtime value.  A
        # combined block-major tensor has one region; an Ascend tuple/list has
        # one region per explicit K/V, index/scale, or conv/SSM component.
        for region in slot.regions:
            for logical_block in logical_blocks:
                block_id = self.resolve_block_id(logical_block, block_map)
                ptr = region.base_ptr + block_id * region.block_stride
                size = region.payload_bytes
                if ptr + size > region.base_ptr + region.buffer_size_bytes:
                    raise ValueError(
                        "KV cache IO region exceeds registered tensor buffer"
                    )
                layout_debug(
                    f"io-segment group={slot.group_id} "
                    f"layer={slot.layer_name} block={block_id} "
                    f"ptr={ptr:#x} size={size}"
                )
                result.append((ptr, size))
        return tuple(result)

    def _component_segments(
        self,
        slot: LayerSlot,
        component: ComponentSlot,
        *,
        block_map: Mapping[int, int],
        token_start: int,
        token_end: int,
        token_block_size: int,
        state: bool,
    ) -> tuple[tuple[int, int], ...]:
        if state:
            return self._state_segments(
                slot, component, block_map, token_end, token_block_size
            )

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

        first = token_start // token_block_size
        last = (token_end - 1) // token_block_size
        for logical_block in range(first, last + 1):
            block_id = self.resolve_block_id(logical_block, block_map)
            logical_begin = max(token_start, logical_block * token_block_size)
            logical_end = min(token_end, (logical_block + 1) * token_block_size)
            # Scale logical tokens to stored states (identity unless the
            # cache compresses several tokens into one state).
            states_per_block = component.states_per_block
            numerator_begin = (
                logical_begin - logical_block * token_block_size
            ) * states_per_block
            numerator_end = (
                logical_end - logical_block * token_block_size
            ) * states_per_block
            if numerator_begin % token_block_size or numerator_end % token_block_size:
                raise ValueError(
                    "Logical token range cannot be represented exactly by tensor layout"
                )
            physical_begin = numerator_begin // token_block_size
            physical_end = numerator_end // token_block_size
            while physical_begin < physical_end:
                row_in_block, state_in_row = divmod(
                    physical_begin, component.states_per_row
                )
                row_end = min(physical_end, (row_in_block + 1) * component.states_per_row)
                row_index = block_id * component.rows_per_block + row_in_block
                ptr = (
                    component.base_ptr
                    + row_index * component.row_stride_bytes
                    + state_in_row * component.bytes_per_state
                )
                size = (row_end - physical_begin) * component.bytes_per_state
                if ptr + size > component.base_ptr + component.buffer_size_bytes:
                    raise ValueError(
                        "KV cache segment exceeds registered tensor buffer"
                    )
                layout_debug(
                    f"segment group={slot.group_id} layer={slot.layer_name} "
                    f"block={block_id} row={row_in_block}/"
                    f"{component.rows_per_block} "
                    f"tokens=[{logical_begin},{logical_end}) "
                    f"ptr={ptr:#x} view_off={ptr - component.base_ptr} size={size}"
                )
                append_segment(ptr, size, block_id)
                physical_begin = row_end
        return tuple(result)

    def _state_segments(
        self,
        component: ComponentSlot,
        block_map: Mapping[int, int],
        token_end: int,
        token_block_size: int,
    ) -> tuple[tuple[int, int], ...]:
        logical_block = max((token_end - 1) // token_block_size, 0)
        block_id = self.resolve_block_id(logical_block, block_map)
        first_row = block_id * component.rows_per_block
        layout_debug(
            f"state-segment block={block_id} rows={component.rows_per_block} "
            f"ptr={component.base_ptr + first_row * component.row_stride_bytes:#x} "
            f"view_off={first_row * component.row_stride_bytes} "
            f"size={component.rows_per_block * component.row_payload_bytes}"
        )
        state_segments: list[tuple[int, int]] = []
        for row in range(component.rows_per_block):
            ptr = component.base_ptr + (first_row + row) * component.row_stride_bytes
            size = component.row_payload_bytes
            if ptr + size > component.base_ptr + component.buffer_size_bytes:
                raise ValueError(
                    "KV cache state segment exceeds registered tensor buffer"
                )
            state_segments.append((ptr, size))
        return tuple(state_segments)
