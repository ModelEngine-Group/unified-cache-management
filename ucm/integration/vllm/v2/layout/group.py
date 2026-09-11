"""Per-group addressing for connector v2: the views are the truth.

Every layer component is one whole-block span taken straight from its
runtime view -- ``view.data_ptr() + block_id * block_stride`` with the
view's payload -- because vLLM's allocation logic carves each view at
its final address (0.26: one allocation per view, never adjacent; 0.29:
one packed backing, views strided per the declarations).  Nothing
probes or re-derives placement.

One special case: on interleaved declared layouts (DSV4 0.29,
``layer_stride < block_stride`` -- a descriptor's layer pages tile each
block slot), whole-batch queries take one descriptor-sized span per
block, ``layer_count * layer_stride`` bytes with the page paddings
riding along, endpoints from the declarations' own arithmetic.  Layered
and sub-block queries stay per-view exact and never touch padding.

Records are block-major, one span per layer; every segment locates
itself as ``record_base + entry_offset + in-component offset``, so a
layerwise batch addresses the very same record bytes a full batch
produced.  Blocks never merge with each other.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .model import (
    LAYOUT_DEBUG,
    ComponentSlot,
    LayerSlot,
    TensorDescriptor,
    layout_debug,
)


@dataclass(frozen=True)
class DescriptorSpan:
    """A descriptor's contiguous per-block span on interleaved layouts."""

    base_ptr: int
    block_stride: int
    span_bytes: int  # layer_count * layer_stride, page paddings included
    record_start: int  # span start inside one block's record


class GroupLayout:
    """Init-time addressing plan for one KV group; queries only translate."""

    def __init__(
        self,
        group_id: int,
        slots: Sequence[LayerSlot],
        *,
        token_block_size: int,
        num_blocks: int,
        descriptors: Sequence[TensorDescriptor] = (),
    ) -> None:
        if num_blocks <= 0 or token_block_size <= 0:
            raise ValueError("num_blocks and token_block_size must be positive")
        self.group_id = group_id
        self.slots = tuple(slots)
        self.token_block_size = token_block_size
        self.num_blocks = num_blocks

        # Block-First special case; empty unless the group's layers tile
        # block slots under one declaration set.
        self.descriptor_spans, layer_spans = self._resolve_spans(descriptors)

        # One entry per component: (slot, component, record_offset,
        # scale_num, scale_den).  record_offset locates the component
        # inside one block's record; scale maps tokens to stored states.
        self.entries: list[tuple] = []
        record = 0
        for slot in self.slots:
            span = layer_spans.get(
                slot.layer_name,
                sum(c.payload_bytes for c in slot.components),
            )
            offset = record
            for component in slot.components:
                self._check_bounds(slot, component)
                divisor = math.gcd(
                    component.states_per_block, token_block_size
                )
                self.entries.append(
                    (
                        slot,
                        component,
                        offset,
                        component.states_per_block // divisor,
                        token_block_size // divisor,
                    )
                )
                offset += component.payload_bytes
            record += span
        self.record_size = record
        if LAYOUT_DEBUG:
            layout_debug(
                f"group-layout group={group_id} layers={len(self.slots)} "
                f"entries={len(self.entries)} "
                f"descriptor-spans={len(self.descriptor_spans)} "
                f"record_per_block={record} token_block={token_block_size}"
            )

    def _resolve_spans(
        self, descriptors: Sequence[TensorDescriptor]
    ) -> tuple[tuple[DescriptorSpan, ...], dict[str, int]]:
        """Interleaved-layout spans and each layer's record span.

        Falls back to the per-view record (empty spans) unless every
        layer of the group is single-component, covered by one
        interleaved descriptor (``0 < layer_stride < block_stride``)
        that stays inside the group, and every page fits its slot.
        """

        if not descriptors:
            return (), {}
        descriptor_at: dict[str, tuple[TensorDescriptor, int]] = {}
        for descriptor in descriptors:
            for position, name in enumerate(descriptor.layers):
                descriptor_at.setdefault(name, (descriptor, position))
        grouped: dict[int, list[str]] = {}
        for slot in self.slots:
            placement = descriptor_at.get(slot.layer_name)
            if placement is None or len(slot.components) != 1:
                return (), {}
            descriptor, _ = placement
            component = slot.components[0]
            if (
                not 0 < descriptor.layer_stride < descriptor.block_stride
                or component.payload_bytes > descriptor.layer_stride
            ):
                return (), {}
            grouped.setdefault(descriptor.descriptor_id, []).append(
                slot.layer_name
            )
        for descriptor in descriptors:
            names = grouped.get(descriptor.descriptor_id)
            if names and len(names) != len(descriptor.layers):
                return (), {}
        spans: list[DescriptorSpan] = []
        record = 0
        for descriptor in descriptors:
            names = grouped.get(descriptor.descriptor_id)
            if not names:
                continue
            first = self._slot_of(descriptor.layers[0])
            span = DescriptorSpan(
                base_ptr=first.components[0].base_ptr,
                block_stride=descriptor.block_stride,
                span_bytes=len(descriptor.layers) * descriptor.layer_stride,
                record_start=record,
            )
            spans.append(span)
            record += span.span_bytes
        layer_spans = {
            name: descriptor.layer_stride
            for name, (descriptor, _) in descriptor_at.items()
            if any(s.layer_name == name for s in self.slots)
        }
        return tuple(spans), layer_spans

    def _slot_of(self, layer_name: str) -> LayerSlot:
        for slot in self.slots:
            if slot.layer_name == layer_name:
                return slot
        raise ValueError(f"Unknown layer {layer_name}")

    def _check_bounds(self, slot: LayerSlot, component: ComponentSlot) -> None:
        if (
            component.base_ptr
            + (self.num_blocks - 1) * component.block_stride
            + component.payload_bytes
            > component.base_ptr + component.buffer_size_bytes
        ):
            raise ValueError(
                f"KV cache layout for {slot.layer_name} exceeds its "
                "registered tensor buffer"
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def segments(
        self,
        block_ids: Sequence[int],
        layers: Sequence[str] | None = None,
        token_range: tuple[int, int] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """(ptr, size) byte spans backing the given blocks.

        ``block_ids`` are physical block IDs carrying ``token_range`` in
        order; ``layers`` restricts the answer to those layer names
        (None = every layer); ``token_range`` is an absolute ``(start,
        end)`` token window (None = whole blocks).
        """

        selected = None if layers is None else frozenset(layers)
        if selected is not None:
            unknown = selected.difference(
                slot.layer_name for slot in self.slots
            )
            if unknown:
                raise ValueError(
                    f"Unknown layers for group {self.group_id}: {sorted(unknown)}"
                )
        if token_range is None:
            spans = tuple(
                (self._checked_block(block_id), 0, self.token_block_size)
                for block_id in block_ids
            )
        else:
            spans = self.spans_for_range(block_ids, *token_range)
        entries, _ = self._emit(spans, 0, selected)
        return tuple((ptr, size) for ptr, size, _ in entries)

    def emit_record(
        self,
        spans: Sequence[tuple[int, int, int]],
        base: int,
        layer_name: str | None = None,
    ) -> tuple[list[tuple[int, int, int]], int]:
        """Segments with record offsets for the given block spans.

        Returns ``(entries, next_base)``; each entry is ``(ptr, size,
        record_offset)``.  ``layer_name`` filters what is emitted but
        never moves anything -- filtered and unfiltered batches share
        record coordinates, so a layerwise load writes into the record a
        full dump produced.  A layer from another group emits nothing.
        State snapshots pass the spans of the last block of the request
        (:meth:`state_plan_range`).
        """

        selected = None if layer_name is None else frozenset((layer_name,))
        return self._emit(spans, base, selected)

    # ------------------------------------------------------------------
    # Span derivation
    # ------------------------------------------------------------------

    def spans_for_range(
        self,
        block_ids: Sequence[int],
        token_start: int,
        token_end: int,
    ) -> tuple[tuple[int, int, int], ...]:
        """(block_id, local_start, local_end) per block carrying the range."""

        token_block_size = self.token_block_size
        if token_end <= token_start:
            raise ValueError("Token range must be non-empty")
        first = token_start // token_block_size
        needed = (token_end - 1) // token_block_size - first + 1
        if len(block_ids) != needed:
            raise ValueError(
                f"Token range [{token_start},{token_end}) needs {needed} "
                f"blocks, got {len(block_ids)}"
            )
        spans = []
        for ordinal, block_id in enumerate(block_ids):
            block_start = (first + ordinal) * token_block_size
            spans.append(
                (
                    self._checked_block(block_id),
                    max(token_start - block_start, 0),
                    min(token_end - block_start, token_block_size),
                )
            )
        return tuple(spans)

    def plan_spans(
        self,
        block_map: Mapping[int, int],
        token_start: int,
        token_end: int,
    ) -> tuple[tuple[int, int, int], ...]:
        """spans_for_range over a plan's logical->physical block map."""

        token_block_size = self.token_block_size
        block_ids = [
            block_map[logical]
            for logical in range(
                token_start // token_block_size,
                (token_end - 1) // token_block_size + 1,
            )
        ]
        return self.spans_for_range(block_ids, token_start, token_end)

    def state_plan_range(
        self, block_map: Mapping[int, int], token_end: int
    ) -> tuple[tuple[int, int, int], ...]:
        """Spans of the last complete block holding a state snapshot."""

        logical = max((token_end - 1) // self.token_block_size, 0)
        if logical not in block_map:
            raise ValueError(f"Missing vLLM block {logical} in the plan")
        block = self._checked_block(block_map[logical])
        return ((block, 0, self.token_block_size),)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def _checked_block(self, block_id: int) -> int:
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(
                f"vLLM block ID {block_id} is outside [0, {self.num_blocks})"
            )
        return block_id

    def _emit(
        self,
        spans: Sequence[tuple[int, int, int]],
        base: int,
        selected: frozenset[str] | None,
    ) -> tuple[list[tuple[int, int, int]], int]:
        token_block_size = self.token_block_size
        if all(
            local_start == 0 and local_end == token_block_size
            for _, local_start, local_end in spans
        ):
            entries = self._whole(spans, base, selected)
        else:
            entries = self._partial(spans, base, selected)
        return entries, base + len(spans) * self.record_size

    def _whole(
        self,
        spans: Sequence[tuple[int, int, int]],
        base: int,
        selected: frozenset[str] | None,
    ) -> list[tuple[int, int, int]]:
        if selected is None and self.descriptor_spans:
            # Whole batches on interleaved layouts: one descriptor-sized
            # span per block, paddings riding inside.
            entries = []
            for ordinal, (block_id, _, _) in enumerate(spans):
                record_base = base + ordinal * self.record_size
                for span in self.descriptor_spans:
                    ptr = span.base_ptr + block_id * span.block_stride
                    if LAYOUT_DEBUG:
                        layout_debug(
                            f"io-span group={self.group_id} block={block_id} "
                            f"ptr={ptr:#x} size={span.span_bytes} "
                            f"record={record_base + span.record_start}"
                        )
                    entries.append(
                        (ptr, span.span_bytes, record_base + span.record_start)
                    )
            return entries
        entries = []
        for ordinal, (block_id, _, _) in enumerate(spans):
            record_base = base + ordinal * self.record_size
            for slot, component, offset, _, _ in self.entries:
                if selected is not None and slot.layer_name not in selected:
                    continue
                ptr = component.base_ptr + block_id * component.block_stride
                if LAYOUT_DEBUG:
                    layout_debug(
                        f"io-segment group={self.group_id} "
                        f"layer={slot.layer_name} block={block_id} "
                        f"ptr={ptr:#x} size={component.payload_bytes} "
                        f"record={record_base + offset}"
                    )
                entries.append(
                    (ptr, component.payload_bytes, record_base + offset)
                )
        return entries

    def _partial(
        self,
        spans: Sequence[tuple[int, int, int]],
        base: int,
        selected: frozenset[str] | None,
    ) -> list[tuple[int, int, int]]:
        """Row-table walk for ranges that cut into blocks."""

        entries = []
        for ordinal, (block_id, local_start, local_end) in enumerate(spans):
            record_base = base + ordinal * self.record_size
            for slot, component, offset, scale_num, scale_den in self.entries:
                if selected is not None and slot.layer_name not in selected:
                    continue
                component_base = (
                    component.base_ptr + block_id * component.block_stride
                )
                for rel, size in _component_segments(
                    component, scale_num, scale_den, local_start, local_end
                ):
                    if LAYOUT_DEBUG:
                        layout_debug(
                            f"segment group={self.group_id} "
                            f"layer={slot.layer_name} block={block_id} "
                            f"tokens=[{local_start},{local_end}) "
                            f"ptr={component_base + rel:#x} size={size} "
                            f"record={record_base + offset + rel}"
                        )
                    entries.append(
                        (
                            component_base + rel,
                            size,
                            record_base + offset + rel,
                        )
                    )
        return entries


def _component_segments(
    component: ComponentSlot,
    scale_num: int,
    scale_den: int,
    local_start: int,
    local_end: int,
) -> list[tuple[int, int]]:
    """Merged byte spans of one component's block for a token window.

    Returns ``(offset_from_block_base, size)`` pairs; ``scale_num /
    scale_den`` maps block-local tokens to stored states (identity
    unless the cache compresses several tokens into one state).
    """

    begin_num = local_start * scale_num
    end_num = local_end * scale_num
    if begin_num % scale_den or end_num % scale_den:
        raise ValueError(
            "Logical token range cannot be represented exactly by tensor layout"
        )
    state_begin = begin_num // scale_den
    state_end = end_num // scale_den
    states_per_row = component.states_per_row
    segments: list[tuple[int, int]] = []
    while state_begin < state_end:
        row_in_block, state_in_row = divmod(state_begin, states_per_row)
        row_end = min(state_end, (row_in_block + 1) * states_per_row)
        offset = (
            row_in_block * component.row_stride_bytes
            + state_in_row * component.bytes_per_state
        )
        size = (row_end - state_begin) * component.bytes_per_state
        if segments and segments[-1][0] + segments[-1][1] == offset:
            segments[-1] = (segments[-1][0], segments[-1][1] + size)
        else:
            segments.append((offset, size))
        state_begin = row_end
    return segments
