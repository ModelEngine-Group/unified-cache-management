"""Per-group KV-cache addressing: the views are the truth.

Each group's layers are walked once, at init.  Every layer component
becomes one whole-block entry read straight from its runtime view --
``view.data_ptr() + block_id * block_stride`` with the view's payload --
because vLLM's allocation logic carves each view at its final address
(0.26: one allocation per view, never adjacent; 0.29: one packed backing,
views strided per the declarations).  A query then answers the core
question -- for these blocks, which (ptr, size) spans back these layers
over this token window -- by pure translation.

One special case: on interleaved declared layouts (Block First, DSV4
0.29: ``layer_stride < block_stride`` -- a descriptor's layer pages tile
each block slot) whole-batch queries take one descriptor-sized span per
block, ``layer_count * layer_stride`` bytes with the page paddings riding
along, endpoints from the declarations' own arithmetic.  Layered and
sub-block queries stay per-view exact and never touch padding.

Records are block-major, one slot-sized span per layer; every segment
locates itself as ``record_base + entry_offset + in-component offset``,
so a layerwise batch addresses the very same record bytes a full batch
produced.  Blocks never merge with each other.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import view
from .view import Component, LAYOUT_DEBUG, layout_debug

if TYPE_CHECKING:
    import torch

    from ..ucm_kv_cache import UCMKVCacheGroupInfo, UCMLayerSpec


@dataclass(frozen=True)
class TensorDescriptor:
    """One vLLM 0.29 ``KVCacheTensor`` declaration, as handed over.

    Layer ``layers[l]``'s block ``b`` starts at ``offset + l *
    layer_stride + b * block_stride`` bytes into the backing.
    """

    layers: tuple[str, ...]
    offset: int
    layer_stride: int
    block_stride: int


@dataclass(frozen=True)
class DescriptorSpan:
    """A descriptor's contiguous per-block span on interleaved layouts."""

    base_ptr: int
    block_stride: int
    span_bytes: int  # layer_count * layer_stride, page paddings included
    record_start: int  # span start inside one block's record


class GroupLayout:
    """One KV group's addressing, resolved once at init.

    ``entries`` is the core mapping: one row per (layer, component) with
    the component's whole-block span, the layer's record offset, and the
    token->state scale.  ``segment(block_ids, layers, token_range)``
    answers spans; ``emit_record`` adds record coordinates for the proxy
    batches.
    """

    def __init__(
        self,
        group: "UCMKVCacheGroupInfo",
        kv_caches: "Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]]",
    ) -> None:
        self.group_id = group.group_id
        self.token_block_size = group.token_block_size
        self.num_blocks = group.layers[0].num_blocks
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        # Walk the group's layers once, in layer order.  One model layer
        # contributes several layer names (attn, indexer.k_cache,
        # swa_cache, compressor states all belong to layer 5), so
        # layer_index is the order key and the name is only a stable
        # tiebreak within the same layer -- the record layout must not
        # depend on the config's enumeration order.
        ordered_layers = sorted(
            group.layers, key=lambda item: (item.layer_index, item.layer_name)
        )
        components_at: dict[str, tuple[Component, ...]] = {}
        for layer in ordered_layers:
            components = view.layer_structures(
                kv_caches[layer.layer_name],
                layer,
                state_snapshot=group.is_state_snapshot,
            )
            if not components:
                raise ValueError(
                    f"Layer {layer.layer_name} registered no addressable view"
                )
            if layer.descriptor is not None:
                for component in components:
                    if component.block_stride != layer.descriptor.block_stride:
                        raise ValueError(
                            f"Layer {layer.layer_name}: view block stride "
                            f"{component.block_stride} disagrees with the "
                            f"declared {layer.descriptor.block_stride}"
                        )
            components_at[layer.layer_name] = components

        self.layer_names = [layer.layer_name for layer in ordered_layers]
        self.descriptor_spans = self._descriptor_spans(
            ordered_layers, components_at
        )
        # Interleaved layouts: each layer's record span is its page slot
        # (layer_stride, paddings riding inside); otherwise the layer's
        # components' payloads.
        layer_spans = (
            {
                layer.layer_name: layer.descriptor.layer_stride
                for layer in ordered_layers
                if layer.descriptor is not None
            }
            if self.descriptor_spans
            else {}
        )
        self.entries: list[tuple] = []
        record = 0
        for layer in ordered_layers:
            components = components_at[layer.layer_name]
            span = layer_spans.get(
                layer.layer_name,
                sum(c.payload_bytes for c in components),
            )
            offset = record
            for component in components:
                divisor = math.gcd(
                    component.states_per_block, self.token_block_size
                )
                self.entries.append(
                    (
                        layer.layer_name,
                        component,
                        offset,
                        component.states_per_block // divisor,
                        self.token_block_size // divisor,
                    )
                )
                offset += component.payload_bytes
            record += span
        self.record_size = record
        if LAYOUT_DEBUG:
            layout_debug(
                f"group-layout group={self.group_id} "
                f"layers={len(self.layer_names)} entries={len(self.entries)} "
                f"descriptor-spans={len(self.descriptor_spans)} "
                f"record_per_block={record} token_block={self.token_block_size}"
            )

    def _descriptor_spans(
        self,
        ordered_layers: "Sequence[UCMLayerSpec]",
        components_at: Mapping[str, tuple[Component, ...]],
    ) -> tuple[DescriptorSpan, ...]:
        """Block First special case; empty unless this group's layers tile
        block slots under whole interleaved declarations."""

        descriptors = []
        seen = set()
        for layer in ordered_layers:
            if layer.descriptor is None or id(layer.descriptor) in seen:
                continue
            seen.add(id(layer.descriptor))
            descriptors.append(layer.descriptor)
        if not descriptors:
            return ()
        grouped: dict[int, list[str]] = {}
        for layer in ordered_layers:
            descriptor = layer.descriptor
            components = components_at[layer.layer_name]
            if descriptor is None or len(components) != 1:
                return ()
            if not 0 < descriptor.layer_stride < descriptor.block_stride:
                return ()  # layer-contiguous placements do not tile slots
            if components[0].payload_bytes > descriptor.layer_stride:
                return ()
            grouped.setdefault(id(descriptor), []).append(layer.layer_name)
        spans: list[DescriptorSpan] = []
        record = 0
        for descriptor in descriptors:
            names = grouped.get(id(descriptor))
            if not names:
                continue
            if len(names) != len(descriptor.layers):
                return ()  # descriptor spans layers outside this group
            spans.append(
                DescriptorSpan(
                    base_ptr=components_at[descriptor.layers[0]][0].base_ptr,
                    block_stride=descriptor.block_stride,
                    span_bytes=len(descriptor.layers) * descriptor.layer_stride,
                    record_start=record,
                )
            )
            record += len(descriptor.layers) * descriptor.layer_stride
        return tuple(spans)

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
        first = token_start // token_block_size
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
        entries = []
        if selected is None and self.descriptor_spans:
            # Whole batches on interleaved layouts: one descriptor-sized
            # span per block, paddings riding inside.
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
        for ordinal, (block_id, _, _) in enumerate(spans):
            record_base = base + ordinal * self.record_size
            for layer_name, component, offset, _, _ in self.entries:
                if selected is not None and layer_name not in selected:
                    continue
                ptr = component.base_ptr + block_id * component.block_stride
                if LAYOUT_DEBUG:
                    layout_debug(
                        f"io-segment group={self.group_id} "
                        f"layer={layer_name} block={block_id} "
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
            for layer_name, component, offset, scale_num, scale_den in self.entries:
                if selected is not None and layer_name not in selected:
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
                            f"layer={layer_name} block={block_id} "
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
    component: Component,
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
