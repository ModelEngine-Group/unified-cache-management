"""Precomputed per-group block addressing for connector v2.

:class:`GroupLayout` resolves one KV group's placement once, at init, into
two structures that queries only translate:

* **Runs** -- contiguous whole-block byte spans.  The group's structures
  (every component of every layer, split into kernel rows) are walked in
  slot order; consecutive spans that touch and share one block stride
  collapse into a shared run.  Because the merge follows from the
  placement arithmetic, it holds identically for every block, so dump and
  load produce the same entries however vLLM scatters physical blocks.
* **Row tables** -- the flat per-component facts (row stride, states per
  row, bytes per state, token-to-state scale) needed to cut a token range
  out of a block (DSV4's sub-block hash granularity, sliding-window
  tails).

Records are block-major: one plan block contributes its runs in order, so
a layer's position inside a record is an init-time constant.  Filtered
(layerwise) batches therefore address the very same record coordinates as
full ones -- a layerwise load writes into the record a full dump wrote.
Blocks never merge with each other: cross-block adjacency is an
allocation accident, not a layout property.

The layerwise contract (every layer, every step, save and load) keeps the
whole-block path pure translation: one multiply-add per block plus a walk
of precomputed slices.  Sub-block ranges take the row-table walk, which
merges only what is actually contiguous.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .model import (
    LAYOUT_DEBUG,
    ComponentSlot,
    LayerSlot,
    layout_debug,
)


@dataclass(frozen=True)
class SegmentRun:
    """One contiguous whole-block byte span shared by adjacent structures.

    Block ``b`` addresses the span at ``base_ptr + b * block_stride``;
    ``record_start`` is the span's offset inside one block's record.
    """

    base_ptr: int
    block_stride: int
    size: int
    record_start: int


@dataclass(frozen=True)
class _LayerSlice:
    """One layer's whole-block bytes inside one run (its rows merged)."""

    run_index: int
    offset: int
    size: int


class GroupLayout:
    """Init-time addressing plan for one KV group.

    ``segments`` answers "which (ptr, size) byte spans back these blocks x
    layers x tokens"; ``record_segments``/``state_record_segments`` add the
    record coordinates the proxy batches need.  Everything structural --
    run merging, per-layer slices, record positions -- is computed here
    once; the queries only translate block IDs.
    """

    def __init__(
        self,
        group_id: int,
        slots: Sequence[LayerSlot],
        *,
        token_block_size: int,
        num_blocks: int,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if token_block_size <= 0:
            raise ValueError("token_block_size must be positive")
        self.group_id = group_id
        self.slots = tuple(slots)
        self.token_block_size = token_block_size
        self.num_blocks = num_blocks

        # Runs are built from per-row spans: a component whose kernel rows
        # are padded (row_stride > row payload) is not contiguous inside a
        # block, so its rows stay separate runs, exactly like the row walk
        # would emit them.
        runs: list[list[int]] = []
        layer_slices: dict[str, list[_LayerSlice]] = {}
        # (slot, component, scale numerator, scale denominator) for the
        # sub-block row walk; states = local_tokens * num / den exactly.
        structures: list[tuple[LayerSlot, ComponentSlot, int, int]] = []
        for slot in self.slots:
            entries: list[_LayerSlice] = []
            for component in slot.components:
                if (
                    component.base_ptr
                    + (num_blocks - 1) * component.block_stride
                    + component.payload_bytes
                    > component.base_ptr + component.buffer_size_bytes
                ):
                    raise ValueError(
                        f"KV cache layout for {slot.layer_name} exceeds its "
                        "registered tensor buffer"
                    )
                for row in range(component.rows_per_block):
                    span_base = component.base_ptr + row * component.row_stride_bytes
                    span_size = component.row_payload_bytes
                    if (
                        runs
                        and runs[-1][1] == component.block_stride
                        and runs[-1][0] + runs[-1][2] == span_base
                    ):
                        runs[-1][2] += span_size
                    else:
                        runs.append([span_base, component.block_stride, span_size])
                    entries.append(
                        _LayerSlice(
                            len(runs) - 1,
                            span_base - runs[-1][0],
                            span_size,
                        )
                    )
                divisor = math.gcd(component.states_per_block, token_block_size)
                structures.append(
                    (
                        slot,
                        component,
                        component.states_per_block // divisor,
                        token_block_size // divisor,
                    )
                )
            # Merge the layer's consecutive slices that share one run (its
            # dense rows and contiguous components); padded rows or foreign
            # components in between keep the slices apart.
            merged: list[_LayerSlice] = []
            for entry in entries:
                if (
                    merged
                    and merged[-1].run_index == entry.run_index
                    and merged[-1].offset + merged[-1].size == entry.offset
                ):
                    merged[-1] = _LayerSlice(
                        entry.run_index,
                        merged[-1].offset,
                        merged[-1].size + entry.size,
                    )
                else:
                    merged.append(entry)
            layer_slices[slot.layer_name] = merged

        record_start = 0
        frozen_runs: list[SegmentRun] = []
        for base_ptr, block_stride, size in runs:
            frozen_runs.append(SegmentRun(base_ptr, block_stride, size, record_start))
            record_start += size
        self.runs = tuple(frozen_runs)
        self.block_record_size = record_start
        self.layer_slices = {
            name: tuple(slices) for name, slices in layer_slices.items()
        }
        self._structures = tuple(structures)
        if LAYOUT_DEBUG:
            layout_debug(
                f"group-layout group={group_id} layers={len(self.slots)} "
                f"runs={len(self.runs)} record_per_block={record_start} "
                f"token_block={token_block_size}"
            )

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def segments(
        self,
        block_ids: Sequence[int],
        layers: Sequence[str] | None = None,
        token_range: tuple[int, int] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """(ptr, size) byte spans backing the given blocks.

        ``block_ids`` are physical block IDs carrying ``token_range`` in
        order (block ``i`` holds the range's ``i``-th slice); ``layers``
        restricts the answer to those layer names (None = every layer of
        the group); ``token_range`` is an absolute ``(start, end)`` token
        window (None = whole blocks).
        """

        selected = None if layers is None else frozenset(layers)
        if selected is not None:
            unknown = selected.difference(self.layer_slices)
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

    def record_segments(
        self,
        base: int,
        block_map: Mapping[int, int],
        token_start: int,
        token_end: int,
        layer_name: str | None = None,
    ) -> tuple[list[tuple[int, int, int]], int]:
        """Segments with record offsets for one plan's token range.

        Returns ``(entries, next_base)`` where each entry is
        ``(ptr, size, record_offset)`` and ``next_base`` is the record
        offset after this group's (full, unfiltered) record.  ``layer_name``
        filters what is emitted but never moves anything: filtered and
        unfiltered batches share record coordinates, so a layerwise load
        writes into the record a full dump produced.  A ``layer_name`` from
        another group simply emits nothing.
        """

        spans = self.plan_spans(block_map, token_start, token_end)
        selected = None if layer_name is None else frozenset((layer_name,))
        return self._emit(spans, base, selected)

    def state_record_segments(
        self,
        base: int,
        block_map: Mapping[int, int],
        token_end: int,
        layer_name: str | None = None,
    ) -> tuple[list[tuple[int, int, int]], int]:
        """State snapshots: the last complete block of the token range."""

        logical = max((token_end - 1) // self.token_block_size, 0)
        if logical not in block_map:
            raise ValueError(f"Missing vLLM block {logical} in the plan")
        spans = (
            (self._checked_block(block_map[logical]), 0, self.token_block_size),
        )
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
        first = token_start // token_block_size
        last = (token_end - 1) // token_block_size
        block_ids = []
        for logical in range(first, last + 1):
            if logical not in block_map:
                raise ValueError(f"Missing vLLM block {logical} in the plan")
            block_ids.append(block_map[logical])
        return self.spans_for_range(block_ids, token_start, token_end)

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
        """Dispatch to the whole-block fast path or the row-table walk."""

        token_block_size = self.token_block_size
        if all(
            local_start == 0 and local_end == token_block_size
            for _, local_start, local_end in spans
        ):
            return self._emit_whole(spans, base, selected)
        return self._emit_partial(spans, base, selected)

    def _emit_whole(
        self,
        spans: Sequence[tuple[int, int, int]],
        base: int,
        selected: frozenset[str] | None,
    ) -> tuple[list[tuple[int, int, int]], int]:
        """Precomputed runs (or selected layers' slices inside them)."""

        record_size = self.block_record_size
        next_base = base + len(spans) * record_size
        slices = self._selected_slices(selected)
        entries: list[tuple[int, int, int]] = []
        for ordinal, (block_id, _, _) in enumerate(spans):
            record_base = base + ordinal * record_size
            for run_index, offset, size in slices:
                run = self.runs[run_index]
                ptr = run.base_ptr + block_id * run.block_stride + offset
                if LAYOUT_DEBUG:
                    layout_debug(
                        f"io-segment group={self.group_id} block={block_id} "
                        f"ptr={ptr:#x} size={size} "
                        f"record={record_base + run.record_start + offset}"
                    )
                entries.append(
                    (ptr, size, record_base + run.record_start + offset)
                )
        return entries, next_base

    def _selected_slices(
        self, selected: frozenset[str] | None
    ) -> tuple[tuple[int, int, int], ...]:
        """(run, offset, size) slices covering the selection.

        With no filter every run is one slice; a layer filter takes the
        selected layers' whole-block slices, merged inside shared runs.
        """

        if selected is None:
            return tuple((index, 0, run.size) for index, run in enumerate(self.runs))
        merged: list[tuple[int, int, int]] = []
        for slot in self.slots:
            if slot.layer_name not in selected:
                continue
            for entry in self.layer_slices[slot.layer_name]:
                if (
                    merged
                    and merged[-1][0] == entry.run_index
                    and merged[-1][1] + merged[-1][2] == entry.offset
                ):
                    merged[-1] = (
                        entry.run_index,
                        merged[-1][1],
                        merged[-1][2] + entry.size,
                    )
                else:
                    merged.append((entry.run_index, entry.offset, entry.size))
        return tuple(merged)

    def _emit_partial(
        self,
        spans: Sequence[tuple[int, int, int]],
        base: int,
        selected: frozenset[str] | None,
    ) -> tuple[list[tuple[int, int, int]], int]:
        """Row-table walk for ranges that cut into blocks.

        Structures are enumerated in slot order so record positions are a
        function of the logical enumeration alone; contiguous segments
        merge, but never across an unselected structure -- a filtered
        batch must not swallow bytes the filter excluded.
        """

        entries: list[tuple[int, int, int]] = []
        offset = base
        open_ptr: int | None = None
        open_size = 0
        open_offset = 0
        for block_id, local_start, local_end in spans:
            open_ptr = None  # a block boundary always splits the merge
            for slot, component, scale_num, scale_den in self._structures:
                selected_here = selected is None or slot.layer_name in selected
                for ptr, size in _component_segments(
                    component,
                    scale_num,
                    scale_den,
                    block_id,
                    local_start,
                    local_end,
                ):
                    if (
                        selected_here
                        and open_ptr is not None
                        and open_ptr + open_size == ptr
                    ):
                        open_size += size
                    else:
                        if open_ptr is not None:
                            entries.append((open_ptr, open_size, open_offset))
                        if selected_here:
                            open_ptr, open_size, open_offset = ptr, size, offset
                        else:
                            open_ptr = None
                    offset += size
                    if LAYOUT_DEBUG:
                        layout_debug(
                            f"segment group={self.group_id} "
                            f"layer={slot.layer_name} block={block_id} "
                            f"tokens=[{local_start},{local_end}) "
                            f"ptr={ptr:#x} size={size} open={selected_here}"
                        )
            if open_ptr is not None:
                entries.append((open_ptr, open_size, open_offset))
                open_ptr = None
        return entries, offset


def _component_segments(
    component: ComponentSlot,
    scale_num: int,
    scale_den: int,
    block_id: int,
    local_start: int,
    local_end: int,
) -> list[tuple[int, int]]:
    """Merged byte spans of one component for a block-local token window."""

    begin_num = local_start * scale_num
    end_num = local_end * scale_num
    if begin_num % scale_den or end_num % scale_den:
        raise ValueError(
            "Logical token range cannot be represented exactly by tensor layout"
        )
    state_begin = begin_num // scale_den
    state_end = end_num // scale_den
    states_per_row = component.states_per_row
    bytes_per_state = component.bytes_per_state
    base = component.base_ptr + block_id * component.block_stride
    segments: list[tuple[int, int]] = []
    while state_begin < state_end:
        row_in_block, state_in_row = divmod(state_begin, states_per_row)
        row_end = min(state_end, (row_in_block + 1) * states_per_row)
        ptr = (
            base
            + row_in_block * component.row_stride_bytes
            + state_in_row * bytes_per_state
        )
        size = (row_end - state_begin) * bytes_per_state
        if segments and segments[-1][0] + segments[-1][1] == ptr:
            segments[-1] = (segments[-1][0], segments[-1][1] + size)
        else:
            segments.append((ptr, size))
        state_begin = row_end
    return segments
