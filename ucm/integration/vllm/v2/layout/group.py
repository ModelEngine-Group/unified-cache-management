"""Per-group KV-cache addressing: flat columns plus one formula.

The group is the unit vLLM allocates by -- one block table per group, of
three kinds (full attention, sliding window, mamba state).  At
registration each group's kv_cache views are walked once and distilled
into parallel columns, one row per tensor view:

    layer_ids         model layer of the view (spec layer_index, MTP-safe)
    layer_names       registered cache name (a layer may contribute several)
    base_ptrs         block-0 address of the view
    block_strides     bytes between consecutive blocks
    state_strides     bytes between consecutive stored states
    states_per_block  stored states one block spans
    record_slots      the view's slot in a whole-block record (Block First:
                     its page position in the descriptor tile chain; else
                     the payload prefix)
    record_size       one block's whole-record bytes

Accepted views have dense rows (view.py), so the states inside a block are
evenly spaced by ``state_strides`` and every token window is one span per
view -- whole blocks included (``local_start = 0``):

    ptr  = base_ptr + block_id * block_stride
           + (local_start * states_per_block // token_block) * state_stride
    size = (local_end - local_start) * states_per_block // token_block
           * state_stride

``extract_segments`` is that formula as array arithmetic over the columns.
It knows no record and no scheduling; the shell composes per-hash-block
record offsets from ``record_slots``/``record_size``.  Two special cases:
a state group's pages are indivisible checkpoints, so windows are forced
to whole blocks; and Block First layouts (0.29 declared, ``layer_stride <
block_stride``) tile their block slot with the descriptors' layer pages,
so an unfiltered whole batch is one IO span per block
(``block_first_segments``), paddings riding inside.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import view
from .view import LAYOUT_DEBUG, TensorView, layout_debug

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
class BlockFirstSpan:
    """One group's contiguous per-block span on a Block First layout.

    The group's descriptors tile the block slot back to back (offset
    chain, paddings riding inside), so one block of this group is one
    IO span: every layer page of every descriptor in it.
    """

    base_ptr: int  # first layer of the first (lowest-offset) descriptor
    block_stride: int
    span_bytes: int  # sum of per-descriptor layer_count * layer_stride


class KVCacheGroupLayout:
    """One KV group's addressing facts, distilled once into columns.

    ``extract_segments`` answers (ptr, size) grids for per-block token
    windows; ``block_first_segments`` the group-span special case;
    ``record_slots``/``record_size`` carry the whole-block record ledger
    for the shell.
    """

    def __init__(
        self,
        group: "UCMKVCacheGroupInfo",
        kv_caches: "Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]]",
    ) -> None:
        self.group_id = group.group_id
        self.token_block_size = group.token_block_size
        self.is_state_snapshot = group.is_state_snapshot
        self.num_blocks = group.layers[0].num_blocks
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        # Walk the group's layers once, in layer order.  One model layer
        # contributes several layer names (attn, indexer.k_cache, swa_cache,
        # compressor states all belong to layer 5), so layer_index is the
        # order key and the name is only a stable tiebreak within the same
        # layer -- the record layout must not depend on the config's
        # enumeration order.
        ordered_layers = sorted(
            group.layers, key=lambda item: (item.layer_index, item.layer_name)
        )
        layer_names: list[str] = []
        layer_ids: list[int] = []
        base_ptrs: list[int] = []
        block_strides: list[int] = []
        state_strides: list[int] = []
        states_per_block: list[int] = []
        payload_bytes: list[int] = []
        views_by_name: dict[str, tuple[TensorView, ...]] = {}
        for layer in ordered_layers:
            tensor_views = view.build_tensor_views(
                kv_caches[layer.layer_name],
                layer,
                state_snapshot=group.is_state_snapshot,
            )
            if not tensor_views:
                raise ValueError(
                    f"Layer {layer.layer_name} registered no addressable view"
                )
            if layer.descriptor is not None:
                for tensor_view in tensor_views:
                    if tensor_view.block_stride_bytes != layer.descriptor.block_stride:
                        raise ValueError(
                            f"Layer {layer.layer_name}: view block stride "
                            f"{tensor_view.block_stride_bytes} disagrees with the "
                            f"declared {layer.descriptor.block_stride}"
                        )
            views_by_name[layer.layer_name] = tensor_views
            for tensor_view in tensor_views:
                layer_names.append(layer.layer_name)
                layer_ids.append(layer.layer_index)
                base_ptrs.append(tensor_view.base_ptr)
                block_strides.append(tensor_view.block_stride_bytes)
                state_strides.append(tensor_view.bytes_per_state)
                states_per_block.append(tensor_view.states_per_block)
                payload_bytes.append(tensor_view.payload_bytes)

        # Flat columns carry the arithmetic; per-layer slices answer
        # "which rows belong to model layer N" without a ragged array.
        self.layer_names: tuple[str, ...] = tuple(layer_names)
        self.layer_ids = np.asarray(layer_ids, dtype=np.int64)
        self.base_ptrs = np.asarray(base_ptrs, dtype=np.int64)
        self.block_strides = np.asarray(block_strides, dtype=np.int64)
        self.state_strides = np.asarray(state_strides, dtype=np.int64)
        self.states_per_block = np.asarray(states_per_block, dtype=np.int64)
        self.payload_bytes = np.asarray(payload_bytes, dtype=np.int64)
        layer_slices: dict[int, slice] = {}
        for row, layer_id in enumerate(layer_ids):
            first = layer_slices.setdefault(layer_id, slice(row, row + 1))
            layer_slices[layer_id] = slice(first.start, row + 1)
        self.layer_slices = layer_slices

        self.block_first, tile_starts = self._block_first_span(
            ordered_layers, views_by_name
        )
        if tile_starts is not None:
            assert self.block_first is not None  # set together with starts
            # Block First record: a layer's slot is its page position inside
            # its descriptor's tile -- the very coordinates the span's bytes
            # land at, so layered queries address the same record a
            # whole-batch span wrote.
            slots: list[int] = []
            for layer in ordered_layers:
                descriptor = layer.descriptor
                assert descriptor is not None  # checked by the builder
                anchor = tile_starts[id(descriptor)] + (
                    layer.descriptor_position * descriptor.layer_stride
                )
                slots.extend((anchor,) * len(views_by_name[layer.layer_name]))
            self.record_slots = np.asarray(slots, dtype=np.int64)
            self.record_size = self.block_first.span_bytes
        else:
            # Layer First record: per-view payload slots, back to back.
            self.record_slots = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(self.payload_bytes)[:-1])
            )
            self.record_size = int(self.payload_bytes.sum())
        if LAYOUT_DEBUG:
            layout_debug(
                f"group-layout group={self.group_id} "
                f"layers={len(set(self.layer_names))} views={len(self.layer_names)} "
                f"state={int(self.is_state_snapshot)} "
                f"block-first={int(self.block_first is not None)} "
                f"record_per_block={self.record_size} token_block={self.token_block_size}"
            )

    def _block_first_span(
        self,
        ordered_layers: "Sequence[UCMLayerSpec]",
        views_by_name: Mapping[str, tuple[TensorView, ...]],
    ) -> tuple[BlockFirstSpan | None, dict[int, int] | None]:
        """The group's Block First span, or ``(None, None)`` to stay per-view.

        The group's descriptors tile its block slot back to back (one
        block is owned by one group, and vLLM places the group's
        descriptors at consecutive offsets), so the whole group is one
        IO span per block.  Requires: every declared layer has a single
        tensor view, every descriptor is interleaved
        (``0 < layer_stride < block_stride``) with its pages fitting the
        stride, and the descriptors' offset chain is gapless.
        """

        descriptors = []
        seen = set()
        for layer in ordered_layers:
            if layer.descriptor is None:
                return None, None  # mixed groups keep the per-view record
            if id(layer.descriptor) in seen:
                continue
            seen.add(id(layer.descriptor))
            descriptors.append(layer.descriptor)
        tiles = []
        for descriptor in descriptors:
            tensor_views = views_by_name[descriptor.layers[0]]
            if len(tensor_views) != 1:
                return None, None  # multi-view layers keep the per-view record
            if not 0 < descriptor.layer_stride < descriptor.block_stride:
                return None, None  # layer-contiguous placements do not tile
            if tensor_views[0].payload_bytes > descriptor.layer_stride:
                return None, None
            tiles.append((descriptor, len(descriptor.layers) * descriptor.layer_stride))
        tiles.sort(key=lambda item: item[0].offset)
        starts: dict[int, int] = {}
        record = 0
        for descriptor, tile_bytes in tiles:
            if descriptor.offset != record:
                return None, None  # offset chain must be gapless
            starts[id(descriptor)] = record
            record += tile_bytes
        first_descriptor = tiles[0][0]
        span = BlockFirstSpan(
            base_ptr=views_by_name[first_descriptor.layers[0]][0].base_ptr,
            block_stride=first_descriptor.block_stride,
            span_bytes=record,
        )
        return span, starts

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def view_mask(
        self,
        layer_names: "Collection[str] | None" = None,
        layer_ids: Sequence[int] | None = None,
    ) -> np.ndarray | None:
        """Boolean column mask selecting views (None = every view).

        Exact names or every cache name of the model layers with these
        IDs (attention and indexer of one layer share the ID).
        """

        if layer_names is not None and layer_ids is not None:
            raise ValueError("Specify either layer_names or layer_ids")
        if layer_ids is not None:
            return np.isin(self.layer_ids, np.asarray(layer_ids, dtype=np.int64))
        if layer_names is None:
            return None
        wanted = set(layer_names)
        return np.asarray(
            [name in wanted for name in self.layer_names], dtype=np.bool_
        )

    def extract_segments(
        self,
        block_ids: Sequence[int],
        local_starts: Sequence[int],
        local_ends: Sequence[int],
        layer_ids: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(ptrs, sizes) grids [n_blocks, n_views] for per-block windows.

        ``block_ids`` carry the windows in logical order: ``local_starts``
        / ``local_ends`` are each block's token window (whole block =
        ``(0, token_block_size)``).  State groups ignore the windows --
        a state page is an indivisible checkpoint, so every window is
        the whole block.  ``layer_ids`` restricts the columns to every
        cache name of those model layers.
        """

        blocks = np.asarray(block_ids, dtype=np.int64)
        if self.is_state_snapshot:
            starts = np.zeros(len(blocks), dtype=np.int64)
            ends = np.full(len(blocks), self.token_block_size, dtype=np.int64)
        else:
            starts = np.asarray(local_starts, dtype=np.int64)
            ends = np.asarray(local_ends, dtype=np.int64)
        self._checked_blocks(blocks)
        begins = starts[:, None] * self.states_per_block[None, :]
        stops = ends[:, None] * self.states_per_block[None, :]
        unrepresentable = (begins % self.token_block_size) | (
            stops % self.token_block_size
        )
        if unrepresentable.any():
            names = ", ".join(
                self.layer_names[index]
                for index in np.nonzero(unrepresentable.any(axis=0))[0][:4]
            )
            raise ValueError(
                f"Logical token windows cannot be represented exactly by "
                f"tensor layout (views starting at {names})"
            )
        state_begin = begins // self.token_block_size
        state_end = stops // self.token_block_size
        ptrs = (
            self.base_ptrs[None, :]
            + blocks[:, None] * self.block_strides[None, :]
            + state_begin * self.state_strides[None, :]
        )
        sizes = (state_end - state_begin) * self.state_strides[None, :]
        if layer_ids is not None:
            mask = np.isin(self.layer_ids, np.asarray(layer_ids, dtype=np.int64))
            ptrs = ptrs[:, mask]
            sizes = sizes[:, mask]
        return ptrs, sizes

    def block_first_segments(
        self, block_ids: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """One group-sized (ptr, size) per block, paddings riding inside.

        The Block First special case for whole unfiltered batches: the
        descriptors tile the block slot back to back, so the entire slot
        is one IO span.
        """

        span = self.block_first
        assert span is not None  # caller checks block_first is not None
        blocks = np.asarray(block_ids, dtype=np.int64)
        self._checked_blocks(blocks)
        ptrs = span.base_ptr + blocks * span.block_stride
        sizes = np.full(len(blocks), span.span_bytes, dtype=np.int64)
        return ptrs, sizes

    def _checked_blocks(self, blocks: np.ndarray) -> None:
        if len(blocks) and ((blocks < 0) | (blocks >= self.num_blocks)).any():
            bad = blocks[(blocks < 0) | (blocks >= self.num_blocks)][0]
            raise ValueError(
                f"vLLM block ID {int(bad)} is outside [0, {self.num_blocks})"
            )
