"""Physical KV-cache access, compiled once and resolved over block IDs.

``compile_access`` converts local token ranges to per-segment byte offsets and
sizes. ``BlockAccess.resolve`` supplies the physical block IDs. Neither needs
hash keys or UCM window rules; record_layout.py composes their results into
records. Whole Block First spans retain padding for the existing fast path.

Each column is a contiguous token segment, not necessarily an entire view.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import view
from .view import LayerView, MemorySegment

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
class BlockFirstView:
    """One group's contiguous per-block span on a Block First layout.

    The group's descriptors tile the block slot back to back (offset
    chain, paddings riding inside), so one block of this group is one
    IO span: every layer page of every descriptor in it.
    """

    base_ptr: int  # first layer of the first (lowest-offset) descriptor
    block_stride: int
    block_size_bytes: int  # sum of per-descriptor layer_count * layer_stride


@dataclass(frozen=True)
class BlockAccess:
    """Physical access pattern, with arrays shaped [window block, segment].

    No hash keys or storage-record offsets live here. A pattern repeats for
    each window passed to resolve(); dynamic token offsets are per block.
    Sizes and fixed byte offsets are compiled once. Columns represent segments;
    one component view can contribute multiple head segments.
    """

    layout: "KVCacheGroupLayout"
    block_byte_offsets: np.ndarray
    segment_bytes: np.ndarray
    selected_segments: np.ndarray | None = None

    def resolve(self, block_ids, *, token_offsets=None, segment_mask=None):
        """Return independent writable [block, segment] pointer/size grids."""
        ptrs, fixed_sizes = self._resolve(
            block_ids, token_offsets=token_offsets, segment_mask=segment_mask
        )
        return ptrs, np.tile(fixed_sizes, (len(ptrs) // len(fixed_sizes), 1))

    def resolve_ptrs(self, block_ids, *, token_offsets=None, segment_mask=None):
        """Resolve pointers without expanding the already compiled size template."""
        return self._resolve(
            block_ids, token_offsets=token_offsets, segment_mask=segment_mask
        )[0]

    def _resolve(
        self,
        block_ids: Sequence[int] | np.ndarray,
        *,
        token_offsets: np.ndarray | None = None,
        segment_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return [block, segment] pointer/size grids in input block order.

        Dynamic token offsets add to the compiled start. Use a zero-start
        template when supplying absolute local starts (the FA sub-block case).
        """
        blocks = np.asarray(block_ids, dtype=np.uint64)
        if blocks.ndim != 1:
            raise ValueError("Block IDs must be one-dimensional")
        self.layout._checked_blocks(blocks)
        rows = len(self.segment_bytes)
        if len(blocks) % rows:
            raise ValueError("Block IDs must contain complete access windows")
        repeats = len(blocks) // rows
        if self.selected_segments is not None:
            segment_mask = (
                self.selected_segments
                if segment_mask is None
                else self.selected_segments & segment_mask
            )
        columns: slice | np.ndarray = (
            slice(None) if segment_mask is None else segment_mask
        )
        # Keep the fixed template small. Broadcast it across access windows
        # into this call's pointer buffer instead of tiling source offsets.
        byte_offsets = self.block_byte_offsets[:, columns]
        fixed_sizes = self.segment_bytes[:, columns]
        dynamic_offsets = None
        if token_offsets is not None:
            starts = np.asarray(token_offsets, dtype=np.int64)
            if starts.shape != blocks.shape:
                raise ValueError("One dynamic token offset is required per block")
            if ((starts < 0) | (starts >= self.layout.token_block_size)).any():
                raise ValueError("Dynamic token offset must lie inside a block")
            dynamic_offsets = self.layout.tokens_to_segment_bytes(
                starts.astype(np.uint64), segment_mask
            ).reshape(repeats, rows, fixed_sizes.shape[1])
            ends = dynamic_offsets + byte_offsets + fixed_sizes
        else:
            ends = byte_offsets + fixed_sizes
        if (ends > self.layout.payload_bytes[columns]).any():
            raise ValueError("Access range extends beyond a physical block")
        ptrs = np.empty((len(blocks), fixed_sizes.shape[1]), dtype=np.uint64)
        np.multiply(blocks[:, None], self.layout.block_strides[None, columns], out=ptrs)
        ptrs += self.layout.base_ptrs[None, columns]
        windows = ptrs.reshape(repeats, rows, fixed_sizes.shape[1])
        windows += byte_offsets
        if dynamic_offsets is not None:
            windows += dynamic_offsets
        return ptrs, fixed_sizes


class KVCacheGroupLayout:
    """One KV group's addressing facts, distilled once into columns.

    ``extract_segments`` answers (ptr, size) grids for per-block token
    windows; ``block_first_segments`` the group-span special case;
    Storage offsets and record packing are owned by GroupRecordLayout.
    """

    def __init__(
        self,
        group: "UCMKVCacheGroupInfo",
        kv_caches: "Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]]",
        *,
        device_type: str = "npu",
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
        segments_by_name: dict[str, tuple[MemorySegment, ...]] = {}
        self.layer_views: dict[str, LayerView] = {}
        for layer in ordered_layers:
            layer_view = view.build_layer_view(
                kv_caches[layer.layer_name],
                layer,
                state_snapshot=group.is_state_snapshot,
                device_type=device_type,
            )
            self.layer_views[layer.layer_name] = layer_view
            segments = layer_view.segments
            if not segments:
                raise ValueError(
                    f"Layer {layer.layer_name} registered no addressable view"
                )
            if layer.descriptor is not None:
                for segment in segments:
                    if segment.block_stride_bytes != layer.descriptor.block_stride:
                        raise ValueError(
                            f"Layer {layer.layer_name}: view block stride "
                            f"{segment.block_stride_bytes} disagrees with the "
                            f"declared {layer.descriptor.block_stride}"
                        )
            segments_by_name[layer.layer_name] = segments
            for segment in segments:
                layer_names.append(layer.layer_name)
                layer_ids.append(layer.layer_index)
                base_ptrs.append(segment.base_ptr)
                block_strides.append(segment.block_stride_bytes)
                state_strides.append(segment.bytes_per_state)
                states_per_block.append(segment.states_per_block)
                payload_bytes.append(segment.payload_bytes)

        # Flat columns carry the arithmetic; per-layer slices answer
        # "which columns belong to model layer N" without a ragged array.
        self.layer_names: tuple[str, ...] = tuple(layer_names)
        self.layer_ids = np.asarray(layer_ids, dtype=np.uint64)
        self.base_ptrs = np.asarray(base_ptrs, dtype=np.uint64)
        self.block_strides = np.asarray(block_strides, dtype=np.uint64)
        self.state_strides = np.asarray(state_strides, dtype=np.uint64)
        self.states_per_block = np.asarray(states_per_block, dtype=np.uint64)
        self.payload_bytes = np.asarray(payload_bytes, dtype=np.uint64)
        layer_slices: dict[int, slice] = {}
        for row, layer_id in enumerate(layer_ids):
            first = layer_slices.setdefault(layer_id, slice(row, row + 1))
            layer_slices[layer_id] = slice(first.start, row + 1)
        self.layer_slices = layer_slices

        self.block_first = self._block_first_span(ordered_layers, segments_by_name)

    def _block_first_span(
        self,
        ordered_layers: "Sequence[UCMLayerSpec]",
        segments_by_name: Mapping[str, tuple[MemorySegment, ...]],
    ) -> BlockFirstView | None:
        """The group's Block First span, or ``None`` for fragmented source memory.

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
                return None  # mixed groups keep the per-view record
            if id(layer.descriptor) in seen:
                continue
            seen.add(id(layer.descriptor))
            descriptors.append(layer.descriptor)
        tiles = []
        for descriptor in descriptors:
            segments = segments_by_name[descriptor.layers[0]]
            # The fast path is valid only if every layer's segments form a
            # contiguous page at its declared address, including split HNC.
            first_base = segments[0].base_ptr
            for position, name in enumerate(descriptor.layers):
                if len(self.layer_views[name].components) != 1:
                    return None
                cursor = first_base + position * descriptor.layer_stride
                total_payload = 0
                for segment in segments_by_name[name]:
                    if segment.base_ptr != cursor:
                        return None
                    cursor += segment.payload_bytes
                    total_payload += segment.payload_bytes
                if total_payload > descriptor.layer_stride:
                    return None
            if not 0 < descriptor.layer_stride < descriptor.block_stride:
                return None  # layer-contiguous placements do not tile
            tiles.append((descriptor, len(descriptor.layers) * descriptor.layer_stride))
        tiles.sort(key=lambda item: item[0].offset)
        block_offset = 0
        first_descriptor = tiles[0][0]
        first_base = segments_by_name[first_descriptor.layers[0]][0].base_ptr
        for descriptor, tile_bytes in tiles:
            # Declarations alone cannot prove that separate views share the
            # same backing or that their slots advance together across blocks.
            if (
                descriptor.block_stride != first_descriptor.block_stride
                or segments_by_name[descriptor.layers[0]][0].base_ptr
                != first_base + descriptor.offset
            ):
                return None
            if descriptor.offset != block_offset:
                return None  # offset chain must be gapless
            block_offset += tile_bytes
        span = BlockFirstView(
            base_ptr=segments_by_name[first_descriptor.layers[0]][0].base_ptr,
            block_stride=first_descriptor.block_stride,
            block_size_bytes=block_offset,
        )
        return span

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def tokens_to_segment_bytes(
        self, tokens: int | np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Per-segment bytes whole tokens occupy -- one conversion, two roles.

        States are evenly spaced inside a block (states_per_block states
        per token_block tokens), so "the bytes t tokens occupy" and "the
        bytes the first t tokens of a block span" are the same number: a
        partial block's live size (template row 0) and a window head's
        pointer skip (a mid-block start) both read here.  FA sub-span
        heads pass a per-key array and get a (key, segment) grid back.
        Windows that are not a whole number of states raise.
        """

        # outer product == the (K, 1) x (1, V) broadcast: one scalar (or
        # one row) of token counts against every segment's state column.
        columns: slice | np.ndarray = slice(None) if mask is None else mask
        states = np.multiply.outer(tokens, self.states_per_block[columns])
        leftover = states % self.token_block_size
        if leftover.any():
            bad = np.nonzero(leftover.reshape(-1, leftover.shape[-1]).any(axis=0))[0][
                :4
            ]
            selected_names = np.asarray(self.layer_names)[columns]
            names = ", ".join(selected_names[index] for index in bad)
            raise ValueError(
                "Token windows are not representable exactly by tensor "
                f"layout (views starting at {names})"
            )
        return (states // self.token_block_size) * self.state_strides[columns]

    def segment_count(self, mask: "np.ndarray | None") -> int:
        return len(self.layer_names) if mask is None else int(mask.sum())

    def segment_mask(
        self,
        layer_names: "Collection[str] | None" = None,
        layer_ids: Sequence[int] | None = None,
    ) -> np.ndarray | None:
        """Boolean column mask selecting segments (None = every segment).

        Exact names or every cache name of the model layers with these
        IDs (attention and indexer of one layer share the ID).
        """

        if layer_names is not None and layer_ids is not None:
            raise ValueError("Specify either layer_names or layer_ids")
        if layer_ids is not None:
            return np.isin(self.layer_ids, np.asarray(layer_ids, dtype=np.uint64))
        if layer_names is None:
            return None
        wanted = set(layer_names)
        return np.asarray([name in wanted for name in self.layer_names], dtype=np.bool_)

    def compile_access(
        self,
        *,
        token_offsets: int | Sequence[int] | np.ndarray = 0,
        token_counts: int | Sequence[int] | np.ndarray | None = None,
        layer_ids: Sequence[int] | None = None,
    ) -> BlockAccess:
        """Compile local token ranges into physical byte offsets and sizes.

        Scalars apply to every row; arrays describe a repeating block window.
        Omitted counts extend to the block end. State pages are indivisible.
        Head-separated components have one column per head segment.
        """
        starts = np.atleast_1d(np.asarray(token_offsets, dtype=np.int64))
        counts = (
            self.token_block_size - starts
            if token_counts is None
            else np.atleast_1d(np.asarray(token_counts, dtype=np.int64))
        )
        starts, counts = np.broadcast_arrays(starts, counts)
        if starts.ndim != 1 or not len(starts):
            raise ValueError("Token ranges must be non-empty one-dimensional arrays")
        if (
            (starts < 0) | (counts <= 0) | (starts + counts > self.token_block_size)
        ).any():
            raise ValueError("Token range must lie inside one physical block")
        if self.is_state_snapshot and (
            (starts != 0).any() or (counts != self.token_block_size).any()
        ):
            raise ValueError("State snapshots require whole-block access")
        return BlockAccess(
            self,
            self.tokens_to_segment_bytes(starts.astype(np.uint64)),
            self.tokens_to_segment_bytes(counts.astype(np.uint64)),
            self.segment_mask(layer_ids=layer_ids),
        )

    def extract_segments(
        self,
        block_ids: Sequence[int],
        local_starts: Sequence[int],
        local_ends: Sequence[int],
        layer_ids: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility query using the same addressing path as dispatch.

        State queries retain their existing whole-page semantics.
        """
        if not len(block_ids):
            empty = np.empty(
                (0, self.segment_count(self.segment_mask(layer_ids=layer_ids))),
                dtype=np.uint64,
            )
            return empty, empty.copy()
        if self.is_state_snapshot:
            access = self.compile_access()
        else:
            starts = np.asarray(local_starts, dtype=np.int64)
            ends = np.asarray(local_ends, dtype=np.int64)
            access = self.compile_access(
                token_offsets=starts, token_counts=ends - starts
            )
        return access.resolve(
            block_ids, segment_mask=self.segment_mask(layer_ids=layer_ids)
        )

    def block_first_segments(
        self, block_ids: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """One group-sized (ptr, size) per block, paddings riding inside.

        The Block First special case for whole unfiltered batches: the
        descriptors tile the block slot back to back, so the entire slot
        is one IO span.
        """

        span = self.block_first
        assert span is not None  # caller checks block_first is not None
        blocks = np.asarray(block_ids, dtype=np.uint64)
        self._checked_blocks(blocks)
        ptrs = span.base_ptr + blocks * span.block_stride
        sizes = np.full(len(blocks), span.block_size_bytes, dtype=np.uint64)
        return ptrs, sizes

    def _checked_blocks(self, blocks: np.ndarray) -> None:
        if len(blocks) and (blocks >= self.num_blocks).any():
            bad = blocks[blocks >= self.num_blocks][0]
            raise ValueError(
                f"vLLM block ID {int(bad)} is outside [0, {self.num_blocks})"
            )
