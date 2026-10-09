"""Physical KV-cache access, compiled once and resolved over block IDs.

``compile_access`` converts local token ranges to per-segment byte offsets and
sizes. ``BlockAccess.resolve_ptrs`` supplies the physical block IDs. Neither needs
hash keys or UCM window rules; store_layout.py composes their results into
records. Whole Block First spans retain padding for the fast path, including
State pages. Partial-token support is one policy for the entire group.

Each column is a contiguous token segment, not necessarily an entire view.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import view
from .view import LayerView

if TYPE_CHECKING:
    import torch

    from .kv_cache import UCMKVCacheGroupInfo, UCMLayerSpec


@dataclass(frozen=True)
class TensorDescriptor:
    """One native ``KVCacheTensor`` declaration, as handed over.

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
    each window passed to resolve_ptrs(); dynamic token offsets are per block.
    Sizes and fixed byte offsets are compiled once. Columns represent segments;
    one component view can contribute multiple head segments.
    """

    layout: "KVCacheGroupLayout"
    block_byte_offsets: np.ndarray
    segment_bytes: np.ndarray

    def resolve_ptrs(
        self,
        block_ids: Sequence[int] | np.ndarray,
        *,
        token_offsets: np.ndarray | None = None,
        segment_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a [block, segment] pointer grid in input block order.

        Dynamic token offsets add to the compiled start. Use a zero-start
        template when supplying absolute local starts (the FA sub-block case).
        """
        blocks = np.asarray(block_ids, dtype=np.uint64)
        self.layout._checked_blocks(blocks)
        blocks_per_chunk = len(self.segment_bytes)
        count = len(blocks) // blocks_per_chunk
        columns: slice | np.ndarray = (
            slice(None) if segment_mask is None else segment_mask
        )
        # Keep the fixed template small. Broadcast it across access windows
        # into this call's pointer buffer instead of tiling source offsets.
        byte_offsets = self.block_byte_offsets[:, columns]
        fixed_sizes = self.segment_bytes[:, columns]
        ptrs = (
            self.layout.base_ptrs[None, columns]
            + blocks.reshape(count, blocks_per_chunk, 1)
            * self.layout.block_strides[None, columns]
            + byte_offsets
        )
        if token_offsets is not None:
            starts = np.asarray(token_offsets, dtype=np.uint64)
            if (starts >= self.layout.token_block_size).any():
                raise ValueError("Dynamic token offset must lie inside a block")
            dynamic_offsets = self.layout.tokens_to_segment_bytes(
                starts, segment_mask
            ).reshape(count, blocks_per_chunk, fixed_sizes.shape[1])
            ends = dynamic_offsets + byte_offsets + fixed_sizes
            # Static ranges were checked by compile_access. Only the
            # per-call displacement needs a fresh source-bounds check.
            if (ends > self.layout.payload_bytes[columns]).any():
                raise ValueError("Access range extends beyond a physical block")
            ptrs += dynamic_offsets
        return ptrs.reshape(len(blocks), fixed_sizes.shape[1])


class KVCacheGroupLayout:
    """One KV group's addressing facts, distilled once into columns.

    ``compile_access`` builds per-block token-range templates;
    ``block_first_segments`` handles the group-span special case.
    Storage offsets and record packing are owned by GroupStoreLayout.
    """

    def __init__(
        self,
        group: "UCMKVCacheGroupInfo",
        kv_caches: "Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]]",
        *,
        layers: "Sequence[UCMLayerSpec]",
        block_first_layout: bool,
        device_type: str = "npu",
    ) -> None:
        self.group_id = group.group_id
        self.token_block_size = group.token_block_size
        self.is_state_snapshot = group.is_state_snapshot
        self.layers = tuple(
            sorted(layers, key=lambda item: (item.layer_index, item.layer_name))
        )
        self.num_blocks = self.layers[0].num_blocks

        # Walk the group's layers once, in layer order.  One model layer
        # contributes several layer names (attn, indexer.k_cache, swa_cache,
        # compressor states all belong to layer 5), so layer_index is the
        # order key and the name is only a stable tiebreak within the same
        # layer -- the record layout must not depend on the config's
        # enumeration order.
        ordered_layers = self.layers
        layer_names: list[str] = []
        layer_ids: list[int] = []
        base_ptrs: list[int] = []
        block_strides: list[int] = []
        state_strides: list[int] = []
        states_per_block: list[int] = []
        payload_bytes: list[int] = []
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
            for segment in segments:
                layer_names.append(layer.layer_name)
                layer_ids.append(layer.layer_index)
                base_ptrs.append(segment.base_ptr)
                block_strides.append(segment.block_stride_bytes)
                state_strides.append(segment.bytes_per_state)
                states_per_block.append(segment.states_per_block)
                payload_bytes.append(segment.payload_bytes)

        # Flat columns carry the arithmetic; layer IDs/names select columns.
        self.layer_names: tuple[str, ...] = tuple(layer_names)
        self.layer_ids = np.asarray(layer_ids, dtype=np.uint64)
        self.base_ptrs = np.asarray(base_ptrs, dtype=np.uint64)
        self.block_strides = np.asarray(block_strides, dtype=np.uint64)
        self.state_strides = np.asarray(state_strides, dtype=np.uint64)
        self.states_per_block = np.asarray(states_per_block, dtype=np.uint64)
        self.payload_bytes = np.asarray(payload_bytes, dtype=np.uint64)
        self.supports_partial_tokens = all(
            component.supports_partial_tokens
            for layer_view in self.layer_views.values()
            for component in layer_view.components
        )
        self.block_first = self._block_first_span() if block_first_layout else None

    def _block_first_span(self) -> BlockFirstView | None:
        """Use CUDA/CPU BLHNC/BLNHC placement declarations directly.

        The native allocator places a group's descriptors consecutively
        from offset zero in one backing. GLM's special slot descriptors
        have layer-outer strides and do not qualify. Ascend never enters
        this path; its registered components use per-layer segments.
        """
        descriptors: dict[int, TensorDescriptor] = {}
        for layer in self.layers:
            descriptor = layer.descriptor
            if descriptor is None:
                return None
            descriptors[id(descriptor)] = descriptor
        first = min(descriptors.values(), key=lambda item: item.offset)
        if any(
            not 0 < descriptor.layer_stride < descriptor.block_stride
            or descriptor.block_stride != first.block_stride
            for descriptor in descriptors.values()
        ):
            return None
        return BlockFirstView(
            base_ptr=self.layer_views[first.layers[0]].segments[0].base_ptr,
            block_stride=first.block_stride,
            block_size_bytes=sum(
                len(descriptor.layers) * descriptor.layer_stride
                for descriptor in descriptors.values()
            ),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def tokens_to_segment_bytes(
        self, tokens: int | np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Per-segment bytes whole tokens occupy -- one conversion, two roles.

        Partial access requires every view in the group to be token-contiguous.
        Multi-head HNC accepts only zero or the whole block span; its head-major bytes
        cannot be sliced using a per-token byte ratio. FA sub-block starts
        may be supplied as arrays. Layer selection does not relax this policy.
        Windows must contain whole stored states.
        """

        # outer product == the (K, 1) x (1, V) broadcast: one scalar (or
        # one row) of token counts against every segment's state column.
        columns: slice | np.ndarray = slice(None) if mask is None else mask
        token_counts = np.asarray(tokens)
        if not self.supports_partial_tokens and (
            (token_counts != 0) & (token_counts != self.token_block_size)
        ).any():
            raise ValueError(f"KV cache group {self.group_id} requires whole-block access")
        states = np.multiply.outer(tokens, self.states_per_block[columns])
        leftover = states % self.token_block_size
        if leftover.any():
            raise ValueError(
                f"Token ranges must align with stored states in group {self.group_id}"
            )
        return (states // self.token_block_size) * self.state_strides[columns]

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
    ) -> BlockAccess:
        """Compile local token ranges for every segment in this group.

        Scalars apply to every row; arrays describe a repeating block window.
        Omitted counts extend to the block end. State pages are indivisible.
        Physically separated heads have one column each. Multi-head HNC
        requires whole blocks even when its heads form one compact segment.
        """
        starts = np.atleast_1d(np.asarray(token_offsets, dtype=np.int64))
        counts = (
            self.token_block_size - starts
            if token_counts is None
            else np.atleast_1d(np.asarray(token_counts, dtype=np.int64))
        )
        starts, counts = np.broadcast_arrays(starts, counts)
        if (
            (starts < 0) | (counts <= 0) | (starts + counts > self.token_block_size)
        ).any():
            raise ValueError("Token range must lie inside one physical block")
        byte_offsets = self.tokens_to_segment_bytes(starts.astype(np.uint64))
        segment_bytes = self.tokens_to_segment_bytes(counts.astype(np.uint64))
        return BlockAccess(self, byte_offsets, segment_bytes)

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
        if (blocks >= self.num_blocks).any():
            bad = blocks[blocks >= self.num_blocks][0]
            raise ValueError(
                f"vLLM block ID {int(bad)} is outside [0, {self.num_blocks})"
            )
