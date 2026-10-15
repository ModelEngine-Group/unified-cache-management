"""Compile a shared chunk contract against registered HBM geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .group import KVCacheGroupLayout
from .view import LAYOUT_DEBUG, layout_debug

if TYPE_CHECKING:
    from .kv_cache import UCMKVCacheSpec


@dataclass(frozen=True)
class GroupStoreLayout:
    """One group's byte templates within a chunk, independent of keys/plans.

    The shared spec owns the chunk's block count. HBM supplies physical block
    bases and segment geometry; this object owns ranges, sizes and offsets.
    Selecting layers preserves their configured positions in the stored value.
    """

    spec: UCMKVCacheSpec
    group_layout: KVCacheGroupLayout
    block_byte_offsets: np.ndarray
    segment_bytes: np.ndarray
    ucm_block_offsets: np.ndarray
    store_bytes: int
    group_block_offsets: np.ndarray
    whole_block_bytes: int
    merge_whole_blocks: bool

    @classmethod
    def build(
        cls,
        spec: UCMKVCacheSpec,
        layout: KVCacheGroupLayout,
        *,
        use_layerwise: bool,
    ) -> GroupStoreLayout:
        group = spec.groups[layout.group_id]
        block_tokens = group.tokens_per_block
        if group.is_sliding_window:
            partial_tokens = (group.tail_tokens or 0) % block_tokens
        else:
            partial_tokens = spec.tokens_per_chunk % block_tokens

        rows = spec.blocks_per_chunk[group.group_id]
        starts = np.zeros(rows, dtype=np.uint64)
        counts = np.full(rows, block_tokens, dtype=np.uint64)
        if partial_tokens:
            if not layout.supports_partial_tokens:
                raise ValueError(
                    f"KV cache group {layout.group_id} requires whole-block access"
                )
            if (
                np.uint64(partial_tokens)
                * layout.states_per_block
                % np.uint64(block_tokens)
            ).any():
                raise ValueError(
                    f"Token ranges must align with stored states in group {layout.group_id}"
                )
            counts[0] = partial_tokens
            if group.is_sliding_window:
                starts[0] = block_tokens - partial_tokens
        byte_offsets = layout.tokens_to_bytes(starts, block_tokens)
        sizes = layout.tokens_to_bytes(counts, block_tokens)

        merge_whole_blocks = (
            not use_layerwise and layout.block_first is not None and partial_tokens == 0
        )
        if merge_whole_blocks:
            span = layout.block_first
            group_block_offsets = layout.base_ptrs - np.uint64(span.base_ptr)
            whole_block_bytes = span.block_size_bytes
            ucm_block_offsets = np.arange(rows, dtype=np.uint64)[:, None] * np.uint64(
                whole_block_bytes
            )
            store_bytes = rows * whole_block_bytes
        else:
            group_block_offsets = np.cumsum(layout.payload_bytes) - layout.payload_bytes
            whole_block_bytes = int(layout.payload_bytes.sum())
            ucm_block_offsets = np.cumsum(sizes).reshape(sizes.shape) - sizes
            store_bytes = int(sizes.sum())
        if LAYOUT_DEBUG:
            layout_debug(
                f"store-layout group={group.group_id} segments={len(layout.layer_names)} "
                f"block-first={int(merge_whole_blocks)} "
                f"blocks_per_chunk={rows} partial_tokens={partial_tokens} "
                f"store_bytes={store_bytes}"
            )
        return cls(
            spec=spec,
            group_layout=layout,
            block_byte_offsets=byte_offsets,
            segment_bytes=sizes,
            ucm_block_offsets=ucm_block_offsets,
            store_bytes=store_bytes,
            group_block_offsets=group_block_offsets,
            whole_block_bytes=whole_block_bytes,
            merge_whole_blocks=merge_whole_blocks,
        )

    def resolve_matrices(
        self,
        blocks: np.ndarray,
        token_starts: np.ndarray,
        *,
        segment_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return [key, IO entry] offsets, ptrs, sizes from per-call buffers.

        Non-SWA offsets are the supplied per-key logical chunk starts modulo
        the group's block span; SWA uses its fixed tail template.
        Static sizes/offsets are broadcast from copies, so retained transfers
        cannot mutate compiled templates or be overwritten by subsequent calls.
        """
        layout = self.group_layout
        key_count = len(token_starts)
        blocks_per_chunk = self.spec.blocks_per_chunk[layout.group_id]
        if self.merge_whole_blocks and segment_mask is None:
            ptrs = layout.resolve_block_first_ptrs(blocks)
            offsets = self.ucm_block_offsets.reshape(-1).copy()
            sizes = np.full(blocks_per_chunk, self.whole_block_bytes, dtype=np.uint64)
        else:
            columns = slice(None) if segment_mask is None else segment_mask
            fixed_offsets = self.block_byte_offsets[:, columns]
            fixed_sizes = self.segment_bytes[:, columns]
            ptrs = layout.resolve_ptrs(blocks, segment_mask=segment_mask).reshape(
                key_count, blocks_per_chunk, fixed_sizes.shape[1]
            )
            ptrs += fixed_offsets
            group = self.spec.groups[layout.group_id]
            block_tokens = group.tokens_per_block
            if not group.is_sliding_window:
                starts = token_starts % np.uint64(block_tokens)
                dynamic_offsets = layout.tokens_to_bytes(
                    starts, block_tokens, segment_mask
                )[:, None, :]
                ptrs += dynamic_offsets
            if self.merge_whole_blocks:
                offsets = (
                    self.ucm_block_offsets + self.group_block_offsets[columns][None, :]
                ).reshape(-1)
            else:
                offsets = self.ucm_block_offsets[:, columns].reshape(-1).copy()
            sizes = fixed_sizes.reshape(-1).copy()
        shape = (key_count, len(sizes))
        return (
            np.broadcast_to(offsets, shape),
            ptrs.reshape(shape),
            np.broadcast_to(sizes, shape),
        )
