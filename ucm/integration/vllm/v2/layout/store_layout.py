"""Compile one group's part of a UCM record against registered HBM geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .group import KVCacheGroupLayout
from .view import LAYOUT_DEBUG, layout_debug

if TYPE_CHECKING:
    from .kv_cache import UCMKVCacheGroupInfo


@dataclass(frozen=True)
class GroupStoreLayout:
    """One group's byte templates within a UCM block, independent of plans.

    HBM supplies physical block bases and segment geometry; this object owns
    token ranges, sizes and offsets. Selecting layers keeps their configured
    positions in the stored value.
    """

    group: UCMKVCacheGroupInfo
    group_layout: KVCacheGroupLayout
    block_byte_offsets: np.ndarray
    segment_bytes: np.ndarray
    ucm_block_offsets: np.ndarray
    store_bytes: int
    group_block_offsets: np.ndarray
    whole_block_bytes: int
    merge_whole_blocks: bool
    # FA UCM blocks smaller than a vLLM block: each row reads its own slice.
    dynamic_token_offsets: bool

    @classmethod
    def build(
        cls,
        group: UCMKVCacheGroupInfo,
        layout: KVCacheGroupLayout,
        ucm_block_size: int,
        *,
        use_layerwise: bool,
    ) -> GroupStoreLayout:
        token_block_size = group.token_block_size
        if group.is_sliding_window:
            partial_tokens = (group.tail_tokens or 0) % token_block_size
        elif group.is_state_snapshot:
            partial_tokens = 0
        else:
            # An FA UCM block smaller than a vLLM block stores one slice of it.
            partial_tokens = ucm_block_size % token_block_size
        is_fa_slice = bool(partial_tokens) and not group.is_sliding_window
        if partial_tokens and not layout.supports_partial_tokens:
            if is_fa_slice:
                raise ValueError(
                    f"FA group {group.group_id} cannot store {partial_tokens}-token "
                    f"slices of its {token_block_size}-token blocks: its KV layout "
                    "is not token-contiguous (for example multi-head HNC). Use a "
                    f"ucm_cache_block_size that is a multiple of {token_block_size}."
                )
            else:
                raise ValueError(
                    f"KV cache group {group.group_id} requires whole-block access"
                )
        if (
            np.uint64(partial_tokens)
            * layout.states_per_block
            % np.uint64(token_block_size)
        ).any():
            raise ValueError(
                f"Token ranges must align with stored states in group {group.group_id}"
            )

        # Zero-tail groups join no route, but still have a valid layout.
        num_tail_blocks = max(group.tail_blocks, 1)
        token_starts = np.zeros(num_tail_blocks, dtype=np.uint64)
        token_counts = np.full(num_tail_blocks, token_block_size, dtype=np.uint64)
        if partial_tokens:
            token_counts[0] = partial_tokens
            if group.is_sliding_window:
                token_starts[0] = token_block_size - partial_tokens
        block_byte_offsets = layout.tokens_to_bytes(token_starts, token_block_size)
        segment_bytes = layout.tokens_to_bytes(token_counts, token_block_size)

        # Full Block First pages have one storage entry, including padding.
        merge_whole_blocks = (
            not use_layerwise and layout.block_first is not None and partial_tokens == 0
        )
        if merge_whole_blocks:
            block_first_view = layout.block_first
            # Layerwise IO keeps each segment at its original page position.
            group_block_offsets = layout.base_ptrs - np.uint64(
                block_first_view.base_ptr
            )
            whole_block_bytes = block_first_view.block_size_bytes
            ucm_block_offsets = np.arange(num_tail_blocks, dtype=np.uint64)[
                :, None
            ] * np.uint64(whole_block_bytes)
            store_bytes = num_tail_blocks * whole_block_bytes
        else:
            group_block_offsets = np.cumsum(layout.payload_bytes) - layout.payload_bytes
            whole_block_bytes = int(layout.payload_bytes.sum())
            ucm_block_offsets = (
                np.cumsum(segment_bytes).reshape(segment_bytes.shape) - segment_bytes
            )
            store_bytes = int(segment_bytes.sum())
        if LAYOUT_DEBUG:
            layout_debug(
                f"store-layout group={group.group_id} "
                f"segments={len(layout.layer_names)} "
                f"block-first={int(merge_whole_blocks)} "
                f"tail_blocks={group.tail_blocks} partial_tokens={partial_tokens} "
                f"store_bytes={store_bytes}"
            )
        return cls(
            group=group,
            group_layout=layout,
            block_byte_offsets=block_byte_offsets,
            segment_bytes=segment_bytes,
            ucm_block_offsets=ucm_block_offsets,
            store_bytes=store_bytes,
            group_block_offsets=group_block_offsets,
            whole_block_bytes=whole_block_bytes,
            merge_whole_blocks=merge_whole_blocks,
            dynamic_token_offsets=is_fa_slice,
        )

    def resolve_matrices(
        self,
        vllm_block_ids: np.ndarray,
        token_starts: np.ndarray,
        *,
        segment_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return [UCM block, IO entry] offsets, ptrs and sizes.

        token_starts holds each UCM block's first token. An FA slice reads from
        that token's place in its vLLM block; other groups use the fixed
        template. Static sizes and offsets are broadcast from copies, so a
        retained transfer cannot change a compiled template or another call.
        """
        layout = self.group_layout
        num_ucm_blocks = len(token_starts)
        num_tail_blocks = max(self.group.tail_blocks, 1)
        if self.merge_whole_blocks and segment_mask is None:
            segment_ptrs = layout.resolve_block_first_ptrs(vllm_block_ids)
            store_offsets = self.ucm_block_offsets.reshape(-1).copy()
            segment_sizes = np.full(
                num_tail_blocks, self.whole_block_bytes, dtype=np.uint64
            )
        else:
            segment_columns: slice | np.ndarray = (
                slice(None) if segment_mask is None else segment_mask
            )
            fixed_segment_sizes = self.segment_bytes[:, segment_columns]
            segment_ptrs = layout.resolve_ptrs(
                vllm_block_ids, segment_mask=segment_mask
            ).reshape(num_ucm_blocks, num_tail_blocks, fixed_segment_sizes.shape[1])
            segment_ptrs += self.block_byte_offsets[:, segment_columns]
            if self.dynamic_token_offsets:
                token_block_size = np.uint64(self.group.token_block_size)
                segment_ptrs += layout.tokens_to_bytes(
                    np.asarray(token_starts, dtype=np.uint64) % token_block_size,
                    self.group.token_block_size,
                    segment_mask,
                )[:, None, :]
            if self.merge_whole_blocks:
                store_offsets = (
                    self.ucm_block_offsets
                    + self.group_block_offsets[segment_columns][None, :]
                ).reshape(-1)
            else:
                store_offsets = (
                    self.ucm_block_offsets[:, segment_columns].reshape(-1).copy()
                )
            segment_sizes = fixed_segment_sizes.reshape(-1).copy()
        matrix_shape = (num_ucm_blocks, len(segment_sizes))
        return (
            np.broadcast_to(store_offsets, matrix_shape),
            segment_ptrs.reshape(matrix_shape),
            np.broadcast_to(segment_sizes, matrix_shape),
        )
