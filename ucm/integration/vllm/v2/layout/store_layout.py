"""Compose physical block accesses into a group's part of a UCM record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .group import BlockAccess, KVCacheGroupLayout
from .view import LAYOUT_DEBUG, layout_debug

if TYPE_CHECKING:
    from .kv_cache import UCMKVCacheGroupInfo


@dataclass(frozen=True)
class GroupStoreLayout:
    """One group's storage template, independent of UCM block ids and plans.

    Offsets are relative to this group's start. The caller chooses participating
    groups and adds their storage bases to form the complete stored value.
    Whole Block First slots include padding. Selecting segments preserves the
    configured storage offsets; it does not repack the stored value.
    """

    vllm_blocks_per_ucm_block: int
    access: BlockAccess
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
        token_block_size = layout.token_block_size
        if group.is_sliding_window:
            partial_tokens = (group.tail_tokens or 0) % token_block_size
        elif group.is_state_snapshot:
            partial_tokens = 0
        else:
            # An FA UCM block smaller than a vLLM block stores one slice of it.
            partial_tokens = ucm_block_size % token_block_size
        is_fa_slice = bool(partial_tokens) and not group.is_sliding_window
        if is_fa_slice and not layout.supports_partial_tokens:
            raise ValueError(
                f"FA group {group.group_id} cannot store {partial_tokens}-token "
                f"slices of its {token_block_size}-token blocks: its KV layout is "
                "not token-contiguous (for example multi-head HNC). Use a "
                f"ucm_cache_block_size that is a multiple of {token_block_size}."
            )

        # Zero-tail groups join no route, but still have a valid layout.
        num_tail_blocks = max(group.tail_blocks, 1)
        token_offsets = np.zeros(num_tail_blocks, dtype=np.uint64)
        token_counts = np.full(num_tail_blocks, token_block_size, dtype=np.uint64)
        if partial_tokens:
            token_counts[0] = partial_tokens
            if group.is_sliding_window:
                token_offsets[0] = token_block_size - partial_tokens
        access = layout.compile_access(
            token_offsets=token_offsets, token_counts=token_counts
        )

        # Full Block First pages have one storage entry, including padding.
        merge_whole_blocks = (
            not use_layerwise and layout.block_first is not None and partial_tokens == 0
        )
        if merge_whole_blocks:
            group_block_offsets = np.zeros(1, dtype=np.uint64)
            whole_block_bytes = layout.block_first.block_size_bytes
        else:
            group_block_offsets = np.cumsum(layout.payload_bytes) - layout.payload_bytes
            whole_block_bytes = int(layout.payload_bytes.sum())

        if merge_whole_blocks:
            ucm_block_offsets = (
                np.arange(num_tail_blocks, dtype=np.uint64)[:, None] * whole_block_bytes
            )
            store_bytes = num_tail_blocks * whole_block_bytes
        else:
            segment_sizes = access.segment_bytes
            ucm_block_offsets = (
                np.cumsum(segment_sizes).reshape(segment_sizes.shape) - segment_sizes
            )
            store_bytes = int(segment_sizes.sum())
        if LAYOUT_DEBUG:
            layout_debug(
                f"group-layout group={group.group_id} "
                f"layers={len(set(layout.layer_names))} segments={len(layout.layer_names)} "
                f"state={int(group.is_state_snapshot)} "
                f"block-first={int(layout.block_first is not None)} "
                f"block_size_bytes={whole_block_bytes} token_block={token_block_size} "
                f"tail_blocks={group.tail_blocks} span={partial_tokens} "
                f"store_bytes={store_bytes}"
            )
        return cls(
            vllm_blocks_per_ucm_block=group.tail_blocks,
            access=access,
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
        num_ucm_blocks: int,
        *,
        token_offsets: np.ndarray | None = None,
        segment_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return [UCM block, segment] arrays; invariant sizes/offsets are broadcast.

        Broadcast backing is copied once per call, so even mutation of an
        array's base cannot corrupt a compiled template or a retained batch.
        """
        layout = self.access.layout
        if self.merge_whole_blocks and segment_mask is None:
            segment_ptrs, _ = layout.block_first_segments(vllm_block_ids)
            store_offsets = self.ucm_block_offsets.reshape(-1).copy()
            segment_sizes = np.full(
                self.vllm_blocks_per_ucm_block, self.whole_block_bytes, dtype=np.uint64
            )
        else:
            segment_ptrs = self.access.resolve_ptrs(
                vllm_block_ids, token_offsets=token_offsets, segment_mask=segment_mask
            )
            segment_columns: slice | np.ndarray = (
                slice(None) if segment_mask is None else segment_mask
            )
            if self.merge_whole_blocks:
                # Layerwise IO keeps each segment at its original page position.
                segment_offsets = layout.base_ptrs[segment_columns] - np.uint64(
                    layout.block_first.base_ptr
                )
                store_offsets = (
                    self.ucm_block_offsets + segment_offsets[None, :]
                ).reshape(-1)
            else:
                store_offsets = (
                    self.ucm_block_offsets[:, segment_columns].reshape(-1).copy()
                )
            segment_sizes = (
                self.access.segment_bytes[:, segment_columns].reshape(-1).copy()
            )
        matrix_shape = (num_ucm_blocks, len(segment_sizes))
        return (
            np.broadcast_to(store_offsets, matrix_shape),
            segment_ptrs.reshape(matrix_shape),
            np.broadcast_to(segment_sizes, matrix_shape),
        )
