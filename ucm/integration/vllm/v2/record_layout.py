"""Compose physical block accesses into a group's part of a UCM record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .layout.group import BlockAccess, KVCacheGroupLayout
from .layout.view import LAYOUT_DEBUG, layout_debug

if TYPE_CHECKING:
    from .ucm_kv_cache import UCMKVCacheGroupInfo


@dataclass(frozen=True)
class GroupRecordLayout:
    """One group's record template; offsets are relative to the group start.

    One FA key identifies one UCM block (only its covered blocks):

        UCM block
        +-- group 0 (dispatch_routes order)
        |   +-- window block 0 (oldest)
        |   |   +-- layer/component/segment payloads
        |   +-- window block 1
        |       +-- layer/component/segment payloads
        +-- group 1
            +-- window block 0
                +-- layer/component/segment payloads

    ``ucm_block_offsets`` holds this group's local positions; the caller adds
    the group base to produce offsets within the complete UCM block.

    Whole Block First slots include their existing padding. Selecting layers
    preserves record offsets; it does not repack the record.
    """

    blocks_per_key: int
    access: BlockAccess
    ucm_block_offsets: np.ndarray
    record_bytes: int
    group_block_offsets: np.ndarray
    whole_block_bytes: int
    merge_whole_blocks: bool
    dynamic_token_offsets: bool

    @classmethod
    def build(
        cls,
        group: UCMKVCacheGroupInfo,
        layout: KVCacheGroupLayout,
        ucm_block_size: int,
    ) -> GroupRecordLayout:
        block_tokens = layout.token_block_size
        if group.is_sliding_window:
            partial_tokens = (group.tail_tokens or 0) % block_tokens
        elif group.is_state_snapshot:
            partial_tokens = 0
        else:
            partial_tokens = ucm_block_size % block_tokens

        # Zero-tail groups join no route, but still have a valid layout.
        rows = max(group.tail_blocks, 1)
        token_offsets = np.zeros(rows, dtype=np.uint64)
        token_counts = np.full(rows, block_tokens, dtype=np.uint64)
        if partial_tokens:
            token_counts[0] = partial_tokens
            if group.is_sliding_window:
                token_offsets[0] = block_tokens - partial_tokens
        access = layout.compile_access(
            token_offsets=token_offsets, token_counts=token_counts
        )

        # Whole-block destination placement belongs here, not in the physical
        # address resolver. Retain the established padded Block First format.
        if layout.block_first is not None:
            slots: list[int] = []
            for layer in sorted(
                group.layers, key=lambda item: (item.layer_index, item.layer_name)
            ):
                descriptor = layer.descriptor
                assert descriptor is not None
                anchor = (
                    descriptor.offset
                    + layer.descriptor_position * descriptor.layer_stride
                )
                segments = layout.layer_views[layer.layer_name].segments
                slots.extend(
                    anchor + segment.base_ptr - segments[0].base_ptr
                    for segment in segments
                )
            group_block_offsets = np.asarray(slots, dtype=np.uint64)
            whole_block_bytes = layout.block_first.block_size_bytes
        else:
            group_block_offsets = np.cumsum(layout.payload_bytes) - layout.payload_bytes
            whole_block_bytes = int(layout.payload_bytes.sum())

        ucm_block_offsets = np.empty_like(access.segment_bytes)
        cursor = 0
        for row in range(rows):
            if row == 0 and partial_tokens:
                sizes = access.segment_bytes[row]
                ucm_block_offsets[row] = np.cumsum(sizes) - sizes
                cursor += int(sizes.sum())
            else:
                ucm_block_offsets[row] = cursor + group_block_offsets
                cursor += whole_block_bytes
        if LAYOUT_DEBUG:
            layout_debug(
                f"group-layout group={group.group_id} "
                f"layers={len(set(layout.layer_names))} segments={len(layout.layer_names)} "
                f"state={int(group.is_state_snapshot)} "
                f"block-first={int(layout.block_first is not None)} "
                f"block_size_bytes={whole_block_bytes} token_block={block_tokens} "
                f"tail_blocks={group.tail_blocks} span={partial_tokens} "
                f"record_bytes={cursor}"
            )
        return cls(
            blocks_per_key=group.tail_blocks,
            access=access,
            ucm_block_offsets=ucm_block_offsets,
            record_bytes=cursor,
            group_block_offsets=group_block_offsets,
            whole_block_bytes=whole_block_bytes,
            # Merging is a transfer decision: source span and destination
            # positions must describe exactly the same byte arrangement.
            merge_whole_blocks=(
                partial_tokens == 0
                and layout.block_first is not None
                and np.array_equal(
                    group_block_offsets,
                    layout.base_ptrs - np.uint64(layout.block_first.base_ptr),
                )
            ),
            dynamic_token_offsets=bool(partial_tokens) and not group.is_sliding_window,
        )

    def resolve(
        self,
        blocks: np.ndarray,
        key_count: int,
        *,
        token_offsets: np.ndarray | None = None,
        segment_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Return group-relative record offsets, pointers, sizes, entries/key."""
        layout = self.access.layout
        if (
            layout.block_first is not None
            and segment_mask is None
            and self.merge_whole_blocks
        ):
            ptrs, sizes = layout.block_first_segments(blocks)
            offsets = (
                np.arange(len(blocks), dtype=np.uint64) % self.blocks_per_key
            ) * self.whole_block_bytes
            return offsets, ptrs, sizes, self.blocks_per_key

        ptrs, sizes = self.access.resolve(
            blocks, token_offsets=token_offsets, segment_mask=segment_mask
        )
        columns: slice | np.ndarray = (
            slice(None) if segment_mask is None else segment_mask
        )
        offsets = np.tile(self.ucm_block_offsets[:, columns], (key_count, 1))
        return (
            offsets.reshape(-1),
            ptrs.reshape(-1),
            sizes.reshape(-1),
            self.blocks_per_key * layout.segment_count(segment_mask),
        )

    def resolve_matrices(
        self,
        blocks: np.ndarray,
        key_count: int,
        *,
        token_offsets: np.ndarray | None = None,
        segment_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return [key, segment] arrays; invariant sizes/offsets are broadcast.

        Broadcast backing is copied once per call, so even mutation of an
        array's base cannot corrupt a compiled template or a retained batch.
        """
        layout = self.access.layout
        if self.merge_whole_blocks and segment_mask is None:
            ptrs, _ = layout.block_first_segments(blocks)
            offsets = (
                np.arange(self.blocks_per_key, dtype=np.uint64) * self.whole_block_bytes
            )
            sizes = np.full(
                self.blocks_per_key, self.whole_block_bytes, dtype=np.uint64
            )
        else:
            ptrs = self.access.resolve_ptrs(
                blocks, token_offsets=token_offsets, segment_mask=segment_mask
            )
            columns: slice | np.ndarray = (
                slice(None) if segment_mask is None else segment_mask
            )
            offsets = self.ucm_block_offsets[:, columns].reshape(-1).copy()
            sizes = self.access.segment_bytes[:, columns].reshape(-1).copy()
        shape = (key_count, len(sizes))
        return (
            np.broadcast_to(offsets, shape),
            ptrs.reshape(shape),
            np.broadcast_to(sizes, shape),
        )
