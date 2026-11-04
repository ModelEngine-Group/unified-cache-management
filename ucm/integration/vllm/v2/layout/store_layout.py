"""Compile complete distributed blocks into a ready-to-use group store layout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from ..runtime import ParallelRank, ParallelTopology
    from .group import GroupLayoutMetadata, KVCacheGroupLayout
    from .kv_cache import UCMKVCacheGroupSpec, UCMKVCacheSpec


PPStageMetadata: TypeAlias = tuple[int, Mapping[int, "GroupLayoutMetadata"], str | None]


class UCMShard(NamedTuple):
    pp_rank: int
    tp_shard: int
    dcp_shard: int


def shard_replicas(
    topology: ParallelTopology,
    pp_rank: int,
    tp_shards: int,
    dcp_sharded: bool,
) -> dict[UCMShard, tuple[ParallelRank, ...]]:
    """Enumerate effective pairs, never the TP x DCP Cartesian product."""
    from ..runtime import ParallelRank

    owners: dict[UCMShard, list[ParallelRank]] = {}
    for pcp_rank in range(topology.pcp_size):
        for tp_rank in range(topology.tp_size):
            dcp_shard = 0
            if dcp_sharded and topology.dcp_size > 1:
                if topology.pcp_size == 1:
                    dcp_shard = tp_rank % topology.dcp_size
                elif topology.dcp_size == topology.pcp_size:
                    dcp_shard = pcp_rank
                else:
                    dcp_shard = tp_rank * topology.pcp_size + pcp_rank
            shard = UCMShard(
                pp_rank, tp_rank * tp_shards // topology.tp_size, dcp_shard
            )
            owners.setdefault(shard, []).append(
                ParallelRank(tp_rank, pp_rank, pcp_rank)
            )
    return {shard: tuple(ranks) for shard, ranks in sorted(owners.items())}


@dataclass(frozen=True)
class GroupStoreLayout:
    """Final local IO sizes and offsets within a complete distributed chunk."""

    spec: UCMKVCacheSpec
    group_id: int
    group_layout: KVCacheGroupLayout | None
    tp_shards: int
    shard: UCMShard | None
    replica_index: int
    replica_count: int
    # Matrices: [blocks in this chunk/window, local segments].
    # HBM offsets are relative to each segment's local block base.
    segment_byte_offsets_in_block: np.ndarray
    chunk_segment_bytes: np.ndarray
    # Store offsets are relative to this native group's payload in the object.
    chunk_segment_offsets: np.ndarray
    # Local segments' positions in one complete distributed block (1-D).
    block_segment_offsets: np.ndarray
    whole_block_nbytes: int
    whole_chunk_nbytes: int

    @property
    def group(self) -> UCMKVCacheGroupSpec:
        return self.spec.groups[self.group_id]

    @classmethod
    def build(
        cls,
        kv_cache_spec: UCMKVCacheSpec,
        group_id: int,
        group_layout: KVCacheGroupLayout | None,
        pp_metadata: Sequence[PPStageMetadata],
        parallel_topology: ParallelTopology,
        parallel_rank: ParallelRank,
    ) -> GroupStoreLayout:
        """Assemble a full block, expand to a chunk, then create the final layout."""
        group_spec = kv_cache_spec.groups[group_id]
        pp_group_metadata = [
            (pp_rank, group_metadata_by_id[group_id])
            for pp_rank, group_metadata_by_id, _ in pp_metadata
        ]
        tp_shards = lcm(
            *(
                group_metadata.tp_shards
                for _, group_metadata in pp_group_metadata
                if group_metadata.sizes
            )
        )
        global_block_sizes, global_states_per_block = [], []
        local_segment_start, local_segment_count, local_shard = 0, 0, None
        replica_index, replica_count = 0, 1
        for pp_rank, group_metadata in pp_group_metadata:
            if not group_metadata.sizes:
                continue
            for ucm_shard, replica_ranks in shard_replicas(
                parallel_topology, pp_rank, tp_shards, group_spec.dcp_sharded
            ).items():
                if parallel_rank in replica_ranks:
                    local_segment_start, local_segment_count = (
                        len(global_block_sizes),
                        len(group_metadata.sizes),
                    )
                    local_shard = ucm_shard
                    replica_index, replica_count = (
                        replica_ranks.index(parallel_rank),
                        len(replica_ranks),
                    )
                global_block_sizes.extend(group_metadata.sizes)
                global_states_per_block.extend(group_metadata.states_per_block)
        if not global_block_sizes and group_spec.restore_policy is not None:
            raise ValueError(f"No worker stores persistent group {group_id}")

        block_token_span = group_spec.block_token_span
        blocks_per_chunk = max(kv_cache_spec.blocks_per_chunk[group_id], 1)
        block_token_counts = np.full(
            blocks_per_chunk, block_token_span, dtype=np.uint64
        )
        block_token_offsets = np.zeros(blocks_per_chunk, dtype=np.uint64)
        if group_spec.is_sliding_window:
            tail_tokens = group_spec.tail_tokens or 0
            if tail_tokens == 0:
                block_token_counts[:] = 0
            elif tail_tokens % block_token_span:
                block_token_counts[0] = tail_tokens % block_token_span
                block_token_offsets[0] = block_token_span - int(block_token_counts[0])
        elif (
            not group_spec.is_state_snapshot
            and kv_cache_spec.chunk_size < block_token_span
        ):
            block_token_counts[0] = kv_cache_spec.chunk_size
        block_segment_bytes = np.asarray(global_block_sizes, dtype=np.uint64)
        block_segment_states = np.asarray(global_states_per_block, dtype=np.uint64)
        for token_count in block_token_counts:
            if token_count and token_count != block_token_span:
                if not all(
                    group_metadata.supports_partial_tokens
                    for _, group_metadata in pp_group_metadata
                    if group_metadata.sizes
                ):
                    raise ValueError(
                        f"KV cache group {group_id} requires whole-block access: "
                        f"its KV layout is not token-contiguous for {int(token_count)}-token slices"
                    )
                if (
                    token_count * block_segment_states % np.uint64(block_token_span)
                ).any():
                    raise ValueError(
                        f"Token ranges must align with stored states in group {group_id}"
                    )
        chunk_segment_bytes = (
            block_token_counts[:, None]
            * block_segment_bytes[None, :]
            // np.uint64(block_token_span)
        )
        chunk_segment_offsets = (
            np.cumsum(chunk_segment_bytes).reshape(chunk_segment_bytes.shape)
            - chunk_segment_bytes
        )
        local_segment_slice = slice(
            local_segment_start, local_segment_start + local_segment_count
        )
        block_segment_offsets = np.cumsum(block_segment_bytes) - block_segment_bytes
        return cls(
            spec=kv_cache_spec,
            group_id=group_id,
            group_layout=group_layout,
            tp_shards=tp_shards,
            shard=local_shard,
            replica_index=replica_index,
            replica_count=replica_count,
            segment_byte_offsets_in_block=block_token_offsets[:, None]
            * block_segment_bytes[local_segment_slice][None, :]
            // np.uint64(block_token_span),
            chunk_segment_bytes=chunk_segment_bytes[:, local_segment_slice].copy(),
            chunk_segment_offsets=chunk_segment_offsets[:, local_segment_slice].copy(),
            block_segment_offsets=block_segment_offsets[local_segment_slice].copy(),
            whole_block_nbytes=int(block_segment_bytes.sum()),
            whole_chunk_nbytes=int(chunk_segment_bytes.sum()),
        )

    def writer_rows(self, chunk_indices: np.ndarray) -> np.ndarray:
        """One writer owns every local segment of this group at a chunk index."""
        return np.flatnonzero(chunk_indices % self.replica_count == self.replica_index)

    def resolve_matrices(
        self,
        vllm_block_ids: np.ndarray,
        token_starts: np.ndarray,
        *,
        segment_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Final [chunk,entry] offsets, ptrs, sizes; layout compilation is over."""
        rows = len(token_starts)
        cols = slice(None) if segment_mask is None else segment_mask
        fixed = self.chunk_segment_bytes[:, cols].copy()
        block_count = max(self.spec.blocks_per_chunk[self.group_id], 1)
        ptrs = self.group_layout.resolve_ptrs(
            vllm_block_ids, segment_mask=segment_mask
        ).reshape(rows, block_count, fixed.shape[1])
        ptrs += self.segment_byte_offsets_in_block[:, cols]
        if not self.group.is_sliding_window:
            ptrs += self.group_layout.tokens_to_bytes(
                np.asarray(token_starts, dtype=np.uint64)
                % np.uint64(self.group.block_token_span),
                self.group.block_token_span,
                segment_mask,
            )[:, None, :]
        shape = (rows, fixed.size)
        return (
            np.broadcast_to(
                self.chunk_segment_offsets[:, cols].copy().reshape(-1), shape
            ),
            ptrs.reshape(shape),
            np.broadcast_to(fixed.reshape(-1), shape),
        )
