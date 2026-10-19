"""Scheduler-side lookup: how many tokens can be loaded from the Store."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .layout import UCMKVCacheSpec
from .ucm_block_id import AttentionType, ShardKey, make_local_ucm_block_ids
from .ucm_proxy import UCMProxyAdapter
from .ucm_sharding import UCMSharding, groups_by_attention_type


class UCMLookup:
    """Count how many tokens of a request, after its HBM hit, can be loaded
    from the Store instead of recomputed.

    A block of an attention type hits only when all shards of all its record
    classes are stored. A pipeline stage without groups of a record class
    stores an empty record, so lookup does not need to know which layers each
    rank has. WA dumps skip incomplete windows, so lookup never picks a
    partial record.
    """

    def __init__(
        self,
        spec: UCMKVCacheSpec,
        proxy: UCMProxyAdapter,
        sharding: UCMSharding,
        *,
        recompute_tokens: int = 1,
    ) -> None:
        self.spec = spec
        self.proxy = proxy
        self.recompute_tokens = max(int(recompute_tokens), 0)
        self._attention_types = tuple(
            attention_type for attention_type, _ in groups_by_attention_type(spec)
        )
        # All (record class, shard) pairs a block of each attention type needs.
        self._record_shards: dict[AttentionType, tuple[tuple[int, ShardKey], ...]] = {
            attention_type: tuple(
                (record_class.class_index, shard_key)
                for record_class in sharding.record_classes(attention_type)
                for shard_key in sharding.all_shard_keys(record_class)
            )
            for attention_type in self._attention_types
        }

    def hit_tokens(
        self,
        common_ucm_block_ids: Sequence[bytes],
        num_tokens: int,
        hbm_hit_tokens: int,
    ) -> int:
        """Tokens that can be loaded after hbm_hit_tokens.

        FA must hit as a continuous prefix. WA and State only store the
        restore boundary, so they must hit at a boundary the FA prefix
        reaches. Without FA, any full block can be the restore boundary.
        """
        ucm_block_size = self.spec.ucm_cache_block_size
        # Only scan full blocks before the last recompute_tokens tokens; those
        # tokens are always recomputed.
        last_block = max(num_tokens - self.recompute_tokens, 0) // ucm_block_size
        first_block = hbm_hit_tokens // ucm_block_size
        if AttentionType.FA in self._attention_types:
            fa_hit_end_block = self._prefix_hit_end_block(
                AttentionType.FA, common_ucm_block_ids, first_block, last_block
            )
            restore_end_token = max(fa_hit_end_block * ucm_block_size, hbm_hit_tokens)
        else:
            restore_end_token = max(last_block * ucm_block_size, hbm_hit_tokens)
        boundary_types = [
            attention_type
            for attention_type in self._attention_types
            if attention_type is not AttentionType.FA
        ]
        if boundary_types:
            boundary_hit_end_block = self._common_boundary_hit_end_block(
                boundary_types,
                common_ucm_block_ids,
                first_block,
                restore_end_token // ucm_block_size,
            )
            restore_end_token = max(
                boundary_hit_end_block * ucm_block_size, hbm_hit_tokens
            )
        return max(restore_end_token - hbm_hit_tokens, 0)

    def _local_ucm_block_ids(
        self,
        attention_type: AttentionType,
        record_shard: tuple[int, ShardKey],
        common_ucm_block_ids: Sequence[bytes],
    ) -> tuple[bytes, ...]:
        class_index, shard_key = record_shard
        return make_local_ucm_block_ids(
            common_ucm_block_ids, attention_type, class_index, shard_key
        )

    def _prefix_hit_end_block(
        self,
        attention_type: AttentionType,
        common_ucm_block_ids: Sequence[bytes],
        first_block: int,
        end_block: int,
    ) -> int:
        """End (exclusive) of the blocks from first_block that all hit, or
        first_block when the first block misses.

        A block hits only when every record shard has it, so the hit prefix is
        the shortest prefix of all shards. The Store's prefix lookup stops at
        the first miss, and each later shard only scans up to the shortest
        prefix so far.
        """
        hit_end_block = max(end_block, first_block)
        # _record_shards contains all unique kv shards on all ranks
        # here we guarantee it hits on all ranks
        for record_shard in self._record_shards[attention_type]:
            if hit_end_block > first_block:
                # The Store returns the offset of the last hit, -1 for none.
                last_hit_offset = self.proxy.lookup_on_prefix(
                    self._local_ucm_block_ids(
                        attention_type,
                        record_shard,
                        common_ucm_block_ids[first_block:hit_end_block],
                    ),
                    np.arange(first_block, hit_end_block, dtype=np.int64),
                )
                hit_end_block = first_block + last_hit_offset + 1
            else:
                break
        return hit_end_block

    def _common_boundary_hit_end_block(
        self,
        boundary_types: Sequence[AttentionType],
        common_ucm_block_ids: Sequence[bytes],
        first_block: int,
        end_block: int,
    ) -> int:
        """Find the last block that every shard has, and return the block
        after it. Return first_block if there is no such block.

        Check each shard from the end to find its last hit, and keep the
        earliest one. If the shards do not agree, check again below it.
        Stop when all shards agree.
        """
        record_shards = [
            (attention_type, record_shard)
            for attention_type in boundary_types
            for record_shard in self._record_shards[attention_type]
        ]
        hit_end_block = max(end_block, first_block)
        is_common_hit = False
        while hit_end_block > first_block and not is_common_hit:
            pass_end_block = hit_end_block
            for attention_type, record_shard in record_shards:
                if hit_end_block > first_block:
                    # The Store returns the offset of the latest hit, -1 for
                    # none.
                    latest_hit_offset = self.proxy.lookup_on_reverse(
                        self._local_ucm_block_ids(
                            attention_type,
                            record_shard,
                            common_ucm_block_ids[first_block:hit_end_block],
                        ),
                        np.arange(first_block, hit_end_block, dtype=np.int64),
                    )
                    hit_end_block = first_block + latest_hit_offset + 1
                else:
                    break
            is_common_hit = hit_end_block == pass_end_block
        return hit_end_block
