"""Scheduler-side lookup: how many tokens can be loaded from the Store."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .layout import UCMKVCacheSpec
from .ucm_chunk_id import AttentionType, ShardKey, make_local_ucm_chunk_ids
from .ucm_proxy import UCMProxyAdapter
from .ucm_sharding import UCMSharding, groups_by_attention_type


class UCMLookup:
    """Count how many tokens of a request, after its HBM hit, can be loaded
    from the Store instead of recomputed.

    A chunk of an attention type hits only when all shards of all its record
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
        # All (record class, shard) pairs a chunk of each attention type needs.
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
        common_ucm_chunk_ids: Sequence[bytes],
        num_tokens: int,
        hbm_hit_tokens: int,
    ) -> int:
        """Tokens that can be loaded after hbm_hit_tokens.

        FA must hit as a continuous prefix. WA and State only store the
        restore boundary, so they must hit at a boundary the FA prefix
        reaches. Without FA, any full chunk can be the restore boundary.
        """
        tokens_per_chunk = self.spec.tokens_per_chunk
        # Only scan full chunks before the last recompute_tokens tokens; those
        # tokens are always recomputed.
        last_chunk = max(num_tokens - self.recompute_tokens, 0) // tokens_per_chunk
        first_chunk = hbm_hit_tokens // tokens_per_chunk
        if AttentionType.FA in self._attention_types:
            fa_hit_end_chunk = self._prefix_hit_end_chunk(
                AttentionType.FA, common_ucm_chunk_ids, first_chunk, last_chunk
            )
            restore_end_token = max(fa_hit_end_chunk * tokens_per_chunk, hbm_hit_tokens)
        else:
            restore_end_token = max(last_chunk * tokens_per_chunk, hbm_hit_tokens)
        boundary_types = [
            attention_type
            for attention_type in self._attention_types
            if attention_type is not AttentionType.FA
        ]
        if boundary_types:
            boundary_hit_end_chunk = self._common_boundary_hit_end_chunk(
                boundary_types,
                common_ucm_chunk_ids,
                first_chunk,
                restore_end_token // tokens_per_chunk,
            )
            restore_end_token = max(
                boundary_hit_end_chunk * tokens_per_chunk, hbm_hit_tokens
            )
        return max(restore_end_token - hbm_hit_tokens, 0)

    def _local_ucm_chunk_ids(
        self,
        attention_type: AttentionType,
        record_shard: tuple[int, ShardKey],
        common_ucm_chunk_ids: Sequence[bytes],
    ) -> tuple[bytes, ...]:
        class_index, shard_key = record_shard
        return make_local_ucm_chunk_ids(
            common_ucm_chunk_ids, attention_type, class_index, shard_key
        )

    def _prefix_hit_end_chunk(
        self,
        attention_type: AttentionType,
        common_ucm_chunk_ids: Sequence[bytes],
        first_chunk: int,
        end_chunk: int,
    ) -> int:
        """End (exclusive) of the chunks from first_chunk that all hit, or
        first_chunk when the first chunk misses.

        A chunk hits only when every record shard has it, so the hit prefix is
        the shortest prefix of all shards. The Store's prefix lookup stops at
        the first miss, and each later shard only scans up to the shortest
        prefix so far.
        """
        hit_end_chunk = max(end_chunk, first_chunk)
        # _record_shards contains all unique kv shards on all ranks
        # here we guarantee it hits on all ranks
        for record_shard in self._record_shards[attention_type]:
            if hit_end_chunk > first_chunk:
                # The Store returns the offset of the last hit, -1 for none.
                last_hit_offset = self.proxy.lookup_on_prefix(
                    self._local_ucm_chunk_ids(
                        attention_type,
                        record_shard,
                        common_ucm_chunk_ids[first_chunk:hit_end_chunk],
                    ),
                    np.arange(first_chunk, hit_end_chunk, dtype=np.int64),
                )
                hit_end_chunk = first_chunk + last_hit_offset + 1
            else:
                break
        return hit_end_chunk

    def _common_boundary_hit_end_chunk(
        self,
        boundary_types: Sequence[AttentionType],
        common_ucm_chunk_ids: Sequence[bytes],
        first_chunk: int,
        end_chunk: int,
    ) -> int:
        """Find the last chunk that every shard has, and return the chunk
        after it. Return first_chunk if there is no such chunk.

        Check each shard from the end to find its last hit, and keep the
        earliest one. If the shards do not agree, check again below it.
        Stop when all shards agree.
        """
        record_shards = [
            (attention_type, record_shard)
            for attention_type in boundary_types
            for record_shard in self._record_shards[attention_type]
        ]
        hit_end_chunk = max(end_chunk, first_chunk)
        is_common_hit = False
        while hit_end_chunk > first_chunk and not is_common_hit:
            pass_end_chunk = hit_end_chunk
            for attention_type, record_shard in record_shards:
                if hit_end_chunk > first_chunk:
                    # The Store returns the offset of the latest hit, -1 for
                    # none.
                    latest_hit_offset = self.proxy.lookup_on_reverse(
                        self._local_ucm_chunk_ids(
                            attention_type,
                            record_shard,
                            common_ucm_chunk_ids[first_chunk:hit_end_chunk],
                        ),
                        np.arange(first_chunk, hit_end_chunk, dtype=np.int64),
                    )
                    hit_end_chunk = first_chunk + latest_hit_offset + 1
                else:
                    break
            is_common_hit = hit_end_chunk == pass_end_chunk
        return hit_end_chunk
