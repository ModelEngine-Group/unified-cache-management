"""Decide which ranks hold the same KV data and how records are sharded.

A record class is the groups of one attention type that are sharded the
same way across TP and DCP. Each record class is stored as its own record.
Ranks with the same ShardKey hold the same data: they read the same record
and share the work of writing it. Scheduler and workers get the same result
from the same config.
"""

from __future__ import annotations

from dataclasses import dataclass

from .layout import UCMKVCacheGroupInfo, UCMKVCacheSpec
from .parallel import ParallelLayout
from .ucm_block_id import AttentionType, ShardKey, validate_shard_key_encoding

# (tp_rank, pp_rank, pcp_rank) of one worker.
ParallelRank = tuple[int, int, int]


def groups_by_attention_type(
    spec: UCMKVCacheSpec,
) -> tuple[tuple[AttentionType, tuple[UCMKVCacheGroupInfo, ...]], ...]:
    """Attention types in dispatch order, each with the groups it stores.

    FA holds the full-attention groups; WA the sliding groups that
    re-store a window tail (tail 0 groups store nothing); State the
    mamba snapshot groups. Absent attention types are omitted, and
    dispatch plans index groups in this order.
    """

    attention_type_groups: list[
        tuple[AttentionType, tuple[UCMKVCacheGroupInfo, ...]]
    ] = []
    if spec.fa_groups:
        attention_type_groups.append((AttentionType.FA, spec.fa_groups))
    wa_groups_with_tail = tuple(
        group for group in spec.wa_groups if (group.tail_tokens or 0) > 0
    )
    if wa_groups_with_tail:
        attention_type_groups.append((AttentionType.WA, wa_groups_with_tail))
    if spec.state_groups:
        attention_type_groups.append((AttentionType.STATE, spec.state_groups))
    return tuple(attention_type_groups)


@dataclass(frozen=True, eq=False)
class UCMRecordClass:
    """KV groups stored together as one record.

    Compared by identity: UCMSharding creates each record class once.
    """

    attention_type: AttentionType
    class_index: int  # index within the attention type
    groups: tuple[UCMKVCacheGroupInfo, ...]
    # Position of each group in its attention type's group list (plan order).
    group_positions: tuple[int, ...]
    # Number of different head shards across TP. 1 means every TP rank holds
    # the same data.
    num_head_shards: int
    # Tokens are sharded across DCP ranks.
    is_token_sharded: bool


class UCMSharding:
    """Record classes and the replicas of each shard."""

    def __init__(self, spec: UCMKVCacheSpec, parallel_layout: ParallelLayout) -> None:
        self.parallel_layout = parallel_layout
        all_parallel_ranks = tuple(
            (tp_rank, pp_rank, pcp_rank)
            for pp_rank in range(parallel_layout.pp)
            for pcp_rank in range(parallel_layout.pcp)
            for tp_rank in range(parallel_layout.tp)
        )
        self._record_classes: dict[AttentionType, tuple[UCMRecordClass, ...]] = {}
        self._replica_ranks_by_shard: dict[
            UCMRecordClass, dict[ShardKey, list[ParallelRank]]
        ] = {}
        for attention_type, groups in groups_by_attention_type(spec):
            # Group positions of each shard type, in first-seen order.
            group_positions_by_shard_type: dict[tuple[int, bool], list[int]] = {}
            for group_position, group in enumerate(groups):
                group_positions_by_shard_type.setdefault(
                    self._group_shard_type(group), []
                ).append(group_position)
            record_classes = tuple(
                UCMRecordClass(
                    attention_type,
                    class_index,
                    tuple(groups[group_position] for group_position in group_positions),
                    tuple(group_positions),
                    num_head_shards,
                    is_token_sharded,
                )
                for class_index, (
                    (num_head_shards, is_token_sharded),
                    group_positions,
                ) in enumerate(group_positions_by_shard_type.items())
            )
            self._record_classes[attention_type] = record_classes
            for record_class in record_classes:
                replica_ranks_by_shard = self._replica_ranks_by_shard.setdefault(
                    record_class, {}
                )
                for parallel_rank in all_parallel_ranks:
                    shard_key = self.shard_key(record_class, parallel_rank)
                    validate_shard_key_encoding(record_class.class_index, shard_key)
                    replica_ranks_by_shard.setdefault(shard_key, []).append(
                        parallel_rank
                    )

    def _group_shard_type(self, group: UCMKVCacheGroupInfo) -> tuple[int, bool]:
        """How a group is sharded in this topology: (number of head shards,
        whether tokens are sharded by DCP).

        Groups with the same shard type share a record. If the TP sharding is
        unknown or does not divide TP, every TP rank is treated as different,
        which is always safe.
        """
        tp_size = self.parallel_layout.tp
        if group.tp_shards is not None and tp_size % group.tp_shards == 0:
            num_head_shards = group.tp_shards
        else:
            num_head_shards = tp_size
        shard_by_dcp = self.parallel_layout.dcp > 1 and group.dcp_sharded
        return num_head_shards, shard_by_dcp

    def _dcp_rank(self, tp_rank: int, pcp_rank: int) -> int:
        """DCP rank as vLLM builds the groups: inside TP without PCP, otherwise
        across PCP first and then TP."""
        parallel_layout = self.parallel_layout
        if parallel_layout.dcp == 1:
            return 0
        elif parallel_layout.pcp == 1:
            return tp_rank % parallel_layout.dcp
        elif parallel_layout.dcp == parallel_layout.pcp:
            return pcp_rank
        else:
            return tp_rank * parallel_layout.pcp + pcp_rank

    def record_classes(
        self, attention_type: AttentionType
    ) -> tuple[UCMRecordClass, ...]:
        return self._record_classes[attention_type]

    def shard_key(
        self, record_class: UCMRecordClass, parallel_rank: ParallelRank
    ) -> ShardKey:
        """The shard of a record class that this rank holds.

        head_key follows vLLM's KV head replication: consecutive TP ranks share
        one head shard (QKVParallelLinear loads head tp_rank // replicas).
        token_key is the rank's place in its DCP group, built the same way as
        vLLM's DCP groups.
        """
        tp_rank, pp_rank, pcp_rank = parallel_rank
        return ShardKey(
            head_key=tp_rank * record_class.num_head_shards // self.parallel_layout.tp,
            token_key=(
                self._dcp_rank(tp_rank, pcp_rank)
                if record_class.is_token_sharded
                else 0
            ),
            pp_key=pp_rank,
        )

    def all_shard_keys(self, record_class: UCMRecordClass) -> tuple[ShardKey, ...]:
        """All shards of a record class. A block hits only if all are stored."""
        return tuple(self._replica_ranks_by_shard[record_class])

    def replica_index_and_count(
        self, record_class: UCMRecordClass, parallel_rank: ParallelRank
    ) -> tuple[int, int]:
        """(index of this rank, number of replicas) in its shard."""
        replica_ranks = self._replica_ranks_by_shard[record_class][
            self.shard_key(record_class, parallel_rank)
        ]
        return replica_ranks.index(tuple(parallel_rank)), len(replica_ranks)
