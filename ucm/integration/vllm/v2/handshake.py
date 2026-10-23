"""Worker-to-scheduler handshake for the group sharding.

vLLM gives the scheduler only one spec of each uniform group, while workers
see every layer's spec. Workers send the sharding they use, and the
scheduler takes it so lookup checks the shards workers really write.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)

from ucm.logger import init_logger

from .layout import UCMKVCacheSpec

logger = init_logger(__name__)

# (group_id, tp_shards, dcp_sharded) of each stored group.
GroupSharding = tuple[tuple[int, int | None, bool], ...]


@dataclass(frozen=True)
class UCMShardingHandshake(KVConnectorHandshakeMetadata):
    group_sharding: GroupSharding


def group_sharding(spec: UCMKVCacheSpec) -> GroupSharding:
    return tuple(
        (group.group_id, group.tp_shards, group.dcp_sharded)
        for group in spec.groups
        if group.is_persistent
    )


def apply_worker_sharding(
    spec: UCMKVCacheSpec, worker_metadata: Iterable[object]
) -> UCMKVCacheSpec:
    """Return the scheduler spec with the workers' group sharding.

    All workers must send the same sharding. Returns the same spec object
    when nothing changes.
    """
    declarations = {
        metadata.group_sharding
        for metadata in worker_metadata
        if isinstance(metadata, UCMShardingHandshake)
    }
    if len(declarations) == 0:
        logger.warning(
            "No UCM sharding handshake from workers; the scheduler uses the "
            "sharding of its own representative specs."
        )
        return spec
    elif len(declarations) > 1:
        raise ValueError(
            f"Workers use different UCM group sharding: {list(declarations)}"
        )
    else:
        (worker_sharding,) = declarations
        scheduler_sharding = group_sharding(spec)
        if [item[0] for item in worker_sharding] != [
            item[0] for item in scheduler_sharding
        ]:
            raise ValueError(
                f"Workers store groups {[item[0] for item in worker_sharding]}, "
                f"but the scheduler stores {[item[0] for item in scheduler_sharding]}"
            )
        groups = list(spec.groups)
        for group_id, tp_shards, dcp_sharded in worker_sharding:
            group = groups[group_id]
            # Context parallel has already scaled block sizes by dcp_sharded,
            # and vLLM keeps dcp_sharded the same in a uniform group.
            if dcp_sharded != group.dcp_sharded:
                raise ValueError(
                    f"Group {group_id}: workers have dcp_sharded={dcp_sharded}, "
                    f"but the scheduler has {group.dcp_sharded}"
                )
            elif tp_shards != group.tp_shards:
                logger.info(
                    f"Group {group_id}: using the workers' tp_shards={tp_shards} "
                    f"instead of the scheduler's {group.tp_shards}"
                )
                groups[group_id] = replace(group, tp_shards=tp_shards)
        if groups == list(spec.groups):
            return spec
        else:
            return replace(spec, groups=tuple(groups))
