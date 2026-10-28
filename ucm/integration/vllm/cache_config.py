"""Cache capacity, domain geometry and worker-local NUMA hints for KV and EC."""

from typing import Any

from ucm.logger import init_logger
from ucm.store.cache.capabilities import cache_core_version, uses_partitioned_cache

logger = init_logger(__name__)


def configure_cache_capacity(
    config: dict[str, Any], parallel: Any, stores_per_dp: int = 1
) -> None:
    """Split the node budget across local DP domains and their Cache stores.

    Pure external LB exposes one DP per service, not the physical node's DP
    count. Keep its per-DP capacity and ask the deployer to divide the budget.
    """
    partitioned = uses_partitioned_cache()
    if not partitioned and cache_core_version() != 1:
        return
    capacity = config.get("cache_buffer_capacity_gb")
    if capacity is None:
        capacity = 160
    capacity = int(capacity)
    # Zero selects the C++ Cache default; never pass a rounded-down zero back.
    if capacity == 0:
        capacity = 160
    external_lb = bool(getattr(parallel, "data_parallel_external_lb", False))
    local_dp_size = (
        1 if external_lb else int(getattr(parallel, "data_parallel_size_local", 1))
    )
    if local_dp_size <= 0 or stores_per_dp <= 0:
        raise ValueError(
            f"Invalid Cache capacity partition: local_dp_size={local_dp_size}, "
            f"stores_per_dp={stores_per_dp}."
        )
    private_ranks = (
        cache_rank_count(parallel)
        if not partitioned and not config.get("share_buffer_enable", False)
        else 1
    )
    divisor = local_dp_size * stores_per_dp * private_ranks
    per_store = capacity // divisor
    if per_store < 1:
        raise ValueError(
            f"cache_buffer_capacity_gb={capacity} is too small for "
            f"local_dp_size={local_dp_size}, stores_per_dp={stores_per_dp}, "
            f"private_ranks={private_ranks}; set at least {divisor}GB."
        )
    if external_lb:
        logger.warning_once(
            "External DP load balancing: UCM cannot determine the physical "
            "node's DP count. cache_buffer_capacity_gb applies to each DP "
            "instance independently (FA/WA stores split it internally). "
            "If multiple DP instances share a node, manually set "
            "cache_buffer_capacity_gb = target node Cache capacity / actual "
            "local DP count. Otherwise total Cache allocations may exceed "
            "the node budget and initialization may fail."
        )
    config["cache_buffer_capacity_gb"] = per_store
    logger.info(
        "Cache capacity: configured=%sGB, local_dp_size=%s, stores_per_dp=%s, "
        "private_ranks=%s, per_store=%sGB, external_lb=%s.",
        capacity,
        local_dp_size,
        stores_per_dp,
        private_ranks,
        per_store,
        external_lb,
    )


def cache_rank_count(parallel: Any) -> int:
    """Count workers on this machine in one DP-isolated cache domain."""
    world = int(getattr(parallel, "world_size", parallel.tensor_parallel_size))
    local = int(getattr(parallel, "local_world_size", world))
    count = min(world, local)
    if count <= 0:
        raise ValueError(f"Invalid Cache worker count: world={world}, local={local}.")
    return count


def configure_cache_numa(
    config: dict[str, Any], device: Any, device_id: int, parallel: Any, local_rank: int
) -> None:
    """Pass real affinity, or the worker's position among all local DP workers.

    The C++ store intersects online memory nodes with the process's allowed
    nodes. Do not synthesize affinity from CPU numbers or use the Cache-domain
    rank as the fallback rank when local DP metadata is available.
    """
    if not uses_partitioned_cache():
        return
    node = device.get_numa_node(device_id)
    if node is not None:
        config["cache_detected_numa_node"] = node
        return
    local_dp_rank = getattr(parallel, "data_parallel_rank_local", None)
    local_dp_size = int(getattr(parallel, "data_parallel_size_local", 0))
    if local_dp_rank is not None and local_dp_size > 0:
        local_workers = cache_rank_count(parallel)
        rank = int(local_dp_rank) * local_workers + local_rank
        count = local_dp_size * local_workers
    else:
        model_size = parallel.pipeline_parallel_size * parallel.tensor_parallel_size
        rank = (
            int(parallel.data_parallel_index) * model_size + parallel.rank % model_size
        )
        count = int(parallel.world_size)
    if not 0 <= rank < count:
        raise ValueError(
            f"Invalid fallback NUMA worker geometry: rank={rank}, count={count}."
        )
    config["cache_fallback_numa_rank"] = rank
    config["cache_fallback_numa_rank_count"] = count
