"""Cache capacity, domain geometry and worker-local NUMA hints for KV and EC."""

from typing import Any

from ucm.logger import init_logger

logger = init_logger(__name__)


def configure_cache_capacity(
    config: dict[str, Any], parallel: Any, stores_per_dp: int = 1
) -> None:
    """Split the node budget across local DP domains and their Cache stores.

    Pure external LB exposes one DP per service, not the physical node's DP
    count. Keep its per-DP capacity and ask the deployer to divide the budget.
    """
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
    divisor = local_dp_size * stores_per_dp
    per_store = capacity // divisor
    if per_store < 1:
        raise ValueError(
            f"cache_buffer_capacity_gb={capacity} is too small for "
            f"local_dp_size={local_dp_size}, stores_per_dp={stores_per_dp}, "
            f"set at least {divisor}GB."
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
        "per_store=%sGB, external_lb=%s.",
        capacity,
        local_dp_size,
        stores_per_dp,
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


def _fallback_numa_geometry(parallel: Any, local_rank: int) -> tuple[int, int]:
    local_dp_rank = getattr(parallel, "data_parallel_rank_local", None)
    local_dp_size = int(getattr(parallel, "data_parallel_size_local", 0))
    if local_dp_rank is not None and local_dp_size > 0:
        local_workers = cache_rank_count(parallel)
        dp_offset = int(local_dp_rank) * local_workers
        count = local_dp_size * local_workers
        # CUDA multiprocessing can pass a rank that already includes the DP offset.
        if dp_offset <= local_rank < dp_offset + local_workers:
            rank = local_rank
        elif 0 <= local_rank < local_workers:
            rank = dp_offset + local_rank
        else:
            raise ValueError(
                f"Invalid worker local rank for NUMA fallback: local_rank={local_rank}, "
                f"local_dp_rank={local_dp_rank}, workers_per_dp={local_workers}."
            )
    else:
        model_size = parallel.pipeline_parallel_size * parallel.tensor_parallel_size
        if model_size <= 0:
            raise ValueError(
                f"Invalid fallback NUMA model parallel size: {model_size}."
            )
        rank = (
            int(parallel.data_parallel_index) * model_size + parallel.rank % model_size
        )
        count = int(parallel.world_size)
    if not 0 <= rank < count:
        raise ValueError(
            f"Invalid fallback NUMA worker geometry: rank={rank}, count={count}."
        )
    return rank, count


def configure_cache_numa(
    config: dict[str, Any], device: Any, device_id: int, parallel: Any, local_rank: int
) -> None:
    """Pass device affinity and the geometry needed if that node is unavailable."""
    node = device.get_numa_node(device_id)
    if node is not None:
        config["cache_detected_numa_node"] = node
    try:
        rank, count = _fallback_numa_geometry(parallel, local_rank)
    except (AttributeError, TypeError, ValueError) as error:
        # Without node-wide hints, CacheStore can rank-slice within its own domain.
        config.pop("cache_fallback_numa_rank", None)
        config.pop("cache_fallback_numa_rank_count", None)
        logger.warning(
            "NUMA worker geometry unavailable for device %s, "
            "using Cache-domain rank when needed: %s",
            device_id,
            error,
        )
        return
    config["cache_fallback_numa_rank"] = rank
    config["cache_fallback_numa_rank_count"] = count
