"""Cache-domain geometry and worker-local NUMA hints shared by KV and EC."""

from typing import Any

from ucm.store.cache.capabilities import uses_partitioned_cache


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
