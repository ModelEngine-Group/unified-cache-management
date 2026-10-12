"""Process-local bridge between Cache2 NUMA placement and CPU affinity.

Cache2 resolves its fallback NUMA node from the worker-local rank.  CPU
binding happens later in vLLM-Ascend, after the Store has been created, so the
connector publishes the resolved fallback placement here for the binding
patch to consume.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

_ONLINE_PATH = Path("/sys/devices/system/node/online")
_HAS_MEMORY_PATH = Path("/sys/devices/system/node/has_memory")
_PROC_STATUS_PATH = Path("/proc/self/status")
_MEMS_ALLOWED_PREFIX = "Mems_allowed_list:"


@dataclass(frozen=True)
class CacheNumaPlacement:
    node: int
    fallback_rank: int
    available_nodes: tuple[int, ...]


@dataclass(frozen=True)
class UcmCoreBundle:
    source_npu: int
    cores: tuple[int, ...]
    health_cores: tuple[int, ...]


_cache_numa_placement: Optional[CacheNumaPlacement] = None


def _parse_index_list(text: str) -> list[int]:
    """Parse a Linux list such as ``0-3,8`` into sorted unique indices."""
    values: list[int] = []
    seen: set[int] = set()
    for raw_token in text.strip().split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(f"invalid empty index in {text!r}")
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first, last = int(first_text), int(last_text)
            if last < first:
                raise ValueError(f"descending index range {token!r}")
        else:
            first = last = int(token)
        for value in range(first, last + 1):
            if value < 0 or value in seen:
                raise ValueError(f"invalid or duplicated index {value}")
            seen.add(value)
            values.append(value)
    return sorted(values)


def _read_mems_allowed(status_path: Path) -> list[int]:
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(_MEMS_ALLOWED_PREFIX):
            return _parse_index_list(line[len(_MEMS_ALLOWED_PREFIX) :])
    raise ValueError(f"cannot find {_MEMS_ALLOWED_PREFIX} in {status_path}")


def available_cache_numa_nodes(
    online_path: Path = _ONLINE_PATH,
    has_memory_path: Path = _HAS_MEMORY_PATH,
    status_path: Path = _PROC_STATUS_PATH,
) -> list[int]:
    """Mirror Cache2's online/has-memory/mems-allowed NUMA intersection."""
    online = set(_parse_index_list(online_path.read_text(encoding="utf-8")))
    has_memory = set(
        _parse_index_list(has_memory_path.read_text(encoding="utf-8"))
    )
    allowed = set(_read_mems_allowed(status_path))
    nodes = sorted(online & has_memory & allowed)
    if not nodes:
        raise ValueError("no usable Cache NUMA memory node")
    return nodes


def publish_round_robin_cache_numa_placement(
    fallback_rank: int,
) -> CacheNumaPlacement:
    """Publish the exact node selected by Cache2's fallback policy."""
    if fallback_rank < 0:
        raise ValueError(f"fallback rank must be non-negative, got {fallback_rank}")
    nodes = tuple(available_cache_numa_nodes())
    placement = CacheNumaPlacement(
        node=nodes[fallback_rank % len(nodes)],
        fallback_rank=fallback_rank,
        available_nodes=nodes,
    )
    global _cache_numa_placement
    _cache_numa_placement = placement
    return placement


def clear_cache_numa_placement() -> None:
    global _cache_numa_placement
    _cache_numa_placement = None


def get_cache_numa_placement() -> Optional[CacheNumaPlacement]:
    return _cache_numa_placement


def select_numa_local_ucm_bundle(
    placement: CacheNumaPlacement,
    assign_ucm: Mapping[int, Sequence[int]],
    assign_ucm_health: Mapping[int, Sequence[int]],
    cpu_to_node: Mapping[int, int],
) -> Optional[UcmCoreBundle]:
    """Select this fallback rank's unique pre-reserved bundle on its Cache node.

    vLLM-Ascend has already reserved one UCM bundle per NPU.  Bundles are
    grouped by physical NUMA node and then reassigned in fallback-rank order;
    this is a permutation, so worker/ACL/release assignments stay untouched
    and no UCM cores overlap.
    """
    nodes = placement.available_nodes
    if not nodes or placement.node != nodes[placement.fallback_rank % len(nodes)]:
        return None

    bundles: list[UcmCoreBundle] = []
    for npu, cores_value in assign_ucm.items():
        cores = tuple(cores_value)
        health_cores = tuple(assign_ucm_health.get(npu, ()))
        all_cores = cores + health_cores
        if not all_cores:
            continue
        bundle_nodes = {cpu_to_node.get(core) for core in all_cores}
        if bundle_nodes != {placement.node}:
            continue
        bundles.append(UcmCoreBundle(npu, cores, health_cores))

    bundles.sort(
        key=lambda bundle: (
            min(bundle.cores + bundle.health_cores),
            bundle.source_npu,
        )
    )
    ordinal = placement.fallback_rank // len(nodes)
    if ordinal >= len(bundles):
        return None
    return bundles[ordinal]
