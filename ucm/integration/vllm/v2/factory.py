"""Construct Store chains/routes and rank-scoped Proxy adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .parallel import AllShardLookup, ParallelLayout
from .test_stores import FileTestStore, MemoryTestStore
from .ucm_proxy import ParallelProxy, PipelineProxy, UCMProxyAdapter


def _storage_root(launch_config: dict[str, Any]) -> Path:
    explicit = launch_config.get("v2_storage_path")
    if explicit:
        return Path(str(explicit))

    connectors = launch_config.get("ucm_connectors")
    if (
        not isinstance(connectors, (list, tuple))
        or len(connectors) != 1
        or not isinstance(connectors[0], dict)
    ):
        raise ValueError(
            "connector v2 requires exactly one ucm_connectors entry or "
            "v2_storage_path"
        )
    store_config = connectors[0].get("ucm_connector_config") or {}
    if not isinstance(store_config, dict):
        raise ValueError("ucm_connector_config must be a mapping")
    backends = store_config.get("storage_backends")
    if isinstance(backends, (list, tuple)):
        backends = backends[0] if backends else None
    elif isinstance(backends, str) and os.pathsep in backends:
        backends = backends.split(os.pathsep, 1)[0]
    if not backends:
        raise ValueError("connector v2 requires storage_backends or v2_storage_path")
    return Path(str(backends))


def create_store(
    config: dict[str, Any],
    *,
    scope: Path,
    backend: FileTestStore | MemoryTestStore | None = None,
) -> FileTestStore | MemoryTestStore:
    kind = config.get("type")
    if kind == "test_file":
        if backend is not None:
            raise ValueError("test_file must be the final Store in a pipeline")
        if not config.get("path"):
            raise ValueError("test_file requires path")
        return FileTestStore(Path(config["path"]) / scope)
    if kind == "test_memory":
        return MemoryTestStore(
            backend=backend, capacity_bytes=int(config.get("capacity_bytes", 268435456))
        )
    raise ValueError(f"Unknown v2 Store type: {kind!r}")


def create_proxy(
    launch_config: dict[str, Any],
    *,
    namespace: str,
    record_token_span: int,
    parallel_layout: ParallelLayout,
    rank: int | None,
) -> UCMProxyAdapter:
    stores_config = launch_config.get("v2_stores")
    proxy_config = launch_config.get("v2_proxy")
    legacy_file = stores_config is None and proxy_config is None
    if legacy_file:
        stores_config = {
            "file": {"type": "test_file", "path": _storage_root(launch_config)}
        }
        proxy_config = {"type": "pipeline", "stores": ["file"]}
    if not isinstance(stores_config, dict) or not isinstance(proxy_config, dict):
        raise ValueError("v2_stores and v2_proxy must both be mappings")
    kind = proxy_config.get("type")
    if kind == "pipeline":
        names = proxy_config.get("stores", [])
        token_ends = []
    elif kind == "parallel":
        routes = proxy_config.get("routes", [])
        names = [route["store"] for route in routes]
        token_ends = [route["token_end"] for route in routes]
    else:
        raise ValueError(f"Unknown v2 Proxy type: {kind!r}")
    if not names or len(set(names)) != len(names):
        raise ValueError("Proxy must reference distinct named Stores")
    if any(name not in stores_config for name in names):
        raise ValueError("Proxy references an undefined Store")
    for name in names:
        definition = stores_config[name]
        if not isinstance(definition, dict) or definition.get("type") not in (
            "test_file",
            "test_memory",
        ):
            raise ValueError(f"Invalid v2 Store definition: {name!r}")
    scope = Path(".ucm-v2") / namespace
    if parallel_layout.world_size > 1:
        scope /= parallel_layout.namespace
    if rank is not None and not 0 <= rank < parallel_layout.world_size:
        raise ValueError("Worker rank is outside the parallel topology")
    # Each process owns its memory cache; file Stores supply shared visibility.
    # Independent memory-only routes cannot be queried across worker processes.
    if kind == "parallel" and any(
        stores_config[name]["type"] == "test_memory" for name in names
    ):
        raise ValueError(
            "Process-local test_memory routes require in-process construction; use shared Stores in connector config"
        )
    if kind == "pipeline" and stores_config[names[-1]]["type"] == "test_memory":
        raise ValueError("Connector test pipelines require a shared final Store")
    shards = []
    try:
        for index in range(parallel_layout.world_size):
            shard_scope = (
                scope / f"rank-{index}" if parallel_layout.world_size > 1 else scope
            )
            if kind == "pipeline":
                store = None
                for name in reversed(names):
                    store = create_store(
                        stores_config[name],
                        scope=shard_scope if legacy_file else shard_scope / name,
                        backend=store,
                    )
                shards.append(PipelineProxy(store))
            else:
                stores = [
                    create_store(stores_config[name], scope=shard_scope / name)
                    for name in names
                ]
                shards.append(ParallelProxy(stores, token_ends, record_token_span))
    except Exception:
        for shard in shards:
            shard.close()
        raise
    local = shards[rank or 0]
    proxy = AllShardLookup(local, shards) if len(shards) > 1 else local
    return UCMProxyAdapter(proxy)
