"""Construct the configured Stores and the one Proxy adapter over them."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .test_stores import FileTestStore, MemoryTestStore
from .ucm_proxy import ParallelProxy, PipelineProxy, UCMProxyAdapter


def _storage_root(launch_config: dict[str, Any]) -> Path:
    explicit_path = launch_config.get("v2_storage_path")
    if explicit_path:
        return Path(str(explicit_path))

    connector_configs = launch_config.get("ucm_connectors")
    if (
        not isinstance(connector_configs, (list, tuple))
        or len(connector_configs) != 1
        or not isinstance(connector_configs[0], dict)
    ):
        raise ValueError(
            "connector v2 requires exactly one ucm_connectors entry or "
            "v2_storage_path"
        )
    store_config = connector_configs[0].get("ucm_connector_config") or {}
    if not isinstance(store_config, dict):
        raise ValueError("ucm_connector_config must be a mapping")
    storage_backend = store_config.get("storage_backends")
    if isinstance(storage_backend, (list, tuple)):
        storage_backend = storage_backend[0] if storage_backend else None
    elif isinstance(storage_backend, str) and os.pathsep in storage_backend:
        storage_backend = storage_backend.split(os.pathsep, 1)[0]
    if not storage_backend:
        raise ValueError("connector v2 requires storage_backends or v2_storage_path")
    return Path(str(storage_backend))


def create_store(
    config: dict[str, Any],
    *,
    backend: FileTestStore | MemoryTestStore | None = None,
) -> FileTestStore | MemoryTestStore:
    """Build one Store from its own config; the connector adds no paths."""
    store_type = config.get("type")
    if store_type == "test_file":
        if backend is not None:
            raise ValueError("test_file must be the final Store in a pipeline")
        if not config.get("path"):
            raise ValueError("test_file requires path")
        return FileTestStore(Path(config["path"]))
    if store_type == "test_memory":
        return MemoryTestStore(
            backend=backend, capacity_bytes=int(config.get("capacity_bytes", 268435456))
        )
    raise ValueError(f"Unknown v2 Store type: {store_type!r}")


def create_proxy(
    launch_config: dict[str, Any],
    *,
    record_token_span: int,
) -> UCMProxyAdapter:
    """One Proxy per process. Records differ only by UCM chunk id: model,
    layout and topology are in the hash seed; ranks share each object key."""
    stores_config = launch_config.get("v2_stores")
    proxy_config = launch_config.get("v2_proxy")
    if stores_config is None and proxy_config is None:
        stores_config = {
            "file": {"type": "test_file", "path": _storage_root(launch_config)}
        }
        proxy_config = {"type": "pipeline", "stores": ["file"]}
    if not isinstance(stores_config, dict) or not isinstance(proxy_config, dict):
        raise ValueError("v2_stores and v2_proxy must both be mappings")
    proxy_type = proxy_config.get("type")
    if proxy_type == "pipeline":
        store_names = proxy_config.get("stores", [])
        token_ends = []
    elif proxy_type == "parallel":
        routes = proxy_config.get("routes", [])
        store_names = [route["store"] for route in routes]
        token_ends = [route["token_end"] for route in routes]
    else:
        raise ValueError(f"Unknown v2 Proxy type: {proxy_type!r}")
    if not store_names or len(set(store_names)) != len(store_names):
        raise ValueError("Proxy must reference distinct named Stores")
    if any(store_name not in stores_config for store_name in store_names):
        raise ValueError("Proxy references an undefined Store")
    for store_name in store_names:
        store_config = stores_config[store_name]
        if not isinstance(store_config, dict) or store_config.get("type") not in (
            "test_file",
            "test_memory",
        ):
            raise ValueError(f"Invalid v2 Store definition: {store_name!r}")
    file_paths = [
        Path(stores_config[store_name]["path"])
        for store_name in store_names
        if stores_config[store_name]["type"] == "test_file"
        and stores_config[store_name].get("path")
    ]
    if len(set(file_paths)) != len(file_paths):
        raise ValueError("Each test_file Store needs its own path")
    # Each process owns its memory cache; file Stores supply shared visibility.
    # Independent memory-only routes cannot be queried across worker processes.
    if proxy_type == "parallel" and any(
        stores_config[store_name]["type"] == "test_memory" for store_name in store_names
    ):
        raise ValueError(
            "Process-local test_memory routes require in-process construction; use shared Stores in connector config"
        )
    if (
        proxy_type == "pipeline"
        and stores_config[store_names[-1]]["type"] == "test_memory"
    ):
        raise ValueError("Connector test pipelines require a shared final Store")
    if proxy_type == "pipeline":
        backend_store = None
        for store_name in reversed(store_names):
            backend_store = create_store(
                stores_config[store_name], backend=backend_store
            )
        return UCMProxyAdapter(PipelineProxy(backend_store))
    route_stores = [
        create_store(stores_config[store_name]) for store_name in store_names
    ]
    return UCMProxyAdapter(ParallelProxy(route_stores, token_ends, record_token_span))
