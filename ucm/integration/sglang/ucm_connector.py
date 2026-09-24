import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import torch
import yaml
from sglang.srt.distributed.parallel_state import get_world_group

from ucm.store.factory_v1 import UcmConnectorFactoryV1

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import (
        HiCacheStorageConfig,
        HiCacheStorageExtraInfo,
        PoolTransfer,
    )
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def _load_extra_config_from_yaml_env() -> Optional[Dict[str, Any]]:
    cfg_path = os.environ.get("UNIFIEDCACHE_CONFIG_FILE")
    if not cfg_path:
        return None

    p = Path(cfg_path)
    if not p.is_file():
        return None

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"UNIFIEDCACHE_CONFIG_FILE YAML root must be a dict, got {type(data)}"
        )
    return data


@dataclass
class UnifiedCacheStoreConfig:
    module_path: str
    name: str
    config: Dict[str, Any]

    def fixed_size_config(
        self, namespace: str, tensor_size: int, tensors_per_block: int = 1
    ) -> Dict[str, Any]:
        """Build an isolated fixed-size Posix configuration for one v2 pool."""
        cfg = dict(self.config)
        safe_namespace = re.sub(r"[^A-Za-z0-9_.-]", "_", namespace)
        storage_backends = []
        for path in self.config["storage_backends"]:
            component_path = Path(path) / "sglang_v2" / safe_namespace
            # PosixStore creates its data directories but expects the configured
            # storage root itself to exist.
            component_path.mkdir(parents=True, exist_ok=True)
            storage_backends.append(str(component_path))
        cfg["storage_backends"] = storage_backends
        cfg["tensor_size"] = tensor_size
        block_size = tensor_size * tensors_per_block
        cfg["shard_size"] = block_size
        cfg["block_size"] = block_size
        return cfg

    @staticmethod
    def load_from_config(
        storage_config: "HiCacheStorageConfig", mem_pool_host: "HostKVCache"
    ) -> "UnifiedCacheStoreConfig":
        extra = dict(getattr(storage_config, "extra_config", None) or {})
        if "kv_connector_extra_config" not in extra:
            yaml_extra = _load_extra_config_from_yaml_env()
            if yaml_extra is not None:
                extra.update(yaml_extra)
        if not extra:
            raise ValueError(
                "Missing extra_config: storage_config.extra_config is None and "
                "UNIFIEDCACHE_CONFIG_FILE is not set or cannot be loaded"
            )

        kvc = extra.get("kv_connector_extra_config")
        if kvc is None:
            raise ValueError(
                "Missing config: extra_config['kv_connector_extra_config']"
            )

        is_logical_pool = getattr(mem_pool_host, "kv_buffer", None) is None
        if is_logical_pool:
            # DeepSeek V4 uses a LogicalHostPool as its KV anchor. It owns only
            # allocation indices; physical data and sizes arrive later through
            # register_mem_host_pool_v2(). These placeholders are never passed
            # to a store factory and are overwritten by fixed_size_config().
            tensor_size = 0
            block_size = 0
        else:
            page_size = mem_pool_host.page_size
            page_bytes = page_size * mem_pool_host.get_size_per_token()
            tensor_size = page_bytes if storage_config.is_mla_model else page_bytes // 2
            block_size = tensor_size * (1 if storage_config.is_mla_model else 2)

        ucm_cfg = kvc.get("ucm_connector_config")
        name = kvc.get("ucm_connector_name")
        module_path = kvc.get("ucm_connector_module_path")
        if ucm_cfg is None:
            raise ValueError(
                "Missing config: kv_connector_extra_config['ucm_connector_config']"
            )
        if name is None:
            raise ValueError(
                "Missing config: kv_connector_extra_config['ucm_connector_name']"
            )

        cfg = dict(ucm_cfg)
        cfg["store_pipeline"] = "Posix"
        cfg["storage_backends"] = [
            path for path in cfg["storage_backends"].split(":") if path
        ]
        cfg["device_id"] = get_world_group().local_rank
        cfg["tensor_size"] = tensor_size
        cfg["shard_size"] = block_size
        cfg["block_size"] = block_size
        cfg["stream_number"] = 8

        return UnifiedCacheStoreConfig(module_path=module_path, name=name, config=cfg)


class SglangUcmConnector:
    def __init__(
        self,
        store,
        mem_pool_host: "HostKVCache",
        storage_config: "HiCacheStorageConfig",
        storage_backends: List[str],
    ):
        self.store = store
        self.mem_pool_host = mem_pool_host
        self.storage_backends = storage_backends

        self.dtype = mem_pool_host.dtype
        self.page_size = mem_pool_host.page_size
        self.model = storage_config.model_name
        self.is_mla = storage_config.is_mla_model
        self.cache_nums = 1 if self.is_mla else 2
        self.tp_rank = storage_config.tp_rank
        self.tp_size = storage_config.tp_size

        self.config_suffix = self._build_config_suffix()
        self.ucm_store_config: Optional[UnifiedCacheStoreConfig] = None
        self.registered_pools: Dict[Any, Any] = {}
        self.pool_components: Dict[Any, List[tuple[Any, int]]] = {}
        self.pool_component_sizes: Dict[Any, List[int]] = {}
        self.flattened_pools: set[Any] = set()

    @classmethod
    def from_hicache(
        cls,
        storage_config: "HiCacheStorageConfig",
        mem_pool_host: "HostKVCache",
    ) -> "SglangUcmConnector":
        if mem_pool_host is None:
            raise ValueError("mem_pool_host must be provided for UnifiedCache")
        ucm_store_config = UnifiedCacheStoreConfig.load_from_config(
            storage_config, mem_pool_host
        )
        store = None
        if getattr(mem_pool_host, "kv_buffer", None) is not None:
            store = UcmConnectorFactoryV1.create_connector(
                ucm_store_config.name,
                ucm_store_config.config,
                ucm_store_config.module_path,
            )
        connector = cls(
            store,
            mem_pool_host,
            storage_config,
            ucm_store_config.config["storage_backends"],
        )
        connector.ucm_store_config = ucm_store_config
        return connector

    @staticmethod
    def _pool_value(pool_name: Any) -> str:
        return str(getattr(pool_name, "value", pool_name))

    @staticmethod
    def _flatten_page_meta(
        ptr_list: Sequence[Any], size_list: Sequence[Any], page_count: int
    ) -> tuple[List[List[int]], List[List[int]]]:
        """Normalize HostKVCache metadata into one component list per page."""
        if page_count == 0:
            return [], []
        if (
            len(ptr_list) == page_count
            and ptr_list
            and isinstance(ptr_list[0], (list, tuple))
        ):
            page_ptrs = [list(values) for values in ptr_list]
            page_sizes = [list(values) for values in size_list]
        else:
            if len(ptr_list) % page_count != 0:
                raise ValueError(
                    f"Host pool returned {len(ptr_list)} buffers for {page_count} pages"
                )
            width = len(ptr_list) // page_count
            page_ptrs = [
                list(ptr_list[i * width : (i + 1) * width]) for i in range(page_count)
            ]
            page_sizes = [
                list(size_list[i * width : (i + 1) * width]) for i in range(page_count)
            ]
        if any(len(p) != len(s) for p, s in zip(page_ptrs, page_sizes)):
            raise ValueError("Host pool returned mismatched pointer and size metadata")
        widths = {len(values) for values in page_ptrs}
        if len(widths) != 1:
            raise ValueError("Host pool component count must be stable across pages")
        return page_ptrs, page_sizes

    def register_pool_v2(self, host_pool: "HostKVCache", pool_name: Any) -> None:
        """Register an arbitrary hybrid pool and provision fixed-size stores lazily."""
        self.registered_pools[pool_name] = host_pool
        if self.ucm_store_config is None:
            raise RuntimeError("UCM connector configuration is not initialized")
        page_size = int(getattr(host_pool, "page_size", 1) or 1)
        probe_indices = torch.arange(page_size, dtype=torch.int64)
        ptrs, sizes = host_pool.get_page_buffer_meta(probe_indices)
        _, page_sizes = self._flatten_page_meta(ptrs, sizes, 1)
        pool_value = self._pool_value(pool_name)
        if pool_value == "mamba":
            # Mamba owns heterogeneous temporal/SSM and convolution buffers.
            # Store one fixed-size flattened page instead of provisioning one
            # PosixStore per component. This deliberately trades an L2 copy for
            # fewer stores and page-level storage atomicity.
            total_size = sum(int(size) for size in page_sizes[0] if int(size) > 0)
            if total_size <= 0:
                raise ValueError("Mamba hybrid pool has no non-empty components")
            namespace = f"{pool_value}_flat_{total_size}"
            cfg = self.ucm_store_config.fixed_size_config(namespace, total_size)
            store = UcmConnectorFactoryV1.create_connector(
                self.ucm_store_config.name,
                cfg,
                self.ucm_store_config.module_path,
            )
            self.pool_components[pool_name] = [(store, total_size)]
            self.pool_component_sizes[pool_name] = [total_size]
            self.flattened_pools.add(pool_name)
            return

        component_sizes = [int(size) for size in page_sizes[0] if int(size) > 0]
        if not component_sizes:
            raise ValueError(f"Hybrid pool {pool_value!r} has no non-empty components")
        if len(set(component_sizes)) != 1:
            raise ValueError(
                f"Hybrid pool {pool_value!r} contains unequal component sizes "
                f"{component_sizes}; asymmetric pools are not supported"
            )

        component_size = component_sizes[0]
        component_count = len(component_sizes)
        namespace = f"{pool_value}_{component_size}x{component_count}"
        cfg = self.ucm_store_config.fixed_size_config(
            namespace, component_size, component_count
        )
        store = UcmConnectorFactoryV1.create_connector(
            self.ucm_store_config.name,
            cfg,
            self.ucm_store_config.module_path,
        )
        self.pool_components[pool_name] = [(store, component_size)]
        self.pool_component_sizes[pool_name] = component_sizes

    def _component_key(self, logical_key: str, pool_name: Any, index: int) -> bytes:
        # A logical primary anchor means the physical payload is wholly owned
        # by the v2 pools.  Such layouts follow SGLang's replicated-MLA owner
        # semantics: TP0 persists one copy and every rank queries/loads that
        # same object.  A physical primary anchor can have rank-sharded hybrid
        # sidecars, so keep the TP rank in that case.
        tp_suffix = (
            ""
            if self._is_logical_anchor()
            else f"__v2_tp_{self.tp_rank}_{self.tp_size}"
        )
        physical = (
            f"{logical_key}{self.config_suffix}"
            f"{tp_suffix}"
            f"__pool_{self._pool_value(pool_name)}__component_{index}"
        )
        return self._encode_key(physical)

    def _transfer_meta(self, transfer: "PoolTransfer"):
        host_pool = self.registered_pools.get(transfer.name)
        if host_pool is None:
            raise ValueError(f"Unregistered UCM hybrid pool: {transfer.name}")
        keys = list(transfer.keys or [])
        if not keys:
            return keys, [], []
        page_size = int(getattr(host_pool, "page_size", 1) or 1)
        if transfer.host_indices is None or len(transfer.host_indices) != (
            len(keys) * page_size
        ):
            raise ValueError(
                f"Hybrid pool {transfer.name} expects "
                f"{len(keys) * page_size} host indices"
            )
        ptrs, sizes = host_pool.get_page_buffer_meta(transfer.host_indices)
        page_ptrs, page_sizes = self._flatten_page_meta(ptrs, sizes, len(keys))
        expected = self.pool_component_sizes[transfer.name]
        for sizes_for_page in page_sizes:
            actual = [int(size) for size in sizes_for_page if int(size) > 0]
            if actual != expected:
                raise ValueError(
                    f"Hybrid pool {transfer.name} component sizes changed: "
                    f"expected {expected}, got {actual}"
                )
        page_ptrs = [
            [ptr for ptr, size in zip(ptrs_for_page, sizes_for_page) if int(size) > 0]
            for ptrs_for_page, sizes_for_page in zip(page_ptrs, page_sizes)
        ]
        return keys, page_ptrs, page_sizes

    def batch_io_v2(
        self, transfers: List["PoolTransfer"], is_set: bool
    ) -> Dict[str, List[bool]]:
        results: Dict[str, List[bool]] = {}
        for transfer in transfers:
            if transfer.name in self.flattened_pools:
                page_results = self._batch_io_flattened_pool(transfer, is_set)
                results[transfer.name] = page_results
                continue
            keys, page_ptrs, _ = self._transfer_meta(transfer)
            page_results = [True] * len(keys)
            store, _ = self.pool_components[transfer.name][0]
            encoded = [self._component_key(key, transfer.name, 0) for key in keys]
            try:
                task = (
                    store.dump_data(encoded, [0] * len(keys), page_ptrs)
                    if is_set
                    else store.load_data(encoded, [0] * len(keys), page_ptrs)
                )
                store.wait(task)
            except RuntimeError as exc:
                logger.error(
                    "UnifiedCache %s failed for pool %s: %s",
                    "dump" if is_set else "load",
                    transfer.name,
                    exc,
                )
                page_results = [False] * len(keys)
            results[transfer.name] = page_results
        return results

    def _batch_io_flattened_pool(
        self, transfer: "PoolTransfer", is_set: bool
    ) -> List[bool]:
        """Transfer a heterogeneous pool as one contiguous page per key."""
        host_pool = self.registered_pools[transfer.name]
        keys = list(transfer.keys or [])
        if not keys:
            return []
        page_size = int(getattr(host_pool, "page_size", 1) or 1)
        host_indices = transfer.host_indices
        if host_indices is None or len(host_indices) != len(keys) * page_size:
            raise ValueError(
                f"Hybrid pool {transfer.name} expects "
                f"{len(keys) * page_size} host indices"
            )

        page_offsets = [
            int(host_indices[index * page_size].item()) for index in range(len(keys))
        ]
        if is_set:
            staging_pages = [
                host_pool.get_data_page(offset, flat=True) for offset in page_offsets
            ]
        else:
            staging_pages = [host_pool.get_dummy_flat_data_page() for _ in page_offsets]

        store, expected_size = self.pool_components[transfer.name][0]
        actual_sizes = [page.numel() * page.element_size() for page in staging_pages]
        if any(size != expected_size for size in actual_sizes):
            raise ValueError(
                f"Hybrid pool {transfer.name} flattened page size changed: "
                f"expected {expected_size}, got {actual_sizes}"
            )

        encoded = [self._component_key(key, transfer.name, 0) for key in keys]
        pointers = [[page.data_ptr()] for page in staging_pages]
        try:
            task = (
                store.dump_data(encoded, [0] * len(keys), pointers)
                if is_set
                else store.load_data(encoded, [0] * len(keys), pointers)
            )
            store.wait(task)
        except RuntimeError as exc:
            logger.error(
                "UnifiedCache %s failed for flattened pool %s: %s",
                "dump" if is_set else "load",
                transfer.name,
                exc,
            )
            return [False] * len(keys)

        if not is_set:
            for offset, page in zip(page_offsets, staging_pages):
                host_pool.set_from_flat_data_page(offset, page)
        return [True] * len(keys)

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        from sglang.srt.mem_cache.hicache_storage import (
            PoolHitPolicy,
            PoolTransferResult,
        )

        # Some hybrid layouts (notably DeepSeek V4) use a logical KV anchor;
        # all physical payloads live in v2 pools in that case.
        kv_pages = (
            len(keys)
            if getattr(self.mem_pool_host, "kv_buffer", None) is None
            else self.batch_exists(keys, extra_info)
        )
        hit_counts = {"kv": kv_pages} if kv_pages else {}
        # Retain every valid stop point: a trailing state window can make a
        # shorter prefix restorable even when a longer one is not.
        restorable = list(range(1, kv_pages + 1))

        for transfer in pool_transfers or []:
            if not restorable:
                break
            components = self.pool_components.get(transfer.name)
            if components is None:
                raise ValueError(f"Unregistered UCM hybrid pool: {transfer.name}")
            store, _ = components[0]
            trailing_keys = list(transfer.keys or [])
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                encoded = [
                    self._component_key(key, transfer.name, 0)
                    for key in keys[:kv_pages]
                ]
                page_exists = [bool(value) for value in store.lookup(encoded)]
                boundary = (
                    page_exists.index(False) if False in page_exists else kv_pages
                )
                pool_restorable = list(range(1, boundary + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                # Trailing pools are sparse: an object marks a candidate
                # prefix end instead of contributing to a contiguous prefix.
                # Lookup returns an independent existence bit for every
                # candidate, so all legal trailing windows can be validated in
                # one request. Placeholder values carry only the window size
                # and are never storage keys.
                encoded = [
                    self._component_key(key, transfer.name, 0)
                    for key in keys[:kv_pages]
                ]
                trailing = max(1, len(trailing_keys))
                pool_restorable = []
                boundary = 0
                page_exists = [bool(value) for value in store.lookup(encoded)]
                for prefix_len in range(kv_pages, 0, -1):
                    window_start = max(0, prefix_len - trailing)
                    if not all(page_exists[window_start:prefix_len]):
                        continue
                    pool_restorable.append(prefix_len)
                    if boundary == 0:
                        boundary = prefix_len
            else:
                raise ValueError(f"Unsupported pool hit policy: {transfer.hit_policy}")

            if boundary:
                hit_counts[transfer.name] = boundary
            allowed = set(pool_restorable)
            restorable = [value for value in restorable if value in allowed]
        final_pages = restorable[-1] if restorable else 0
        return PoolTransferResult(final_pages, hit_counts, restorable)

    def close(self) -> None:
        seen = set()
        for store in [self.store] + [
            store
            for components in self.pool_components.values()
            for store, _ in components
        ]:
            if id(store) in seen:
                continue
            seen.add(id(store))
            close = getattr(store, "close", None)
            if callable(close):
                close()

    def _encode_key(self, key: str) -> bytes:
        return hashlib.md5(key.encode("utf-8")).digest()

    def _encode_keys(self, keys: List[str]) -> List[bytes]:
        return [self._encode_key(key) for key in keys]

    def _build_config_suffix(self) -> str:
        model_name = "-".join(self.model.split("/")) if self.model else ""
        if self.is_mla:
            return f"_{model_name}"
        return f"_{model_name}_{self.tp_rank}_{self.tp_size}"

    def _get_physical_key(self, logical_key: str) -> str:
        return logical_key + self.config_suffix

    def _get_physical_keys(self, logical_keys: List[str]) -> List[str]:
        return [self._get_physical_key(key) for key in logical_keys]

    def _is_logical_anchor(self) -> bool:
        return (
            self.store is None
            and getattr(self.mem_pool_host, "kv_buffer", None) is None
        )

    def _require_primary_store(self):
        if self.store is None:
            raise RuntimeError("UnifiedCache primary KV store is not initialized")
        return self.store

    def _generate_task(
        self,
        encoded_keys: List[bytes],
        host_indices: torch.Tensor,
    ):
        if not encoded_keys:
            return [], [], []

        shard_index_list = [0] * len(encoded_keys)
        ptr_list, _ = self.mem_pool_host.get_page_buffer_meta(host_indices)

        if not self.is_mla:
            ptr_list = [list(p) for p in zip(ptr_list[::2], ptr_list[1::2])]
        else:
            ptr_list = [[p] for p in ptr_list]

        return encoded_keys, shard_index_list, ptr_list

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        if not keys:
            return []
        if self._is_logical_anchor():
            # DeepSeek V4's KV anchor contains indices only. Physical payloads
            # are restored by the accompanying batch_get_v2() transfers.
            return [True] * len(keys)

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        key_list, shard_index_list, ptr_list = self._generate_task(
            encoded_keys, host_indices
        )

        store = self._require_primary_store()
        task = store.load_data(key_list, shard_index_list, ptr_list)
        try:
            store.wait(task)
        except RuntimeError as e:
            logger.error(f"UnifiedCache load KVCache failed: {e}")
            return [False] * len(keys)

        return [True] * len(keys)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        if not keys:
            return []
        if self._is_logical_anchor():
            # HybridCacheController still invokes the primary v1 path after
            # writing v2 sidecars. The logical anchor has no bytes to persist.
            return [True] * len(keys)

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        key_list, shard_index_list, ptr_list = self._generate_task(
            encoded_keys, host_indices
        )

        store = self._require_primary_store()
        task = store.dump_data(key_list, shard_index_list, ptr_list)
        try:
            store.wait(task)
        except RuntimeError as e:
            logger.error(f"UnifiedCache dump KVCache failed: {e}")
            return [False] * len(keys)

        return [True] * len(keys)

    def exists(self, key: str) -> bool:
        if self._is_logical_anchor():
            return True
        if self.is_mla and self.tp_rank != 0:
            return True

        store = self._require_primary_store()
        result = store.lookup(self._encode_keys([self._get_physical_key(key)]))
        return result[0] == 1

    def batch_exists(
        self, keys: List[str], extra_info: Optional["HiCacheStorageExtraInfo"] = None
    ) -> int:
        if not keys:
            return 0
        if self._is_logical_anchor():
            return len(keys)
        if self.is_mla and self.tp_rank != 0:
            return len(keys)

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        return self._require_primary_store().lookup_on_prefix(encoded_keys) + 1

    def get_stats(self):
        return None
