import logging
from typing import Any, Dict, List, Optional

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

try:
    from sglang.srt.mem_cache.hicache_storage import (
        PoolName,
        PoolTransfer,
        PoolTransferResult,
    )
except ImportError:
    PoolName = None
    PoolTransfer = Any
    PoolTransferResult = Any

try:
    from sglang.srt.mem_cache.pool_host import HostKVCache
except ImportError:
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

from ucm.integration.sglang.ucm_connector import SglangUcmConnector

logger = logging.getLogger(__name__)


class UnifiedCacheStore(HiCacheStorage):
    """HiCache L3 backend backed by UCM zero-copy load/store operations."""

    def __init__(
        self,
        storage_config: Optional[HiCacheStorageConfig] = None,
        context: Optional[Any] = None,
    ):
        if storage_config is None:
            raise ValueError("storage_config must be provided for UnifiedCacheStore.")

        self.storage_config = storage_config
        self.connector: Optional[SglangUcmConnector] = None
        self.store = None
        self.mem_pool_host: Optional[HostKVCache] = None
        self.registered_pools: Dict[str, Any] = {}
        self.pool_connectors: Dict[str, SglangUcmConnector] = {}

        if isinstance(context, HostKVCache):
            self.register_mem_pool_host(context)

    def _ensure_initialized(self) -> SglangUcmConnector:
        if self.connector is None or self.store is None or self.mem_pool_host is None:
            raise RuntimeError(
                "UnifiedCacheStore is not initialized yet. "
                "SGLang should call register_mem_pool_host() before storage operations."
            )
        return self.connector

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        if mem_pool_host.layout != "page_first":
            raise ValueError(
                "UnifiedCacheStore currently requires --hicache-mem-layout page_first, "
                f"got {mem_pool_host.layout!r}."
            )

        logger.info(
            "Registering SGLang UCM main mem pool: layout=%s, page_size=%s, dtype=%s",
            getattr(mem_pool_host, "layout", None),
            getattr(mem_pool_host, "page_size", None),
            getattr(mem_pool_host, "dtype", None),
        )
        self.mem_pool_host = mem_pool_host
        if self.connector is None:
            self.connector = SglangUcmConnector.from_hicache(
                self.storage_config, mem_pool_host, store_dir="kv"
            )
            self.store = self.connector.store
        else:
            self.connector.mem_pool_host = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        if self._is_kv_pool(host_pool_name):
            logger.debug("Skipping duplicate SGLang UCM KV pool registration.")
            return

        pool_name = self._pool_name_value(host_pool_name)
        self.registered_pools[pool_name] = host_pool
        pool_connector = self.pool_connectors.get(pool_name)
        if pool_connector is None:
            pool_connector = self._create_pool_connector(host_pool, host_pool_name)
            self.pool_connectors[pool_name] = pool_connector
        logger.info(
            "Registering SGLang UCM v2 mem pool: pool=%s, host_pool_type=%s, "
            "page_size=%s, store_dir=%s",
            pool_name,
            type(host_pool).__name__,
            getattr(host_pool, "page_size", None),
            pool_name,
        )

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        connector = self._ensure_initialized()
        if PoolTransferResult is None or PoolName is None:
            raise NotImplementedError(
                "SGLang batch_exists_v2 requires a HiCacheStorage with PoolTransferResult."
            )
        if not keys:
            return PoolTransferResult.empty()

        if getattr(connector.mem_pool_host, "kv_buffer", True) is None:
            kv_pages = len(keys)
        else:
            kv_pages = connector.batch_exists(keys, extra_info)

        hit_count: dict = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages
        for transfer in pool_transfers or []:
            if self._is_kv_pool(transfer.name):
                continue
            if final_pages == 0:
                break
            pool_pages = self._get_pool_connector(transfer.name).batch_exists(
                keys[:kv_pages], extra_info
            )
            if pool_pages:
                hit_count[transfer.name] = pool_pages
            final_pages = min(final_pages, pool_pages)

        logger.debug(
            "SGLang UCM batch_exists_v2 result: keys=%s, kv_pages=%s, "
            "final_pages=%s, hit_count=%s",
            len(keys),
            kv_pages,
            final_pages,
            hit_count,
        )
        return PoolTransferResult(final_pages, hit_count)

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, is_set=False, extra_info=extra_info)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, is_set=True, extra_info=extra_info)

    @staticmethod
    def _pool_name_value(pool_name: Any) -> str:
        return str(getattr(pool_name, "value", pool_name))

    def _is_kv_pool(self, pool_name: Any) -> bool:
        if PoolName is not None and pool_name == PoolName.KV:
            return True
        return self._pool_name_value(pool_name) == "kv"

    def _create_pool_connector(
        self,
        host_pool: HostKVCache,
        pool_name: Any,
    ) -> SglangUcmConnector:
        store_dir = self._pool_name_value(pool_name)
        logger.info(
            "Creating SGLang UCM side-pool connector: pool=%s, store_dir=%s",
            self._pool_name_value(pool_name),
            store_dir,
        )
        return SglangUcmConnector.from_hicache(
            self.storage_config,
            host_pool,
            store_dir=store_dir,
        )

    def _get_pool_connector(self, pool_name: Any) -> SglangUcmConnector:
        if self._is_kv_pool(pool_name):
            return self._ensure_initialized()
        pool_key = self._pool_name_value(pool_name)
        pool_connector = self.pool_connectors.get(pool_key)
        if pool_connector is None:
            raise ValueError(f"Unregistered SGLang UCM pool: {pool_name}")
        return pool_connector

    def _batch_io_v2(
        self,
        transfers: List[PoolTransfer],
        is_set: bool,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ):
        self._ensure_initialized()
        results: dict = {}
        for transfer in transfers:
            pool_name = transfer.name
            keys = list(transfer.keys or [])
            if not keys:
                results[str(pool_name)] = []
                continue

            pool_connector = self._get_pool_connector(pool_name)
            logger.debug(
                "Routing SGLang UCM v2 transfer to connector: op=%s, pool=%s, keys=%s",
                "set" if is_set else "get",
                self._pool_name_value(pool_name),
                len(keys),
            )
            if is_set:
                results[pool_name] = pool_connector.batch_set_v1(
                    keys, transfer.host_indices, extra_info
                )
            else:
                results[pool_name] = pool_connector.batch_get_v1(
                    keys, transfer.host_indices, extra_info
                )
        return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._ensure_initialized().batch_get_v1(keys, host_indices, extra_info)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._ensure_initialized().batch_set_v1(keys, host_indices, extra_info)

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError(
            "UnifiedCacheStore only supports the zero-copy batch_get_v1 interface."
        )

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        raise NotImplementedError(
            "UnifiedCacheStore only supports the zero-copy batch_get_v1 interface."
        )

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError(
            "UnifiedCacheStore only supports the zero-copy batch_set_v1 interface."
        )

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError(
            "UnifiedCacheStore only supports the zero-copy batch_set_v1 interface."
        )

    def exists(self, key: str) -> bool:
        return self._ensure_initialized().exists(key)

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        return self._ensure_initialized().batch_exists(keys, extra_info)

    def clear(self) -> bool:
        logger.warning("UnifiedCacheStore does not implement clear(); skipping.")
        return False

    def close(self) -> None:
        connector = self.connector
        if connector is not None:
            connector.close()
        closed_connector_ids = set()
        for pool_connector in self.pool_connectors.values():
            connector_id = id(pool_connector)
            if connector_id in closed_connector_ids:
                continue
            closed_connector_ids.add(connector_id)
            pool_connector.close()

    def get_stats(self):
        connector = self.connector
        return None if connector is None else connector.get_stats()
