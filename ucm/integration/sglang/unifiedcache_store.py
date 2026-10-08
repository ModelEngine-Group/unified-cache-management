import logging
from typing import Any, List, Optional

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolTransfer,
    PoolTransferResult,
)
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

        if isinstance(context, HostKVCache):
            self.register_mem_pool_host(context)

    def _ensure_initialized(self) -> SglangUcmConnector:
        if self.connector is None or self.mem_pool_host is None:
            raise RuntimeError(
                "UnifiedCacheStore is not initialized yet. "
                "SGLang should call register_mem_pool_host() before storage operations."
            )
        return self.connector

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        # HybridCacheController registers the HostPoolGroup allocation facade,
        # while the v1 storage path must operate on its physical KV anchor.
        # The group deliberately does not expose HostKVCache data APIs such as
        # get_size_per_token() or get_page_buffer_meta().
        anchor_entry = getattr(mem_pool_host, "anchor_entry", None)
        storage_host_pool = (
            anchor_entry.host_pool if anchor_entry is not None else mem_pool_host
        )

        super().register_mem_pool_host(storage_host_pool)
        if storage_host_pool.layout != "page_first":
            raise ValueError(
                "UnifiedCacheStore currently requires --hicache-mem-layout page_first, "
                f"got {storage_host_pool.layout!r}."
            )

        self.mem_pool_host = storage_host_pool
        if self.connector is None:
            self.connector = SglangUcmConnector.from_hicache(
                self.storage_config, storage_host_pool
            )
            self.store = self.connector.store
        else:
            self.connector.mem_pool_host = storage_host_pool

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        # SGLang registers the KV anchor through both APIs; v1 already owns it.
        if str(getattr(host_pool_name, "value", host_pool_name)) == "kv":
            return
        self._ensure_initialized().register_pool_v2(host_pool, host_pool_name)

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        return self._ensure_initialized().batch_exists_v2(
            keys, pool_transfers, extra_info
        )

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._ensure_initialized().batch_io_v2(transfers, is_set=False)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._ensure_initialized().batch_io_v2(transfers, is_set=True)

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
        if self.connector is not None:
            self.connector.close()

    def get_stats(self):
        connector = self.connector
        return None if connector is None else connector.get_stats()
