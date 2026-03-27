import logging
from typing import Any, List, Optional

import torch

from ucm.integration.sglang.ucm_connector import SglangUcmConnector

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

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

        self.mem_pool_host = mem_pool_host
        if self.connector is None:
            self.connector = SglangUcmConnector.from_hicache(
                self.storage_config, mem_pool_host
            )
            self.store = self.connector.store
        else:
            self.connector.mem_pool_host = mem_pool_host

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
        close = getattr(self.store, "close", None)
        if callable(close):
            close()

    def get_stats(self):
        connector = self.connector
        return None if connector is None else connector.get_stats()
