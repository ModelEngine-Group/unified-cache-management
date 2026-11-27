import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
import time

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
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


class NfsstoreUnifiedCacheStore(HiCacheStorage):
    def __init__(
        self,
        storage_config: Optional["HiCacheStorageConfig"] = None,
        context: Optional[Any] = None,
    ):
        if storage_config is None:
            raise ValueError("storage_config must be provided for NfsstoreUnifiedCacheStore.")

        self.storage_config = storage_config
        self.store = None
        self.mem_pool_host: Optional["HostKVCache"] = None

        self.dtype = None
        self.page_size = 0
        self.model = ""
        self.is_mla = False
        self.cache_nums = 1
        self.tp_rank = 0
        self.tp_size = 1
        self.config_suffix = ""

        if isinstance(context, HostKVCache):
            self.register_mem_pool_host(context)

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host
        self.dtype = mem_pool_host.dtype
        self.page_size = mem_pool_host.page_size
        self.model = self.storage_config.model_name
        self.is_mla = self.storage_config.is_mla_model
        self.cache_nums = 1 if self.is_mla else 2
        self.tp_rank = self.storage_config.tp_rank
        self.tp_size = self.storage_config.tp_size
        self.k_size_per_token = mem_pool_host.layer_num * mem_pool_host.kv_lora_rank * self.dtype.itemsize
        self.v_size_per_token = mem_pool_host.layer_num * mem_pool_host.qk_rope_head_dim * self.dtype.itemsize
        self.k_size = self.page_size * self.k_size_per_token
        self.v_size = self.page_size * self.v_size_per_token
        self.config_suffix = self._build_config_suffix()

        if self.store is None:
            self.store = self._create_store()
        else:
            raise RuntimeError("NfsstoreUnifiedCacheStore already initialized")

    def _create_store(self):
        extra = dict(getattr(self.storage_config, "extra_config", None) or {})
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

        page_bytes = self.page_size * self.mem_pool_host.get_size_per_token()
        self.block_size = page_bytes

        ucm_cfg = kvc.get("ucm_connector_config")
        
        from ucm.store.nfsstore import ucmnfsstore
        store = ucmnfsstore.NFSStore()
        storage_backends = [
            path for path in ucm_cfg.get("storage_backends", "").split(":") if path
        ]
        transfer_enable = True
        param = ucmnfsstore.NFSStore.Config(storage_backends, self.block_size, transfer_enable)
        param.transferDeviceId = -1
        param.transferIoDirect = False

        ret = store.Setup(param)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize NFSStore, errcode: {ret}")

        return store

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

    def _generate_task(
        self,
        encoded_keys: List[bytes],
        host_indices: torch.Tensor,
    ):
        if not encoded_keys:
            return [], [], [], []

        block_ids = []
        offset_list = []
        ptr_list = []
        element_size_list = []
        k_buffer_data_ptr = self.mem_pool_host.k_buffer.data_ptr()
        v_buffer_data_ptr = self.mem_pool_host.v_buffer.data_ptr()
        key_index = 0
        indices = host_indices.tolist()
        for index in range(0, len(indices), self.page_size):
            k_ptr = k_buffer_data_ptr + indices[index] * self.k_size_per_token
            v_ptr = v_buffer_data_ptr + indices[index] * self.v_size_per_token
            block_ids.extend([encoded_keys[key_index], encoded_keys[key_index]])
            key_index += 1

            offset_list.extend([0, self.k_size])
            ptr_list.extend([k_ptr, v_ptr])
            element_size_list.extend([self.k_size, self.v_size])

        return block_ids, offset_list, ptr_list, element_size_list

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        if not keys:
            return []

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        block_ids, offset_list, ptr_list, size_list = self._generate_task(
            encoded_keys, host_indices
        )
        load_start_time = time.perf_counter() * 1000
        task = self.store.LoadToHost(block_ids, offset_list, ptr_list, size_list)
        try:
            self.store.Wait(task)
        except RuntimeError as e:
            logger.error(f"NFSStore load KVCache failed: {e}")
            return [False] * len(keys)
        load_end_time = time.perf_counter() * 1000
        load_speed = (
            len(keys)
            * self.block_size
            / (load_end_time - load_start_time)
            / 1024
            / 1024
        )  # GB/s

        logger.info(
            f"UnifiedCache load completed for {len(keys)} keys, "
            f"total size: {len(keys) * self.block_size / 1024 / 1024:.2f} MB, "
            f"time: {load_end_time - load_start_time:.2f} ms, "
            f"speed: {load_speed:.2f} GB/s"
        )
        return [True] * len(keys)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        if not keys:
            return []

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        block_ids, offset_list, ptr_list, size_list = self._generate_task(
            encoded_keys, host_indices
        )
        self.store.AllocBatch(encoded_keys)
        dump_start_time = time.perf_counter() * 1000
        task = self.store.DumpFromHost(block_ids, offset_list, ptr_list, size_list)
        try:
            ret = self.store.Wait(task)
            if ret != 0:
                self.store.CommitBatch(encoded_keys, False)
                logger.error("Failed to wait dump task.")
            else:
                self.store.CommitBatch(encoded_keys, True)
        except RuntimeError as e:
            logger.error(f"NFSStore dump KVCache failed: {e}")
            return [False] * len(keys)
        dump_end_time = time.perf_counter() * 1000
        dump_speed = (
            len(keys)
            * self.block_size
            / (dump_end_time - dump_start_time)
            / 1024
            / 1024
        )  # GB/s

        logger.info(
            f"UnifiedCache dump completed for {len(keys)} keys, "
            f"total size: {len(keys) * self.block_size / 1024 / 1024:.2f} MB, "
            f"time: {dump_end_time - dump_start_time:.2f} ms, "
            f"speed: {dump_speed:.2f} GB/s"
        )
        return [True] * len(keys)

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        raise NotImplementedError(
            "NfsstoreUnifiedCacheStore only supports the zero-copy batch_get_v1 interface."
        )

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        raise NotImplementedError(
            "NfsstoreUnifiedCacheStore only supports the zero-copy batch_get_v1 interface."
        )

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError(
            "NfsstoreUnifiedCacheStore only supports the zero-copy batch_set_v1 interface."
        )

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        raise NotImplementedError(
            "NfsstoreUnifiedCacheStore only supports the zero-copy batch_set_v1 interface."
        )

    def exists(self, key: str) -> bool:
        if self.is_mla and self.tp_rank != 0:
            return True
        result = self.store.Lookup(self._encode_keys([self._get_physical_key(key)]))
        return result[0] == 1

    def batch_exists(
        self, keys: List[str], extra_info: Optional["HiCacheStorageExtraInfo"] = None
    ) -> int:
        if not keys:
            return 0
        if self.is_mla and self.tp_rank != 0:
            return len(keys)
        results = self.store.LookupBatch(self._encode_keys(self._get_physical_keys(keys)))
        logger.info(f"=====look up results:{results}")
        for i, exists in enumerate(results):
            if not exists:
                return i
        return len(keys)

    def clear(self) -> bool:
        logger.warning("NfsstoreUnifiedCacheStore does not implement clear(); skipping.")
        return False

    def close(self) -> None:
        pass

    def get_stats(self):
        return None
