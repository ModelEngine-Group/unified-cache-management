import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import yaml
import time
from sglang.srt.distributed.parallel_state import get_world_group

from ucm.store.factory_v1 import UcmConnectorFactoryV1

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import (
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


@dataclass
class UnifiedCacheStoreConfig:
    module_path: str
    name: str
    config: Dict[str, Any]

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
        if mem_pool_host.layout == "page_first_kv_split":
            cfg["store_pipeline"] = "Cache|Posix"
            k_size = mem_pool_host.layer_num * mem_pool_host.kv_lora_rank * mem_pool_host.dtype.itemsize * page_size
            v_size = mem_pool_host.layer_num * mem_pool_host.qk_rope_head_dim * mem_pool_host.dtype.itemsize * page_size
            cfg["tensor_size_list"] = [k_size, v_size]
            safe_model_name = "-".join(storage_config.model_name.split("/")) if storage_config.model_name else ""
            cfg["unique_id"] = f"sglang{safe_model_name}"
            cfg["cache_buffer_capacity_gb"] = 64
            cfg["io_direct"] = True
            cfg["cache_use_host_buffer"] = True
        else:
            cfg["store_pipeline"] = "Posix"
            cfg["tensor_size"] = tensor_size

        cfg["storage_backends"] = [
            path for path in cfg["storage_backends"].split(":") if path
        ]
        cfg["device_id"] = get_world_group().local_rank
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
        page_size = mem_pool_host.page_size
        page_bytes = page_size * mem_pool_host.get_size_per_token()
        tensor_size = page_bytes if storage_config.is_mla_model else page_bytes // 2
        self.block_data_size = tensor_size * (1 if storage_config.is_mla_model else 2)
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

        self.is_kv_split = mem_pool_host.layout == "page_first_kv_split"
        if self.is_kv_split:
            self.k_size_per_token = mem_pool_host.layer_num * mem_pool_host.kv_lora_rank * self.dtype.itemsize
            self.v_size_per_token = mem_pool_host.layer_num * mem_pool_host.qk_rope_head_dim * self.dtype.itemsize
        self.config_suffix = self._build_config_suffix()

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
        store = UcmConnectorFactoryV1.create_connector(
            ucm_store_config.name, ucm_store_config.config, ucm_store_config.module_path
        )
        return cls(
            store,
            mem_pool_host,
            storage_config,
            ucm_store_config.config["storage_backends"],
        )

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

    def _generate_task_split(
        self,
        encoded_keys: List[bytes],
        host_indices: torch.Tensor,
    ):
        if not encoded_keys:
            return [], [], []
        shard_index_list = [0] * len(encoded_keys)
        block_ids = []
        ptr_list = []
        k_buffer_data_ptr = self.mem_pool_host.k_buffer.data_ptr()
        v_buffer_data_ptr = self.mem_pool_host.v_buffer.data_ptr()
        key_index = 0
        indices = host_indices.tolist()
        for index in range(0, len(indices), self.page_size):
            k_ptr = k_buffer_data_ptr + indices[index] * self.k_size_per_token
            v_ptr = v_buffer_data_ptr + indices[index] * self.v_size_per_token
            block_ids.append(encoded_keys[key_index])
            key_index += 1
            ptr_list.append([k_ptr, v_ptr])
        return block_ids, shard_index_list, ptr_list

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

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        if self.is_kv_split:
            key_list, shard_index_list, ptr_list = self._generate_task_split(encoded_keys, host_indices)
        else:
            key_list, shard_index_list, ptr_list = self._generate_task(encoded_keys, host_indices)

        load_start_time = time.perf_counter() * 1000
        task = self.store.load_data(key_list, shard_index_list, ptr_list)
        try:
            self.store.wait(task)
            load_end_time = time.perf_counter() * 1000
        except RuntimeError as e:
            logger.error(f"UnifiedCache load KVCache failed: {e}")
            return [False] * len(keys)
        load_speed = (
            len(keys)
            * self.block_data_size
            / (load_end_time - load_start_time)
            / 1024
            / 1024
        )  # GB/s

        logger.info(
            f"UnifiedCache load completed for {len(keys)} keys, "
            f"total size: {len(keys) * self.block_data_size / 1024 / 1024:.2f} MB, "
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
        if self.is_kv_split:
            key_list, shard_index_list, ptr_list = self._generate_task_split(encoded_keys, host_indices)
        else:
            key_list, shard_index_list, ptr_list = self._generate_task(encoded_keys, host_indices)
        dump_start_time = time.perf_counter() * 1000
        task = self.store.dump_data(key_list, shard_index_list, ptr_list)
        try:
            self.store.wait(task)
        except RuntimeError as e:
            logger.error(f"UnifiedCache dump KVCache failed: {e}")
            return [False] * len(keys)
        dump_end_time = time.perf_counter() * 1000
        dump_speed = (
            len(keys)
            * self.block_data_size
            / (dump_end_time - dump_start_time)
            / 1024
            / 1024
        )  # GB/s

        logger.info(
            f"UnifiedCache dump completed for {len(keys)} keys, "
            f"total size: {len(keys) * self.block_data_size / 1024 / 1024:.2f} MB, "
            f"time: {dump_end_time - dump_start_time:.2f} ms, "
            f"speed: {dump_speed:.2f} GB/s"
        )
        return [True] * len(keys)

    def exists(self, key: str) -> bool:
        if self.is_mla and self.tp_rank != 0:
            return True

        result = self.store.lookup(self._encode_keys([self._get_physical_key(key)]))
        return result[0] == 1

    def batch_exists(
        self, keys: List[str], extra_info: Optional["HiCacheStorageExtraInfo"] = None
    ) -> int:
        if not keys:
            return 0
        if self.is_mla and self.tp_rank != 0:
            return len(keys)

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        return self.store.lookup_on_prefix(encoded_keys) + 1

    def get_stats(self):
        return None
