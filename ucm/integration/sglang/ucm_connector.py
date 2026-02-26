import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import xxhash
import yaml
from sglang.srt.distributed.parallel_state import get_world_group

from ucm.store.factory_v1 import UcmConnectorFactoryV1

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import (
        HiCacheStorageConfig,
        HiCacheStorageExtraInfo,
    )
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

UCM_META_BYTES: bytes | None = None
UCM_SEED_HASH = "UCM_HASH_SEED"


def uc_get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    if UCM_META_BYTES is None:
        raise RuntimeError(
            "UCM_META_BYTES is None, do not use uc_get_hash_str before register_uc_hasher"
        )

    hasher = xxhash.xxh64()
    hasher.update(UCM_META_BYTES)

    if prior_hash is None:
        prior_hash = UCM_SEED_HASH
    hasher.update(prior_hash.encode("utf-8"))

    for t in token_ids:
        if isinstance(t, tuple):
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


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
    name: str
    config: Dict[str, Any]

    @staticmethod
    def load_from_config(
        storage_config: "HiCacheStorageConfig", mem_pool_host: "HostKVCache"
    ) -> "UnifiedCacheStoreConfig":
        extra = getattr(storage_config, "extra_config", None)
        if extra is None:
            extra = _load_extra_config_from_yaml_env()
        if extra is None:
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

        return UnifiedCacheStoreConfig(name=name, config=cfg)


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

        self.register_uc_hasher()

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
            ucm_store_config.name, ucm_store_config.config
        )
        return cls(
            store,
            mem_pool_host,
            storage_config,
            ucm_store_config.config["storage_backends"],
        )

    def register_uc_hasher(self) -> None:
        global UCM_META_BYTES

        if self.is_mla:
            meta = f"{self.model}"
        else:
            meta = f"{self.model}:{self.tp_size}:{self.dtype}:{self.tp_rank}"
        UCM_META_BYTES = meta.encode("utf-8")

    def _encode_keys(self, keys: List[str]) -> List[bytes]:
        return [key.encode("utf-8") for key in keys]

    def _generate_task(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
    ):
        key_list = self._encode_keys(keys)
        shard_index_list = [0] * len(key_list)
        ptr_list, _ = self.mem_pool_host.get_page_buffer_meta(host_indices)

        if not self.is_mla:
            ptr_list = [list(p) for p in zip(ptr_list[::2], ptr_list[1::2])]
        else:
            ptr_list = [[p] for p in ptr_list]

        return key_list, shard_index_list, ptr_list

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional["HiCacheStorageExtraInfo"] = None,
    ) -> List[bool]:
        key_list, shard_index_list, ptr_list = self._generate_task(keys, host_indices)

        task = self.store.load_data(key_list, shard_index_list, ptr_list)
        try:
            self.store.wait(task)
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
        key_list, shard_index_list, ptr_list = self._generate_task(keys, host_indices)

        task = self.store.dump_data(key_list, shard_index_list, ptr_list)
        try:
            self.store.wait(task)
        except RuntimeError as e:
            logger.error(f"UnifiedCache dump KVCache failed: {e}")
            return [False] * len(keys)

        return [True] * len(keys)

    def exists(self, key: str) -> bool:
        if self.is_mla and self.tp_rank != 0:
            return True

        result = self.store.lookup(self._encode_keys([key]))
        return result[0] == 1

    def batch_exists(
        self, keys: List[str], extra_info: Optional["HiCacheStorageExtraInfo"] = None
    ) -> int:
        if self.is_mla and self.tp_rank != 0:
            return len(keys)

        return self.store.lookup_on_prefix(self._encode_keys(keys)) + 1

    def get_stats(self):
        return None
