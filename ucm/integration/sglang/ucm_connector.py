import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
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


def _normalize_storage_backends(storage_backends: Any) -> List[str]:
    if isinstance(storage_backends, str):
        return [path for path in storage_backends.split(":") if path]
    if isinstance(storage_backends, Sequence) and not isinstance(
        storage_backends, (str, bytes)
    ):
        return [str(path) for path in storage_backends if str(path)]
    raise ValueError(
        "storage_backends must be a ':' separated string or a non-empty sequence"
    )


def _safe_dir_segment(value: Any) -> str:
    raw = str(getattr(value, "value", value))
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw)
    return safe or "store"


def _storage_backends_for_store(base_backends: List[str], store_dir: str) -> List[str]:
    backends = []
    for backend in base_backends:
        path = Path(backend) / store_dir
        path.mkdir(parents=True, exist_ok=True)
        backends.append(str(path))
    logger.info(
        "SGLang UCM store directory prepared: store_dir=%s, backends=%s",
        store_dir,
        backends,
    )
    return backends


def _config_for_store_dir(config: Dict[str, Any], store_dir: str) -> Dict[str, Any]:
    cfg = dict(config)
    cfg["storage_backends"] = _storage_backends_for_store(
        _normalize_storage_backends(config["storage_backends"]),
        store_dir,
    )
    return cfg


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
        storage_config: "HiCacheStorageConfig", mem_pool_host: "HostKVCache", store_dir: Optional[str] = None
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
        if hasattr(mem_pool_host, "get_size_per_token") and callable(mem_pool_host.get_size_per_token):
            page_bytes = page_size * mem_pool_host.get_size_per_token()
        elif hasattr(mem_pool_host, "size_per_token"):
            page_bytes = page_size * mem_pool_host.size_per_token
        else:
            raise ValueError(
                "mem_pool_host must have either get_size_per_token() method or size_per_token attribute"
            )
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
        cfg["store_pipeline"] = "Cache|Posix"
        cfg["storage_backends"] = _normalize_storage_backends(cfg["storage_backends"])
        cfg["device_id"] = get_world_group().local_rank
        cfg["tensor_size_list"] = [tensor_size]
        cfg["tensor_size"] = tensor_size
        safe_model_name = "-".join(storage_config.model_name.split("/")) if storage_config.model_name else ""
        cfg["unique_id"] = f"sglang{safe_model_name}" if store_dir is None else f"sglang{safe_model_name}_{store_dir}"
        cfg["cache_buffer_capacity_gb"] = 64
        cfg["io_direct"] = True
        cfg["cache_use_host_buffer"] = True
        cfg["shard_size"] = block_size
        cfg["block_size"] = block_size
        cfg["stream_number"] = 8
        logger.info(
            "Loaded SGLang UCM config: connector=%s, module=%s, model=%s, "
            "is_mla=%s, tensor_size=%s, block_size=%s, base_backends=%s",
            name,
            module_path,
            storage_config.model_name,
            storage_config.is_mla_model,
            tensor_size,
            block_size,
            cfg["storage_backends"],
        )

        return UnifiedCacheStoreConfig(
            module_path=module_path,
            name=name,
            config=cfg,
        )


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

    @classmethod
    def from_hicache(
        cls,
        storage_config: "HiCacheStorageConfig",
        mem_pool_host: "HostKVCache",
        store_dir: Optional[str] = None,
    ) -> "SglangUcmConnector":
        if mem_pool_host is None:
            raise ValueError("mem_pool_host must be provided for UnifiedCache")
        ucm_store_config = UnifiedCacheStoreConfig.load_from_config(
            storage_config, mem_pool_host, store_dir
        )
        store_config = (
            _config_for_store_dir(ucm_store_config.config, store_dir)
            if store_dir is not None
            else ucm_store_config.config
        )
        logger.info(
            "Creating SGLang UCM store: connector=%s, store_dir=%s, backends=%s",
            ucm_store_config.name,
            store_dir,
            store_config["storage_backends"],
        )
        store = UcmConnectorFactoryV1.create_connector(
            ucm_store_config.name, store_config, ucm_store_config.module_path
        )
        return cls(
            store,
            mem_pool_host,
            storage_config,
            store_config["storage_backends"],
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
        key_list, shard_index_list, ptr_list = self._generate_task(
            encoded_keys, host_indices
        )

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
        if not keys:
            return []

        encoded_keys = self._encode_keys(self._get_physical_keys(keys))
        key_list, shard_index_list, ptr_list = self._generate_task(
            encoded_keys, host_indices
        )

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

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()