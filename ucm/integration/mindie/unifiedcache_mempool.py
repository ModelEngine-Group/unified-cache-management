import fcntl
import functools
import hashlib
import json
import os
import time
from datetime import datetime
from enum import IntFlag
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mindie_llm.utils.file_utils import safe_open
from mindie_llm.utils.log.logging import logger
from pydantic import BaseModel, Field

from .base import MemPool

WORK_ROLE = "worker"
SCHEDULER_ROLE = "scheduler"
LOAD_OK = 0
DUMP_OK = 0
LOAD_ERROR = -1
DUMP_ERROR = -1
TASK_INVALID = 0

_ENABLE_TIME_STAT = os.getenv("MINDIE_UC_TIME_STAT", "0") == "1"
_STORAGE_PLATFORM = os.environ.get("STORAGE_PLATFORM")


def uc_timeit(name: str):
    """
    UC api cost decorator
    """

    def decorator(func):
        if not _ENABLE_TIME_STAT:
            # if MINDIE_UC_TIME_STAT not set, direct exec func
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                cost_ms = (time.perf_counter() - start) * 1000
                logger.info(f"[UC][TIME] {name} cost {cost_ms:.3f} ms")

        return wrapper

    return decorator


def get_dual_consensus_uids(
    coordination_file="/dev/shm/ucm_dual_uid.json",
    prefix_a="k_store_uid",
    timeout=180,
):
    """
    Coordinate across multiple processes to ensure consistency
    and generate dual timestamp-based UIDs in a single transaction.
    :return: mla_store_uid
    """
    os.makedirs(os.path.dirname(coordination_file), exist_ok=True)

    with open(coordination_file, "a+") as f:
        try:
            # 1. lock between process
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            content = f.read()
            now = int(time.time())
            data = {}
            if content:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    pass
            last_ts = data.get("timestamp", 0)

            # 2. check whether the config is valid
            if not data.get("mla_store_uid") or (now - last_ts) > timeout:
                # get uid
                date_str = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
                mla_store_uid = f"{prefix_a}_{date_str}"
                new_data = {
                    "timestamp": now,
                    "mla_store_uid": mla_store_uid,
                }

                # write into new config
                f.seek(0)
                f.truncate()
                f.write(json.dumps(new_data))
                f.flush()
                os.fsync(f.fileno())
                logger.info(f"[UC] npu worker: {os.getpid()} set UIDs: {mla_store_uid}")
            else:
                mla_store_uid = data["mla_store_uid"]
                logger.info(f"[UC] npu worker: {os.getpid()} sync UID: {mla_store_uid}")
            return mla_store_uid
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def str_to_md5_bytes(key: str) -> bytes:
    return hashlib.md5(key.encode("utf-8")).digest()


class BypassMode(IntFlag):
    NONE = 0
    GET = 1
    PUT = 2
    EXISTS = 4


class MempoolConfig(BaseModel):
    is_mla: bool = False
    scp_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    dp_size: int = 1
    sp_size: int = 1


class PipelineStoreConfig(BaseModel):
    store_pipeline: str = "Cache|Posix"
    storage_backends: List[str] = [""]
    unique_id: str
    device_id: int = -1
    timeout_ms: Optional[int] = None

    tensor_size_list: List[int] = [0]
    shard_size: int = 0
    block_size: int = 0

    cache_buffer_capacity_gb: Optional[int] = None
    cache_stream_number: Optional[int] = None
    share_buffer_enable: bool = False
    waiting_queue_depth: Optional[int] = None
    running_queue_depth: Optional[int] = None

    posix_data_trans_concurrency: Optional[int] = None
    posix_lookup_concurrency: Optional[int] = None
    io_direct: Optional[bool] = None


class KVCSStoreConfig(PipelineStoreConfig):
    store_pipeline: str = "KVCS"
    kvcs_instance_name: str
    kvcs_store_id: int
    kvcs_tls_enable: bool = False
    kvcs_ucm_over_tcp_ip_list: str
    kvcs_block_size: int = 128
    kvcs_sliding_window_size: int = Field(default=100, ge=10, le=1000)
    kvcs_failure_rate_threshold: int = Field(default=10, ge=0, le=100)
    kvcs_consecutive_fail_limit: int = Field(default=5, ge=1, le=100)


class ConfigParser:
    @staticmethod
    def safe_load(path: str) -> Dict[str, Any]:
        with safe_open(path, "r") as f:
            return json.load(f)

    @classmethod
    def parse_mindie_config(cls, mindie_config: Dict[str, Any]):
        backend = mindie_config.get("BackendConfig") or mindie_config.get(
            "mindie_server_prefill_config"
        ).get("BackendConfig")

        model_config = backend.get("ModelDeployConfig").get("ModelConfig")[0]

        dp = model_config.get("dp", 1)
        tp = model_config.get("tp", 1)
        sp = model_config.get("sp", 1)
        cp = model_config.get("cp", 1)
        world_size = model_config.get("worldSize", 1)
        scp_size = sp * cp
        if tp * dp == 1 or tp * cp == 1:
            # tp is default set to world_size
            tp = world_size

        return {
            "dp_size": dp,
            "tp_size": tp,
            "sp_size": sp,
            "cp_size": cp,
            "scp_size": scp_size,
            "backend_raw": backend,
            "weight_path": model_config.get("modelWeightPath"),
        }

    @classmethod
    def parse_model_config(cls, weight_path: str, scp_size: int):
        share_buffer_enable = False
        is_mla = False

        config_json = os.path.join(weight_path, "config.json")
        model_arc = ConfigParser.safe_load(config_json)
        if "kv_lora_rank" in model_arc:
            # for deepseek model
            is_mla = True
            share_buffer_enable = scp_size == 1

        return is_mla, share_buffer_enable

    @classmethod
    def check_kvcs_certificates(cls):
        certs_to_check = [
            "/etc/kvcs/tls/ca.crt",
            "/etc/kvcs/tls/tls.crt" "/etc/kvcs/tls/tls.key",
        ]
        missing_certs = []

        for cert_path in certs_to_check:
            path = Path(cert_path)
            if not path.is_file():
                missing_certs.append(cert_path)

        if missing_certs:
            raise FileNotFoundError(
                f"KVCS certificates missing, mempool initialization failed: {', '.join(missing_certs)}. "
            )

    @classmethod
    def get_io_size_info(cls, kv_caches: Any):
        if kv_caches is None:
            return [0], 0

        k_tensor = kv_caches[0][0][0]
        v_tensor = kv_caches[0][1][0]

        k_io_size = k_tensor.numel() * k_tensor.element_size()
        v_io_size = v_tensor.numel() * v_tensor.element_size()
        num_layers = len(kv_caches)

        tensor_size_list = [k_io_size] * num_layers + [v_io_size] * num_layers
        shard_size = sum(tensor_size_list)

        return tensor_size_list, shard_size

    @classmethod
    def get_unified_config(
        cls, config_path: str, device_id: int, kv_caches: Any
    ) -> Tuple[MempoolConfig, PipelineStoreConfig]:
        uc_config = ConfigParser.safe_load(config_path)
        mindie_config = ConfigParser.safe_load(uc_config.get("mindie_config_path"))

        parallel_config = ConfigParser.parse_mindie_config(mindie_config)
        is_mla, share_buffer_enable = ConfigParser.parse_model_config(
            parallel_config["weight_path"], parallel_config["scp_size"]
        )
        tensor_sizes, shard_size = ConfigParser.get_io_size_info(kv_caches)

        mempool_config = MempoolConfig(is_mla=is_mla, **parallel_config)
        store_data = {
            **uc_config,
            "device_id": device_id,
            "share_buffer_enable": share_buffer_enable,
            "unique_id": get_dual_consensus_uids(),
            "tensor_size_list": tensor_sizes,
            "shard_size": shard_size,
            "block_size": shard_size,
            "kvcs_report_divisor": (
                1 if mempool_config.scp_size > 1 else mempool_config.tp_size
            ),
        }

        if os.getenv("NUM_ACCELERATOR") is not None:
            config_cls = PipelineStoreConfig
        else:
            platform_map = {
                None: KVCSStoreConfig,
                "ASERIES": KVCSStoreConfig,
                "PACIFIC9950": PipelineStoreConfig,
            }
            if _STORAGE_PLATFORM in platform_map:
                config_cls = platform_map[_STORAGE_PLATFORM]
            else:
                raise ValueError(f"Invalid STORAGE_PLATFORM: '{_STORAGE_PLATFORM}'. ")

        store_config = config_cls(**store_data)

        if config_cls == KVCSStoreConfig and store_config.kvcs_tls_enable:
            ConfigParser.check_kvcs_certificates()

        return mempool_config, store_config


class UnifiedCacheMempool(MemPool):

    def __init__(self, config_path, role, **kwargs):
        device_id = kwargs.get("device_id", -1)
        kv_caches = kwargs.get("kv_caches", None)

        self.runtime_cfg, self.store_cfg = ConfigParser.get_unified_config(
            config_path, device_id, kv_caches
        )

        logger.info(
            f"[UC] uc_config: {str(self.store_cfg.model_dump(exclude_none=True))}"
        )
        self.uc_store = self._init_store_engine()

        logger.info("[UC]: Initialize unifiedcache success.")

        self._setup_runtime_states()

    def _init_store_engine(self):
        from ucm.store.factory_v1 import UcmConnectorFactoryV1

        if isinstance(self.store_cfg, KVCSStoreConfig):
            module_path = "ucm.store.kvcs.connector"
            class_name = "UcmKvcsStore"
        else:
            module_path = "ucm.store.pipeline.connector"
            class_name = "UcmPipelineStore"
        return UcmConnectorFactoryV1.create_connector(
            class_name, self.store_cfg.model_dump(exclude_none=True), module_path
        )

    def _setup_runtime_states(self):
        bypass_val = int(os.getenv("BYPASS_UC", "0"))
        self.bypass = BypassMode(bypass_val)
        logger.info(f"[UC] bypass:{self.bypass.name}")

        self.tp_rank = -1

    def _get_tp_rank_0_hash_key(self, keys):
        # should be aligned with the get_prefix_keys func in prefix_cache_plugin.py
        rank0_keys = []
        for key in keys:
            parts = key.split("_")
            cur_tp = int(parts[1])
            if self.tp_rank < 0:
                self.tp_rank = cur_tp
            if cur_tp == 0:
                return keys, cur_tp
            parts[1] = "0"
            rank0_key = "_".join(parts)
            rank0_keys.append(rank0_key)
        return rank0_keys, cur_tp

    def _get_tensors_for_store(self, mindie_tensors):
        kv_tensors = []
        num_blocks = len(mindie_tensors)
        for i in range(num_blocks):
            kv_flat_list = [item for sublist in mindie_tensors[i] for item in sublist]
            kv_tensors.append(kv_flat_list)
        return kv_tensors

    def _check_task(self, task, store_name):
        if task is None or task.task_id == TASK_INVALID:
            logger.error(f"[UC][{store_name}] invalid task: {task}")
            return False
        return True

    def _wait_tasks(self, tasks):
        for store, task, name in tasks:
            try:
                store.wait(task)
            except RuntimeError as e:
                logger.error(
                    f"[UC][{name}] wait failed, task_id={task.task_id}, err={e}"
                )
                return False
        return True

    @uc_timeit("exists")
    def exists(self, keys: Union[str, List], **kwargs) -> bool:
        """
        Judge whether current key is in store
        current only scheduler call the exists api for each tp_rank/scp_rank's block_hash_key

        Args:
            keys (Union[str, List]): MindIE block prefix_key.

        Returns:
            Bool
        """
        if self.bypass & BypassMode.EXISTS:
            return False

        if isinstance(keys, str):
            keys = [keys]

        if self.runtime_cfg.is_mla and self.runtime_cfg.scp_size == 1:
            keys, cur_tp = self._get_tp_rank_0_hash_key(keys)
            # for deepseek model, all tp rank share the same cache, only need lookup block_hash_key once
            if cur_tp > 0:
                return True
        hash_keys = [str_to_md5_bytes(k) for k in keys]

        try:
            found_idx = self.uc_store.lookup_on_prefix(hash_keys)
            return found_idx >= 0

        except RuntimeError as e:
            logger.error(f"[UC][exists] lookup exception: {e}")
            # current mindie do not handle exception
            return False

    @uc_timeit("batch_exist")
    def batch_exist(self, keys: List[str]) -> List[bool]:
        """
        Check whether current req's blocks are in store in batch way, and return the result in order.
        keys are ordered differently for different parallel strategy, for example:
            DPTP:  blk0tp0 blk0tp1 blk0tp2 blk0tp3 ... blk1tp0 blk1tp1 blk1tp2 blk1tp3 ...
            CPSP:  blk0 blk1 blk2 blk3 ...
        """
        if not isinstance(keys, list):
            logger.error(
                f"[UC][batch_exist] keys type should be List[str], got {type(keys)}"
            )
            return [False]

        n = len(keys)
        assert n > 0, f"[UC][batch_exist] keys list is empty"

        if self.bypass & BypassMode.EXISTS:
            return [False] * n

        try:
            if self.runtime_cfg.is_mla and self.runtime_cfg.scp_size == 1:
                tp_size = self.runtime_cfg.tp_size

                if tp_size <= 0 or n % tp_size != 0:
                    logger.error(f"[UC][batch_exist] invalid tp_size: {tp_size}")
                    return [False] * n

                # NOTE: keys layout MUST BE: blk0tp0 blk0tp1 ... blk1tp0 blk1tp1 ...
                block_tp0_keys = keys[::tp_size]

                hash_keys = [str_to_md5_bytes(k) for k in block_tp0_keys]
                found_idx = self.uc_store.lookup_on_prefix(hash_keys)

                num_blocks = len(block_tp0_keys)
                num_hit_blocks = found_idx + 1
                return [
                    block_idx < num_hit_blocks
                    for block_idx in range(num_blocks)
                    for _ in range(tp_size)
                ]

            hash_keys = [str_to_md5_bytes(k) for k in keys]
            found_idx = self.uc_store.lookup_on_prefix(hash_keys)

        except RuntimeError as e:
            logger.error(f"[UC][batch_exist] lookup exception: {e}")
            return [False] * n

        return [True] * (found_idx + 1) + [False] * (n - found_idx - 1)

    @uc_timeit("put")
    def put(
        self, keys: Union[str, List[str]], tensors: Union[torch.Tensor, List], **kwargs
    ) -> Any:
        """
        Put kvcache of MindIE npu-cache into store

        Args:
            keys (List[str]): mindie block prefix_key.
            tensors (Union[torch.Tensor, List]): mindie block prefix_key.
        """

        if self.bypass & BypassMode.PUT:
            return DUMP_OK

        if self.runtime_cfg.is_mla and self.runtime_cfg.scp_size == 1:
            tp0_keys, _ = self._get_tp_rank_0_hash_key(keys)
            keys = tp0_keys[self.tp_rank :: self.runtime_cfg.tp_size]
            if not keys:
                return DUMP_OK
            tensors = tensors[self.tp_rank :: self.runtime_cfg.tp_size]

        hash_keys = [str_to_md5_bytes(k) for k in keys]
        tasks = []
        try:
            shard_indexes = [0 for _ in range(len(hash_keys))]
            kv_tensors = self._get_tensors_for_store(tensors)
            task = self.uc_store.dump_data(hash_keys, shard_indexes, kv_tensors)
            if not self._check_task(task, "k_store"):
                return DUMP_ERROR
            tasks.append((self.uc_store, task, "k_store"))

        except RuntimeError as e:
            logger.error(f"[UC][put] dump exception: {e}")
            return DUMP_ERROR

        ret = DUMP_OK if self._wait_tasks(tasks) else DUMP_ERROR
        return ret

    @uc_timeit("get")
    def get(
        self, keys: Union[str, List[str]], tensors: Union[torch.Tensor, List], **kwargs
    ) -> Any:
        """
        Get kvcache from store for MindIE npu-cache

        Args:
            keys (List[str]): MindIE block prefix_key.
            tensors (Union[torch.Tensor, List]): tensors in MindIE npu-cache.
        """
        # bypass uc
        if self.bypass & BypassMode.GET:
            return LOAD_OK

        if self.runtime_cfg.is_mla and self.runtime_cfg.scp_size == 1:
            keys, _ = self._get_tp_rank_0_hash_key(keys)

        hash_keys = [str_to_md5_bytes(k) for k in keys]
        tasks = []

        try:
            shard_indexes = [0 for _ in range(len(hash_keys))]
            kv_tensors = self._get_tensors_for_store(tensors)
            task = self.uc_store.load_data(hash_keys, shard_indexes, kv_tensors)
            if not self._check_task(task, "k_store"):
                return LOAD_ERROR
            tasks.append((self.uc_store, task, "k_store"))

        except RuntimeError as e:
            logger.error(f"[UC][get] load exception: {e}")
            return LOAD_ERROR
        ret = LOAD_OK if self._wait_tasks(tasks) else LOAD_ERROR
        return ret
