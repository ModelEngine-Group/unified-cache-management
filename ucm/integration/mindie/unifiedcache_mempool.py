import fcntl
import functools
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Union

import torch
from mindie_llm.utils.file_utils import safe_open
from mindie_llm.utils.log.logging import logger

from .base import MemPool

WORK_ROLE = "worker"
SCHEDULER_ROLE = "scheduler"
LOAD_OK = 0
DUMP_OK = 0
LOAD_ERROR = -1
DUMP_ERROR = -1
TASK_INVALID = 0

_ENABLE_TIME_STAT = os.getenv("MINDIE_UC_TIME_STAT", "0") == "1"


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


@dataclass
class UnifiedCacheConfig:
    storage_backends: str = ""
    mla_store_uid: str = ""
    is_mla: bool = False
    share_buffer_enable: bool = False

    scp_size: int = 1
    cp_size: int = 1
    dp_size: int = 1
    sp_size: int = 1
    tp_size: int = 1

    @classmethod
    def parse_config(cls, config_path: str):

        with safe_open(config_path) as fin:
            uc_config = json.load(fin)

        storage_backends = uc_config.get("storage_backends")

        mindie_config_path = uc_config.get("mindie_config_path")
        with safe_open(mindie_config_path) as fin:
            mindie_config = json.load(fin)
        BackendConfig = mindie_config.get("BackendConfig", None)
        if BackendConfig is None:
            BackendConfig = mindie_config.get("mindie_server_prefill_config").get(
                "BackendConfig"
            )

        schedule_config = BackendConfig.get("ScheduleConfig")

        blocksize = schedule_config.get("cacheBlockSize", 128)
        maxPrefillTokens = schedule_config.get("maxPrefillTokens", 32768)
        if not maxPrefillTokens:
            logger.error(f"[UC] please set maxPrefillTokens in mindie config")
        # current only support single model
        model_config = BackendConfig.get("ModelDeployConfig").get("ModelConfig")[0]

        dp = model_config.get("dp", 1)
        tp = model_config.get("tp", 1)
        sp = model_config.get("sp", 1)
        cp = model_config.get("cp", 1)
        world_size = model_config.get("worldSize", 1)
        scp_size = sp * cp
        if tp * dp == 1 or tp * cp == 1:
            # tp is default set to world_size
            tp = world_size

        if cp > 1:
            world_size = cp * tp

        if dp > 1:
            world_size = dp * tp

        model_weight_path = model_config.get("modelWeightPath")
        model_weight_path = (
            model_weight_path
            if model_weight_path.endswith("/")
            else model_weight_path + "/"
        )
        with safe_open(model_weight_path + "config.json") as fin:
            model_arc_config = json.load(fin)

        # should be same as the local_world_size in worker process command,
        # eg: mindie_llm_backend --local_rank 0 --local_world_size 2

        # dp only support 1 and 2
        # for two node with 16 cards,
        mla_store_uid = get_dual_consensus_uids()

        share_buffer_enable = False
        is_mla = False
        if "kv_lora_rank" in model_arc_config.keys():
            # for deepseek model
            is_mla = True
            if scp_size == 1:
                share_buffer_enable = True

        return UnifiedCacheConfig(
            scp_size=scp_size,
            dp_size=dp,
            cp_size=cp,
            sp_size=sp,
            tp_size=tp,
            storage_backends=storage_backends,
            is_mla=is_mla,
            mla_store_uid=mla_store_uid,
            share_buffer_enable=share_buffer_enable,
        )


class UnifiedCacheMempool(MemPool):

    def __init__(self, config_path, role, **kwargs):
        device_id = kwargs.get("device_id", -1)
        kv_caches = kwargs.get("kv_caches", None)
        try:
            kv_store_config = self._get_store_config(config_path, device_id, kv_caches)
        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise
        from ucm.store.pipeline.connector import UcmPipelineStore

        self.uc_store = UcmPipelineStore(kv_store_config)

        logger.info("[UC]: Initialize unifiedcache success.")
        # current for attn_tp strategy of mla, we need to get current tp rank
        self.tp_rank = -1

        bypass_val = int(os.getenv("BYPASS_UC", "0"))
        self.bypass_exists = bool(bypass_val & 0b100)  # 或 & 4
        self.bypass_put = bool(bypass_val & 0b010)  # 或 & 2
        self.bypass_get = bool(bypass_val & 0b001)  # 或 & 1
        logger.info(
            f"[UC]: bypass_exists {self.bypass_exists} bypass_put {self.bypass_put} bypass_get {self.bypass_get}"
        )

    def _get_store_default_config(
        self, storage_backends, unique_id, device_id, share_buffer_enable
    ):
        store_config = {
            "store_pipeline": "Cache|Posix",
            "storage_backends": storage_backends,
            "unique_id": unique_id,
            "device_id": device_id,
            "timeout_ms": 30000,
            "tensor_size_list": [0],
            "shard_size": 0,
            "block_size": 0,
            # for cache store
            "cache_buffer_capacity_gb": 64,
            "cache_stream_number": 32,
            "share_buffer_enable": share_buffer_enable,
            "waiting_queue_depth": 16,
            "running_queue_depth": 1024,
            # for posix store
            "posix_data_trans_concurrency": 32,
            "posix_lookup_concurrency": 8,
            "io_direct": False,
        }
        return store_config

    def _get_store_config(self, config_path, device_id, kv_caches):
        uc_config = UnifiedCacheConfig.parse_config(config_path)
        logger.info(f"[UC] parsed unified cache config: {str(uc_config)}")
        self.uc_config = uc_config
        store_config = self._get_store_default_config(
            uc_config.storage_backends,
            uc_config.mla_store_uid,
            device_id,
            uc_config.share_buffer_enable,
        )
        if kv_caches is not None:
            k_tensor = kv_caches[0][0][0]
            v_tensor = kv_caches[0][1][0]
            k_io_size = k_tensor.element_size() * k_tensor.numel()
            v_io_size = v_tensor.element_size() * v_tensor.numel()
            self.num_layers = len(kv_caches)
            store_config["tensor_size_list"] = [k_io_size] * self.num_layers + [
                v_io_size
            ] * self.num_layers
            store_config["shard_size"] = sum(store_config["tensor_size_list"])
            store_config["block_size"] = store_config["shard_size"]

        logger.info(f"[UC] pipeline store config: {str(store_config)}")
        return store_config

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
        if self.bypass_exists:
            return False

        if isinstance(keys, str):
            keys = [keys]

        if self.uc_config.is_mla and self.uc_config.scp_size == 1:
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

        if self.bypass_exists:
            return [False] * n

        try:
            if self.uc_config.is_mla and self.uc_config.scp_size == 1:
                tp_size = self.uc_config.tp_size

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

        if self.bypass_put:
            return DUMP_OK

        if self.uc_config.is_mla and self.uc_config.scp_size == 1:
            tp0_keys, _ = self._get_tp_rank_0_hash_key(keys)
            keys = tp0_keys[self.tp_rank :: self.uc_config.tp_size]
            if not keys:
                return DUMP_OK
            tensors = tensors[self.tp_rank :: self.uc_config.tp_size]

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
        if self.bypass_get:
            return LOAD_OK

        if self.uc_config.is_mla and self.uc_config.scp_size == 1:
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
