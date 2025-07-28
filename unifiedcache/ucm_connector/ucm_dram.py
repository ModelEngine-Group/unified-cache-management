import torch
from dataclasses import dataclass
from typing import List, Dict, Optional

from unifiedcache.logger import init_logger
from unifiedcache.ucm_connector import Task, UcmKVStoreBase

logger = init_logger(__name__)

SUCCESS = 0
FAILURE = -1

@dataclass
class DramTask(Task):
    task_id: str = '1'
    event: Optional[torch.npu.Event] = None


class UcmDram(UcmKVStoreBase):
    """
    Dram Connector
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.dram_cache: Dict[str, any] = {}
        self.max_cache_byte = int(config["max_cache_size"])
        self.kv_block_size = config["kv_block_size"]
        self.max_block_num = self.max_cache_byte//self.kv_block_size
        if config["role"] == "scheduler":
            self.cached_blocks = set()
    
    def create(self, block_ids: List[str]) -> int:
        """
        create kv cache space in storafe

        Args:
            block_ids (List[str]): vLLM block hash.
        Returns:
            success mask
        """
        return SUCCESS

    def lookup(self, block_ids: List[str]) -> List[bool]:
        """
        Get number of blocks that can be loaded from the
        external KV cache.

        Args:
            block_ids (List[str]): vLLM block hash.

        Returns:
            hit block mask, True -> hit
        """
        hit_list = [block_id in self.cached_blocks for block_id in block_ids]
        return hit_list

    def prefetch(self, block_ids: List[str]) -> None:
        """
        prefetch kv cache to high speed cache according to block_ids.

        Args:
            block_ids (List[str]): vLLM block hash.
        """
        pass

    def load(self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]) -> Task:
        """
        load kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            dst_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        task = DramTask()
        stream = torch.npu.Stream()
        task.event = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(stream):
            for i, block_id in enumerate(block_ids):
                key = block_id + '_' + str(offset[i])
                dst_tensor[i].copy_(self.dram_cache[key], non_blocking=True)
            task.event.record(stream=stream)
        logger.debug(f"load block {block_ids} finished.")
        return task

    def dump(self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]) -> Task:
        """
        dump kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        task = DramTask()
        if len(self.dram_cache) > self.max_block_num:
            logger.warning("Dram cache usage exceeds limit! No more kv cache offload! Try to increase your initial max_cache_size.")
            task.task_id = "-1"
            return task
        else:
            torch.npu.current_stream().synchronize()
            stream = torch.npu.Stream()
            task.event = torch.npu.Event(enable_timing = True)
            with torch.npu.stream(stream):
                for i, block_id in enumerate(block_ids):
                    key = block_id + '_' + str(offset[i])
                    self.dram_cache[key] = src_tensor[i].cpu()
                task.event.record(stream=stream)
        logger.debug(f"dump block {block_ids} finished.")
        return task

    def wait(self, task: DramTask) -> int:
        """
        wait kv cache kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - success
            others - failed.
        """
        if task.task_id == "-1":
            logger.warning("Dump failure with full cache usage!")
            return FAILURE
        try:
            event = task.event
            event.synchronize()
            return SUCCESS
        except Exception as e:
            logger.error(f"Error waiting cache for block IDs: {e}")
            return FAILURE

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        """
        commit kv cache, now kv cache can be reused.

        Args:
            block_ids (List[str]): vLLM block hash.
            is_success(bool): if False, we need release block
        """
        if is_success:
            self.cached_blocks.update(block_ids)