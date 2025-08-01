#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from unifiedcache.logger import init_logger
from unifiedcache.ucm_connector import Task, UcmKVStoreBase

logger = init_logger(__name__)

SUCCESS = 0
FAILURE = -1

if torch.cuda.is_available():
    device = torch.cuda
elif hasattr(torch, 'npu') and torch.npu.is_available():
    device = torch.npu
else:
    raise RuntimeError(
        "No supported accelerator found. "
        "Please ensure either CUDA or NPU is available."
    )


@dataclass
class DramTask(Task):
    task_id: str = '1'
    event: Optional[Any] = None


class UcmDram(UcmKVStoreBase):
    """
    Dram Connector
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.dram_cache: Dict[str, any] = {}
        self.max_cache_byte = int(config.get("max_cache_size", 5368709120))
        self.kv_block_size = int(config.get("kv_block_size", 262144))
        self.max_block_num = self.max_cache_byte // self.kv_block_size
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
        stream = device.Stream()
        task.event = device.Event(enable_timing=True)
        with device.stream(stream):
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
            logger.warning(
                "Dram cache usage exceeds limit! No more kv cache offload! Try to increase your initial max_cache_size.")
            task.task_id = "-1"
            return task
        else:
            device.current_stream().synchronize()
            stream = device.Stream()
            task.event = device.Event(enable_timing=True)
            with device.stream(stream):
                for i, block_id in enumerate(block_ids):
                    key = block_id + '_' + str(offset[i])
                    self.dram_cache[key] = src_tensor[i].to('cpu', non_blocking=True)
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
