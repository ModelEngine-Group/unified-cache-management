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
from typing import List, Dict
from unifiedcache.logger import init_logger
from unifiedcache.ucm_connector import Task, UcmKVStoreBase
from unifiedcache.ucm_connector import ucmlocalstore

logger = init_logger(__name__)


@dataclass
class LocalTask(Task):
    task_id: int

    def get_id(self) -> int:
        return self.task_id

class UcmLocalStore(UcmKVStoreBase):
    """
    Local connector
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        capacity = int(config["capacity"])
        cache_size = int(config["cacheSize"])
        device_id = int(config["device"])
        param = ucmlocalstore.SetupParam(capacity, cache_size)
        param.deviceId = device_id
        ret = ucmlocalstore.Setup(param)
        if ret != 0:
            msg = f"Failed to initialize ucmlocalstore, errcode: {ret}."
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.info("Succeed in initializing ucmlocalstore.")

    def create(self, block_ids: List[str]) -> int:
        """
        create kv cache space in storage

        Args:
            block_ids (List[str]): vLLM block hash.
        Returns:
            success mask
        """
        logger.info("create finished.")

    def lookup(self, block_ids: List[str]) -> List[bool]:
        """
        Get number of blocks that can be loaded from the
        external KV cache.

        Args:
            block_ids (List[str]): vLLM block hash.

        Returns:
            hit block mask, True -> hit
        """
        ret = ucmlocalstore.Lookup(block_ids)
        logger.info("Succeed in looking up kv cache blocks.")
        return ret

    def prefetch(self, block_ids: List[str]) -> None:
        """
        prefetch kv cache to high speed cache according to block_ids.

        Args:
            block_ids (List[str]): vLLM block hash.
        """
        # TODO
        logger.info("prefetch finished.")

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
        dst_tensor_ptr = [t.data_ptr() for t in dst_tensor]
        dst_tensor_size = [t.numel() * t.element_size() for t in dst_tensor]
        task_id = ucmlocalstore.ReadFromDram(block_ids, offset, dst_tensor_ptr, dst_tensor_size)
        logger.debug(f"Succeed in loading kv cache, task id: {task_id}, offset {offset}.")
        return LocalTask(task_id=task_id)


    def dump(self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]) -> Task:
        """
        dump kv cache to dram.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        src_tensor_ptr = [t.data_ptr() for t in src_tensor]
        task_id = ucmlocalstore.WriteToDram(block_ids, src_tensor_ptr)
        logger.debug(f"Succeed in dumping kv cache, task id: {task_id}.")
        return LocalTask(task_id=task_id)

    def wait(self, task: Task) -> int:
        """
        wait kv cache kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - success
            others - failed.
        """
        if not isinstance(task, LocalTask):
            logger.error("This is not LocalTask")
            return -1
        ret = ucmlocalstore.Wait(task.task_id)
        if ret != 0:
            logger.error(f"Failed to wait for kv cache transfer task, errcode: {ret}.")
        else:
            logger.debug("Succeed in waiting for kv cache transfer task.")
        return ret

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        """
        commit kv cache, now kv cache can be reused.

        Args:
            block_ids (List[str]): vLLM block hash.
            is_success(bool): if False, we need release block
        """
        logger.debug("commit finished.")
