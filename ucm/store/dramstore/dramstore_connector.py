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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ucm.logger import init_logger
from ucm.store.dramstore import ucmdramstore
from ucm.store.ucmstore import Task, UcmKVStoreBase

logger = init_logger(__name__)

SUCCESS = 0
FAILURE = -1

if torch.cuda.is_available():
    device = torch.cuda
elif hasattr(torch, "npu") and torch.npu.is_available():
    device = torch.npu
else:
    raise RuntimeError(
        "No supported accelerator found. "
        "Please ensure either CUDA or NPU is available."
    )


@dataclass
class DramTask(Task):
    task_id: int
    # task_id: str = "1"
    # event: Optional[Any] = None


class UcmDramStore(UcmKVStoreBase):
    """
    Dram Connector
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.store = ucmdramstore.DRAMStore()

        capacity = int(config.get("capacity", 1073741824))  # Default 1GB
        block_size = int(config.get("kv_block_size", 262144))  # Default 256KB
        stream_number = int(config.get("stream_number", 32))
        timeout_ms = int(config.get("timeout_ms", 30000))

        param = ucmdramstore.DRAMStore.Config(
            capacity, block_size, stream_number, timeout_ms
        )

        ret = self.store.Setup(param)
        if ret != 0:
            msg = f"Failed to initialize ucmdramstore, errcode: {ret}."
            raise RuntimeError(msg)

    def cc_store(self) -> int:
        return self.store.CCStoreImpl()

    def create(self, block_ids: List[str]) -> List[int]:
        return self.store.AllocBatch(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        return self.store.LookupBatch(block_ids)

    def prefetch(self, block_ids: List[str]) -> None:
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        dst_tensor_ptr = [t.data_ptr() for t in dst_tensor]
        dst_tensor_size = [t.numel() * t.element_size() for t in dst_tensor]
        task_id = self.store.Load(block_ids, offset, dst_tensor_ptr, dst_tensor_size)
        return DramTask(task_id=task_id)

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        src_tensor_ptr = [t.data_ptr() for t in src_tensor]
        src_tensor_size = [t.numel() * t.element_size() for t in src_tensor]
        task_id = self.store.Dump(block_ids, offset, src_tensor_ptr, src_tensor_size)
        return DramTask(task_id=task_id)

    def fetch_data(
        self,
        block_ids: List[str],
        offset: List[int],
        dst_addr: List[int],
        size: List[int],
    ) -> Task:
        task_id = self.store.Load(block_ids, offset, dst_addr, size)
        return DramTask(task_id=task_id)

    def dump_data(
        self,
        block_ids: List[str],
        offset: List[int],
        src_addr: List[int],
        size: List[int],
    ) -> Task:
        task_id = self.store.Dump(block_ids, offset, src_addr, size)
        return DramTask(task_id=task_id)

    def wait(self, task: DramTask) -> int:
        return self.store.Wait(task.task_id)

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        self.store.CommitBatch(block_ids, is_success)

    def check(self, task: Task) -> int:
        return self.store.Check(task.task_id)
