# -*- coding: utf-8 -*-
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
from typing import Dict, List, Tuple

import torch

from ucm.store.pcstore import ucmpcstore
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1


@dataclass
class PcTask(Task):
    task_id: int


class UcmPcStore(UcmKVStoreBaseV1):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.store = ucmpcstore.PcStore()
        storage_backends = [
            path for path in config["storage_backends"].split(":") if path
        ]
        block_size = config.get("kv_block_size", 33554432)
        transfer_enable = True if config["role"] == "worker" else False
        param = ucmpcstore.PcStore.Config(storage_backends, block_size, transfer_enable)
        if transfer_enable:
            param.transferDeviceId = config["device"]
            param.transferIoSize = config["io_size"]
            param.transferIoDirect = config.get("use_direct", False)
            param.transferStreamNumber = config.get("stream_number", 8)
            param.transferBufferNumber = config.get("buffer_number", 4096)
            param.transferScatterGatherEnable = config.get("use_scatter_gatter", False)
        ret = self.store.Setup(param)
        if ret != 0:
            msg = f"Failed to initialize ucmpcstore, errcode: {ret}."
            raise RuntimeError(msg)

    def cc_store(self) -> int:
        return self.store.CCStoreImpl()

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        return self.store.LookupBatch(block_ids)

    def prefetch(self, block_ids: List[bytes]) -> None:
        pass

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        dst_tensor_ptrs = [[t.data_ptr() for t in tensors] for tensors in dst_tensor]
        task_id = self.store.LoadToDevice(block_ids, dst_tensor_ptrs)
        return PcTask(task_id=task_id)

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        src_tensor_ptrs = [[t.data_ptr() for t in tensors] for tensors in src_tensor]
        task_id = self.store.DumpFromDevice(block_ids, src_tensor_ptrs)
        return PcTask(task_id=task_id)

    def fetch_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]],
    ) -> Task:
        task_id = self.store.LoadToDevice(block_ids, dst_addr)
        return task_id

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]],
    ) -> Task:
        task_id = self.store.DumpFromDevice(block_ids, src_addr)
        return task_id

    def wait(self, task: Task) -> None:
        ret = self.store.Wait(task.task_id)
        if ret != 0:
            raise RuntimeError(f"Wait failed for task {task.task_id}, return={ret}")

    def check(self, task: Task) -> bool:
        return self.store.Check(task.task_id)
