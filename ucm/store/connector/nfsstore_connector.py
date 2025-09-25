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

from ucm.store.connector import ucmnfsstore
from ucm.store.connector.ucmstore import Task, UcmKVStoreBase


@dataclass
class NfsTask(Task):
    task_id: int


class UcmNfsStore(UcmKVStoreBase):
    def __init__(self, config: Dict):
        super().__init__(config)
        storage_backends = [
            path for path in config["storage_backends"].split(":") if path
        ]
        block_size = int(config["kv_block_size"])
        transfer_enable = True if config["role"] == "worker" else False
        param = ucmnfsstore.NfsStore.Config(
            storage_backends, block_size, transfer_enable
        )
        if transfer_enable:
            param.transferDeviceId = config["device"]
            param.transferIoSize = config["io_size"]
        ret = ucmnfsstore.Setup(param)
        if ret != 0:
            msg = f"Failed to initialize ucmnfsstore, errcode: {ret}."
            raise RuntimeError(msg)

    def cc_store(self) -> int:
        return ucmnfsstore.CCStoreImpl()

    def create(self, block_ids: List[str]) -> List[int]:
        return ucmnfsstore.AllocBatch(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        return ucmnfsstore.LookupBatch(block_ids)

    def prefetch(self, block_ids: List[str]) -> None:
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        dst_tensor_ptr = [t.data_ptr() for t in dst_tensor]
        dst_tensor_size = [t.numel() * t.element_size() for t in dst_tensor]
        task_id = ucmnfsstore.LoadToDevice(
            block_ids, offset, dst_tensor_ptr, dst_tensor_size
        )
        return NfsTask(task_id=task_id)

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        src_tensor_ptr = [t.data_ptr() for t in src_tensor]
        src_tensor_size = [t.numel() * t.element_size() for t in src_tensor]
        task_id = ucmnfsstore.DumpFromDevice(
            block_ids, offset, src_tensor_ptr, src_tensor_size
        )
        return NfsTask(task_id=task_id)

    def wait(self, task: Task) -> int:
        return ucmnfsstore.Wait(task.task_id)

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        ucmnfsstore.CommitBatch(block_ids, is_success)

    def check(self, task: Task) -> Tuple[int, bool]:
        return ucmnfsstore.Check(task.task_id)
