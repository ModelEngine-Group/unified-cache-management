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
import array
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from ucm.store.cache import ucmcachestore
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1


@dataclass
class CacheTransTask(Task):
    task_id: int


class UcmCacheStore(UcmKVStoreBaseV1):
    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__(config)
        key_mapping = {
            "store_backend": "storeBackend",
            "unique_id": "uniqueId",
            "device_id": "deviceId",
            "tensor_size": "tensorSize",
            "shard_size": "shardSize",
            "block_size": "blockSize",
            "buffer_size": "bufferSize",
            "share_buffer_enable": "shareBufferEnable",
            "waiting_queue_depth": "waitingQueueDepth",
            "running_queue_depth": "runningQueueDepth",
            "timeout_ms": "timeoutMs",
        }
        self.store = ucmcachestore.CacheStore()
        param = ucmcachestore.CacheStore.Config()
        for key, value in config.items():
            attr = key_mapping.get(key)
            if attr and hasattr(param, attr):
                setattr(param, attr, value)
        self.store.Setup(param)

    def cc_store(self) -> int:
        return self.store.Self()

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        flat = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        res = self.store.Lookup(flat)
        return np.frombuffer(res, dtype=bool)

    def prefetch(self, block_ids: List[bytes]) -> None:
        flat = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        self.store.Prefetch(flat)

    def _tensor_normalize(self, tensors: List[List[torch.Tensor]]) -> np.ndarray:
        n_rows = len(tensors)
        n_cols = len(tensors[0])
        flat = np.fromiter(
            (t for row in tensors for t in row), dtype=object, count=n_rows * n_cols
        )
        ptrs = np.vectorize(torch.Tensor.data_ptr, otypes=[np.uint64])(flat)
        return ptrs.reshape(n_rows, n_cols)

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        addrs = self._tensor_normalize(dst_tensor)
        task_id = self.store.Load(ids, indexes, addrs)
        return CacheTransTask(task_id)

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        addrs = self._tensor_normalize(src_tensor)
        task_id = self.store.Dump(ids, indexes, addrs)
        return CacheTransTask(task_id)

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]],
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        addrs = np.array(dst_addr, np.uint64)
        task_id = self.store.Load(ids, indexes, addrs)
        return CacheTransTask(task_id)

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]],
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        addrs = np.array(src_addr, np.uint64)
        task_id = self.store.Dump(ids, indexes, addrs)
        return CacheTransTask(task_id)

    def wait(self, task: Task) -> None:
        return self.store.Wait(task.task_id)

    def check(self, task: Task) -> bool:
        return self.store.Check(task.task_id)
