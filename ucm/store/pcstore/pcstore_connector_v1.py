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

import numpy as np
import torch

from ucm.store.pcstore import ucmpcstore
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1


@dataclass
class UcmPcTask(Task):
    task_id: int


class UcmPcStoreV1(UcmKVStoreBaseV1):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.store = ucmpcstore.PcStore()
        storage_backends = config["storage_backends"]
        block_size = config.get("block_size", 0)
        transfer_enable = True if int(config["device_id"]) >= 0 else False
        param = ucmpcstore.PcStore.Config(storage_backends, block_size, transfer_enable)
        key_mapping = {
            "unique_id": "uniqueId",
            "io_direct": "transferIoDirect",
            "local_rank_size": "transferLocalRankSize",
            "device_id": "transferDeviceId",
            "stream_number": "transferStreamNumber",
            "tensor_size": "transferIoSize",
            "buffer_number": "transferBufferNumber",
            "timeout_ms": "transferTimeoutMs",
            "use_scatter_gather": "transferScatterGatherEnable",
        }
        for key, value in config.items():
            attr = key_mapping.get(key)
            if attr and hasattr(param, attr):
                setattr(param, attr, value)
        ret = self.store.Setup(param)
        if ret != 0:
            msg = f"Failed to initialize ucmpcstore, errcode: {ret}."
            raise RuntimeError(msg)

    def cc_store(self) -> int:
        """Return a low-level C/C++ pointer to the underlying store.

        Returns:
            An opaque ``int`` representing the ``Store*`` instance that can
            be passed to native code.
        """
        return self.store.CCStoreImpl()

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        """Check presence of blocks in external storage.

        Args:
            block_ids: List of vLLM block hashes (raw bytes).

        Returns:
            A list of booleans, ``True`` if the corresponding block exists in
            storage, ``False`` otherwise. The order matches ``block_ids``.
        """
        return self.store.LookupBatch(block_ids)

    def prefetch(self, block_ids: List[bytes]) -> None:
        """Asynchronously prefetch blocks into high-speed cache.

        Args:
            block_ids: List of vLLM block hashes to prefetch.
        """
        pass

    def _flatten_tensor_ptrs(
        self, block_ids: List[bytes], tensors: List[List[torch.Tensor]]
    ) -> Tuple[List[bytes], List[int]]:
        n_blocks = len(block_ids)
        m_addrs = len(tensors[0])
        total_len = n_blocks * m_addrs
        flat_ids = [None] * total_len
        flat_ptrs = [None] * total_len
        ids_arr = flat_ids
        ptrs_arr = flat_ptrs
        data_ptr_method = torch.Tensor.data_ptr
        idx = 0
        for bid, tensor_list in zip(block_ids, tensors):
            for t in tensor_list:
                ids_arr[idx] = bid
                ptrs_arr[idx] = data_ptr_method(t)
                idx += 1
        return flat_ids, flat_ptrs

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        """Initiate transfer of KV cache from storage to device.

        Args:
            block_ids: Hashes of the blocks to load.
            shard_index: Shard index for each block.
            dst_tensor: Double-list structure where ``dst_tensor[i][j]`` is the
                destination PyTorch tensor on device for block ``i``, tensor ``j``.

        Returns:
            A ``Task`` handle that can be used to check or wait for completion.
        """
        ids, addrs = self._flatten_tensor_ptrs(block_ids, dst_tensor)
        task_id = self.store.LoadToDevice(ids, addrs)
        return UcmPcTask(task_id=task_id)

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        """Initiate transfer of KV cache from device to storage.

        Args:
            block_ids: Hashes of the blocks to write.
            shard_index: Shard index for each block.
            src_tensor: Double-list structure where ``src_tensor[i][j]`` is the
                source PyTorch tensor on device for block ``i``, tensor ``j``.

        Returns:
            A ``Task`` handle that can be used to check or wait for completion.
        """
        ids, addrs = self._flatten_tensor_ptrs(block_ids, src_tensor)
        task_id = self.store.DumpFromDevice(ids, addrs)
        return UcmPcTask(task_id=task_id)

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]],
    ) -> Task:
        """Low-level fetch: copy KV data to device pointers.

        Args:
            block_ids: Block hashes to load.
            shard_index: Shard index for each block.
            dst_addr: Double-list of ``int`` pointers (as Python ``int``) to
                pre-allocated device buffers.

        Returns:
            A ``Task`` handle for the asynchronous copy.
        """
        block_ids_np = np.array(block_ids, dtype=object)
        dst_addr_np = np.array(dst_addr, dtype=int)
        ids = np.repeat(block_ids_np, dst_addr_np.shape[1])
        addrs = dst_addr_np.ravel()
        task_id = self.store.LoadToDevice(ids, addrs)
        return UcmPcTask(task_id=task_id)

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]],
    ) -> Task:
        """Low-level dump: copy KV data from device pointers.

        Args:
            block_ids: Block hashes to store.
            shard_index: Shard index for each block.
            src_addr: Double-list of ``int`` pointers to device buffers.

        Returns:
            A ``Task`` handle for the asynchronous copy.
        """
        block_ids_np = np.array(block_ids, dtype=object)
        src_addr_np = np.array(src_addr, dtype=int)
        ids = np.repeat(block_ids_np, src_addr_np.shape[1])
        addrs = src_addr_np.ravel()
        task_id = self.store.DumpFromDevice(ids, addrs)
        return UcmPcTask(task_id=task_id)

    def wait(self, task: Task) -> None:
        """Block until the given transfer task completes.

        Args:
            task: Task handle returned by ``load``, ``dump``, ``load_data``,
                or ``dump_data``.
        """
        ret = self.store.Wait(task.task_id)
        if ret != 0:
            msg = f"Failed to wait task({task.task_id}), errcode: {ret}."
            raise RuntimeError(msg)

    def check(self, task: Task) -> bool:
        """Non-blocking poll for task completion.

        Args:
            task: Task handle returned by any transfer method.

        Returns:
            ``True`` if the task has finished, ``False`` if still in-flight.
        """
        return self.store.Check(task.task_id)
