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
import os
import secrets
import time
from typing import List

import cupy
import numpy as np

from ucm.store.ds3fs.connector import UcmDs3fsStore
from ucm.store.ucmstore import Task


class Ds3fsStoreOnly:
    def __init__(
        self,
        block_size: int,
        storage_backends: List[str],
    ):
        ds3fs_config = {}
        ds3fs_config["storage_backends"] = storage_backends
        ds3fs_config["device_id"] = 0
        ds3fs_config["tensor_size"] = block_size
        ds3fs_config["shard_size"] = block_size
        ds3fs_config["block_size"] = block_size
        ds3fs_config["io_direct"] = True
        ds3fs_config["stream_number"] = 16
        self.ds3fs = UcmDs3fsStore(ds3fs_config)

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        return self.ds3fs.lookup(block_ids)

    def prefetch(self, block_ids: List[bytes]) -> None:
        return self.ds3fs.prefetch(block_ids)

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]],
    ) -> Task:
        return self.ds3fs.load_data(block_ids, shard_index, dst_addr)

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]],
    ) -> Task:
        return self.ds3fs.dump_data(block_ids, shard_index, src_addr)

    def wait(self, task: Task) -> None:
        return self.ds3fs.wait(task)

    def check(self, task: Task) -> bool:
        return self.ds3fs.check(task)


def e2e_test(
    store: Ds3fsStoreOnly,
    block_size: int,
    block_num: int,
):
    block_ids = [secrets.token_bytes(16) for _ in range(block_num)]

    founds = store.lookup(block_ids)
    assert not all(founds), "Blocks should not exist before dump"

    shard_indexes = [0 for _ in range(block_num)]

    src_data = []
    src_arrays = []
    src_mems = []
    for i in range(block_num):
        mem = cupy.cuda.alloc_pinned_memory(block_size)
        arr = np.frombuffer(mem, dtype=np.uint8, count=block_size)
        arr.flags.writeable = True
        arr[:] = np.random.randint(0, 256, block_size, dtype=np.uint8)
        src_data.append([mem.ptr])
        src_arrays.append(arr.copy())
        src_mems.append(mem)

    task = store.dump_data(block_ids, shard_indexes, src_data)
    store.wait(task)

    founds = store.lookup(block_ids)
    assert all(founds), "Blocks should exist after dump"

    dst_data = []
    dst_arrays = []
    dst_mems = []
    for i in range(block_num):
        mem = cupy.cuda.alloc_pinned_memory(block_size)
        arr = np.frombuffer(mem, dtype=np.uint8, count=block_size)
        arr.flags.writeable = True
        arr[:] = 0
        dst_data.append([mem.ptr])
        dst_arrays.append(arr)
        dst_mems.append(mem)

    task = store.load_data(block_ids, shard_indexes, dst_data)
    store.wait(task)

    for i, (src_arr, dst_arr) in enumerate(zip(src_arrays, dst_arrays)):
        if not np.array_equal(src_arr, dst_arr):
            diff_mask = src_arr != dst_arr
            num_diff = diff_mask.sum()
            print(f"DIFF at block {i}: {num_diff} bytes differ")
            print(f"  src sample: {src_arr[diff_mask][:10]}")
            print(f"  dst sample: {dst_arr[diff_mask][:10]}")
            assert False, f"Data mismatch at block {i}"


def main():
    block_size = 1048576 * 4
    block_num = 1024
    storage_backends = ["."]
    test_batch_number = 64

    store = Ds3fsStoreOnly(block_size, storage_backends)

    for i in range(test_batch_number):
        e2e_test(store, block_size, block_num)

    time.sleep(10)


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "debug"
    main()
