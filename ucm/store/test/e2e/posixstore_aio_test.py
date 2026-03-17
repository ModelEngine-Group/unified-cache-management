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
import mmap
import secrets
import time

import numpy as np

from ucm.store.factory_v1 import UcmConnectorFactoryV1, UcmKVStoreBaseV1


def create_store(shard_size: int, block_size: int, worker: bool) -> UcmKVStoreBaseV1:
    module_path = "ucm.store.pipeline.connector"
    class_name = "UcmPipelineStore"
    config = {}
    config["store_pipeline"] = "Posix"
    config["storage_backends"] = ["./build/data"]
    config["posix_io_engine"] = "aio"
    config["tensor_size"] = shard_size
    config["shard_size"] = shard_size
    config["block_size"] = block_size
    config["device_id"] = 0 if worker else -1
    return UcmConnectorFactoryV1.create_connector(class_name, config, module_path)


def make_array(size, alignment=262144, dtype=np.uint8) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    total_bytes = size * itemsize
    mm = mmap.mmap(-1, total_bytes + alignment)
    raw_array = np.frombuffer(mm, dtype=np.uint8, count=total_bytes + alignment)
    raw_ptr = raw_array.__array_interface__["data"][0]
    aligned_addr = (raw_ptr + alignment - 1) & ~(alignment - 1)
    offset = aligned_addr - raw_ptr
    array = raw_array[offset : offset + total_bytes].view(dtype=dtype)
    return array


def dump(worker, block_ids, block_ptr, block_number, shard_size, shard_number):
    total_size = shard_size * shard_number * block_number
    tp = time.perf_counter()
    tasks = []
    for i in range(shard_number):
        idxes = [i for _ in range(block_number)]
        ptrs = [[ptr + i * shard_size] for ptr in block_ptr]
        tasks.append(worker.dump_data(block_ids, idxes, ptrs))
    for task in tasks:
        worker.wait(task)
    cost = time.perf_counter() - tp
    print(
        f"Dump [{shard_size} x {shard_number} x {block_number}]: "
        f"cost={cost * 1e3:.3f}ms, bw={total_size / cost / 1e9:.3f}GB/s."
    )


def load(worker, block_ids, block_ptr, block_number, shard_size, shard_number):
    total_size = shard_size * shard_number * block_number
    tp = time.perf_counter()
    tasks = []
    for i in range(shard_number):
        idxes = [i for _ in range(block_number)]
        ptrs = [[ptr + i * shard_size] for ptr in block_ptr]
        tasks.append(worker.load_data(block_ids, idxes, ptrs))
    for task in tasks:
        worker.wait(task)
    cost = time.perf_counter() - tp
    print(
        f"Load [{shard_size} x {shard_number} x {block_number}]: "
        f"cost={cost * 1e3:.3f}ms, bw={total_size / cost / 1e9:.3f}GB/s."
    )


if __name__ == "__main__":
    shard_size = 256 * 1024
    shard_number = 16
    block_size = shard_size * shard_number
    block_number = 4096
    epoch_number = 32
    worker = create_store(shard_size, block_size, True)
    block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
    block_data = [make_array(block_size) for _ in range(block_number)]
    block_ptr = [block.ctypes.data for block in block_data]
    for _ in range(epoch_number):
        dump(worker, block_ids, block_ptr, block_number, shard_size, shard_number)
    for _ in range(epoch_number):
        load(worker, block_ids, block_ptr, block_number, shard_size, shard_number)
