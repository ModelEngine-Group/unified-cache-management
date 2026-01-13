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

import numpy as np

from ucm.store.posix.connector import UcmKVStoreBaseV1, UcmPosixStore


def setup(
    backends: list[str],
    block_size: int,
    stream_number: int,
    io_direct: bool,
    io_async: bool,
    worker: bool,
) -> UcmKVStoreBaseV1:
    config = {}
    config["storage_backends"] = backends
    config["tensor_size"] = block_size
    config["shard_size"] = block_size
    config["block_size"] = block_size
    config["stream_number"] = stream_number
    config["io_direct"] = io_direct
    config["io_async"] = io_async
    config["device_id"] = 0 if worker else -1
    return UcmPosixStore(config)


def aligned_array(size, alignment=4096, dtype=np.uint8):
    extra = alignment
    buf = np.empty(size + extra, dtype=dtype)
    address = buf.ctypes.data
    offset = (alignment - (address % alignment)) % alignment
    aligned_view = buf[offset : offset + size]
    return aligned_view


def main():
    backends = ["./build/data"]
    block_size = 1048576
    stream_number = 8
    io_direct = True
    io_async = False
    worker = setup(backends, block_size, stream_number, io_direct, io_async, True)
    scheduler = setup(backends, block_size, stream_number, io_direct, io_async, False)
    batch_number = 64
    batch_size = 1024
    data_size = block_size * batch_size
    raw_data1 = [aligned_array(block_size) for _ in range(batch_size)]
    raw_data2 = [aligned_array(block_size) for _ in range(batch_size)]
    data1 = [[d.ctypes.data] for d in raw_data1]
    data2 = [[d.ctypes.data] for d in raw_data2]
    for idx in range(batch_number):
        block_ids = [secrets.token_bytes(16) for _ in range(batch_size)]
        shard_idxes = [0 for _ in range(batch_size)]

        tp = time.perf_counter()
        founds = scheduler.lookup(block_ids)
        cost_fully_lookup1 = time.perf_counter() - tp
        assert not any(founds)

        tp = time.perf_counter()
        found_idx = scheduler.lookup_on_prefix(block_ids)
        cost_prefix_lookup1 = time.perf_counter() - tp
        assert found_idx == -1

        tp = time.perf_counter()
        handle = worker.dump_data(block_ids, shard_idxes, data1)
        worker.wait(handle)
        cost_dump = time.perf_counter() - tp

        tp = time.perf_counter()
        founds = scheduler.lookup(block_ids)
        cost_fully_lookup2 = time.perf_counter() - tp
        assert all(founds)

        tp = time.perf_counter()
        found_idx = scheduler.lookup_on_prefix(block_ids)
        cost_prefix_lookup2 = time.perf_counter() - tp
        assert found_idx == batch_size - 1

        tp = time.perf_counter()
        handle = worker.load_data(block_ids, shard_idxes, data2)
        worker.wait(handle)
        cost_load = time.perf_counter() - tp

        bw_dump = data_size / cost_dump
        bw_load = data_size / cost_load
        print(
            f"[{idx:03}/{batch_number:03}] [{block_size}] [{batch_size}] "
            f"fully_lookup1={cost_fully_lookup1 * 1e3:.3f}ms, "
            f"prefix_lookup1={cost_prefix_lookup1 * 1e3:.3f}ms, "
            f"fully_lookup2={cost_fully_lookup2 * 1e3:.3f}ms, "
            f"prefix_lookup2={cost_prefix_lookup2 * 1e3:.3f}ms, "
            f"dump={cost_dump * 1e3:.3f}ms, load={cost_load * 1e3:.3f}ms, "
            f"bw_dump={bw_dump / 1e9:.3f}GB/s, bw_load={bw_load / 1e9:.3f}GB/s."
        )


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "info"
    main()
