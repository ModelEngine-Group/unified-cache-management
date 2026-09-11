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
import argparse
import mmap
import multiprocessing
import secrets
import time

import numpy as np

from ucm.store.factory_v1 import UcmConnectorFactoryV1, UcmKVStoreBaseV1

worker_number = 1
shard_size = 8 * 1024 * 1024
shard_number = 1
block_number = 64
dump_epoch_number = 32
load_epoch_number = 32
storage_backends = ["./build/data"]
posix_data_trans_concurrency = 32
posix_io_engine = "aio"
io_direct = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run POSIX AIO store benchmark")
    parser.add_argument("--worker-number", type=int, default=worker_number)
    parser.add_argument("--shard-size", type=int, default=shard_size)
    parser.add_argument("--shard-number", type=int, default=shard_number)
    parser.add_argument("--block-number", type=int, default=block_number)
    parser.add_argument("--dump-epoch-number", type=int, default=dump_epoch_number)
    parser.add_argument("--load-epoch-number", type=int, default=load_epoch_number)
    parser.add_argument(
        "--posix-data-trans-concurrency",
        type=int,
        default=posix_data_trans_concurrency,
        help="posix data transfer concurrency (psync worker count).",
    )
    parser.add_argument(
        "--posix-io-engine",
        choices=["psync", "aio"],
        default=posix_io_engine,
        help="posix io engine.",
    )
    parser.add_argument(
        "--io-direct",
        action=argparse.BooleanOptionalAction,
        default=io_direct,
        help="use O_DIRECT for aligned file I/O (use --no-io-direct to disable).",
    )
    parser.add_argument(
        "--storage-backend",
        action="append",
        default=None,
        help="Storage backend path; may be repeated",
    )
    return parser.parse_args(argv)


def apply_args(args):
    global worker_number
    global shard_size
    global shard_number
    global block_number
    global dump_epoch_number
    global load_epoch_number
    global storage_backends
    global posix_data_trans_concurrency
    global posix_io_engine
    global io_direct

    worker_number = args.worker_number
    shard_size = args.shard_size
    shard_number = args.shard_number
    block_number = args.block_number
    dump_epoch_number = args.dump_epoch_number
    load_epoch_number = args.load_epoch_number
    if args.storage_backend is not None:
        storage_backends = args.storage_backend
    posix_data_trans_concurrency = args.posix_data_trans_concurrency
    posix_io_engine = args.posix_io_engine
    io_direct = args.io_direct


def create_worker(device_id: int) -> UcmKVStoreBaseV1:
    module_path = "ucm.store.pipeline.connector"
    class_name = "UcmPipelineStore"
    config = {}
    config["store_pipeline"] = "Posix"
    config["posix_io_engine"] = posix_io_engine
    config["io_direct"] = io_direct
    config["posix_data_trans_concurrency"] = posix_data_trans_concurrency
    config["storage_backends"] = storage_backends
    config["tensor_size"] = shard_size
    config["shard_size"] = shard_size
    config["block_size"] = shard_size * shard_number
    config["device_id"] = device_id
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


def dump(epoch, device_id, worker, block_ids, block_ptr):
    total_size = shard_size * shard_number * block_number
    costs = []
    for i in range(shard_number):
        idxes = [i for _ in range(block_number)]
        ptrs = [[ptr + i * shard_size] for ptr in block_ptr]
        tp = time.perf_counter()
        task = worker.dump_data(block_ids, idxes, ptrs)
        worker.wait(task)
        costs.append(time.perf_counter() - tp)
    total_cost = np.sum(costs)
    print(
        f"epoch={epoch:03}, worker={device_id:02}, "
        f"dump=[{shard_size} x {block_number} x {shard_number}], "
        f"avg_cost={np.average(costs) * 1e3:.3f}ms, "
        f"p99_cost={np.percentile(costs, 99) * 1e3:.3f}ms, "
        f"total_cost={total_cost * 1e3:.3f}ms, "
        f"bw={total_size / total_cost / 1e9:.3f}GB/s."
    )


def load(epoch, device_id, worker, block_ids, block_ptr):
    total_size = shard_size * shard_number * block_number
    costs = []
    for i in range(shard_number):
        idxes = [i for _ in range(block_number)]
        ptrs = [[ptr + i * shard_size] for ptr in block_ptr]
        tp = time.perf_counter()
        task = worker.load_data(block_ids, idxes, ptrs)
        worker.wait(task)
        costs.append(time.perf_counter() - tp)
    total_cost = np.sum(costs)
    print(
        f"epoch={epoch:03}, worker={device_id:02}, "
        f"load=[{shard_size} x {block_number} x {shard_number}], "
        f"avg_cost={np.average(costs) * 1e3:.3f}ms, "
        f"p99_cost={np.percentile(costs, 99) * 1e3:.3f}ms, "
        f"total_cost={total_cost * 1e3:.3f}ms, "
        f"bw={total_size / total_cost / 1e9:.3f}GB/s."
    )


def worker_loop(device_id, barrier):
    store = create_worker(device_id)
    block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
    block_data = [make_array(shard_size * shard_number) for _ in range(block_number)]
    block_ptr = [block.ctypes.data for block in block_data]
    barrier.wait()
    for epoch in range(dump_epoch_number):
        dump(epoch, device_id, store, block_ids, block_ptr)
        barrier.wait()
    for epoch in range(load_epoch_number):
        load(epoch, device_id, store, block_ids, block_ptr)
        barrier.wait()


if __name__ == "__main__":
    apply_args(parse_args())
    barrier = multiprocessing.Barrier(worker_number)
    workers = []
    for i in range(worker_number):
        p = multiprocessing.Process(target=worker_loop, args=(i, barrier))
        workers.append(p)
        p.start()
    for w in workers:
        w.join()
