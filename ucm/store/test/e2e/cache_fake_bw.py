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

import torch

from ucm.store.pipeline.connector import UcmPipelineStore

store_pipeline = "Cache|Fake"
device_type = "npu"
device_id = 0
tensor_size_list = [131072, 65536, 32768]
block_number = 100
dump_epoch_number = 32
load_epoch_number = 32
dtype = torch.bfloat16
cache_buffer_capacity_gb = 8
cache_stream_number = 4
waiting_queue_depth = 16
running_queue_depth = 1024
timeout_ms = 10000
check_data = False
cache_sdma_direct = False
io_direct = True
share_buffer_enable = True
shard_number = 1


def setup_device():
    if device_type == "cuda":
        torch.cuda.set_device(device_id)
    else:
        import torch_npu  # noqa: F401

        torch.npu.set_device(device_id)
    return f"{device_type}:{device_id}"


def synchronize_device():
    if device_type == "cuda":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()


def create_worker(unique_id: str) -> UcmPipelineStore:
    shard_size = sum(tensor_size_list)
    config = {}
    config["store_pipeline"] = store_pipeline
    config["unique_id"] = unique_id
    config["tensor_size_list"] = tensor_size_list
    config["shard_size"] = shard_size
    config["block_size"] = shard_size
    config["share_buffer_enable"] = share_buffer_enable
    config["io_direct"] = io_direct
    config["cache_buffer_capacity_gb"] = cache_buffer_capacity_gb
    config["cache_stream_number"] = cache_stream_number
    config["cache_sdma_direct"] = cache_sdma_direct
    config["waiting_queue_depth"] = waiting_queue_depth
    config["running_queue_depth"] = running_queue_depth
    config["timeout_ms"] = timeout_ms
    config["device_id"] = device_id
    return UcmPipelineStore(config)


def make_tensors(device: str):
    return make_sized_tensors(device, torch.rand)


def make_empty_tensors(device: str):
    return make_sized_tensors(device, torch.empty)


def make_sized_tensors(device: str, factory):
    element_size = torch.empty((), dtype=dtype).element_size()
    tensors = []
    for _ in range(block_number):
        row = []
        for tensor_size in tensor_size_list:
            if tensor_size % element_size != 0:
                raise ValueError(
                    f"tensor size {tensor_size} is not divisible by {element_size}"
                )
            row.append(
                factory([tensor_size // element_size], dtype=dtype, device=device)
            )
        tensors.append(row)
    return tensors


def check_tensors(src_tensors, dst_tensors):
    for row_idx, (src_row, dst_row) in enumerate(zip(src_tensors, dst_tensors)):
        for col_idx, (src_tensor, dst_tensor) in enumerate(zip(src_row, dst_row)):
            if not torch.equal(src_tensor, dst_tensor):
                raise AssertionError(f"tensor mismatch at [{row_idx}][{col_idx}]")


def dump(epoch: int, device: str, device_id: int, worker, block_ids):
    src_tensors = make_tensors(device)
    total_size = sum(tensor_size_list) * block_number * shard_number
    costs = []
    for shard_idx in range(shard_number):
        shard_indexes = [shard_idx for _ in range(block_number)]
        synchronize_device()
        tp = time.perf_counter()
        task = worker.dump(block_ids, shard_indexes, src_tensors)
        worker.wait(task)
        costs.append(time.perf_counter() - tp)
    print_result("dump", epoch, device_id, costs, total_size)
    if check_data:
        return src_tensors
    return None


def load(epoch: int, device: str, device_id: int, worker, block_ids, src_tensors=None):
    dst_tensors = make_empty_tensors(device)
    total_size = sum(tensor_size_list) * block_number * shard_number
    costs = []
    for shard_idx in range(shard_number):
        shard_indexes = [shard_idx for _ in range(block_number)]
        synchronize_device()
        tp = time.perf_counter()
        task = worker.load(block_ids, shard_indexes, dst_tensors)
        worker.wait(task)
        costs.append(time.perf_counter() - tp)
    synchronize_device()
    if check_data and src_tensors is not None:
        check_tensors(src_tensors, dst_tensors)
    print_result("load", epoch, device_id, costs, total_size)


def percentile(values, percent):
    ordered = sorted(values)
    index = int((len(ordered) - 1) * percent / 100)
    return ordered[index]


def print_result(direction: str, epoch: int, device_id: int, costs, total_size: int):
    total_cost = sum(costs)
    avg_cost = total_cost / len(costs)
    p99_cost = percentile(costs, 99)
    print(
        f"epoch={epoch:03}, worker={device_id:02}, "
        f"{direction}=[{sum(tensor_size_list)} x {block_number} x {shard_number}], "
        f"avg_cost={avg_cost * 1e3:.3f}ms, "
        f"p99_cost={p99_cost * 1e3:.3f}ms, "
        f"total_cost={total_cost * 1e3:.3f}ms, "
        f"bw={total_size / total_cost / 1e9:.3f}GB/s."
    )


def main():
    os.environ["UC_LOGGER_LEVEL"] = "warning"
    device = setup_device()
    unique_id = secrets.token_hex(8)
    worker = create_worker(unique_id)
    print(
        f"{store_pipeline} one-layer benchmark: device={device}, "
        f"block_number={block_number}, tensor_size_list={tensor_size_list}, "
        f"shard_size={sum(tensor_size_list)}, dtype={dtype}, "
        f"cache_sdma_direct={cache_sdma_direct}, io_direct={io_direct}, "
        f"share_buffer_enable={share_buffer_enable}"
    )

    records = []
    for epoch in range(dump_epoch_number):
        block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
        src_tensors = dump(epoch, device, device_id, worker, block_ids)
        records.append((block_ids, src_tensors))

    for epoch in range(load_epoch_number):
        block_ids, src_tensors = records[epoch % len(records)]
        load(epoch, device, device_id, worker, block_ids, src_tensors)


if __name__ == "__main__":
    main()
