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
import statistics
import time

import torch

from ucm.store.pipeline.connector import UcmPipelineStore

store_pipeline = "Cache|Fake"
device_type = "npu"
device_id = 0
tensor_size_list = [32768]
block_number = 64
warmup_epoch_number = 2
test_epoch_number = 10
dtype = torch.bfloat16
cache_buffer_capacity_gb = 8
cache_stream_number = 4
waiting_queue_depth = 16
running_queue_depth = 1024
timeout_ms = 10000
check_data = False


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
    config["share_buffer_enable"] = True
    config["cache_buffer_capacity_gb"] = cache_buffer_capacity_gb
    config["cache_stream_number"] = cache_stream_number
    config["waiting_queue_depth"] = waiting_queue_depth
    config["running_queue_depth"] = running_queue_depth
    config["timeout_ms"] = timeout_ms
    config["device_id"] = device_id
    return UcmPipelineStore(config)


def make_tensors(device: str):
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
                torch.rand(
                    [tensor_size // element_size],
                    dtype=dtype,
                    device=device,
                )
            )
        tensors.append(row)
    return tensors


def make_empty_like(tensors):
    return [[torch.empty_like(tensor) for tensor in row] for row in tensors]


def check_tensors(src_tensors, dst_tensors):
    for row_idx, (src_row, dst_row) in enumerate(zip(src_tensors, dst_tensors)):
        for col_idx, (src_tensor, dst_tensor) in enumerate(zip(src_row, dst_row)):
            if not torch.equal(src_tensor, dst_tensor):
                raise AssertionError(f"tensor mismatch at [{row_idx}][{col_idx}]")


def build_result(direction: str, epoch: int, measured: bool, total_bytes: int, cost: float):
    return {
        "direction": direction,
        "epoch": epoch,
        "measured": measured,
        "total_bytes": total_bytes,
        "cost_ms": cost * 1e3,
        "bw_gbs": total_bytes / cost / 1e9 if cost > 0 else 0.0,
    }


def dump_once(worker, epoch: int, measured: bool, block_ids, src_tensors):
    total_bytes = sum(tensor_size_list) * block_number
    shard_indexes = [0 for _ in range(block_number)]
    synchronize_device()
    tp = time.perf_counter()
    task = worker.dump(block_ids, shard_indexes, src_tensors)
    worker.wait(task)
    cost = time.perf_counter() - tp
    return build_result("d2h", epoch, measured, total_bytes, cost)


def load_once(worker, epoch: int, measured: bool, block_ids, dst_tensors):
    total_bytes = sum(tensor_size_list) * block_number
    shard_indexes = [0 for _ in range(block_number)]
    synchronize_device()
    tp = time.perf_counter()
    task = worker.load(block_ids, shard_indexes, dst_tensors)
    worker.wait(task)
    cost = time.perf_counter() - tp
    return build_result("h2d", epoch, measured, total_bytes, cost)


def print_result(result):
    phase = "measure" if result["measured"] else "warmup"
    print(
        f"{phase} epoch={result['epoch']:03} {result['direction']} "
        f"bytes={result['total_bytes']} "
        f"cost={result['cost_ms']:.3f}ms "
        f"bw={result['bw_gbs']:.3f}GB/s"
    )


def print_summary(name: str, results):
    avg_cost = statistics.fmean(result["cost_ms"] for result in results)
    avg_bw = statistics.fmean(result["bw_gbs"] for result in results)
    print(
        f"summary {name}: epochs={len(results)} "
        f"avg_cost={avg_cost:.3f}ms avg_bw={avg_bw:.3f}GB/s"
    )


def main():
    os.environ["UC_LOGGER_LEVEL"] = "warning"
    device = setup_device()
    unique_id = secrets.token_hex(8)
    worker = create_worker(unique_id)
    print(
        f"{store_pipeline} one-layer benchmark: device={device}, "
        f"block_number={block_number}, tensor_size_list={tensor_size_list}, "
        f"shard_size={sum(tensor_size_list)}, dtype={dtype}"
    )

    dump_results = []
    load_results = []
    total_epochs = warmup_epoch_number + test_epoch_number
    for epoch in range(total_epochs):
        measured = epoch >= warmup_epoch_number
        block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
        src_tensors = make_tensors(device)
        dump_result = dump_once(worker, epoch, measured, block_ids, src_tensors)
        print_result(dump_result)

        dst_tensors = make_empty_like(src_tensors)
        load_result = load_once(worker, epoch, measured, block_ids, dst_tensors)
        print_result(load_result)
        if check_data:
            synchronize_device()
            check_tensors(src_tensors, dst_tensors)
        if measured:
            dump_results.append(dump_result)
            load_results.append(load_result)

    print_summary("d2h", dump_results)
    print_summary("h2d", load_results)


if __name__ == "__main__":
    main()
