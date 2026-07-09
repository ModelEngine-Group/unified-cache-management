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

store_pipeline = "Cache|Posix"
device_type = "npu"
device_id = 0
tensor_size_list = [32768]
block_number = 100
warmup_epoch_number = 2
test_epoch_number = 10
dtype = torch.bfloat16
cache_buffer_capacity_gb = 8
cache_stream_number = 4
waiting_queue_depth = 16
running_queue_depth = 1024
timeout_ms = 10000
check_data = False
cache_sdma_direct = True
backend_ready_timeout_s = 60
backend_ready_poll_interval_s = 0.001

storage_backends = ["./build/data"]
posix_io_engine = "aio"
io_direct = True
posix_data_trans_concurrency = 32
posix_lookup_concurrency = 32
cache_load_backend_only = True


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


def create_cache_worker(unique_id: str) -> UcmPipelineStore:
    shard_size = sum(tensor_size_list)
    config = {}
    config["store_pipeline"] = store_pipeline
    config["storage_backends"] = storage_backends
    config["posix_io_engine"] = posix_io_engine
    config["io_direct"] = io_direct
    config["posix_data_trans_concurrency"] = posix_data_trans_concurrency
    config["posix_lookup_concurrency"] = posix_lookup_concurrency
    config["cache_load_backend_only"] = cache_load_backend_only
    config["unique_id"] = unique_id
    config["tensor_size_list"] = tensor_size_list
    config["shard_size"] = shard_size
    config["block_size"] = shard_size
    config["share_buffer_enable"] = True
    config["cache_buffer_capacity_gb"] = cache_buffer_capacity_gb
    config["cache_stream_number"] = cache_stream_number
    config["cache_sdma_direct"] = cache_sdma_direct
    config["waiting_queue_depth"] = waiting_queue_depth
    config["running_queue_depth"] = running_queue_depth
    config["timeout_ms"] = timeout_ms
    config["device_id"] = device_id
    return UcmPipelineStore(config)


def create_posix_scheduler() -> UcmPipelineStore:
    config = {}
    config["store_pipeline"] = "Posix"
    config["storage_backends"] = storage_backends
    config["posix_io_engine"] = posix_io_engine
    config["io_direct"] = io_direct
    config["posix_lookup_concurrency"] = posix_lookup_concurrency
    config["timeout_ms"] = timeout_ms
    config["device_id"] = -1
    return UcmPipelineStore(config)


def make_storage_dirs():
    for path in storage_backends:
        os.makedirs(path, exist_ok=True)


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
        "cost_s": cost,
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
    return build_result("d2h+backend submit", epoch, measured, total_bytes, cost)


def wait_backend_ready(scheduler, block_ids):
    deadline = time.perf_counter() + backend_ready_timeout_s
    tp = time.perf_counter()
    while True:
        founds = scheduler.lookup(block_ids)
        if bool(founds.all()):
            return time.perf_counter() - tp
        if time.perf_counter() >= deadline:
            ready_count = int(founds.sum())
            raise TimeoutError(
                f"backend commit timeout: ready={ready_count}/{len(block_ids)}"
            )
        time.sleep(backend_ready_poll_interval_s)


def build_complete_dump_result(epoch: int, measured: bool, total_bytes: int, cost: float):
    return build_result("d2h+backend complete", epoch, measured, total_bytes, cost)


def load_once(worker, epoch: int, measured: bool, block_ids, dst_tensors):
    total_bytes = sum(tensor_size_list) * block_number
    shard_indexes = [0 for _ in range(block_number)]
    synchronize_device()
    tp = time.perf_counter()
    task = worker.load(block_ids, shard_indexes, dst_tensors)
    worker.wait(task)
    cost = time.perf_counter() - tp
    return build_result("backend+h2d", epoch, measured, total_bytes, cost)


def print_result(result):
    phase = "measure" if result["measured"] else "warmup"
    print(
        f"{phase} epoch={result['epoch']:03} {result['direction']} "
        f"bytes={result['total_bytes']} "
        f"cost={result['cost_ms']:.3f}ms "
        f"bw={result['bw_gbs']:.3f}GB/s"
    )


def print_summary(name: str, results):
    total_bytes = sum(result["total_bytes"] for result in results)
    total_cost_s = sum(result["cost_s"] for result in results)
    avg_cost = statistics.fmean(result["cost_ms"] for result in results)
    bw_gbs = total_bytes / total_cost_s / 1e9 if total_cost_s > 0 else 0.0
    print(
        f"summary {name}: epochs={len(results)} "
        f"avg_cost={avg_cost:.3f}ms bw={bw_gbs:.3f}GB/s"
    )


def main():
    os.environ["UC_LOGGER_LEVEL"] = "warning"
    make_storage_dirs()
    device = setup_device()
    dump_worker = create_cache_worker(secrets.token_hex(8))
    load_worker = create_cache_worker(secrets.token_hex(8))
    scheduler = create_posix_scheduler()
    print(
        f"{store_pipeline} one-layer benchmark: device={device}, "
        f"block_number={block_number}, tensor_size_list={tensor_size_list}, "
        f"shard_size={sum(tensor_size_list)}, dtype={dtype}, "
        f"storage_backends={storage_backends}, posix_io_engine={posix_io_engine}, "
        f"cache_load_backend_only={cache_load_backend_only}, "
        f"cache_sdma_direct={cache_sdma_direct}"
    )

    dump_submit_results = []
    dump_complete_results = []
    load_results = []
    total_epochs = warmup_epoch_number + test_epoch_number
    for epoch in range(total_epochs):
        measured = epoch >= warmup_epoch_number
        block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
        src_tensors = make_tensors(device)
        dump_result = dump_once(dump_worker, epoch, measured, block_ids, src_tensors)
        print_result(dump_result)
        backend_wait_cost = wait_backend_ready(scheduler, block_ids)
        dump_complete_result = build_complete_dump_result(
            epoch,
            measured,
            dump_result["total_bytes"],
            dump_result["cost_s"] + backend_wait_cost,
        )
        print_result(dump_complete_result)

        dst_tensors = make_empty_like(src_tensors)
        load_result = load_once(load_worker, epoch, measured, block_ids, dst_tensors)
        print_result(load_result)
        if check_data:
            synchronize_device()
            check_tensors(src_tensors, dst_tensors)
        if measured:
            dump_submit_results.append(dump_result)
            dump_complete_results.append(dump_complete_result)
            load_results.append(load_result)

    print_summary("d2h+backend submit", dump_submit_results)
    print_summary("d2h+backend complete", dump_complete_results)
    print_summary("backend+h2d", load_results)


if __name__ == "__main__":
    main()
