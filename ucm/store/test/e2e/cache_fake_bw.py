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
import multiprocessing
import os
import secrets
import time

import torch

from ucm.store.pipeline.connector import UcmPipelineStore

store_pipeline = "Cache|Fake"
device_type = "npu"

# =========================== User configuration ===========================
model_name = "glm-5.2"

# Fill tensor_size_list with the per-layer tensor byte sizes of the target
# deployment before running a profile. MLA worker 0 writes the shared block
# ids and every worker loads them; GQA workers use their own block ids.
MODEL_PROFILES = {
    "glm-5.2": {
        "worker_mode": "mla",
        "worker_number": 8,
        "share_buffer_enable": True,
        "tensor_size_list": [],
    },
    "minimax-m2.7": {
        "worker_mode": "gqa",
        "worker_number": 8,
        "share_buffer_enable": False,
        "tensor_size_list": [],
    },
    "dsv4": {
        "worker_mode": "mla",
        "worker_number": 8,
        "share_buffer_enable": True,
        "tensor_size_list": [],
    },
}

model_profile = MODEL_PROFILES.get(model_name)
if model_profile is None:
    available_models = ", ".join(MODEL_PROFILES)
    raise ValueError(
        f"unsupported model {model_name!r}; choose one of: {available_models}"
    )
worker_mode = model_profile["worker_mode"]
worker_number = model_profile["worker_number"]
share_buffer_enable = model_profile["share_buffer_enable"]
tensor_size_list = model_profile["tensor_size_list"]

# ======================== Benchmark configuration =========================
block_number = 100
dump_epoch_number = 32
load_epoch_number = 32
cache_sdma_direct = False


def setup_device(device_id: int):
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


def create_worker(unique_id: str, device_id: int) -> UcmPipelineStore:
    shard_size = sum(tensor_size_list)
    config = {}
    config["store_pipeline"] = store_pipeline
    config["unique_id"] = unique_id
    config["tensor_size_list"] = tensor_size_list
    config["shard_size"] = shard_size
    config["block_size"] = shard_size
    config["share_buffer_enable"] = share_buffer_enable
    config["cache_sdma_direct"] = cache_sdma_direct
    config["device_id"] = device_id
    return UcmPipelineStore(config)


def make_tensors(device: str):
    return make_sized_tensors(device, torch.rand)


def make_empty_tensors(device: str):
    return make_sized_tensors(device, torch.empty)


def make_sized_tensors(device: str, factory):
    dtype = torch.bfloat16
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


def dump(epoch: int, device: str, device_id: int, worker, block_ids):
    src_tensors = make_tensors(device)
    total_size = sum(tensor_size_list) * block_number
    shard_indexes = [0 for _ in range(block_number)]
    synchronize_device()
    tp = time.perf_counter()
    task = worker.dump(block_ids, shard_indexes, src_tensors)
    worker.wait(task)
    print_result("dump", epoch, device_id, time.perf_counter() - tp, total_size)


def load(epoch: int, device: str, device_id: int, worker, block_ids):
    dst_tensors = make_empty_tensors(device)
    total_size = sum(tensor_size_list) * block_number
    shard_indexes = [0 for _ in range(block_number)]
    synchronize_device()
    tp = time.perf_counter()
    task = worker.load(block_ids, shard_indexes, dst_tensors)
    worker.wait(task)
    synchronize_device()
    print_result("load", epoch, device_id, time.perf_counter() - tp, total_size)


def print_result(direction: str, epoch: int, device_id: int, cost: float, total_size: int):
    print(
        f"epoch={epoch:03}, worker={device_id:02}, "
        f"{direction}=[{sum(tensor_size_list)} x {block_number}], "
        f"cost={cost * 1e3:.3f}ms, "
        f"bw={total_size / cost / 1e9:.3f}GB/s."
    )


def worker_loop(
    device_id: int,
    barrier: multiprocessing.Barrier,
    unique_id: str,
    block_id_records,
):
    os.environ["UC_LOGGER_LEVEL"] = "warning"
    device = setup_device(device_id)
    worker = create_worker(unique_id, device_id)
    print(
        f"{store_pipeline} one-layer benchmark: device={device}, "
        f"model={model_name}, worker_mode={worker_mode}, "
        f"worker_number={worker_number}, "
        f"block_number={block_number}, tensor_size_list={tensor_size_list}, "
        f"shard_size={sum(tensor_size_list)}, dtype={torch.bfloat16}, "
        f"cache_sdma_direct={cache_sdma_direct}, "
        f"share_buffer_enable={share_buffer_enable}"
    )

    barrier.wait()
    for epoch, block_ids in enumerate(block_id_records):
        if worker_mode == "gqa" or device_id == 0:
            dump(epoch, device, device_id, worker, block_ids)
        barrier.wait()

    for epoch in range(load_epoch_number):
        record_idx = epoch % len(block_id_records)
        load(
            epoch,
            device,
            device_id,
            worker,
            block_id_records[record_idx],
        )
        barrier.wait()


def make_block_id_records():
    return [
        [secrets.token_bytes(16) for _ in range(block_number)]
        for _ in range(dump_epoch_number)
    ]


if __name__ == "__main__":
    if not tensor_size_list:
        raise ValueError(
            f"{model_name} tensor_size_list is empty; fill it in MODEL_PROFILES "
            "before running the benchmark"
        )
    barrier = multiprocessing.Barrier(worker_number)
    unique_id = secrets.token_hex(8)
    shared_block_id_records = make_block_id_records()
    workers = []
    for device_id in range(worker_number):
        block_id_records = (
            shared_block_id_records
            if worker_mode == "mla"
            else make_block_id_records()
        )
        process = multiprocessing.Process(
            target=worker_loop,
            args=(device_id, barrier, unique_id, block_id_records),
        )
        workers.append(process)
        process.start()
    for process in workers:
        process.join()
