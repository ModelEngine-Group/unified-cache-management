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
import torch_npu

from ucm.store.pipeline.connector import UcmPipelineStore


def cmp_and_print_diff(a, b, rtol=0.0, atol=0.0):
    for r, (row_a, row_b) in enumerate(zip(a, b)):
        for c, (ta, tb) in enumerate(zip(row_a, row_b)):
            if not torch.allclose(ta, tb, rtol=rtol, atol=atol):
                mask = ~torch.isclose(ta, tb, rtol=rtol, atol=atol)
                diff_a = ta[mask].cpu()
                diff_b = tb[mask].cpu()
                print(f"DIFF at [{r}][{c}]  total {mask.sum().item()} element(s)")
                print("  a val:", diff_a.flatten())
                print("  b val:", diff_b.flatten())
                assert False


def e2e_test(
    worker: UcmPipelineStore,
    scheduler: UcmPipelineStore,
    tensor_size: int,
    layer_size: int,
    chunk_size: int,
    request_size: int,
    device_id: int,
):
    chunk_block_ids = [secrets.token_bytes(16) for _ in range(request_size)]

    # lookup at 0% hit
    tp = time.perf_counter()
    founds = scheduler.lookup(chunk_block_ids)
    cost_fully_lookup1 = time.perf_counter() - tp
    assert not any(founds)

    tp = time.perf_counter()
    found_idx = scheduler.lookup_on_prefix(chunk_block_ids)
    cost_prefix_lookup1 = time.perf_counter() - tp
    assert found_idx == -1

    # make tensor randomly
    shard_indexes = [0 for _ in range(request_size)]
    src_tensors = [
        [
            torch.rand(
                [tensor_size // 2],
                dtype=torch.bfloat16,
                device="npu:{}".format(device_id),
            )
            for _ in range(layer_size * chunk_size)
        ]
        for _ in range(request_size)
    ]

    # dump
    tp = time.perf_counter()
    task = worker.dump(chunk_block_ids, shard_indexes, src_tensors)
    worker.wait(task)
    cost_dump = time.perf_counter() - tp

    # lookup at 100% hit
    tp = time.perf_counter()
    founds = scheduler.lookup(chunk_block_ids)
    cost_fully_lookup2 = time.perf_counter() - tp
    assert all(founds)

    tp = time.perf_counter()
    found_idx = scheduler.lookup_on_prefix(chunk_block_ids)
    cost_prefix_lookup2 = time.perf_counter() - tp
    assert found_idx + 1 == request_size

    # load
    dst_tensors = [[torch.empty_like(t) for t in row] for row in src_tensors]
    tp = time.perf_counter()
    task = worker.load(chunk_block_ids, shard_indexes, dst_tensors)
    worker.wait(task)
    cost_load = time.perf_counter() - tp

    # compare
    cmp_and_print_diff(src_tensors, dst_tensors)

    # report
    data_size = tensor_size * layer_size * chunk_size * request_size
    bw_dump = data_size / cost_dump
    bw_load = data_size / cost_load
    print(
        f"[{tensor_size}-{layer_size}-{chunk_size}-{request_size}] "
        f"fully_lookup1={cost_fully_lookup1 * 1e3:.3f}ms, "
        f"prefix_lookup1={cost_prefix_lookup1 * 1e3:.3f}ms, "
        f"fully_lookup2={cost_fully_lookup2 * 1e3:.3f}ms, "
        f"prefix_lookup2={cost_prefix_lookup2 * 1e3:.3f}ms, "
        f"dump={cost_dump * 1e3:.3f}ms, load={cost_load * 1e3:.3f}ms, "
        f"bw_dump={bw_dump / 1e9:.3f}GB/s, bw_load={bw_load / 1e9:.3f}GB/s."
    )


def main():
    tensor_size = 32768
    layer_size = 64
    chunk_size = 4
    request_size = chunk_size * 16
    device_id = int(os.environ.get("DEVICE_ID", "0"))

    local_hostname = os.environ.get("MOONCAKE_LOCAL_HOST", "127.0.0.1")
    master_address = os.environ.get("MOONCAKE_MASTER", "127.0.0.1:50088")
    metadata_server = os.environ.get("MOONCAKE_METADATA", "P2PHANDSHAKE")
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "ascend")

    chunk_block_size = tensor_size * layer_size * chunk_size
    tensor_size_list = [tensor_size] * (layer_size * chunk_size)

    config = {}
    config["store_pipeline"] = "Mooncake"
    config["unique_id"] = secrets.token_hex(8)
    config["local_hostname"] = local_hostname
    config["master_server_address"] = master_address
    config["metadata_server"] = metadata_server
    config["protocol"] = protocol
    config["global_segment_size_gb"] = 32
    config["local_buffer_size_gb"] = 1
    config["replica_num"] = 1
    config["tensor_size"] = tensor_size
    config["tensor_size_list"] = tensor_size_list
    config["shard_size"] = chunk_block_size
    config["block_size"] = chunk_block_size
    config["timeout_ms"] = 10000

    torch.npu.set_device(device_id)

    worker = UcmPipelineStore(config | {"device_id": device_id})
    scheduler = UcmPipelineStore(config)

    test_batch_number = 512
    for _ in range(test_batch_number):
        e2e_test(
            worker,
            scheduler,
            tensor_size,
            layer_size,
            chunk_size,
            request_size,
            device_id,
        )


if __name__ == "__main__":
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")
    main()
