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
import secrets
import time

import torch

from ucm.store.factory_v1 import UcmConnectorFactoryV1, UcmKVStoreBaseV1

device_id = 0
shard_size = 64 * 1024
shard_number = 64
block_number = 1024
storage_backends = ["./build/data"]


def create_worker(store_pipeline, unique_id) -> UcmKVStoreBaseV1:
    module_path = "ucm.store.pipeline.connector"
    class_name = "UcmPipelineStore"
    config = {}
    config["store_pipeline"] = store_pipeline
    config["tensor_size"] = shard_size
    config["shard_size"] = shard_size * shard_number
    config["block_size"] = shard_size * shard_number
    config["device_id"] = device_id
    config["unique_id"] = unique_id
    config["share_buffer_enable"] = True
    config["cache_buffer_capacity_gb"] = 8
    config["storage_backends"] = storage_backends
    return UcmConnectorFactoryV1.create_connector(class_name, config, module_path)


def make_tensors(device):
    return [
        [
            torch.rand([shard_size // 2], dtype=torch.bfloat16, device=device)
            for _ in range(shard_number)
        ]
        for _ in range(block_number)
    ]


def cmp_and_print_diff(a, b, rtol=0.0, atol=0.0):
    for r, (row_a, row_b) in enumerate(zip(a, b)):
        for c, (ta, tb) in enumerate(zip(row_a, row_b)):
            if not torch.allclose(ta, tb, rtol=rtol, atol=atol):
                mask = ~torch.isclose(ta, tb, rtol=rtol, atol=atol)
                diff_a = ta[mask].cpu()
                diff_b = tb[mask].cpu()
                print(
                    f"DIFF at d{tb.device}[{r}][{c}]  total {mask.sum().item()} element(s)"
                )
                print("  a val:", diff_a.flatten())
                print("  b val:", diff_b.flatten())
                assert False


def dump(worker, block_ids, block_tensors):
    shard_indexes = [0 for _ in range(len(block_ids))]
    tp = time.perf_counter()
    task = worker.dump(block_ids, shard_indexes, block_tensors)
    worker.wait(task)
    cost = time.perf_counter() - tp
    print(f"Dump data({cost * 1e3:.3f}ms) successfullyl: {block_tensors[0][0]}")


def load(worker, block_ids):
    block_tensors = make_tensors("cuda:{}".format(device_id))
    shard_indexes = [0 for _ in range(len(block_ids))]
    tp = time.perf_counter()
    task = worker.load(block_ids, shard_indexes, block_tensors)
    worker.wait(task)
    cost = time.perf_counter() - tp
    print(f"Load data({cost * 1e3:.3f}ms) successfullyl: {block_tensors[0][0]}")
    return block_tensors


if __name__ == "__main__":
    unique_id = secrets.token_hex(8)
    dumper = create_worker("Posix", unique_id)
    loader = create_worker("Cache|Posix", unique_id)
    block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
    src_tensors = make_tensors("cpu:0")
    dump(dumper, block_ids, src_tensors)
    hbm_tensors = load(loader, block_ids)
    dst_tensors = [[t.to("cpu:0") for t in row] for row in hbm_tensors]
    cmp_and_print_diff(src_tensors, dst_tensors)
    hbm_tensors = load(loader, block_ids)
    dst_tensors = [[t.to("cpu:0") for t in row] for row in hbm_tensors]
    cmp_and_print_diff(src_tensors, dst_tensors)
