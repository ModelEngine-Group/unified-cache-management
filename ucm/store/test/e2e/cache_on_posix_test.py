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
    founds = scheduler.lookup(chunk_block_ids)
    assert not all(founds)
    shard_indexes = [0 for _ in range(request_size)]
    src_tensors = [
        [
            torch.rand(
                [tensor_size // 2],
                dtype=torch.bfloat16,
                device="cuda:{}".format(device_id),
            )
            for _ in range(layer_size * chunk_size)
        ]
        for _ in range(request_size)
    ]
    task = worker.dump(chunk_block_ids, shard_indexes, src_tensors)
    worker.wait(task)
    dst_tensors = [[torch.empty_like(t) for t in row] for row in src_tensors]
    task = worker.load(chunk_block_ids, shard_indexes, dst_tensors)
    worker.wait(task)
    cmp_and_print_diff(src_tensors, dst_tensors)


def main():
    tensor_size = 262144
    layer_size = 64
    chunk_size = 4
    request_size = chunk_size * 16
    storage_backends = ["."]
    device_id = 1
    chunk_block_size = tensor_size * layer_size * chunk_size
    config = {}
    config["store_pipeline"] = "Cache|Posix"
    config["device_id"] = device_id
    config["storage_backends"] = storage_backends
    config["unique_id"] = secrets.token_hex(8)
    config["tensor_size"] = tensor_size
    config["shard_size"] = chunk_block_size
    config["block_size"] = chunk_block_size
    config["buffer_size"] = chunk_block_size * 2048
    config["share_buffer_enable"] = True
    config["waiting_queue_depth"] = 16
    config["running_queue_depth"] = 1024
    config["io_direct"] = True
    config["stream_number"] = 16
    config["timeout_ms"] = 30000
    worker = UcmPipelineStore(config)
    config["device_id"] = -1
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
    time.sleep(10)


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "debug"
    main()
