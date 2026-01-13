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
    tensors1 = [
        [
            torch.rand(
                [tensor_size // 2],
                dtype=torch.bfloat16,
                device="cuda:{}".format(device_id),
            )
            for _ in range(layer_size)
        ]
        for _ in range(request_size)
    ]
    chunk_number = request_size // chunk_size
    chunk_ids = [secrets.token_bytes(16) for _ in range(chunk_number)]
    founds = scheduler.lookup(chunk_ids)
    assert not any(founds)
    assert scheduler.lookup_on_prefix(chunk_ids) == -1
    tasks1 = []
    for i in range(layer_size):
        shard_indexes = [i for _ in range(chunk_number)]
        layer_tensors = [block[i] for block in tensors1]
        src_tensors = [
            layer_tensors[i_chunk * chunk_size : (i_chunk + 1) * chunk_size]
            for i_chunk in range(chunk_number)
        ]
        task = worker.dump(chunk_ids, shard_indexes, src_tensors)
        tasks1.append(task)
    for task in tasks1:
        worker.wait(task)
    founds = scheduler.lookup(chunk_ids)
    assert all(founds)
    assert scheduler.lookup_on_prefix(chunk_ids) + 1 == chunk_number
    tensors2 = [[torch.empty_like(t) for t in row] for row in tensors1]
    tasks2 = []
    for i in range(layer_size):
        shard_indexes = [i for _ in range(chunk_number)]
        layer_tensors = [block[i] for block in tensors2]
        dst_tensors = [
            layer_tensors[i_chunk * chunk_size : (i_chunk + 1) * chunk_size]
            for i_chunk in range(chunk_number)
        ]
        task = worker.load(chunk_ids, shard_indexes, dst_tensors)
        tasks2.append(task)
    for task in tasks2:
        worker.wait(task)
    cmp_and_print_diff(tensors1, tensors2)


def main():
    tensor_size = 32768
    layer_size = 64
    chunk_size = 4
    request_size = chunk_size * 16
    device_id = 1
    config = {}
    config["store_pipeline"] = "Cache|Empty"
    config["unique_id"] = secrets.token_hex(8)
    config["tensor_size"] = tensor_size
    config["shard_size"] = tensor_size * chunk_size
    config["block_size"] = tensor_size * chunk_size * layer_size
    config["share_buffer_enable"] = True
    worker = UcmPipelineStore(config | {"device_id": device_id})
    scheduler = UcmPipelineStore(config)
    test_batch_number = 32
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
    os.environ["UC_LOGGER_LEVEL"] = "debug"
    main()
