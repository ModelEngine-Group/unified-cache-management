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
    tensor_size: list[int],
    layer_size: int,
    request_size: int,
    device_id: int,
):
    block_ids = [secrets.token_bytes(16) for _ in range(request_size)]
    founds = scheduler.lookup(block_ids)
    assert not any(founds)
    found_idx = scheduler.lookup_on_prefix(block_ids)
    assert found_idx == -1
    src_tensors1 = [
        [
            torch.rand(
                [tensor_size[0] // 2],
                dtype=torch.bfloat16,
                device="cuda:{}".format(device_id),
            )
            for _ in range(layer_size)
        ]
        for _ in range(request_size)
    ]
    src_tensors2 = [
        [
            torch.rand(
                [tensor_size[1] // 2],
                dtype=torch.bfloat16,
                device="cuda:{}".format(device_id),
            )
            for _ in range(layer_size)
        ]
        for _ in range(request_size)
    ]
    dst_tensors1 = [[torch.empty_like(t) for t in row] for row in src_tensors1]
    dst_tensors2 = [[torch.empty_like(t) for t in row] for row in src_tensors2]
    dump_tasks = []
    for i in range(layer_size):
        src_tensors = [
            [src_tensors1[j][i], src_tensors2[j][i]] for j in range(request_size)
        ]
        task = worker.dump(block_ids, [i] * request_size, src_tensors)
        dump_tasks.append(task)
    for task in dump_tasks:
        worker.wait(task)
    founds = scheduler.lookup(block_ids)
    assert all(founds)
    found_idx = scheduler.lookup_on_prefix(block_ids)
    assert found_idx + 1 == request_size
    load_tasks = []
    for i in range(layer_size):
        dst_tensors = [
            [dst_tensors1[j][i], dst_tensors2[j][i]] for j in range(request_size)
        ]
        task = worker.load(block_ids, [i] * request_size, dst_tensors)
        load_tasks.append(task)
    for task in load_tasks:
        worker.wait(task)
    cmp_and_print_diff(src_tensors1, dst_tensors1)
    cmp_and_print_diff(src_tensors2, dst_tensors2)


def main():
    tensor_size = [32768, 4096]
    layer_size = 64
    request_size = 64
    device_id = 0
    shard_size = sum(tensor_size)
    block_size = shard_size * layer_size
    config = {}
    config["store_pipeline"] = "Cache|Empty"
    config["unique_id"] = secrets.token_hex(8)
    config["tensor_size_list"] = tensor_size
    config["shard_size"] = shard_size
    config["block_size"] = block_size
    config["share_buffer_enable"] = True
    config["cache_buffer_capacity_gb"] = 4
    worker = UcmPipelineStore(config | {"device_id": device_id})
    scheduler = UcmPipelineStore(config)
    test_batch_number = 4
    for _ in range(test_batch_number):
        e2e_test(
            worker,
            scheduler,
            tensor_size,
            layer_size,
            request_size,
            device_id,
        )


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "debug"
    main()
