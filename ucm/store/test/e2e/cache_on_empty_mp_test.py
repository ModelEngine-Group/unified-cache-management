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
import secrets

import torch

from ucm.store.factory_v1 import UcmConnectorFactoryV1, UcmKVStoreBaseV1

worker_number = 8
shard_size = 64 * 1024
shard_number = 27
block_number = 64


def create_worker(unique_id, device_id) -> UcmKVStoreBaseV1:
    module_path = "ucm.store.pipeline.connector"
    class_name = "UcmPipelineStore"
    config = {}
    config["store_pipeline"] = "Cache|Empty"
    config["tensor_size"] = shard_size
    config["shard_size"] = shard_size * shard_number
    config["block_size"] = shard_size * shard_number
    config["device_id"] = device_id
    config["unique_id"] = unique_id
    config["share_buffer_enable"] = True
    config["cache_buffer_capacity_gb"] = 32
    return UcmConnectorFactoryV1.create_connector(class_name, config, module_path)


def make_tensors(device_id):
    return [
        [
            torch.rand(
                [shard_size // 2],
                dtype=torch.bfloat16,
                device="cuda:{}".format(device_id),
            )
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


def worker_routine(unique_id, device_id, barrier, block_ids, src_tensors):
    torch.cuda.set_device(device="cuda:{}".format(device_id))
    dst_tensors = make_tensors(device_id)
    worker = create_worker(unique_id, device_id)
    shard_indexes = [0 for _ in range(block_number)]
    if device_id == 0:
        task = worker.dump(block_ids, shard_indexes, src_tensors)
        worker.wait(task)
        print(f"Device({device_id}) dump data successfullyl: {src_tensors[0][0]}")
    barrier.wait()
    task = worker.load(block_ids, shard_indexes, dst_tensors)
    worker.wait(task)
    if device_id == 0:
        cmp_and_print_diff(src_tensors, dst_tensors)
    else:
        dst_tensors_on_0 = [[t.to("cuda:0") for t in row] for row in dst_tensors]
        cmp_and_print_diff(src_tensors, dst_tensors_on_0)
    print(f"Device({device_id}) load data successfullyl: {dst_tensors[0][0]}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    barrier = multiprocessing.Barrier(worker_number)
    unique_id = secrets.token_hex(8)
    workers = []
    block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
    block_tensors = make_tensors(0)
    for i in range(worker_number):
        p = multiprocessing.Process(
            target=worker_routine,
            args=(unique_id, i, barrier, block_ids, block_tensors),
        )
        workers.append(p)
        p.start()
    for w in workers:
        w.join()
