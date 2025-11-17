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
from typing import List

import torch

from ucm.store.pcstore.pcstore_connector import UcmPcStore
from ucm.store.ucmstore import UcmKVStoreBase


def setup_store(storage_backends, block_size, device_id, io_size) -> UcmKVStoreBase:
    config = {}
    config["storage_backends"] = storage_backends
    config["kv_block_size"] = block_size
    config["role"] = "worker"
    config["device"] = device_id
    config["io_size"] = io_size
    return UcmPcStore(config)


def make_buffers(
    block_number, device_id, batch_size, block_dim, block_len, block_layer
):
    hashes = [secrets.token_hex(16) for _ in range(block_number)]
    tensors = [
        [
            torch.rand(
                [block_dim, block_len],
                dtype=torch.bfloat16,
                device="cuda:{}".format(device_id),
            )
            for _ in range(block_layer)
        ]
        for _ in range(batch_size)
    ]
    return hashes, tensors


def embed(store: UcmKVStoreBase, hashes: List[str], tensors: List[List[torch.Tensor]]):
    results = store.create(hashes)
    assert sum(results) == 0
    block_ids = []
    offsets = []
    layers = []
    for hash_id, block in zip(hashes, tensors):
        offset = 0
        for layer in block:
            block_ids.append(hash_id)
            offsets.append(offset)
            layers.append(layer)
            offset += layer.untyped_storage().size()
    task = store.dump(block_ids, offsets, layers)
    assert task.task_id > 0
    ret = store.wait(task)
    assert ret == 0
    store.commit(hashes, True)


def fetch(store: UcmKVStoreBase, hashes: List[str], tensors: List[List[torch.Tensor]]):
    founds = store.lookup(hashes)
    for found in founds:
        assert found
    block_ids = []
    offsets = []
    layers = []
    for hash_id, block in zip(hashes, tensors):
        offset = 0
        for layer in block:
            block_ids.append(hash_id)
            offsets.append(offset)
            layers.append(layer)
            offset += layer.untyped_storage().size()
    task = store.load(block_ids, offsets, layers)
    assert task.task_id > 0
    ret = store.wait(task)
    assert ret == 0


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


def store_all_hashes(hashes):
    kvcache_block_hashes_file = "kvcache_block_hashes.txt"
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, kvcache_block_hashes_file)
    with open(file_path, "w", encoding="utf-8") as file:
        for hs in hashes:
            file.write(hs + "\n")


def main():
    storage_backends = "/home/share/mag1c/"
    block_number = 4096
    device_id = 1
    block_dim = 576
    block_len = 64
    block_elem_size = 2
    block_layer = 61
    io_size = block_dim * block_len * block_elem_size
    block_size = io_size * block_layer
    batch_size = 64
    store = setup_store(storage_backends, block_size, device_id, io_size)
    hashes, tensors = make_buffers(
        block_number, device_id, batch_size, block_dim, block_len, block_layer
    )
    total_batches = (block_number + batch_size - 1) // batch_size
    for batch in range(total_batches):
        start = batch_size * batch
        end = min(start + batch_size, block_number)
        tensors2 = [[torch.empty_like(t) for t in row] for row in tensors]
        embed(store, hashes[start:end], tensors)
        time.sleep(1)
        fetch(store, hashes[start:end], tensors2)
        cmp_and_print_diff(tensors, tensors2)
    store_all_hashes(hashes)


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "debug"
    main()
