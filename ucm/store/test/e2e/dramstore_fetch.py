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
import random
from typing import List

import torch

from ucm.store.dramstore.dramstore_connector import UcmDramStore
from ucm.store.ucmstore import UcmKVStoreBase


def setup_store(
    capacity, block_size, stream_number, device_id, timeout_ms
) -> UcmKVStoreBase:
    config = {}
    config["capacity"] = capacity
    config["kv_block_size"] = block_size
    config["stream_number"] = stream_number
    config["device_id"] = device_id
    config["timeout_ms"] = timeout_ms
    return UcmDramStore(config)


def get_hashes(batch_size, batch_number):
    kvcache_block_hashes_file = "kvcache_block_hashes.txt"
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, kvcache_block_hashes_file)
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    total = [line.strip() for line in lines]
    hashes = []
    for _ in range(batch_number):
        hashes.extend(random.sample(total, batch_size))
    return hashes


def make_buffers(device_id, batch_size, block_dim, block_len, block_layer):
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
    return tensors


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


def main():
    # storage_backends = "."
    block_number = 4096
    device_id = 1
    block_dim = 576
    block_len = 128
    block_elem_size = 2
    block_layer = 61
    io_size = block_dim * block_len * block_elem_size
    block_size = io_size * block_layer
    batch_size = 256
    batch_number = 64
    stream_number = 10
    timeout_ms = 1000000
    capacity = block_number * block_size * 2
    store = setup_store(capacity, block_size, stream_number, device_id, timeout_ms)
    hashes = get_hashes(batch_size, batch_number)
    tensors = make_buffers(device_id, batch_size, block_dim, block_len, block_layer)
    for batch in range(batch_number):
        start = batch_size * batch
        end = start + batch_size
        fetch(store, hashes[start:end], tensors)


if __name__ == "__main__":
    os.environ["UC_LOGGER_LEVEL"] = "debug"
    main()
