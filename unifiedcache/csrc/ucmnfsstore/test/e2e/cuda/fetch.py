# -*- coding: utf-8 -*-
import os
import random
import torch
import ucmnfsstore as ucmstore

storage_backends = [
    "/root/space/tmp",
]


def setup_uc(block_size, device_id):
    param = ucmstore.SetupParam(storage_backends, block_size, True)
    param.transferDeviceId = device_id
    param.transferStreamNumber = 32
    ret = ucmstore.Setup(param)
    assert ret == 0


def load_hashes(batch_size, batch_number):
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
            torch.rand([block_dim, block_len], dtype=torch.bfloat16, device="cuda:{}".format(device_id))
            for _ in range(block_layer)
        ]
        for _ in range(batch_size)
    ]
    return tensors


def fetch(hashes, tensors):
    data_id = []
    data_off = []
    data_addr = []
    data_len = []
    for hash_id, block in zip(hashes, tensors):
        offset = 0
        for layer in block:
            size = layer.untyped_storage().size()
            data_id.append(hash_id)
            data_addr.append(layer.data_ptr())
            data_off.append(offset)
            data_len.append(size)
            offset += size
    task = ucmstore.LoadToDevice(data_id, data_off, data_addr, data_len)
    assert task > 0
    ret = ucmstore.Wait(task)
    assert ret == 0


def main():
    device_id = 0
    block_dim = 576
    block_len = 128
    block_elem_size = 2
    block_layer = 61
    block_size = block_dim * block_len * block_elem_size * block_layer
    batch_size = 256
    batch_number = 64
    setup_uc(block_size, device_id)
    hashes = load_hashes(batch_size, batch_number)
    tensors = make_buffers(device_id, batch_size, block_dim, block_len, block_layer)
    for batch in range(batch_number):
        start = batch_size * batch
        end = start + batch_size
        fetch(hashes[start:end], tensors)


if __name__ == "__main__":
    main()
