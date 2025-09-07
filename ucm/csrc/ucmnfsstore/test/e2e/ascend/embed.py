# -*- coding: utf-8 -*-
import os
import secrets

import torch
import torch_npu
import ucmnfsstore as ucmstore

# 预埋的KVCacheBlock个数
kvcache_block_number = 4096
# KVCacheBlock数据的存放位置，可以是多个位置，但指向的远端目录必须是同一个
storage_backends = [
    "/root/space/tmp_data",
]


def setup_uc(block_size, device_id):
    param = ucmstore.SetupParam(storage_backends, block_size, True)
    param.transferDeviceId = device_id
    ret = ucmstore.Setup(param)
    assert ret == 0


def make_buffers(
    block_number, device_id, batch_size, block_dim, block_len, block_layer
):
    hashes = [secrets.token_hex(16) for _ in range(block_number)]
    tensors = [
        [
            torch.rand(
                [block_dim, block_len],
                dtype=torch.bfloat16,
                device="npu:{}".format(device_id),
            )
            for _ in range(block_layer)
        ]
        for _ in range(batch_size)
    ]
    return hashes, tensors


def embed(hashes, tensors):
    ret = ucmstore.AllocBatch(hashes)
    assert sum(ret) == 0
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
    task_id = ucmstore.DumpFromDevice(data_id, data_off, data_addr, data_len)
    assert task_id > 0
    ret = ucmstore.Wait(task_id)
    assert ret == 0
    ucmstore.CommitBatch(hashes, True)


def store_all_hashes(hashes):
    kvcache_block_hashes_file = "kvcache_block_hashes.txt"
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, kvcache_block_hashes_file)
    with open(file_path, "w", encoding="utf-8") as file:
        for hs in hashes:
            file.write(hs + "\n")


def main():
    device_id = 7
    block_dim = 576
    block_len = 128
    block_elem_size = 2
    block_layer = 61
    block_size = block_dim * block_len * block_elem_size * block_layer
    batch_size = 256
    setup_uc(block_size, device_id)
    hashes, tensors = make_buffers(
        kvcache_block_number, device_id, batch_size, block_dim, block_len, block_layer
    )
    total_batches = (kvcache_block_number + batch_size - 1) // batch_size
    for batch in range(total_batches):
        start = batch_size * batch
        end = min(start + batch_size, kvcache_block_number)
        embed(hashes[start:end], tensors)
    store_all_hashes(hashes)


if __name__ == "__main__":
    os.environ["UCMSTORE_LOGGER_LEVEL"] = "info"
    main()
