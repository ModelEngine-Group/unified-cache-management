# -*- coding: utf-8 -*-
from unifiedcache.csrc.ucmnfsstore.output.lib import ucmnfsstore as ucmstore
import secrets
import torch

workspace = [
    "/root/space/kvcache.mthreads/unifiedcache/csrc/ucmnfsstore/sample/data",
]


def gen_tensor(dim: list, zero: bool):
    if zero:
        return torch.zeros(dim, dtype=torch.bfloat16)
    else:
        return torch.rand(dim, dtype=torch.bfloat16)


def setup_uc(block_size):
    param = ucmstore.SetupParam(workspace, block_size, True)
    ret = ucmstore.Setup(param)
    assert ret == 0


def dump_to_uc(hashes, tensors):
    ret = ucmstore.Alloc(hashes)
    assert ret == 0
    data_id = []
    data_off = []
    data_addr = []
    data_len = []
    for i in range(len(tensors)):
        block = tensors[i]
        offset = 0
        for j in range(len(block)):
            size = block[j].numel() * block[j].element_size()
            data_id.append(hashes[i])
            data_addr.append(block[j].data_ptr())
            data_off.append(offset)
            data_len.append(size)
            offset += size
    task_id = ucmstore.DumpFromHost(data_id, data_off, data_addr, data_len)
    assert task_id > 0
    ret = ucmstore.Wait(task_id)
    assert ret == 0
    ucmstore.Commit(hashes, True)


def fetch_from_uc(hashes, tensors):
    data_id = []
    data_off = []
    data_addr = []
    data_len = []
    for i in range(len(tensors)):
        block = tensors[i]
        offset = 0
        for j in range(len(block)):
            size = block[j].numel() * block[j].element_size()
            data_id.append(hashes[i])
            data_addr.append(block[j].data_ptr())
            data_off.append(offset)
            data_len.append(size)
            offset += size
    for _ in range(10):
        task_id = ucmstore.LoadToHost(data_id, data_off, data_addr, data_len)
        assert task_id > 0
        ret = ucmstore.Wait(task_id)
        assert ret == 0


def transfer_blocks(block_dim, block_len, block_layer, block_number):
    hashes = [secrets.token_hex(16) for _ in range(block_number)]
    tensors = [[gen_tensor([block_dim, block_len], False) for _ in range(block_layer)] for _ in range(block_number)]
    founds = ucmstore.Lookup(hashes)
    for found in founds:
        assert not found
    dump_to_uc(hashes, tensors)
    founds = ucmstore.Lookup(hashes)
    for found in founds:
        assert found
    tensors2 = [[gen_tensor([block_dim, block_len], True) for _ in range(block_layer)] for _ in range(block_number)]
    fetch_from_uc(hashes, tensors2)
    for i in range(block_number):
        for j in range(block_layer):
            assert tensors[i][j].equal(tensors2[i][j])


def main():
    block_dim = 576
    block_len = 128
    block_elem_size = 2
    block_layer = 61
    block_number = 4
    block_size = block_dim * block_len * block_elem_size * block_layer
    setup_uc(block_size)
    transfer_blocks(block_dim, block_len, block_layer, block_number)


if __name__ == "__main__":
    main()
