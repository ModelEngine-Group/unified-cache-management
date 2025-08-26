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
# -*- coding: utf-8 -*-
import secrets

import torch
import torch_npu

from unifiedcache.csrc.ucmnfsstore.output.lib import ucmnfsstore as ucmstore

workspace = [
    "/root/space/kvcache.mthreads/unifiedcache/csrc/ucmnfsstore/sample/data",
]


def gen_tensor(dim: list, zero: bool, device_id: int):
    if zero:
        return torch.zeros(dim, dtype=torch.bfloat16, device=f"npu:{device_id}")
    else:
        return torch.rand(dim, dtype=torch.bfloat16, device=f"npu:{device_id}")


def setup_uc(block_size, device_id):
    param = ucmstore.SetupParam(workspace, block_size, True)
    param.transferDeviceId = device_id
    ret = ucmstore.Setup(param)
    assert ret == 0


def dump_to_uc(hashes, tensors):
    cpu_tensors = [[layer.cpu() for layer in block] for block in tensors]
    ret = ucmstore.AllocBatch(hashes)
    assert ret == 0
    data_id = []
    data_off = []
    data_addr = []
    data_len = []
    for i in range(len(cpu_tensors)):
        block = cpu_tensors[i]
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
    ucmstore.CommitBatch(hashes, True)


def fetch_from_uc(hashes, tensors):
    cpu_tensors = [
        [torch.empty_like(layer, device="cpu") for layer in block] for block in tensors
    ]
    data_id = []
    data_off = []
    data_addr = []
    data_len = []
    for i in range(len(cpu_tensors)):
        block = cpu_tensors[i]
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
    for i in range(len(tensors)):
        for j in range(len(tensors[i])):
            tensors[i][j].copy_(cpu_tensors[i][j])


def transfer_blocks(block_dim, block_len, block_layer, block_number, device_id):
    hashes = [secrets.token_hex(16) for _ in range(block_number)]
    tensors = [
        [
            gen_tensor([block_dim, block_len], False, device_id)
            for _ in range(block_layer)
        ]
        for _ in range(block_number)
    ]
    founds = ucmstore.LookupBatch(hashes)
    for found in founds:
        assert not found
    dump_to_uc(hashes, tensors)
    founds = ucmstore.LookupBatch(hashes)
    for found in founds:
        assert found
    tensors2 = [
        [
            gen_tensor([block_dim, block_len], True, device_id)
            for _ in range(block_layer)
        ]
        for _ in range(block_number)
    ]
    fetch_from_uc(hashes, tensors2)
    for i in range(block_number):
        for j in range(block_layer):
            assert tensors[i][j].equal(tensors2[i][j])


def main():
    device_id = 7
    block_dim = 576
    block_len = 128
    block_elem_size = 2
    block_layer = 61
    block_number = 4
    block_size = block_dim * block_len * block_elem_size * block_layer
    setup_uc(block_size, device_id)
    transfer_blocks(block_dim, block_len, block_layer, block_number, device_id)


if __name__ == "__main__":
    main()
