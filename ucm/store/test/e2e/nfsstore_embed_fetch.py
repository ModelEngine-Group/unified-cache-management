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
import csv
import math
import os
import secrets
import time
from typing import Dict, List, Tuple

import torch

from ucm.store.nfsstore.nfsstore_connector import UcmNfsStore
from ucm.store.ucmstore import UcmKVStoreBase


def setup(
    storage_backends,
    block_size,
    device_id,
    io_size,
    transferStreamNumber,
    transferIoDirect,
) -> UcmKVStoreBase:
    config = {
        "storage_backends": storage_backends,
        "kv_block_size": block_size,
        "role": "worker",
        "device": device_id,
        "io_size": io_size,
        "transferStreamNumber": transferStreamNumber,
        "transferIoDirect": transferIoDirect,
    }
    return UcmNfsStore(config)


def make_aligned_tensor(shape, dtype, device, alignment=4096):
    numl = math.prod(shape)
    dtype_size = torch.tensor(1, dtype=dtype).element_size()
    total_byters = numl * dtype_size

    padded_bytes = total_byters + alignment
    storage = torch.ByteTensor(padded_bytes).to(device)

    ptr = storage.data_ptr()
    offset = ptr % alignment
    if offset != 0:
        aligned_ptr = ptr + (alignment - offset)
    else:
        aligned_ptr = ptr

    aligned_storage = storage[(aligned_ptr - ptr) :].view(dtype)
    tensor = aligned_storage[:numl].view(shape)
    tensor.storage_ref = storage
    return tensor


def make_buffers(
    block_number, device_id, batch_size, head_dim, block_len, block_layer, num_head, kv
):
    hashes = [secrets.token_hex(16) for _ in range(block_number)]
    kv_caches = {}
    for i in range(block_layer):
        kv_caches[i] = make_aligned_tensor(
            [kv, block_number, block_len, num_head, head_dim],
            dtype=torch.float16,
            device=f"cuda:{device_id}",
        )
    return hashes, kv_caches


def store_all_hashes(hashes: List[str]):
    file_path = os.path.join(os.path.dirname(__file__), "kvcache_block_hashes.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for h in hashes:
            f.write(h + "\n")


def load_hashes_from_file() -> List[str]:
    file_path = os.path.join(os.path.dirname(__file__), "kvcache_block_hashes.txt")
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def embed(
    store: UcmKVStoreBase,
    hashes: List[str],
    kvcaches: Dict[int, torch.Tensor],
    mla: bool,
):
    start_time = time.perf_counter()

    total_block_ids, total_offsets, total_tensors = [], [], []
    total_size = 0

    for i, hash_val in enumerate(hashes):
        offset = 0
        for layer_id, kv_layer in kvcaches.items():
            k_tensor = kv_layer[0][i]  # kv=1
            total_tensors.append(k_tensor)
            total_block_ids.append(hash_val)
            total_offsets.append(offset)
            sz = k_tensor.numel() * k_tensor.element_size()
            offset += sz
            total_size += sz

            if not mla:
                v_tensor = kv_layer[1][i]
                total_tensors.append(v_tensor)
                total_block_ids.append(hash_val)
                total_offsets.append(offset)
                sz = v_tensor.numel() * v_tensor.element_size()
                offset += sz
                total_size += sz

    task = store.dump(total_block_ids, total_offsets, total_tensors)
    store.wait(task)

    elapsed_time = time.perf_counter() - start_time
    throughput_gbps = (total_size / (1024**3)) / elapsed_time if elapsed_time > 0 else 0

    print(
        f"WRITE: Data Size={(total_size / (1024 ** 3)):.4f} GB, Time={elapsed_time:.4f} s, "
        f"Speed={throughput_gbps:.4f} GB/s"
    )

    return total_size, elapsed_time, throughput_gbps


def fetch(
    store: UcmKVStoreBase,
    hashes: List[str],
    kvcaches: Dict[int, torch.Tensor],
    mla: bool,
):
    start_time = time.perf_counter()

    founds = store.lookup(hashes)
    for f in founds:
        assert f, "Cache block miss detected"

    block_ids, offsets, tensors = [], [], []
    total_size = 0

    for i, hash_val in enumerate(hashes):
        offset = 0
        for layer_id, kv_layer in kvcaches.items():
            k_tensor = kv_layer[0][i]  # kv=1
            block_ids.append(hash_val)
            offsets.append(offset)
            tensors.append(k_tensor)
            sz = k_tensor.numel() * k_tensor.element_size()
            offset += sz
            total_size += sz

            if not mla:
                v_tensor = kv_layer[1][i]
                block_ids.append(hash_val)
                offsets.append(offset)
                tensors.append(v_tensor)
                sz = v_tensor.numel() * v_tensor.element_size()
                offset += sz
                total_size += sz

    task = store.load(block_ids, offsets, tensors)
    ret = store.wait(task)
    assert ret == 0, "Load operation failed"

    elapsed_time = time.perf_counter() - start_time
    throughput_gbps = (total_size / (1024**3)) / elapsed_time if elapsed_time > 0 else 0

    print(
        f"READ: Data Size={(total_size / (1024 ** 3)):.4f} GB, Time={elapsed_time:.4f} s, "
        f"Speed={throughput_gbps:.4f} GB/s"
    )

    return total_size, elapsed_time, throughput_gbps


def run(
    storage_backends: str,
    device_id: int,
    repeat: int,
    num_head: int,
    block_len: int,
    transferStreamNumber: int,
    num_tokens: int,
    block_layer: int,
    head_size: int,
    block_elem_size: int,
    kv: int,
    mla: bool,
    transferIoDirect: bool,
    operation_mode: str = "both",  #  "write_only", "read_only", or "both"
) -> Tuple[float, float, float, float, float, float]:
    """
    Run a single test with given parameters and return performance metrics.

    Returns:
        Tuple of (avg_w_size, avg_w_time, avg_w_bw, avg_r_time, avg_r_bw, avg_r_size)
    """

    block_dim = head_size * num_head
    io_size = block_dim * block_len * block_elem_size
    block_size = io_size * block_layer
    batch_size = int(num_tokens / block_len)
    real_blocks = batch_size + 10

    w_bw_list, r_bw_list = [], []
    w_time_list, r_time_list = [], []
    w_size_sum, r_size_sum = 0.0, 0.0

    store = setup(
        storage_backends,
        block_size,
        device_id,
        io_size,
        transferStreamNumber,
        transferIoDirect,
    )

    for r in range(repeat):
        print(f"\n--- Round {r+1} ---")

        if operation_mode in ["write_only", "both"]:
            hashes, kvcaches = make_buffers(
                real_blocks,
                device_id,
                batch_size,
                head_size,
                block_len,
                block_layer,
                num_head,
                kv,
            )

            results = store.create(hashes[:batch_size])
            assert sum(results) == 0, "Create operation failed"

            w_size, w_time, w_bw = embed(
                store,
                hashes[:batch_size],
                kvcaches,
                mla,
            )
            store.commit(hashes[:batch_size], True)

            if r == 0:
                store_all_hashes(hashes[:batch_size])

            w_bw_list.append(w_bw)
            w_time_list.append(w_time)
            w_size_sum += w_size

            if operation_mode == "write_only":
                del kvcaches, hashes
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.empty_cache()

        if operation_mode in ["read_only", "both"]:
            if operation_mode == "read_only":
                saved_hashes = load_hashes_from_file()
                if not saved_hashes:
                    raise RuntimeError("No saved hashes found for read operation")

                _, kvcaches = make_buffers(
                    real_blocks,
                    device_id,
                    batch_size,
                    head_size,
                    block_len,
                    block_layer,
                    num_head,
                    kv,
                )

                r_size, r_time, r_bw = fetch(
                    store,
                    saved_hashes[:batch_size],
                    kvcaches,
                    mla,
                )
            else:
                r_size, r_time, r_bw = fetch(
                    store,
                    hashes[:batch_size],
                    kvcaches,
                    mla,
                )

            r_bw_list.append(r_bw)
            r_time_list.append(r_time)
            r_size_sum += r_size

            if operation_mode == "read_only":
                del kvcaches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.empty_cache()
            else:
                del kvcaches, hashes
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.empty_cache()

    del store
    avg_w_bw = sum(w_bw_list) / len(w_bw_list) if w_bw_list else 0.0
    avg_r_bw = sum(r_bw_list) / len(r_bw_list) if r_bw_list else 0.0
    avg_w_time = sum(w_time_list) / len(w_time_list) if w_time_list else 0.0
    avg_r_time = sum(r_time_list) / len(r_time_list) if r_time_list else 0.0
    avg_w_size = w_size_sum / (1024**3) / len(w_time_list) if w_time_list else 0.0
    avg_r_size = r_size_sum / (1024**3) / len(r_time_list) if r_time_list else 0.0

    return avg_w_size, avg_w_time, avg_w_bw, avg_r_time, avg_r_bw, avg_r_size
