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
"""
Mooncake|Posix pipeline e2e test.

Tests:
  1. Correctness: dump → lookup → load → compare
  2. Posix fallback: dump → evict from Mooncake pool → load (should hit Posix)
  3. Bandwidth: dump + load throughput

Follows the same pattern as cache_on_posix_test.py but uses Mooncake as frontend
and Posix as backend persistence.

Prerequisites:
  1. mooncake master_service running (default 127.0.0.1:50088)
  2. libmooncakestore.so + libposixstore.so built
  3. NPU device available (ascend protocol)

Usage:
  python mooncake_on_posix_test.py
  python mooncake_on_posix_test.py --device-id 1
  python mooncake_on_posix_test.py --tensors-per-block 4
  MOONCAKE_MASTER=10.0.0.1:50088 python mooncake_on_posix_test.py
"""
import argparse
import os
import secrets
import shutil
import time

import numpy as np
import torch
import torch_npu

from ucm.store.pipeline.connector import UcmPipelineStore


def cmp_and_print_diff(a, b):
    for r, (row_a, row_b) in enumerate(zip(a, b)):
        for c, (ta, tb) in enumerate(zip(row_a, row_b)):
            if not torch.equal(ta, tb):
                diff_mask = ta != tb
                diff = diff_mask.sum().item()
                total = ta.numel()
                first_idx = diff_mask.nonzero(as_tuple=True)[0][0].item()
                print(
                    f"  FAIL at [{r}][{c}]: {diff}/{total} elements differ, "
                    f"first diff at index {first_idx}"
                )
                print(
                    f"    src[{first_idx}:{first_idx+8}]: "
                    f"{ta[first_idx:first_idx+8].cpu().tolist()}"
                )
                print(
                    f"    dst[{first_idx}:{first_idx+8}]: "
                    f"{tb[first_idx:first_idx+8].cpu().tolist()}"
                )
                assert False, "Data mismatch"


class NpuBufferPool:
    def __init__(self, block_count, tensor_size, tensors_per_block, device_id):
        self.block_count = block_count
        self.tensor_size = tensor_size
        self.tensors_per_block = tensors_per_block
        self.device = f"npu:{device_id}"
        total = block_count * tensors_per_block * tensor_size
        self.src_pool = torch.randint(
            0, 256, [total], dtype=torch.uint8, device=self.device
        )
        self.dst_pool = torch.zeros(total, dtype=torch.uint8, device=self.device)

    def register(self, store):
        store.register_memory(
            self.src_pool.data_ptr(),
            self.src_pool.numel() * self.src_pool.element_size(),
        )
        store.register_memory(
            self.dst_pool.data_ptr(),
            self.dst_pool.numel() * self.dst_pool.element_size(),
        )

    def _slice_block(self, pool, block_idx):
        base = block_idx * self.tensors_per_block * self.tensor_size
        return [
            pool[base + t * self.tensor_size : base + (t + 1) * self.tensor_size]
            for t in range(self.tensors_per_block)
        ]

    def src_tensors(self, count=None):
        n = count or self.block_count
        return [self._slice_block(self.src_pool, i) for i in range(n)]

    def dst_tensors(self, count=None):
        n = count or self.block_count
        return [self._slice_block(self.dst_pool, i) for i in range(n)]

    def clear_dst(self):
        self.dst_pool.zero_()

    @property
    def block_bytes(self):
        return self.tensors_per_block * self.tensor_size


def create_worker_store(
    device_id,
    tensor_size,
    tensors_per_block,
    local_hostname,
    master_address,
    metadata_server,
    protocol,
    global_segment_size,
    local_buffer_size,
    replica_num,
    storage_backends,
):
    tensor_size_list = [tensor_size] * tensors_per_block
    shard_size = tensor_size * tensors_per_block
    config = {
        "store_pipeline": "Mooncake|Posix",
        "unique_id": secrets.token_hex(8),
        "local_hostname": local_hostname,
        "master_server_address": master_address,
        "metadata_server": metadata_server,
        "protocol": protocol,
        "global_segment_size": global_segment_size,
        "local_buffer_size": local_buffer_size,
        "replica_num": replica_num,
        "tensor_size_list": tensor_size_list,
        "tensor_size": tensor_size,
        "shard_size": shard_size,
        "block_size": shard_size,
        "device_id": device_id,
        "storage_backends": storage_backends,
        "io_direct": False,
    }
    return UcmPipelineStore(config)


def create_scheduler_store(
    tensor_size,
    tensors_per_block,
    local_hostname,
    master_address,
    metadata_server,
    protocol,
    global_segment_size,
    local_buffer_size,
    replica_num,
    storage_backends,
):
    shard_size = tensor_size * tensors_per_block
    config = {
        "store_pipeline": "Mooncake|Posix",
        "unique_id": secrets.token_hex(8),
        "local_hostname": local_hostname,
        "master_server_address": master_address,
        "metadata_server": metadata_server,
        "protocol": protocol,
        "global_segment_size": global_segment_size,
        "local_buffer_size": local_buffer_size,
        "replica_num": replica_num,
        "tensor_size": tensor_size,
        "shard_size": shard_size,
        "block_size": shard_size,
        "storage_backends": storage_backends,
        "io_direct": False,
    }
    return UcmPipelineStore(config)


def test_correctness(worker, scheduler, pool, block_count):
    ts = pool.tensor_size
    tpb = pool.tensors_per_block
    print(
        f"\n=== Correctness Test: {block_count} blocks x {tpb} tensors x {ts} bytes ==="
    )

    block_ids = [secrets.token_bytes(16) for _ in range(block_count)]
    shard_indexes = [0] * block_count

    src = pool.src_tensors(block_count)
    torch.npu.synchronize()
    src_cpu = [[t.cpu().clone() for t in row] for row in src]

    # lookup before dump — all miss
    founds = scheduler.lookup(block_ids)
    assert not any(founds), f"Expected all miss, got {sum(founds)} hits"
    print("  lookup (before dump): all miss — OK")

    prefix_idx = scheduler.lookup_on_prefix(block_ids)
    assert prefix_idx == -1, f"Expected -1, got {prefix_idx}"
    print("  lookup_on_prefix (before dump): -1 — OK")

    # dump (writes to Mooncake pool + async persist to Posix)
    task = worker.dump(block_ids, shard_indexes, src)
    worker.wait(task)
    torch.npu.synchronize()
    print("  dump: OK")

    # lookup after dump — all hit
    founds = scheduler.lookup(block_ids)
    assert all(founds), f"Expected all hit, got {sum(founds)}/{block_count}"
    print("  lookup (after dump): all hit — OK")

    prefix_idx = scheduler.lookup_on_prefix(block_ids)
    assert (
        prefix_idx + 1 == block_count
    ), f"Expected {block_count - 1}, got {prefix_idx}"
    print(f"  lookup_on_prefix (after dump): {prefix_idx} — OK")

    # load back
    pool.clear_dst()
    torch.npu.synchronize()
    dst = pool.dst_tensors(block_count)
    task = worker.load(block_ids, shard_indexes, dst)
    worker.wait(task)
    torch.npu.synchronize()
    print("  load: OK")

    # compare
    dst_cpu = [[t.cpu() for t in row] for row in dst]
    cmp_and_print_diff(src_cpu, dst_cpu)
    print("  data compare: all match — OK")


def test_posix_fallback(worker, scheduler, pool, block_count):
    ts = pool.tensor_size
    tpb = pool.tensors_per_block
    print(
        f"\n=== Posix Fallback Test: {block_count} blocks x {tpb} tensors x {ts} bytes ==="
    )

    block_ids = [secrets.token_bytes(16) for _ in range(block_count)]
    shard_indexes = [0] * block_count

    src = pool.src_tensors(block_count)
    torch.npu.synchronize()
    src_cpu = [[t.cpu().clone() for t in row] for row in src]

    # dump (Mooncake pool + Posix persist)
    task = worker.dump(block_ids, shard_indexes, src)
    worker.wait(task)
    torch.npu.synchronize()
    print("  dump: OK")

    # wait a bit for async Posix persist to complete
    time.sleep(2)

    # verify lookup hits (data in Mooncake pool)
    founds = scheduler.lookup(block_ids)
    assert all(founds), f"Expected all hit, got {sum(founds)}/{block_count}"
    print("  lookup (Mooncake pool): all hit — OK")

    # NOTE: To truly test Posix fallback, we would need to evict data from
    # Mooncake pool (e.g. by filling it up or calling remove). For now we
    # just verify the dump+load path works end-to-end. A full eviction test
    # requires Mooncake master eviction or manual remove API.
    print("  (full Posix fallback test requires Mooncake eviction — skipped)")

    # load back and verify
    pool.clear_dst()
    torch.npu.synchronize()
    dst = pool.dst_tensors(block_count)
    task = worker.load(block_ids, shard_indexes, dst)
    worker.wait(task)
    torch.npu.synchronize()

    dst_cpu = [[t.cpu() for t in row] for row in dst]
    cmp_and_print_diff(src_cpu, dst_cpu)
    print("  data compare: all match — OK")


def test_bandwidth(worker, pool, epochs):
    bc = pool.block_count
    ts = pool.tensor_size
    tpb = pool.tensors_per_block
    total_bytes = ts * tpb * bc
    print(
        f"\n=== Bandwidth Test: {bc} blocks x {tpb} tensors x {ts} bytes, "
        f"{epochs} epochs ==="
    )

    shard_indexes = [0] * bc
    src = pool.src_tensors()
    dst = pool.dst_tensors()
    block_ids = [secrets.token_bytes(16) for _ in range(bc)]

    # dump (single shot)
    torch.npu.synchronize()
    tp = time.perf_counter()
    task = worker.dump(block_ids, shard_indexes, src)
    worker.wait(task)
    torch.npu.synchronize()
    dump_cost = time.perf_counter() - tp
    print(
        f"  dump: cost={dump_cost * 1e3:.3f}ms, "
        f"bw={total_bytes / dump_cost / 1e9:.3f} GB/s"
    )

    # load (repeated reads)
    load_costs = []
    for epoch in range(epochs):
        pool.clear_dst()
        torch.npu.synchronize()
        tp = time.perf_counter()
        task = worker.load(block_ids, shard_indexes, dst)
        worker.wait(task)
        torch.npu.synchronize()
        cost = time.perf_counter() - tp
        load_costs.append(cost)
        print(
            f"  load epoch={epoch:03d}, cost={cost * 1e3:.3f}ms, "
            f"bw={total_bytes / cost / 1e9:.3f} GB/s"
        )

    load_avg = np.mean(load_costs)
    load_p50 = np.percentile(load_costs, 50)
    load_p99 = np.percentile(load_costs, 99)
    print(
        f"  load summary: avg={load_avg * 1e3:.3f}ms, "
        f"p50={load_p50 * 1e3:.3f}ms, p99={load_p99 * 1e3:.3f}ms, "
        f"avg_bw={total_bytes / load_avg / 1e9:.3f} GB/s"
    )


def main():
    parser = argparse.ArgumentParser(description="Mooncake|Posix pipeline e2e test")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--tensor-size",
        type=int,
        default=8 * 1024 * 1024,
        help="bytes per tensor (default 8MB)",
    )
    parser.add_argument("--tensors-per-block", type=int, default=2)
    parser.add_argument(
        "--block-count", type=int, default=16, help="number of blocks per batch"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--correctness-blocks", type=int, default=4)
    parser.add_argument(
        "--storage-dir", type=str, default="./build/mooncake_posix_test_data"
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument("--skip-bandwidth", action="store_true")
    args = parser.parse_args()

    local_hostname = os.environ.get("MOONCAKE_LOCAL_HOST", "127.0.0.1:12345")
    master_address = os.environ.get("MOONCAKE_MASTER", "127.0.0.1:50088")
    metadata_server = os.environ.get("MOONCAKE_METADATA", "P2PHANDSHAKE")
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "ascend")
    global_segment_size = int(
        os.environ.get("MOONCAKE_SEGMENT_SIZE", str(512 * 1024 * 1024))
    )
    local_buffer_size = int(
        os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", str(256 * 1024 * 1024))
    )
    replica_num = int(os.environ.get("MOONCAKE_REPLICA_NUM", "1"))
    device_id = args.device_id

    # Prepare Posix storage directory
    storage_backends = [args.storage_dir]
    os.makedirs(args.storage_dir, exist_ok=True)

    # NPU context
    torch.npu.set_device(device_id)
    print(f"NPU device {device_id} initialized")

    print("=" * 60)
    print("Mooncake|Posix Pipeline E2E Test")
    print("=" * 60)
    print(f"  local_hostname     = {local_hostname}")
    print(f"  master_address     = {master_address}")
    print(f"  protocol           = {protocol}")
    print(f"  global_segment_size= {global_segment_size}")
    print(f"  local_buffer_size  = {local_buffer_size}")
    print(f"  device_id          = {device_id}")
    print(f"  tensor_size        = {args.tensor_size}")
    print(f"  tensors_per_block  = {args.tensors_per_block}")
    print(f"  block_count        = {args.block_count}")
    print(f"  storage_dir        = {args.storage_dir}")
    print()

    common_args = dict(
        tensor_size=args.tensor_size,
        tensors_per_block=args.tensors_per_block,
        local_hostname=local_hostname,
        master_address=master_address,
        metadata_server=metadata_server,
        protocol=protocol,
        global_segment_size=global_segment_size,
        local_buffer_size=local_buffer_size,
        replica_num=replica_num,
        storage_backends=storage_backends,
    )

    worker = create_worker_store(device_id=device_id, **common_args)
    scheduler = create_scheduler_store(**common_args)

    # Allocate NPU buffer pool and register
    max_blocks = max(args.correctness_blocks, args.block_count)
    pool = NpuBufferPool(
        max_blocks, args.tensor_size, args.tensors_per_block, device_id
    )
    pool.register(worker)
    print(
        f"  Registered NPU buffer pool: {max_blocks} x "
        f"{args.tensors_per_block} tensors x {args.tensor_size} bytes"
    )

    if not args.skip_correctness:
        test_correctness(worker, scheduler, pool, args.correctness_blocks)

    if not args.skip_fallback:
        test_posix_fallback(worker, scheduler, pool, args.correctness_blocks)

    if not args.skip_bandwidth:
        test_bandwidth(worker, pool, args.epochs)

    print("\n=== All tests passed ===")

    # Cleanup Posix test data
    if os.path.exists(args.storage_dir):
        shutil.rmtree(args.storage_dir, ignore_errors=True)
        print(f"  Cleaned up {args.storage_dir}")


if __name__ == "__main__":
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")
    main()
