#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# Compare io_uring transfer performance on:
#   1) local SSD path
#   2) NFS over RDMA path
#
# Example:
# python ucm/store/test/e2e/compare_uring_local_vs_nfs.py \
#   --local-backend /mnt/local_ssd/ucm \
#   --nfs-backend /mnt/nfs_rdma/ucm \
#   --iterations 20 --warmup 3 --batch-size 256 --block-size 1048576

import argparse
import mmap
import os
import secrets
import statistics
import time
from dataclasses import dataclass

import numpy as np

from ucm.store.pipeline.connector import UcmKVStoreBaseV1, UcmPipelineStore


def make_array(size: int, alignment: int = 4096, dtype=np.uint8) -> np.ndarray:
    item_size = np.dtype(dtype).itemsize
    total_bytes = size * item_size
    mm = mmap.mmap(-1, total_bytes + alignment)
    raw = np.frombuffer(mm, dtype=np.uint8, count=total_bytes + alignment)
    raw_ptr = raw.__array_interface__["data"][0]
    aligned_addr = (raw_ptr + alignment - 1) & ~(alignment - 1)
    offset = aligned_addr - raw_ptr
    arr = raw[offset : offset + total_bytes].view(dtype=dtype)
    return arr


def setup_store(
    backend: str,
    block_size: int,
    data_trans_concur: int,
    lookup_concur: int,
    io_direct: bool,
    use_io_uring: bool,
    worker: bool,
) -> UcmKVStoreBaseV1:
    cfg = {
        "store_pipeline": "Posix",
        "storage_backends": [backend],
        "tensor_size": block_size,
        "shard_size": block_size,
        "block_size": block_size,
        "posix_data_trans_concurrency": data_trans_concur,
        "posix_lookup_concurrency": lookup_concur,
        "io_direct": io_direct,
        "device_id": 0 if worker else -1,
        "posix_use_io_uring": use_io_uring,
    }
    return UcmPipelineStore(cfg)


@dataclass
class BackendResult:
    name: str
    dump_ms: list[float]
    load_ms: list[float]
    dump_bw_gbps: list[float]
    load_bw_gbps: list[float]


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), p))


def run_backend_case(
    name: str,
    backend_root: str,
    iterations: int,
    warmup: int,
    block_size: int,
    batch_size: int,
    data_trans_concur: int,
    lookup_concur: int,
    io_direct: bool,
    use_io_uring: bool,
) -> BackendResult:
    run_dir = os.path.join(backend_root, f"uring_demo_{name}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    worker = setup_store(
        backend=run_dir,
        block_size=block_size,
        data_trans_concur=data_trans_concur,
        lookup_concur=lookup_concur,
        io_direct=io_direct,
        use_io_uring=use_io_uring,
        worker=True,
    )
    scheduler = setup_store(
        backend=run_dir,
        block_size=block_size,
        data_trans_concur=data_trans_concur,
        lookup_concur=lookup_concur,
        io_direct=io_direct,
        use_io_uring=use_io_uring,
        worker=False,
    )

    raw_src = [make_array(block_size) for _ in range(batch_size)]
    raw_dst = [make_array(block_size) for _ in range(batch_size)]
    src_ptrs = [[arr.ctypes.data] for arr in raw_src]
    dst_ptrs = [[arr.ctypes.data] for arr in raw_dst]
    shard_indices = [0 for _ in range(batch_size)]
    payload_bytes = block_size * batch_size

    dump_ms: list[float] = []
    load_ms: list[float] = []
    dump_bw: list[float] = []
    load_bw: list[float] = []

    total_rounds = warmup + iterations
    for round_idx in range(total_rounds):
        block_ids = [secrets.token_bytes(16) for _ in range(batch_size)]

        t0 = time.perf_counter()
        dump_h = worker.dump_data(block_ids, shard_indices, src_ptrs)
        worker.wait(dump_h)
        dump_cost = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        load_h = worker.load_data(block_ids, shard_indices, dst_ptrs)
        worker.wait(load_h)
        load_cost = (time.perf_counter() - t1) * 1000.0

        if round_idx >= warmup:
            dump_ms.append(dump_cost)
            load_ms.append(load_cost)
            dump_bw.append(payload_bytes / (dump_cost / 1000.0) / 1e9)
            load_bw.append(payload_bytes / (load_cost / 1000.0) / 1e9)

    # Keep explicit references till benchmark ends.
    _ = scheduler
    _ = worker
    _ = raw_src
    _ = raw_dst

    return BackendResult(
        name=name,
        dump_ms=dump_ms,
        load_ms=load_ms,
        dump_bw_gbps=dump_bw,
        load_bw_gbps=load_bw,
    )


def print_result(res: BackendResult) -> None:
    print(f"\n=== {res.name} ===")
    print(
        "dump(ms): "
        f"mean={statistics.mean(res.dump_ms):.3f}, "
        f"p50={pct(res.dump_ms, 50):.3f}, "
        f"p95={pct(res.dump_ms, 95):.3f}"
    )
    print(
        "load(ms): "
        f"mean={statistics.mean(res.load_ms):.3f}, "
        f"p50={pct(res.load_ms, 50):.3f}, "
        f"p95={pct(res.load_ms, 95):.3f}"
    )
    print(
        "dump(GB/s): "
        f"mean={statistics.mean(res.dump_bw_gbps):.3f}, "
        f"p50={pct(res.dump_bw_gbps, 50):.3f}, "
        f"p95={pct(res.dump_bw_gbps, 5):.3f} (lower is worse)"
    )
    print(
        "load(GB/s): "
        f"mean={statistics.mean(res.load_bw_gbps):.3f}, "
        f"p50={pct(res.load_bw_gbps, 50):.3f}, "
        f"p95={pct(res.load_bw_gbps, 5):.3f} (lower is worse)"
    )


def print_compare(local_res: BackendResult, nfs_res: BackendResult) -> None:
    local_dump = statistics.mean(local_res.dump_bw_gbps)
    nfs_dump = statistics.mean(nfs_res.dump_bw_gbps)
    local_load = statistics.mean(local_res.load_bw_gbps)
    nfs_load = statistics.mean(nfs_res.load_bw_gbps)
    print("\n=== NFS / Local ratio (mean BW) ===")
    print(f"dump ratio: {nfs_dump / local_dump:.3f}x")
    print(f"load ratio: {nfs_load / local_load:.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare io_uring performance between local SSD and NFS over RDMA."
    )
    parser.add_argument("--local-backend", required=True, help="Local SSD backend directory")
    parser.add_argument("--nfs-backend", required=True, help="NFS over RDMA backend directory")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=1024 * 1024)
    parser.add_argument("--data-trans-concur", type=int, default=2)
    parser.add_argument("--lookup-concur", type=int, default=8)
    parser.add_argument("--io-direct", action="store_true", default=False)
    parser.add_argument("--disable-uring", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("UC_LOGGER_LEVEL", "warn")

    use_uring = not args.disable_uring
    print(
        "Config: "
        f"use_io_uring={use_uring}, io_direct={args.io_direct}, "
        f"iterations={args.iterations}, warmup={args.warmup}, "
        f"batch_size={args.batch_size}, block_size={args.block_size}"
    )

    local = run_backend_case(
        name="local",
        backend_root=args.local_backend,
        iterations=args.iterations,
        warmup=args.warmup,
        block_size=args.block_size,
        batch_size=args.batch_size,
        data_trans_concur=args.data_trans_concur,
        lookup_concur=args.lookup_concur,
        io_direct=args.io_direct,
        use_io_uring=use_uring,
    )
    nfs = run_backend_case(
        name="nfs",
        backend_root=args.nfs_backend,
        iterations=args.iterations,
        warmup=args.warmup,
        block_size=args.block_size,
        batch_size=args.batch_size,
        data_trans_concur=args.data_trans_concur,
        lookup_concur=args.lookup_concur,
        io_direct=args.io_direct,
        use_io_uring=use_uring,
    )

    print_result(local)
    print_result(nfs)
    print_compare(local, nfs)


if __name__ == "__main__":
    main()
