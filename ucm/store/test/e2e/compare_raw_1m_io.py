#!/usr/bin/env python3
# MIT License
#
# Raw 1MiB IO benchmark for local SSD vs NFS-over-RDMA.
#
# Example:
# python ucm/store/test/e2e/compare_raw_1m_io.py \
#   --local-dir /mnt/local_ssd/bench \
#   --nfs-dir /mnt/nfs_rdma/bench \
#   --file-size-mb 4096 \
#   --loops 2

import argparse
import hashlib
import os
import statistics
import time
from dataclasses import dataclass

import numpy as np


KB = 1024
MB = 1024 * KB
BLOCK_SIZE = 1 * MB


@dataclass
class IoStats:
    write_ms: list[float]
    read_ms: list[float]
    write_bw_gbps: float
    read_bw_gbps: float
    checksum: str


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def run_one_path(target_dir: str, file_size_mb: int, loops: int, fsync: bool) -> IoStats:
    os.makedirs(target_dir, exist_ok=True)
    total_bytes = file_size_mb * MB
    n_blocks = total_bytes // BLOCK_SIZE
    if n_blocks == 0:
        raise ValueError("--file-size-mb must be >= 1")
    total_bytes = n_blocks * BLOCK_SIZE

    payload = os.urandom(BLOCK_SIZE)
    read_buf_size = BLOCK_SIZE
    write_lat_ms: list[float] = []
    read_lat_ms: list[float] = []
    total_write_s = 0.0
    total_read_s = 0.0
    digest = hashlib.sha256()

    for loop_idx in range(loops):
        file_path = os.path.join(target_dir, f"raw_1m_io_loop{loop_idx}.bin")
        fd = os.open(file_path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o644)
        try:
            start_write = time.perf_counter()
            for blk in range(n_blocks):
                off = blk * BLOCK_SIZE
                t0 = time.perf_counter()
                n = os.pwrite(fd, payload, off)
                t1 = time.perf_counter()
                if n != BLOCK_SIZE:
                    raise RuntimeError(f"short pwrite: {n} != {BLOCK_SIZE}")
                write_lat_ms.append((t1 - t0) * 1000.0)
            if fsync:
                os.fdatasync(fd)
            total_write_s += time.perf_counter() - start_write

            start_read = time.perf_counter()
            for blk in range(n_blocks):
                off = blk * BLOCK_SIZE
                t0 = time.perf_counter()
                data = os.pread(fd, read_buf_size, off)
                t1 = time.perf_counter()
                if len(data) != BLOCK_SIZE:
                    raise RuntimeError(f"short pread: {len(data)} != {BLOCK_SIZE}")
                read_lat_ms.append((t1 - t0) * 1000.0)
                digest.update(data[:64])
            total_read_s += time.perf_counter() - start_read
        finally:
            os.close(fd)

    write_bw = (total_bytes * loops) / total_write_s / 1e9
    read_bw = (total_bytes * loops) / total_read_s / 1e9
    return IoStats(
        write_ms=write_lat_ms,
        read_ms=read_lat_ms,
        write_bw_gbps=write_bw,
        read_bw_gbps=read_bw,
        checksum=digest.hexdigest()[:16],
    )


def show(name: str, stats: IoStats) -> None:
    print(f"\n=== {name} ===")
    print(
        f"write BW: {stats.write_bw_gbps:.3f} GB/s | "
        f"lat p50/p95/p99: {pct(stats.write_ms, 50):.3f}/{pct(stats.write_ms, 95):.3f}/{pct(stats.write_ms, 99):.3f} ms"
    )
    print(
        f"read  BW: {stats.read_bw_gbps:.3f} GB/s | "
        f"lat p50/p95/p99: {pct(stats.read_ms, 50):.3f}/{pct(stats.read_ms, 95):.3f}/{pct(stats.read_ms, 99):.3f} ms"
    )
    print(f"checksum(sample): {stats.checksum}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw 1MiB IO benchmark for local SSD and NFS over RDMA."
    )
    parser.add_argument("--local-dir", required=True, help="Directory on local SSD")
    parser.add_argument("--nfs-dir", required=True, help="Directory on NFS over RDMA mount")
    parser.add_argument(
        "--file-size-mb",
        type=int,
        default=2048,
        help="Per-loop file size in MiB (rounded down to 1MiB blocks)",
    )
    parser.add_argument("--loops", type=int, default=2, help="Number of loops per backend")
    parser.add_argument(
        "--fsync",
        action="store_true",
        default=False,
        help="Call fdatasync() after write loop (more durable, usually slower)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Config: "
        f"block=1MiB, file_size={args.file_size_mb}MiB, loops={args.loops}, fsync={args.fsync}"
    )

    local_stats = run_one_path(args.local_dir, args.file_size_mb, args.loops, args.fsync)
    nfs_stats = run_one_path(args.nfs_dir, args.file_size_mb, args.loops, args.fsync)

    show("Local SSD", local_stats)
    show("NFS over RDMA", nfs_stats)

    print("\n=== NFS / Local ratio ===")
    print(f"write BW ratio: {nfs_stats.write_bw_gbps / local_stats.write_bw_gbps:.3f}x")
    print(f"read  BW ratio: {nfs_stats.read_bw_gbps / local_stats.read_bw_gbps:.3f}x")


if __name__ == "__main__":
    main()
