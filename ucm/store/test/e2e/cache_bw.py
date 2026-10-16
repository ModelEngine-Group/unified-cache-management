# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Cache2 load bandwidth: Cache hits or pre-seeded, first-use Posix hits.

See cache_bw.md for examples, workload shapes, and measurement boundaries.
Torch and UCM are imported only inside workers; --help/--dry-run need neither.
"""

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import mmap
import multiprocessing
import os
import re
import secrets
import shutil
import signal
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

# Sizes are bytes per block, before storage alignment. MiniMax is TP8;
# DeepSeek V4 describes the existing FA-only profile, not its separate WA store.
MODEL_PROFILES = {
    "glm-5.1": ([131072, 16384, 32768], 78, 16, "mla"),
    "glm-5.2": ([131072, 16384, 32768], 78, 16, "mla"),
    "minimax-m2.7-tp8": ([32768, 32768], 62, 8, "gqa"),
    "dsv4": ([131072, 16384, 256] * 21 + [4096] * 20, 1, 8, "mla"),
    "custom": ([1048576], 1, 1, "gqa"),
}


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def byte_size(text):
    match = re.fullmatch(r"(\d+)\s*([KMG]?)(?:i?B)?", text.strip(), re.IGNORECASE)
    if not match or int(match[1]) == 0:
        raise argparse.ArgumentTypeError("expected bytes or a size such as 176K/2MiB")
    power = {"": 0, "K": 1, "M": 2, "G": 3}[match[2].upper()]
    return int(match[1]) * 1024**power


def align_up(size, alignment):
    return (size + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class Workload:
    tensor_sizes: tuple[int, ...]
    shards: int
    shared: bool

    @property
    def payload_size(self):
        return sum(self.tensor_sizes)

    @property
    def shard_size(self):
        return align_up(self.payload_size, 4096)

    @property
    def block_size(self):
        return self.shard_size * self.shards


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("cache-hit", "posix-hit"), required=True)
    parser.add_argument("--model", choices=MODEL_PROFILES, default="glm-5.1")
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--workers", type=positive_int)
    parser.add_argument("--blocks", type=positive_int, default=512)
    parser.add_argument("--layers", type=positive_int)
    parser.add_argument(
        "--layout", choices=("layerwise", "whole-block", "single-layer")
    )
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument(
        "--io-size", type=byte_size, help="override with one tensor per layer"
    )
    sizes.add_argument(
        "--tensor-sizes", help="override bytes per tensor, e.g. 128K,16K,32K"
    )
    parser.add_argument("--worker-mode", choices=("mla", "gqa"))
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--epoch-interval-ms", type=float, default=15)
    parser.add_argument("--cache-capacity-gb", type=positive_int, default=32)
    parser.add_argument(
        "--cache-copy", choices=("auto", "plain", "aggregation", "sdma"), default="auto"
    )
    parser.add_argument("--cache-streams", type=positive_int)
    parser.add_argument(
        "--cache-numa-nodes", help="one NUMA node per worker; omitted: rank slicing"
    )
    parser.add_argument(
        "--storage-backends", nargs="+", default=["./build/cache-bw-data"]
    )
    parser.add_argument("--posix-engine", choices=("psync", "aio"), default="psync")
    parser.add_argument("--posix-concurrency", type=positive_int, default=32)
    parser.add_argument("--posix-lookup-concurrency", type=positive_int, default=16)
    parser.add_argument(
        "--direct-io", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--timeout-ms", type=positive_int, default=60000)
    parser.add_argument(
        "--process-timeout",
        type=positive_int,
        default=1800,
        help="seconds per process group",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check every loaded tensor outside timed transfers",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="keep this run's isolated Posix directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the plan without importing Torch/UCM",
    )
    args = parser.parse_args(argv)
    try:
        workload = resolve_workload(args)
        validate(args, workload)
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    return args, workload


def resolve_workload(args):
    tensors, layers, workers, mode = MODEL_PROFILES[args.model]
    args.workers = args.workers or workers
    args.layers = args.layers or layers
    args.worker_mode = args.worker_mode or mode
    args.layout = args.layout or ("whole-block" if args.layers == 1 else "layerwise")
    if args.model == "dsv4" and (args.layers != 1 or args.layout != "whole-block"):
        raise ValueError(
            "dsv4 is FA-only/whole-block; use --model custom for other layouts"
        )
    if args.io_size is not None:
        tensors = [args.io_size]
    elif args.tensor_sizes is not None:
        tensors = [byte_size(item) for item in args.tensor_sizes.split(",")]
    shards = args.layers if args.layout == "layerwise" else 1
    if args.layout == "whole-block":
        tensors = tensors * args.layers
    return Workload(tuple(tensors), shards, args.worker_mode == "mla")


def reserved_slots(args):
    return align_up(1024, args.workers)


def validate(args, workload):
    if args.workers > 128:
        raise ValueError("Cache2 supports at most 128 local ranks")
    if (
        args.warmup < 0
        or not math.isfinite(args.epoch_interval_ms)
        or args.epoch_interval_ms < 0
    ):
        raise ValueError("warmup and epoch interval must be non-negative")
    if args.cache_streams is not None and args.cache_streams > 32:
        raise ValueError("cache streams must be in [1, 32]")
    if args.mode == "posix-hit" and args.posix_engine == "aio" and not args.direct_io:
        raise ValueError("Posix aio requires --direct-io")
    nodes = args.cache_numa_nodes
    args.cache_numa_nodes = (
        None if nodes is None else [int(node) for node in nodes.split(",")]
    )
    if args.cache_numa_nodes is not None and (
        len(args.cache_numa_nodes) != args.workers or min(args.cache_numa_nodes) < 0
    ):
        raise ValueError(
            "cache NUMA nodes must contain one non-negative node per worker"
        )
    slots = (args.cache_capacity_gb << 30) // workload.shard_size
    reserved = reserved_slots(args)
    per_rank = slots // args.workers
    owned_blocks = (
        math.ceil(args.blocks / args.workers) if workload.shared else args.blocks
    )
    if slots < 2 * reserved:
        raise ValueError("cache capacity is too small for Cache2's reserved slots")
    if args.mode == "posix-hit" and per_rank < owned_blocks:
        raise ValueError(
            "cache capacity cannot hold one load task; increase --cache-capacity-gb"
        )
    if (
        args.mode == "cache-hit"
        and per_rank - reserved // args.workers < owned_blocks * workload.shards
    ):
        raise ValueError(
            "cache capacity cannot keep one complete request resident; increase --cache-capacity-gb"
        )


def block_ids(args, run_id, record, rank):
    key_rank = 0 if args.worker_mode == "mla" else rank
    return [
        hashlib.blake2b(
            f"{run_id}:{key_rank}:{record}:{block}".encode(), digest_size=16
        ).digest()
        for block in range(args.blocks)
    ]


def owned_indexes(args, rank):
    return (
        range(rank, args.blocks, args.workers)
        if args.worker_mode == "mla"
        else range(args.blocks)
    )


def sample_byte(block_id, shard):
    return (
        1
        + int.from_bytes(
            hashlib.blake2b(
                block_id + shard.to_bytes(8, "little"), digest_size=2
            ).digest(),
            "little",
        )
        % 254
    )


def store_config(args, workload, run_id, rank, *, seed=False):
    config = {
        "store_pipeline": (
            "Posix"
            if seed
            else ("Cache" if args.mode == "cache-hit" else "Cache|Posix")
        ),
        "unique_id": run_id + ("_seed" if seed else "_load"),
        "device_id": rank,
        "tensor_size_list": list(workload.tensor_sizes),
        "shard_size": workload.shard_size,
        "block_size": workload.block_size,
        "timeout_ms": args.timeout_ms,
    }
    if seed or args.mode == "posix-hit":
        config.update(
            storage_backends=args.storage_backends,
            tensor_size=workload.shard_size if seed else 0,
            io_direct=args.direct_io,
            posix_io_engine=args.posix_engine,
            posix_data_trans_concurrency=args.posix_concurrency,
            posix_lookup_concurrency=args.posix_lookup_concurrency,
            posix_gc_enable=False,
        )
    if seed:
        return config
    config.update(
        local_rank_size=args.workers,
        share_buffer_enable=workload.shared,
        cache_buffer_capacity_gb=args.cache_capacity_gb,
        cache_load_exclusive_buffer_number=reserved_slots(args),
        cache_fallback_numa_rank=rank,
        cache_fallback_numa_rank_count=args.workers,
    )
    if args.cache_numa_nodes is not None:
        config["cache_detected_numa_node"] = args.cache_numa_nodes[rank]
    if args.cache_copy != "auto":
        config["cache_sdma_direct"] = args.cache_copy == "sdma"
        config["cache_io_aggregation"] = args.cache_copy == "aggregation"
    if args.cache_streams is not None:
        config["cache_stream_number"] = args.cache_streams
    return config


def open_store(config):
    from ucm.logger import init_logger
    from ucm.store.pipeline.connector import UcmPipelineStore

    init_logger(__name__)
    return UcmPipelineStore(config)


def setup_device(args, rank):
    import torch

    if args.device == "npu":
        importlib.import_module("torch_npu")
    runtime = getattr(torch, args.device)
    runtime.set_device(rank)
    return torch, runtime, f"{args.device}:{rank}"


def make_rows(torch, workload, device, count):
    # Separate KV views, with consecutive blocks within each view as in a cache tensor.
    views = [
        torch.empty((count, size), dtype=torch.uint8, device=device)
        for size in workload.tensor_sizes
    ]
    return [[view[block] for view in views] for block in range(count)]


def fill_rows(rows, ids, shard):
    for row, block_id in zip(rows, ids):
        for tensor in row:
            tensor.fill_(sample_byte(block_id, shard))


def transfer_request(store, runtime, rows, ids, workload, *, dump=False, verify=False):
    costs = []
    operation = store.dump if dump else store.load
    for shard in range(workload.shards):
        if dump:
            fill_rows(rows, ids, shard)
        elif verify:
            for row in rows:
                for tensor in row:
                    tensor.zero_()
        indexes = [shard] * len(ids)
        runtime.synchronize()
        start = time.perf_counter()
        task = operation(ids, indexes, rows)
        store.wait(task)
        runtime.synchronize()
        costs.append(time.perf_counter() - start)
        if verify and not dump:
            for row, block_id in zip(rows, ids):
                for tensor in row:
                    if not bool((tensor == sample_byte(block_id, shard)).all().item()):
                        raise RuntimeError(
                            f"data mismatch: block={block_id.hex()} shard={shard}"
                        )
    return costs


def seed_worker(args, workload, run_id, rank, barrier):
    store = open_store(store_config(args, workload, run_id, rank, seed=True))
    indexes = list(owned_indexes(args, rank))
    barrier.wait()
    if indexes:
        # Posix consumes host pointers even though device_id >= 0 enables its transfer queue.
        # mmap gives page-aligned addresses; each shard is also 4 KiB aligned for O_DIRECT.
        with mmap.mmap(-1, len(indexes) * workload.shard_size) as buffer:
            address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            pointers = [
                [address + index * workload.shard_size] for index in range(len(indexes))
            ]
            for record in range(args.warmup + args.epochs):
                all_ids = block_ids(args, run_id, record, rank)
                ids = [all_ids[index] for index in indexes]
                for shard in range(workload.shards):
                    for index, block_id in enumerate(ids):
                        start = index * workload.shard_size
                        buffer[start : start + workload.shard_size] = (
                            bytes([sample_byte(block_id, shard)]) * workload.shard_size
                        )
                    task = store.dump_data(ids, [shard] * len(ids), pointers)
                    store.wait(task)
                print(
                    f"phase=seed rank={rank:02} record={record:03} blocks={len(ids)}",
                    flush=True,
                )
    barrier.wait()


def load_worker(args, workload, run_id, rank, barrier, results):
    torch, runtime, device = setup_device(args, rank)
    store = open_store(store_config(args, workload, run_id, rank))
    barrier.wait()
    if args.mode == "cache-hit":
        indexes = list(owned_indexes(args, rank))
        if indexes:
            all_ids = block_ids(args, run_id, 0, rank)
            ids = [all_ids[index] for index in indexes]
            rows = make_rows(torch, workload, device, len(ids))
            costs = transfer_request(store, runtime, rows, ids, workload, dump=True)
            print_sample(
                "prefill",
                rank,
                0,
                costs,
                workload.payload_size * len(ids) * workload.shards,
            )
            del rows
        barrier.wait()
    rows = make_rows(torch, workload, device, args.blocks)
    for record in range(args.warmup + args.epochs):
        ids = block_ids(args, run_id, record if args.mode == "posix-hit" else 0, rank)
        barrier.wait()
        costs = transfer_request(
            store, runtime, rows, ids, workload, verify=args.verify
        )
        warmup = record < args.warmup
        epoch = record if warmup else record - args.warmup
        print_sample(
            "warmup" if warmup else "load",
            rank,
            epoch,
            costs,
            workload.payload_size * args.blocks * workload.shards,
        )
        if not warmup:
            start = (rank * args.epochs + epoch) * workload.shards
            results[start : start + workload.shards] = costs
        barrier.wait()
        if args.epoch_interval_ms:
            time.sleep(args.epoch_interval_ms / 1000)


def worker_entry(phase, args, workload, run_id, rank, barrier, results):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("UCM_LOG_LEVEL", "info")
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")
    try:
        if phase == "seed":
            seed_worker(args, workload, run_id, rank, barrier)
        else:
            load_worker(args, workload, run_id, rank, barrier, results)
    except BaseException:
        barrier.abort()
        raise


def run_group(context, phase, args, workload, run_id, results):
    barrier = context.Barrier(args.workers, timeout=args.process_timeout)
    workers = []
    try:
        for rank in range(args.workers):
            worker = context.Process(
                target=worker_entry,
                args=(phase, args, workload, run_id, rank, barrier, results),
            )
            worker.start()
            workers.append(worker)
        deadline = time.monotonic() + args.process_timeout
        while any(worker.is_alive() for worker in workers):
            failed = next(
                (worker for worker in workers if worker.exitcode not in (None, 0)), None
            )
            if failed is not None:
                raise RuntimeError(
                    f"{phase} worker pid={failed.pid} exited with code {failed.exitcode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{phase} exceeded --process-timeout={args.process_timeout}s"
                )
            time.sleep(0.1)
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(
                    f"{phase} worker pid={worker.pid} exited with code {worker.exitcode}"
                )
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def print_sample(phase, rank, epoch, costs, byte_count):
    total = sum(costs)
    print(
        f"phase={phase} rank={rank:02} epoch={epoch:03} tasks={len(costs)} "
        f"task_avg={total / len(costs) * 1e3:.3f}ms "
        f"task_p90={percentile(costs, 0.9) * 1e3:.3f}ms task_p99={percentile(costs, 0.99) * 1e3:.3f}ms "
        f"request_cost={total * 1e3:.3f}ms bw={byte_count / total / 1e9:.3f}GB/s",
        flush=True,
    )


def print_summary(args, workload, results):
    print("\nLoad summary (warmup excluded; ALL pools equal-sized per-rank samples)")
    print(
        "rank   samples request_mean_ms request_p90_ms request_p99_ms task_mean_ms task_p90_ms task_p99_ms bandwidth_GB/s"
    )
    stride = args.epochs * workload.shards
    per_rank = [
        list(results[rank * stride : (rank + 1) * stride])
        for rank in range(args.workers)
    ]
    byte_count = workload.payload_size * args.blocks * workload.shards
    for label, costs in [("ALL", list(results))] + [
        (str(rank), costs) for rank, costs in enumerate(per_rank)
    ]:
        if any(cost <= 0 for cost in costs):
            raise RuntimeError(
                "missing load samples; refusing to print incomplete results"
            )
        requests = [
            sum(costs[index : index + workload.shards])
            for index in range(0, len(costs), workload.shards)
        ]
        mean = sum(requests) / len(requests)
        print(
            f"{label:<6} {len(requests):>7} {mean * 1e3:>15.3f} "
            f"{percentile(requests, 0.9) * 1e3:>14.3f} {percentile(requests, 0.99) * 1e3:>14.3f} "
            f"{sum(costs) / len(costs) * 1e3:>12.3f} {percentile(costs, 0.9) * 1e3:>11.3f} "
            f"{percentile(costs, 0.99) * 1e3:>11.3f} {byte_count / mean / 1e9:>14.3f}"
        )


def describe(args, workload):
    records = args.warmup + args.epochs if args.mode == "posix-hit" else 1
    copies = 1 if workload.shared else args.workers
    return {
        "mode": args.mode,
        "model": args.model,
        "scope": "FA-only (WA excluded)" if args.model == "dsv4" else "KV",
        "device": args.device,
        "workers": args.workers,
        "layout": args.layout,
        "blocks_per_rank_load": args.blocks,
        "tensor_count_per_task_block": len(workload.tensor_sizes),
        "payload_bytes_per_task_block": workload.payload_size,
        "shard_size": workload.shard_size,
        "block_size": workload.block_size,
        "load_tasks_per_request": workload.shards,
        "h2d_bytes_per_rank_request": workload.payload_size
        * args.blocks
        * workload.shards,
        "unique_seed_records": records,
        "seed_storage_bytes": workload.block_size * args.blocks * records * copies,
        "cache_capacity_gib": args.cache_capacity_gb,
        "cache_copy": args.cache_copy,
        "cache_streams": args.cache_streams,
        "numa_nodes": args.cache_numa_nodes or "rank-slice; no CPU affinity",
        "warmup_requests": args.warmup,
        "measured_requests": args.epochs,
        "pipeline": (
            "Cache"
            if args.mode == "cache-hit"
            else "Posix seed -> fresh Cache|Posix workers"
        ),
        "posix_engine": args.posix_engine if args.mode == "posix-hit" else None,
        "direct_io": args.direct_io if args.mode == "posix-hit" else None,
    }


def main(argv=None):
    args, workload = parse_args(argv)
    print(json.dumps(describe(args, workload), indent=2), flush=True)
    if args.dry_run:
        return
    context = multiprocessing.get_context("spawn")
    run_id = secrets.token_hex(12)
    results = context.Array(
        "d", args.workers * args.epochs * workload.shards, lock=False
    )
    with ExitStack() as cleanup:
        if args.mode == "posix-hit":
            paths = []
            for root in args.storage_backends:
                root = Path(root).resolve()
                root.mkdir(parents=True, exist_ok=True)
                path = tempfile.mkdtemp(prefix="ucm-cache-bw-", dir=root)
                paths.append(path)
                if not args.keep_data:
                    # Only these newly created, run-owned directories are removed.
                    cleanup.callback(shutil.rmtree, path)
            args.storage_backends = paths
            print(
                f"Posix run directories: {paths}; keep_data={args.keep_data}",
                flush=True,
            )
            run_group(context, "seed", args, workload, run_id, results)
        run_group(context, "load", args, workload, run_id, results)
        print_summary(args, workload, results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
