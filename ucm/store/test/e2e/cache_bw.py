# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Cache load bandwidth: Cache hits or pre-seeded, first-use Posix hits.

--help/--dry-run need neither Torch nor an installed UCM runtime.
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

LOAD_DURATION = "cache_load_duration_ms"
LOAD_BYTES = "cache_load_bytes_total"


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
    parser.add_argument(
        "--cache-core",
        choices=("v2", "legacy"),
        default="v2",
        help="v2 for Cache2; legacy for develop or A5 cache/cc builds",
    )
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
    parser.add_argument(
        "--epoch-interval-ms",
        type=float,
        default=15,
        help="pause between complete requests, outside timing (default: 15 ms)",
    )
    parser.add_argument(
        "--cache-capacity-gb",
        type=positive_int,
        default=32,
        help="total Cache capacity across workers in GiB, for either cache core",
    )
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
    if args.cache_core == "legacy":
        if args.cache_numa_nodes is not None:
            raise ValueError("legacy Cache does not support --cache-numa-nodes")
        if not workload.shared and args.cache_capacity_gb % args.workers:
            raise ValueError(
                "legacy GQA requires --cache-capacity-gb divisible by --workers "
                "to configure equal whole-GiB private buffers"
            )
        if args.cache_copy == "sdma" and args.cache_streams not in (None, 1):
            raise ValueError("legacy Cache SDMA uses one stream; use --cache-streams 1")
    slots = (args.cache_capacity_gb << 30) // workload.shard_size
    reserved = reserved_slots(args)
    per_rank = slots // args.workers
    owned_blocks = (
        math.ceil(args.blocks / args.workers) if workload.shared else args.blocks
    )
    if slots < 2 * reserved:
        raise ValueError("cache capacity is too small for Cache's reserved slots")
    if args.cache_core == "legacy" and not workload.shared and per_rank < 1024:
        raise ValueError("legacy Cache requires at least 1024 slots per private buffer")
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
    legacy = args.cache_core == "legacy"
    config = {
        "store_pipeline": (
            "Posix"
            if seed
            else (
                ("Cache|Empty" if legacy else "Cache")
                if args.mode == "cache-hit"
                else "Cache|Posix"
            )
        ),
        "unique_id": run_id + ("_seed" if seed else "_load"),
        "device_id": rank,
        "io_direct": args.direct_io,
        "tensor_size_list": list(workload.tensor_sizes),
        "shard_size": workload.shard_size,
        "block_size": workload.block_size,
        "timeout_ms": args.timeout_ms,
    }
    if seed or args.mode == "posix-hit":
        config.update(
            storage_backends=args.storage_backends,
            tensor_size=workload.shard_size if seed else 0,
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
    )
    if legacy:
        if not workload.shared:
            # Legacy GQA has an independent buffer and dispatch domain per rank.
            config["local_rank_size"] = 1
            config["cache_buffer_capacity_gb"] //= args.workers
            config["cache_load_exclusive_buffer_number"] //= args.workers
    else:
        config.update(
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


def load_duration_buckets():
    from ucm.default_metrics_config import get_default_metrics_config

    definition = next(
        item
        for item in get_default_metrics_config()["histogram"]
        if item["name"] == LOAD_DURATION
    )
    bounds = tuple(float(value) for value in definition["buckets"])
    return bounds if bounds and math.isinf(bounds[-1]) else bounds + (math.inf,)


def setup_load_metrics(bounds):
    from ucm.default_metrics_config import get_default_metrics_config
    from ucm.metrics_config import setup_ucm_metrics
    from ucm.shared.metrics import ucmmetrics

    if load_duration_buckets() != bounds:
        raise RuntimeError("parent/worker metrics bucket definitions differ")
    # Use the service's default definitions, without a dispatcher/HTTP consumer.
    setup_ucm_metrics(get_default_metrics_config())
    return ucmmetrics


@dataclass(frozen=True)
class LoadMetrics:
    duration_ms: float
    byte_count: int
    bucket_counts: tuple[int, ...]

    @property
    def count(self):
        # C++ returns disjoint bucket counts, not Prometheus cumulative buckets.
        return sum(self.bucket_counts)


def collect_load_metrics(
    metrics,
    bounds,
    expected_tasks,
    expected_bytes,
    timeout_ms,
    *,
    expect_cache_hit=False,
):
    """Drain this request's C++ deltas, including a possibly unfinished epilog."""
    counts = [0] * len(bounds)
    total_ms = 0.0
    byte_count = 0
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        counters, _, histograms = metrics.get_all_stats_and_clear()
        if expect_cache_hit and counters.get("cache_load_backend_shards_total", 0) != 0:
            raise RuntimeError(
                "cache-hit run submitted backend reads; prefilled data is not fully resident"
            )
        if LOAD_DURATION in histograms:
            buckets, duration_ms = histograms[LOAD_DURATION]
            if len(buckets) != len(bounds) or any(
                not math.isfinite(count) or count < 0 or int(count) != count
                for count in buckets
            ):
                raise RuntimeError("invalid C++ load histogram buckets")
            if not math.isfinite(duration_ms) or duration_ms < 0:
                raise RuntimeError("invalid C++ load duration sum")
            counts = [left + int(right) for left, right in zip(counts, buckets)]
            total_ms += float(duration_ms)
        byte_delta = float(counters.get(LOAD_BYTES, 0))
        if (
            not math.isfinite(byte_delta)
            or byte_delta < 0
            or not byte_delta.is_integer()
        ):
            raise RuntimeError("invalid C++ load byte counter")
        byte_count += int(byte_delta)
        count = sum(counts)
        if count > expected_tasks or byte_count > expected_bytes:
            raise RuntimeError(
                f"unexpected load metrics: tasks={count}/{expected_tasks} "
                f"bytes={byte_count}/{expected_bytes}; another workload may be active"
            )
        if count == expected_tasks and byte_count == expected_bytes:
            if not math.isfinite(total_ms) or total_ms <= 0:
                raise RuntimeError("invalid C++ load duration sum")
            return LoadMetrics(total_ms, byte_count, tuple(counts))
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"incomplete load metrics: tasks={count}/{expected_tasks} "
                f"bytes={byte_count}/{expected_bytes}; check metrics setup/shared library "
                "and ensure no other consumer drains this process"
            )
        # Latch can become ready just before its metrics epilog finishes. Do not
        # start the next request or silently fall back to Python wall time.
        time.sleep(0.001)


@dataclass
class BenchmarkResults:
    bounds: tuple[float, ...]
    api_costs: object
    duration_ms: object
    byte_counts: object
    bucket_counts: object


def create_results(context, args, workload):
    bounds = load_duration_buckets()
    requests = args.workers * args.epochs
    return BenchmarkResults(
        bounds,
        context.Array("d", requests * workload.shards, lock=False),
        context.Array("d", requests, lock=False),
        context.Array("Q", requests, lock=False),
        context.Array("Q", requests * len(bounds), lock=False),
    )


def save_result(results, index, sample, api_costs):
    start = index * len(api_costs)
    results.api_costs[start : start + len(api_costs)] = api_costs
    results.duration_ms[index] = sample.duration_ms
    results.byte_counts[index] = sample.byte_count
    start = index * len(results.bounds)
    results.bucket_counts[start : start + len(results.bounds)] = sample.bucket_counts


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


@dataclass(frozen=True)
class PreparedTransfer:
    # Keep tensors alive until every task using their raw addresses has completed.
    rows: list
    block_ids: tuple[bytes, ...]
    tasks: tuple


def prepare_transfer(rows, ids, workload):
    """Build reusable buffers matching PipelineStore.Load/Dump's buffer ABI."""
    ids = tuple(ids)
    packed_ids = b"".join(ids)
    address_row = ctypes.c_uint64 * len(workload.tensor_sizes)
    addresses = (address_row * len(rows))(
        *(address_row(*(tensor.data_ptr() for tensor in row)) for row in rows)
    )
    tasks = tuple(
        (packed_ids, (ctypes.c_uint64 * len(ids))(*([shard] * len(ids))), addresses)
        for shard in range(workload.shards)
    )
    return PreparedTransfer(rows, ids, tasks)


def transfer_request(store, runtime, request, *, dump=False, verify=False):
    costs = []
    # The Python load/load_data wrappers rebuild buffers on every call. Submit the
    # same raw buffers directly; the C++ task construction/queue/wait stay timed.
    operation = store.store_.Dump if dump else store.store_.Load
    wait = store.store_.Wait
    rows, ids = request.rows, request.block_ids
    runtime.synchronize()
    for shard, parameters in enumerate(request.tasks):
        if dump:
            fill_rows(rows, ids, shard)
        elif verify:
            for row in rows:
                for tensor in row:
                    tensor.zero_()
        if dump or verify:
            runtime.synchronize()
        start = time.perf_counter()
        task = operation(*parameters)
        wait(task)
        # Cache task completion already synchronizes its copy streams. A second
        # device-wide sync here would charge unrelated runtime work to the load.
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
    metrics = setup_load_metrics(results.bounds)
    store = open_store(store_config(args, workload, run_id, rank))
    barrier.wait()
    if args.mode == "cache-hit":
        indexes = list(owned_indexes(args, rank))
        if indexes:
            all_ids = block_ids(args, run_id, 0, rank)
            ids = [all_ids[index] for index in indexes]
            rows = make_rows(torch, workload, device, len(ids))
            prefill = prepare_transfer(rows, ids, workload)
            costs = transfer_request(store, runtime, prefill, dump=True)
            print_api_sample(
                "prefill",
                rank,
                0,
                costs,
                workload.payload_size * len(ids) * workload.shards,
            )
            del prefill, rows
        barrier.wait()
    rows = make_rows(torch, workload, device, args.blocks)
    records = args.warmup + args.epochs
    requests = [
        prepare_transfer(rows, block_ids(args, run_id, record, rank), workload)
        for record in range(records if args.mode == "posix-hit" else 1)
    ]
    if args.mode == "cache-hit":
        requests *= records
    # The C++ API returns and clears interval deltas, not cumulative snapshots.
    # Discard prefill/init data; each warmup is collected but never saved below.
    metrics.get_all_stats_and_clear()
    for record, request in enumerate(requests):
        barrier.wait()
        costs = transfer_request(store, runtime, request, verify=args.verify)
        sample = collect_load_metrics(
            metrics,
            results.bounds,
            workload.shards,
            workload.shard_size * args.blocks * workload.shards,
            args.timeout_ms,
            expect_cache_hit=args.mode == "cache-hit",
        )
        warmup = record < args.warmup
        epoch = record if warmup else record - args.warmup
        print_sample(
            "warmup" if warmup else "load",
            rank,
            epoch,
            sample,
            results.bounds,
            costs,
        )
        if not warmup:
            save_result(results, rank * args.epochs + epoch, sample, costs)
        barrier.wait()
        if args.epoch_interval_ms and record + 1 < records:
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


def histogram_quantile(bounds, counts, quantile):
    """Prometheus-style interpolation over the C++ non-cumulative counts."""
    target = quantile * sum(counts)
    previous_count = 0
    lower = 0.0
    for upper, count in zip(bounds, counts):
        if count and previous_count + count >= target:
            if math.isinf(upper):
                return lower if math.isfinite(lower) else math.nan
            return lower + (upper - lower) * (target - previous_count) / count
        previous_count += count
        lower = upper
    return math.nan


def print_api_sample(phase, rank, epoch, costs, byte_count):
    total = sum(costs)
    print(
        f"phase={phase} rank={rank:02} epoch={epoch:03} source=python_api tasks={len(costs)} "
        f"task_avg={total / len(costs) * 1e3:.3f}ms "
        f"task_p90={percentile(costs, 0.9) * 1e3:.3f}ms task_p99={percentile(costs, 0.99) * 1e3:.3f}ms "
        f"request_cost={total * 1e3:.3f}ms bw={byte_count / total / 1e9:.3f}GB/s",
        flush=True,
    )


def print_sample(phase, rank, epoch, sample, bounds, api_costs):
    print(
        f"phase={phase} rank={rank:02} epoch={epoch:03} source={LOAD_DURATION} tasks={sample.count} "
        f"task_avg={sample.duration_ms / sample.count:.3f}ms "
        f"task_p90~={histogram_quantile(bounds, sample.bucket_counts, 0.9):.3f}ms "
        f"task_p99~={histogram_quantile(bounds, sample.bucket_counts, 0.99):.3f}ms "
        f"request_cost={sample.duration_ms:.3f}ms "
        f"bw={sample.byte_count / sample.duration_ms / 1e6:.3f}GB/s "
        f"api_request_cost={sum(api_costs) * 1e3:.3f}ms",
        flush=True,
    )


def print_summary(args, workload, results):
    print(
        f"\nLoad summary (source={LOAD_DURATION}; warmup excluded; ALL merges rank histograms)"
    )
    print(
        "rank   requests tasks request_mean_ms request_p90_ms request_p99_ms task_mean_ms task_p90_ms~ task_p99_ms~ bandwidth_GB/s api_request_mean_ms"
    )
    width = len(results.bounds)
    if any(not math.isfinite(cost) or cost <= 0 for cost in results.api_costs):
        raise RuntimeError("missing/invalid API samples; refusing incomplete results")
    for index, duration in enumerate(results.duration_ms):
        counts = results.bucket_counts[index * width : (index + 1) * width]
        if (
            not math.isfinite(duration)
            or duration <= 0
            or sum(counts) != workload.shards
            or results.byte_counts[index]
            != workload.shard_size * args.blocks * workload.shards
        ):
            raise RuntimeError(
                "missing/invalid load metrics; refusing incomplete results"
            )
    groups = [("ALL", range(args.workers * args.epochs))] + [
        (str(rank), range(rank * args.epochs, (rank + 1) * args.epochs))
        for rank in range(args.workers)
    ]
    for label, indexes in groups:
        requests = [results.duration_ms[index] for index in indexes]
        total_ms = sum(requests)
        counts = [
            sum(results.bucket_counts[index * width + bucket] for index in indexes)
            for bucket in range(width)
        ]
        byte_count = sum(results.byte_counts[index] for index in indexes)
        api_ms = sum(
            sum(
                results.api_costs[
                    index * workload.shards : (index + 1) * workload.shards
                ]
            )
            * 1e3
            for index in indexes
        )
        mean = sum(requests) / len(requests)
        print(
            f"{label:<6} {len(requests):>8} {sum(counts):>5} {mean:>15.3f} "
            f"{percentile(requests, 0.9):>14.3f} {percentile(requests, 0.99):>14.3f} "
            f"{total_ms / sum(counts):>12.3f} "
            f"{histogram_quantile(results.bounds, counts, 0.9):>12.3f} "
            f"{histogram_quantile(results.bounds, counts, 0.99):>12.3f} "
            f"{byte_count / total_ms / 1e6:>14.3f} {api_ms / len(requests):>19.3f}"
        )
    print(
        "~ task percentiles are histogram estimates; bandwidth = metric bytes / metric duration."
    )


def describe(args, workload):
    records = args.warmup + args.epochs if args.mode == "posix-hit" else 1
    copies = 1 if workload.shared else args.workers
    config = store_config(args, workload, "preview", 0)
    return {
        "mode": args.mode,
        "cache_core": args.cache_core,
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
        "cache_config_capacity_gib": config["cache_buffer_capacity_gb"],
        "cache_local_rank_size": config["local_rank_size"],
        "cache_copy": args.cache_copy,
        "cache_streams": args.cache_streams,
        "numa_nodes": (
            "legacy runtime allocation; no NUMA override or CPU affinity"
            if args.cache_core == "legacy"
            else args.cache_numa_nodes or "rank-slice; no CPU affinity"
        ),
        "warmup_requests": args.warmup,
        "measured_requests": args.epochs,
        "epoch_interval_ms": args.epoch_interval_ms,
        "timing": f"C++ {LOAD_DURATION}; Python Load + Wait reported separately",
        "bandwidth": f"{LOAD_BYTES} / {LOAD_DURATION} sum (not mean of per-task rates)",
        "pipeline": (
            config["store_pipeline"]
            if args.mode == "cache-hit"
            else "Posix seed -> fresh Cache|Posix workers"
        ),
        "posix_engine": args.posix_engine if args.mode == "posix-hit" else None,
        "direct_io": args.direct_io,
    }


def main(argv=None):
    args, workload = parse_args(argv)
    print(json.dumps(describe(args, workload), indent=2), flush=True)
    if args.dry_run:
        return
    context = multiprocessing.get_context("spawn")
    run_id = secrets.token_hex(12)
    results = create_results(context, args, workload)
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
