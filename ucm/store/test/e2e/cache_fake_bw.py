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
# Usage:
# 1. Select model/mode and set worker_number/layer_number below.
#    GLM defaults to 16/78; MiniMax to 8/62; DSV4 to 8 and ignores layer_number.
# 2. Tune the workload and optional SDMA/CPU-affinity switches as needed.
# 3. Run this script with python3.
import multiprocessing
import os
import secrets
import signal
import subprocess
import sys
import time

import torch

store_pipeline = "Cache|Fake"
device_type = "npu"

# =========================== User configuration ===========================
# Model profile: glm-5.2, minimax-m2.7, or dsv4.
model_name = "glm-5.2"
# True uses layerwise transfer; False uses non-layerwise. DSV4 ignores it.
use_layerwise = True
# Worker process/NPU count: GLM defaults to 16; MiniMax/DSV4 default to 8.
worker_number = 16
# Only used by GLM/MiniMax non-layerwise transfers: GLM=78, MiniMax=62.
layer_number = 78

# ======================== Benchmark configuration =========================
# Number of cache blocks transferred in each epoch.
block_number = 100
# Number of measured dump epochs.
dump_epoch_number = 16
# Number of measured load epochs.
load_epoch_number = 128
# Warmup epochs excluded from the final statistics.
warmup_epoch_number = 5
# Pause between adjacent epochs, in milliseconds.
epoch_interval_ms = 15
# Enable Cache SDMA Direct transfers.
cache_sdma_direct = False
# Bind each worker and its UCM store threads to NUMA-local CPU cores.
worker_cpu_affinity_enable = False

# MLA writes once from worker 0 and all workers load the same block ids, while
# GQA workers use their own block ids. GLM and MiniMax support layerwise and
# non-layerwise transfers; DSV4 always uses non-layerwise transfers.
MODEL_PROFILES = {
    "glm-5.2": {
        "worker_mode": "mla",
        "share_buffer_enable": True,
        "layer_tensor_size_list": [131072, 16384, 32768],
    },
    # MiniMax tensor sizes below are for TP=8; adjust them for other TP sizes.
    "minimax-m2.7": {
        "worker_mode": "gqa",
        "share_buffer_enable": False,
        "layer_tensor_size_list": [32768, 32768],
    },
    "dsv4": {
        "worker_mode": "mla",
        "share_buffer_enable": True,
        "full_tensor_size_list": [
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            131072,
            16384,
            256,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
            4096,
        ],
    },
}


def resolve_tensor_size_list(profile):
    layer_tensor_sizes = profile.get("layer_tensor_size_list")
    if use_layerwise and layer_tensor_sizes is not None:
        return layer_tensor_sizes

    full_tensor_sizes = profile.get("full_tensor_size_list")
    if full_tensor_sizes is not None:
        return full_tensor_sizes
    return profile["layer_tensor_size_list"] * layer_number


model_profile = MODEL_PROFILES.get(model_name)
if model_profile is None:
    available_models = ", ".join(MODEL_PROFILES)
    raise ValueError(
        f"unsupported model {model_name!r}; choose one of: {available_models}"
    )
worker_mode = model_profile["worker_mode"]
share_buffer_enable = model_profile["share_buffer_enable"]
tensor_size_list = resolve_tensor_size_list(model_profile)
shard_size = (sum(tensor_size_list) + 4095) // 4096 * 4096
effective_use_layerwise = (
    use_layerwise and model_profile.get("layer_tensor_size_list") is not None
)
transfer_mode = "layerwise" if effective_use_layerwise else "non-layerwise"
bytes_per_block = sum(tensor_size_list)
bytes_per_epoch = bytes_per_block * block_number
dump_worker_number = worker_number if worker_mode == "gqa" else 1
total_dump_bytes = (
    bytes_per_epoch
    * (warmup_epoch_number + dump_epoch_number)
    * dump_worker_number
)


def parse_cpu_list(cpu_list: str):
    cpu_cores = []
    for part in cpu_list.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            cpu_cores.extend(range(min(start, end), max(start, end) + 1))
        else:
            cpu_cores.append(int(part))
    return list(dict.fromkeys(cpu_cores))


def split_cpu_cores(cpu_cores, group_number):
    base, extra = divmod(len(cpu_cores), group_number)
    groups = []
    start = 0
    for group_id in range(group_number):
        group_size = base + (group_id < extra)
        groups.append(cpu_cores[start : start + group_size])
        start += group_size
    return groups


def get_visible_npu_ids():
    visible_devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES") or os.getenv(
        "ASCEND_VISIBLE_DEVICES"
    )
    npu_ids = (
        parse_cpu_list(visible_devices)
        if visible_devices
        else list(range(worker_number))
    )
    if len(npu_ids) < worker_number:
        raise RuntimeError(
            f"worker_number={worker_number} exceeds visible NPU number "
            f"{len(npu_ids)}"
        )
    return npu_ids[:worker_number]


def get_topology_cpu_pools(npu_ids, available_cpu_cores):
    command_env = os.environ.copy()
    command_env.update({"LC_ALL": "C", "LANG": "C", "LC_MESSAGES": "C"})
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "topo"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=command_env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"warning: failed to query NPU CPU affinity: {error}")
        return None
    if result.returncode != 0:
        print(
            "warning: npu-smi topology query failed, falling back to lscpu "
            f"NUMA mapping: {result.stderr.strip()}"
        )
        return None

    npu_affinities = {}
    logic_npu_id = 0
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("NPU"):
            continue
        affinity = line.split()[-1]
        if affinity != "Affinity":
            npu_affinities[logic_npu_id] = parse_cpu_list(affinity)
        logic_npu_id += 1

    allowed_cpu_cores = set(available_cpu_cores)
    cpu_pools = [
        sorted(allowed_cpu_cores.intersection(npu_affinities.get(npu_id, [])))
        for npu_id in npu_ids
    ]
    if any(not cpu_pool for cpu_pool in cpu_pools):
        print(
            "warning: incomplete NPU CPU affinity, falling back to lscpu "
            "NUMA mapping"
        )
        return None
    return cpu_pools


def get_fallback_numa_cpu_pools(available_cpu_cores):
    command_env = os.environ.copy()
    command_env.update({"LC_ALL": "C", "LANG": "C", "LC_MESSAGES": "C"})
    try:
        result = subprocess.run(
            ["lscpu", "-e=CPU,NODE"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=command_env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"failed to query CPU NUMA topology: {error}") from error
    if result.returncode != 0:
        raise RuntimeError(f"lscpu NUMA query failed: {result.stderr.strip()}")

    allowed_cpu_cores = set(available_cpu_cores)
    numa_cpu_cores = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2 or not all(part.lstrip("-").isdigit() for part in parts):
            continue
        cpu_id, numa_id = map(int, parts)
        if numa_id >= 0 and cpu_id in allowed_cpu_cores:
            numa_cpu_cores.setdefault(numa_id, []).append(cpu_id)
    if not numa_cpu_cores:
        raise RuntimeError("no available NUMA-local CPU cores found")

    numa_ids = sorted(numa_cpu_cores)
    return [
        sorted(numa_cpu_cores[numa_ids[worker_id * len(numa_ids) // worker_number]])
        for worker_id in range(worker_number)
    ]


def make_cpu_affinity_core_groups():
    if not worker_cpu_affinity_enable:
        empty_groups = [[] for _ in range(worker_number)]
        return empty_groups, empty_groups
    available_cpu_cores = sorted(os.sched_getaffinity(0))
    if len(available_cpu_cores) < worker_number * 2:
        raise RuntimeError(
            f"NUMA-aware affinity requires at least two CPU cores per worker: "
            f"worker_number={worker_number}, available={len(available_cpu_cores)}"
        )
    npu_ids = get_visible_npu_ids()
    npu_cpu_pools = get_topology_cpu_pools(npu_ids, available_cpu_cores)
    affinity_source = "npu-smi topo"
    if npu_cpu_pools is None:
        npu_cpu_pools = get_fallback_numa_cpu_pools(available_cpu_cores)
        affinity_source = "lscpu NUMA fallback"

    grouped_worker_ids = {}
    for worker_id, cpu_pool in enumerate(npu_cpu_pools):
        grouped_worker_ids.setdefault(tuple(cpu_pool), []).append(worker_id)

    worker_cpu_core_groups = [[] for _ in range(worker_number)]
    store_cpu_core_groups = [[] for _ in range(worker_number)]
    assigned_cpu_cores = set()
    for cpu_pool, worker_ids in grouped_worker_ids.items():
        per_worker_pools = split_cpu_cores(list(cpu_pool), len(worker_ids))
        for worker_id, per_worker_pool in zip(worker_ids, per_worker_pools):
            if len(per_worker_pool) < 2:
                raise RuntimeError(
                    f"NPU {npu_ids[worker_id]} has fewer than two available "
                    f"CPU cores: {per_worker_pool}"
                )
            overlap = assigned_cpu_cores.intersection(per_worker_pool)
            if overlap:
                raise RuntimeError(
                    f"overlapping CPU affinity cores detected for worker "
                    f"{worker_id}: {sorted(overlap)}"
                )
            assigned_cpu_cores.update(per_worker_pool)
            middle = max(1, len(per_worker_pool) // 2)
            worker_cpu_core_groups[worker_id] = per_worker_pool[:middle]
            store_cpu_core_groups[worker_id] = per_worker_pool[middle:]
            print(
                f"CPU affinity plan: source={affinity_source}, "
                f"worker={worker_id}, npu={npu_ids[worker_id]}, "
                f"worker_cores={worker_cpu_core_groups[worker_id]}, "
                f"store_cores={store_cpu_core_groups[worker_id]}"
            )
    return worker_cpu_core_groups, store_cpu_core_groups


def setup_device(device_id: int):
    if device_type == "cuda":
        torch.cuda.set_device(device_id)
    else:
        import torch_npu  # noqa: F401

        torch.npu.set_device(device_id)
    return f"{device_type}:{device_id}"


def synchronize_device():
    if device_type == "cuda":
        torch.cuda.synchronize()
    else:
        torch.npu.synchronize()


def configure_ucm_logging():
    os.environ["UCM_LOG_LEVEL"] = "info"
    os.environ["UC_LOGGER_LEVEL"] = "info"


def create_cache_worker(
    pipeline_store_cls, unique_id: str, device_id: int, store_cpu_affinity_cores
):
    config = {}
    config["store_pipeline"] = store_pipeline
    config["unique_id"] = unique_id
    config["tensor_size_list"] = tensor_size_list
    config["shard_size"] = shard_size
    config["block_size"] = shard_size
    config["share_buffer_enable"] = share_buffer_enable
    config["io_direct"] = True
    config["cache_load_backend_only"] = True
    config["cache_buffer_capacity_gb"] = 32
    config["cache_stream_number"] = 4
    config["cache_sdma_direct"] = cache_sdma_direct
    config["cache_sdma_direct_launch_granularity"] = "shard"
    config["timeout_ms"] = 30000
    config["device_id"] = device_id
    if store_cpu_affinity_cores:
        config["cpu_affinity_cores"] = store_cpu_affinity_cores
    return pipeline_store_cls(config)


def create_cache_scheduler(
    pipeline_store_cls, unique_id: str, store_cpu_affinity_cores
):
    config = {}
    config["store_pipeline"] = store_pipeline
    config["cache_load_backend_only"] = True
    config["unique_id"] = unique_id
    # Keep scheduler tensor sizes and shard size unset so the C++ defaults are
    # used; an empty Python list is parsed as vector<any> and fails any_cast.
    config["block_size"] = shard_size
    config["share_buffer_enable"] = share_buffer_enable
    config["io_direct"] = True
    config["cache_buffer_capacity_gb"] = 32
    config["cache_sdma_direct"] = cache_sdma_direct
    config["cache_sdma_direct_launch_granularity"] = "shard"
    config["timeout_ms"] = 30000
    config["device_id"] = -1
    if store_cpu_affinity_cores:
        config["cpu_affinity_cores"] = store_cpu_affinity_cores
    return pipeline_store_cls(config)


def make_tensors(device: str):
    return make_sized_tensors(device, torch.rand)


def make_empty_tensors(device: str):
    return make_sized_tensors(device, torch.empty)


def make_sized_tensors(device: str, factory):
    dtype = torch.bfloat16
    element_size = torch.empty((), dtype=dtype).element_size()
    tensors = []
    for _ in range(block_number):
        row = []
        for tensor_size in tensor_size_list:
            if tensor_size % element_size != 0:
                raise ValueError(
                    f"tensor size {tensor_size} is not divisible by {element_size}"
                )
            row.append(
                factory([tensor_size // element_size], dtype=dtype, device=device)
            )
        tensors.append(row)
    return tensors


def dump(
    epoch: int, device: str, device_id: int, worker, block_ids, warmup: bool
) -> float:
    src_tensors = make_tensors(device)
    total_size = sum(tensor_size_list) * block_number
    shard_indexes = [0 for _ in range(block_number)]
    synchronize_device()
    tp = time.perf_counter()
    task = worker.dump(block_ids, shard_indexes, src_tensors)
    worker.wait(task)
    cost = time.perf_counter() - tp
    print_result("dump", epoch, device_id, cost, total_size, warmup)
    return cost


def load(
    epoch: int, device: str, device_id: int, worker, block_ids, warmup: bool
) -> float:
    dst_tensors = make_empty_tensors(device)
    total_size = sum(tensor_size_list) * block_number
    shard_indexes = [0 for _ in range(block_number)]
    synchronize_device()
    tp = time.perf_counter()
    task = worker.load(block_ids, shard_indexes, dst_tensors)
    worker.wait(task)
    synchronize_device()
    cost = time.perf_counter() - tp
    print_result("load", epoch, device_id, cost, total_size, warmup)
    return cost


def wait_backend_ready(scheduler, block_ids, timeout_s=60, poll_interval_s=0.001):
    deadline = time.perf_counter() + timeout_s
    while True:
        founds = scheduler.lookup(block_ids)
        if bool(founds.all()):
            return
        if time.perf_counter() >= deadline:
            ready_count = int(founds.sum())
            raise TimeoutError(
                f"backend ready timeout: ready={ready_count}/{len(block_ids)}"
            )
        time.sleep(poll_interval_s)


def print_result(
    direction: str,
    epoch: int,
    device_id: int,
    cost: float,
    total_size: int,
    warmup: bool,
):
    phase = "warmup" if warmup else "benchmark"
    print(
        f"phase={phase}, epoch={epoch:03}, worker={device_id:02}, "
        f"{direction}=[{sum(tensor_size_list)} x {block_number}], "
        f"cost={cost * 1e3:.3f}ms, "
        f"bw={total_size / cost / 1e9:.3f}GB/s."
    )


def percentile(sorted_values, percent):
    position = (len(sorted_values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def format_statistics(values):
    sorted_values = sorted(values)
    statistics = (
        ("avg", sum(sorted_values) / len(sorted_values)),
        ("min", sorted_values[0]),
        ("p50", percentile(sorted_values, 50)),
        ("p90", percentile(sorted_values, 90)),
        ("p99", percentile(sorted_values, 99)),
        ("max", sorted_values[-1]),
    )
    return ", ".join(f"{name}={value:.3f}" for name, value in statistics)


def print_benchmark_summary(dump_cost_records, load_cost_records):
    total_size = sum(tensor_size_list) * block_number
    print("\n================ Benchmark summary ================")
    for direction, records in (
        ("dump", dump_cost_records),
        ("load", load_cost_records),
    ):
        costs = [cost for cost in records if cost > 0]
        latencies_ms = [cost * 1e3 for cost in costs]
        bandwidths_gbps = [total_size / cost / 1e9 for cost in costs]
        print(f"{direction}: samples={len(costs)}")
        print(f"  latency(ms): {format_statistics(latencies_ms)}")
        print(f"  bandwidth(GB/s): {format_statistics(bandwidths_gbps)}")


def worker_loop(
    device_id: int,
    barrier: multiprocessing.Barrier,
    unique_id: str,
    worker_cpu_affinity_cores,
    store_cpu_affinity_cores,
    block_id_records,
    backend_block_ids,
    dump_cost_records,
    load_cost_records,
    completed_worker_number,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    if worker_cpu_affinity_cores:
        os.sched_setaffinity(0, worker_cpu_affinity_cores)
    configure_ucm_logging()

    # Import UCM inside the spawned worker so its async logger owns a live
    # logging thread in this process.
    from ucm.logger import init_logger  # pylint: disable=import-outside-toplevel
    from ucm.store.pipeline.connector import (  # pylint: disable=import-outside-toplevel
        UcmPipelineStore,
    )

    logger = init_logger(__name__)
    logger.info(
        "Cache Fake benchmark worker %d initialized UC logging at info level.",
        device_id,
    )
    device = setup_device(device_id)
    worker = create_cache_worker(
        UcmPipelineStore, unique_id, device_id, store_cpu_affinity_cores
    )
    scheduler = (
        create_cache_scheduler(
            UcmPipelineStore, unique_id, store_cpu_affinity_cores
        )
        if device_id == 0
        else None
    )
    print(
        f"{store_pipeline} benchmark: device={device}, "
        f"model={model_name}, transfer_mode={transfer_mode}, "
        f"worker_mode={worker_mode}, layer_number={layer_number}, "
        f"worker_number={worker_number}, "
        f"block_number={block_number}, tensor_number={len(tensor_size_list)}, "
        f"tensor_size_list={tensor_size_list}, "
        f"bytes_per_block={bytes_per_block}, bytes_per_epoch={bytes_per_epoch}, "
        f"total_dump_bytes={total_dump_bytes}, "
        f"shard_size={shard_size}, dtype={torch.bfloat16}, "
        f"warmup_epoch_number={warmup_epoch_number}, "
        f"epoch_interval_ms={epoch_interval_ms}, "
        f"cache_sdma_direct={cache_sdma_direct}, "
        f"share_buffer_enable={share_buffer_enable}, "
        f"worker_cpu_affinity_enable={worker_cpu_affinity_enable}, "
        f"worker_cpu_affinity_cores={worker_cpu_affinity_cores}, "
        f"store_cpu_affinity_cores={store_cpu_affinity_cores}, "
        f"multiprocessing_start_method={multiprocessing.get_start_method()}"
    )

    barrier.wait()
    for record_idx, block_ids in enumerate(block_id_records):
        warmup = record_idx < warmup_epoch_number
        epoch = record_idx if warmup else record_idx - warmup_epoch_number
        if worker_mode == "gqa" or device_id == 0:
            cost = dump(epoch, device, device_id, worker, block_ids, warmup)
            if not warmup:
                dump_cost_records[device_id * dump_epoch_number + epoch] = cost
        barrier.wait()
        if record_idx + 1 < len(block_id_records):
            time.sleep(epoch_interval_ms / 1000)

    if device_id == 0:
        wait_backend_ready(scheduler, backend_block_ids)
    barrier.wait()

    total_load_epoch_number = warmup_epoch_number + load_epoch_number
    for load_idx in range(total_load_epoch_number):
        warmup = load_idx < warmup_epoch_number
        epoch = load_idx if warmup else load_idx - warmup_epoch_number
        record_idx = load_idx % len(block_id_records)
        cost = load(
            epoch,
            device,
            device_id,
            worker,
            block_id_records[record_idx],
            warmup,
        )
        if not warmup:
            load_cost_records[device_id * load_epoch_number + epoch] = cost
        barrier.wait()
        if load_idx + 1 < total_load_epoch_number:
            time.sleep(epoch_interval_ms / 1000)
    sys.stdout.flush()
    sys.stderr.flush()
    with completed_worker_number.get_lock():
        completed_worker_number.value += 1


def make_block_id_records():
    return [
        [secrets.token_bytes(16) for _ in range(block_number)]
        for _ in range(warmup_epoch_number + dump_epoch_number)
    ]


def cleanup_workers(workers, unique_id: str):
    for process in workers:
        if process.is_alive():
            process.terminate()
    for process in workers:
        process.join(timeout=10)
    for process in workers:
        if process.is_alive():
            process.kill()
            process.join()

    for prefix in ("uc_shm_cache_", "uc_shm_fake_"):
        path = f"/dev/shm/{prefix}{unique_id}"
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


stop_requested = False


def stop_on_suspend(_signum, _frame):
    global stop_requested
    stop_requested = True


if __name__ == "__main__":
    if not tensor_size_list:
        raise ValueError(
            f"{model_name} tensor_size_list is empty; fill it in MODEL_PROFILES "
            "before running the benchmark"
        )
    configure_ucm_logging()
    process_context = multiprocessing.get_context("spawn")
    barrier = process_context.Barrier(worker_number)
    unique_id = secrets.token_hex(8)
    shared_block_id_records = make_block_id_records()
    worker_block_id_records = (
        [shared_block_id_records] * worker_number
        if worker_mode == "mla"
        else [make_block_id_records() for _ in range(worker_number)]
    )
    backend_block_id_records = (
        shared_block_id_records
        if worker_mode == "mla"
        else [
            block_ids
            for block_id_records in worker_block_id_records
            for block_ids in block_id_records
        ]
    )
    backend_block_ids = [
        block_id
        for block_ids in backend_block_id_records
        for block_id in block_ids
    ]
    worker_cpu_core_groups, store_cpu_core_groups = make_cpu_affinity_core_groups()
    dump_cost_records = process_context.Array(
        "d", worker_number * dump_epoch_number, lock=False
    )
    load_cost_records = process_context.Array(
        "d", worker_number * load_epoch_number, lock=False
    )
    completed_worker_number = process_context.Value("i", 0)
    workers = []
    signal.signal(signal.SIGTSTP, stop_on_suspend)
    try:
        for device_id in range(worker_number):
            process = process_context.Process(
                target=worker_loop,
                args=(
                    device_id,
                    barrier,
                    unique_id,
                    worker_cpu_core_groups[device_id],
                    store_cpu_core_groups[device_id],
                    worker_block_id_records[device_id],
                    backend_block_ids if device_id == 0 else None,
                    dump_cost_records,
                    load_cost_records,
                    completed_worker_number,
                ),
            )
            workers.append(process)
            process.start()
            if stop_requested:
                raise KeyboardInterrupt

        while any(process.is_alive() for process in workers):
            if completed_worker_number.value == worker_number:
                break
            if stop_requested:
                raise KeyboardInterrupt
            failed = next(
                (process for process in workers if process.exitcode not in (None, 0)),
                None,
            )
            if failed is not None:
                raise RuntimeError(
                    f"worker pid={failed.pid} exited with code {failed.exitcode}"
                )
            time.sleep(0.1)
        if completed_worker_number.value != worker_number:
            failed = next(
                (
                    process
                    for process in workers
                    if process.exitcode not in (None, 0)
                ),
                None,
            )
            if failed is not None:
                raise RuntimeError(
                    f"worker pid={failed.pid} exited with code {failed.exitcode}"
                )
            raise RuntimeError("workers exited before completing the benchmark")
        print_benchmark_summary(dump_cost_records, load_cost_records)
    except KeyboardInterrupt:
        print("benchmark interrupted; cleaning up workers and shared memory")
    finally:
        cleanup_workers(workers, unique_id)
