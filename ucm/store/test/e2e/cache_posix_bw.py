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
import multiprocessing
import os
import secrets
import signal
import sys
import time

import torch

store_pipeline = "Cache|Posix"
device_type = "npu"

# ======================== Benchmark configuration =========================
block_number = 100
dump_epoch_number = 16
load_epoch_number = 16
warmup_epoch_number = 5
epoch_interval_ms = 15
cache_sdma_direct = True
storage_backends = ["./build/data"]
worker_cpu_affinity_enable = True
ucm_log_level = "info"

# =========================== User configuration ===========================
model_name = "glm-5.2"

# Fill tensor_size_list with the per-layer tensor byte sizes of the target
# deployment before running a profile. MLA writes once from worker 0 and all
# workers load the same block ids, while GQA workers use their own block ids.
MODEL_PROFILES = {
    "glm-5.2": {
        "worker_mode": "mla",
        "worker_number": 8,
        "share_buffer_enable": True,
        "tensor_size_list": [131072, 32768, 16384],
    },
    "minimax-m2.7": {
        "worker_mode": "gqa",
        "worker_number": 8,
        "share_buffer_enable": False,
        "tensor_size_list": [32768, 32768],
    },
    "dsv4": {
        "worker_mode": "mla",
        "worker_number": 8,
        "share_buffer_enable": True,
        "tensor_size_list": [
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

model_profile = MODEL_PROFILES.get(model_name)
if model_profile is None:
    available_models = ", ".join(MODEL_PROFILES)
    raise ValueError(
        f"unsupported model {model_name!r}; choose one of: {available_models}"
    )
worker_mode = model_profile["worker_mode"]
worker_number = model_profile["worker_number"]
share_buffer_enable = model_profile["share_buffer_enable"]
tensor_size_list = model_profile["tensor_size_list"]
shard_size = (sum(tensor_size_list) + 4095) // 4096 * 4096


def make_worker_cpu_core_groups():
    if not worker_cpu_affinity_enable:
        return [[] for _ in range(worker_number)]
    available_cpu_cores = sorted(os.sched_getaffinity(0))
    if len(available_cpu_cores) < worker_number:
        raise RuntimeError(
            f"worker_number={worker_number} exceeds available CPU core number "
            f"{len(available_cpu_cores)}"
        )
    return [
        available_cpu_cores[worker_id::worker_number]
        for worker_id in range(worker_number)
    ]


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
    os.environ["UCM_LOG_LEVEL"] = ucm_log_level
    os.environ["UC_LOGGER_LEVEL"] = ucm_log_level


def create_cache_worker(
    pipeline_store_cls, unique_id: str, device_id: int, cpu_affinity_cores
):
    config = {}
    config["store_pipeline"] = store_pipeline
    config["storage_backends"] = storage_backends
    config["posix_io_engine"] = "aio"
    config["io_direct"] = True
    config["posix_data_trans_concurrency"] = 32
    config["posix_lookup_concurrency"] = 32
    config["cache_load_backend_only"] = True
    config["unique_id"] = unique_id
    config["tensor_size_list"] = tensor_size_list
    config["shard_size"] = shard_size
    config["block_size"] = shard_size
    config["share_buffer_enable"] = share_buffer_enable
    config["cache_buffer_capacity_gb"] = 8
    config["cache_stream_number"] = 4
    config["cache_sdma_direct"] = cache_sdma_direct
    config["cache_sdma_direct_launch_granularity"] = "shard"
    config["waiting_queue_depth"] = 16
    config["running_queue_depth"] = 1024
    config["timeout_ms"] = 10000
    config["device_id"] = device_id
    if cpu_affinity_cores:
        config["cpu_affinity_cores"] = cpu_affinity_cores
    return pipeline_store_cls(config)


def create_posix_scheduler(pipeline_store_cls, cpu_affinity_cores):
    config = {}
    config["store_pipeline"] = "Posix"
    config["storage_backends"] = storage_backends
    config["posix_io_engine"] = "aio"
    config["io_direct"] = True
    config["posix_lookup_concurrency"] = 32
    config["timeout_ms"] = 10000
    config["device_id"] = -1
    if cpu_affinity_cores:
        config["cpu_affinity_cores"] = cpu_affinity_cores
    return pipeline_store_cls(config)


def make_storage_dirs():
    for path in storage_backends:
        os.makedirs(path, exist_ok=True)


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
                f"backend commit timeout: ready={ready_count}/{len(block_ids)}"
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
    cpu_affinity_cores,
    block_id_records,
    backend_block_ids,
    dump_cost_records,
    load_cost_records,
    completed_worker_number,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    if cpu_affinity_cores:
        os.sched_setaffinity(0, cpu_affinity_cores)
    configure_ucm_logging()

    # Import UCM inside the spawned worker so its async logger owns a live
    # logging thread in this process.
    from ucm.logger import init_logger  # pylint: disable=import-outside-toplevel
    from ucm.store.pipeline.connector import (  # pylint: disable=import-outside-toplevel
        UcmPipelineStore,
    )

    logger = init_logger(__name__)
    logger.info(
        "Cache Posix benchmark worker %d initialized UC logging at level %s.",
        device_id,
        ucm_log_level,
    )
    make_storage_dirs()
    device = setup_device(device_id)
    worker = create_cache_worker(
        UcmPipelineStore, unique_id, device_id, cpu_affinity_cores
    )
    scheduler = (
        create_posix_scheduler(UcmPipelineStore, cpu_affinity_cores)
        if device_id == 0
        else None
    )
    print(
        f"{store_pipeline} one-layer benchmark: device={device}, "
        f"model={model_name}, worker_mode={worker_mode}, "
        f"worker_number={worker_number}, "
        f"block_number={block_number}, tensor_size_list={tensor_size_list}, "
        f"shard_size={shard_size}, dtype={torch.bfloat16}, "
        f"warmup_epoch_number={warmup_epoch_number}, "
        f"epoch_interval_ms={epoch_interval_ms}, "
        f"storage_backends={storage_backends}, "
        f"cache_sdma_direct={cache_sdma_direct}, "
        f"worker_cpu_affinity_enable={worker_cpu_affinity_enable}, "
        f"cpu_affinity_cores={cpu_affinity_cores}, "
        f"ucm_log_level={ucm_log_level}, "
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
    worker_cpu_core_groups = make_worker_cpu_core_groups()
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
