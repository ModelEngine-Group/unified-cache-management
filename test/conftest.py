from __future__ import annotations

import dataclasses
import datetime
import datetime as dt
import logging
import os
import platform as pf
import random
import sys
import uuid
from pathlib import Path

import pynvml
import pytest
from common.capture_utils import export_vars, set_test_info
from common.config_utils import config_utils as config_instance
from common.uc_eval.utils.data_class import ModelConfig

# ---------------- Constants ----------------
PRJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PRJ_ROOT))
logger = logging.getLogger(__name__)


# ---------------- CLI Options ----------------
def pytest_addoption(parser):
    parser.addoption(
        "--stage", action="store", default="", help="Filter by stage marker (1,2,3,+)"
    )
    parser.addoption(
        "--feature", action="store", default="", help="Filter by feature marker"
    )
    parser.addoption(
        "--platform", action="store", default="", help="Filter by platform marker"
    )


# ---------------- Test Filtering ----------------
def pytest_collection_modifyitems(config, items):
    kept = items[:]

    markers = [m.split(":", 1)[0].strip() for m in config.getini("markers")]
    for name in markers:
        opt = config.getoption(f"--{name}", "").strip()
        if not opt:
            continue

        if name == "stage" and opt.endswith("+"):
            min_stage = int(opt[:-1])
            kept = [
                it
                for it in kept
                if any(int(v) >= min_stage for v in _get_marker_args(it, "stage"))
            ]
        else:
            wanted = {x.strip() for x in opt.split(",") if x.strip()}
            kept = [
                it
                for it in kept
                if any(v in wanted for v in _get_marker_args(it, name))
            ]

    config.hook.pytest_deselected(items=[i for i in items if i not in kept])
    items[:] = kept


def _get_marker_args(item, marker_name):
    """Extract only args (not kwargs) from markers, as strings."""
    return [
        str(arg) for mark in item.iter_markers(name=marker_name) for arg in mark.args
    ]


# ---------------- Report Setup ----------------
def _prepare_report_dir(config: pytest.Config) -> Path:
    cfg = config_instance.get_config("reports", {})
    base_dir = Path(cfg.get("base_dir", "reports"))
    prefix = cfg.get("directory_prefix", "pytest")
    if cfg.get("use_timestamp", False):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = base_dir / f"{prefix}_{ts}"
    else:
        report_dir = base_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def _setup_html_report(config: pytest.Config, report_dir: Path) -> None:
    reports_config = config_instance.get_config("reports", {})
    html_cfg = reports_config.get("html", {})
    if not html_cfg.get("enabled", True):
        if hasattr(config.option, "htmlpath"):
            config.option.htmlpath = None
        print("HTML report disabled according to config.yaml")
        return

    html_filename = html_cfg.get("filename", "report.html")
    config.option.htmlpath = str(report_dir / html_filename)
    config.option.self_contained_html = True
    print("HTML report enabled")


# ---------------- Build ID & Session Init ----------------
def _generate_build_info(config: pytest.Config) -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cli_parts = []
    markers = [m.split(":", 1)[0].strip() for m in config.getini("markers")]
    for opt in markers:
        val = config.getoption(opt, "")
        if val:
            cli_parts.append(f"{opt}={val}")
    args_part = " ".join(f"_--{p}" for p in cli_parts) if cli_parts else "all_cases"
    return f"pytest{args_part}"


# ---------------- Pytest Hooks ----------------
def pytest_configure(config: pytest.Config) -> None:
    """The global configuration will be executed directly upon entering pytest."""
    print(f"Starting Test Session: {dt.datetime.now():%Y-%m-%d %H:%M:%S}")

    # Set up report directory
    report_dir = _prepare_report_dir(config)
    config._report_dir = report_dir  # Attach to config for later use
    _setup_html_report(config, report_dir)

    # Generate and register build ID into DB
    test_items = _generate_build_info(config)
    test_id = str(uuid.uuid4())
    set_test_info(test_id, test_items)


def pytest_sessionstart(session):
    print("")
    print("-" * 60)
    print(f"{'Python':<10} │ {pf.python_version()}")
    print(f"{'Platform':<10} │ {pf.system()} {pf.release()}")
    print("-" * 60)


def pytest_sessionfinish(session, exitstatus):
    report_dir = getattr(session.config, "_report_dir", "reports")
    print("")
    print("-" * 60)
    print(f"{'Reports at':<10} │ {report_dir}")
    print("Test session ended")
    print("-" * 60)


# ---------------- Fixtures ----------------


@export_vars
def pytest_runtest_logreport(report):
    """
    Called after each test phase. We only care about 'call' (the actual test).
    """
    if report.when != "call":
        return

    status = report.outcome.upper()  # 'passed', 'failed', 'skipped' → 'PASSED', etc.
    test_result = {
        "test_case": report.nodeid,
        "status": status,
        # "duration": report.duration,
        "error": str(report.longrepr) if report.failed else None,
    }
    return {"_name": "test_case_info", "_data": test_result}


# GPU lock files are stored on the host-mounted shared directory so all runners
# on the same machine coordinate through the same set of lock files.
GPU_LOCK_DIR = os.environ.get("GPU_LOCK_DIR", "/workspace/test_results/gpu_locks")


def get_free_gpu(required_memory_mb):
    """Find a free GPU and acquire an exclusive file lock on it.

    Returns (gpu_id, free_mb, utilization, lock_file) on success, or
    (None, 0, 0, None) when no suitable GPU is available.
    The caller is responsible for releasing the lock by calling
    fcntl.flock(lock_file, fcntl.LOCK_UN) and closing lock_file.
    """
    import fcntl

    mem_needed_with_buffer = int(required_memory_mb * 1.3)  # add buffer to avoid OOM

    def try_acquire_lock(gpu_index, free_mb, total_bytes):
        """Try to acquire flock for gpu_index. Returns lock_file on success, None on failure."""
        lock_path = os.path.join(GPU_LOCK_DIR, f"gpu_{gpu_index}.lock")
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return None
        utilization = required_memory_mb * (1024**2) / total_bytes if total_bytes else 0
        return lock_file, free_mb, utilization

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        device_indices = list(range(device_count))
        random.shuffle(device_indices)

        # Collect memory info for all GPUs
        gpu_infos = []
        for i in device_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = info.free / 1024**2
            used_ratio = info.used / info.total if info.total else 1.0
            gpu_infos.append((i, free_mb, info.total, used_ratio))

        # Pass 1: prefer idle GPUs (used memory < 5%)
        for i, free_mb, total, used_ratio in gpu_infos:
            if used_ratio >= 0.05:
                continue
            result = try_acquire_lock(i, free_mb, total)
            if result:
                lock_file, free_mb, utilization = result
                return i, free_mb, utilization, lock_file

        # Pass 2: fall back to any GPU with enough free memory
        for i, free_mb, total, _ in gpu_infos:
            if free_mb < mem_needed_with_buffer:
                continue
            result = try_acquire_lock(i, free_mb, total)
            if result:
                lock_file, free_mb, utilization = result
                return i, free_mb, utilization, lock_file
    finally:
        pynvml.nvmlShutdown()
    return None, 0, 0, None


@pytest.fixture(autouse=True)
def setup_gpu_resource(request):
    import fcntl

    gpu_mem_marker = request.node.get_closest_marker("gpu_mem")
    gpu_count_marker = request.node.get_closest_marker("gpu_count")

    if not gpu_mem_marker and not gpu_count_marker:
        # No GPU markers, skip GPU resource setup
        yield
        return

    if gpu_count_marker:
        # Handle gpu_count marker - allocate multiple GPUs
        gpu_count = gpu_count_marker.args[0]
        mem_needed = gpu_mem_marker.args[0] if gpu_mem_marker else 0
        allocated_gpus = []
        lock_files = []

        try:
            for i in range(gpu_count):
                gpu_id, free_in_mb, gpu_utilization, lock_file = get_free_gpu(
                    mem_needed
                )
                if gpu_id is None:
                    pytest.fail(
                        f"Failed to {f'allocate GPU with {mem_needed}MB(+30% buffer) free memory' if mem_needed else 'allocate GPU'} {i + 1}/{gpu_count}"
                    )

                allocated_gpus.append(gpu_id)
                lock_files.append(lock_file)
                print(
                    f"Allocating GPU {gpu_id} (slot {i + 1}/{gpu_count}) with {free_in_mb}MB free memory"
                )

            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, allocated_gpus))
            print(f"Allocated GPUs: {allocated_gpus}")

            yield  # test runs here while locks are held

        finally:
            for lock_file in lock_files:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
    else:
        # Handle gpu_mem marker - allocate single GPU with memory requirement
        mem_needed = gpu_mem_marker.args[0]
        gpu_id, free_in_mb, gpu_utilization, lock_file = get_free_gpu(mem_needed)
        if gpu_id is None:
            pytest.fail(
                f"No GPU with {mem_needed}MB(+30% buffer) free memory available"
            )

        print(
            f"Allocating GPU {gpu_id} with {free_in_mb}MB free memory, gpu utilization for test {gpu_utilization:.4%}"
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if gpu_utilization:
            os.environ["E2E_TEST_GPU_MEMORY_UTILIZATION"] = str(gpu_utilization)

        yield  # test runs here while the lock is held

        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    cfg = config_instance.get_config("models") or {}
    field_names = [field.name for field in dataclasses.fields(ModelConfig)]
    kwargs = {k: v for k, v in cfg.items() if k in field_names and v is not None}
    return ModelConfig(**kwargs)


# ---------------- Session Finish Hook ----------------


def pytest_sessionfinish(session, exitstatus):
    backup_dir = config_instance.get_nested_config("database.backup") or "results/"
    backup_dir = Path(backup_dir).resolve()

    if not backup_dir.exists():
        logger.warning(f"Backup directory not found: {backup_dir}, skipping conversion")
        return

    jsonl_files = list(backup_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {backup_dir}, skipping conversion")
        return

    success_count = 0
    for jsonl_file in jsonl_files:
        try:
            from common.capture_results.localFile import jsonl_to_csv

            csv_file = jsonl_to_csv(jsonl_file, flatten=True)
            logger.debug(f"Converted: {jsonl_file.name} → {csv_file.name}")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to convert {jsonl_file.name}: {e}", exc_info=True)
    logger.info(f"Converted {success_count} JSONL files to CSV")
