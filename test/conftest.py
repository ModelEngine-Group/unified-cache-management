from __future__ import annotations

import dataclasses
import datetime as dt
import importlib
import logging
import os
import platform as pf
import random
import sys
from pathlib import Path

import pynvml
import pytest
from common.capture_utils import export_vars
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
def _generate_build_id(config: pytest.Config) -> str:
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
    build_id = _generate_build_id(config)
    config._build_id = build_id

    for item in config_instance.get_config("results", []):
        if isinstance(item, dict) and item:
            backend_name = next(iter(item.keys()))
            mod = importlib.import_module(f"common.capture_results.{backend_name}")
            mod.set_build_id(build_id)


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


def get_free_gpu(required_memory_mb):
    try:
        mem_needed_with_buffer = int(
            required_memory_mb * 1.3
        )  # add buffer to avoid OOM
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        device_indices = list(range(device_count))
        random.shuffle(device_indices)
        for i in device_indices:  # random order to reduce collisions
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_in_mb = info.free / 1024**2
            if free_in_mb >= mem_needed_with_buffer:
                utilization = (
                    required_memory_mb * (1024**2) / info.total if info.total else 0
                )
                return i, free_in_mb, utilization
    finally:
        pynvml.nvmlShutdown()
    return None, 0, 0


@pytest.fixture(autouse=True)
def setup_gpu_resource(request):
    marker = request.node.get_closest_marker("gpu_mem")
    if marker:
        mem_needed = marker.args[0]
        gpu_id, free_in_mb, gpu_utilization = get_free_gpu(mem_needed)
        if gpu_id is not None:
            print(
                f"Allocating GPU {gpu_id} with {free_in_mb}MB free memory, gpu utilization for test {gpu_utilization:.4%}"
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            if gpu_utilization:
                os.environ["E2E_TEST_GPU_MEMORY_UTILIZATION"] = str(gpu_utilization)
        else:
            pytest.fail(
                f"No GPU with {mem_needed}MB(+30% buffer) free memory available"
            )


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

    logger.info(
        f"Starting JSONL to CSV conversion for {len(jsonl_files)} files in {backup_dir}"
    )

    success_count = 0
    for jsonl_file in jsonl_files:
        try:
            from common.capture_results.localFile import jsonl_to_csv

            csv_file = jsonl_to_csv(jsonl_file, flatten=True)
            logger.info(f"Converted: {jsonl_file.name} → {csv_file.name}")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to convert {jsonl_file.name}: {e}", exc_info=True)

    logger.info(
        f"Conversion complete: {success_count}/{len(jsonl_files)} files converted"
    )
