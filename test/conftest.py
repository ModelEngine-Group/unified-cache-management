from __future__ import annotations

import datetime as dt
import platform as pf
import sys
from functools import wraps
from pathlib import Path

import pytest
from common.config_utils import config_utils as config_instance
from common.db_utils import database_connection, write_to_db

# ---------------- Constants ----------------
PRJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PRJ_ROOT))


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
        if name == "forked":
            continue
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
    args_part = "_".join(cli_parts) if cli_parts else "all_cases"
    return f"pytest_{ts}_{args_part}"


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
    database_connection(build_id)


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
    write_to_db("test_case_info", test_result)
