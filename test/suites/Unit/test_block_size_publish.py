"""CPU tests for the worker-to-scheduler store block_size handoff files."""

import ast
import glob
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "ucm_connector.py"
SELECTED_NAMES = {"_worker_publish_block_size", "_scheduler_read_block_size"}

ENGINE_PID = 4242
WORKER_PID = 5151


def _load_handoff_source():
    """Extract the handoff helpers so the test needs neither vLLM nor torch."""

    tree = ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8-sig"))
    nodes = [
        node for node in tree.body if getattr(node, "name", None) in SELECTED_NAMES
    ]
    assert {node.name for node in nodes} == SELECTED_NAMES
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, str(CONNECTOR_PATH), "exec")


HANDOFF_CODE = _load_handoff_source()


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    """Run the helpers against tmp_path as a worker of ENGINE_PID."""

    namespace = {
        "_SHM_DIR": str(tmp_path),
        "glob": glob,
        "os": os,
        "time": time,
        "logger": Mock(),
        "world": SimpleNamespace(rank_in_group=0),
    }
    namespace["get_world_group"] = lambda: namespace["world"]
    exec(HANDOFF_CODE, namespace)

    def as_process(pid, ppid):
        monkeypatch.setattr(os, "getpid", lambda: pid)
        monkeypatch.setattr(os, "getppid", lambda: ppid)

    # Worker processes are children of the EngineCore process.
    as_process(WORKER_PID, ENGINE_PID)
    return SimpleNamespace(
        dir=tmp_path,
        ns=namespace,
        publish=namespace["_worker_publish_block_size"],
        read=namespace["_scheduler_read_block_size"],
        as_process=as_process,
    )


def as_scheduler(handoff):
    # The scheduler runs inside the EngineCore process itself.
    handoff.as_process(ENGINE_PID, 1)


def test_suffixed_file_name_and_round_trip(handoff):
    handoff.publish(4096, 0, store_suffix="fa")
    assert (handoff.dir / f"ucm_blocksize_{ENGINE_PID}_fa").read_text() == "4096"
    as_scheduler(handoff)
    assert handoff.read(store_suffix="fa") == 4096


def test_fa_and_wa_files_do_not_overwrite(handoff):
    handoff.publish(1000, 0, store_suffix="fa")
    handoff.publish(2000, 0, store_suffix="wa")
    as_scheduler(handoff)
    assert handoff.read(store_suffix="fa") == 1000
    assert handoff.read(store_suffix="wa") == 2000


def test_unsuffixed_name_unchanged_for_direct_connector(handoff):
    handoff.publish(8192, 0)
    assert (handoff.dir / f"ucm_blocksize_{ENGINE_PID}").read_text() == "8192"
    as_scheduler(handoff)
    assert handoff.read() == 8192
    assert handoff.read(store_suffix="fa") is None


def test_no_temp_file_left_behind(handoff):
    handoff.publish(4096, 0, store_suffix="fa")
    assert [p.name for p in handoff.dir.iterdir()] == [f"ucm_blocksize_{ENGINE_PID}_fa"]


@pytest.mark.parametrize("dp_rank,rank_in_group", [(1, 0), (0, 1)])
def test_only_dp0_rank0_publishes(handoff, dp_rank, rank_in_group):
    handoff.ns["world"] = SimpleNamespace(rank_in_group=rank_in_group)
    handoff.publish(4096, dp_rank, store_suffix="fa")
    assert list(handoff.dir.iterdir()) == []


def test_stale_cleanup_covers_suffixed_files(handoff):
    old = time.time() - 601
    stale = [
        handoff.dir / "ucm_blocksize_111",
        handoff.dir / "ucm_blocksize_111_fa",
        handoff.dir / "ucm_blocksize_111_wa",
    ]
    fresh = handoff.dir / "ucm_blocksize_222_wa"
    for path in [*stale, fresh]:
        path.write_text("1")
    for path in stale:
        os.utime(path, (old, old))

    handoff.publish(4096, 0, store_suffix="fa")

    assert not any(path.exists() for path in stale)
    assert fresh.exists()
    assert (handoff.dir / f"ucm_blocksize_{ENGINE_PID}_fa").exists()


def test_scheduler_prefers_own_pid_then_parent(handoff):
    (handoff.dir / f"ucm_blocksize_{ENGINE_PID}_fa").write_text("100")
    (handoff.dir / "ucm_blocksize_1_fa").write_text("200")
    as_scheduler(handoff)
    assert handoff.read(store_suffix="fa") == 100

    # A scheduler in a child of EngineCore still finds the published file.
    handoff.as_process(9999, ENGINE_PID)
    assert handoff.read(store_suffix="fa") == 100


def test_missing_file_returns_none_and_names_paths(handoff):
    as_scheduler(handoff)
    assert handoff.read(store_suffix="wa") is None
    message = handoff.ns["logger"].warning.call_args.args[0]
    assert f"ucm_blocksize_{ENGINE_PID}_wa" in message
    assert "ucm_blocksize_1_wa" in message
    assert "fall back" not in message


@pytest.mark.parametrize("content", ["abc", "", "0", "-4096"])
def test_invalid_content_returns_none(handoff, content):
    (handoff.dir / f"ucm_blocksize_{ENGINE_PID}_fa").write_text(content)
    as_scheduler(handoff)
    assert handoff.read(store_suffix="fa") is None
    handoff.ns["logger"].warning.assert_called_once()
