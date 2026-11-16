"""CPU-only checks for the PP-aware layer-wise connector store wiring.

Like test_hla_connector.py, extract the reviewed methods from
ucm_connector.py with AST so the tests run without torch, vLLM or NPU
libraries. The filesystem is faked as well, so the tests only assert the
stage directory layout the connector *asks for* rather than touching a real
backend. They do not exercise the C++ stores, shard-index mapping or
end-to-end PP accuracy. Reset checks use real loopback sockets, including
two separate worker processes, with in-memory stand-ins for native stores.

Run with ``--noconftest``; the repository conftest pulls in the end-to-end
harness (pynvml, evaluator configs) which these tests do not need.
"""

import ast
import copy
import importlib.util
import math
import multiprocessing
import posixpath
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_PATH = ROOT / "ucm/integration/vllm/ucm_connector.py"
RESET_PATH = ROOT / "ucm/integration/vllm/cache_reset.py"
reset_spec = importlib.util.spec_from_file_location("ucm_cache_reset", RESET_PATH)
reset_module = importlib.util.module_from_spec(reset_spec)
reset_spec.loader.exec_module(reset_module)


class KVConnectorRole(Enum):
    WORKER = "worker"
    SCHEDULER = "scheduler"


CLASS_NAME = "UCMLayerWiseConnector"
METHODS = {
    "_pp_stage_name",
    "_pp_stage_paths",
    "_configure_pp_store",
    "_lookup_all_stages",
    "_get_lookup_pool",
    "shutdown",
    "get_handshake_metadata",
    "set_xfer_handshake_metadata_pp_aware",
    "reset_cache",
}

DIRECT_CLASS = "UCMDirectConnector"
DIRECT_METHODS = {
    "_lookup_all_stages",
    "_lookup_one_stage",
    "_record_load_error",
    "get_block_ids_with_load_errors",
    "reset_cache",
}

BACKEND = "/data/ucm"


class FakePath:
    """Minimal os.path stand-in; records the directories the code creates."""

    def __init__(self, existing=()):
        self.existing = set(existing)
        self.created: list[str] = []

    def isdir(self, path) -> bool:
        return str(path) in self.existing

    def join(self, *parts) -> str:
        return posixpath.join(*parts)

    def makedirs(self, path, exist_ok: bool = False) -> None:
        path = str(path)
        if path in self.existing:
            if exist_ok:
                return
            raise FileExistsError(path)
        self.existing.add(path)
        self.created.append(path)


class FakeOs:
    def __init__(self, existing=()):
        self.path = FakePath(existing)

    def makedirs(self, path, exist_ok: bool = False) -> None:
        self.path.makedirs(path, exist_ok=exist_ok)


def _load_connector_class(existing=(BACKEND,)):
    nodes = []
    for node in ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME:
            node.bases = [ast.Name(id=DIRECT_CLASS, ctx=ast.Load())]
            node.body = [
                item for item in node.body if getattr(item, "name", None) in METHODS
            ]
            nodes.append(node)
    if not nodes:
        raise AssertionError(f"{CLASS_NAME} missing from ucm_connector.py")

    factory = MagicMock()
    fake_os = FakeOs(existing)
    logger = MagicMock()
    namespace = {
        "__name__": __name__,
        "Any": Any,
        "Optional": Optional,
        "math": math,
        "os": fake_os,
        "copy": copy,
        "logger": logger,
        "UcmConnectorFactoryV1": SimpleNamespace(create_connector=factory),
        "ThreadPoolExecutor": ThreadPoolExecutor,
        DIRECT_CLASS: _load_direct_class(),
        "KVConnectorRole": KVConnectorRole,
        "CacheResetServer": lambda reset: reset_module.CacheResetServer(
            reset, host="127.0.0.1"
        ),
        "reset_cache_endpoints": reset_module.reset_cache_endpoints,
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    connector_cls = namespace[CLASS_NAME]
    connector_cls._test_logger = logger
    return connector_cls, factory, fake_os


def _load_direct_class():
    """Extract the base connector's load-error bookkeeping and lookup helpers."""
    nodes = []
    for node in ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.ClassDef) and node.name == DIRECT_CLASS:
            node.bases = []
            node.body = [
                item
                for item in node.body
                if getattr(item, "name", None) in DIRECT_METHODS
            ]
            nodes.append(node)
    if not nodes:
        raise AssertionError(f"{DIRECT_CLASS} missing from ucm_connector.py")

    namespace = {
        "__name__": __name__,
        "Any": Any,
        "Optional": Optional,
        "ucmmetrics": MagicMock(),
        "logger": MagicMock(),
        "os": FakeOs(),
        "ThreadPoolExecutor": ThreadPoolExecutor,
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace[DIRECT_CLASS]


def _make_connector(
    cls,
    *,
    pp_size: int = 2,
    world_size: int = 8,
    tp_size: int = 4,
    is_mla: bool = False,
    gc_owner: bool = False,
    buffer_gb: int = 128,
    posix_gb: int = 0,
    store_name: str = "UcmPipelineStore",
    module_path: Optional[str] = None,
    storage_backends: Any = BACKEND,
):
    connector_config = {
        "store_pipeline": "Cache|Posix",
        "share_buffer_enable": True,
        "cache_buffer_capacity_gb": buffer_gb,
        "posix_capacity_gb": posix_gb,
    }
    if storage_backends is not None:
        connector_config["storage_backends"] = storage_backends

    entry = {
        "ucm_connector_name": store_name,
        "ucm_connector_config": connector_config,
    }
    if module_path is not None:
        entry["ucm_connector_module_path"] = module_path

    conn = cls.__new__(cls)
    conn._role = KVConnectorRole.SCHEDULER
    conn._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
    )
    conn._get_world_size = lambda: world_size
    conn.tp_size = tp_size
    conn.is_mla = is_mla
    conn.unique_id = "engine"
    conn._gc_owner = gc_owner
    conn.connector_configs = [entry]
    conn._set_default_shm_buffer_capacity = lambda config: config.setdefault(
        "cache_buffer_capacity_gb", buffer_gb
    )
    return conn


def _metadata(allocations: list[tuple[int, int]]) -> dict:
    """Handshake metadata keyed by (pp_rank, 0), as vLLM delivers it."""
    return {
        (rank, 0): {
            "block_size": 12288 + rank,
            "shard_size": 4096,
            "layer_start": start,
            "layer_count": count,
        }
        for rank, (start, count) in enumerate(allocations)
    }


# --------------------------------------------------------------------------
# stage naming / directories
# --------------------------------------------------------------------------


def test_stage_name_encodes_layer_allocation():
    cls, _, _ = _load_connector_class()

    assert cls._pp_stage_name(1, 4, 4) == "pp_1_l4-4"
    assert cls._pp_stage_name(0, 0, 2) == "pp_0_l0-2"


def test_stage_name_separates_repartitioned_stages():
    """The 4,4,4 -> 2,4,6 case: stage 1 keeps 4 layers but starts at 2."""
    cls, _, _ = _load_connector_class()

    before = cls._pp_stage_name(1, 4, 4)
    after = cls._pp_stage_name(1, 2, 4)

    assert before != after


def test_stage_paths_creates_layer_scoped_directory():
    cls, _, fake_os = _load_connector_class()

    paths = cls._pp_stage_paths(BACKEND, 1, 4, 4)

    assert paths == [f"{BACKEND}/pp_1_l4-4"]
    assert fake_os.path.created == [f"{BACKEND}/pp_1_l4-4"]


def test_stage_paths_do_not_reuse_directory_after_repartition():
    cls, _, fake_os = _load_connector_class()

    cls._pp_stage_paths(BACKEND, 1, 4, 4)
    cls._pp_stage_paths(BACKEND, 1, 2, 4)

    assert fake_os.path.created == [
        f"{BACKEND}/pp_1_l4-4",
        f"{BACKEND}/pp_1_l2-4",
    ]


def test_stage_paths_is_idempotent_for_existing_directory():
    cls, _, fake_os = _load_connector_class(existing={BACKEND, f"{BACKEND}/pp_0_l0-4"})

    paths = cls._pp_stage_paths(BACKEND, 0, 0, 4)

    assert paths == [f"{BACKEND}/pp_0_l0-4"]
    # exist_ok=True: an already present stage directory is reused, not an error.
    assert fake_os.path.created == []


def test_stage_paths_rejects_empty_root():
    cls, _, _ = _load_connector_class()

    with pytest.raises(ValueError, match="nonempty shared Posix roots"):
        cls._pp_stage_paths("", 0, 0, 4)


def test_stage_paths_rejects_missing_root():
    cls, _, _ = _load_connector_class(existing={BACKEND})

    with pytest.raises(FileNotFoundError, match="does not exist"):
        cls._pp_stage_paths("/data/absent", 0, 0, 4)


def test_stage_paths_supports_multiple_roots():
    cls, _, fake_os = _load_connector_class(existing={"/data/a", "/data/b"})

    paths = cls._pp_stage_paths("/data/a:/data/b", 3, 6, 2)

    assert paths == ["/data/a/pp_3_l6-2", "/data/b/pp_3_l6-2"]
    assert fake_os.path.created == ["/data/a/pp_3_l6-2", "/data/b/pp_3_l6-2"]


# --------------------------------------------------------------------------
# _configure_pp_store
# --------------------------------------------------------------------------


def test_configure_pp_store_splits_budget_per_stage():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=2, world_size=8, tp_size=4, posix_gb=1024)
    config = copy.deepcopy(conn.connector_configs[0]["ucm_connector_config"])

    conn._configure_pp_store(config, 1, 4, 4)

    # 8 ranks on the node / tp 4 = 2 stages share the node buffer budget.
    assert config["cache_buffer_capacity_gb"] == 64
    # Disk capacity is split across all PP stages, not just the local ones.
    assert config["posix_capacity_gb"] == 512
    assert config["storage_backends"] == [f"{BACKEND}/pp_1_l4-4"]
    assert config["local_rank_size"] == 4


def test_configure_pp_store_namespaces_cache_id_by_layer_allocation():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls)
    before = copy.deepcopy(conn.connector_configs[0]["ucm_connector_config"])
    after = copy.deepcopy(conn.connector_configs[0]["ucm_connector_config"])

    conn._configure_pp_store(before, 1, 4, 4)
    conn._configure_pp_store(after, 1, 2, 4)

    assert before["unique_id"] == "engine_pp_1_l4-4"
    assert after["unique_id"] == "engine_pp_1_l2-4"


def test_configure_pp_store_rejects_zero_buffer_capacity():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=2, world_size=8, tp_size=4, buffer_gb=1)
    config = copy.deepcopy(conn.connector_configs[0]["ucm_connector_config"])

    with pytest.raises(ValueError, match="cache_buffer_capacity_gb=1 is too small"):
        conn._configure_pp_store(config, 0, 0, 4)


def test_configure_pp_store_rejects_zero_posix_capacity():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=4, posix_gb=3)
    config = copy.deepcopy(conn.connector_configs[0]["ucm_connector_config"])

    with pytest.raises(ValueError, match="posix_capacity_gb=3 is too small"):
        conn._configure_pp_store(config, 0, 0, 3)


def test_configure_pp_store_requires_storage_backends():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls, storage_backends=None)
    config = copy.deepcopy(conn.connector_configs[0]["ucm_connector_config"])

    with pytest.raises(ValueError, match="storage_backends"):
        conn._configure_pp_store(config, 0, 0, 4)


def test_configure_pp_store_splits_domain_budget_with_legacy_share_flag_disabled():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls, buffer_gb=8)
    config = copy.deepcopy(conn.connector_configs[0]["ucm_connector_config"])
    config["share_buffer_enable"] = False

    conn._configure_pp_store(config, 0, 0, 4)

    assert config["cache_buffer_capacity_gb"] == 4


# --------------------------------------------------------------------------
# get_handshake_metadata
# --------------------------------------------------------------------------


def test_get_handshake_metadata_is_none_without_pp():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=1)
    conn._is_pp = False

    assert conn.get_handshake_metadata() is None


def test_get_handshake_metadata_publishes_geometry_and_allocation():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls)
    conn._is_pp = True
    conn._pp_store_sizes = {
        "block_size": 12288,
        "shard_size": 4096,
        "layer_start": 4,
        "layer_count": 4,
    }
    conn.store = SimpleNamespace(reset_cache=lambda: True)

    try:
        metadata = conn.get_handshake_metadata()
        endpoint = metadata.pop("cache_reset_endpoint")
        assert metadata == conn._pp_store_sizes
        assert reset_module.reset_cache_endpoints([endpoint]) is True
        assert conn.get_handshake_metadata()["cache_reset_endpoint"] == endpoint
        assert "cache_reset_endpoint" not in conn._pp_store_sizes
    finally:
        conn.shutdown()


# --------------------------------------------------------------------------
# set_xfer_handshake_metadata_pp_aware
# --------------------------------------------------------------------------


def test_handshake_creates_one_store_per_stage():
    cls, factory, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=2)
    conn._is_pp = True

    conn.set_xfer_handshake_metadata_pp_aware(_metadata([(0, 4), (4, 4)]))

    assert factory.call_count == 2
    assert len(conn._pp_stores) == 2
    assert conn.store is conn._pp_stores[0]


def test_handshake_is_idempotent():
    cls, factory, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=2)
    conn._is_pp = True
    metadata = _metadata([(0, 4), (4, 4)])

    conn.set_xfer_handshake_metadata_pp_aware(metadata)
    first_stores = list(conn._pp_stores)
    conn.set_xfer_handshake_metadata_pp_aware(metadata)

    # A re-sent handshake after sleep/wake_up must not rebuild (and leak) stores.
    assert factory.call_count == 2
    assert conn._pp_stores == first_stores


def test_handshake_uses_configured_connector_and_module_path():
    cls, factory, _ = _load_connector_class()
    conn = _make_connector(
        cls, pp_size=1, store_name="CustomStore", module_path="custom.store"
    )
    conn._is_pp = True

    conn.set_xfer_handshake_metadata_pp_aware(_metadata([(0, 4)]))

    args = factory.call_args
    assert args.args[0] == "CustomStore"
    assert args.args[2] == "custom.store"


def test_handshake_reports_missing_stage_metadata():
    cls, _, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=2)
    conn._is_pp = True

    # Stage 0 is complete, stage 1 never reported.
    with pytest.raises(ValueError, match=r"PP stage 1, worker coordinates zero"):
        conn.set_xfer_handshake_metadata_pp_aware(_metadata([(0, 4)]))


def test_handshake_requires_layer_allocation():
    """Without the allocation the scheduler cannot name the stage directory."""
    cls, factory, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=1)
    conn._is_pp = True
    metadata = {(0, 0): {"block_size": 12288, "shard_size": 4096}}

    with pytest.raises(ValueError, match="missing required fields:.*layer_start"):
        conn.set_xfer_handshake_metadata_pp_aware(metadata)

    assert factory.call_count == 0


def test_handshake_keeps_layer_allocation_out_of_store_config():
    """layer_start/layer_count are scheduler inputs, not store options."""
    cls, factory, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=1)
    conn._is_pp = True

    conn.set_xfer_handshake_metadata_pp_aware(_metadata([(0, 4)]))

    store_config = factory.call_args.args[1]
    assert "layer_start" not in store_config
    assert "layer_count" not in store_config


def test_handshake_only_pp0_keeps_cache_pipeline():
    cls, factory, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=2)
    conn._is_pp = True

    conn.set_xfer_handshake_metadata_pp_aware(_metadata([(0, 4), (4, 4)]))

    pp0_config = factory.call_args_list[0].args[1]
    pp1_config = factory.call_args_list[1].args[1]
    # PP0 keeps the configured Cache|Posix pipeline and owns hit decisions.
    assert pp0_config["store_pipeline"] == "Cache|Posix"
    # Later stages are Posix-only: they serve GC and hotness, never hits.
    assert pp1_config["store_pipeline"] == "Posix"
    assert pp1_config["unique_id"] == "engine_pp_1_l4-4"
    assert pp1_config["storage_backends"] == [f"{BACKEND}/pp_1_l4-4"]
    # Geometry comes from the handshake, not from a per-layer size estimate.
    assert pp1_config["block_size"] == 12289
    assert pp1_config["shard_size"] == 4096


def test_handshake_is_noop_without_pp():
    cls, factory, _ = _load_connector_class()
    conn = _make_connector(cls, pp_size=1)
    conn._is_pp = False

    conn.set_xfer_handshake_metadata_pp_aware(_metadata([(0, 4)]))

    assert factory.call_count == 0


def _reset_handshake(cls, factory, stack, *, fail_stage=None, missing_rank=None):
    """Use real reset sockets while scheduler stores remain Cache|Posix/Posix."""
    metadata = _metadata([(0, 4), (4, 4)])
    dram = [{b"block"}, {b"block"}]
    persistent = [{b"block"}, {b"block"}]
    calls = [0, 0]
    for pp_rank in range(2):

        def reset(stage=pp_rank):
            calls[stage] += 1
            if stage == fail_stage:
                raise RuntimeError("reset failed")
            dram[stage].clear()
            return True

        server = reset_module.CacheResetServer(reset, host="127.0.0.1")
        stack.callback(server.close)
        for tp_rank in range(2):
            rank = (pp_rank, tp_rank)
            metadata[rank] = dict(metadata[(pp_rank, 0)])
            if rank != missing_rank:
                metadata[rank]["cache_reset_endpoint"] = server.endpoint
            else:
                metadata[rank].pop("cache_reset_endpoint", None)
    conn = _make_connector(cls, tp_size=2, world_size=4)
    conn._is_pp = True
    conn.set_xfer_handshake_metadata_pp_aware(metadata)
    assert factory.call_args_list[1].args[1]["store_pipeline"] == "Posix"
    assert all(
        "cache_reset_endpoint" not in call.args[1] for call in factory.call_args_list
    )
    return conn, dram, persistent, calls


def test_pp_reset_clears_all_worker_domains_and_keeps_persistent_data():
    cls, factory, _ = _load_connector_class(Caps(partitioned=True))
    with ExitStack() as stack:
        conn, dram, persistent, calls = _reset_handshake(cls, factory, stack)

        assert conn.reset_cache() is True
        assert dram == [set(), set()]
        assert persistent == [{b"block"}, {b"block"}]
        # Two TP workers share each stage's host domain; reset it only once.
        assert calls == [1, 1]
        # Scheduler Posix-only stores cannot clear worker Cache memory.
        factory.return_value.reset_cache.assert_not_called()


@pytest.mark.parametrize("failed_stage", [0, 1])
def test_pp_reset_reports_failure_and_still_attempts_other_stages(failed_stage):
    cls, factory, _ = _load_connector_class(Caps(partitioned=True))
    with ExitStack() as stack:
        conn, dram, _, calls = _reset_handshake(
            cls, factory, stack, fail_stage=failed_stage
        )

        assert conn.reset_cache() is False
        assert calls == [1, 1]
        assert dram[failed_stage] == {b"block"}
        assert dram[1 - failed_stage] == set()


def test_pp_reset_rejects_incomplete_worker_handshake_before_clearing():
    cls, factory, _ = _load_connector_class(Caps(partitioned=True))
    with ExitStack() as stack:
        conn, dram, _, calls = _reset_handshake(
            cls, factory, stack, missing_rank=(1, 1)
        )

        assert conn.reset_cache() is False
        assert dram == [{b"block"}, {b"block"}]
        assert calls == [0, 0]


def test_pp_reset_before_handshake_reports_failure():
    cls, _, _ = _load_connector_class(Caps())
    conn = _make_connector(cls)
    conn._is_pp = True

    assert conn.reset_cache() is False


@pytest.mark.parametrize(
    "is_pp,role", [(False, KVConnectorRole.SCHEDULER), (True, KVConnectorRole.WORKER)]
)
def test_local_reset_uses_the_existing_store_hook(is_pp, role):
    cls, _, _ = _load_connector_class(Caps())
    conn = _make_connector(cls)
    conn._is_pp = is_pp
    conn._role = role
    conn.store = SimpleNamespace(reset_cache=MagicMock(return_value=True))

    assert conn.reset_cache() is True
    conn.store.reset_cache.assert_called_once_with()


def test_reset_reply_waits_for_worker_completion():
    entered = threading.Event()
    release = threading.Event()

    def reset():
        entered.set()
        return release.wait(timeout=3)

    server = reset_module.CacheResetServer(reset, host="127.0.0.1")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(reset_module.reset_cache_endpoints, [server.endpoint])
            try:
                assert entered.wait(timeout=3)
                assert not result.done()
            finally:
                release.set()
            assert result.result(timeout=3) is True
    finally:
        server.close()


def test_reset_rejects_wrong_token_without_touching_cache():
    reset = MagicMock(return_value=True)
    server = reset_module.CacheResetServer(reset, host="127.0.0.1")
    try:
        wrong = {**server.endpoint, "token": "00" * 32}
        assert reset_module.reset_cache_endpoints([wrong]) is False
        reset.assert_not_called()
    finally:
        server.close()


def test_reset_reports_closed_endpoint_and_continues_to_live_domain():
    closed = reset_module.CacheResetServer(lambda: True, host="127.0.0.1")
    closed.close()
    reset = MagicMock(return_value=True)
    live = reset_module.CacheResetServer(reset, host="127.0.0.1")
    try:
        assert (
            reset_module.reset_cache_endpoints(
                [closed.endpoint, live.endpoint], timeout=0.5
            )
            is False
        )
        reset.assert_called_once_with()
    finally:
        live.close()


def test_reset_timeout_reports_failure_and_continues_to_other_domains():
    release = threading.Event()
    slow = reset_module.CacheResetServer(
        lambda: release.wait(timeout=3), host="127.0.0.1"
    )
    reset = MagicMock(return_value=True)
    live = reset_module.CacheResetServer(reset, host="127.0.0.1")
    try:
        assert (
            reset_module.reset_cache_endpoints(
                [slow.endpoint, live.endpoint], timeout=0.05
            )
            is False
        )
        reset.assert_called_once_with()
    finally:
        release.set()
        slow.close()
        live.close()


def test_pp_reset_collects_separate_hosts_within_the_same_stage():
    cls, factory, _ = _load_connector_class(Caps(partitioned=True))
    metadata = _metadata([(0, 4), (4, 4)])
    for pp_rank in range(2):
        for tp_rank in range(2):
            rank = (pp_rank, tp_rank)
            metadata[rank] = dict(metadata[(pp_rank, 0)])
            metadata[rank]["cache_reset_endpoint"] = {
                "host": f"node-{tp_rank}",
                "port": 1234,
                "token": "00" * 32,
            }
    conn = _make_connector(cls, tp_size=2, world_size=2)
    conn._is_pp = True
    conn.set_xfer_handshake_metadata_pp_aware(metadata)

    assert sorted(endpoint["host"] for endpoint in conn._pp_reset_endpoints) == [
        "node-0",
        "node-0",
        "node-1",
        "node-1",
    ]


def _reset_worker_process(pipe):
    dram = {b"block"}
    persistent = {b"block"}

    def reset():
        dram.clear()
        return True

    server = reset_module.CacheResetServer(reset, host="127.0.0.1")
    try:
        pipe.send(server.endpoint)
        while pipe.recv() == "inspect":
            pipe.send((dram, persistent))
    finally:
        server.close()
        pipe.close()


def test_pp_reset_reaches_two_separate_worker_processes():
    cls, factory, _ = _load_connector_class(Caps(partitioned=True))
    ctx = multiprocessing.get_context("spawn")
    workers = []
    try:
        metadata = _metadata([(0, 4), (4, 4)])
        for pp_rank in range(2):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_reset_worker_process, args=(child,))
            process.start()
            child.close()
            workers.append((process, parent))
            assert parent.poll(10)
            metadata[(pp_rank, 0)]["cache_reset_endpoint"] = parent.recv()
        conn = _make_connector(cls, tp_size=1, world_size=2)
        conn._is_pp = True
        conn.set_xfer_handshake_metadata_pp_aware(metadata)

        assert conn.reset_cache() is True
        for _, pipe in workers:
            pipe.send("inspect")
            assert pipe.poll(3)
            assert pipe.recv() == (set(), {b"block"})
    finally:
        for process, pipe in workers:
            if process.is_alive():
                pipe.send("close")
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            pipe.close()


# --------------------------------------------------------------------------
# Hits require every PP stage to hold the block
# --------------------------------------------------------------------------


def _store_with_hit(hit: int):
    return SimpleNamespace(lookup_on_prefix=lambda block_ids: hit)


def _lookup_connector(hits: list[int]):
    """Build a layer-wise connector whose stage stores report the hit indices."""
    cls, _, _ = _load_connector_class()
    connector = cls.__new__(cls)
    stores = [_store_with_hit(hit) for hit in hits]
    connector.store = stores[0]
    connector._pp_stores = stores
    connector._is_pp = True
    # _lookup_one_stage lives on the base class, which is not extracted here.
    connector._lookup_one_stage = lambda store, block_ids: store.lookup_on_prefix(
        block_ids
    )
    return connector


def _direct_lookup_connector(hit: int):
    """Build a non-PP connector, which keeps the base single-store lookup."""
    cls = _load_direct_class()
    connector = cls.__new__(cls)
    connector.store = _store_with_hit(hit)
    connector._rank_consistency = SimpleNamespace(
        lookup_on_prefix=lambda store, block_ids: store.lookup_on_prefix(block_ids)
    )
    return connector


def test_lookup_reports_the_minimum_hit_across_stages():
    connector = _lookup_connector([5, 5, 5])
    try:
        assert connector._lookup_all_stages([b"k"] * 8) == 5
    finally:
        connector.shutdown()

    connector = _lookup_connector([5, 2, 4])
    try:
        assert connector._lookup_all_stages([b"k"] * 8) == 2
    finally:
        connector.shutdown()


def test_lookup_queries_every_stage_in_parallel():
    hits = [5, 2, 4]
    connector = _lookup_connector(hits)
    try:
        assert connector._lookup_all_stages([b"k"] * 8) == 2
        pool = connector._lookup_pool
        assert pool is not None
        # One worker per stage, so the remote queries overlap.
        assert pool._max_workers == len(hits)
        # The same pool is reused across lookups (threads are not per-call).
        assert connector._lookup_all_stages([b"k"] * 8) == 2
        assert connector._lookup_pool is pool
    finally:
        connector.shutdown()
    assert connector._lookup_pool is None


def test_lookup_misses_when_any_stage_misses():
    """PP0 holds the block, PP1 does not -> the block must not be a hit."""
    connector = _lookup_connector([7, -1])
    try:
        assert connector._lookup_all_stages([b"k"] * 8) == -1
    finally:
        connector.shutdown()

    connector = _lookup_connector([-1, 7])
    assert connector._lookup_all_stages([b"k"] * 8) == -1


def test_lookup_stops_early_only_on_a_full_miss():
    calls: list[int] = []

    def make(hit: int, index: int):
        def lookup(block_ids):
            calls.append(index)
            return hit

        return SimpleNamespace(lookup_on_prefix=lookup)

    cls, _, _ = _load_connector_class()
    connector = cls.__new__(cls)
    connector._is_pp = True
    connector._lookup_one_stage = lambda store, block_ids: store.lookup_on_prefix(
        block_ids
    )
    connector.store = make(0, 0)
    connector._pp_stores = [make(0, 0), make(9, 1)]

    # PP0 hitting the first block is not a final answer: a later stage can
    # still miss everything, which makes the common prefix empty.
    assert connector._lookup_all_stages([b"k"] * 8) == 0
    assert calls == [0, 1]

    calls.clear()
    connector._pp_stores = [make(0, 0), make(0, 1), make(-1, 2)]
    # A later stage hitting only the first block must not stop the scan before
    # the stage that misses everything is reached.
    assert connector._lookup_all_stages([b"k"] * 8) == -1
    assert calls == [0, 1, 2]

    calls.clear()
    connector._pp_stores = [make(-1, 0), make(9, 1)]
    # A full miss on PP0 stops the scan before its successors are queried.
    assert connector._lookup_all_stages([b"k"] * 8) == -1
    assert calls == [0]


def test_lookup_without_pp_uses_the_single_store():
    connector = _direct_lookup_connector(3)

    assert connector._lookup_all_stages([b"k"] * 4) == 3


def test_lookup_propagates_store_errors():
    """The caller records the error and treats the lookup as a miss."""

    def boom(block_ids):
        raise OSError("posix lookup failed")

    cls, _, _ = _load_connector_class()
    connector = cls.__new__(cls)
    connector._is_pp = True
    connector._lookup_one_stage = lambda store, block_ids: store.lookup_on_prefix(
        block_ids
    )
    connector.store = _store_with_hit(3)
    connector._pp_stores = [_store_with_hit(3), SimpleNamespace(lookup_on_prefix=boom)]

    try:
        with pytest.raises(OSError, match="posix lookup failed"):
            connector._lookup_all_stages([b"k"] * 4)
    finally:
        connector.shutdown()


def test_partial_stage_load_error_still_surfaces_as_a_safety_net():
    """Even with intersection a block can be evicted between lookup and load;
    that stage load error must still reach vLLM exactly once."""
    cls = _load_direct_class()
    connector = cls.__new__(cls)
    connector._invalid_block_ids = set()

    connector._record_load_error("connector_load_wait_errors_total", [7])

    assert connector.get_block_ids_with_load_errors() == {7}
    # vLLM consumed the report; the same block must not be reported twice.
    assert connector.get_block_ids_with_load_errors() == set()


# --------------------------------------------------------------------------
# CP + PP uses the layer-wise parent
# --------------------------------------------------------------------------


def test_cp_connector_accepts_pp_and_reaches_layerwise_parent():
    tree = ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8-sig"))
    cp_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UCMCPConnector"
    )
    init = next(
        node
        for node in cp_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    class ParentReached(RuntimeError):
        pass

    class Parent:
        def __init__(self, *args):
            raise ParentReached("layerwise parent reached")

    node = ast.ClassDef(
        name="CPUnderTest",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[init],
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    namespace = {"Parent": Parent}
    exec(
        compile(ast.fix_missing_locations(module), str(CONNECTOR_PATH), "exec"),
        namespace,
    )
    config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
    with pytest.raises(ParentReached, match="layerwise parent reached"):
        namespace["CPUnderTest"](config, None)
