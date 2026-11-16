"""CPU-only checks for the outer UCMConnector cache reset handshake.

The reviewed methods are extracted from ucm_connector.py with AST so the
tests run without torch, vLLM or NPU libraries. Reset requests use real
loopback sockets and in-memory stand-ins for the native Cache stores.

Run with ``--noconftest``; the repository conftest pulls in the end-to-end
harness (pynvml, evaluator configs) which these tests do not need.
"""

import ast
import importlib.util
import json
import multiprocessing
import os
import socket
import threading
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR_PATH = ROOT / "ucm/integration/vllm/ucm_connector.py"
HMA_PATH = ROOT / "ucm/integration/vllm/hma_connector.py"
RESET_PATH = ROOT / "ucm/integration/vllm/cache_reset.py"
reset_spec = importlib.util.spec_from_file_location("ucm_cache_reset", RESET_PATH)
reset_module = importlib.util.module_from_spec(reset_spec)
reset_spec.loader.exec_module(reset_module)


class KVConnectorRole(Enum):
    WORKER = "worker"
    SCHEDULER = "scheduler"


class _Logger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append(str(message))

    def info(self, message, *args, **kwargs):
        self.infos.append(str(message))


OUTER_METHODS = {
    "_valid_cache_reset_endpoint",
    "_cache_reset_supported",
    "_cache_reset_ranks",
    "_cache_reset_pp_enabled",
    "_all_workers_on_this_node",
    "_cache_reset_domain",
    "_get_cache_reset_server",
    "_build_cache_reset_plan",
    "_local_cache_reset",
    "reset_cache",
    "get_handshake_metadata",
    "set_xfer_handshake_metadata_pp_aware",
    "shutdown",
}


def _load_outer_class(scheduler_ip: str = "127.0.0.1"):
    """Compile only the outer connector's handshake/reset methods."""
    nodes = []
    for node in ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.ClassDef) and node.name == "UCMConnector":
            node.bases = []
            node.body = [
                item
                for item in node.body
                if getattr(item, "name", None) in OUTER_METHODS
            ]
            nodes.append(node)
    if not nodes:
        raise AssertionError("UCMConnector missing from ucm_connector.py")

    logger = _Logger()
    namespace = {
        "__name__": __name__,
        "Optional": Optional,
        "KVConnectorRole": KVConnectorRole,
        "CacheResetServer": lambda reset, host=None: reset_module.CacheResetServer(
            reset, host=host or "127.0.0.1"
        ),
        "reset_cache_endpoints": reset_module.reset_cache_endpoints,
        "get_local_ip": lambda: scheduler_ip,
        "logger": logger,
        "_record_connector_interface_duration": lambda func: func,
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace["UCMConnector"], namespace


def _load_fawa_reset_method():
    """Return the production UCMFAWAConnector.reset_cache implementation."""
    tree = ast.parse(HMA_PATH.read_text(encoding="utf-8-sig"))
    connector = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UCMFAWAConnector"
    )
    method = next(
        node
        for node in connector.body
        if isinstance(node, ast.FunctionDef) and node.name == "reset_cache"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(HMA_PATH), "exec"), namespace)
    return namespace["reset_cache"]


def _parallel_config(
    *,
    pp: int,
    tp: int,
    world_size: int,
    local_world_size: Optional[int],
    nnodes: int,
    dp: int,
    dp_local: int,
    expose_local_size: bool,
    pcp: int = 1,
    dcp: int = 1,
    backend=None,
):
    config = SimpleNamespace(
        pipeline_parallel_size=pp,
        tensor_parallel_size=tp,
        world_size=world_size,
        nnodes=nnodes,
        data_parallel_size=dp,
        data_parallel_size_local=dp_local,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        distributed_executor_backend=backend,
    )
    if expose_local_size:
        config.local_world_size = (
            world_size if local_world_size is None else local_world_size
        )
    return config


def _make_outer(
    *,
    role=KVConnectorRole.SCHEDULER,
    pp: int = 1,
    tp: int = 1,
    world_size: int = 1,
    local_world_size: Optional[int] = None,
    nnodes: int = 1,
    dp: int = 1,
    dp_local: int = 1,
    expose_local_size: bool = True,
    pcp: int = 1,
    dcp: int = 1,
    backend=None,
    scheduler_ip: str = "127.0.0.1",
    inner=None,
):
    cls, namespace = _load_outer_class(scheduler_ip)
    outer = cls.__new__(cls)
    outer._role = role
    outer._vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(
            pp=pp,
            tp=tp,
            world_size=world_size,
            local_world_size=local_world_size,
            nnodes=nnodes,
            dp=dp,
            dp_local=dp_local,
            expose_local_size=expose_local_size,
            pcp=pcp,
            dcp=dcp,
            backend=backend,
        )
    )
    outer.connector = _DirectInner() if inner is None else inner
    outer._cache_reset_server = None
    outer._cache_reset_plan = None
    return outer, namespace


class _DirectInner:
    """Direct/HLA-style single-store connector."""

    _is_pp = False

    def __init__(self, unique_id: str = "engine", result=True):
        self.unique_id = unique_id
        self.result = result
        self.resets = 0

    def get_handshake_metadata(self):
        return None

    def reset_cache(self):
        self.resets += 1
        return self.result


class _PPInner:
    _is_pp = True

    def __init__(self):
        self.unique_id = "engine"
        self.handshake_calls = []
        self.resets = 0

    def get_handshake_metadata(self):
        return {
            "block_size": 12288,
            "shard_size": 4096,
            "layer_start": 0,
            "layer_count": 2,
        }

    def set_xfer_handshake_metadata_pp_aware(self, metadata):
        self.handshake_calls.append(metadata)

    def reset_cache(self):
        self.resets += 1
        return True


def _endpoint(server, domain: str) -> dict:
    return {**server.endpoint, "host": "127.0.0.1", "domain": domain}


def _metadata(endpoint_by_rank: dict, pp_size: int = 1) -> dict:
    metadata = {}
    for rank, endpoint in endpoint_by_rank.items():
        host = endpoint.get("host") if isinstance(endpoint, dict) else None
        domain = endpoint.get("domain") if isinstance(endpoint, dict) else None
        metadata[rank] = {
            "cache_reset_endpoint": endpoint,
            "cache_reset_host": host,
            "cache_reset_domain": domain,
        }
    return metadata


def _metadata_with_hosts(endpoint_by_rank: dict, hosts: dict) -> dict:
    """Attach topology hosts while still connecting through loopback."""
    metadata = _metadata(endpoint_by_rank)
    for rank, host in hosts.items():
        metadata[rank]["cache_reset_host"] = host
    return metadata


# --------------------------------------------------------------------------
# Worker handshake
# --------------------------------------------------------------------------


def test_worker_publishes_reset_endpoint_and_clears_direct_store():
    inner = _DirectInner()
    outer, _ = _make_outer(role=KVConnectorRole.WORKER, inner=inner)
    try:
        metadata = outer.get_handshake_metadata()

        endpoint = metadata["cache_reset_endpoint"]
        assert endpoint["host"] == "127.0.0.1"
        assert endpoint["domain"] == "engine"
        assert reset_module.reset_cache_endpoints([endpoint]) is True
        assert inner.resets == 1
        # The endpoint is stable for a repeated handshake.
        assert outer.get_handshake_metadata()["cache_reset_endpoint"] == endpoint
    finally:
        outer.shutdown()


def test_worker_reset_callback_clears_fawa_fa_and_wa():
    fawa_reset_cache = _load_fawa_reset_method()

    class FawaInner:
        _is_pp = False
        unique_id = "engine"

        def __init__(self):
            self.fa = {b"block"}
            self.wa = {b"block"}
            self.backend = {b"block"}
            self.fa_store = SimpleNamespace(reset_cache=self._reset_fa)
            self.wa_store = SimpleNamespace(reset_cache=self._reset_wa)
            self.window_group_ids = [1]

        def _reset_fa(self):
            self.fa.clear()
            return True

        def _reset_wa(self):
            self.wa.clear()
            return True

        def get_handshake_metadata(self):
            return None

    # Use the production FAWA override, not a copy of its logic.
    FawaInner.reset_cache = fawa_reset_cache
    inner = FawaInner()
    outer, _ = _make_outer(role=KVConnectorRole.WORKER, inner=inner)
    try:
        endpoint = outer.get_handshake_metadata()["cache_reset_endpoint"]

        assert reset_module.reset_cache_endpoints([endpoint]) is True
        assert inner.fa == set()
        assert inner.wa == set()
        assert inner.backend == {b"block"}
    finally:
        outer.shutdown()


def test_worker_keeps_pp_geometry_when_reset_server_creation_fails():
    inner = _PPInner()
    expected = inner.get_handshake_metadata()
    outer, namespace = _make_outer(
        role=KVConnectorRole.WORKER, pp=2, tp=1, world_size=2, inner=inner
    )

    def fail_server(reset, host=None):
        raise OSError("cannot bind reset port")

    namespace["CacheResetServer"] = fail_server
    metadata = outer.get_handshake_metadata()

    assert metadata["block_size"] == expected["block_size"]
    assert metadata["cache_reset_host"] == "127.0.0.1"
    assert metadata["cache_reset_domain"] == "engine"
    assert "cache_reset_endpoint" not in metadata
    assert outer._cache_reset_server is None
    assert any(
        "reset server unavailable" in item for item in namespace["logger"].warnings
    )


def test_worker_without_pp_or_reset_server_publishes_only_topology():
    outer, namespace = _make_outer(role=KVConnectorRole.WORKER, inner=_DirectInner())
    namespace["CacheResetServer"] = lambda reset, host=None: (_ for _ in ()).throw(
        OSError("cannot bind reset port")
    )

    metadata = outer.get_handshake_metadata()

    assert metadata["cache_reset_host"] == "127.0.0.1"
    assert metadata["cache_reset_domain"] == "engine"
    assert "cache_reset_endpoint" not in metadata


def test_scheduler_does_not_publish_reset_endpoint():
    scheduler, _ = _make_outer(role=KVConnectorRole.SCHEDULER, inner=_PPInner())

    assert scheduler.get_handshake_metadata() is None


# --------------------------------------------------------------------------
# Scheduler dispatch
# --------------------------------------------------------------------------


def test_single_node_tp_reset_stays_local_without_tcp():
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        pp=1,
        tp=2,
        world_size=2,
        local_world_size=2,
        inner=inner,
    )
    remote_reset = MagicMock(return_value=True)
    with ExitStack() as stack:
        server = reset_module.CacheResetServer(remote_reset, host="127.0.0.1")
        stack.callback(server.close)
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata(
                {
                    (0, 0): _endpoint(server, "engine"),
                    (0, 1): _endpoint(server, "engine"),
                }
            )
        )

        assert outer.reset_cache() is True
        local_reset.assert_called_once_with()
        remote_reset.assert_not_called()


def test_pp_reset_reaches_every_stage_even_on_one_node():
    inner = _PPInner()
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        pp=2,
        tp=1,
        world_size=2,
        local_world_size=2,
        inner=inner,
    )
    calls = [MagicMock(return_value=True), MagicMock(return_value=True)]
    with ExitStack() as stack:
        servers = [
            reset_module.CacheResetServer(calls[stage], host="127.0.0.1")
            for stage in range(2)
        ]
        for server in servers:
            stack.callback(server.close)
        metadata = _metadata(
            {
                (0, 0): _endpoint(servers[0], "engine"),
                (1, 0): _endpoint(servers[1], "engine"),
            }
        )
        outer.set_xfer_handshake_metadata_pp_aware(metadata)

        assert outer.reset_cache() is True
        assert calls[0].call_count == 1
        assert calls[1].call_count == 1
        assert inner.handshake_calls == [metadata]


def test_tp_workers_sharing_a_domain_are_reset_once():
    remote_reset = MagicMock(return_value=True)
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="scheduler", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=1,
        scheduler_ip="10.0.0.1",
        inner=inner,
    )
    with ExitStack() as stack:
        server = reset_module.CacheResetServer(remote_reset, host="127.0.0.1")
        stack.callback(server.close)
        endpoint = _endpoint(server, "shared-domain")
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata({(0, 0): endpoint, (0, 1): dict(endpoint)})
        )

        assert outer.reset_cache() is True
        remote_reset.assert_called_once_with()
        local_reset.assert_not_called()


def test_scheduler_domain_mismatch_uses_remote_reset():
    """Same host is not enough: the scheduler must own the same Cache domain."""
    local_reset = MagicMock(return_value=True)
    remote_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="scheduler-domain", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=1,
        world_size=1,
        local_world_size=1,
        inner=inner,
    )
    with ExitStack() as stack:
        server = reset_module.CacheResetServer(remote_reset, host="127.0.0.1")
        stack.callback(server.close)
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata({(0, 0): _endpoint(server, "worker-domain")})
        )

        assert outer._cache_reset_plan["local"] is False
        assert outer.reset_cache() is True
        remote_reset.assert_called_once_with()
        local_reset.assert_not_called()


def test_partial_handshake_never_resets_only_some_domains():
    remote_reset = MagicMock(return_value=True)
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="scheduler", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=1,
        inner=inner,
    )
    with ExitStack() as stack:
        server = reset_module.CacheResetServer(remote_reset, host="127.0.0.1")
        stack.callback(server.close)
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata({(0, 0): _endpoint(server, "engine")})
        )

        assert outer._cache_reset_plan["complete"] is False
        assert outer.reset_cache() is False
        remote_reset.assert_not_called()
        local_reset.assert_not_called()


@pytest.mark.parametrize(
    "bad_endpoint",
    [
        None,
        {"host": "", "port": 1234, "token": "00"},
        {"host": "127.0.0.1", "port": 0, "token": "00"},
        {"host": "127.0.0.1", "port": "1234", "token": "00"},
        {"host": "127.0.0.1", "port": 1234, "token": "not-hex"},
    ],
)
def test_missing_or_damaged_endpoint_disables_remote_reset(bad_endpoint):
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="scheduler", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=1,
        world_size=2,
        local_world_size=1,
        inner=inner,
    )
    outer.set_xfer_handshake_metadata_pp_aware(_metadata({(0, 0): bad_endpoint}))

    assert outer._cache_reset_plan["complete"] is False
    assert outer.reset_cache() is False
    local_reset.assert_not_called()


def test_malformed_metadata_does_not_break_startup():
    inner = _PPInner()
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        pp=2,
        tp=1,
        world_size=2,
        inner=inner,
    )

    outer.set_xfer_handshake_metadata_pp_aware(
        {(0, 0): {"block_size": 1}, (1, 0): {"block_size": 1}}
    )

    assert inner.handshake_calls
    assert outer._cache_reset_plan["complete"] is False
    assert outer.reset_cache() is False


def test_ray_geometry_with_multiple_worker_hosts_uses_remote_reset():
    """Ray may report one node while the handshake proves several hosts."""
    local_reset = MagicMock(return_value=True)
    calls = [MagicMock(return_value=True), MagicMock(return_value=True)]
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=2,
        nnodes=1,
        backend="ray",
        scheduler_ip="10.0.0.1",
        inner=inner,
    )
    with ExitStack() as stack:
        servers = [
            reset_module.CacheResetServer(calls[rank], host="127.0.0.1")
            for rank in range(2)
        ]
        for server in servers:
            stack.callback(server.close)
        metadata = _metadata_with_hosts(
            {
                (0, 0): _endpoint(servers[0], "engine"),
                (0, 1): _endpoint(servers[1], "engine"),
            },
            {(0, 0): "10.0.0.2", (0, 1): "10.0.0.3"},
        )
        outer.set_xfer_handshake_metadata_pp_aware(metadata)

        assert outer._cache_reset_plan["complete"] is True
        assert outer._cache_reset_plan["local"] is False
        assert outer.reset_cache() is True
        assert [call.call_count for call in calls] == [1, 1]
        local_reset.assert_not_called()


def test_workers_on_one_remote_host_still_use_remote_reset():
    local_reset = MagicMock(return_value=True)
    remote_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=2,
        scheduler_ip="10.0.0.1",
        inner=inner,
    )
    with ExitStack() as stack:
        server = reset_module.CacheResetServer(remote_reset, host="127.0.0.1")
        stack.callback(server.close)
        endpoint = _endpoint(server, "engine")
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata_with_hosts(
                {(0, 0): endpoint, (0, 1): dict(endpoint)},
                {(0, 0): "10.0.0.9", (0, 1): "10.0.0.9"},
            )
        )

        assert outer._cache_reset_plan["local"] is False
        assert outer.reset_cache() is True
        remote_reset.assert_called_once_with()
        local_reset.assert_not_called()


def test_single_node_geometry_with_missing_worker_fails_closed():
    local_reset = MagicMock(return_value=True)
    remote_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=2,
        nnodes=1,
        inner=inner,
    )
    with ExitStack() as stack:
        server = reset_module.CacheResetServer(remote_reset, host="127.0.0.1")
        stack.callback(server.close)
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata({(0, 0): _endpoint(server, "engine")})
        )

        assert outer._cache_reset_plan["complete"] is False
        assert outer.reset_cache() is False
        remote_reset.assert_not_called()
        local_reset.assert_not_called()


def test_pcp_with_dcp_disabled_rejects_reset_without_affecting_startup():
    local_reset = MagicMock(return_value=True)
    remote_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=4,
        local_world_size=2,
        pcp=2,
        dcp=1,
        scheduler_ip="10.0.0.1",
        inner=inner,
    )
    with ExitStack() as stack:
        server = reset_module.CacheResetServer(remote_reset, host="127.0.0.1")
        stack.callback(server.close)
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata({(0, 0): _endpoint(server, "engine")})
        )

        assert outer._cache_reset_supported() is False
        assert outer._cache_reset_plan["complete"] is False
        assert outer.reset_cache() is False
        remote_reset.assert_not_called()
        local_reset.assert_not_called()


def test_pp_geometry_errors_still_fail_startup():
    class RaisingInner(_PPInner):
        def set_xfer_handshake_metadata_pp_aware(self, metadata):
            raise ValueError("PP stage 1 handshake metadata has no layer allocation")

    inner = RaisingInner()
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        pp=2,
        tp=1,
        world_size=2,
        inner=inner,
    )

    with pytest.raises(ValueError, match="no layer allocation"):
        outer.set_xfer_handshake_metadata_pp_aware(
            {(0, 0): {"block_size": 1}, (1, 0): {"block_size": 1}}
        )


def test_connection_failure_reports_failure_and_continues():
    closed = reset_module.CacheResetServer(lambda: True, host="127.0.0.1")
    closed.close()
    live_reset = MagicMock(return_value=True)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=1,
        inner=SimpleNamespace(unique_id="scheduler", reset_cache=MagicMock()),
    )
    with ExitStack() as stack:
        live = reset_module.CacheResetServer(live_reset, host="127.0.0.1")
        stack.callback(live.close)
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata(
                {
                    (0, 0): _endpoint(closed, "closed-domain"),
                    (0, 1): _endpoint(live, "live-domain"),
                }
            )
        )

        assert outer.reset_cache() is False
        live_reset.assert_called_once_with()


def test_cleanup_failure_reports_failure():
    def fail_reset():
        raise RuntimeError("native reset failed")

    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=1,
        world_size=2,
        local_world_size=1,
        inner=SimpleNamespace(unique_id="scheduler", reset_cache=MagicMock()),
    )
    with ExitStack() as stack:
        server = reset_module.CacheResetServer(fail_reset, host="127.0.0.1")
        stack.callback(server.close)
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata({(0, 0): _endpoint(server, "engine")})
        )

        assert outer.reset_cache() is False


def test_timeout_reports_failure_and_continues():
    release = threading.Event()
    slow_reset = MagicMock(side_effect=lambda: release.wait(timeout=3))
    live_reset = MagicMock(return_value=True)
    outer, namespace = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=1,
        inner=SimpleNamespace(unique_id="scheduler", reset_cache=MagicMock()),
    )
    with ExitStack() as stack:
        slow = reset_module.CacheResetServer(slow_reset, host="127.0.0.1")
        stack.callback(slow.close)
        live = reset_module.CacheResetServer(live_reset, host="127.0.0.1")
        stack.callback(live.close)
        real_reset_endpoints = namespace["reset_cache_endpoints"]
        namespace["reset_cache_endpoints"] = lambda endpoints: real_reset_endpoints(
            endpoints, timeout=0.05
        )
        outer.set_xfer_handshake_metadata_pp_aware(
            _metadata(
                {
                    (0, 0): _endpoint(slow, "slow-domain"),
                    (0, 1): _endpoint(live, "live-domain"),
                }
            )
        )

        try:
            assert outer.reset_cache() is False
        finally:
            release.set()
        live_reset.assert_called_once_with()


# --------------------------------------------------------------------------
# Older vLLM without the handshake
# --------------------------------------------------------------------------


def test_old_vllm_single_node_falls_back_to_local_reset():
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=2,
        inner=inner,
    )

    assert outer.reset_cache() is True
    local_reset.assert_called_once_with()


def test_old_vllm_cross_node_without_handshake_refuses():
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=1,
        inner=inner,
    )

    assert outer.reset_cache() is False
    local_reset.assert_not_called()


def test_pp_without_handshake_refuses_even_on_one_node():
    inner = _PPInner()
    outer, _ = _make_outer(pp=2, tp=1, world_size=2, inner=inner)

    assert outer.reset_cache() is False
    assert inner.resets == 0


def test_old_vllm_single_node_multi_dp_uses_local_reset_not_global_nnodes():
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        nnodes=2,
        dp=4,
        dp_local=2,
        expose_local_size=False,
        inner=inner,
    )

    assert outer._all_workers_on_this_node() is True
    assert outer.reset_cache() is True
    local_reset.assert_called_once_with()


def test_old_vllm_cross_node_multi_dp_refuses():
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        nnodes=4,
        dp=2,
        dp_local=1,
        expose_local_size=False,
        inner=inner,
    )

    assert outer._all_workers_on_this_node() is False
    assert outer.reset_cache() is False
    local_reset.assert_not_called()


def test_old_vllm_ray_topology_without_handshake_fails_closed():
    local_reset = MagicMock(return_value=True)
    inner = SimpleNamespace(unique_id="engine", reset_cache=local_reset)
    outer, _ = _make_outer(
        role=KVConnectorRole.SCHEDULER,
        tp=2,
        world_size=2,
        local_world_size=2,
        nnodes=1,
        backend="ray",
        inner=inner,
    )

    assert outer._all_workers_on_this_node() is False
    assert outer.reset_cache() is False
    local_reset.assert_not_called()


def test_cache_reset_server_thread_start_failure_releases_socket(monkeypatch):
    created = []

    class FakeTCPServer:
        def __init__(self, address, handler):
            self.server_address = ("127.0.0.1", 23456)
            self.closed = False
            created.append(self)

        def serve_forever(self, **kwargs):
            raise AssertionError("server thread must not run")

        def server_close(self):
            self.closed = True

        def shutdown(self):
            pass

    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(reset_module.socketserver, "TCPServer", FakeTCPServer)
    monkeypatch.setattr(reset_module.threading, "Thread", FailingThread)

    with pytest.raises(RuntimeError, match="thread start failed"):
        reset_module.CacheResetServer(lambda: True, host="127.0.0.1")

    assert created and created[0].closed is True


# --------------------------------------------------------------------------
# Two independent worker processes
# --------------------------------------------------------------------------


def _send_json(conn, payload):
    conn.sendall(json.dumps(payload).encode("utf-8") + b"\n")


def _send_command(conn, command):
    conn.sendall(command.encode("utf-8") + b"\n")


def _cache_reset_worker(control_address):
    """Worker process publishing its endpoint through the outer connector."""
    state = {"dram": {b"block"}, "backend": {b"block"}}

    def reset():
        state["dram"].clear()
        return True

    inner = SimpleNamespace(
        unique_id=f"worker-{os.getpid()}",
        get_handshake_metadata=lambda: None,
        reset_cache=reset,
    )
    cls, _ = _load_outer_class()
    outer = cls.__new__(cls)
    outer._role = KVConnectorRole.WORKER
    outer._vllm_config = SimpleNamespace(
        parallel_config=_parallel_config(
            pp=1,
            tp=1,
            world_size=1,
            local_world_size=1,
            nnodes=1,
            dp=1,
            dp_local=1,
            expose_local_size=True,
        )
    )
    outer.connector = inner
    outer._cache_reset_server = None
    outer._cache_reset_plan = None
    conn = socket.create_connection(control_address, timeout=20)
    reader = conn.makefile("rb")
    try:
        metadata = outer.get_handshake_metadata()
        _send_json(conn, metadata)
        while True:
            line = reader.readline()
            if not line:
                break
            command = line.decode("utf-8").strip()
            if command == "inspect":
                _send_json(
                    conn,
                    {
                        "dram": sorted(item.decode() for item in state["dram"]),
                        "backend": sorted(item.decode() for item in state["backend"]),
                    },
                )
            elif command == "load":
                if b"block" in state["dram"]:
                    _send_json(conn, "cache")
                elif b"block" in state["backend"]:
                    _send_json(conn, "backend")
                else:
                    _send_json(conn, "miss")
            else:
                break
    finally:
        reader.close()
        conn.close()
        outer.shutdown()


def test_tp_across_two_worker_processes_resets_cache_but_keeps_backend():
    ctx = multiprocessing.get_context("spawn")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    workers = []
    try:
        metadata = {}
        for tp_rank in range(2):
            process = ctx.Process(
                target=_cache_reset_worker, args=(listener.getsockname(),)
            )
            process.start()
            conn, _ = listener.accept()
            conn.settimeout(20)
            reader = conn.makefile("rb")
            workers.append((process, conn, reader))
            metadata[(0, tp_rank)] = json.loads(reader.readline())

        outer, _ = _make_outer(
            role=KVConnectorRole.SCHEDULER,
            tp=2,
            world_size=2,
            local_world_size=1,
            scheduler_ip="127.0.0.2",
            inner=SimpleNamespace(unique_id="scheduler", reset_cache=MagicMock()),
        )
        outer.set_xfer_handshake_metadata_pp_aware(metadata)

        assert outer._cache_reset_plan["complete"] is True
        assert outer._cache_reset_plan["local"] is False
        assert outer.reset_cache() is True
        for process, conn, reader in workers:
            _send_command(conn, "inspect")
            response = reader.readline()
            assert response, f"worker {process.pid} exited {process.exitcode}"
            assert json.loads(response) == {
                "dram": [],
                "backend": ["block"],
            }
            _send_command(conn, "load")
            assert json.loads(reader.readline()) == "backend"
    finally:
        listener.close()
        for process, conn, reader in workers:
            if process.is_alive():
                try:
                    _send_command(conn, "close")
                except OSError:
                    pass
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            reader.close()
            conn.close()
