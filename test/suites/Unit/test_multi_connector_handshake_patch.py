import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def handshake_patch(monkeypatch):
    base_name = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    base = ModuleType(base_name)
    base.KVConnectorHandshakeMetadata = type("KVConnectorHandshakeMetadata", (), {})
    monkeypatch.setitem(sys.modules, base_name, base)
    path = (
        Path(__file__).parents[3]
        / "ucm/integration/vllm/patch/multi_connector_handshake_patch.py"
    )
    spec = importlib.util.spec_from_file_location("ucm_handshake_patch_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("rank", "setter"),
    [
        (0, "set_xfer_handshake_metadata"),
        ((0, 1), "set_xfer_handshake_metadata_pp_aware"),
        ((0, 1, 2), "set_xfer_handshake_metadata_pp_aware"),
    ],
)
def test_routes_handshakes_to_their_own_children(handshake_patch, rank, setter):
    class MultiConnector:
        def set_xfer_handshake_metadata_pp_aware(self, metadata):
            raise AssertionError("expected this method to be patched")

    class AscendMultiConnector(MultiConnector):
        pass

    handshake_patch.patch_handshake(MultiConnector)
    worker = MultiConnector()
    first = Mock()
    first.get_handshake_metadata.return_value = "pd"
    second = Mock()
    second.get_handshake_metadata.return_value = "ucm"
    silent = Mock()
    silent.get_handshake_metadata.return_value = None
    worker._connectors = [first, silent, second]
    scheduler = AscendMultiConnector()
    scheduler._connectors = [Mock(), Mock(), Mock()]

    report = pickle.loads(pickle.dumps(worker.get_handshake_metadata()))
    getattr(scheduler, setter)({rank: report})

    getattr(scheduler._connectors[0], setter).assert_called_once_with({rank: "pd"})
    getattr(scheduler._connectors[1], setter).assert_not_called()
    getattr(scheduler._connectors[2], setter).assert_called_once_with({rank: "ucm"})
    second.get_handshake_metadata.return_value = None
    first.get_handshake_metadata.return_value = None
    assert worker.get_handshake_metadata() is None


def test_merges_child_handshakes_from_pcp_replicas(handshake_patch):
    rank = (0, 0)
    make_handshake = handshake_patch.MultiConnectorHandshake
    worker_reports = [
        {rank: make_handshake({0: "pd rank 0", 1: "ucm representative"})},
        None,
        {rank: make_handshake({0: "pd rank 1"})},
    ]

    class Executor:
        def get_kv_connector_handshake_metadata(self):
            return worker_reports

    handshake_patch.patch_handshake_collection(Executor)
    executor = Executor()
    [merged] = executor.get_kv_connector_handshake_metadata()
    assert merged[rank].by_connector == {0: "pd rank 1", 1: "ucm representative"}

    worker_reports[:] = [{rank: "standalone connector"}]
    assert executor.get_kv_connector_handshake_metadata() is worker_reports
