"""CPU-only checks for the vLLM MultiConnector handshake backport."""

import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def patch_module(monkeypatch):
    utils = ModuleType("ucm.integration.vllm.patch.utils")
    utils.when_imported = MagicMock(side_effect=lambda _: lambda func: func)
    logger = ModuleType("ucm.logger")
    logger.init_logger = MagicMock()
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, logger.__name__, logger)
    path = (
        Path(__file__).resolve().parents[3]
        / "ucm/integration/vllm/patch/v0300/multi_connector_handshake_patch.py"
    )
    spec = importlib.util.spec_from_file_location("ucm_handshake_patch_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    utils.when_imported.assert_called_once_with(
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector"
    )
    return module


@pytest.fixture
def vllm_module(monkeypatch):
    class MultiConnector:
        get_handshake_metadata = MagicMock()
        set_xfer_handshake_metadata = MagicMock()
        set_xfer_handshake_metadata_pp_aware = MagicMock()

    module = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.multi_connector")
    module_parts = module.__name__.split(".")
    for depth in range(1, len(module_parts)):
        package = ModuleType(".".join(module_parts[:depth]))
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package.__name__, package)
    module.MultiConnector = MultiConnector
    module.KVConnectorHandshakeMetadata = type("KVConnectorHandshakeMetadata", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def multi_connector_cls(patch_module, vllm_module):
    patch_module.patch_multi_connector_handshake(vllm_module)
    return vllm_module.MultiConnector


@pytest.mark.parametrize(
    "setter, rank_keys",
    [
        ("set_xfer_handshake_metadata", (0, 1)),
        ("set_xfer_handshake_metadata_pp_aware", ((0, 0), (1, 0))),
        ("set_xfer_handshake_metadata_pp_aware", ((0, 0, 0), (1, 0, 0))),
    ],
)
def test_dispatches_each_workers_metadata_to_the_matching_child(
    multi_connector_cls, setter, rank_keys
):
    class AscendMultiConnector(multi_connector_cls):
        pass

    mooncake, observer, ucm = [MagicMock() for _ in range(3)]
    observer.get_handshake_metadata.return_value = None
    connector = AscendMultiConnector()
    connector._connectors = [mooncake, observer, ucm]
    mooncake_metadata = {}
    ucm_metadata = {}
    gathered = {}
    for stage, rank in enumerate(rank_keys):
        mooncake_metadata[rank] = SimpleNamespace(handshake_port=30000 + stage * 8)
        ucm_metadata[rank] = {
            "block_size": 128,
            "shard_size": 64,
            "layer_start": 42 * stage,
            "layer_count": (42, 36)[stage],
        }
        mooncake.get_handshake_metadata.return_value = mooncake_metadata[rank]
        ucm.get_handshake_metadata.return_value = ucm_metadata[rank]
        gathered[rank] = connector.get_handshake_metadata()

    getattr(connector, setter)(pickle.loads(pickle.dumps(gathered)))

    getattr(mooncake, setter).assert_called_once_with(mooncake_metadata)
    getattr(ucm, setter).assert_called_once_with(ucm_metadata)
    getattr(observer, setter).assert_called_once_with({})


def test_empty_and_legacy_worker_handshake(multi_connector_cls):
    child = MagicMock()
    child.get_handshake_metadata.return_value = None
    connector = multi_connector_cls()
    connector._connectors = [child]

    assert connector.get_handshake_metadata() is None
    legacy_metadata = {(0, 0): object()}
    connector.set_xfer_handshake_metadata_pp_aware(legacy_metadata)
    assert (
        child.set_xfer_handshake_metadata_pp_aware.call_args.args[0] is legacy_metadata
    )


def test_skips_vllm_when_the_upstream_fix_is_present(patch_module, vllm_module):
    cls = vllm_module.MultiConnector
    cls._sub_connector_handshake_metadata = MagicMock()
    native_metadata = type("MultiKVConnectorHandshakeMetadata", (), {})
    vllm_module.MultiKVConnectorHandshakeMetadata = native_metadata
    original_get = cls.get_handshake_metadata
    original_set = cls.set_xfer_handshake_metadata
    original_pp_set = cls.set_xfer_handshake_metadata_pp_aware

    patch_module.patch_multi_connector_handshake(vllm_module)

    assert vllm_module.MultiKVConnectorHandshakeMetadata is native_metadata
    assert cls.get_handshake_metadata is original_get
    assert cls.set_xfer_handshake_metadata is original_set
    assert cls.set_xfer_handshake_metadata_pp_aware is original_pp_set
