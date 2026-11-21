"""CPU-only checks for UCM's Ascend MultiConnector handshake patch."""

import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def multi_connector_cls(monkeypatch):
    class MultiConnector:
        def get_handshake_metadata(self):
            return self.original_get()

        def set_xfer_handshake_metadata_pp_aware(self, metadata):
            self.original_set(metadata)

    utils = ModuleType("ucm.integration.vllm.patch.utils")
    utils.when_imported = MagicMock(side_effect=lambda _: lambda func: func)
    logger = ModuleType("ucm.logger")
    logger.init_logger = MagicMock()
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, logger.__name__, logger)
    path = (
        Path(__file__).resolve().parents[3]
        / "ucm/integration/vllm/patch/v0300/vllm_ascend/multi_connector_pp_handshake_patch.py"
    )
    spec = importlib.util.spec_from_file_location("ucm_pp_handshake_patch_test", path)
    patch_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, patch_module)
    spec.loader.exec_module(patch_module)
    utils.when_imported.assert_called_once_with(
        "vllm_ascend.distributed.kv_transfer.ascend_multi_connector"
    )

    module = SimpleNamespace(AscendMultiConnector=MultiConnector)
    patch_module.patch_multi_connector_pp_handshake(module)
    patched_get = MultiConnector.get_handshake_metadata
    patch_module.patch_multi_connector_pp_handshake(module)
    assert MultiConnector.get_handshake_metadata is patched_get
    return MultiConnector


def _connector(cls, pp_size, children, *, ucm=True):
    connector = cls()
    connector._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        kv_transfer_config=SimpleNamespace(
            has_connector=lambda name: ucm and name == "UCMConnector"
        ),
    )
    connector._connectors = children
    connector.original_get = MagicMock(return_value="original")
    connector.original_set = MagicMock()
    return connector


@pytest.mark.parametrize("rank_keys", [((0, 0), (1, 0)), ((0, 0, 0), (1, 0, 0))])
def test_pp_dispatches_each_workers_metadata_to_the_matching_child(
    multi_connector_cls, rank_keys
):
    mooncake, ucm, observer = [MagicMock() for _ in range(3)]
    observer.get_handshake_metadata.return_value = None
    connector = _connector(multi_connector_cls, 2, [mooncake, ucm, observer])
    mooncake_metadata = {}
    ucm_metadata = {}
    gathered = {}
    for stage, ranks in enumerate(rank_keys):
        mooncake_metadata[ranks] = SimpleNamespace(handshake_port=30000 + stage * 8)
        ucm_metadata[ranks] = {
            "block_size": 128,
            "shard_size": 64,
            "layer_start": 42 * stage,
            "layer_count": (42, 36)[stage],
        }
        mooncake.get_handshake_metadata.return_value = mooncake_metadata[ranks]
        ucm.get_handshake_metadata.return_value = ucm_metadata[ranks]
        gathered[ranks] = connector.get_handshake_metadata()

    connector.set_xfer_handshake_metadata_pp_aware(pickle.loads(pickle.dumps(gathered)))

    mooncake.set_xfer_handshake_metadata_pp_aware.assert_called_once_with(
        mooncake_metadata
    )
    ucm.set_xfer_handshake_metadata_pp_aware.assert_called_once_with(ucm_metadata)
    observer.set_xfer_handshake_metadata_pp_aware.assert_not_called()


def test_empty_and_legacy_worker_handshake(multi_connector_cls):
    child = MagicMock()
    child.get_handshake_metadata.return_value = None
    connector = _connector(multi_connector_cls, 2, [child])

    assert connector.get_handshake_metadata() is None
    with pytest.raises(ValueError, match="same UCM patch"):
        connector.set_xfer_handshake_metadata_pp_aware({(0, 0): object()})
    child.set_xfer_handshake_metadata_pp_aware.assert_not_called()


@pytest.mark.parametrize("pp_size,ucm", [(1, True), (2, False)])
def test_unrelated_connectors_keep_the_original_handshake(
    multi_connector_cls, pp_size, ucm
):
    child = MagicMock()
    connector = _connector(multi_connector_cls, pp_size, [child], ucm=ucm)
    metadata = {(0, 0): object()}

    assert connector.get_handshake_metadata() == "original"
    connector.set_xfer_handshake_metadata_pp_aware(metadata)

    child.get_handshake_metadata.assert_not_called()
    child.set_xfer_handshake_metadata_pp_aware.assert_not_called()
    connector.original_set.assert_called_once_with(metadata)
