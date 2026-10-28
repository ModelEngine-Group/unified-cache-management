"""CPU tests for FAWA Ascend block geometry conversion."""

import ast
import copy
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTOR_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "hma_connector.py"
DETECT_MODULE = "vllm_ascend.core.kv_cache_interface"


def _load_geometry_symbols():
    """Load only the geometry helpers, without importing vLLM or torch."""

    tree = ast.parse(CONNECTOR_PATH.read_text(encoding="utf-8-sig"))
    selected_nodes = [
        node
        for node in tree.body
        if getattr(node, "name", None)
        in {"BlockGeometry", "_ascend_block_size_is_logical"}
    ]
    connector_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UCMFAWAConnector"
    )
    # Keep only the method under test and drop the vLLM base classes.
    stripped_cls = copy.copy(connector_cls)
    stripped_cls.bases = []
    stripped_cls.keywords = []
    stripped_cls.body = [
        node
        for node in connector_cls.body
        if getattr(node, "name", None)
        in {
            "ascend_block_geometry",
            "ascend_compression_ratio",
            "_get_ascend_base_block_size",
            "_specs_for_group",
        }
        or (
            isinstance(node, ast.Assign)
            and any(
                getattr(target, "id", None)
                in {"ASCEND_SUPPORTED_VLLM_BLOCK_SIZES", "ASCEND_C4_COMPRESS_RATIO"}
                for target in node.targets
            )
        )
    ]
    selected_nodes.append(stripped_cls)
    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"dataclass": dataclass, "lru_cache": lru_cache, "logger": Mock()}
    exec(compile(module, str(CONNECTOR_PATH), "exec"), namespace)
    return namespace


SYMBOLS = _load_geometry_symbols()
BlockGeometry = SYMBOLS["BlockGeometry"]
detect_is_logical = SYMBOLS["_ascend_block_size_is_logical"]
geometry = SYMBOLS["UCMFAWAConnector"].ascend_block_geometry


def spec(block_size, compress_ratio=None):
    attrs = {"block_size": block_size}
    if compress_ratio is not None:
        attrs["compress_ratio"] = compress_ratio
    return SimpleNamespace(**attrs)


def nested(**members):
    return SimpleNamespace(block_size=0, kv_cache_specs=members)


# (name, physical-convention block_size, logical-convention block_size,
#  compress_ratio, expected geometry) for user block_size=128.
DSV4_GROUPS = [
    ("c4", 128, 512, 4, BlockGeometry(512, 128)),
    ("c128", 128, 16384, 128, BlockGeometry(16384, 128)),
    ("swa", 128, 128, None, BlockGeometry(128, 128)),
    ("c4_state", 8, 8, None, BlockGeometry(8, 8)),
]


@pytest.mark.parametrize("name,physical_bs,logical_bs,ratio,expected", DSV4_GROUPS)
def test_both_conventions_resolve_to_same_geometry(
    name, physical_bs, logical_bs, ratio, expected
):
    old = geometry(spec(physical_bs, ratio), block_size_is_logical=False)
    new = geometry(spec(logical_bs, ratio), block_size_is_logical=True)
    assert old == new == expected


@pytest.mark.parametrize("user_bs", [32, 64, 128])
def test_c4_physical_matches_user_block_size(user_bs):
    new = geometry(spec(user_bs * 4, 4), block_size_is_logical=True)
    old = geometry(spec(user_bs, 4), block_size_is_logical=False)
    assert new.physical == old.physical == user_bs


def test_nested_consistent_geometry():
    group = nested(
        **{
            "model.layers.2.attn": spec(512, 4),
            "model.layers.2.indexer": spec(512, 4),
        }
    )
    assert geometry(group, block_size_is_logical=True) == BlockGeometry(512, 128)


def test_runtime_tokens_per_state_specs_find_c4_base_block_size(detection):
    """v0.30 main uses tokens_per_state in the captured DSV4 worker config."""
    detection(True)
    c4_specs = {
        "model.layers.2.indexer": SimpleNamespace(
            block_size=512, tokens_per_state=4, sliding_window=None
        ),
        "model.layers.2.attn": SimpleNamespace(
            block_size=512, tokens_per_state=4, sliding_window=None
        ),
    }
    c4 = SimpleNamespace(
        layer_names=list(c4_specs),
        kv_cache_spec=SimpleNamespace(block_size=512, kv_cache_specs=c4_specs),
    )
    config = SimpleNamespace(kv_cache_groups=[c4])
    connector = SYMBOLS["UCMFAWAConnector"]
    assert connector._get_ascend_base_block_size(config) == 128
    assert geometry(c4.kv_cache_spec) == BlockGeometry(512, 128)


@pytest.mark.parametrize("ratio", [4, 128])
def test_tokens_per_state_geometry_matches_compress_ratio(ratio):
    runtime_spec = SimpleNamespace(block_size=128 * ratio, tokens_per_state=ratio)
    assert geometry(runtime_spec, block_size_is_logical=True) == BlockGeometry(
        128 * ratio, 128
    )


def test_nested_mixed_geometry_reports_layers():
    group = nested(
        **{
            "model.layers.2.attn": spec(512, 4),
            "model.layers.3.attn": spec(16384, 128),
        }
    )
    with pytest.raises(ValueError) as exc_info:
        geometry(group, block_size_is_logical=True)
    message = str(exc_info.value)
    assert "block_size_is_logical=True" in message
    assert "(logical=512, physical=128)" in message
    assert "(logical=16384, physical=128)" in message
    assert "model.layers.2.attn" in message
    assert "model.layers.3.attn" in message


@pytest.mark.parametrize("block_size,ratio", [(0, 4), (-128, 4), (128, 0), (128, -4)])
@pytest.mark.parametrize("is_logical", [False, True])
def test_non_positive_values_rejected(block_size, ratio, is_logical):
    with pytest.raises(ValueError, match="must be positive"):
        geometry(spec(block_size, ratio), block_size_is_logical=is_logical)


def test_logical_block_size_must_divide_ratio():
    with pytest.raises(ValueError, match="divisible by compress ratio"):
        geometry(spec(510, 4), block_size_is_logical=True)
    # The physical convention multiplies instead, so any positive size works.
    assert geometry(spec(510, 4), block_size_is_logical=False) == BlockGeometry(
        2040, 510
    )


@pytest.fixture
def detection(monkeypatch):
    """Control the vllm_ascend helper import seen by the convention probe."""

    detect_is_logical.cache_clear()

    def set_helper_available(available):
        for name in ("vllm_ascend", "vllm_ascend.core"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        if available:
            module = ModuleType(DETECT_MODULE)
            module.get_storage_block_size = lambda kv_cache_spec: 0
            monkeypatch.setitem(sys.modules, DETECT_MODULE, module)
        else:
            # A None entry makes the import raise ImportError.
            monkeypatch.setitem(sys.modules, DETECT_MODULE, None)

    yield set_helper_available
    detect_is_logical.cache_clear()


@pytest.mark.parametrize("available,expected", [(True, True), (False, False)])
def test_detection_follows_helper_presence(detection, available, expected):
    detection(available)
    assert detect_is_logical() is expected


def test_detection_selects_convention_when_not_given(detection):
    detection(True)
    assert geometry(spec(512, 4)) == BlockGeometry(512, 128)
    detect_is_logical.cache_clear()
    detection(False)
    assert geometry(spec(128, 4)) == BlockGeometry(512, 128)


def test_detection_result_is_cached(detection):
    detection(True)
    assert detect_is_logical() is True
    detection(False)
    assert detect_is_logical() is True


@pytest.mark.parametrize("helper_available", [True, False])
def test_explicit_flag_overrides_detection(detection, helper_available):
    """Partial cherry-picks pass the runner convention explicitly."""

    detection(helper_available)
    assert geometry(spec(128, 4), block_size_is_logical=False) == BlockGeometry(
        512, 128
    )
    assert geometry(spec(512, 4), block_size_is_logical=True) == BlockGeometry(512, 128)
