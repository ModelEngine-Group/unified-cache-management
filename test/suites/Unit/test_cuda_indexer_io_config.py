"""CPU-only IO policy checks, following the layout tests' AST loading approach."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def policy():
    source = (
        Path(__file__).resolve().parents[3] / "ucm/integration/vllm/ucm_connector.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_cuda_indexer_io_config"
    )
    platform = SimpleNamespace(is_cuda=Mock(return_value=True))
    namespace = {"current_platform": platform, "logger": Mock()}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[function.name], platform


def cache_config(layout, name="model.layers.0.self_attn.indexer.k_cache"):
    specs = {
        "model.layers.0.self_attn.attn": SimpleNamespace(),
        name: SimpleNamespace(),
    }
    if layout == "new":
        groups = [SimpleNamespace(kv_cache_specs=specs)]
    elif layout == "wrapped":
        groups = [
            SimpleNamespace(
                layer_names=list(specs),
                kv_cache_spec=SimpleNamespace(kv_cache_specs=specs),
            )
        ]
    else:
        groups = [
            SimpleNamespace(layer_names=[n], kv_cache_spec=s) for n, s in specs.items()
        ]
    return SimpleNamespace(kv_cache_groups=groups)


@pytest.mark.parametrize("layout", ["new", "wrapped", "merged"])
@pytest.mark.parametrize("layerwise", [None, False, True])
@pytest.mark.parametrize("async_load", [False, True])
def test_cuda_indexer_forces_whole_block_io(policy, layout, layerwise, async_load):
    apply, _ = policy
    config = {"use_request_async_load": async_load, "ucm_connectors": []}
    if layerwise is not None:
        config["use_layerwise"] = layerwise
    apply(config, cache_config(layout))
    assert config == {
        "use_layerwise": False,
        "use_request_async_load": True,
        "ucm_connectors": [],
    }


@pytest.mark.parametrize("cache", [None, SimpleNamespace(kv_cache_groups=[])])
def test_missing_cache_metadata_preserves_config(policy, cache):
    apply, _ = policy
    config = {}
    apply(config, cache)
    assert config == {}


def test_already_direct_is_unchanged(policy):
    apply, _ = policy
    config = {"use_layerwise": False, "use_request_async_load": True}
    apply(config, cache_config("new"))
    assert config == {"use_layerwise": False, "use_request_async_load": True}


@pytest.mark.parametrize("layout", ["new", "wrapped", "merged"])
@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.self_attn.indexer.k_cache", True),
        ("model.layers.0.self_attn.attn.index_cache", True),
        ("model.layers.0.self_attn.attn", False),
        ("model.layers.0.self_attn.indexer_aux.k_cache", False),
        ("model.layers.0.self_attn.attn.index_cache_aux", False),
    ],
)
def test_cache_names_across_layouts(policy, layout, cuda, name, expected):
    apply, platform = policy
    platform.is_cuda.return_value = cuda
    config = {"use_layerwise": True}
    apply(config, cache_config(layout, name))
    assert config["use_layerwise"] == (not (cuda and expected))
    if cuda and expected:
        assert config["use_request_async_load"] is True
    else:
        assert config == {"use_layerwise": True}


def test_layer_names_need_no_spec_metadata(policy):
    apply, _ = policy
    group = SimpleNamespace(layer_names=["model.layers.0.indexer.k_cache"])
    config = {"use_layerwise": True}
    apply(config, SimpleNamespace(kv_cache_groups=[group]))
    assert config["use_layerwise"] is False
