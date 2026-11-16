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
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_cuda_indexer_io_config"
    )
    platform = SimpleNamespace(is_cuda=Mock(return_value=True))
    namespace = {"current_platform": platform, "logger": Mock()}
    exec(
        compile(
            ast.Module(
                body=[
                    next(
                        n
                        for n in tree.body
                        if getattr(n, "name", None) == "_has_layer_component"
                    ),
                    function,
                ],
                type_ignores=[],
            ),
            str(source),
            "exec",
        ),
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


@pytest.mark.parametrize("layout", ["new", "wrapped", "merged"])
@pytest.mark.parametrize("layerwise", [False, True])
def test_cuda_indexer_pp_rejected_before_config_mutation(policy, layout, layerwise):
    apply, _ = policy
    config = {"use_layerwise": layerwise, "use_request_async_load": False}
    before = config.copy()
    with pytest.raises(RuntimeError, match="pipeline_parallel_size=1"):
        apply(config, cache_config(layout), pipeline_parallel_size=2)
    assert config == before


@pytest.mark.parametrize(
    "cuda,name",
    [
        (False, "model.layers.0.indexer.k_cache"),
        (True, "model.layers.0.self_attn.attn"),
    ],
)
def test_other_pp_models_preserve_config(policy, cuda, name):
    apply, platform = policy
    platform.is_cuda.return_value = cuda
    config = {"use_layerwise": True}
    apply(config, cache_config("new", name), pipeline_parallel_size=2)
    assert config == {"use_layerwise": True}


@pytest.fixture
def connector_selection(policy, monkeypatch):
    import os
    import sys
    from types import ModuleType

    apply, platform = policy
    source = (
        Path(__file__).resolve().parents[3] / "ucm/integration/vllm/ucm_connector.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UCMConnector"
    )
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]

    class Base:
        def __init__(self, vllm_config, **kwargs):
            self._vllm_config = vllm_config

    direct, layerwise = Mock(), Mock()
    hla = ModuleType("ucm.integration.vllm.hla_connector")
    hla.UCMHybridLinearAttentionConnector = Mock()
    hla.UCMHybridLinearAttentionConnector.supports_kv_cache_layout.return_value = False
    hla.UCMHybridLinearAttentionLayerWiseConnector = Mock()
    hla.is_cuda_blnhc_layout = Mock(return_value=False)
    hma = ModuleType("ucm.integration.vllm.hma_connector")
    hma.UCMFAWAConnector = Mock()
    hma.UCMFAWAConnector.can_handle_kv_cache_config.return_value = False
    monkeypatch.setitem(sys.modules, hla.__name__, hla)
    monkeypatch.setitem(sys.modules, hma.__name__, hma)
    ns = {
        "KVConnectorBase_V1": Base,
        "SupportsHMA": type("SupportsHMA", (), {}),
        "KVConnectorRole": SimpleNamespace(SCHEDULER="scheduler", WORKER="worker"),
        "Config": lambda transfer: SimpleNamespace(get_config=lambda: transfer.config),
        "_apply_cuda_indexer_io_config": apply,
        "current_platform": platform,
        "os": os,
        "logger": Mock(),
        "get_world_group": lambda: SimpleNamespace(rank=0),
        "UCMDirectConnector": direct,
        "UCMLayerWiseConnector": layerwise,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    connector = ns["UCMConnector"]
    connector._setup_ucm_metrics = Mock()
    return connector, direct, layerwise


@pytest.mark.parametrize("role", ["scheduler", "worker"])
@pytest.mark.parametrize("layout", ["new", "wrapped"])
@pytest.mark.parametrize("pp,indexer", [(2, True), (1, True), (2, False)])
def test_connector_initialization_selects_pp_path(
    connector_selection, role, layout, pp, indexer
):
    connector, direct, layerwise = connector_selection
    config = {"use_layerwise": True, "use_request_async_load": False}
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(engine_id="test", config=config),
    )
    name = (
        "model.layers.0.self_attn.indexer.k_cache"
        if indexer
        else "model.layers.0.self_attn.attn"
    )
    cache = cache_config(layout, name)
    if pp == 2 and indexer:
        before = config.copy()
        with pytest.raises(RuntimeError, match="pipeline_parallel_size=1") as exc:
            connector(vllm_config, role, cache)
        assert "please set use_layerwise=True" not in str(exc.value)
        assert config == before
        direct.assert_not_called()
        layerwise.assert_not_called()
    else:
        instance = connector(vllm_config, role, cache)
        selected, other = (direct, layerwise) if indexer else (layerwise, direct)
        selected.assert_called_once_with(vllm_config, role, cache)
        other.assert_not_called()
        assert instance.connector is selected.return_value
        assert config == {
            "use_layerwise": not indexer,
            "use_request_async_load": indexer,
        }


@pytest.mark.parametrize(
    "group",
    [
        SimpleNamespace(),
        SimpleNamespace(layer_names=[], kv_cache_specs=None),
        SimpleNamespace(layer_names=None, kv_cache_specs={}),
    ],
)
def test_missing_layer_names_preserves_config(policy, group):
    apply, _ = policy
    config = {"use_layerwise": True}
    apply(config, SimpleNamespace(kv_cache_groups=[group]))
    assert config == {"use_layerwise": True}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.INDEXER.k_cache", True),
        ("model.layers.0.attn.INDEX_CACHE", True),
        ("model.layers.0.INDEXER_AUX.k_cache", False),
        ("model.layers.0.attn.INDEX_CACHE_AUX", False),
    ],
)
def test_empty_layer_names_falls_back_to_specs(policy, name, expected):
    apply, _ = policy
    group = SimpleNamespace(layer_names=[], kv_cache_specs={name: SimpleNamespace()})
    config = {"use_layerwise": True}
    apply(config, SimpleNamespace(kv_cache_groups=[group]))
    assert config["use_layerwise"] is (not expected)
