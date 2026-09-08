"""CPU tests for M3 hook ordering and graph configuration contracts."""

import ast
import importlib.util
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_ROOT = ROOT / "ucm/integration/vllm/patch"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    logger = Mock()
    stub_module(monkeypatch, "ucm.logger", init_logger=lambda _: logger)
    hooks = {}

    def when_imported(name):
        def register(fn):
            hooks[name] = fn
            return fn

        return register

    stub_module(
        monkeypatch, "ucm.integration.vllm.patch.utils", when_imported=when_imported
    )
    connector = Mock()
    connector.has_connector_metadata.return_value = True
    ctx = SimpleNamespace(attn_metadata={"layer.attn": object()})
    transfer = stub_module(
        monkeypatch,
        "vllm.distributed.kv_transfer",
        has_kv_transfer_group=Mock(return_value=True),
        is_v1_kv_transfer_group=Mock(return_value=True),
        get_kv_transfer_group=Mock(return_value=connector),
    )
    stub_module(monkeypatch, "vllm.forward_context", get_forward_context=lambda: ctx)
    module = load_module(
        PATCH_ROOT / "v0260/vllm_ascend/minimax_m3_kv_transfer_patch.py",
        "m3_kv_patch_test",
    )
    return SimpleNamespace(
        patch=module, connector=connector, ctx=ctx, transfer=transfer, hooks=hooks
    )


def make_layer(runtime, trace, layer_name="layer.attn"):
    class SparseAttention:
        def __init__(self):
            self.layer_name = layer_name
            self.kv_cache = (object(), object())
            self.indexer_cache = []

        def _run_sparse_attention(self, query, *, index_key):
            trace.append(("update_kv", query))
            self.indexer_cache.append(index_key)
            trace.append(("update_indexer", index_key))
            trace.append(("attention", self.layer_name))
            return query

    module = SimpleNamespace(MiniMaxM3SparseAttention=SparseAttention)
    runtime.patch.patch_minimax_m3_kv_hooks(module)
    return module, SparseAttention()


def test_sparse_hooks_wrap_both_cache_updates_and_preserve_arguments(runtime):
    trace = []
    module, layer = make_layer(runtime, trace)
    runtime.patch.patch_minimax_m3_kv_hooks(module)
    runtime.connector.wait_for_layer_load.side_effect = lambda name: trace.append(
        ("wait", name)
    )

    def save(name, cache, metadata):
        assert cache is layer.kv_cache
        assert metadata is runtime.ctx.attn_metadata[name]
        assert layer.indexer_cache == ["index"]
        trace.append(("save", name))

    runtime.connector.save_kv_layer.side_effect = save
    assert layer._run_sparse_attention("query", index_key="index") == "query"
    assert [event[0] for event in trace] == [
        "wait",
        "update_kv",
        "update_indexer",
        "attention",
        "save",
    ]
    runtime.connector.wait_for_layer_load.assert_called_once_with("layer.attn")
    runtime.connector.save_kv_layer.assert_called_once()


@pytest.mark.parametrize(
    "inactive", ["no_group", "not_v1", "no_connector_meta", "profile"]
)
def test_profile_and_no_transfer_do_not_call_hooks(runtime, inactive):
    trace = []
    if inactive == "no_group":
        runtime.transfer.has_kv_transfer_group.return_value = False
    elif inactive == "not_v1":
        runtime.transfer.is_v1_kv_transfer_group.return_value = False
    elif inactive == "no_connector_meta":
        runtime.connector.has_connector_metadata.return_value = False
    else:
        runtime.ctx.attn_metadata = None
    _, layer = make_layer(runtime, trace)
    layer._run_sparse_attention("q", index_key="i")
    assert len(trace) == 3
    runtime.connector.wait_for_layer_load.assert_not_called()
    runtime.connector.save_kv_layer.assert_not_called()


@pytest.mark.parametrize("failure", ["wait", "attention", "save"])
def test_hook_errors_propagate_without_saving_failed_attention(runtime, failure):
    class SparseAttention:
        layer_name = "layer.attn"
        kv_cache = (object(), object())

        def _run_sparse_attention(self):
            if failure == "attention":
                raise RuntimeError("attention failed")

    runtime.patch.patch_minimax_m3_kv_hooks(
        SimpleNamespace(MiniMaxM3SparseAttention=SparseAttention)
    )
    if failure == "wait":
        runtime.connector.wait_for_layer_load.side_effect = RuntimeError("wait failed")
    if failure == "save":
        runtime.connector.save_kv_layer.side_effect = RuntimeError("save failed")
    with pytest.raises(RuntimeError, match=failure):
        SparseAttention()._run_sparse_attention()
    if failure != "save":
        runtime.connector.save_kv_layer.assert_not_called()


@pytest.fixture
def connector_symbols(runtime, monkeypatch):
    config_module = load_module(ROOT / "ucm/utils.py", "m3_ucm_config_test")
    tree = ast.parse(
        (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(encoding="utf-8-sig")
    )
    selected = []
    methods = {
        "UCMConnector": {"requires_piecewise_for_cudagraph", "__init__"},
        "UCMDirectConnector": {
            "_generate_dispatch_meta",
            "_get_full_hit_recompute_tokens",
        },
        "UCMLayerWiseConnector": {"start_load_kv", "wait_for_layer_load"},
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "RequestMeta",
            "RequestDispatchMeta",
        }:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in methods:
            node.bases = [ast.Name(id="Base", ctx=ast.Load())]
            node.body = [
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name in methods[node.name]
            ]
            selected.append(node)

    class Base:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._vllm_config = vllm_config

        def _setup_ucm_metrics(self, *args):
            pass

    ns = dict(
        __name__=__name__,
        Base=Base,
        Config=config_module.Config,
        SimpleNamespace=SimpleNamespace,
        dataclass=dataclass,
        field=field,
        KVConnectorRole=SimpleNamespace(WORKER="worker", SCHEDULER="scheduler"),
        UCMLiteConnector=Mock(),
        logger=Mock(),
        time=time,
        ucmmetrics=Mock(),
    )
    monitor_tree = ast.parse(
        (
            ROOT / "ucm/integration/vllm/inference_duration_monitor_connector.py"
        ).read_text(encoding="utf-8-sig")
    )
    selected.append(
        next(
            node
            for node in monitor_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "inference_duration_monitor_enabled"
        )
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<ucm-connector-contracts>", "exec"),
        ns,
    )
    stub_module(
        monkeypatch,
        "ucm.integration.vllm.inference_duration_monitor_connector",
        inference_duration_monitor_enabled=ns["inference_duration_monitor_enabled"],
    )
    stub_module(
        monkeypatch,
        "ucm.integration.vllm.hla_connector",
        UCMHybridLinearAttentionConnector=SimpleNamespace(
            supports_kv_cache_layout=lambda _: False
        ),
        UCMHybridLinearAttentionLayerWiseConnector=Mock(),
    )
    stub_module(
        monkeypatch,
        "ucm.integration.vllm.hma_connector",
        UCMFAWAConnector=SimpleNamespace(can_handle_kv_cache_config=lambda _: False),
    )
    return SimpleNamespace(**ns)


def test_dense_to_sparse_load_chain_reaches_every_layer(runtime, connector_symbols):
    cls = connector_symbols.UCMLayerWiseConnector
    connector = cls.__new__(cls)
    names = [f"model.layers.{i}.self_attn.attn" for i in range(60)]
    metadata = SimpleNamespace(
        request_meta={"request": SimpleNamespace(load_block_ids=([b"block"], [1]))}
    )
    connector._connector_metadata = metadata
    connector._get_connector_metadata = lambda: metadata
    connector._dumped_layer_ids = set()
    connector.load_tasks = defaultdict(dict)
    connector.request_data = []
    connector._failure_req_ids = set()
    connector.tp_rank = 0
    connector.tp_size = 4
    connector.is_mla = False
    connector.first_layer_id = 0
    connector.layer_ids = list(range(60))
    connector.layer_name_to_id = dict(zip(names, range(60)))
    connector.kv_cache_layout = Mock()
    connector.kv_cache_layout.extract_block_addrs.return_value = np.zeros((60, 1, 3))
    loaded, waited, saved = [], [], []

    def submit(layer_id, row, meta):
        assert row == layer_id
        loaded.append(layer_id)
        connector.load_tasks[layer_id]["request"] = layer_id

    connector._submit_request_load_tasks_for_layer = submit
    connector._rank_consistency = SimpleNamespace(wait_load=waited.append)
    connector.has_connector_metadata = lambda: True
    connector.save_kv_layer = lambda name, *args: saved.append(
        connector.layer_name_to_id[name]
    )
    runtime.transfer.get_kv_transfer_group.return_value = connector
    runtime.ctx.attn_metadata = dict.fromkeys(names, object())
    connector.start_load_kv(runtime.ctx)
    for i, name in enumerate(names):
        if i < 3:
            connector.wait_for_layer_load(name)
            connector.save_kv_layer(name)
        else:
            _, layer = make_layer(runtime, [], name)
            layer._run_sparse_attention("q", index_key="i")
    assert loaded == waited == saved == list(range(60))
    assert not connector.load_tasks


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, False),
        ({"use_layerwise": False}, False),
        ({"use_layerwise": True}, False),
        ({"use_inference_duration_monitor": True}, True),
        ({"use_layerwise": True, "use_inference_duration_monitor": True}, True),
    ],
)
@pytest.mark.parametrize("yaml_backed", [False, True])
def test_graph_requirement_resolves_actual_ucm_config(
    connector_symbols, config, expected, yaml_backed
):
    import yaml

    extra = {"UCM_CONFIG_FILE": "ucm.yaml"} if yaml_backed else config
    with patch("builtins.open", mock_open(read_data=yaml.safe_dump(config))):
        assert (
            connector_symbols.UCMConnector.requires_piecewise_for_cudagraph(extra)
            is expected
        )


class GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL_DECODE_ONLY = 3


def make_config(mode, layerwise):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="minimax_m3_vl")
        ),
        kv_transfer_config=SimpleNamespace(
            engine_id="m3_dp0",
            kv_connector_extra_config={"use_layerwise": layerwise},
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=mode, splitting_ops=None),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        additional_config={"ascend_compilation_config": {"enable_static_kernel": True}},
    )


@pytest.mark.parametrize(
    "mode", [GraphMode.NONE, GraphMode.PIECEWISE, GraphMode.FULL_DECODE_ONLY]
)
@pytest.mark.parametrize("layerwise", [False, True])
def test_connector_preserves_requested_graph_mode(connector_symbols, mode, layerwise):
    config = make_config(mode, layerwise=layerwise)
    connector = connector_symbols.UCMConnector(config, "scheduler")
    expected_cls = (
        connector_symbols.UCMLayerWiseConnector
        if layerwise
        else connector_symbols.UCMDirectConnector
    )
    assert isinstance(connector.connector, expected_cls)
    assert config.compilation_config.cudagraph_mode is mode
    assert config.additional_config["ascend_compilation_config"]["enable_static_kernel"]


def test_yaml_graph_requirement_uses_same_precedence_as_connector(connector_symbols):
    extra = {"UCM_CONFIG_FILE": "ucm.yaml", "use_inference_duration_monitor": True}
    with patch("builtins.open", mock_open(read_data="use_layerwise: true\n")):
        assert not connector_symbols.UCMConnector.requires_piecewise_for_cudagraph(
            extra
        )


def test_prefill_saves_blocks_but_decode_does_not(connector_symbols):
    symbols = connector_symbols
    dispatch = symbols.UCMDirectConnector.__new__(symbols.UCMDirectConnector)
    dispatch.block_size = 64
    dispatch.cp_world_size = 1
    request = symbols.RequestMeta(
        ucm_block_ids=[b"block0", b"block1"], num_token_ids=128
    )
    for num_tokens, new_blocks, expected_dump in [
        (64, [10], [b"block0"]),
        (64, [11], [b"block1"]),
        (1, [12], []),
    ]:
        meta = dispatch._generate_dispatch_meta(
            request, num_tokens, new_blocks, need_load=False
        )
        assert meta.load_block_ids == ([], [])
        assert meta.dump_block_ids[0] == expected_dump


def test_resumed_request_generates_load_metadata(connector_symbols):
    symbols = connector_symbols
    dispatch = symbols.UCMDirectConnector.__new__(symbols.UCMDirectConnector)
    dispatch.block_size = 64
    dispatch.cp_world_size = 1
    request = symbols.RequestMeta(
        ucm_block_ids=[b"block0", b"block1"],
        num_token_ids=128,
        total_hit_block_num=1,
        token_processed=64,
    )
    meta = dispatch._generate_dispatch_meta(request, 1, [10, 11], need_load=True)
    assert meta.load_block_ids == ([b"block0"], [10])
    assert meta.dump_block_ids == ([], [])


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("spec_tokens", [None, 0, 3])
def test_full_hit_recompute_protection_is_preserved(
    connector_symbols, layerwise, spec_tokens
):
    cls = connector_symbols.UCMDirectConnector
    connector = cls.__new__(cls)
    connector.use_layerwise = layerwise
    connector._vllm_config = SimpleNamespace(
        speculative_config=(
            None
            if spec_tokens is None
            else SimpleNamespace(num_speculative_tokens=spec_tokens)
        )
    )
    expected = (spec_tokens or 0) + 2 if layerwise else 1
    assert connector._get_full_hit_recompute_tokens() == expected
    assert connector._get_full_hit_recompute_tokens() == expected


def test_patch_registers_only_sparse_attention_hook(runtime):
    assert set(runtime.hooks) == {
        "vllm_ascend.models.minimax_m3.minimax_m3",
    }
    tree = ast.parse((PATCH_ROOT / "apply_patch.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.match_case) and isinstance(
            node.pattern, ast.MatchValue
        ):
            if (
                isinstance(node.pattern.value, ast.Constant)
                and node.pattern.value.value == "0.26.0"
            ):
                assert any(
                    isinstance(stmt, ast.Import)
                    and any(
                        name.name.endswith("minimax_m3_kv_transfer_patch")
                        for name in stmt.names
                    )
                    for stmt in node.body
                )
                return
    pytest.fail("Missing vLLM-Ascend 0.26.0 patch route")
