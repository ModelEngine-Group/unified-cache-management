"""CPU-only checks for CUDA indexer wait scheduling; run with --noconftest."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest


@pytest.fixture
def methods():
    path = Path(__file__).resolve().parents[3] / "ucm/integration/vllm/ucm_connector.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if getattr(n, "name", "") == "UCMLayerWiseConnector"
    )
    init = next(n for n in cls.body if getattr(n, "name", "") == "__init__")
    gate = next(
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Attribute)
        and n.targets[0].attr == "_wait_ahead_for_indexer"
    )
    helper = next(
        n for n in tree.body if getattr(n, "name", "") == "_has_layer_component"
    )
    hooks = [
        n
        for n in cls.body
        if getattr(n, "name", "") in ("start_load_kv", "wait_for_layer_load")
    ]
    namespace = {"time": SimpleNamespace(perf_counter=lambda: 0)}
    module = ast.Module(body=[helper, *hooks], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace, compile(
        ast.Module(body=[gate], type_ignores=[]), str(path), "exec"
    )


@pytest.mark.parametrize(
    "cuda,name,expected",
    [
        (True, "model.layers.0.attn", False),
        (True, "model.layers.0.indexer.k_cache", True),
        (True, "model.layers.0.index_cache", True),
        (True, "model.layers.0.indexer_proj", False),
        (True, "model.layers.0.INDEXER.k_cache", True),
        (False, "model.layers.0.indexer.k_cache", False),
        (True, None, False),
    ],
)
@pytest.mark.parametrize("field", ["layer_names", "kv_cache_specs"])
def test_wait_schedule_only_changes_for_cuda_indexer(
    methods, cuda, name, expected, field
):
    namespace, gate = methods
    config = (
        None
        if name is None
        else SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    **{field: [name] if field == "layer_names" else {name: object()}}
                )
            ]
        )
    )
    metadata = SimpleNamespace(request_meta={})
    connector = SimpleNamespace(
        need_load=True,
        _connector_metadata=metadata,
        _get_connector_metadata=lambda: metadata,
        first_layer_id=4,
        layer_ids=[4, 5],
        layer_name_to_id={"attn4": 4, "attn5": 5},
        _submit_request_load_tasks_for_layer=Mock(),
        _wait_layer_and_prefetch_next=Mock(),
    )
    exec(
        gate,
        {
            **namespace,
            "self": connector,
            "kv_cache_config": config,
            "current_platform": SimpleNamespace(is_cuda=lambda: cuda),
        },
    )
    assert connector._wait_ahead_for_indexer is expected
    namespace["start_load_kv"](connector, None)
    assert connector._wait_layer_and_prefetch_next.call_args_list == (
        [call(4)] if expected else []
    )
    connector._wait_layer_and_prefetch_next.reset_mock()
    namespace["wait_for_layer_load"](connector, "attn4")
    assert connector._wait_layer_and_prefetch_next.call_args_list == (
        [call(4), call(5)] if expected else [call(4)]
    )
    connector._wait_layer_and_prefetch_next.reset_mock()
    namespace["wait_for_layer_load"](connector, "attn5")
    connector._wait_layer_and_prefetch_next.assert_called_once_with(5)
