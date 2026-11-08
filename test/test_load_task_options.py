"""CPU-only regression coverage of per-task backend load options."""

import array
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def methods_type(path, class_name, names, **namespace):
    source = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in source.body if getattr(node, "name", "") == class_name)
    methods = [node for node in cls.body if getattr(node, "name", "") in names]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *methods], type_ignores=[])
    )
    exec(compile(module, path, "exec"), namespace)
    cls = type(class_name, (), {name: namespace[name] for name in names})
    namespace[class_name] = cls
    return cls


@pytest.fixture
def pipeline():
    cls = methods_type(
        "ucm/store/pipeline/connector.py",
        "UcmPipelineStore",
        {"load", "load_data", "supports_distributed_backend_load"},
        array=array,
        np=np,
        UcmPipelineStoreTransTask=lambda task_id: SimpleNamespace(task_id=task_id),
    )
    store = cls()
    store.store_ = SimpleNamespace(Load=Mock(return_value=41))
    return store


@pytest.mark.parametrize("distribute", [False, True])
def test_rank_consistency_forwards_task_option_and_retains_store(pipeline, distribute):
    cls = methods_type(
        "ucm/integration/vllm/rank_consistency.py",
        "RankConsistencyManager",
        {"__init__", "submit_load", "wait_load"},
        Any=Any,
    )
    manager = cls(is_scheduler=False, use_consistency_manager=False)
    pipeline.wait = Mock()
    ptrs = np.array([[123]], dtype=np.uint64)
    key = b"k" * 16
    task = manager.submit_load(
        pipeline,
        {"req": [key]},
        [key],
        [0],
        ptrs,
        distribute_backend_load=distribute,
    )
    args = pipeline.store_.Load.call_args.args
    assert args[0].tobytes() == key
    assert list(args[1]) == [0]
    assert args[2] is ptrs
    assert args[3:] == (0, distribute)
    assert task.task_id == 41
    manager.wait_load(task)
    pipeline.wait.assert_called_once_with(task)
    assert not manager._load_task_contexts


@pytest.mark.parametrize("distribute", [False, True])
def test_rank_consistency_accepts_store_with_original_load_signature(distribute):
    cls = methods_type(
        "ucm/integration/vllm/rank_consistency.py",
        "RankConsistencyManager",
        {"__init__", "submit_load", "wait_load"},
        Any=Any,
    )
    manager = cls(is_scheduler=False, use_consistency_manager=False)
    task = object()
    calls = []

    def load_data(keys, shards, ptrs):
        calls.append((keys, shards, ptrs))
        return task

    store = SimpleNamespace(load_data=load_data, wait=Mock())
    ptrs = [[123]]
    result = manager.submit_load(
        store,
        {"req": [b"key"]},
        [b"key"],
        [0],
        ptrs,
        distribute_backend_load=distribute,
    )
    assert result is task
    assert calls == [([b"key"], [0], ptrs)]
    manager.wait_load(task)
    store.wait.assert_called_once_with(task)


@pytest.mark.parametrize("distribute", [False, True])
def test_custom_pipeline_override_keeps_original_load_signature(pipeline, distribute):
    cls = methods_type(
        "ucm/integration/vllm/rank_consistency.py",
        "RankConsistencyManager",
        {"__init__", "submit_load", "wait_load"},
        Any=Any,
    )
    manager = cls(is_scheduler=False, use_consistency_manager=False)
    task = object()

    class CustomPipeline(type(pipeline)):
        def load_data(self, keys, shards, ptrs):
            return task

    store = CustomPipeline()
    store.wait = Mock()
    assert not store.supports_distributed_backend_load
    assert (
        manager.submit_load(
            store,
            {"req": [b"key"]},
            [b"key"],
            [0],
            [[123]],
            distribute_backend_load=distribute,
        )
        is task
    )
    manager.wait_load(task)
    store.wait.assert_called_once_with(task)


@pytest.mark.parametrize("distribute", [False, True])
def test_pipeline_load_option_preserves_prerequisite_event(pipeline, distribute):
    pipeline.load_data(
        [b"k" * 16],
        [2],
        [[123]],
        99,
        distribute_backend_load=distribute,
    )
    args = pipeline.store_.Load.call_args.args
    assert args[2].tolist() == [[123]]
    assert args[3:] == (99, distribute)


def test_tensor_load_forwards_task_option(pipeline):
    ptrs = np.array([[123]], dtype=np.uint64)
    pipeline._tensor_normalize = lambda tensors: ptrs
    pipeline.load([b"k" * 16], [0], [[object()]], distribute_backend_load=True)
    assert pipeline.store_.Load.call_args.args[2] is ptrs
    assert pipeline.store_.Load.call_args.args[3:] == (0, True)


def test_load_defaults_to_local_backend_fill(pipeline):
    pipeline.load_data([b"k" * 16], [0], [[123]])
    assert pipeline.store_.Load.call_args.args[3:] == (0, False)
