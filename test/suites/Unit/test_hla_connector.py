"""CPU-only checks for HLA decisions and transfer metadata.

Like test_kv_cache_layout.py, load production methods via AST to avoid
requiring torch, vLLM or NPU libraries. These tests do not emulate kernels
or establish the safety of every model-selection/state-copy configuration.
"""

import ast
import concurrent.futures
import math
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HLA_PATH = ROOT / "ucm/integration/vllm/hla_connector.py"


class KVCacheSpec(SimpleNamespace):
    pass


class FullAttentionSpec(KVCacheSpec):
    pass


class MLAAttentionSpec(FullAttentionSpec):
    def __init__(self, **kwargs):
        kwargs.setdefault("is_index_group_leader", False)
        kwargs.setdefault("tokens_per_state", 1)
        super().__init__(**kwargs)


class MambaSpec(KVCacheSpec):
    pass


class UniformTypeKVCacheSpecs(KVCacheSpec):
    pass


class FakeTensor:
    def __init__(self, ptr, shape, strides, element_size=2):
        self.ptr = ptr
        self.shape = shape
        self.strides = strides
        self.item_size = element_size

    def numel(self):
        return math.prod(self.shape)

    def dim(self):
        return len(self.shape)

    def stride(self, dim):
        return self.strides[dim]

    def element_size(self):
        return self.item_size

    def data_ptr(self):
        return self.ptr


@pytest.fixture
def hla():
    functions = {
        "_kv_cache_tensor_layers",
        "layer_name_to_kv_cache_spec",
        "block_size_from_kv_cache_spec",
        "is_mamba_align_kv_cache_spec",
        "participates_in_prefix_caching",
        "extend_non_null",
        "kv_cache_layout_name",
        "is_cuda_blnhc_layout",
        "is_npu_layer_major_blnhc",
        "npu_flash_next_store_columns",
        "npu_flash_next_column_owners",
        "ple_stage_row_map",
        "group_state_flags",
        "state_group_ids",
        "blnhc_layer_role",
        "blnhc_group_role",
    }
    methods = {
        "KVCacheGroupManager": {
            "__init__",
            "num_groups",
            "compute_block_hashes",
            "compute_all_group_block_ids",
        },
        "HybridLinearAttentionLayout": {
            "_tokens_per_state",
            "_has_glm53_layout",
            "_build_layout",
            "_build_glm53_ascend_layout",
            "_validate_glm53_ascend_descriptors",
            "_build_glm53_ascend_row",
            "_finalize_glm53_ascend_columns",
            "_finalize_layout_arrays",
            "_extract_group_addrs",
        },
        "UCMHybridLinearAttentionConnector": {
            "_scope_blocks",
            "_scope_blocks_pcp",
            "_setup_cp_scoping",
            "_make_other_rank_hashers",
            "_append_mamba_align_state_block",
            "_generate_hla_dispatch_meta",
            "update_state_after_alloc",
            "build_connector_meta",
            "start_load_kv",
            "_start_load_kv_async",
            "_poll_pending_hla_loads",
            "get_finished",
            "get_transfer_results",
            "request_finished_all_groups",
            "shutdown",
            "_supports_request_async_load",
            "_mark_load_failed",
        },
        "UCMHybridLinearAttentionLayerWiseConnector": {
            "layerwise_engine_config_error",
            "register_kv_caches",
            "_build_checkpoint_windows",
            "_submit_layer_load",
            "_submit_window_load",
            "_wait_window_load",
            "_wait_layer_load_pair",
            "_wait_ple_on_same_layer",
            "_submit_layer_dump",
            "_drain_unwaited_loads",
            "wait_for_layer_load",
            "save_kv_layer",
            "_plan_row_saves",
            "wait_for_save",
            "start_load_kv",
            "_submit_request_load_tasks_for_row",
            "_wait_row_load",
            "_mark_load_failed",
            "_record_layerwise_load_bytes",
        },
    }
    nodes = []
    found_unhashed_roles = False
    for node in ast.parse(HLA_PATH.read_text(encoding="utf-8-sig")).body:
        if getattr(node, "name", None) in functions | {
            "GroupInfo",
            "HLARequestMeta",
            "HLARequestDispatchMeta",
            "HLAPendingLoad",
        }:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in methods:
            node.bases = []
            node.body = [
                item
                for item in node.body
                if getattr(item, "name", None) in methods[node.name]
                or (
                    node.name == "UCMHybridLinearAttentionLayerWiseConnector"
                    and isinstance(item, ast.Assign)
                )
            ]
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_BLNHC_UNHASHED_ROLES"
            for target in node.targets
        ):
            nodes.append(node)
            found_unhashed_roles = True
    if not found_unhashed_roles:
        raise AssertionError(
            "_BLNHC_UNHASHED_ROLES assignment missing from hla_connector.py"
        )
    connector_path = ROOT / "ucm/integration/vllm/ucm_connector.py"
    connector_names = {
        "KVCacheSegment",
        "RequestMeta",
        "RequestDispatchMeta",
        "PendingLoadTask",
        "UCMConnectorMetadata",
        "build_kv_hash_meta",
        "_get_sparse_c8_flags",
        "_record_counter",
    }
    base_nodes = [
        node
        for node in ast.parse(connector_path.read_text(encoding="utf-8-sig")).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in connector_names
    ]
    for node in base_nodes:
        if node.name == "UCMConnectorMetadata":
            node.bases = []
    nodes = base_nodes + nodes
    namespace = {
        "__name__": __name__,
        "math": math,
        "time": time,
        "np": np,
        "defaultdict": defaultdict,
        "dataclass": dataclass,
        "field": field,
        "Callable": Callable,
        "List": List,
        "Optional": Optional,
        # Stand-in for the engine's transfer-results type so the async
        # failure channel can be asserted without a vllm dependency.
        "KVConnectorTransferResults": SimpleNamespace,
        "KVCacheSpec": KVCacheSpec,
        "FullAttentionSpec": FullAttentionSpec,
        "MLAAttentionSpec": MLAAttentionSpec,
        "MambaSpec": MambaSpec,
        "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
        "torch": SimpleNamespace(Tensor=FakeTensor),
        "logger": MagicMock(),
        "ucmmetrics": MagicMock(),
        "current_platform": SimpleNamespace(
            device_type="npu", is_cuda_alike=lambda: False
        ),
        "KVConnectorRole": SimpleNamespace(WORKER="WORKER", SCHEDULER="SCHEDULER"),
        "Task": object,
        "Future": Future,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "request_hasher_module": SimpleNamespace(
            _layout_source_digest=lambda: "test-layout-digest"
        ),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(HLA_PATH), "exec"), namespace)
    namespace["UCMHybridLinearAttentionLayerWiseConnector"]._mark_load_failed = (
        namespace["UCMHybridLinearAttentionConnector"]._mark_load_failed
    )
    namespace[
        "UCMHybridLinearAttentionLayerWiseConnector"
    ]._supports_request_async_load = classmethod(
        namespace[
            "UCMHybridLinearAttentionConnector"
        ]._supports_request_async_load.__func__
    )
    return SimpleNamespace(**namespace)


def group(spec, *names):
    return SimpleNamespace(kv_cache_spec=spec, layer_names=list(names))


@pytest.fixture
def direct_load_connector(hla):
    connector = hla.UCMHybridLinearAttentionConnector()
    unscoped = [b"kv0", b"kv1", b"state", b"state"]
    scoped = [b"kv0", b"kv1", b"state@1", b"state@1"]
    blocks = [10, 11, 20, 21]
    groups = [0, 0, 1, 1]
    dispatch = SimpleNamespace(
        load_full_attn_count=2, load_block_ids=(unscoped, blocks)
    )
    metadata = hla.UCMConnectorMetadata(request_meta={"req": dispatch})
    connector._get_connector_metadata = lambda: metadata
    connector._reset_transfer_state = lambda: None
    connector._prepare_load_plan = lambda _: None
    connector.request_data = [("req", unscoped, scoped, blocks, groups)]
    connector.is_mla = True
    connector.need_load = True
    connector.layer_order = ["mla", "state"]
    # store.wait must be callable: async submissions dispatch background
    # store.wait calls into the (lazy) executor.
    connector.store = SimpleNamespace(wait=lambda task: None)
    connector._failure_req_ids = set()
    connector._invalid_block_ids = set()
    connector._connector_worker_meta = SimpleNamespace(mark_failed=MagicMock())
    connector._load_bytes_total = 0
    connector._pending_hla_loads = {}
    connector._failed_hla_loads = set()
    connector._finished_async_load_req_ids = set()
    connector._load_wait_executor = None
    connector._load_wait_workers = 2
    connector.use_request_async_load = False
    connector.device = SimpleNamespace(synchronize=MagicMock())
    layout = hla.HybridLinearAttentionLayout.__new__(hla.HybridLinearAttentionLayout)
    layout.direct_group_bytes = {0: 64, 1: 16}
    layout.extract_block_addrs = lambda ids, gids: np.array(
        list(zip(ids, gids)), dtype=np.uint64
    )
    connector.kv_cache_layout = layout
    return connector


@pytest.mark.parametrize(
    "failure", [None, "submit_mla", "submit_state", "wait_mla", "wait_state"]
)
def test_direct_load_splits_shared_and_private_tasks_and_drains_failures(
    hla, direct_load_connector, failure
):
    connector = direct_load_connector
    submissions = []
    waits = []

    def submit(store, requests, keys, shards, ptrs, *, distribute_backend_load):
        label = "mla" if distribute_backend_load else "state"
        assert store is connector.store
        submissions.append((label, requests, keys, shards, ptrs.tolist()))
        if failure == f"submit_{label}":
            raise RuntimeError("submit failed")
        return SimpleNamespace(label=label)

    def wait(task):
        waits.append(task.label)
        if failure == f"wait_{task.label}":
            raise RuntimeError("wait failed")

    connector._rank_consistency = SimpleNamespace(submit_load=submit, wait_load=wait)
    connector.start_load_kv(SimpleNamespace())
    connector.device.synchronize.assert_called_once()
    assert submissions[0] == (
        "mla",
        {"req": [b"kv0", b"kv1"]},
        [b"kv0", b"kv1"],
        [0, 0],
        [[10, 0], [11, 0]],
    )
    if failure != "submit_mla":
        assert submissions[1] == (
            "state",
            {"req": [b"state", b"state"]},
            [b"state@1", b"state@1"],
            [0, 0],
            [[20, 1], [21, 1]],
        )
    expected_waits = [] if failure == "submit_mla" else ["mla"]
    if failure not in ("submit_mla", "submit_state"):
        expected_waits.append("state")
    assert waits == expected_waits
    stats = hla.ucmmetrics.update_stats.call_args.args[0]
    assert stats["load_requests_num"] == (0 if failure else 1)
    if failure:
        assert connector._failure_req_ids == {"req"}
        assert connector._invalid_block_ids == {10, 11, 20, 21}
    else:
        assert stats["load_blocks_num"] == 4
        assert stats["load_bytes_total"] == 160
        assert not connector._failure_req_ids


@pytest.mark.parametrize("shared", [False, True])
def test_direct_load_skips_empty_task_type(hla, direct_load_connector, shared):
    connector = direct_load_connector
    start, end = (0, 2) if shared else (2, 4)
    request_id, unscoped, scoped, blocks, groups = connector.request_data[0]
    connector.request_data = [
        (
            request_id,
            unscoped[start:end],
            scoped[start:end],
            blocks[start:end],
            groups[start:end],
        )
    ]
    dispatch = connector._get_connector_metadata().request_meta[request_id]
    dispatch.load_full_attn_count = 2 if shared else 0
    connector._rank_consistency = SimpleNamespace(
        submit_load=MagicMock(return_value=object()), wait_load=MagicMock()
    )
    connector.start_load_kv(SimpleNamespace())
    connector._rank_consistency.submit_load.assert_called_once()
    assert connector._rank_consistency.submit_load.call_args.kwargs == {
        "distribute_backend_load": shared
    }
    connector._rank_consistency.wait_load.assert_called_once()


def test_gqa_direct_load_keeps_all_rank_scoped_keys_in_one_task(direct_load_connector):
    connector = direct_load_connector
    connector.is_mla = False
    connector._rank_consistency = SimpleNamespace(
        submit_load=MagicMock(return_value=object()), wait_load=MagicMock()
    )
    connector.start_load_kv(SimpleNamespace())
    connector._rank_consistency.submit_load.assert_called_once()
    call = connector._rank_consistency.submit_load.call_args
    assert call.args[2] == [b"kv0", b"kv1", b"state@1", b"state@1"]
    assert call.kwargs == {"distribute_backend_load": False}
    connector._rank_consistency.wait_load.assert_called_once()


def test_async_alloc_dispatches_checkpoint_state_only(hla, state_connector):
    connector = state_connector
    connector.use_request_async_load = True
    req_meta = _restore_request_meta(hla)
    # Async allocation extends the state table with exactly the external-hit
    # blocks — no scratch tail, so the running slot is out of bounds here.
    req_meta.group_vllm_block_ids[1] = [21, 22]
    connector.requests_meta = {"req": req_meta}
    blocks = SimpleNamespace(get_block_ids=lambda: req_meta.group_vllm_block_ids)

    connector.update_state_after_alloc(
        SimpleNamespace(request_id="req"), blocks, num_external_tokens=8
    )

    dispatch = connector._pending_async_load_dispatches["req"]
    assert dispatch.load_async
    assert dispatch.load_block_ids == (
        [b"fa0", b"fa1", b"checkpoint"],
        [11, 12, 22],
    )
    assert dispatch.dump_block_ids == ([], [])

    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    metadata = connector.build_connector_meta(output)
    assert metadata.request_meta["req"] is dispatch
    assert connector._pending_async_load_dispatches == {}

    output.scheduled_new_reqs = [
        SimpleNamespace(req_id="req", block_ids=req_meta.group_vllm_block_ids)
    ]
    output.num_scheduled_tokens = {"req": 1}
    resumed = connector.build_connector_meta(output).request_meta["req"]
    assert resumed.load_block_ids == ([], [])
    assert not resumed.load_async
    assert connector.request_finished_all_groups(
        SimpleNamespace(request_id="req"), req_meta.group_vllm_block_ids
    ) == (False, None)


def test_async_flag_dispatches_all_requests_without_waiting(direct_load_connector):
    connector = direct_load_connector
    connector.use_request_async_load = True
    sync_data = connector.request_data[0]
    connector.request_data.append(("async-req", *sync_data[1:]))
    metadata = connector._get_connector_metadata()
    metadata.request_meta["req"].load_async = True
    metadata.request_meta["async-req"] = SimpleNamespace(
        load_async=True,
        load_full_attn_count=2,
        load_block_ids=(list(sync_data[1]), list(sync_data[3])),
    )

    def submit(store, requests, keys, shards, ptrs, *, distribute_backend_load):
        return SimpleNamespace(request_id=next(iter(requests)))

    connector._rank_consistency = SimpleNamespace(
        submit_load=submit, wait_load=MagicMock()
    )
    connector.start_load_kv(SimpleNamespace())

    connector.device.synchronize.assert_not_called()
    connector._rank_consistency.wait_load.assert_not_called()
    assert set(connector._pending_hla_loads) == {"req", "async-req"}
    assert len(connector._pending_hla_loads["async-req"].tasks) == 2


def test_async_direct_load_waits_for_mla_and_state_before_completion(
    hla,
    direct_load_connector,
):
    connector = direct_load_connector
    connector.use_request_async_load = True
    dispatch = connector._get_connector_metadata().request_meta["req"]
    dispatch.load_async = True
    # Each background store.wait blocks on its own event, so the test drives
    # per-task completion exactly like the Store deadline would.
    done = {"mla": threading.Event(), "state": threading.Event()}
    waits = []

    def submit(store, requests, keys, shards, ptrs, *, distribute_backend_load):
        return SimpleNamespace(label="mla" if distribute_backend_load else "state")

    connector.store = SimpleNamespace(
        wait=lambda task: done[task.label].wait(timeout=5)
    )

    def wait_load(task, wait=None):
        # Mirror rank_consistency.wait_load: invoke the background wait
        # callable so its result/failure surfaces here.
        waits.append(task.label)
        if wait is not None:
            wait()

    connector._rank_consistency = SimpleNamespace(
        submit_load=submit,
        wait_load=wait_load,
    )
    connector.start_load_kv(SimpleNamespace())
    try:
        connector.device.synchronize.assert_not_called()
        assert not waits
        pending = connector._pending_hla_loads["req"]
        assert len(pending.tasks) == 2
        assert not any(f.done() for f in pending.wait_futures.values())
        assert connector.get_finished(set()) == (None, None)

        done["mla"].set()
        assert connector.get_finished(set()) == (None, None)

        done["state"].set()
        # Let the background threads observe the events before polling.
        concurrent.futures.wait(pending.wait_futures.values(), timeout=5)
        # Finished req ids trigger the cancel-drain path: the request's
        # loads are drained immediately and reported as complete.
        assert connector.get_finished({"req"}) == (None, {"req"})
        assert waits == ["mla", "state"]
        assert not connector._pending_hla_loads
        assert not connector._invalid_block_ids
        hla.ucmmetrics.update_stats.assert_called_with(
            {
                "load_bytes_total": 160,  # 2 MLA blocks * 64 + 2 state * 16
                "load_blocks_num": 4,
                "load_requests_num": 1,
            }
        )

        assert connector.get_finished({"req", "sync-req"}) == (None, None)
    finally:
        for event in done.values():
            event.set()
        connector.shutdown()


def test_synchronous_finished_req_returns_no_async_completion(
    direct_load_connector,
):
    assert direct_load_connector.get_finished({"req"}) == (None, None)


def test_async_direct_load_failure_drains_remaining_state(direct_load_connector):
    connector = direct_load_connector
    connector.use_request_async_load = True
    connector._get_connector_metadata().request_meta["req"].load_async = True
    state_done = threading.Event()

    def submit(store, requests, keys, shards, ptrs, *, distribute_backend_load):
        return SimpleNamespace(label="mla" if distribute_backend_load else "state")

    def wait(task):
        if task.label == "mla":
            raise RuntimeError("MLA load failed")
        state_done.wait(timeout=5)

    connector.store = SimpleNamespace(wait=wait)
    connector._rank_consistency = SimpleNamespace(
        submit_load=submit,
        # Mirror rank_consistency.wait_load: invoke the background wait
        # callable so the injected RuntimeError surfaces and fails the load.
        wait_load=lambda task, wait=None: None if wait is None else wait(),
    )
    connector.start_load_kv(SimpleNamespace())

    assert connector.get_finished(set()) == (None, None)
    assert "req" in connector._pending_hla_loads

    state_done.set()
    concurrent.futures.wait(
        connector._pending_hla_loads["req"].wait_futures.values(), timeout=5
    )
    results = connector.get_transfer_results(set())
    assert results.finished_recving == {"req"}
    assert results.failed_recving == {"req"}
    assert not results.finished_sending
    assert not connector._pending_hla_loads
    assert not connector._invalid_block_ids
    connector._connector_worker_meta.mark_failed.assert_called_with("req")


def test_async_direct_load_submit_failure_does_not_use_flat_block_ids(
    direct_load_connector,
):
    connector = direct_load_connector
    connector.use_request_async_load = True
    connector._get_connector_metadata().request_meta["req"].load_async = True

    def submit(*args, **kwargs):
        raise RuntimeError("submit failed")

    connector._rank_consistency = SimpleNamespace(submit_load=submit)
    connector.start_load_kv(SimpleNamespace())

    # An immediate submit failure completes the request in the same call:
    # it lands in _finished_async_load_req_ids, then flows through
    # get_transfer_results as failed_recving without flat block ids.
    results = connector.get_transfer_results(set())
    assert results.finished_recving == {"req"}
    assert results.failed_recving == {"req"}
    assert not results.finished_sending
    assert not connector._invalid_block_ids
    connector._connector_worker_meta.mark_failed.assert_called_with("req")


def test_request_async_is_scoped_to_direct_hla(hla):
    assert hla.UCMHybridLinearAttentionConnector._supports_request_async_load()
    assert (
        not hla.UCMHybridLinearAttentionLayerWiseConnector._supports_request_async_load()
    )


@pytest.mark.parametrize(
    "is_mla,group_id,expected", [(True, 0, True), (True, 1, False), (False, 0, False)]
)
def test_layerwise_load_distributes_only_shared_mla_group(
    hla, is_mla, group_id, expected
):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.is_mla = is_mla
    connector.layer_order = ["layer"]
    connector.layer_to_group = {"layer": group_id}
    connector.layer_to_slot = {"layer": 0}
    connector._submitted_positions = set()
    connector._submitted_group_slots = set()
    connector._failure_req_ids = set()
    connector._load_bytes_by_pair = defaultdict(int)
    connector.load_tasks = defaultdict(dict)
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            group(MLAAttentionSpec()),
            group(MambaSpec(mamba_cache_mode="align")),
        ]
    )
    connector._state_group_ids = hla.state_group_ids(connector._kv_cache_config)
    metadata = hla.UCMConnectorMetadata(request_meta={})
    connector._get_connector_metadata = lambda: metadata
    connector.request_data = [("req", [b"key"], [b"scoped"], [10], [group_id])]
    connector.store = object()
    layout = hla.HybridLinearAttentionLayout.__new__(hla.HybridLinearAttentionLayout)
    layout.extract_layer_addrs = lambda name, blocks: np.array([[10]], dtype=np.uint64)
    layout.layer_column_bytes = {"layer": 16}
    connector.kv_cache_layout = layout
    task = object()
    connector._rank_consistency = SimpleNamespace(
        submit_load=MagicMock(return_value=task)
    )
    connector._submit_layer_load(0)
    assert connector._rank_consistency.submit_load.call_args.kwargs == {
        "distribute_backend_load": expected
    }
    assert connector.load_tasks[(group_id, 0)]["req"] is task
    assert connector._load_bytes_by_pair[(group_id, 0)] == 16


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"layers": ["new"]}, ["new"]),
        ({"shared_by": ["old"]}, ["old"]),
        ({"layers": ["new"], "shared_by": ["old"]}, ["new"]),
        ({"layers": [], "shared_by": ["old"]}, []),
        ({"layers": None, "shared_by": ["old"]}, ["old"]),
        ({"shared_by": None}, []),
        ({"layers": None, "shared_by": None}, []),
        ({"layers": ("a", "b")}, ("a", "b")),
        ({}, []),
    ],
)
def test_tensor_layer_names_preserve_values_and_handle_none(hla, fields, expected):
    result = hla._kv_cache_tensor_layers(SimpleNamespace(**fields))
    assert result == expected
    assert isinstance(result, type(expected))


@pytest.mark.parametrize(
    "layer_name,expected",
    [
        ("model.layers.0.linear_attn.raw_key", "raw_key"),
        # Engine QSA owner name: dotted segment is raw_key_cache, not raw_key.
        ("model.layers.0.indexer.raw_key_cache", "raw_key"),
        ("model.layers.0.linear_attn.k_compress", "compressed"),
        ("model.layers.0.linear_attn.ple_state", "ple"),
        ("model.layers.0.self_attn.attn", ""),
        ("model.layers.0.linear_attn.raw_keypad", ""),
    ],
)
def test_blnhc_layer_role(hla, layer_name, expected):
    assert hla.blnhc_layer_role(layer_name) == expected


def test_blnhc_group_role_requires_every_layer(hla):
    assert hla.blnhc_group_role(["model.layers.0.linear_attn.ple_state"]) == "ple"
    assert (
        hla.blnhc_group_role(
            [
                "model.layers.0.self_attn.attn",
                "model.layers.0.linear_attn.k_compress",
            ]
        )
        == ""
    )


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, True),
        ({"prefix_cacheable": False}, False),
        ({"participates_in_prefix_caching": False}, False),
        ({"prefix_cacheable": True, "participates_in_prefix_caching": False}, False),
        ({"prefix_cacheable": False, "participates_in_prefix_caching": True}, False),
        ({"prefix_cacheable": True, "participates_in_prefix_caching": True}, True),
    ],
)
def test_prefix_eligibility_checks_both_descriptor_versions(hla, fields, expected):
    assert hla.participates_in_prefix_caching(KVCacheSpec(**fields)) is expected


def test_nested_uniform_prefix_eligibility(hla):
    tail = KVCacheSpec(prefix_cacheable=False)
    nested = UniformTypeKVCacheSpecs(kv_cache_specs={"tail": tail})
    outer = UniformTypeKVCacheSpecs(
        kv_cache_specs={"attention": FullAttentionSpec(), "nested": nested}
    )
    assert not hla.participates_in_prefix_caching(outer)
    tail.prefix_cacheable = True
    assert hla.participates_in_prefix_caching(outer)


def test_tail_is_excluded_from_hashes_and_lcm_without_renumbering(hla):
    config = SimpleNamespace(
        kv_cache_groups=[
            group(FullAttentionSpec(block_size=4), "mla"),
            group(KVCacheSpec(block_size=7, prefix_cacheable=False), "tail"),
            group(MambaSpec(block_size=6, mamba_cache_mode="align"), "gdn"),
        ]
    )
    hasher = MagicMock(return_value=b"seed")
    hash_request = MagicMock(return_value=[b"a", b"b", b"c"])
    hasher.make_request_block_hasher.return_value = hash_request
    manager = hla.KVCacheGroupManager(config, hasher, b"base")
    assert manager.lcm_block_size == 12
    assert [g.group_id for g in manager.full_attn_groups] == [0]
    assert [g.group_id for g in manager.non_prefix_groups] == [1]
    assert [g.group_id for g in manager.state_groups] == [2]
    assert manager.compute_all_group_block_ids(
        SimpleNamespace(all_token_ids=list(range(12)))
    ) == [[b"a", b"b", b"c"], [], [b"", b""]]
    hash_request.assert_called_once()


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, 1),
        ({"compress_ratio": 4}, 4),
        ({"tokens_per_state": 2, "compress_ratio": 4}, 2),
    ],
)
def test_compression_field_fallback(hla, fields, expected):
    assert (
        hla.HybridLinearAttentionLayout._tokens_per_state(KVCacheSpec(**fields))
        == expected
    )


def _glm_detection_layout(hla, hf_fields):
    specs = {
        "mla": MLAAttentionSpec(tokens_per_state=1),
        "indexer": MLAAttentionSpec(tokens_per_state=4),
    }
    config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(layers=["mla", "indexer", "state"])],
        kv_cache_groups=[
            group(UniformTypeKVCacheSpecs(kv_cache_specs=specs), *specs),
            group(MambaSpec(mamba_cache_mode="align"), "state"),
        ],
    )
    layout = hla.HybridLinearAttentionLayout()
    layout.kv_cache_config = config
    layout.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf_fields))
    )
    return layout


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"architectures": ["Glm5NextForConditionalGeneration"]}, True),
        ({"architectures": ["OtherModel", "Glm5NextForConditionalGeneration"]}, True),
        ({"architectures": ["Glm5NextForCausalLM"]}, False),
        ({"architectures": ["OtherHybridForConditionalGeneration"]}, False),
        ({"architectures": ["Glm5NextForConditionalGenerationExtra"]}, False),
        ({"architectures": []}, False),
        ({"architectures": None}, False),
        ({}, False),
    ],
)
def test_glm_detection_uses_exact_supported_architecture(hla, fields, expected):
    layout = _glm_detection_layout(hla, fields)
    assert layout._has_glm53_layout(layout.kv_cache_config) is expected


@pytest.mark.parametrize("tag", [None, "other_model", "glm5_next"])
def test_similar_structure_and_spec_tags_do_not_override_other_architecture(hla, tag):
    layout = _glm_detection_layout(
        hla, {"architectures": ["OtherHybridForConditionalGeneration"]}
    )
    specs = layout.kv_cache_config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    for spec in specs.values():
        spec.model_version = tag
    assert not layout._has_glm53_layout(layout.kv_cache_config)


@pytest.mark.parametrize(
    "device,expected", [("npu", True), ("cuda", True), ("cpu", False)]
)
def test_glm_detection_preserves_platform_guard(hla, device, expected):
    layout = _glm_detection_layout(
        hla, {"architectures": ["Glm5NextForConditionalGeneration"]}
    )
    hla.current_platform.device_type = device
    hla.current_platform.is_cuda_alike = lambda: device == "cuda"
    assert layout._has_glm53_layout(layout.kv_cache_config) is expected


def test_glm_detection_requires_allocations(hla):
    layout = _glm_detection_layout(
        hla, {"architectures": ["Glm5NextForConditionalGeneration"]}
    )
    layout.kv_cache_config.kv_cache_tensors = []
    assert not layout._has_glm53_layout(layout.kv_cache_config)


def test_supported_architecture_selects_ascend_layout(hla):
    layout = _glm_detection_layout(
        hla, {"architectures": ["Glm5NextForConditionalGeneration"]}
    )
    layout._build_glm53_ascend_layout = MagicMock()
    caches = {"mla": object()}
    layout._build_layout(caches)
    layout._build_glm53_ascend_layout.assert_called_once_with(caches)


def test_supported_architecture_explicitly_rejects_cuda(hla):
    layout = _glm_detection_layout(
        hla, {"architectures": ["Glm5NextForConditionalGeneration"]}
    )
    hla.current_platform.device_type = "cuda"
    hla.current_platform.is_cuda_alike = lambda: True
    with pytest.raises(
        NotImplementedError, match="CUDA KV cache layout is not supported"
    ):
        layout._build_layout({})


@pytest.mark.parametrize(
    "architecture", ["OtherHybridForConditionalGeneration", "Glm5NextForCausalLM"]
)
def test_other_architectures_keep_generic_cuda_path(hla, architecture):
    layout = _glm_detection_layout(hla, {"architectures": [architecture]})
    hla.current_platform.device_type = "cuda"
    hla.current_platform.is_cuda_alike = lambda: True
    layout._build_glm53_ascend_layout = MagicMock()
    # Exercise the generic dispatch branch without requiring physical tensors.
    layout._collect_shared_tensor_info = MagicMock(return_value=([], []))
    layout._build_layout({})
    layout._collect_shared_tensor_info.assert_called_once()
    layout._build_glm53_ascend_layout.assert_not_called()


def test_empty_rope_is_skipped_before_contiguity_checks(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    key = FakeTensor(1000, (4, 2, 4), (8, 4, 1))
    rope = FakeTensor(2000, (4, 2, 0), (8, 4, 1))
    rows = layout._build_glm53_ascend_row(
        "mla",
        {"mla": (key, rope)},
        SimpleNamespace(size=64),
        KVCacheSpec(page_size_bytes=32),
    )
    assert len(rows) == 1
    segment = rows[0]
    assert (
        segment.ptr,
        segment.copy_size,
        segment.block_stride,
        segment.buffer_size,
    ) == (
        1000,
        32,
        32,
        64,
    )


def test_padded_state_copies_only_live_bytes(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    rows = layout._build_glm53_ascend_row(
        "gdn",
        {"gdn": FakeTensor(1000, (2, 2, 3), (16, 3, 1))},
        SimpleNamespace(size=64),
        KVCacheSpec(page_size_bytes=32),
    )
    assert (rows[0].copy_size, rows[0].block_stride, rows[0].buffer_size) == (
        12,
        32,
        44,
    )


def test_nonempty_noncontiguous_view_is_rejected(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    with pytest.raises(ValueError, match="Non-contiguous Ascend KV block"):
        layout._build_glm53_ascend_row(
            "mla",
            {"mla": FakeTensor(1000, (2, 2, 3), (16, 4, 1))},
            SimpleNamespace(size=64),
            KVCacheSpec(page_size_bytes=32),
        )


@pytest.mark.parametrize("compressed_first", [False, True])
def test_glm_families_preserve_column_order_and_pad_rows(hla, compressed_first):
    specs = {
        "mla0": MLAAttentionSpec(page_size_bytes=16, tokens_per_state=1),
        "mla1": MLAAttentionSpec(page_size_bytes=16, tokens_per_state=1),
        "index": MLAAttentionSpec(page_size_bytes=16, tokens_per_state=4),
    }
    names = ["index", "mla0", "mla1"] if compressed_first else ["mla0", "index", "mla1"]
    layout = hla.HybridLinearAttentionLayout()
    layout.num_blocks = 2
    layout.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            group(UniformTypeKVCacheSpecs(kv_cache_specs=specs), *names),
            group(KVCacheSpec(page_size_bytes=16, prefix_cacheable=False), "tail"),
            group(MambaSpec(page_size_bytes=16), "gdn"),
        ],
        kv_cache_tensors=[
            SimpleNamespace(layers=[name], size=32, block_stride=16)
            for name in [*names, "tail", "gdn"]
        ],
    )

    def tensor(ptr):
        return FakeTensor(ptr, (2, 2), (8, 1))

    layout._build_glm53_ascend_layout(
        {
            "mla0": tensor(1000),
            "mla1": tensor(3000),
            "index": tensor(2000),
            "gdn": (tensor(5000), tensor(5004)),
        }
    )
    first = [2000, 1000] if compressed_first else [1000, 2000]
    second = [0, 3000] if compressed_first else [3000, 0]
    assert layout.base_ptrs.tolist() == [*first, 5000, 5004, *second, 0, 0]
    assert layout.row_tensor_size_lists == [[4, 4, 4, 4], [4, 4, 4, 4]]
    assert layout.row_shard_sizes == [16, 16]
    assert set(layout.group_layouts) == {0, 2}
    assert layout.preload_all_rows and layout.save_after_forward
    assert layout.layer_name_to_row["mla1"] == 1
    assert "tail" not in layout.layer_name_to_row
    addresses = layout._extract_group_addrs([1, 1], [0, 2])
    assert addresses[0].tolist() == [
        first[0] + 16,
        first[1] + 16,
        0,
        0,
        *[value + 16 if value else 0 for value in second],
        0,
        0,
    ]
    assert addresses[1].tolist() == [0, 0, 5016, 5020, 0, 0, 0, 0]
    np.testing.assert_array_equal(
        layout._extract_group_addrs([1, 1], [0, 2], layout.row_slices[1]),
        addresses[:, 4:],
    )


def test_group_address_validation_and_empty_batch(hla):
    layout = hla.HybridLinearAttentionLayout()
    layout.base_ptrs = np.array([1000], dtype=np.uint64)
    layout.group_layouts = {0: (layout.base_ptrs, np.array([16], dtype=np.uint64))}
    with pytest.raises(ValueError, match="lengths differ"):
        layout._extract_group_addrs([1], [])
    with pytest.raises(ValueError, match="No hybrid physical layout"):
        layout._extract_group_addrs([1], [5])
    empty = layout._extract_group_addrs([], [])
    assert empty.shape == (0, 1)
    assert empty.flags.c_contiguous


@pytest.fixture
def save_connector(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector._metadata_factory = hla.UCMConnectorMetadata
    specs = {
        "attn0": FullAttentionSpec(),
        "state2": MambaSpec(),
        "attn3": FullAttentionSpec(),
        "index4": MLAAttentionSpec(tokens_per_state=4),
        "state9": MambaSpec(),
        "attn8": FullAttentionSpec(),
        "tail100": KVCacheSpec(prefix_cacheable=False),
    }
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[group(spec, name) for name, spec in specs.items()]
    )
    connector.kv_cache_layout = SimpleNamespace(save_after_forward=False)
    connector.layer_name_to_id = {
        "attn0": 0,
        "state2": 2,
        "attn3": 3,
        "index4": 4,
        "state9": 9,
        "attn8": 8,
        "tail100": 100,
    }
    connector.layer_name_to_row = {
        "attn0": 0,
        "state2": 0,
        "attn3": 1,
        "index4": 1,
        "state9": 2,
        "attn8": 3,
        "tail100": 0,
    }
    connector.row_ids = [0, 1, 2, 3]
    connector.commit_row_id = 3
    return connector


def test_row_saves_wait_for_last_component_and_exclude_commit(save_connector):
    # index4 cannot be a callback; tail100 must not delay row 0.
    # state9 has no subsequent callback, so row 2 stays for post-forward.
    assert save_connector._plan_row_saves() == {"attn3": [0], "attn8": [1]}


def test_no_attention_callback_leaves_rows_for_post_forward(save_connector):
    save_connector._kv_cache_config.kv_cache_groups = [
        group(MambaSpec(), name) for name in save_connector.layer_name_to_row
    ]
    assert save_connector._plan_row_saves() == {}


def test_glm_saves_are_all_deferred(save_connector):
    save_connector.kv_cache_layout.save_after_forward = True
    assert save_connector._plan_row_saves() == {}


@pytest.mark.parametrize("already_saved", [set(), {0, 1}])
def test_post_forward_backfills_rows_before_commit(save_connector, already_saved):
    connector = save_connector
    events = []
    metadata = connector._metadata_factory(request_meta={})
    connector._connector_metadata = metadata
    connector._get_connector_metadata = lambda: metadata
    connector._dumped_row_ids = set(already_saved)
    connector.dump_tasks = {}
    connector._dump_transfer_data = (None, None, None, {"req"})
    connector._layerwise_save_bytes = 0
    connector.is_save = True
    connector.enable_event_sync = False
    connector._rank_consistency = SimpleNamespace(
        finish_dump=lambda requests: events.append(("finish", requests))
    )

    def submit(row, meta):
        assert meta is metadata
        events.append(("submit", row))
        connector._dumped_row_ids.add(row)

    connector._submit_dump_row = submit
    connector._wait_dump_rows = lambda rows: events.append(("wait", rows))
    connector.wait_for_save()
    assert events == [
        *(("submit", row) for row in [0, 1, 2] if row not in already_saved),
        ("wait", [0, 1, 2]),
        ("submit", 3),
        ("wait", [3]),
        ("finish", {"req"}),
    ]
    assert connector._dump_transfer_data is None
    assert not connector.is_save


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("is_dump", [False, True])
def test_rank_scoping_keeps_keys_blocks_and_groups_paired(hla, is_mla, rank, is_dump):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector.tp_rank = rank
    connector.tp_size = 2
    connector.is_mla = is_mla
    connector.request_hasher = lambda key: b"rank1:" + key
    result = connector._scope_blocks([b"mla", b"gdn"], [11, 22], [0, 2], 1, is_dump)
    selected = [1] if is_mla and rank == 1 and is_dump else [0, 1]
    expected_keys = [b"mla", b"gdn"]
    expected_scoped = [
        b"mla" if is_mla or rank == 0 else b"rank1:mla",
        b"gdn" if rank == 0 else b"rank1:gdn",
    ]
    assert result == tuple(
        [values[index] for index in selected]
        for values in [expected_keys, expected_scoped, [11, 22], [0, 2]]
    )


def test_null_filter_keeps_group_pairing(hla):
    keys, blocks, groups = [], [], []
    hla.extend_non_null(keys, blocks, groups, [b"a", b"b", b"c"], [3, 0, 5], 7)
    assert (keys, blocks, groups) == ([b"a", b"c"], [3, 5], [7, 7])


@pytest.fixture
def state_connector(hla):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector.is_mla = True
    connector.group_manager = SimpleNamespace(
        num_groups=2,
        lcm_block_size=4,
        groups_by_id=[
            SimpleNamespace(
                block_size=4,
                is_full_attention=True,
                is_mamba_align=False,
                is_state_hash=False,
            ),
            SimpleNamespace(
                block_size=4,
                is_full_attention=False,
                is_mamba_align=True,
                is_state_hash=True,
            ),
        ],
        compute_mamba_align_state_hash=MagicMock(return_value=b"checkpoint"),
    )
    connector._pending_async_load_dispatches = {}
    connector._async_load_req_ids = set()
    connector._pending_hla_loads = {}
    connector._failed_hla_loads = set()
    connector._finished_async_load_req_ids = set()
    return connector


@pytest.mark.parametrize(
    "seq_len,scheduled,table,reason,expected",
    [
        (8, 1, [21, 22, 23, 99, 100], "load", [22, 23]),
        (8, 4, [21, 22, 23, 99, 100], "load", [22, 23]),
        (8, 0, [21, 22, 99, 100], "load", [22]),
        (7, 1, [21, 22, 99, 100], "load", [22]),
        (8, 1, [21, 22, 22, 99, 100], "load", [22]),
        (8, 1, [21, 0, 23, 99, 100], "load", [23]),
        (8, 1, [21, 22, 0, 99, 100], "load", [22]),
        (8, 1, [21, 0, 0, 99, 100], "load", []),
        (8, 4, [21, 22, 23, 99, 100], "dump", [22]),
    ],
)
def test_state_restore_selects_checkpoint_and_running_not_scratch(
    hla, state_connector, seq_len, scheduled, table, reason, expected
):
    req_meta = hla.HLARequestMeta(
        group_vllm_block_ids=[[11, 12, 13], table],
        group_ucm_block_ids=[[b"a", b"b", b"c"], []],
    )
    keys, blocks, groups = [], [], []
    state_connector._append_mamba_align_state_block(
        keys, blocks, groups, req_meta, "req", 1, seq_len, reason, scheduled
    )
    assert blocks == expected
    assert keys == [b"checkpoint"] * len(expected)
    assert groups == [1] * len(expected)
    assert 99 not in blocks and 100 not in blocks
    if expected:
        state_connector.group_manager.compute_mamba_align_state_hash.assert_called_once_with(
            state_connector.group_manager.groups_by_id[1],
            seq_len,
            req_meta.group_ucm_block_ids,
        )
    else:
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_checkpoint", "missing_hash"])
def test_state_restore_without_checkpoint_slot_appends_nothing(
    hla, state_connector, failure
):
    # Without the checkpoint slot there is nothing worth restoring: a lone
    # running-slot seed would compute from state no future request could
    # reuse, so the whole state restore is skipped.
    tables = {
        "missing_checkpoint": [21],
        "missing_hash": [21, 22, 23],
    }
    if failure == "missing_hash":
        state_connector.group_manager.compute_mamba_align_state_hash.return_value = None
    req_meta = hla.HLARequestMeta(
        group_vllm_block_ids=[[], tables[failure]],
        group_ucm_block_ids=[[], []],
    )
    keys, blocks, groups = [b"existing"], [7], [0]
    state_connector._append_mamba_align_state_block(
        keys, blocks, groups, req_meta, "req", 1, 8, "load", 1
    )
    assert (keys, blocks, groups) == ([b"existing"], [7], [0])
    assert hla.logger.error.called


def test_state_restore_tolerates_missing_running_slot(hla, state_connector):
    # The checkpoint restore (correctness) survives a running slot that is
    # out of bounds; only the promotion-seeding copy is skipped.
    req_meta = hla.HLARequestMeta(
        group_vllm_block_ids=[[], [21, 22]],
        group_ucm_block_ids=[[], []],
    )
    keys, blocks, groups = [b"existing"], [7], [0]
    state_connector._append_mamba_align_state_block(
        keys, blocks, groups, req_meta, "req", 1, 8, "load", 1
    )
    assert (keys, blocks, groups) == (
        [b"existing", b"checkpoint"],
        [7, 22],
        [0, 1],
    )
    hla.logger.error.assert_called_once()


def _restore_request_meta(hla, *, hbm_hits=0):
    return hla.HLARequestMeta(
        hbm_hit_block_num=hbm_hits,
        total_hit_block_num=2,
        num_token_ids=40,
        token_processed=8,
        group_ucm_block_ids=[
            [b"fa0", b"fa1", b"fa2", b"fa3"],
            [b"", b"", b"", b""],
        ],
        group_vllm_block_ids=[
            [11, 12, 13, 14],
            [21, 22, 23, 99, 100],
        ],
    )


@pytest.mark.parametrize("format_name", ["legacy_list", "resumed_flags", "resumed_set"])
@pytest.mark.parametrize("resumed", [False, True])
def test_cached_request_only_reloads_on_resume(
    hla, state_connector, format_name, resumed
):
    req_meta = _restore_request_meta(hla)
    state_connector.requests_meta = {"req": req_meta}
    if format_name == "legacy_list":
        cached = [
            SimpleNamespace(
                req_id="req",
                new_block_ids=(
                    tuple(list(ids) for ids in req_meta.group_vllm_block_ids)
                    if resumed
                    else ([], [])
                ),
                resumed_from_preemption=resumed,
            )
        ]
    else:
        cached = SimpleNamespace(
            req_ids=["req"],
            new_block_ids=[
                (
                    tuple(list(ids) for ids in req_meta.group_vllm_block_ids)
                    if resumed
                    else None
                )
            ],
        )
        if format_name == "resumed_flags":
            cached.resumed_from_preemption = [resumed]
        else:
            cached.resumed_req_ids = {"req"} if resumed else set()
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req": 1},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    dispatch = state_connector.build_connector_meta(output).request_meta["req"]
    if resumed:
        assert dispatch.load_block_ids == (
            [b"fa0", b"fa1", b"checkpoint", b"checkpoint"],
            [11, 12, 22, 23],
        )
        assert dispatch.load_group_ids == [0, 0, 1, 1]
        assert not dispatch.load_async
    else:
        # Keep a positive external-hit count to ensure the continuation
        # gate, rather than the absence of a hit, prevents a reload.
        assert dispatch.load_block_ids == ([], [])
        assert dispatch.load_group_ids == []
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


@pytest.mark.parametrize("hbm_hits", [0, 1, 2])
def test_new_request_restores_only_external_hit_increment(
    hla, state_connector, hbm_hits
):
    req_meta = _restore_request_meta(hla, hbm_hits=hbm_hits)
    state_connector.requests_meta = {"req": req_meta}
    output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req", block_ids=([11, 12, 13], [21, 22, 23, 99]))
        ],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={"req": 1},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    dispatch = state_connector.build_connector_meta(output).request_meta["req"]
    if hbm_hits < 2:
        expected_attention_keys = [b"fa0", b"fa1"][hbm_hits:]
        expected_attention_blocks = [11, 12][hbm_hits:]
        assert dispatch.load_block_ids == (
            expected_attention_keys + [b"checkpoint", b"checkpoint"],
            expected_attention_blocks + [22, 23],
        )
        assert dispatch.load_group_ids == [0] * (2 - hbm_hits) + [1, 1]
    else:
        assert dispatch.load_block_ids == ([], [])
        state_connector.group_manager.compute_mamba_align_state_hash.assert_not_called()


@pytest.mark.parametrize(
    "new_tokens,expected",
    [
        (8, [16]),
        (14, []),
    ],
)
def test_dump_writes_state_only_when_step_ends_on_lcm_boundary(
    hla, state_connector, new_tokens, expected
):
    req_meta = _restore_request_meta(hla)
    state_connector._generate_hla_dispatch_meta(
        req_meta, new_tokens, ([], []), request_id="req", need_load=False
    )
    seq_lens = [
        call.args[1]
        for call in (
            state_connector.group_manager.compute_mamba_align_state_hash.call_args_list
        )
    ]
    assert seq_lens == expected


@pytest.mark.parametrize("row_count", [2, 3])
def test_glm_preload_waits_all_state_rows_before_returning(
    hla, state_connector, row_count
):
    dispatch = state_connector._generate_hla_dispatch_meta(
        _restore_request_meta(hla), 1, ([], []), request_id="req"
    )
    metadata = hla.UCMConnectorMetadata(request_meta={"req": dispatch})
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector._get_connector_metadata = lambda: metadata
    connector.tp_rank = 0
    connector.tp_size = 1
    connector.is_mla = True
    connector._scope_blocks = state_connector._scope_blocks
    state_connector.tp_rank = 0
    state_connector.tp_size = 1
    connector.row_ids = list(range(row_count))
    connector._load_prefetch_rows = 0
    connector._row_shard_size = 16
    connector.load_tasks = defaultdict(dict)
    connector.request_data = []
    connector._failure_req_ids = set()
    connector._submitted_load_rows = set()
    connector._dumped_row_ids = set()
    connector._load_block_counts = {}
    connector._invalid_block_ids = set()
    connector._connector_worker_meta = SimpleNamespace(mark_failed=MagicMock())
    connector.store = object()
    events = []
    memory = {
        row: {22: b"old-checkpoint", 23: b"old-running", 99: b"scratch"}
        for row in connector.row_ids
    }

    def addresses(blocks, row, group_ids):
        assert blocks == [11, 12, 22, 23]
        assert group_ids == [0, 0, 1, 1]
        return np.array([[block] for block in blocks], dtype=np.uint64)

    connector.kv_cache_layout = SimpleNamespace(
        preload_all_rows=True, extract_block_addrs_for_row=addresses
    )
    connector.device = SimpleNamespace(synchronize=lambda: events.append(("sync",)))

    def submit(store, requests, keys, shards, ptrs):
        assert store is connector.store
        assert requests == {"req": dispatch.load_block_ids[0]}
        row = shards[0]
        assert shards == [row] * len(keys)
        events.append(("submit", row))
        return SimpleNamespace(row=row, keys=list(keys), blocks=ptrs[:, 0].tolist())

    def wait(task):
        events.append(("wait", task.row))
        # Fake the external DMA: pre-existing destination contents are
        # replaced only when this row's load is completed.
        for key, block in zip(task.keys, task.blocks):
            if block in memory[task.row]:
                memory[task.row][block] = key

    connector._rank_consistency = SimpleNamespace(submit_load=submit, wait_load=wait)
    connector.start_load_kv(SimpleNamespace())
    assert events == [
        ("sync",),
        *(("submit", row) for row in connector.row_ids),
        *(("wait", row) for row in connector.row_ids),
    ]
    assert not connector.load_tasks
    assert connector._layerwise_load_bytes == row_count * 4 * 16
    assert connector._layerwise_load_bytes_recorded
    # A consumer proceeding after start_load_kv sees both restored slots
    # for every row, while the unused speculative tail remains untouched.
    for values in memory.values():
        assert values == {22: b"checkpoint", 23: b"checkpoint", 99: b"scratch"}


def test_npu_flash_next_whole_store_columns(hla):
    columns, shard, block = hla.npu_flash_next_store_columns(
        12, 524288, 32768, 184320, False
    )
    assert columns == [524288] * 12 + [32768] * 12 + [184320]
    assert shard == block == 6868992


def test_npu_flash_next_layerwise_columns(hla):
    columns, shard, block = hla.npu_flash_next_store_columns(
        12, 524288, 32768, 184320, True
    )
    assert columns == [524288, 32768]
    assert shard == 557056
    assert block == 6684672


def test_npu_flash_next_column_owners(hla):
    whole = hla.npu_flash_next_column_owners(12, 0, [1, 2, 3], 4, False)
    assert whole[0] == {(index, 0) for index in range(24)}
    assert whole[1] == {(index, 0) for index in range(12)}
    assert whole[4] == {(24, 0)}
    layerwise = hla.npu_flash_next_column_owners(12, 0, [1], 4, True)
    assert layerwise[0] == {(type_id, slot) for type_id in (0, 1) for slot in range(12)}
    assert layerwise[1] == {(0, slot) for slot in range(12)}
    assert layerwise[4] == {(0, 0)}


def test_ple_stage_row_map_skips_null_and_duplicates(hla):
    assert hla.ple_stage_row_map([0, 7, 7, 3, 0]) == {7: 0, 3: 1}
    assert hla.ple_stage_row_map([]) == {}


def test_non_mamba_ple_is_a_state_group(hla):
    config = SimpleNamespace(
        kv_cache_layout="BLNHC",
        kv_cache_groups=[
            group(FullAttentionSpec(), "layers.0.self_attn"),
            group(FullAttentionSpec(), "layers.1.ple"),
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                layers=["layers.0.self_attn", "layers.1.linear"],
                layer_stride=200,
                block_stride=50,
            )
        ],
    )
    assert hla.state_group_ids(config) == {1}
    assert hla.group_state_flags(config, config.kv_cache_groups[1]) == (False, True)


def test_is_npu_layer_major_blnhc(hla):
    def config(name, tensors):
        return SimpleNamespace(kv_cache_layout=name, kv_cache_tensors=tensors)

    multi = SimpleNamespace(
        layers=["a", "b"], layer_stride=2104881696, block_stride=524288
    )
    assert hla.is_npu_layer_major_blnhc(config("BLNHC", [multi]))
    block_first = SimpleNamespace(
        layers=["a", "b"], layer_stride=100, block_stride=524288
    )
    assert not hla.is_npu_layer_major_blnhc(config("BLNHC", [block_first]))
    assert not hla.is_npu_layer_major_blnhc(config("LBNHC", [multi]))
    ple_only = SimpleNamespace(
        layers=["ple"], layer_stride=695070720, block_stride=184320
    )
    assert not hla.is_npu_layer_major_blnhc(config("BLNHC", [ple_only]))


@pytest.mark.parametrize(
    "architecture,device,expect_direct",
    [
        # The gate is platform-wide: on CUDA every architecture falls back to
        # direct mode -- including the text-only variants and unknown models
        # that an architecture allowlist would silently miss.
        ("Qwen3_5ForConditionalGeneration", "cuda", True),
        ("Qwen3_5ForCausalLM", "cuda", True),
        ("Glm5NextForConditionalGeneration", "cuda", True),
        ("Glm5NextForCausalLM", "cuda", True),
        ("Qwen4ExpForConditionalGeneration", "cuda", True),
        ("Qwen4ExpForCausalLM", "cuda", True),
        ("KimiK3ForConditionalGeneration", "cuda", True),
        ("KimiLinearForCausalLM", "cuda", True),
        ("UnknownForConditionalGeneration", "cuda", True),
        ("Qwen3_5ForConditionalGeneration", "npu", False),
        ("Glm5NextForConditionalGeneration", "npu", False),
        ("Qwen4ExpForConditionalGeneration", "npu", False),
        ("KimiK3ForConditionalGeneration", "npu", False),
        ("KimiLinearForCausalLM", "npu", False),
        ("UnknownForConditionalGeneration", "npu", False),
    ],
)
def test_layerwise_platform_gate_selects_direct_fallback(
    hla, architecture, device, expect_direct
):
    hla.current_platform.device_type = device
    hla.current_platform.is_cuda_alike = lambda: device == "cuda"
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture])
        )
    )
    fallback_log = (
        hla.UCMHybridLinearAttentionLayerWiseConnector.layerwise_engine_config_error(
            config
        )
    )
    if expect_direct:
        assert fallback_log == (
            "Layerwise mode is not supported on "
            f"{device.upper()}; falling back to direct mode."
        )
    else:
        assert fallback_log is None


def test_checkpoint_windows_exclude_indexers_and_keep_mtp_and_tail(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    specs = {
        "linear0": MambaSpec(mamba_cache_mode="align"),
        "index0": MLAAttentionSpec(is_index_group_leader=False, tokens_per_state=4),
        "full0": MLAAttentionSpec(is_index_group_leader=True),
        "linear1": MambaSpec(mamba_cache_mode="align"),
        "index1": MLAAttentionSpec(is_index_group_leader=False, tokens_per_state=4),
        # Dense Kimi MLA leaves is_index_group_leader at its default False.
        "full1": MLAAttentionSpec(),
        "mtp_linear": MambaSpec(mamba_cache_mode="align"),
        "mtp_full": FullAttentionSpec(),
        "tail": MambaSpec(mamba_cache_mode="align"),
    }
    connector.layer_order = list(specs)
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[group(spec, name) for name, spec in specs.items()]
    )

    connector._build_checkpoint_windows()

    assert connector.checkpoint_windows == [
        ["linear0", "index0", "full0"],
        ["linear1", "index1", "full1"],
        ["mtp_linear", "mtp_full"],
        ["tail"],
    ]
    assert connector._checkpoint_to_window == {
        "full0": 0,
        "full1": 1,
        "mtp_full": 2,
    }


def test_checkpoint_windows_are_empty_for_a_linear_only_stage(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.layer_order = ["linear0", "linear1"]
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            group(MambaSpec(mamba_cache_mode="align"), name)
            for name in connector.layer_order
        ]
    )
    connector._build_checkpoint_windows()
    assert connector.checkpoint_windows == []
    assert connector._checkpoint_to_window == {}


def test_flash_next_layer_order_keeps_ple_before_checkpoint_and_mtp_last(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    names = ["full3", "linear2", "ple2", "mtp_full"]
    layout = SimpleNamespace(
        slot_shard_size=16,
        tensor_size_list=[16],
        num_slots=1,
        layer_anchors={name: object() for name in names},
        layer_name_to_id={
            "full3": 3,
            "linear2": 2,
            "ple2": 2,
            "mtp_full": 0,
        },
        layer_name_to_slot={name: 0 for name in names},
    )
    mtp_group = group(FullAttentionSpec(), "mtp_full")
    mtp_group.is_eagle_group = True
    connector._kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            group(MambaSpec(mamba_cache_mode="align"), "linear2", "ple2"),
            group(FullAttentionSpec(), "full3"),
            mtp_group,
        ]
    )

    def register_common(_kv_caches):
        connector.layer_to_group = {
            "linear2": 0,
            "ple2": 0,
            "full3": 1,
            "mtp_full": 2,
        }
        connector.layer_to_slot = layout.layer_name_to_slot
        return layout, None, None

    connector._register_common = register_common
    connector._create_store_bound = MagicMock()
    connector._build_checkpoint_windows = MagicMock()

    connector.register_kv_caches({})

    assert connector.layer_order == ["linear2", "ple2", "full3", "mtp_full"]


def test_layer_hook_waits_same_layer_ple_once(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.need_load = True
    connector._connector_metadata = object()
    connector.kv_cache_layout = SimpleNamespace(ple_layer_name="ple2")
    connector.layer_order = ["linear2", "ple2"]
    connector.layer_pos = {"linear2": 0, "ple2": 1}
    connector.layer_name_to_id = {"linear2": 2, "ple2": 2}
    connector.layer_to_group = {"linear2": 0, "ple2": 1}
    connector.layer_to_slot = {"linear2": 0, "ple2": 0}
    connector._waited_layers = set()
    connector._waited_group_slots = set()
    events = []
    submitted = set()

    def submit(pos):
        if 0 <= pos < len(connector.layer_order) and pos not in submitted:
            submitted.add(pos)
            events.append(("submit", pos))

    connector._submit_layer_load = submit
    connector._wait_layer_load_pair = lambda pair: events.append(("wait", pair))

    connector.wait_for_layer_load("linear2")
    connector.wait_for_layer_load("linear2")

    assert events == [("wait", (0, 0)), ("submit", 1), ("wait", (1, 0))]
    assert connector._waited_layers == {"linear2", "ple2"}


def _start_load_connector(hla, windows):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.need_load = True
    connector.layer_order = [name for window in windows for name in window]
    connector.checkpoint_windows = windows
    connector.kv_cache_layout = SimpleNamespace(ple_layer_name=None)
    connector.device = SimpleNamespace(synchronize=MagicMock())
    connector._reset_transfer_state = MagicMock()
    connector._prepare_load_plan = MagicMock()
    connector._get_connector_metadata = MagicMock(
        return_value=hla.UCMConnectorMetadata(request_meta={})
    )
    connector._record_load_bytes = MagicMock()
    return connector


def test_ascend_start_load_prefetches_two_layers(hla):
    connector = _start_load_connector(hla, [["l0"], ["l1"], ["l2"], ["l3"]])
    events = []
    connector._submit_layer_load = lambda pos: events.append(("submit", pos))

    connector.start_load_kv(SimpleNamespace())

    assert events == [("submit", 0), ("submit", 1)]
    connector._record_load_bytes.assert_not_called()


def test_ascend_layer_load_pipelines_each_layer(hla):
    windows = [
        ["linear0", "full0"],
        ["linear1", "full1"],
        ["tail_linear"],
    ]
    connector = _start_load_connector(hla, windows)
    connector._connector_metadata = object()
    connector.layer_pos = {name: pos for pos, name in enumerate(connector.layer_order)}
    connector.layer_to_group = {name: 0 for name in connector.layer_order}
    connector.layer_to_slot = connector.layer_pos
    connector._submitted_positions = set()
    connector._submitted_group_slots = set()
    connector._waited_group_slots = set()
    connector._waited_layers = set()
    events = []

    def submit(pos):
        if (
            0 <= pos < len(connector.layer_order)
            and pos not in connector._submitted_positions
        ):
            connector._submitted_positions.add(pos)
            events.append(("submit", pos))

    connector._submit_layer_load = submit
    connector._wait_layer_load_pair = lambda pair: events.append(("wait", pair[1]))
    connector._record_load_bytes = lambda: events.append(("record",))

    connector.start_load_kv(SimpleNamespace())
    assert events == [("submit", 0), ("submit", 1)]

    connector.wait_for_layer_load("linear0")
    connector.wait_for_layer_load("linear0")
    assert events[2:] == [("wait", 0), ("submit", 2)]
    in_flight = {
        pos
        for pos in connector._submitted_positions
        if (0, pos) not in connector._waited_group_slots
    }
    assert in_flight == {1, 2}

    connector.wait_for_layer_load("full0")
    connector.wait_for_layer_load("linear1")
    connector.wait_for_layer_load("full1")
    connector.wait_for_layer_load("tail_linear")
    assert events == [
        ("submit", 0),
        ("submit", 1),
        ("wait", 0),
        ("submit", 2),
        ("wait", 1),
        ("submit", 3),
        ("wait", 2),
        ("submit", 4),
        ("wait", 3),
        ("wait", 4),
        ("record",),
    ]


def test_wait_for_save_submits_tail_without_checkpoint(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.need_load = False
    connector._connector_metadata = object()
    connector.layer_order = ["linear0", "full0", "tail0", "tail1"]
    connector.checkpoint_windows = [["linear0", "full0"], ["tail0", "tail1"]]
    connector._checkpoint_to_window = {"full0": 0}
    connector._dumped_layers = {"linear0", "full0"}
    connector._deferred_dump_layers = set()
    connector._dump_plan = None
    connector._dumped_group_slots = set()
    connector._submitted_group_slots = set()
    connector._waited_group_slots = set()
    connector._dump_tasks = []
    connector.is_save = False
    connector._ensure_dump_plan = MagicMock()
    dumped = []
    connector._submit_layer_dump = lambda name: dumped.append(name)
    connector._submit_unowned_slot_dumps = MagicMock()
    connector._submit_commit_dump = MagicMock()
    connector._wait_dump_tasks = MagicMock()
    connector._record_load_bytes = MagicMock()

    connector.wait_for_save()

    assert dumped == ["tail0", "tail1"]
    connector._wait_dump_tasks.assert_called_once_with()


def test_checkpoint_save_batches_window_and_defers_commit_layers(hla):
    connector = hla.UCMHybridLinearAttentionLayerWiseConnector()
    connector.checkpoint_windows = [
        ["linear0", "index0", "full0"],
        ["linear1", "full1"],
    ]
    connector._checkpoint_to_window = {"full0": 0, "full1": 1}
    connector._connector_metadata = object()
    connector._dumped_layers = set()
    connector._deferred_dump_layers = {"full0"}
    dumped = []

    def submit(name):
        dumped.append(name)
        connector._dumped_layers.add(name)

    connector._submit_layer_dump = submit
    connector.save_kv_layer("linear0", None, None)
    assert dumped == []

    connector.save_kv_layer("full0", None, None)
    connector.save_kv_layer("full0", None, None)
    assert dumped == ["linear0", "index0"]
    assert "full0" not in connector._dumped_layers

    connector.save_kv_layer("full1", None, None)
    connector.save_kv_layer("full1", None, None)
    assert dumped == ["linear0", "index0", "linear1", "full1"]


# ---------------------------------------------------------------------------
# Pure-PCP scoping: _scope_blocks dump grid and replica dedup.
#
# PCP policy (see UCMHybridLinearAttentionConnector._scope_blocks_pcp): no
# hash changes; replicated content spreads across the PCP axis (MLA slice
# grid widens to tp_size x pcp_world_size, each TP position's KDA / non-MLA
# shard dump spreads across its PCP ranks), except TP=1 where every PCP
# rank stores its full KDA section under a pcp_rank namespace; loads are
# never sliced except for that TP=1 namespace.
# ---------------------------------------------------------------------------


def _cp_scope_connector(hla, tp_rank, tp_size, pcp_rank, pcp_world_size, is_mla=True):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector.tp_rank = tp_rank
    connector.tp_size = tp_size
    connector.is_mla = is_mla
    connector.pcp_rank = pcp_rank
    connector.pcp_world_size = pcp_world_size
    connector.request_hasher = lambda key: b"rk:" + key
    return connector


def _cp_scope_ids(n_mla=12, n_kda=3):
    mla = [bytes([i]) for i in range(n_mla)]
    kda = [bytes([0x80 + i]) for i in range(n_kda)]
    return mla + kda, list(range(n_mla + n_kda)), [0] * n_mla + [1] * n_kda, n_mla


def test_scope_blocks_pcp_dump_slices_partition_the_grid(hla):
    """MLA dump slices over the tp x pcp grid are disjoint and complete."""
    ucm, vllm, groups, n = _cp_scope_ids()
    mla_ids = set(ucm[:n])
    seen = set()
    for pcp_rank in range(2):
        for tp_rank in range(4):
            connector = _cp_scope_connector(hla, tp_rank, 4, pcp_rank, 2)
            grid = 4 * 2
            g = pcp_rank * 4 + tp_rank
            expected = ucm[n * g // grid : n * (g + 1) // grid]
            _, scoped, _, _ = connector._scope_blocks(ucm, vllm, groups, n, True)
            rank_mla = [key for key in scoped if key in mla_ids]
            assert rank_mla == expected
            assert not seen & set(rank_mla)
            seen.update(rank_mla)
    assert seen == mla_ids


def test_scope_blocks_pcp_spreads_kda_across_pcp_ranks(hla):
    """KDA shards spread across a TP position's PCP ranks; union = one copy."""
    ucm, vllm, groups, n = _cp_scope_ids()
    for tp_rank in (0, 1):
        seen_scoped = set()
        seen_context = set()
        for pcp_rank in (0, 1):
            connector = _cp_scope_connector(hla, tp_rank, 4, pcp_rank, 2)
            context, scoped, scoped_vllm, scoped_groups = connector._scope_blocks(
                ucm, vllm, groups, n, True
            )
            grid = 4 * 2
            g = pcp_rank * 4 + tp_rank
            mla_len = n * (g + 1) // grid - n * g // grid
            kda_raw = ucm[n:]
            start = len(kda_raw) * pcp_rank // 2
            end = len(kda_raw) * (pcp_rank + 1) // 2
            expected_raw = kda_raw[start:end]
            assert context[mla_len:] == expected_raw
            if tp_rank == 0:
                assert scoped[mla_len:] == expected_raw
            else:
                assert scoped[mla_len:] == [b"rk:" + key for key in expected_raw]
            assert scoped_vllm[mla_len:] == vllm[n:][start:end]
            assert scoped_groups[mla_len:] == groups[n:][start:end]
            seen_scoped.update(scoped[mla_len:])
            seen_context.update(context[mla_len:])
        if tp_rank == 0:
            assert seen_context == set(ucm[n:])
        else:
            assert seen_scoped == {b"rk:" + key for key in ucm[n:]}


def test_scope_blocks_pcp_loads_are_never_sliced(hla):
    """Every PCP rank loads the full MLA set plus its own KDA namespace."""
    ucm, vllm, groups, n = _cp_scope_ids()
    for pcp_rank in (0, 1):
        connector = _cp_scope_connector(hla, 1, 4, pcp_rank, 2)
        _, scoped, scoped_vllm, scoped_groups = connector._scope_blocks(
            ucm, vllm, groups, n, False
        )
        assert scoped[:n] == ucm[:n]
        assert scoped[n:] == [b"rk:" + key for key in ucm[n:]]
        assert scoped_vllm == vllm
        assert scoped_groups == groups


def test_scope_blocks_without_pcp_matches_tp_only_scoping(hla):
    """pcp_world_size=1 keeps the historical TP-only behavior."""
    ucm, vllm, groups, n = _cp_scope_ids()
    connector = _cp_scope_connector(hla, 1, 4, 0, 1)
    _, scoped, _, _ = connector._scope_blocks(ucm, vllm, groups, n, True)
    start, end = n * 1 // 4, n * 2 // 4
    assert scoped[: end - start] == ucm[start:end]
    assert scoped[end - start :] == [b"rk:" + key for key in ucm[n:]]


def test_scope_blocks_non_mla_pcp_spreads_dump_across_pcp_ranks(hla):
    ucm = [b"a", b"b"]
    vllm, groups = [11, 22], [0, 0]
    scoped_by_pcp = []
    for pcp_rank in (0, 1):
        connector = _cp_scope_connector(hla, 1, 4, pcp_rank, 2, is_mla=False)
        context, scoped, scoped_vllm, scoped_groups = connector._scope_blocks(
            ucm, vllm, groups, 0, True
        )
        expected = ucm[pcp_rank : pcp_rank + 1]
        assert context == expected
        assert scoped == [b"rk:" + key for key in expected]
        assert scoped_vllm == vllm[pcp_rank : pcp_rank + 1]
        assert scoped_groups == groups[pcp_rank : pcp_rank + 1]
        scoped_by_pcp.extend(scoped)
    assert scoped_by_pcp == [b"rk:" + b"a", b"rk:" + b"b"]
    for pcp_rank in (0, 1):
        connector = _cp_scope_connector(hla, 1, 4, pcp_rank, 2, is_mla=False)
        _, scoped_load, _, _ = connector._scope_blocks(ucm, vllm, groups, 0, False)
        assert scoped_load == [b"rk:" + key for key in ucm]


def test_scope_blocks_tp1_pcp_kda_full_section_per_rank_namespace(hla):
    """TP=1+PCP: every PCP rank stores its FULL KDA section under a
    pcp_rank namespace on both dump and load -- self-consistent for any
    content distribution (sharded or replicated); MLA still spreads
    across the pcp grid."""
    ucm, vllm, groups, n = _cp_scope_ids()
    mla_ids = set(ucm[:n])
    seen_mla = set()
    for pcp_rank in range(3):
        prefix = b"shard%d:" % pcp_rank
        connector = hla.UCMHybridLinearAttentionConnector()
        connector.tp_rank = 0
        connector.tp_size = 1
        connector.is_mla = True
        connector.pcp_rank = pcp_rank
        connector.pcp_world_size = 3
        connector._pcp_shard_hasher = lambda key, p=prefix: p + key
        connector.request_hasher = lambda key: b"rk:" + key
        expected_kda = (
            list(ucm[n:]) if pcp_rank == 0 else [prefix + key for key in ucm[n:]]
        )
        # dump: MLA slice across the pcp grid; KDA full under its namespace.
        context, scoped, scoped_vllm, scoped_groups = connector._scope_blocks(
            ucm, vllm, groups, n, True
        )
        mla_len = n * (pcp_rank + 1) // 3 - n * pcp_rank // 3
        assert scoped[:mla_len] == ucm[n * pcp_rank // 3 : n * (pcp_rank + 1) // 3]
        assert context[mla_len:] == ucm[n:]
        assert scoped[mla_len:] == expected_kda
        assert scoped_vllm[mla_len:] == vllm[n:]
        assert scoped_groups[mla_len:] == groups[n:]
        seen_mla.update(scoped[:mla_len])
        # load: full MLA set + this rank's namespace KDA section.
        _, scoped_load, scoped_vllm_load, _ = connector._scope_blocks(
            ucm, vllm, groups, n, False
        )
        assert scoped_load[:n] == ucm[:n]
        assert scoped_load[n:] == expected_kda
        assert scoped_vllm_load == vllm
    assert seen_mla == mla_ids


def test_scope_blocks_non_mla_tp1_pcp_splits_gqa_and_gdn(hla):
    """TP=1+PCP non-MLA: GQA stays shared (spread on dump); GDN state takes
    the pcp_rank namespace with a full section per rank, dump and load."""
    ucm = [b"a", b"b", b"s0", b"s1", b"s2"]
    vllm = [10, 11, 20, 21, 22]
    groups = [0, 0, 1, 1, 1]
    for pcp_rank in range(2):
        prefix = b"shard%d:" % pcp_rank
        connector = hla.UCMHybridLinearAttentionConnector()
        connector.tp_rank = 0
        connector.tp_size = 1
        connector.is_mla = False
        connector.pcp_rank = pcp_rank
        connector.pcp_world_size = 2
        connector._state_group_ids = {1}
        connector._pcp_shard_hasher = lambda key, p=prefix: p + key
        connector.request_hasher = lambda key: b"rk:" + key
        expected_gdn = (
            list(ucm[2:]) if pcp_rank == 0 else [prefix + key for key in ucm[2:]]
        )
        # dump: GQA spread (1 of 2 blocks each), GDN full under the namespace.
        context, scoped, scoped_vllm, scoped_groups = connector._scope_blocks(
            ucm, vllm, groups, 0, True
        )
        assert scoped == ucm[pcp_rank : pcp_rank + 1] + expected_gdn
        assert context == ucm[pcp_rank : pcp_rank + 1] + ucm[2:]
        assert scoped_vllm == vllm[pcp_rank : pcp_rank + 1] + vllm[2:]
        assert scoped_groups == [0, 1, 1, 1]
        # load: full GQA set + this rank's namespaced GDN section.
        _, scoped_load, scoped_vllm_load, _ = connector._scope_blocks(
            ucm, vllm, groups, 0, False
        )
        assert scoped_load == ucm[:2] + expected_gdn
        assert scoped_vllm_load == vllm


def _cp_vllm_config(tp=1, pcp=1, dcp=1):
    return SimpleNamespace(
        model_config=SimpleNamespace(model="test-model", dtype="bfloat16"),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
        ),
        speculative_config=None,
        additional_config={},
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=SimpleNamespace(),
    )


def test_build_kv_hash_meta_has_no_pcp_field(hla):
    """build_kv_hash_meta stays CP-free: non-hybrid models never see the
    suffix, and the tp field already separates tp1+pcp2 from tp2+pcp1."""
    tp1_meta = hla.build_kv_hash_meta(_cp_vllm_config(pcp=2), 0)
    tp2_meta = hla.build_kv_hash_meta(_cp_vllm_config(tp=2), 0)
    assert "pcp" not in tp1_meta
    assert tp1_meta != tp2_meta


class _FakeMetaHasher:
    def __init__(self, meta):
        self.meta_bytes = meta.encode()
        self.seed = b"seed:" + self.meta_bytes


def _patch_hasher_modules(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "ucm.integration.vllm.request_hasher",
        SimpleNamespace(RequestHasher=_FakeMetaHasher),
    )
    monkeypatch.setitem(
        sys.modules,
        "ucm.integration.vllm.ucm_connector",
        SimpleNamespace(build_kv_hash_meta=lambda cfg, rank: f"meta:{rank}"),
    )


def test_setup_cp_scoping_scheduler_tp1_rebuilds_hasher_with_suffix(hla, monkeypatch):
    """At TP=1+PCP the scheduler's own hasher carries the suffix: rank 0's
    shard keys ARE the scheduler's chain and state hashes."""
    _patch_hasher_modules(monkeypatch)
    connector = hla.UCMHybridLinearAttentionConnector()
    connector._role = hla.KVConnectorRole.SCHEDULER
    connector._setup_cp_scoping(_cp_vllm_config(pcp=2))
    assert connector.request_hasher.meta_bytes == b"meta:0:pcp=2"
    assert connector._seed == b"seed:meta:0:pcp=2"
    assert connector.pcp_world_size == 2
    assert connector.pcp_rank == 0  # scheduler never touches the group
    assert connector._pcp_shard_hasher is None


def test_setup_cp_scoping_scheduler_never_touches_the_pcp_group(hla):
    """The scheduler process does not initialize the PCP group under the
    multiprocess executor: the pcp size must come from parallel_config.
    TP>1 needs no hasher rebuild at all."""
    connector = hla.UCMHybridLinearAttentionConnector()
    connector._role = hla.KVConnectorRole.SCHEDULER
    connector._setup_cp_scoping(_cp_vllm_config(tp=8, pcp=2))
    assert connector.pcp_world_size == 2
    assert connector.pcp_rank == 0
    assert connector._pcp_shard_hasher is None


def test_setup_cp_scoping_dcp_guard_fires_before_the_pcp_early_return(hla):
    """pcp=1 + dcp>1 must raise instead of silently returning."""
    connector = hla.UCMHybridLinearAttentionConnector()
    connector._role = hla.KVConnectorRole.SCHEDULER
    with pytest.raises(ValueError, match="pure PCP only"):
        connector._setup_cp_scoping(_cp_vllm_config(dcp=2))


def test_setup_cp_scoping_pcp1_scheduler_is_a_noop(hla):
    connector = hla.UCMHybridLinearAttentionConnector()
    connector._role = hla.KVConnectorRole.SCHEDULER
    connector._setup_cp_scoping(_cp_vllm_config())
    assert connector.pcp_world_size == 1
    assert connector.pcp_rank == 0
    assert connector._pcp_shard_hasher is None


def test_make_other_rank_hashers_covers_pcp_shards_at_tp1(hla, monkeypatch):
    """State shards live per-TP at TP>1 and per-pcp-shard at TP=1+PCP; the
    base is_mla shortcut no longer skips hybrid KDA namespaces."""
    _patch_hasher_modules(monkeypatch)
    connector = hla.UCMHybridLinearAttentionConnector()
    tp_hashers = connector._make_other_rank_hashers(_cp_vllm_config(tp=4))
    assert [hasher.meta_bytes for hasher in tp_hashers] == [
        b"meta:1",
        b"meta:2",
        b"meta:3",
    ]
    pcp_hashers = connector._make_other_rank_hashers(_cp_vllm_config(pcp=3))
    assert [hasher.meta_bytes for hasher in pcp_hashers] == [
        b"meta:1:pcp=3",
        b"meta:2:pcp=3",
    ]
    assert connector._make_other_rank_hashers(_cp_vllm_config()) == []
