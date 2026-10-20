"""CPU coverage of production Decode planners and FAWA Store dispatch.

Load the selected production symbols through AST, as the geometry tests do:
this suite must also run without vLLM, torch or the native UCM extension.
Device DMA and engine hook ordering still require the documented NPU tests.
"""

import ast
import copy
import ctypes
import importlib.util
import io
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Tuple
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "ucm/integration/vllm"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HASH = load_module("decode_test_hasher", SOURCE / "request_hasher.py")
LAYOUT = load_module("decode_test_layout", SOURCE / "replicated_mla_layout.py")


def load_symbols():
    namespace = dict(
        __name__="decode_test_symbols",
        dataclass=dataclass,
        field=field,
        Any=Any,
        Optional=Optional,
        np=np,
        json=json,
        logger=logging.getLogger("decode-writeback-test"),
        IncrementalRequestHasher=HASH.IncrementalRequestHasher,
        KVConnectorMetadata=object,
        KVConnectorWorkerMetadata=object,
        SchedulerOutput=object,
        Task=object,
        UcmKVStoreBaseV1=object,
        ReplicatedMLAStoreLayout=LAYOUT.ReplicatedMLAStoreLayout,
        RequestHasher=HASH.RequestHasher,
        math=math,
        time=time,
        List=list,
        Tuple=Tuple,
        Sequence=Sequence,
        build_namespace=LAYOUT.build_namespace,
        supports_replicated_mla=LAYOUT.supports_replicated_mla,
        target_layers=LAYOUT.target_layers,
        current_platform=SimpleNamespace(device_type="npu"),
        request_hasher_module=SimpleNamespace(
            _layout_source_digest=lambda: "test-source"
        ),
        extract_layer_index=lambda name: int(re.search(r"layers\.(\d+)", name)[1]),
        torch=SimpleNamespace(Tensor=ByteTensor),
        ucmmetrics=SimpleNamespace(update_stats=Mock()),
    )
    for filename, classes, methods in (
        ("decode_writeback.py", None, {}),
        (
            "ucm_connector.py",
            {
                "RequestMeta",
                "RequestDispatchMeta",
                "UCMConnectorMetadata",
                "UCMWorkerMetadata",
                "PendingDumpTask",
                "build_kv_hash_meta",
                "KVCacheTensorInfo",
                "KVCacheLayout",
            },
            {
                "UCMDirectConnector": {
                    "on_new_request",
                    "_register_writeback_request",
                    "_ensure_writeback_meta",
                    "_debug_keys",
                    "_prepare_writeback_dispatch",
                    "_begin_writeback_jobs",
                    "_complete_writeback_jobs",
                    "_missing_store_rows",
                    "_generate_dispatch_meta",
                    "build_connector_meta",
                    "_track_async_dump_requests",
                    "wait_for_save",
                    "_direct_ptrs",
                    "_wait_pending_dump_task",
                    "_release_dump_event_handle",
                    "_non_negative_ms",
                    "_resolve_writeback_blocks",
                    "_writeback_window_rows",
                }
            },
        ),
        (
            "hma_connector.py",
            {
                "FAWARequestMeta",
                "FAWARequestDispatchMeta",
                "FAWADumpTask",
                "UCMFAWAConnectorMetadata",
                "KVCacheGroupMeta",
                "KVCacheGroupLayout",
                "_build_v41_hash_meta",
            },
            {
                "UCMFAWAConnector": {
                    "_slice_group_block_ids",
                    "_generate_dispatch_meta",
                    "build_connector_meta",
                    "_save_writeback_requests",
                    "_submit_writeback_part",
                    "_submit_dump_task",
                    "_extract_fa_ptr",
                    "_extract_wa_ptr",
                }
            },
        ),
    ):
        tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
        nodes = []
        for node in tree.body:
            name = getattr(node, "name", None)
            if classes is None:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    nodes.append(node)
            elif name in classes:
                nodes.append(node)
            elif name in methods:
                node = copy.deepcopy(node)
                node.bases = (
                    [ast.Name(id="UCMDirectConnector", ctx=ast.Load())]
                    if name == "UCMFAWAConnector"
                    else []
                )
                node.body = [
                    m for m in node.body if getattr(m, "name", None) in methods[name]
                ]
                nodes.append(node)
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, str(SOURCE / filename), "exec"), namespace)
    return namespace


class ByteTensor:
    """Real CPU byte backing with the small tensor-view API layouts require."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def dim(self):
        return self.array.ndim

    def data_ptr(self):
        return self.array.ctypes.data

    def element_size(self):
        return self.array.itemsize

    def stride(self, index=None):
        strides = tuple(value // self.element_size() for value in self.array.strides)
        return strides if index is None else strides[index]

    def __getitem__(self, index):
        return ByteTensor(self.array[index])


SYMS = load_symbols()
State = SYMS["DecodeWritebackState"]
Result = SYMS["WritebackResult"]
Direct = SYMS["UCMDirectConnector"]
FAWA = SYMS["UCMFAWAConnector"]


def request(ids, computed=0):
    return SimpleNamespace(
        request_id="req",
        all_token_ids=list(ids),
        num_computed_tokens=computed,
        num_tokens=len(ids),
    )


def connector(role="kv_consumer", cls=Direct, size=4, tp=1, enabled=True):
    obj = cls()
    obj.save_decode_cache = enabled
    obj.shared_mla = True
    obj._writeback_states = {}
    obj._writeback_generation = obj._next_writeback_job = 0
    obj._writeback_results = {}
    obj.requests_meta = {}
    obj.hash_block_size = obj.block_size = size
    obj.cp_world_size = 1
    obj.tp_size, obj.tp_rank = tp, 0
    obj._vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_role=role))
    obj.persist_token_threshold = 0
    obj.request_hasher = HASH.RequestHasher("same-target-schema")
    obj._seed = obj.request_hasher.seed
    obj.request_block_hasher = obj.request_hasher.make_request_block_hasher(size)
    obj._async_load_req_ids = obj._async_dump_req_ids = set()
    obj._pending_async_load_dispatches = {}
    if cls is FAWA:
        obj.group_metas = {
            0: SYMS["KVCacheGroupMeta"](0, size, 1, size),
            1: SYMS["KVCacheGroupMeta"](1, size, 1, size),
        }
        obj.fa_group_ids, obj.window_group_ids = [0], [1]
        obj.transient_group_ids = set()
        obj.prefix_replay_tokens = 0
        obj.wa_dump_block_wise = True
    return obj


def step(obj, req, scheduled, *, groups=None, new=False, resumed=False):
    factory = SYMS["FAWARequestMeta"] if isinstance(obj, FAWA) else SYMS["RequestMeta"]
    obj.on_new_request(req)
    obj._ensure_writeback_meta(req.request_id, factory)
    cached = SimpleNamespace(
        req_ids=[] if new else [req.request_id],
        new_block_ids=[groups],
        resumed_from_preemption=[resumed],
        num_computed_tokens=[req.num_computed_tokens],
    )
    output = SimpleNamespace(
        scheduled_new_reqs=(
            [
                SimpleNamespace(
                    req_id=req.request_id,
                    block_ids=groups,
                    num_computed_tokens=req.num_computed_tokens,
                    prefill_token_ids=list(req.all_token_ids),
                )
            ]
            if new
            else []
        ),
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={req.request_id: scheduled},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    return obj.build_connector_meta(output).request_meta[req.request_id]


def complete(obj, job, failed=()):
    state = obj._writeback_states[job.request_id]
    return state.complete_save(Result(job, set(range(obj.tp_size)), set(failed)))


@pytest.mark.parametrize("size", [1, 4, 128])
def test_incremental_hash_matches_full_prefill(size):
    req = request(range(size * 4 + 3))
    hasher = HASH.RequestHasher("stable")
    incremental = HASH.IncrementalRequestHasher(hasher, size)
    for end in range(1, len(req.all_token_ids) + 1):
        expected = hasher.make_request_block_hasher(size)(
            request(req.all_token_ids[:end])
        )
        assert incremental.update(req, end) == expected
    assert incremental.update(req, size) == hasher.make_request_block_hasher(size)(
        request(range(size))
    )


@pytest.mark.parametrize("cls", [Direct, FAWA])
@pytest.mark.parametrize("role", ["kv_consumer", "kv_producer", "kv_both"])
def test_no_new_block_progresses_and_next_turn_hash_matches(cls, role):
    obj = connector(role, cls)
    req = request(range(5))
    dispatch = step(obj, req, 5, groups=([11, 12, 13], [21, 22, 23]), new=True)
    assert dispatch.save_job is not None and dispatch.save_job.phase == "prefill"
    assert complete(obj, dispatch.save_job)
    for count in range(6, 9):
        req.all_token_ids.append(count)
        req.num_computed_tokens = count - 1
        dispatch = step(obj, req, 1, groups=None)
        if count < 8 and dispatch.save_job is not None:
            complete(obj, dispatch.save_job)
    job = dispatch.save_job
    assert job is not None and job.phase == "decode" and job.end == 8
    assert complete(obj, job)
    # A later Prefill includes actual generated tokens plus a new user turn.
    next_turn = request(req.all_token_ids + [100, 101, 102])
    lookup = obj.request_block_hasher(next_turn)
    assert set(job.keys).issubset(lookup)
    assert obj._writeback_states["req"].computed_end == 8


@pytest.mark.parametrize("cls", [Direct, FAWA])
def test_consumer_failover_prefill_saves_only_computed_complete_units(cls):
    # Simulate the local compute path after external routing hands the request
    # to a consumer. This does not exercise a real router or failure detector.
    obj = connector("kv_consumer", cls)
    req = request(range(8))
    initial = step(obj, req, 2, groups=([11, 12, 13], [21, 22, 23]), new=True)
    assert initial.save_job is None  # allocated blocks are not computed KV
    req.num_computed_tokens = 2
    first = step(obj, req, 2)
    assert first.save_job.phase == "prefill"
    assert (first.save_job.start, first.save_job.end) == (0, 4)
    assert list(first.save_job.keys) == obj.request_block_hasher(req)[:1]
    assert complete(obj, first.save_job)
    req.num_computed_tokens = 4
    second = step(obj, req, 4)
    assert second.save_job.phase == "prefill"
    assert (second.save_job.start, second.save_job.end) == (4, 8)
    assert complete(obj, second.save_job)
    req.all_token_ids.extend(range(8, 12))
    req.num_computed_tokens = 8
    decoded = step(obj, req, 4)
    assert decoded.save_job.phase == "decode"
    assert (decoded.save_job.start, decoded.save_job.end) == (8, 12)
    assert complete(obj, decoded.save_job)
    next_turn = request(req.all_token_ids + [100, 101])
    assert obj._writeback_states["req"].saved == set(
        obj.request_block_hasher(next_turn)
    )


def test_rejected_speculative_tokens_not_hashed_or_saved():
    obj = connector()
    req = request(range(5))
    initial = step(obj, req, 5, groups=([10, 11, 12],), new=True)
    assert complete(obj, initial.save_job)
    req.all_token_ids.extend([5, 6, 7, 8])  # committed, excluding draft candidates
    req.num_computed_tokens = 5
    dispatch = step(obj, req, 7)  # schedules beyond the committed length
    job = dispatch.save_job
    assert job.end == 8
    assert list(job.keys) == obj.request_block_hasher(request(range(8)))[1:]
    assert obj._writeback_states["req"].computed_end == 9


def test_failed_or_partial_rank_completion_retries_without_advancing():
    obj = connector(tp=2)
    req = request(range(8))
    state = obj._register_writeback_request(req)
    hashes, _ = state.refresh_request(8)
    job = state.plan_save(hashes, 1, [0, 1], "decode")
    assert not state.complete_save(Result(job, {0}, set()))
    assert not state.saved
    assert not state.complete_save(Result(job, {1}, {1}))
    assert not state.pending and not state.saved
    retry = state.plan_save(hashes, 2, [0, 1], "decode")
    assert retry.keys == job.keys
    assert state.complete_save(Result(retry, {0, 1}, set()))
    assert state.saved == set(hashes)


@pytest.mark.parametrize("mrv2", [False, True])
def test_recovery_records_full_reprefill_range_and_ignores_old_generation(mrv2):
    obj = connector()
    req = request(range(8))
    state = obj._register_writeback_request(req)
    hashes, _ = state.refresh_request(8)
    old = state.plan_save(hashes, 1, [0], "decode")
    req.all_token_ids.extend(range(8, 16))
    new_data = SimpleNamespace(prefill_token_ids=list(range(16))) if mrv2 else None
    recovered = obj._register_writeback_request(req, new_data)
    assert recovered.prefill_end == 16
    assert not recovered.complete_save(Result(old, {0}, set()))
    req.num_computed_tokens = 8
    _, decoding = recovered.refresh_request(4)
    assert not decoding
    req.num_computed_tokens = 16
    assert recovered.refresh_request(1)[1]


def test_window_snapshot_uses_completed_pages_not_the_current_partial_page():
    obj = connector(cls=FAWA)
    req = request(range(5))
    initial = step(obj, req, 5, groups=([11, 12, 13], [21, 22, 23]), new=True)
    assert complete(obj, initial.save_job)
    req.all_token_ids.extend(range(5, 10))
    req.num_computed_tokens = 5
    dispatch = step(obj, req, 4)
    assert obj._writeback_states["req"].computed_end == 9
    assert dispatch.save_job.end == 8
    assert dispatch.dump_vllm_block_ids[1] == [22]
    complete(obj, dispatch.save_job)
    req.num_computed_tokens = 9
    req.all_token_ids.extend([10, 11])
    dispatch = step(obj, req, 3)
    assert dispatch.save_job.end == 12
    assert dispatch.dump_wa_block_wise is False
    assert dispatch.dump_vllm_block_ids[1] == [23]


def test_window_snapshot_skips_when_small_state_pages_may_have_expired():
    obj = connector(cls=FAWA)
    obj.group_metas[1] = SYMS["KVCacheGroupMeta"](1, 2, 1, 2)
    req = request(range(5))
    initial = step(
        obj, req, 5, groups=([11, 12, 13], [21, 22, 23, 24, 25, 26]), new=True
    )
    assert complete(obj, initial.save_job)
    req.all_token_ids.extend(range(5, 12))
    req.num_computed_tokens = 5
    assert step(obj, req, 6).save_job is None  # remainder 3 >= state page 2
    req.num_computed_tokens = 11
    assert step(obj, req, 1).save_job.end == 12


def test_snapshot_guard_accounts_for_writes_of_uncommitted_mtp_candidates():
    obj = connector(cls=FAWA)
    req = request(range(5))
    initial = step(obj, req, 5, groups=([11, 12, 13], [21, 22, 23]), new=True)
    assert complete(obj, initial.save_job)
    req.all_token_ids.extend(range(5, 8))
    req.num_computed_tokens = 7
    # Committed key ends at 8, but forward will write through position 12.
    # The key must not authorize a snapshot from uncommitted future pages.
    assert step(obj, req, 6).save_job is None
    state = obj._writeback_states["req"]
    assert state.computed_end == 8 and state.scheduled_end == 13


def test_expired_window_source_does_not_publish_or_leave_a_pending_job():
    obj = connector(cls=FAWA)
    req = request(range(5))
    initial = step(obj, req, 5, groups=([11, 12], [21, 0]), new=True)
    assert complete(obj, initial.save_job)
    req.all_token_ids.extend(range(5, 8))
    req.num_computed_tokens = 7
    dispatch = step(obj, req, 1)
    assert dispatch.save_job is None and not dispatch.dump_keys
    assert not obj._writeback_states["req"].pending


@pytest.mark.parametrize("cls", [Direct, FAWA])
@pytest.mark.parametrize("mrv2", [False, True])
def test_resumed_scheduled_request_saves_reprefill_and_resets_generation(cls, mrv2):
    obj = connector(cls=cls)
    req = request(range(5))
    initial = step(obj, req, 5, groups=([11, 12], [21, 22]), new=True)
    assert complete(obj, initial.save_job)
    old = obj._writeback_states["req"]
    req.all_token_ids.extend(range(5, 8))
    req.num_computed_tokens = 7
    pending = step(obj, req, 1).save_job
    req.num_computed_tokens = 0
    resumed = step(obj, req, 4, groups=([31, 32], [41, 42]), new=mrv2, resumed=not mrv2)
    current = obj._writeback_states["req"]
    assert current.generation > old.generation
    assert current.prefill_end == 8
    assert resumed.save_job is not None and resumed.save_job.phase == "prefill"
    assert (resumed.save_job.start, resumed.save_job.end) == (0, 4)
    assert not current.complete_save(Result(pending, {0}, set()))
    assert current.blocks == [[31, 32], [41, 42]]
    assert complete(obj, resumed.save_job)
    # Continue re-prefilling generated history without allocating another block.
    req.num_computed_tokens = 4
    tail = step(obj, req, 4)
    assert tail.save_job.phase == "prefill"
    assert (tail.save_job.start, tail.save_job.end) == (4, 8)
    assert list(tail.save_job.keys) == obj.request_block_hasher(req)[1:]
    assert complete(obj, tail.save_job)
    req.all_token_ids.extend(range(8, 12))
    req.num_computed_tokens = 8
    decoded = step(obj, req, 4, groups=([33], [43]))
    assert decoded.save_job.phase == "decode" and decoded.save_job.end == 12


@pytest.mark.parametrize("cls", [Direct, FAWA])
def test_disabled_keeps_original_prompt_only_dispatch(cls):
    obj = connector(cls=cls, enabled=False)
    req = request(range(5))
    obj.on_new_request(req)
    factory = SYMS["FAWARequestMeta"] if cls is FAWA else SYMS["RequestMeta"]
    obj.requests_meta["req"] = factory(
        ucm_block_ids=obj.request_block_hasher(req), num_token_ids=5
    )
    step(obj, req, 5, groups=([11, 12], [21, 22]), new=True)
    req.all_token_ids.extend(range(5, 8))
    req.num_computed_tokens = 7
    dispatch = step(obj, req, 1)
    assert dispatch.save_job is None
    keys = dispatch.dump_keys if cls is FAWA else dispatch.dump_block_ids[0]
    assert not keys


def config(tp=1, mtp=False, model="glm_moe_dsa"):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/weights/target",
            revision="same-weights",
            dtype="bf16",
            quantization=None,
            hf_text_config=SimpleNamespace(
                model_type=model,
                num_hidden_layers=2,
                compress_ratios=[0, 4, 128] if mtp else [0, 4],
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=1
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        additional_config={},
        speculative_config=SimpleNamespace(method="mtp") if mtp else None,
    )


def cache(groups=None):
    if groups is None:
        groups = [["model.layers.0.attn"], ["model.layers.1.indexer.k_cache"]]
    return SimpleNamespace(
        num_blocks=64,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=names,
                kv_cache_spec=SimpleNamespace(
                    block_size=128, head_size=512, dtype="bf16", num_kv_heads=1
                ),
            )
            for names in groups
        ],
    )


@pytest.mark.parametrize("model", ["glm_moe_dsa", "deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("p_tp,d_tp", [(1, 2), (2, 1), (2, 4), (4, 2)])
def test_replicated_namespace_ignores_tp_and_draft_but_not_target_schema(
    model, p_tp, d_tp
):
    p, d = config(p_tp, model=model), config(d_tp, mtp=True, model=model)
    pc, dc = cache(), cache(
        [
            ["model.layers.0.attn"],
            ["model.layers.1.indexer.k_cache"],
            ["model.layers.2.attn"],
            ["model.mtp.layers.0.attn"],
        ]
    )
    assert LAYOUT.build_namespace(p, pc) == LAYOUT.build_namespace(d, dc)
    dc.kv_cache_groups[0].kv_cache_spec.dtype = "fp8"
    assert LAYOUT.build_namespace(p, pc) != LAYOUT.build_namespace(d, dc)


def test_sharded_or_cp_layout_not_silently_shared():
    cfg, groups = config(), cache()
    groups.kv_cache_groups[0].kv_cache_spec.num_kv_heads = 2
    with pytest.raises(ValueError, match="not TP-replicated"):
        LAYOUT.build_namespace(cfg, groups)
    cfg.parallel_config.decode_context_parallel_size = 2
    assert LAYOUT.build_namespace(cfg, groups) is None


def test_each_component_uses_its_own_group_block_ids():
    class FakeLayout:
        def __init__(self, caches, *args):
            self.base_ptrs = np.array(list(caches.values()), dtype=np.uint64)
            self.layer_name_to_id = dict.fromkeys(caches, 0)
            self.buffer_sizes = np.array([4096])
            self.tensor_size_lists = self.block_stride_lists = np.array([128])

        def extract_block_addrs(self, ids):
            return self.base_ptrs[None, :] + np.array(ids)[:, None] * 128

    layout = LAYOUT.ReplicatedMLAStoreLayout(
        {"model.layers.0.attn": 1000, "model.layers.1.indexer.k_cache": 9000},
        {},
        config(),
        cache(),
        FakeLayout,
    )
    assert layout.extract_block_addrs_for_groups(([2, 3], [7, 9])).tolist() == [
        [1256, 9896],
        [1384, 10152],
    ]


class MemoryStore:
    def __init__(self, fail=False):
        self.values = {}
        self.fail = fail

    def lookup(self, keys):
        return [key in self.values for key in keys]


class RankManager:
    def __init__(self):
        self.waited = []

    def submit_dump(self, store, by_request, keys, shards, ptrs, event):
        return SimpleNamespace(store=store, keys=keys, ptrs=ptrs.copy())

    def wait_dump(self, task):
        self.waited.append(task)
        if task.store.fail:
            return False
        task.store.values.update(zip(task.keys, task.ptrs.tolist()))
        return True


@pytest.mark.parametrize("wa_failure", [False, True])
def test_fawa_real_dispatch_waits_both_stores_before_ack_and_retries(wa_failure):
    obj = connector(cls=FAWA)
    req = request(range(3))
    initial = step(obj, req, 3, groups=([11, 12], [21, 22]), new=True)
    assert initial.save_job is None
    req.all_token_ids.extend(range(3, 8))
    req.num_computed_tokens = 7
    dispatch = step(obj, req, 1)
    obj.fa_store, obj.wa_store = MemoryStore(), MemoryStore(fail=wa_failure)
    obj.device = SimpleNamespace(destroy_event_handle=Mock())
    obj._get_dump_event_handle = lambda: 7
    obj._rank_consistency = RankManager()
    obj.file_size = {"FA": 128, "WA": 128}
    obj._extract_fa_ptr = lambda keys, start, end, groups: np.array(groups[0])[:, None]
    obj._extract_wa_ptr = lambda keys, groups: np.array(groups[1])[:, None]
    meta = SimpleNamespace(request_meta={"req": dispatch})
    assert not obj._writeback_results
    obj._save_writeback_requests(meta)
    job = dispatch.save_job
    result = obj._writeback_results[job.job_id]
    assert len(obj._rank_consistency.waited) == 2
    assert (0 in result.failed) == wa_failure
    assert obj._writeback_states["req"].complete_save(result) != wa_failure
    assert set(job.keys) == set(obj.fa_store.values)
    if wa_failure:
        assert not obj._writeback_states["req"].saved
        obj.wa_store.fail = False
        obj._writeback_results.clear()
        retry = step(obj, req, 1)
        obj._save_writeback_requests(SimpleNamespace(request_meta={"req": retry}))
        assert len(obj._rank_consistency.waited) == 3  # FA deduplicated on retry
        assert obj._writeback_states["req"].complete_save(
            obj._writeback_results[retry.save_job.job_id]
        )
    assert list(obj.wa_store.values) == [job.keys[-1]]


def test_debug_has_exact_keys_and_real_completion_status(caplog):
    obj = connector()
    req = request(range(8))
    state = obj._register_writeback_request(req)
    hashes, _ = state.refresh_request(8)
    job = state.plan_save(hashes, 1, [0], "decode")
    with caplog.at_level(logging.DEBUG, logger="decode-writeback-test"):
        obj._debug_keys("UCM_PREFILL_LOOKUP", "next", job.keys, hit_count=1)
        obj._complete_writeback_jobs((job,), False)
    records = [
        json.loads(record.message.split("] ", 1)[1]) for record in caplog.records
    ]
    assert records[0]["key"] == job.keys[0].hex() and records[0]["hit"]
    assert not records[1]["hit"]
    assert records[-1]["status"] == "failed" and records[-1]["store_job_id"] == 1


def test_family_store_namespaces_lookup_prefetch_and_tensor_apis():
    raw = Mock()
    fa = LAYOUT.CacheFamilyStore(raw, "FA")
    wa = LAYOUT.CacheFamilyStore(raw, "WA")
    key = b"same logical key"
    assert fa.key_for(key) != wa.key_for(key)
    for wrapper in (fa, wa):
        encoded = [wrapper.key_for(key)]
        wrapper.lookup([key])
        raw.lookup.assert_called_with(encoded)
        wrapper.lookup_on_prefix([key])
        raw.lookup_on_prefix.assert_called_with(encoded)
        wrapper.lookup_on_reverse([key])
        raw.lookup_on_reverse.assert_called_with(encoded)
        wrapper.prefetch([key])
        raw.prefetch.assert_called_with(encoded)
        wrapper.dump([key], [0], [["tensor"]])
        raw.dump.assert_called_with(encoded, [0], [["tensor"]])
        wrapper.load([key], [0], [["tensor"]])
        raw.load.assert_called_with(encoded, [0], [["tensor"]])
    assert LAYOUT.CacheFamilyStore(raw, "FA").key_for(key) == fa.key_for(key)


def test_v41_shared_namespace_survives_flattening_and_draft_placement():
    p, d = config(2, model="deepseek_v41"), config(4, mtp=True, model="deepseek_v41")
    p.model_config.hf_config = p.model_config.hf_text_config
    d.model_config.hf_config = d.model_config.hf_text_config
    pc = cache()
    pc.kv_cache_tensors = [
        SimpleNamespace(
            layers=list(group.layer_names),
            layer_stride=0,
            block_stride=128,
            offset=0,
        )
        for group in pc.kv_cache_groups
    ]
    dc = copy.deepcopy(pc)
    pc.kv_cache_groups[0].kv_cache_spec.kv_cache_specs = {
        pc.kv_cache_groups[0].layer_names[0]: SimpleNamespace(
            tokens_per_state=2, num_kv_heads=1
        ),
    }
    # Scheduler has only a merged representative spec. Execution-only MTP
    # placement is absent from the Prefill target schema.
    draft = "model.layers.2.attn"
    dc.kv_cache_groups.append(cache([[draft]]).kv_cache_groups[0])
    dc.kv_cache_tensors.append(
        SimpleNamespace(
            layers=[draft],
            layer_stride=0,
            block_stride=999,
            offset=0,
        )
    )
    hp = SYMS["_build_v41_hash_meta"](p, 0, pc)
    hd = SYMS["_build_v41_hash_meta"](d, 3, dc)
    assert hp == hd
    dc.kv_cache_tensors[0].block_stride *= 2
    assert SYMS["_build_v41_hash_meta"](d, 3, dc) != hp


@pytest.mark.parametrize("value", ["true", "false", 1, None])
def test_decode_flag_rejects_non_booleans(value):
    with pytest.raises(ValueError, match="boolean"):
        SYMS["validate_decode_config"]({"save_decode_cache": value}, config(), cache())


@pytest.mark.parametrize(
    "parallel",
    [
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
def test_decode_rejects_unsupported_parallelism(parallel):
    cfg = config()
    setattr(cfg.parallel_config, parallel, 2)
    with pytest.raises(ValueError, match="PP=PCP=DCP"):
        SYMS["validate_decode_config"](
            {"save_decode_cache": True, "use_layerwise": False}, cfg, cache()
        )


def test_config_default_off_and_yaml_override(monkeypatch):
    assert not SYMS["validate_decode_config"]({}, config(), None)
    assert SYMS["validate_decode_config"](
        {"save_decode_cache": True, "use_layerwise": False}, config(), cache()
    )
    with pytest.raises(ValueError, match="use_layerwise"):
        SYMS["validate_decode_config"]({"save_decode_cache": True}, config(), cache())
    path = ROOT / "ucm/utils.py"
    node = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if getattr(n, "name", None) == "Config"
    )
    namespace = dict(Any=Any, Dict=dict, logger=Mock())
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    config_class = namespace["Config"]
    monkeypatch.setattr(
        config_class,
        "load_ucm_config_from_yaml",
        staticmethod(
            lambda _path: {"save_decode_cache": False, "ucm_connectors": ["unchanged"]}
        ),
    )
    actual = config_class(
        SimpleNamespace(
            kv_connector_extra_config={
                "UCM_CONFIG_FILE": "test.yaml",
                "save_decode_cache": True,
            }
        )
    ).get_config()
    assert actual == {"save_decode_cache": True, "ucm_connectors": ["unchanged"]}


def test_incremental_hash_preserves_extra_keys_and_multimodal_cursor(monkeypatch):
    def extra(req, start, end, cursor):
        next_cursor = cursor + 1
        return (req.cache_salt, req.lora_request, next_cursor), next_cursor

    monkeypatch.setattr(HASH, "generate_block_hash_extra_keys", extra)
    req = request(range(13))
    req.cache_salt, req.lora_request, req.mm_features = "tenant", "adapter", ["image"]
    hasher = HASH.RequestHasher("extra-key-schema")
    incremental = HASH.IncrementalRequestHasher(hasher, 4)
    for end in (4, 7, 8, 12, 4, 12):
        shortened = copy.copy(req)
        shortened.all_token_ids = req.all_token_ids[:end]
        assert incremental.update(req, end) == hasher.make_request_block_hasher(4)(
            shortened
        )
    assert incremental.mm_cursors == [0, 1, 2, 3]


@pytest.mark.parametrize("unhealthy", [False, True])
def test_rank_wait_dump_returns_real_failure_instead_of_swallowing(unhealthy):
    path = SOURCE / "rank_consistency.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if getattr(n, "name", None) == "RankConsistencyManager"
    )
    method = next(n for n in cls.body if getattr(n, "name", None) == "wait_dump")
    error = type("StoreUnhealthyError", (RuntimeError,), {})
    namespace = dict(Any=Any, StoreUnhealthyError=error)
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    task = object()
    store = SimpleNamespace(
        wait=Mock(side_effect=error("offline") if unhealthy else None)
    )
    manager = SimpleNamespace(
        _dump_task_contexts={id(task): (store, {"req": {b"key"}})},
        _record_dump_failure=Mock(),
    )
    assert namespace["wait_dump"](manager, task) == (not unhealthy)
    assert manager._record_dump_failure.called == unhealthy
    assert not manager._dump_task_contexts


@pytest.mark.parametrize("cls", [Direct, FAWA])
def test_authoritative_block_table_replaces_incremental_ids_after_cow(cls):
    obj = connector(cls=cls)
    req = request(range(8), computed=7)
    obj.on_new_request(req)
    obj._ensure_writeback_meta(
        "req", SYMS["FAWARequestMeta"] if cls is FAWA else SYMS["RequestMeta"]
    )
    dispatch = (
        SYMS["FAWARequestDispatchMeta"]()
        if cls is FAWA
        else SYMS["RequestDispatchMeta"](([], []), ([], []))
    )
    obj._writeback_states["req"].prefill_end = 5
    obj._writeback_states["req"].update_blocks(([1, 2], [3, 4]))
    snapshot = ([31, 32], [41, 42])
    output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(
            get_block_ids=lambda rid: snapshot if rid == "req" else None,
        )
    )
    groups = obj._resolve_writeback_blocks(output, "req")
    obj._prepare_writeback_dispatch("req", dispatch, None, 1, full_groups=groups)
    assert obj._writeback_states["req"].blocks == [list(x) for x in snapshot]
    if cls is Direct:
        assert dispatch.dump_groups == snapshot
    else:
        assert dispatch.dump_vllm_block_ids == ([31, 32], [42])


def test_verifier_replays_real_reasoning_and_keeps_eos_enabled(monkeypatch, capsys):
    script = load_module(
        "decode_test_verifier", ROOT / "benchmarks/verify_decode_writeback.py"
    )
    payloads = []

    def open_mock(url, payload, timeout):
        payloads.append(payload)
        if len(payloads) == 1:
            return io.BytesIO(
                json.dumps(
                    {
                        "id": "turn1",
                        "choices": [
                            {
                                "message": {
                                    "content": "actual answer",
                                    "reasoning_content": "actual thinking",
                                },
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    }
                ).encode()
            )
        event = {"id": "turn2", "choices": [{"delta": {"content": "summary"}}]}
        return io.BytesIO(
            ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
        )

    monkeypatch.setattr(script, "open_request", open_mock)
    monkeypatch.setattr(
        sys, "argv", ["verify", "--url", "http://test", "--model", "test"]
    )
    script.main()
    assert len(payloads) == 2
    assert payloads[1]["messages"][2] == {
        "role": "assistant",
        "content": "actual answer",
        "reasoning_content": "actual thinking",
    }
    assert all(payload["ignore_eos"] is False for payload in payloads)
    assert (
        json.loads(capsys.readouterr().out)["turn2"]["ttft_first_delta_s"] is not None
    )


class ByteStore(MemoryStore):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def put(self, keys, ptrs):
        for key, row in zip(keys, ptrs):
            self.values[key] = b"".join(
                ctypes.string_at(int(ptr), size) for ptr, size in zip(row, self.sizes)
            )

    def get(self, keys, ptrs):
        for key, row in zip(keys, ptrs):
            payload = self.values[key]
            offset = 0
            for ptr, size in zip(row, self.sizes):
                ctypes.memmove(int(ptr), payload[offset : offset + size], size)
                offset += size
            assert offset == len(payload)

    def dump_data(self, keys, shards, ptrs, prerequisite_handle=0):
        return SimpleNamespace(store=self, keys=keys, ptrs=ptrs.copy())

    def load_data(self, keys, shards, ptrs):
        self.get(keys, ptrs)

    def lookup_on_prefix(self, keys):
        for index, key in enumerate(keys):
            if key not in self.values:
                return index - 1
        return len(keys) - 1

    def lookup_on_reverse(self, keys):
        return next(
            (i for i in range(len(keys) - 1, -1, -1) if keys[i] in self.values), -1
        )

    def prefetch(self, keys):
        pass


class ByteRankManager(RankManager):
    def submit_dump(self, store, by_request, keys, shards, ptrs, event):
        return store.dump_data(keys, shards, ptrs, event)

    def wait_dump(self, task):
        self.waited.append(task)
        task.store.put(task.keys, task.ptrs)
        return True


@pytest.mark.parametrize("writer_tp,reader_tp", [(1, 2), (2, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize("model", ["glm_moe_dsa", "deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_real_pointer_put_get_bytes_across_tp_and_source_reuse(
    model, writer_tp, reader_tp, phase
):
    cls = Direct if model == "glm_moe_dsa" else FAWA
    scheduler = connector(cls=cls, tp=writer_tp)
    source_ids = ([11, 12], [21, 22])
    if phase == "prefill":
        req = request(range(8))
        dispatch = step(scheduler, req, 8, groups=source_ids, new=True)
    else:
        req = request(range(3))
        assert step(scheduler, req, 3, groups=source_ids, new=True).save_job is None
        req.all_token_ids.extend(range(3, 8))
        req.num_computed_tokens = 7
        dispatch = step(scheduler, req, 1)
    job = dispatch.save_job
    assert job is not None and job.phase == phase
    sources = [
        (np.arange(32 * 16).reshape(32, 16) + i * 17).astype("uint8") for i in range(2)
    ]
    snapshots = [array.copy() for array in sources]

    def layouts(buffers, cfg):
        if cls is Direct:
            cc = cache()
            cc.num_blocks = 32
            for group in cc.kv_cache_groups:
                group.kv_cache_spec.block_size = 4
            tensors = {
                cc.kv_cache_groups[i].layer_names[0]: ByteTensor(
                    array.reshape(32, 4, 1, 4)
                )
                for i, array in enumerate(buffers)
            }
            return LAYOUT.ReplicatedMLAStoreLayout(
                tensors,
                {},
                cfg,
                cc,
                SYMS["KVCacheLayout"],
            )
        result = {}
        for gid, array in enumerate(buffers):
            name = "model.layers.0." + ("long_kv_cache" if gid == 0 else "swa_cache")
            # The first eight bytes are the visible view, remaining bytes
            # include aliased indexer scale/page content in the V4.1 case.
            whole_page = model == "deepseek_v41"
            view = np.ndarray(
                shape=(32, 4, 1, 2 if whole_page else 4),
                dtype="uint8",
                buffer=array,
                strides=(16, 2 if whole_page else 4, 2 if whole_page else 4, 1),
            )
            result[gid] = SYMS["KVCacheGroupLayout"](
                {name: ByteTensor(view)},
                is_ascend_layout=True,
                expected_block_size=4,
                whole_page_sizes_by_layer={name: 16} if whole_page else {},
            )
        return result

    source_layouts = layouts(sources, config(writer_tp, model=model))
    if cls is Direct:
        stores = [ByteStore([16, 16])]
    else:
        # Same logical hashes, one global backend: records must not overwrite.
        shared_store = ByteStore([16])
        stores = [
            LAYOUT.CacheFamilyStore(shared_store, family) for family in ("FA", "WA")
        ]
    results = {}
    for rank in range(writer_tp):
        worker = connector(cls=cls, tp=writer_tp)
        worker.tp_rank = rank
        worker.device = SimpleNamespace(destroy_event_handle=Mock())
        worker._get_dump_event_handle = lambda: 1
        worker._rank_consistency = ByteRankManager()
        if cls is Direct:
            worker.is_mla = True
            worker._skip_null_vllm_blocks = False
            worker.store, worker.kv_cache_layout = stores[0], source_layouts
            worker._pending_dump_tasks = []
            worker._poll_pending_dump_tasks = lambda: None
            worker._get_connector_metadata = lambda: SYMS["UCMConnectorMetadata"](
                {"req": dispatch}
            )
            worker.enable_event_sync = True
            worker.block_data_size = 32
            worker.wait_for_save()
            assert (
                not worker._writeback_results
                if rank == 0
                else worker._writeback_results
            )
            for pending in worker._pending_dump_tasks:
                worker._wait_pending_dump_task(pending)
        else:
            worker.fa_store, worker.wa_store = stores
            worker.group_layouts = source_layouts
            worker.file_size = {"FA": 16, "WA": 16}
            worker._save_writeback_requests(
                SimpleNamespace(request_meta={"req": dispatch})
            )
        SYMS["merge_writeback_results"](results, worker._writeback_results)
    assert scheduler._writeback_states["req"].complete_save(results[job.job_id])
    # A later request must read persisted bytes, not a dangling source view.
    for array in sources:
        array[:] = 0
    next_keys = scheduler.request_block_hasher(request(list(range(8)) + [100, 101]))
    assert next_keys[:2] == list(job.keys)
    for _rank in range(reader_tp):
        destinations = [np.zeros_like(array) for array in sources]
        destination_layouts = layouts(destinations, config(reader_tp, model=model))
        if cls is Direct:
            ptrs = destination_layouts.extract_block_addrs_for_groups(([2, 3], [4, 5]))
            stores[0].load_data(next_keys[:2], [0, 0], ptrs)
            np.testing.assert_array_equal(
                destinations[0][[2, 3]], snapshots[0][[11, 12]]
            )
            np.testing.assert_array_equal(
                destinations[1][[4, 5]], snapshots[1][[21, 22]]
            )
        else:
            reader = connector(cls=FAWA, tp=reader_tp)
            reader.group_layouts = destination_layouts
            stores[0].load_data(
                next_keys[:2],
                [0, 0],
                reader._extract_fa_ptr(next_keys[:2], 0, 2, ([2, 3], [4])),
            )
            stores[1].load_data(
                next_keys[1:2],
                [0],
                reader._extract_wa_ptr(next_keys[1:2], ([2, 3], [4])),
            )
            np.testing.assert_array_equal(
                destinations[0][[2, 3]], snapshots[0][[11, 12]]
            )
            np.testing.assert_array_equal(destinations[1][4], snapshots[1][22])
