"""Prepared-result consumption through the real native Python method.

Requires the normal UCM/vLLM Python imports and pipeline extension. Device
setup, control transport and C++ I/O are replaced; this is not an NPU test.
"""

import threading
import time
import array
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from ucm.integration.vllm import early_prefetch, ucm_connector
from ucm.integration.vllm.early_connector import (
    EarlyDirectConnector,
    EarlyLayerWiseConnector,
)
from ucm.integration.vllm.early_lookup import current_lookup
from ucm.integration.vllm.early_prefetch import EarlyScheduler, Job
from ucm.integration.vllm.early_owner_worker import OwnerWorker
from ucm.integration.vllm.early_owner import worker_capability
from ucm.integration.vllm.rank_consistency import RankConsistencyManager
from ucm.store.pipeline import connector as native_store
from ucm.store.pipeline import early_connector as early_store
from ucm.store.pipeline.early_connector import EarlyPipelineStore


class TokenView(list):
    reject_reads = False

    def __getitem__(self, item):
        if self.reject_reads:
            raise AssertionError("prepared request tokens were scanned again")
        return super().__getitem__(item)

    def __iter__(self):
        if self.reject_reads:
            raise AssertionError("prepared request tokens were scanned again")
        return super().__iter__()


class Request:
    def __init__(self, request_id="request", token_count=8):
        self.request_id = request_id
        self.all_token_ids = TokenView(range(token_count))
        self.num_tokens = token_count
        self.max_tokens = 16
        self.status = ucm_connector.RequestStatus.WAITING
        self.mm_features = []
        self.lora_request = None
        self.cache_salt = None
        self.prompt_embeds = None


@pytest.mark.parametrize("cls", [EarlyDirectConnector, EarlyLayerWiseConnector])
def test_native_worker_hash_identity_without_scheduler_seed(cls):
    """Workers have rank-specific hashers, but MLA block IDs come from rank 0."""
    caps = []
    for rank in range(4):
        c = object.__new__(cls)
        config = SimpleNamespace(
            model_config=SimpleNamespace(model="/models/glm-smoke", dtype="bfloat16", max_model_len=8192),
            parallel_config=SimpleNamespace(tensor_parallel_size=4, data_parallel_index=0, pipeline_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=4))
        c._vllm_config = config
        c.request_hasher = ucm_connector.RequestHasher(config, rank)
        c.launch_config = {}
        c.is_mla = True
        c.cp_world_size = 1
        c.tp_rank, c.tp_size = rank, 4
        c.hash_block_size = c.block_size = 128
        c.blocks_per_chunk = 1
        c.layer_ids = list(range(8))
        c.use_layerwise = cls is EarlyLayerWiseConnector
        c.unique_id = "native-shared-object"
        info = dict(backing="native-provider-object", nodes=100000, reservable_nodes=90000,
                    shard_bytes=180224, block_bytes=8 * 180224)
        c.store = SimpleNamespace(config=dict(share_buffer_enable=True, shard_size=180224,
                    block_size=8 * 180224, tensor_size_list=[131072, 16384, 32768]), domain_info=lambda: info)
        assert not hasattr(c, "_seed")
        cap = worker_capability(c)
        assert cap.get("ready"), cap
        caps.append(cap)
        assert not hasattr(c, "_seed"), "handshake must not add scheduler state to the native worker"
        config.model_config.model = "/models/different-model"
        changed = worker_capability(c)
        assert changed["layout"] != cap["layout"], "different block identity must be rejected"
    assert len({cap["layout"] for cap in caps}) == 1


@pytest.fixture(params=[EarlyDirectConnector, EarlyLayerWiseConnector])
def connector(request, monkeypatch):
    # Bypass engine/resource initialization only. Token calculation, metadata,
    # identity validation, trace deduplication and Store dispatch are real.
    cls = request.param
    c = object.__new__(cls)
    c.block_size = c.hash_block_size = 4
    c.cp_world_size = 1
    c._seed = b"seed"
    c.request_hasher = Mock(return_value=b"native" * 3)
    c.request_hasher.make_request_block_hasher = lambda size, seed: (
        ucm_connector.RequestHasher.make_request_block_hasher(c.request_hasher, size, seed)
    )
    c._role = ucm_connector.KVConnectorRole.SCHEDULER
    c._bind_request_block_hasher()
    c.enable_record_traces = True
    c.persist_token_threshold = 0
    c.requests_meta = {}
    c._pending_async_load_dispatches = {}
    c._async_load_req_ids = set()
    c.use_request_async_load = False
    c._other_rank_hashers = []
    c.use_layerwise = cls is EarlyLayerWiseConnector
    c._vllm_config = SimpleNamespace(speculative_config=None)
    c._rank_consistency = RankConsistencyManager(is_scheduler=True)
    c.store = object.__new__(EarlyPipelineStore)
    c.store._native_store = Mock()
    c.store.store_ = early_store._ReservedLoad(c.store._native_store)
    c.store.store_.LookupOnPrefix.return_value = 1
    c._early = object.__new__(EarlyScheduler)
    c._early.connector = c
    c._early.lock = threading.RLock()
    c._early.jobs = {}
    c._early.by_key = {}
    c._early.traced_requests = weakref.WeakSet()
    c._early.control = Mock()
    monkeypatch.setattr(early_prefetch, "log", Mock())
    monkeypatch.setattr(ucm_connector, "logger", Mock())
    monkeypatch.setattr(ucm_connector.ucmmetrics, "update_stats", Mock())
    return c


def prepare(c, request, *, state="READY", hit=2):
    job = Job(request, request.all_token_ids, c._early.identity(request))
    job.state, job.hit = state, hit
    job.blocks = [b"a" * 16, b"b" * 16]
    c._early.jobs[request.request_id] = job
    c._early.by_key[job.key] = job
    request.all_token_ids.reject_reads = True
    return job


def test_ready_does_not_expand_the_protected_rank_minimum(connector):
    request = Request()
    prepare(connector, request, hit=0)
    # The underlying lookup would report two blocks, neither is protected.
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.requests_meta[request.request_id].total_hit_block_num == 0
    connector.request_hasher.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_not_called()
    connector.store.store_.Prefetch.assert_not_called()


@pytest.mark.parametrize("computed", [0, 4])
def test_ready_all_l1_survives_l2_miss_and_uses_native_metadata(connector, computed):
    request = Request()
    job = prepare(connector, request)
    connector.store.store_.LookupOnPrefix.return_value = -1
    recompute = 2 if connector.use_layerwise else 1
    expected = 8 - computed - recompute
    assert connector.get_num_new_matched_tokens(request, computed) == (expected, False)
    meta = connector.requests_meta[request.request_id]
    assert meta.ucm_block_ids is job.blocks
    assert (meta.hbm_hit_block_num, meta.total_hit_block_num) == (computed // 4, 2)
    assert meta.token_processed == 8 - recompute
    connector.request_hasher.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_not_called()
    connector.store.store_.Prefetch.assert_not_called()


def test_ready_respects_native_known_missing_prefix(connector):
    request = Request()
    job = prepare(connector, request)
    connector._rank_consistency._tracker.mark_missing({job.blocks[1]})
    assert connector.get_num_new_matched_tokens(request, 0) == (4, False)
    connector.store.store_.LookupOnPrefix.assert_not_called()


@pytest.mark.parametrize("computed, expected", [(0, 8), (8, 0)])
def test_ready_counts_hashes_before_native_cp_conversion(connector, computed, expected):
    connector.block_size, connector.cp_world_size = 8, 2
    request = Request(token_count=16)
    job = prepare(connector, request, hit=3)
    job.blocks.extend([b"c" * 16, b"d" * 16])
    assert connector.get_num_new_matched_tokens(request, computed) == (expected, False)
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_pending_returns_without_native_work(connector):
    request = Request()
    prepare(connector, request, state="PENDING")
    for _ in range(3):
        assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
    assert not connector.requests_meta
    connector.request_hasher.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_not_called()
    early_prefetch.log.info.assert_not_called()
    assert current_lookup.get() is None


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED"])
def test_fallback_reuses_trusted_ids_but_queries_native_store(connector, state):
    request = Request()
    job = prepare(connector, request, state=state, hit=0)
    expected = 6 if connector.use_layerwise else 7
    assert connector.get_num_new_matched_tokens(request, 0) == (expected, False)
    assert connector.requests_meta[request.request_id].ucm_block_ids is job.blocks
    connector.request_hasher.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_called_once()


def test_reused_request_id_does_not_reuse_another_instance(connector):
    previous = Request()
    job = prepare(connector, previous)
    replacement = Request(previous.request_id)
    connector.get_num_new_matched_tokens(replacement, 0)
    assert job.state == "CANCELLED"
    assert connector.request_hasher.call_count == 2
    assert connector.requests_meta[replacement.request_id].ucm_block_ids != job.blocks
    connector.store.store_.LookupOnPrefix.assert_called_once()


def test_trace_once_across_preparation_and_ready_retries(connector):
    request = Request()
    job = prepare(connector, request)
    connector._early.trace_request(request, job.blocks)
    for _ in range(3):
        connector.get_num_new_matched_tokens(request, 0)
    traces = [
        call for call in early_prefetch.log.info.call_args_list
        if call.args[0].startswith("timestamp:")
    ]
    assert len(traces) == 1
    assert not any(
        call.args[0].startswith("timestamp:")
        for call in ucm_connector.logger.info_once.call_args_list
    )
    assert connector.enable_record_traces is True


def test_exception_restores_trace_and_store_lookup_behavior(connector):
    request = Request()
    connector.request_hasher.side_effect = RuntimeError("hash failed")
    # v0.8.0 handles hash failures as a native miss instead of propagating them.
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    ucm_connector.logger.error.assert_called_once()
    assert current_lookup.get() is None
    assert connector.enable_record_traces is True
    assert connector.store.lookup_on_prefix([b"a" * 16, b"b" * 16]) == 1
    connector.store.store_.LookupOnPrefix.assert_called_once()


def test_early_bindings_receive_the_existing_pipeline(connector, monkeypatch):
    reserve = Mock(
        return_value={"handle": 7, "total_hit_blocks": 2, "remaining_l2_work": 0}
    )
    prefetch = Mock(return_value={"l2_read_shards": 0})
    release = Mock()
    monkeypatch.setattr(early_store.ucmpipelinestore, "EarlyReservePrefix", reserve)
    monkeypatch.setattr(early_store.ucmpipelinestore, "EarlyPrefetchReserved", prefetch)
    monkeypatch.setattr(early_store.ucmpipelinestore, "EarlyReleaseReservation", release)
    store = connector.store
    blocks = [b"a" * 16, b"b" * 16]
    assert store.reserve_prefix(blocks)["handle"] == 7
    assert reserve.call_args.args[0] is store._native_store
    assert reserve.call_args.args[1].tobytes() == b"".join(blocks)
    assert store.prefetch_reserved(7) == {"l2_read_shards": 0}
    prefetch.assert_called_once_with(store._native_store, 7)
    store.release_reservation(7)
    release.assert_called_once_with(store._native_store, 7)


def test_early_binding_error_is_not_replaced(connector, monkeypatch):
    error = RuntimeError("native backend failed")
    monkeypatch.setattr(
        early_store.ucmpipelinestore, "EarlyPrefetchReserved", Mock(side_effect=error)
    )
    with pytest.raises(RuntimeError) as raised:
        connector.store.prefetch_reserved(7)
    assert raised.value is error


@pytest.mark.parametrize("early", [False, True])
def test_pipeline_builder_preserves_native_config_and_instance(monkeypatch, early):
    pipeline = Mock()
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(native_store.ucmpipelinestore, "PipelineStore", factory)
    monkeypatch.setattr(native_store, "_preload_metrics", Mock())
    native_builder = native_store.UcmPipelineStoreBuilder.get("Cache|Posix")
    builder = Mock(wraps=native_builder)
    monkeypatch.setitem(
        native_store.UcmPipelineStoreBuilder.registry_, "Cache|Posix", builder
    )
    config = {"store_pipeline": "Cache|Posix", "device_id": 0, "shard_size": 4096}
    cls = EarlyPipelineStore if early else native_store.UcmPipelineStore
    store = cls(config)
    factory.assert_called_once_with()
    builder.assert_called_once()
    assert (store._native_store if early else store.store_) is pipeline
    assert store.config is config
    assert config == {
        "store_pipeline": "Cache|Posix", "device_id": 0, "shard_size": 4096
    }
    calls = pipeline.Stack.call_args_list
    prefix = "Early" if early else ""
    assert [call.args[0] for call in calls] == [f"{prefix}Posix", f"{prefix}Cache"]
    assert calls[0].args[2]["tensor_size"] == config["shard_size"]
    assert calls[1].args[2] == config


def test_pipeline_builder_preserves_native_error(monkeypatch):
    monkeypatch.setattr(native_store.ucmpipelinestore, "PipelineStore", Mock())
    error = ValueError("native builder failed")
    monkeypatch.setitem(
        native_store.UcmPipelineStoreBuilder.registry_,
        "Cache|Posix",
        Mock(side_effect=error),
    )
    with pytest.raises(ValueError) as raised:
        EarlyPipelineStore({"store_pipeline": "Cache|Posix", "device_id": -1})
    assert raised.value is error


def test_missing_early_bindings_fail_before_pipeline_creation(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(native_store.ucmpipelinestore, "PipelineStore", factory)
    monkeypatch.delattr(
        native_store.ucmpipelinestore, "EarlyPrefetchReserved", raising=False
    )
    with pytest.raises(RuntimeError, match="bindings are missing"):
        EarlyPipelineStore({"store_pipeline": "Cache|Posix", "device_id": -1})
    factory.assert_not_called()


@pytest.mark.parametrize("method", ["load", "load_data"])
def test_handoff_uses_native_argument_conversion_and_same_pipeline(monkeypatch, method):
    pipeline = Mock()
    store = object.__new__(EarlyPipelineStore)
    store._native_store = pipeline
    store.store_ = early_store._ReservedLoad(pipeline)
    addresses = np.array([[123, 456]], dtype=np.uint64)
    store._tensor_normalize = Mock(return_value=addresses)
    submit = Mock(return_value=19)
    release = Mock()
    monkeypatch.setattr(early_store.ucmpipelinestore, "EarlyLoad", submit)
    monkeypatch.setattr(early_store.ucmpipelinestore, "EarlyReleaseReservation", release)
    store.set_load_reservations([7, 9])
    getattr(store, method)([b"a" * 16], [3], addresses)
    owner, ids, indexes, addrs, reservations = submit.call_args.args
    assert owner is pipeline
    assert ids.tobytes() == b"a" * 16
    assert list(indexes) == [3]
    assert addrs is addresses
    assert reservations == (7, 9)
    pipeline.Load.assert_not_called()
    store.release_reservation(7)
    assert store.store_.reservations == (9,)
    store.release_reservation(9)
    getattr(store, method)([b"b" * 16], [0], addresses)
    pipeline.Load.assert_not_called()
    assert submit.call_count == 2
    assert submit.call_args.args[-1] == ()


def test_handoff_submit_error_keeps_native_exception(monkeypatch):
    store = object.__new__(EarlyPipelineStore)
    store._native_store = Mock()
    store.store_ = early_store._ReservedLoad(store._native_store)
    store.set_load_reservations([7])
    error = RuntimeError("native HBM load failed")
    monkeypatch.setattr(early_store.ucmpipelinestore, "EarlyLoad", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        store.load_data([b"a" * 16], [0], [[123]])
    assert raised.value is error


def test_worker_bind_selects_only_ready_local_consumers():
    worker = object.__new__(early_prefetch.EarlyWorker)
    worker.owner_protocol = OwnerWorker(worker, {}, "w0.0")
    worker.condition = threading.Condition()
    worker.store = Mock()
    worker.request_keys = {}
    worker.jobs = {
        key: SimpleNamespace(key=key, ready=ready, cancelled=cancelled,
                             loading=False, batches=[SimpleNamespace(reservation={"handle": handle})])
        for key, ready, cancelled, handle in (
            ("ready", True, False, 7), ("pending", False, False, 8),
            ("cancelled", True, True, 9), ("other", True, False, 10)
        )
    }
    metadata = SimpleNamespace(
        early_keys={"r1": "ready", "r2": "pending", "r3": "cancelled"},
        request_meta={request: SimpleNamespace(load_block_ids=([b"a" * 16], []))
                      for request in ("r1", "r2", "r3")},
    )
    worker.bind(metadata)
    worker.store.set_load_reservations.assert_called_once_with([7])
    worker.bind(SimpleNamespace(early_keys={}, request_meta={}))
    worker.store.set_load_reservations.assert_called_with([])


def test_submit_is_once_and_pending_never_waits_for_hash(connector, monkeypatch):
    connector._vllm_config.parallel_config = SimpleNamespace(
        data_parallel_index=0, data_parallel_size=1)
    connector.unique_id, connector.tp_size = "unit-test", 1
    gate = threading.Event()
    entered = threading.Event()
    control = Mock()

    def hashing(*args):
        entered.set()
        assert gate.wait(3)
        return [b"a" * 16, b"b" * 16]

    hasher = Mock(side_effect=hashing)
    monkeypatch.setattr(connector, "request_block_hasher", hasher)
    scheduler = EarlyScheduler(connector, control_factory=lambda *a, **kw: control)
    connector._early = scheduler
    request = Request()
    try:
        start = time.monotonic()
        scheduler.submit(request)
        scheduler.submit(request)
        assert time.monotonic() - start < 0.5
        assert entered.wait(1)
        for _ in range(100):
            assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
        connector.store.store_.LookupOnPrefix.assert_not_called()
        control.send.assert_not_called()
    finally:
        gate.set()
        scheduler.close()
    hasher.assert_called_once()


@pytest.mark.parametrize("is_mla, rank", [(True, 3), (False, 0), (False, 3)])
def test_worker_reservation_uses_native_mla_or_rank_local_identity(is_mla, rank):
    worker = object.__new__(early_prefetch.EarlyWorker)
    worker.owner_protocol = OwnerWorker(worker, {}, f"w0.{rank}")
    worker.condition = threading.Condition()
    worker.closed = False
    worker.control = Mock()
    worker.store = Mock()
    worker.store.reserve_prefix.return_value = {
        "handle": 7, "total_hit_blocks": 1, "remaining_l2_work": 0}
    worker.connector = SimpleNamespace(
        is_mla=is_mla, tp_rank=rank, tp_size=8,
        request_hasher=Mock(return_value=b"r" * 16))
    job = early_prefetch.WorkerJob("job", [b"a" * 16])
    worker._reserve(job)
    expected = [b"r" * 16] if not is_mla and rank else job.blocks
    worker.store.reserve_prefix.assert_called_once_with(expected)
    worker.store.prefetch_reserved.assert_not_called()
    assert job.ready


def test_early_load_binding_preserves_native_dimension_errors():
    # No Store/device is constructed: both calls must fail in the shared native
    # argument converter before either could dereference a pipeline entry.
    pipeline = early_store.ucmpipelinestore.PipelineStore()
    ids = np.frombuffer(b"a" * 16, dtype=np.uint8)
    indexes = array.array("Q")
    addresses = np.zeros((1, 1), dtype=np.uint64)
    with pytest.raises(RuntimeError) as native:
        pipeline.Load(ids, indexes, addresses)
    with pytest.raises(type(native.value)) as prepared:
        early_store.ucmpipelinestore.EarlyLoad(pipeline, ids, indexes, addresses, [7])
    assert str(prepared.value) == str(native.value)


def test_owner_expected_plan_uses_native_hbm_and_missing_filter_without_relookup(connector):
    request = Request(token_count=12)
    job = prepare(connector, request, hit=3)
    job.blocks.append(b"c" * 16)
    job.owner, job.attempt, job.grant = "w0.1", "attempt", 1
    connector._rank_consistency._tracker.mark_missing({job.blocks[2]})
    assert connector.get_num_new_matched_tokens(request, 4) == (4, False)
    assert job.expected == (job.blocks[1],)
    assert connector._early.owner_metadata({request.request_id: job.key})[request.request_id]["expected"] == job.expected
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_owner_all_hbm_hit_does_not_require_native_metadata(connector):
    request = Request()
    job = prepare(connector, request)
    job.owner, job.attempt, job.grant = "w0.0", "attempt", 1
    assert connector.get_num_new_matched_tokens(request, 8) == (0, False)
    assert job.expected == ()
    connector.store.store_.LookupOnPrefix.assert_not_called()


@pytest.mark.parametrize("tokens,computed", [(9, 8), (8, 4), (3, 0)])
def test_pending_hbm_sufficient_cancels_without_any_external_work(connector, tokens, computed):
    request = Request(token_count=tokens)
    job = prepare(connector, request, state="PENDING")
    job.blocks = job.blocks[:tokens // connector.hash_block_size]
    # Native HBM already covers its maximum prefix (last block recomputes). A
    # queued hash/owner result must not hold up the uncached suffix compute.
    assert connector.get_num_new_matched_tokens(request, computed) == (0, False)
    assert job.state == "CANCELLED"
    assert connector._early.jobs[request.request_id] is job
    assert job.key not in connector._early.by_key
    if tokens % connector.block_size == 0:
        # Native metadata must persist the last block after recomputation.
        meta = connector.requests_meta[request.request_id]
        assert meta.hbm_hit_block_num == meta.total_hit_block_num == computed // 4
        assert meta.ucm_block_ids is job.blocks
        dispatch = connector._generate_dispatch_meta(meta, 4, [101, 102])
        assert dispatch.load_block_ids == ([], [])
        assert dispatch.dump_block_ids == ([job.blocks[-1]], [102])
    else:
        assert not connector.requests_meta
    connector._early.control.send.assert_called_once_with({"op": "cancel", "key": job.key})
    connector._early.receive(dict(op="result", key=job.key, state="READY", hit=2))
    assert job.state == "CANCELLED"
    assert connector.get_num_new_matched_tokens(request, computed) == (0, False)
    assert not connector._early.scheduled({request.request_id: Mock()})
    connector._early.control.send.assert_called_once()
    connector.request_hasher.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_hbm_shortcut_does_not_wait_for_or_duplicate_an_unfinished_hash(connector):
    request = Request()
    job = prepare(connector, request, state="PENDING")
    job.blocks = None
    for _ in range(3):
        assert connector.get_num_new_matched_tokens(request, 4) == (None, False)
    assert job.state == "PENDING"
    connector.request_hasher.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_not_called()
    connector._early.control.send.assert_not_called()


def test_pending_partial_hbm_still_waits_and_preserves_protection(connector):
    # More than the final allocation block is missing: keep the READY gate.
    request = Request(token_count=12)
    job = prepare(connector, request, state="PENDING")
    for _ in range(3):
        assert connector.get_num_new_matched_tokens(request, 4) == (None, False)
    assert job.state == "PENDING"
    assert job.pending_probes == 3 and job.first_pending_ns is not None
    connector._early.control.send.assert_not_called()


def test_pending_cp_partial_hashes_cannot_add_an_allocation_block(connector):
    connector.block_size, connector.cp_world_size = 8, 2
    request = Request(token_count=12)
    prepare(connector, request, state="PENDING")
    assert connector.get_num_new_matched_tokens(request, 8) == (0, False)
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_ready_match_and_owner_metadata_do_not_reserve_or_control_roundtrip(connector):
    request = Request(token_count=12)
    job = prepare(connector, request, hit=2)
    job.owner, job.attempt, job.grant = "w0.1", "a", 2
    connector._early.dp_rank = 0
    connector.store.reserve_prefix = Mock(side_effect=AssertionError("schedule must not reserve"))
    # A scheduler allocation retry can present a different native HBM hit.
    assert connector.get_num_new_matched_tokens(request, 0) == (8, False)
    assert connector.get_num_new_matched_tokens(request, 4) == (4, False)
    assert job.expected == (job.blocks[1],)
    meta = SimpleNamespace(load_block_ids=([job.blocks[1]], []))
    for _ in range(2):
        keys = connector._early.scheduled({request.request_id: meta})
        ticket = connector._early.owner_metadata(keys)[request.request_id]
        assert ticket["expected"] == (job.blocks[1],)
        assert ticket["key"] == job.key and ticket["attempt"] == "a"
    assert job.match_calls == 2 and job.match_ns > 0
    assert (job.hbm_tokens, job.external_tokens) == (4, 4)
    connector.store.reserve_prefix.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_not_called()
    connector._early.control.send.assert_not_called()
    connector._early.finish(request)
    connector._early.control.send.assert_called_once_with({"op": "finish", "key": job.key})


def test_legacy_handoff_is_still_ordered_once_before_finish(connector):
    request = Request()
    job = prepare(connector, request)
    connector._early.dp_rank = 0
    meta = SimpleNamespace(load_block_ids=(job.blocks, []))
    connector._early.scheduled({request.request_id: meta})
    connector._early.scheduled({request.request_id: meta})
    connector._early.finish(request)
    assert [c.args[0]["op"] for c in connector._early.control.send.call_args_list] == ["handoff", "finish"]


@pytest.mark.parametrize("operation", ["load", "dump"])
def test_capacity_rejection_uses_native_consistency_without_claiming_persistence(monkeypatch, operation):
    pipeline, observer = Mock(), Mock()
    proxy = early_store._ReservedLoad(pipeline)
    proxy.observer = observer
    observer.begin.return_value = 73
    binding = "EarlyLoad" if operation == "load" else "EarlyDump"
    error = early_store.ucmpipelinestore.EarlyCapacityRejected("NoSpace before I/O")
    monkeypatch.setattr(early_store.ucmpipelinestore, binding, Mock(side_effect=error))
    store = object.__new__(EarlyPipelineStore)
    store.store_ = proxy
    consistency = RankConsistencyManager(is_scheduler=False)
    block = b"a" * 16
    addresses = np.zeros((1, 1), dtype=np.uint64)
    with pytest.raises(native_store.StoreNotFoundError):
        if operation == "load":
            consistency.submit_load(store, {"r": [block]}, [block], [0], addresses)
        else:
            consistency.submit_dump(store, {"r": {block}}, [block], [0], addresses, 0)
    observer.complete.assert_not_called()
    observer.unscoped_failure.assert_not_called()
    if operation == "load":
        observer.rejected.assert_called_once_with(73)
        assert consistency._missing_blocks == {block}
    else:
        consistency.finish_dump({"r"})
        assert not consistency._dump_succeeded_blocks
        pipeline.Dump.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_direct_completion_uses_native_method_without_layerwise_failure_attribute(monkeypatch, failure):
    from ucm.integration.vllm.hla_connector import UCMDirectConnector
    connector = object.__new__(EarlyDirectConnector)
    connector.use_layerwise = False
    connector._early = Mock()
    connector._connector_worker_meta = ucm_connector.UCMWorkerMetadata()
    if failure:
        connector._connector_worker_meta.mark_failed("r")
    metadata = ucm_connector.UCMConnectorMetadata(
        {"r": ucm_connector.RequestDispatchMeta(([], []), ([], []))}, set()
    )
    connector._get_connector_metadata = Mock(return_value=metadata)
    native = Mock()
    monkeypatch.setattr(UCMDirectConnector, "start_load_kv", native)
    connector.start_load_kv(None)
    native.assert_called_once_with(None)
    assert not hasattr(connector, "_failure_req_ids")
    if failure:
        connector._early.loaded.assert_called_once_with({"r"}, reason="load_failure")
    else:
        connector._early.loaded.assert_called_once_with({"r"})
