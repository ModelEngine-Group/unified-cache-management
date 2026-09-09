"""Prepared-result consumption through the real native Python method.

Requires the normal UCM/vLLM Python imports and pipeline extension. Device
setup, control transport and C++ I/O are replaced; this is not an NPU test.
"""

import threading
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ucm.integration.vllm import early_prefetch, ucm_connector
from ucm.integration.vllm.early_connector import (
    EarlyDirectConnector,
    EarlyLayerWiseConnector,
)
from ucm.integration.vllm.early_lookup import current_lookup
from ucm.integration.vllm.early_prefetch import EarlyScheduler, Job
from ucm.integration.vllm.rank_consistency import RankConsistencyManager
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
    c.store.store_ = Mock()
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
    assert reserve.call_args.args[0] is store.store_
    assert reserve.call_args.args[1].tobytes() == b"".join(blocks)
    assert store.prefetch_reserved(7) == {"l2_read_shards": 0}
    prefetch.assert_called_once_with(store.store_, 7)
    store.release_reservation(7)
    release.assert_called_once_with(store.store_, 7)


def test_early_binding_error_is_not_replaced(connector, monkeypatch):
    error = RuntimeError("native backend failed")
    monkeypatch.setattr(
        early_store.ucmpipelinestore, "EarlyPrefetchReserved", Mock(side_effect=error)
    )
    with pytest.raises(RuntimeError) as raised:
        connector.store.prefetch_reserved(7)
    assert raised.value is error
