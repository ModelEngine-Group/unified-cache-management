"""Early integration with v0.8.0 request hashes and native async completion."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_early_ready_reuse import Request, connector, prepare
from test_ucm_request_hasher import _config, _mm_feature, _semantic_extra_keys

import ucm.integration.vllm.request_hasher as hashing
from ucm.integration.vllm.early_connector import EarlyDirectConnector
from ucm.integration.vllm.early_prefetch import Job
from ucm.integration.vllm.ucm_connector import (
    PendingLoadTask,
    RequestDispatchMeta,
    RequestStatus,
    UCMConnectorMetadata,
    UCMDirectConnector,
    UCMWorkerMetadata,
)


@pytest.mark.parametrize(
    "semantics",
    [
        {"mm_features": [_mm_feature("image-a", 0, 3)]},
        {"lora_request": SimpleNamespace(lora_name="adapter-a")},
        {"cache_salt": "tenant-a"},
        {"prompt_embeds": b"abcdefgh"},
    ],
)
def test_background_hash_and_ready_reuse_keep_native_request_semantics(
    connector, monkeypatch, semantics
):
    monkeypatch.setattr(hashing, "generate_block_hash_extra_keys", _semantic_extra_keys)
    connector.request_hasher = hashing.RequestHasher(_config(), 0)
    connector._seed = connector.request_hasher.seed
    connector._bind_request_block_hasher()
    native = connector._native_request_block_hasher
    connector._native_request_block_hasher = Mock(wraps=native)
    request = Request()
    for name, value in semantics.items():
        setattr(request, name, value)
    expected = native(request)
    assert expected != native(Request())

    early = connector._early
    early.dp_rank = 0
    job = Job(request, request.all_token_ids, early.identity(request))
    early.jobs[request.request_id] = job
    early.by_key[job.key] = job
    early._prepare(job)
    assert job.state == "PENDING"
    assert job.blocks == expected
    assert early.control.send.call_args.args[0]["blocks"] == expected

    job.state, job.hit = "READY", len(expected)
    request.all_token_ids.reject_reads = True
    tokens, asynchronous = connector.get_num_new_matched_tokens(request, 0)
    assert tokens == (6 if connector.use_layerwise else 7)
    assert not asynchronous
    assert connector.requests_meta[request.request_id].ucm_block_ids is job.blocks
    connector._native_request_block_hasher.assert_called_once_with(request)
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_native_preemption_clears_async_state_before_ready_reuse(connector):
    request = Request()
    job = prepare(connector, request)
    stale_metadata = object()
    connector.requests_meta[request.request_id] = stale_metadata
    connector._async_load_req_ids.add(request.request_id)
    connector._pending_async_load_dispatches[request.request_id] = object()
    request.status = RequestStatus.PREEMPTED
    expected_tokens = 6 if connector.use_layerwise else 7
    assert connector.get_num_new_matched_tokens(request, 0) == (expected_tokens, False)
    assert request.request_id not in connector._async_load_req_ids
    assert request.request_id not in connector._pending_async_load_dispatches
    metadata = connector.requests_meta[request.request_id]
    assert metadata is not stale_metadata
    assert metadata.ucm_block_ids is job.blocks
    connector.request_hasher.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_async_reservation_is_released_only_after_native_completion(
    monkeypatch, failure
):
    worker = object.__new__(EarlyDirectConnector)
    worker.use_layerwise = False
    worker._early = Mock()
    worker._pending_load_tasks = {}
    worker._finished_async_load_req_ids = set()
    worker._async_dump_req_ids = set()
    worker._rank_consistency = Mock()
    worker._rank_consistency.check_load.side_effect = [False, True]
    worker._connector_worker_meta = UCMWorkerMetadata()
    worker._record_load_error = Mock()
    events = []

    def wait(task):
        events.append("wait")
        if failure:
            raise RuntimeError("native async transfer failed")

    worker._rank_consistency.wait_load.side_effect = wait
    worker._early.loaded.side_effect = lambda ids, reason="handoff": events.append(
        (set(ids), reason)
    )
    metadata = UCMConnectorMetadata(
        {
            "sync": RequestDispatchMeta(([b"s"], [1]), ([], [])),
            "async": RequestDispatchMeta(([b"a"], [2]), ([], []), load_async=True),
        },
        set(),
    )
    worker._get_connector_metadata = lambda: metadata

    def submit(self, context, **kwargs):
        self._pending_load_tasks["async"] = PendingLoadTask(
            task=object(), request_id="async", vllm_block_ids=[2]
        )

    monkeypatch.setattr(UCMDirectConnector, "start_load_kv", submit)
    worker.start_load_kv(None)
    assert events == [({"sync"}, "handoff")]
    assert worker.get_finished(set()) == (None, None)
    assert events == [({"sync"}, "handoff")]
    assert worker.get_finished(set()) == (None, {"async"})
    reason = "load_failure" if failure else "handoff"
    assert events == [({"sync"}, "handoff"), "wait", ({"async"}, reason)]
    assert not worker._pending_load_tasks
    assert worker._connector_worker_meta.load_failed_reqs == (
        {"async"} if failure else set()
    )


@pytest.mark.parametrize("connector", [EarlyDirectConnector], indirect=True)
def test_native_async_dispatch_keeps_early_key_without_forward_tokens(connector):
    request = Request()
    job = prepare(connector, request)
    connector.use_request_async_load = True
    connector._async_dump_req_ids = set()
    connector._early.dp_rank = 0
    tokens, load_async = connector.get_num_new_matched_tokens(request, 0)
    assert (tokens, load_async) == (7, True)
    blocks = SimpleNamespace(get_block_ids=lambda: ([10, 11],))
    connector.update_state_after_alloc(request, blocks, tokens)
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )
    metadata = connector.build_connector_meta(output)
    dispatch = metadata.request_meta[request.request_id]
    assert dispatch.load_async
    assert dispatch.load_block_ids == (job.blocks, [10, 11])
    assert dispatch.dump_block_ids == ([], [])
    assert metadata.early_keys == {request.request_id: job.key}
    assert job.dispatched
    assert not connector._pending_async_load_dispatches
    connector.request_hasher.assert_not_called()
    connector.store.store_.LookupOnPrefix.assert_not_called()
