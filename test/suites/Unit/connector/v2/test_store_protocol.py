"""Connector v2 uses the native Store's snapshots, windows and receipts."""

from dataclasses import replace
from itertools import count
from types import SimpleNamespace as NS
from unittest.mock import Mock, call

import pytest

pytestmark = [pytest.mark.stage(0), pytest.mark.feature("connector_v2")]


@pytest.mark.parametrize(
    ("mode", "eager", "draft_tokens", "expected"),
    [
        ("NONE", False, 3, 1),
        ("PIECEWISE", False, 3, 1),
        ("FULL", False, 0, 2),
        ("FULL_DECODE_ONLY", False, 3, 5),
        ("FULL_AND_PIECEWISE", False, 3, 5),
        ("FULL", True, 3, 1),
    ],
)
def test_connector_sets_recompute_tokens_from_graph_mode(
    monkeypatch, mode, eager, draft_tokens, expected
):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from ucm.integration.vllm.v2 import ucm_connector

    context = Mock()
    monkeypatch.setattr(
        ucm_connector.UCMRuntimeContext, "from_vllm_config", Mock(return_value=context)
    )
    scheduler = Mock()
    monkeypatch.setattr(ucm_connector, "UCMConnectorScheduler", scheduler)
    kv_cache_config = NS(kv_cache_groups=())
    config = NS(
        compilation_config=NS(cudagraph_mode=CUDAGraphMode[mode]),
        model_config=NS(enforce_eager=eager),
        speculative_config=(
            NS(num_speculative_tokens=draft_tokens) if draft_tokens else None
        ),
        kv_transfer_config=NS(),
    )

    ucm_connector.UCMConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
    scheduler.assert_called_once_with(context, recompute_tokens=expected)
    assert not ucm_connector.UCMConnector.requires_piecewise_for_cudagraph({})


def test_store_runtime_passes_through_engine_id_for_both_roles():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from ucm.integration.vllm.v2.runtime import (
        ParallelRank,
        ParallelTopology,
        UCMRuntimeContext,
    )

    # Default vLLM UUID followed by the Ascend suffix from the remote repro.
    engine_id = "04807027-2f9b-4387-b71e-5c265a5ac995-" + "a" * 32
    worker = UCMRuntimeContext(
        role=KVConnectorRole.WORKER,
        device_type="npu",
        engine_id=engine_id,
        global_rank=0,
        rank=ParallelRank(),
        parallel_topology=ParallelTopology(tp_size=2, pp_size=2, pcp_size=2),
        launch_config={},
        use_layerwise=True,
        spec=NS(),
        fingerprint="",
    )
    scheduler = replace(worker, role=KVConnectorRole.SCHEDULER, rank=None)

    assert worker.store_runtime(4096)["unique_id"] == engine_id
    assert scheduler.store_runtime(4096)["unique_id"] == engine_id
    assert scheduler.store_runtime(4096)["worker_number"] == 8
    other_dp = replace(worker, engine_id=engine_id + "_dp1")
    assert other_dp.store_runtime(4096)["unique_id"] == engine_id + "_dp1"


def test_cache_worker_uses_largest_complete_record_without_reducing(monkeypatch):
    import torch.distributed as dist

    from ucm.integration.vllm.v2 import connector_worker
    from ucm.integration.vllm.v2.kv_cache_spec import RestorePolicy
    from ucm.integration.vllm.v2.runtime import ParallelRank, ParallelTopology

    worker = object.__new__(connector_worker.UCMConnectorWorker)
    worker.context = NS(launch_config={}, store_runtime=Mock())
    worker.spec = NS()
    worker.parallel_topology = ParallelTopology(tp_size=2, pcp_size=2)
    worker._parallel_rank = ParallelRank()
    worker._kv_cache_config = NS()
    worker._num_hidden_layers = 1
    worker._store_worker = None
    worker._device = None
    layout = Mock(
        store_sizes_by_policy={RestorePolicy.FA: 4096, RestorePolicy.WA: 8192}
    )
    monkeypatch.setattr(connector_worker, "UCMKVCacheLayout", Mock(return_value=layout))
    client = Mock()
    monkeypatch.setattr(connector_worker, "StoreWorkerClient", client)
    all_reduce = Mock()
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    worker.register_kv_caches({})

    all_reduce.assert_not_called()
    worker.context.store_runtime.assert_called_once_with(8192, device_id=0)
    client.assert_called_once_with({}, worker.context.store_runtime.return_value)
    handshake = worker.get_handshake_metadata()
    assert handshake.cache_block_size == 8192


def test_cache_scheduler_uses_reduced_slot_size_from_handshake(monkeypatch):
    from ucm.integration.vllm.v2 import connector_scheduler
    from ucm.integration.vllm.v2.kv_cache_spec import UCMKVCacheSpec
    from ucm.integration.vllm.v2.runtime import ParallelRank, UCMStorageHandshake

    scheduler = object.__new__(connector_scheduler.UCMConnectorScheduler)
    scheduler.context = NS(launch_config={}, store_runtime=Mock(), fingerprint="pp")
    scheduler._store_scheduler = None
    shared_spec = UCMKVCacheSpec((), 128, "cpu")
    client = Mock()
    monkeypatch.setattr(connector_scheduler, "StoreSchedulerClient", client)
    scheduler.set_xfer_handshake_metadata(
        {
            (0, 0): UCMStorageHandshake(ParallelRank(), 8192, shared_spec),
            (1, 0): UCMStorageHandshake(ParallelRank(pp_rank=1), 8192, shared_spec),
        }
    )

    scheduler.context.store_runtime.assert_called_once_with(8192)
    client.assert_called_once_with({}, scheduler.context.store_runtime.return_value)
    assert scheduler.spec == shared_spec


def test_pp_stages_write_disjoint_shards_into_the_same_key(monkeypatch):
    import numpy as np
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    from ucm.integration.vllm.v2 import kv_cache_layout
    from ucm.integration.vllm.v2.connector_scheduler import UCMDispatchPlan
    from ucm.integration.vllm.v2.connector_worker import UCMConnectorWorker
    from ucm.integration.vllm.v2.group_layout import GroupLayoutMetadata
    from ucm.integration.vllm.v2.kv_cache_spec import (
        KVCacheSpecKind,
        RestorePolicy,
        UCMKVCacheGroupSpec,
        UCMKVCacheSpec,
    )
    from ucm.integration.vllm.v2.runtime import (
        ParallelRank,
        ParallelTopology,
        make_ucm_chunk_ids,
    )
    from ucm.integration.vllm.v2.store_layout import UCMShard

    topology = ParallelTopology(tp_size=2, pp_size=2, pcp_size=2)
    spec = UCMKVCacheSpec(
        tuple(
            UCMKVCacheGroupSpec(i, 16, frozenset({KVCacheSpecKind.FULL_ATTENTION}))
            for i in range(2)
        ),
        32,
        "cpu",
    )
    # One group spans both stages with unequal stage widths. The other has
    # only PP1 layers and uses a TP-replicated shard.
    pp_metadata = [
        {0: GroupLayoutMetadata((64,), (16,), True, 2)},
        {
            0: GroupLayoutMetadata((96,), (16,), True, 2),
            1: GroupLayoutMetadata((32,), (16,), True, 1),
        },
    ]
    reports = [
        {
            group_id: (spec.groups[group_id], metadata)
            for group_id, metadata in stage.items()
        }
        for stage in pp_metadata
    ]
    pp_cpu_group = object()
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: NS(cpu_group=pp_cpu_group),
        raising=False,
    )

    def gather(output, local_report, *, group):
        assert group is pp_cpu_group
        assert local_report in reports
        output[:] = reports

    monkeypatch.setattr(dist, "all_gather_object", gather)
    keys = make_ucm_chunk_ids((b"a" * 15, b"b" * 15), RestorePolicy.FA)
    plan = UCMDispatchPlan(
        RestorePolicy.FA,
        keys,
        np.array([0, 32], dtype=np.uint64),
        (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])),
        (0, 1),
    )
    ranges = {key: [] for key in keys}
    for pp, stage in enumerate(pp_metadata):
        for tp in range(topology.tp_size):
            for pcp in range(topology.pcp_size):
                rank = ParallelRank(tp_rank=tp, pp_rank=pp, pcp_rank=pcp)
                layout = object.__new__(kv_cache_layout.UCMKVCacheLayout)
                layout.spec = spec
                layout.group_layouts = {
                    gid: NS(
                        export_metadata=Mock(return_value=metadata),
                        segment_mask=lambda _: None,
                        resolve_ptrs=lambda ids, **_: np.asarray(ids, dtype=np.uint64)[
                            :, None
                        ],
                        tokens_to_bytes=lambda starts, *_: starts[:, None],
                    )
                    for gid, metadata in stage.items()
                }
                layout.build_store_layouts(topology, rank, chunk_size=32)
                assert layout.store_sizes_by_policy == {RestorePolicy.FA: 704}
                assert layout.store_offsets_by_group == {0: 0, 1: 640}
                assert layout.store_layouts[0].shard == UCMShard(tp, 0, pp)
                worker = object.__new__(UCMConnectorWorker)
                worker.spec = spec
                worker.layout = layout
                batches = worker._build_transfer_batches(plan, is_dump=True)
                for batch in batches:
                    for key, offsets, sizes in zip(
                        batch.ucm_keys,
                        batch.store_offsets,
                        batch.byte_sizes,
                        strict=True,
                    ):
                        ranges[key].extend(
                            (int(offset), int(size))
                            for offset, size in zip(offsets, sizes, strict=True)
                        )
    assert (
        sorted(ranges[keys[0]])
        == sorted(ranges[keys[1]])
        == [
            (0, 64),
            (64, 96),
            (160, 64),
            (224, 96),
            (320, 64),
            (384, 96),
            (480, 64),
            (544, 96),
            (640, 32),
            (672, 32),
        ]
    )


def test_lookup_checks_boundary_keys_only_within_the_fa_hit():
    from ucm.integration.vllm.v2.connector_scheduler import UCMConnectorScheduler
    from ucm.integration.vllm.v2.kv_cache_spec import RestorePolicy

    scheduler = object.__new__(UCMConnectorScheduler)
    scheduler.spec = NS(chunk_size=128)
    scheduler._recompute_tokens = 1
    fa, wa, state = RestorePolicy
    scheduler.groups_by_policy = {fa: (), wa: (), state: ()}
    hashes = (b"a" * 15, b"b" * 15, b"c" * 15, b"d" * 15)
    groups = scheduler._lookup_key_groups(hashes)
    available = set(groups[fa][:3])
    available.update(groups[wa][i] for i in (0, 2))
    available.update(groups[state][:2])
    store = scheduler._store_scheduler = Mock()
    store.lookup_on_prefix.side_effect = lambda request, keys: (
        next(
            (i - 1 for i, key in enumerate(keys) if key not in available), len(keys) - 1
        ),
        False,
    )

    def lookup_on_reverse(request, keys):
        index = next(
            (i for i in reversed(range(len(keys))) if keys[i] in available), -1
        )
        return index, keys == groups[wa][:2]

    store.lookup_on_reverse.side_effect = lookup_on_reverse

    assert scheduler._lookup_hit_tokens("r", hashes, 513, 0) == (128, (groups[wa][1],))
    store.lookup_on_prefix.assert_called_once_with("r", groups[fa])
    assert store.lookup_on_reverse.call_args_list == [
        call("r", groups[wa][:3]),
        call("r", groups[state][:3]),
        call("r", groups[wa][:2]),
        call("r", groups[state][:1]),
        call("r", groups[wa][:1]),
        call("r", groups[state][:1]),
    ]
    store.reset_mock()
    available.add(groups[state][2])
    assert scheduler._lookup_hit_tokens("r", hashes, 513, 128) == (256, ())
    store.lookup_on_prefix.assert_called_once_with("r", groups[fa][1:])
    assert store.lookup_on_reverse.call_args_list == [
        call("r", groups[wa][1:3]),
        call("r", groups[state][1:3]),
    ]
    store.lookup.assert_not_called()


def test_fa_miss_only_offers_fa_prefetch_until_the_boundary_is_known():
    from ucm.integration.vllm.v2.connector_scheduler import UCMConnectorScheduler
    from ucm.integration.vllm.v2.kv_cache_spec import RestorePolicy

    scheduler = object.__new__(UCMConnectorScheduler)
    scheduler.spec = NS(chunk_size=128)
    scheduler._recompute_tokens = 1
    fa, wa = RestorePolicy.FA, RestorePolicy.WA
    scheduler.groups_by_policy = {fa: (), wa: ()}
    hashes = (b"a" * 15, b"b" * 15)
    groups = scheduler._lookup_key_groups(hashes)
    fa_keys = groups[fa]
    boundary_keys = groups[wa]
    store = scheduler._store_scheduler = Mock()
    store.lookup_on_prefix.side_effect = [(-1, True), (1, False)]
    store.lookup_on_reverse.return_value = (-1, True)

    assert scheduler._lookup_hit_tokens("r", hashes, 257, 0) == (0, fa_keys)
    store.lookup_on_reverse.assert_not_called()
    store.lookup.assert_not_called()
    assert scheduler._lookup_hit_tokens("r", hashes, 257, 0) == (0, boundary_keys)
    assert store.lookup_on_prefix.call_args_list == [
        call("r", fa_keys),
        call("r", fa_keys),
    ]
    store.lookup_on_reverse.assert_called_once_with("r", boundary_keys)


def test_store_client_passes_explicit_prefetch_keys_to_the_binding():
    from ucm.integration.vllm.v2.store_client import (
        AddressedToken,
        StoreSchedulerClient,
    )

    client = object.__new__(StoreSchedulerClient)
    native_scheduler = client._scheduler = Mock()
    client._proxy = NS(UcmError=RuntimeError)
    keys = (b"a" * 16, b"b" * 16)
    native_scheduler.prepare_prefetch.return_value = NS(
        keys=[NS(to_bytes=Mock(return_value=key)) for key in keys], payload=b"plan"
    )

    assert client.prepare_prefetch("r", keys) == AddressedToken(keys, b"plan")
    native_scheduler.prepare_prefetch.assert_called_once_with("r", b"".join(keys))


def test_prefetch_rejection_keeps_the_hit_and_allocation_owns_the_get():
    from ucm.integration.vllm.v2.connector_scheduler import UCMConnectorScheduler
    from ucm.integration.vllm.v2.store_client import StoreIOError

    scheduler = object.__new__(UCMConnectorScheduler)
    store = scheduler._store_scheduler = Mock()
    store.prepare_prefetch.side_effect = StoreIOError("no space")
    scheduler._job_ids = count(1, 2)
    scheduler._request_states = {}
    scheduler._pending_jobs = {}
    scheduler._chunk_hasher = Mock(return_value=(b"a" * 16,))
    scheduler._lookup_hit_tokens = Mock(return_value=(128, (b"fa",)))
    scheduler.load_threshold_tokens = 0
    request = NS(
        request_id="r",
        num_preemptions=0,
        all_token_ids=range(257),
        skip_reading_prefix_cache=False,
    )
    scheduler.on_new_request(request)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (128, False)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (128, False)
    assert store.prepare_prefetch.call_args_list == [
        call("r", (b"fa",)),
        call("r", (b"fa",)),
    ]
    state = scheduler._request_states["r"]
    assert not state.needs_load
    scheduler.update_state_after_alloc(request, 128)
    assert state.needs_load and state.external_hit_tokens == 128
    scheduler.update_state_after_alloc(request, 0)
    assert not state.needs_load and state.external_hit_tokens == 0


def test_resume_does_not_wait_for_old_get_and_put_results():
    from ucm.integration.vllm.v2.connector_scheduler import (
        JobOperation,
        RequestState,
        UCMConnectorScheduler,
    )
    from ucm.integration.vllm.v2.connector_worker import JobResult, UCMWorkerMetadata
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import AddressedToken

    scheduler = object.__new__(UCMConnectorScheduler)
    store = scheduler._store_scheduler = Mock()
    request = NS(
        request_id="r",
        num_preemptions=1,
        all_token_ids=range(385),
        skip_reading_prefix_cache=False,
    )
    state = RequestState(request, num_preemptions=0)
    scheduler._request_states = {"r": state}
    scheduler._pending_jobs = {}
    scheduler._offload_jobs = {}
    rank = ParallelRank()
    scheduler._expected_worker_ranks = {rank}
    put_token = AddressedToken((b"put",), b"old-put")
    get_token = AddressedToken((b"get",), b"old-get")
    scheduler._track_prepared_job(1, JobOperation.PUT, "r", put_token)
    scheduler._track_prepared_job(3, JobOperation.GET, "r", get_token)
    scheduler._chunk_hasher = Mock(return_value=(b"a" * 16, b"b" * 16))
    scheduler._lookup_hit_tokens = Mock(return_value=(0, ()))
    scheduler.load_threshold_tokens = 0

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    store.on_request_finished.assert_not_called()
    store.on_new_request.assert_not_called()
    store.complete_put.assert_not_called()
    assert scheduler._request_states["r"] is state
    assert state.num_preemptions == request.num_preemptions
    assert len(state.common_chunk_hashes) == 2
    scheduler.update_state_after_alloc(request, 128)
    scheduler.update_connector_output(
        NS(
            kv_connector_worker_meta=UCMWorkerMetadata(
                job_results={JobResult(1, rank, True), JobResult(3, rank, True)}
            )
        )
    )
    store.complete_put.assert_called_once_with("r", put_token, True)
    store.complete_get.assert_called_once_with("r", get_token)
    assert not scheduler._pending_jobs
    assert state.external_hit_tokens == 128 and state.needs_load


def test_skip_reading_prefix_cache_does_not_prepare_prefetch():
    from ucm.integration.vllm.v2.connector_scheduler import UCMConnectorScheduler

    scheduler = object.__new__(UCMConnectorScheduler)
    store = scheduler._store_scheduler = Mock()
    scheduler._request_states = {}
    scheduler._chunk_hasher = Mock(return_value=(b"a" * 16,))
    scheduler._lookup_hit_tokens = Mock(return_value=(0, (b"fa",)))
    request = NS(
        request_id="r",
        num_preemptions=0,
        all_token_ids=range(257),
        skip_reading_prefix_cache=True,
    )
    scheduler.on_new_request(request)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler._lookup_hit_tokens.assert_not_called()
    store.prepare_prefetch.assert_not_called()
    assert scheduler._request_states["r"].num_tokens == 257


def test_load_failure_only_invalidates_dump_jobs_in_the_same_output():
    from ucm.integration.vllm.v2.connector_scheduler import (
        JobOperation,
        RequestState,
        UCMConnectorScheduler,
    )
    from ucm.integration.vllm.v2.connector_worker import JobResult, UCMWorkerMetadata
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import AddressedToken

    scheduler = object.__new__(UCMConnectorScheduler)
    store = scheduler._store_scheduler = Mock()
    scheduler._pending_jobs = {}
    scheduler._offload_jobs = {}
    rank = ParallelRank()
    scheduler._expected_worker_ranks = {rank}
    scheduler._request_states = {"r": RequestState(NS(num_computed_tokens=0))}
    failed_token = AddressedToken((b"old",), b"old-put")
    other_token = AddressedToken((b"new",), b"new-put")
    scheduler._track_prepared_job(1, JobOperation.PUT, "r", failed_token)
    scheduler._track_prepared_job(3, JobOperation.PUT, "r", other_token)

    scheduler.update_connector_output(
        NS(
            kv_connector_worker_meta=UCMWorkerMetadata(
                failed_load_request_ids={"r"},
                job_results={JobResult(1, rank, True)},
            )
        )
    )
    store.complete_put.assert_called_once_with("r", failed_token, False)

    scheduler.update_connector_output(
        NS(
            kv_connector_worker_meta=UCMWorkerMetadata(
                job_results={JobResult(3, rank, True)}
            )
        )
    )
    assert store.complete_put.call_args_list == [
        call("r", failed_token, False),
        call("r", other_token, True),
    ]


def test_plans_prepare_separately_without_filtering_put_keys():
    import numpy as np

    from ucm.integration.vllm.v2.connector_scheduler import (
        RequestState,
        UCMConnectorScheduler,
        UCMDispatchPlan,
    )
    from ucm.integration.vllm.v2.kv_cache_spec import RestorePolicy
    from ucm.integration.vllm.v2.store_client import AddressedToken

    scheduler = object.__new__(UCMConnectorScheduler)
    store = scheduler._store_scheduler = Mock()
    scheduler._pending_jobs = {}
    scheduler._job_ids = count(1, 2)
    store.prepare_get.side_effect = lambda request, keys: AddressedToken(keys, b"get")
    store.prepare_put.side_effect = lambda request, keys: AddressedToken(keys, b"put")
    state = RequestState(
        NS(request_id="r", num_computed_tokens=0), num_preemptions=0, num_tokens=256
    )
    plans = (
        UCMDispatchPlan(RestorePolicy.FA, (b"fa", b"fa2"), np.array([0, 128]), (), ()),
        UCMDispatchPlan(RestorePolicy.WA, (b"wa",), np.array([128]), (), ()),
    )
    scheduler.build_dispatch_plans = Mock(return_value=plans)
    scheduler._state_dump_plans = Mock(return_value=())

    request = scheduler._build_request_plan("r", state, (), 256)

    assert store.prepare_get.call_args_list == [
        call("r", (b"fa", b"fa2")),
        call("r", (b"wa",)),
    ]
    assert store.prepare_put.call_args_list == [
        call("r", (b"fa", b"fa2")),
        call("r", (b"wa",)),
    ]
    jobs = [plan.prepared_job for plan in request.load_plans + request.dump_plans]
    assert [job.ucm_keys for job in jobs] == [
        (b"fa", b"fa2"),
        (b"wa",),
        (b"fa", b"fa2"),
        (b"wa",),
    ]
    assert len({job.job_id for job in jobs}) == len(jobs)
    store.lookup.assert_not_called()


def test_bulk_transfer_submits_each_plan_with_its_own_job():
    from ucm.integration.vllm.v2.connector_scheduler import (
        RequestMetadata,
        UCMConnectorMetadata,
    )
    from ucm.integration.vllm.v2.connector_worker import UCMConnectorWorker
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import AddressedToken, PreparedJob

    worker = object.__new__(UCMConnectorWorker)
    worker._parallel_rank = ParallelRank()
    worker.use_layerwise = False
    worker._failed_dump_jobs = set()
    worker._completion_metadata = NS(failed_load_request_ids=set())
    worker._layer_load_tasks = {}
    worker._store_worker = Mock()
    worker._record_transfer_event = Mock(return_value=0)
    worker._wait_layer_load = Mock()
    worker._submit_dump = Mock()
    jobs = tuple(
        PreparedJob(i, (key,), AddressedToken((key,), b""))
        for i, key in enumerate((b"fa", b"wa", b"fa", b"wa"), start=1)
    )
    plans = tuple(NS(prepared_job=job, batch=object()) for job in jobs)
    worker._build_transfer_batches = lambda plan, **kwargs: (plan.batch,)
    request = RequestMetadata("r", plans[:2], plans[2:])

    worker.start_load_kv(UCMConnectorMetadata(requests={"r": request}), None)
    worker._submit_request_dumps(request)
    assert worker._store_worker.submit_get.call_args_list == [
        call(
            plan.prepared_job, (plan.batch,), is_layerwise=False, prerequisite_handle=0
        )
        for plan in request.load_plans
    ]
    assert worker._submit_dump.call_args_list == [
        call(plan.prepared_job, (plan.batch,), is_layerwise=False)
        for plan in request.dump_plans
    ]


def test_start_load_kv_leaves_attention_gets_for_layer_hooks():
    from ucm.integration.vllm.v2.connector_scheduler import (
        RequestMetadata,
        UCMConnectorMetadata,
    )
    from ucm.integration.vllm.v2.connector_worker import (
        UCMConnectorWorker,
        UCMWorkerMetadata,
    )
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import TransferTask

    worker = object.__new__(UCMConnectorWorker)
    worker._parallel_rank = ParallelRank()
    worker.use_layerwise = True
    worker.spec = NS(state_groups=())
    worker.layout = NS(layer_ids=(0, 1))
    worker._completion_metadata = UCMWorkerMetadata()
    worker._layer_load_tasks = {}
    tasks = [TransferTask(1, 2), TransferTask(1, 4)]

    def submit(request, plan, layer_id):
        worker._layer_load_tasks[layer_id] = [(request, tasks[layer_id])]

    worker._submit_load = submit
    worker._wait_transfer = Mock()
    metadata = UCMConnectorMetadata(
        requests={"r": RequestMetadata("r", load_plans=(NS(),))}
    )
    worker.start_load_kv(metadata, None)

    worker._wait_transfer.assert_not_called()
    assert set(worker._layer_load_tasks) == {0, 1}


@pytest.mark.parametrize("failure", ["submit", "wait"])
def test_dump_reports_failure_after_draining(failure):
    from ucm.integration.vllm.v2.connector_scheduler import (
        RequestMetadata,
        UCMConnectorMetadata,
    )
    from ucm.integration.vllm.v2.connector_worker import (
        JobResult,
        UCMConnectorWorker,
        UCMWorkerMetadata,
    )
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import StoreIOError, TransferTask

    worker = object.__new__(UCMConnectorWorker)
    worker._parallel_rank = ParallelRank()
    worker.use_layerwise = False
    worker._failed_dump_jobs = set()
    worker._completion_metadata = UCMWorkerMetadata()
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._device = Mock()
    worker._record_transfer_event = Mock(return_value=42)
    worker._store_worker = Mock()
    worker._build_transfer_batches = Mock(return_value=(object(),))
    task = TransferTask(1, None, prerequisite_handle=42)
    if failure == "submit":
        worker._store_worker.submit_put.side_effect = StoreIOError("rejected")
    else:
        worker._store_worker.submit_put.return_value = task

        def wait(task):
            task.success = False
            raise StoreIOError("failed")

        worker._store_worker.wait.side_effect = wait

    plan = NS(prepared_job=NS(job_id=1, ucm_keys=(b"key",)))
    metadata = UCMConnectorMetadata(
        requests={"r": RequestMetadata("r", dump_plans=(plan,))}
    )
    worker.wait_for_save(metadata)
    assert not worker._dump_tasks
    worker._device.destroy_event_handle.assert_called_once_with(42)
    assert worker.build_connector_worker_meta().job_results == {
        JobResult(1, ParallelRank(), False)
    }


@pytest.mark.parametrize("operation", ["get", "put"])
def test_each_window_completes_with_its_token_after_all_ranks(operation):
    from ucm.integration.vllm.v2.connector_scheduler import (
        JobOperation,
        RequestState,
        UCMConnectorScheduler,
    )
    from ucm.integration.vllm.v2.connector_worker import JobResult, UCMWorkerMetadata
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import (
        AddressedToken,
        StoreSchedulerClient,
    )

    class NativeToken:
        pass

    client = object.__new__(StoreSchedulerClient)
    native_scheduler = client._scheduler = Mock()
    client._proxy = NS(AddressedToken=NativeToken, BlockId=bytes, UcmError=RuntimeError)
    complete = getattr(native_scheduler, f"complete_{operation}")
    scheduler = object.__new__(UCMConnectorScheduler)
    scheduler._store_scheduler = client
    scheduler._pending_jobs = {}
    rank0, rank1 = ParallelRank(), ParallelRank(pp_rank=1)
    scheduler._expected_worker_ranks = {rank0, rank1}
    scheduler._offload_jobs = {}
    scheduler._request_states = {"r": RequestState(NS(request_id="r"))}
    op = JobOperation[operation.upper()]
    first = scheduler._track_prepared_job(1, op, "r", AddressedToken((b"fa",), b"A"))
    second = scheduler._track_prepared_job(3, op, "r", AddressedToken((b"wa",), b"B"))

    def report(job, rank, success):
        scheduler.update_connector_output(
            NS(
                kv_connector_worker_meta=UCMWorkerMetadata(
                    job_results={JobResult(job.job_id, rank, success)}
                )
            )
        )

    report(first, rank0, True)
    report(second, rank0, False)
    complete.assert_not_called()
    report(second, rank1, True)
    assert set(scheduler._pending_jobs) == {first.job_id}
    report(first, rank1, True)
    assert not scheduler._pending_jobs
    assert [args[0] for args, _ in complete.call_args_list] == ["r", "r"]
    assert [(args[1].keys, args[1].payload) for args, _ in complete.call_args_list] == [
        ([b"wa"], b"B"),
        ([b"fa"], b"A"),
    ]
    assert [args[2:] for args, _ in complete.call_args_list] == (
        [(False,), (True,)] if operation == "put" else [(), ()]
    )


@pytest.mark.parametrize("success", [True, False])
def test_offload_runs_without_a_live_request_and_releases_its_pin(success):
    from ucm.integration.vllm.v2.connector_scheduler import (
        JobOperation,
        RequestState,
        UCMConnectorScheduler,
    )
    from ucm.integration.vllm.v2.connector_worker import (
        JobResult,
        UCMConnectorWorker,
        UCMWorkerMetadata,
    )
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import AddressedToken

    scheduler = object.__new__(UCMConnectorScheduler)
    store = scheduler._store_scheduler = Mock()
    scheduler._pending_jobs = {}
    scheduler._job_ids = count(1, 2)
    scheduler._expected_worker_ranks = {ParallelRank()}
    scheduler._needs_offload_drain = False
    scheduler._offload_jobs = {}
    scheduler._request_states = {"r": RequestState(NS(request_id="r"), finished=True)}
    token = AddressedToken((b"key",), b"offload")
    store.prepare_offload.return_value = (token,)
    put = scheduler._track_prepared_job(
        scheduler._allocate_job_id(), JobOperation.PUT, "r", token
    )

    def report(job, succeeded):
        scheduler.update_connector_output(
            NS(
                kv_connector_worker_meta=UCMWorkerMetadata(
                    job_results={JobResult(job.job_id, ParallelRank(), succeeded)}
                )
            )
        )

    report(put, True)
    store.complete_put.assert_called_once_with("r", token, True)
    store.on_request_finished.assert_called_once_with("r")
    assert scheduler.has_pending_push_work()
    output = NS(
        kv_connector_block_state=NS(req_ids=()),
        preempted_req_ids=set(),
        finished_req_ids=set(),
    )
    metadata = scheduler.build_connector_meta(output)
    (job,) = metadata.offload_jobs
    assert not metadata.requests
    assert scheduler.has_pending_push_work()
    assert not scheduler.build_connector_meta(output).offload_jobs
    store.complete_offload.assert_not_called()

    worker = object.__new__(UCMConnectorWorker)
    worker._parallel_rank = ParallelRank()
    worker.use_layerwise = False
    worker._prefetch_offload_tasks = {}
    worker._completion_metadata = UCMWorkerMetadata()
    worker._store_worker = Mock()
    task = NS(success=None)
    worker._store_worker.submit_offload.return_value = task
    worker.start_load_kv(metadata, None)
    worker._store_worker.submit_offload.assert_called_once_with(job)
    worker.poll_finished()
    assert worker.build_connector_worker_meta() is None

    task.success = success
    worker.poll_finished()
    scheduler.update_connector_output(
        NS(kv_connector_worker_meta=worker.build_connector_worker_meta())
    )
    store.complete_offload.assert_called_once_with(token)
    assert not scheduler.has_pending_push_work()


@pytest.mark.parametrize(
    ("status", "terminal"),
    [(None, True), ("General (1):: IO failure", True), ("InvalidParam (2)", False)],
)
def test_wait_consumes_a_single_native_job(status, terminal):
    from ucm.integration.vllm.v2.store_client import (
        StoreIOError,
        StoreWorkerClient,
        TransferTask,
    )

    class UcmError(RuntimeError):
        pass

    client = object.__new__(StoreWorkerClient)
    client._proxy = NS(UcmError=UcmError)
    client._worker = Mock()
    task = TransferTask(1, 2, prerequisite_handle=42)
    client._tasks = {2: task}
    if status is None:
        client.wait(task)
        client.wait(task)
    else:
        client._worker.wait.side_effect = UcmError(status)
        with pytest.raises(StoreIOError if terminal else UcmError):
            client.wait(task)
    client._worker.wait.assert_called_once_with(2)
    assert (2 not in client._tasks) is terminal
    assert (task.success is not None) is terminal
    assert task.prerequisite_handle == 42
