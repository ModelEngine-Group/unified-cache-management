"""Prefetch can outlive a worker step without delaying synchronous saves."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

pytestmark = [pytest.mark.stage(0), pytest.mark.feature("connector_v2")]


def make_worker(rank):
    from ucm.integration.vllm.v2.connector_worker import (
        UCMConnectorWorker,
        UCMWorkerMetadata,
    )

    worker = object.__new__(UCMConnectorWorker)
    worker._parallel_rank = rank
    worker._store_worker = Mock()
    worker._prefetch_offload_tasks = {}
    worker._async_loads = {}
    worker._finished_loads = {}
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._dump_submitted_layer_ids = set()
    worker._failed_dump_jobs = set()
    worker._completion_metadata = UCMWorkerMetadata()
    return worker


@pytest.mark.parametrize("success", [True, False])
def test_async_bulk_load_survives_empty_steps_and_finishes_all_plans(success):
    from ucm.integration.vllm.v2.connector_scheduler import (
        RequestMetadata,
        UCMConnectorMetadata,
    )
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import PreparedJob

    worker = make_worker(ParallelRank())
    worker.use_layerwise = False
    worker._device = None
    worker._invalid_vllm_block_ids = set()
    worker._record_transfer_event = Mock(return_value=0)
    worker._build_transfer_batches = Mock(return_value=(object(),))
    tasks = [
        SimpleNamespace(success=None, prerequisite_handle=0),
        SimpleNamespace(success=None, prerequisite_handle=0),
    ]
    worker._store_worker.submit_get.side_effect = tasks
    plans = tuple(
        SimpleNamespace(prepared_job=PreparedJob(job_id, (), None)) for job_id in (1, 3)
    )
    request = RequestMetadata("r", load_plans=plans, load_async=True)
    metadata = UCMConnectorMetadata(requests={"r": request})
    worker.begin_step(metadata)
    worker.start_load_kv(metadata, None)
    worker.wait_for_save(metadata)
    worker._store_worker.wait.assert_not_called()
    assert worker.build_connector_worker_meta() is None

    tasks[0].success = success
    empty = UCMConnectorMetadata()
    worker.begin_step(empty)
    worker.wait_for_save(empty)
    worker.poll_finished()
    assert worker.take_finished_loads() == {}
    assert worker.build_connector_worker_meta() is None

    tasks[1].success = True
    worker.poll_finished()
    assert worker.take_finished_loads() == {"r": success}
    results = worker.build_connector_worker_meta()
    assert {(r.job_id, r.success) for r in results.job_results} == {
        (1, success),
        (3, success),
    }
    assert results.failed_load_request_ids == (set() if success else {"r"})
    assert not worker.get_block_ids_with_load_errors()
    worker.poll_finished()
    assert worker.take_finished_loads() == {}
    assert worker.build_connector_worker_meta() is None


def test_rejected_async_submit_drains_accepted_jobs_before_reporting_failure():
    from ucm.integration.vllm.v2.connector_scheduler import (
        RequestMetadata,
        UCMConnectorMetadata,
    )
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import PreparedJob, StoreIOError

    worker = make_worker(ParallelRank())
    worker.use_layerwise = False
    worker._record_transfer_event = Mock(return_value=0)
    worker._build_transfer_batches = Mock(return_value=(object(),))
    task = SimpleNamespace(success=None, prerequisite_handle=0)
    worker._store_worker.submit_get.side_effect = [task, StoreIOError("rejected")]
    plans = tuple(
        SimpleNamespace(prepared_job=PreparedJob(job_id, (), None)) for job_id in (1, 3)
    )
    metadata = UCMConnectorMetadata(
        requests={"r": RequestMetadata("r", load_plans=plans, load_async=True)}
    )
    worker.start_load_kv(metadata, None)
    worker.poll_finished()
    assert worker.take_finished_loads() == {}
    task.success = True
    worker.poll_finished()
    assert worker.take_finished_loads() == {"r": False}
    assert {r.job_id for r in worker.build_connector_worker_meta().job_results} == {
        1,
        3,
    }


@pytest.mark.parametrize("prepare_failed", [True, False])
def test_async_load_without_local_io_still_reports_receiving(prepare_failed):
    from ucm.integration.vllm.v2.connector_scheduler import (
        RequestMetadata,
        UCMConnectorMetadata,
    )
    from ucm.integration.vllm.v2.runtime import ParallelRank

    worker = make_worker(ParallelRank())
    worker.use_layerwise = False
    worker._build_transfer_batches = Mock(return_value=())
    request = RequestMetadata(
        "r",
        load_plans=(SimpleNamespace(prepared_job=None),) if prepare_failed else (),
        load_async=True,
        prepare_get_failed=prepare_failed,
    )
    metadata = UCMConnectorMetadata(requests={"r": request})
    worker.begin_step(metadata)
    worker.start_load_kv(metadata, None)
    worker.poll_finished()
    worker._store_worker.submit_get.assert_not_called()
    assert worker.take_finished_loads() == {"r": not prepare_failed}
    assert not worker._async_loads


@pytest.mark.parametrize("prepare_fails", [True, False])
def test_async_get_dispatches_without_forward_and_cancel_keeps_its_window(
    prepare_fails,
):
    from itertools import count

    import numpy as np

    from ucm.integration.vllm.v2.connector_scheduler import (
        UCMConnectorScheduler,
        UCMDispatchPlan,
    )
    from ucm.integration.vllm.v2.connector_worker import JobResult, UCMWorkerMetadata
    from ucm.integration.vllm.v2.kv_cache_spec import RestorePolicy
    from ucm.integration.vllm.v2.runtime import ParallelRank
    from ucm.integration.vllm.v2.store_client import StoreIOError

    scheduler = object.__new__(UCMConnectorScheduler)
    scheduler.context = SimpleNamespace(load_async=True)
    store = scheduler._store_scheduler = Mock()
    store.prepare_get.return_value = b"get"
    if prepare_fails:
        store.prepare_get.side_effect = StoreIOError("miss")
    scheduler._request_states = {}
    scheduler._pending_jobs = {}
    scheduler._needs_offload_drain = False
    scheduler._job_ids = count(1, 2)
    rank0, rank1 = ParallelRank(), ParallelRank(tp_rank=1)
    scheduler._expected_worker_ranks = {rank0, rank1}
    scheduler._chunk_hasher = Mock(return_value=(b"a" * 16,))
    scheduler._lookup_hit_tokens = Mock(return_value=(128, ()))
    scheduler.load_threshold_tokens = 0
    plan = UCMDispatchPlan(RestorePolicy.FA, (b"a",), np.array([0]), (), ())
    scheduler.build_dispatch_plans = Mock(return_value=(plan,))
    request = SimpleNamespace(
        request_id="r",
        num_preemptions=0,
        all_token_ids=range(257),
        skip_reading_prefix_cache=False,
        num_computed_tokens=0,
    )
    scheduler.on_new_request(request)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (128, True)
    scheduler.update_state_after_alloc(request, 128, ((1,),))
    state = scheduler._request_states["r"]
    scheduler.update_state_after_alloc(request, 0)
    assert state.external_hit_tokens == 128 and not state.needs_load
    scheduler.request_finished("r")
    store.on_request_finished.assert_not_called()
    output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(req_ids=()),
        num_scheduled_tokens={},
    )
    metadata = scheduler.build_connector_meta(output).requests["r"]
    assert metadata.load_async and metadata.prepare_get_failed == prepare_fails
    assert metadata.load_plans and not metadata.dump_plans
    assert not scheduler.build_connector_meta(output).requests
    if prepare_fails:
        store.on_request_finished.assert_called_once_with("r")
    else:
        job_id = metadata.load_plans[0].prepared_job.job_id
        for rank in (rank0, rank1):
            scheduler.update_connector_output(
                SimpleNamespace(
                    kv_connector_worker_meta=UCMWorkerMetadata(
                        job_results={JobResult(job_id, rank, True)}
                    )
                )
            )
            if rank == rank0:
                store.complete_get.assert_not_called()
        store.complete_get.assert_called_once_with("r", b"get")
        store.on_request_finished.assert_called_once_with("r")


@pytest.mark.parametrize("success", [True, False])
def test_prefetch_survives_steps_and_reports_only_terminal_result(success):
    from ucm.integration.vllm.v2.runtime import ParallelRank

    rank = ParallelRank(tp_rank=0, pp_rank=0, pcp_rank=0)
    worker = make_worker(rank)
    task = SimpleNamespace(success=None)
    worker._store_worker.submit_prefetch.return_value = task
    job = SimpleNamespace(job_id=1)

    worker._execute_prefetch(job)
    worker._store_worker.wait.assert_not_called()
    worker.poll_finished()
    assert worker.build_connector_worker_meta() is None

    # Binding a later empty step preserves the pending prefetch task.
    worker.begin_step(SimpleNamespace(requests={}))
    assert worker._prefetch_offload_tasks == {1: task}

    task.success = success
    worker.poll_finished()
    metadata = worker.build_connector_worker_meta()
    assert {(r.job_id, r.rank, r.success) for r in metadata.job_results} == {
        (1, rank, success)
    }
    assert not worker._prefetch_offload_tasks
    worker.poll_finished()
    assert worker.build_connector_worker_meta() is None


def test_nonrepresentative_rank_acknowledges_without_submitting():
    from ucm.integration.vllm.v2.runtime import ParallelRank

    rank = ParallelRank(tp_rank=1, pp_rank=0, pcp_rank=0)
    worker = make_worker(rank)
    worker._execute_prefetch(SimpleNamespace(job_id=3))
    worker._store_worker.submit_prefetch.assert_not_called()
    result = next(iter(worker.build_connector_worker_meta().job_results))
    assert (result.job_id, result.rank, result.success) == (3, rank, True)


def test_failed_completion_poll_preserves_inflight_prefetch():
    from ucm.integration.vllm.v2.runtime import ParallelRank

    worker = make_worker(ParallelRank(tp_rank=0, pp_rank=0, pcp_rank=0))
    task = SimpleNamespace(success=None)
    worker._prefetch_offload_tasks[1] = task
    worker._store_worker.poll_finished.side_effect = RuntimeError("unknown state")
    with pytest.raises(RuntimeError, match="unknown state"):
        worker.poll_finished()
    assert worker._prefetch_offload_tasks == {1: task}
    assert worker.build_connector_worker_meta() is None


@pytest.mark.parametrize("prefetch_prepared", [True, False])
def test_finish_before_dispatch_unregisters_without_pending_jobs(prefetch_prepared):
    from ucm.integration.vllm.v2.connector_scheduler import (
        JobOperation,
        UCMConnectorScheduler,
    )
    from ucm.integration.vllm.v2.store_client import AddressedToken

    scheduler = object.__new__(UCMConnectorScheduler)
    store = scheduler._store_scheduler = Mock()
    scheduler._request_states = {}
    scheduler._pending_jobs = {}
    request = SimpleNamespace(request_id="r")
    scheduler.on_new_request(request)
    if prefetch_prepared:
        scheduler._track_prepared_job(
            1, JobOperation.PREFETCH, "r", AddressedToken((), b"")
        )
        scheduler._request_states["r"].prefetch_job_id = 1

    scheduler.request_finished("r")
    if prefetch_prepared:
        store.complete_prefetch.assert_called_once_with("r", False)
        assert store.method_calls[-2:] == [
            call.complete_prefetch("r", False),
            call.on_request_finished("r"),
        ]
    else:
        store.complete_prefetch.assert_not_called()
    assert not scheduler._pending_jobs
    store.on_request_finished.assert_called_once_with("r")
    assert not scheduler._request_states


@pytest.mark.parametrize("success", [True, False])
def test_finished_request_waits_for_every_rank_before_prefetch_complete(success):
    from ucm.integration.vllm.v2.connector_scheduler import (
        JobOperation,
        PendingJob,
        RequestState,
        UCMConnectorScheduler,
    )
    from ucm.integration.vllm.v2.connector_worker import JobResult, UCMWorkerMetadata
    from ucm.integration.vllm.v2.runtime import ParallelRank

    rank0 = ParallelRank(tp_rank=0, pp_rank=0, pcp_rank=0)
    rank1 = ParallelRank(tp_rank=1, pp_rank=0, pcp_rank=0)
    scheduler = object.__new__(UCMConnectorScheduler)
    scheduler._store_scheduler = Mock()
    scheduler._store_scheduler.prepare_offload.return_value = ()
    scheduler._expected_worker_ranks = {rank0, rank1}
    scheduler._pending_jobs = {
        1: PendingJob(
            "r", JobOperation.PREFETCH, SimpleNamespace(job_id=1), dispatched=True
        )
    }
    scheduler._needs_offload_drain = False
    request = SimpleNamespace(request_id="r", num_preemptions=0)
    scheduler._request_states = {"r": RequestState(request, prefetch_job_id=1)}

    scheduler.request_finished("r")
    scheduler._store_scheduler.on_request_finished.assert_not_called()
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UCMWorkerMetadata(
                job_results={JobResult(1, rank1, True)}
            )
        )
    )
    scheduler._store_scheduler.complete_prefetch.assert_not_called()
    scheduler._store_scheduler.on_request_finished.assert_not_called()

    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=UCMWorkerMetadata(
                job_results={JobResult(1, rank0, success)}
            )
        )
    )
    scheduler._store_scheduler.complete_prefetch.assert_called_once_with("r", success)
    scheduler._store_scheduler.on_request_finished.assert_called_once_with("r")
    assert not scheduler._request_states
    assert scheduler.has_pending_push_work()
    scheduler.build_connector_meta(
        SimpleNamespace(
            kv_connector_block_state=SimpleNamespace(req_ids=()),
            preempted_req_ids=set(),
            finished_req_ids=set(),
        )
    )
    assert not scheduler.has_pending_push_work()
