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
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._dump_submitted_layer_ids = set()
    worker._failed_dump_jobs = set()
    worker._completion_metadata = UCMWorkerMetadata()
    return worker


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
    scheduler._offload_jobs = {}
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
