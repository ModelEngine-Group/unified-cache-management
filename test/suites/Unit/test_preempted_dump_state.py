"""CPU regressions for retiring preempted UCM dump notifications.

Production connector methods and complete consistency classes execute without
GPU imports. The Store is simulated; queued-step tests control execution order
and do not launch the vLLM engine or reproduce hardware scheduling.
"""

import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "ucm/integration/vllm/ucm_connector.py"


def connector():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "UCMDirectConnector"
    )
    methods = {
        "build_connector_meta",
        "_track_async_dump_requests",
        "_flush_pending_dump_tasks",
        "handle_preemptions",
        "get_finished",
        "request_finished",
        "_wait_pending_dump_task",
        "_release_dump_event_handle",
        "_poll_pending_dump_tasks",
        "_non_negative_ms",
        "wait_for_save",
        "_generate_dispatch_meta",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    cls.bases = []
    cls.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ns = {
        "logger": Mock(),
        "time": time,
        "math": math,
        "ucmmetrics": Mock(),
        "PendingDumpTask": NS,
        "RequestDispatchMeta": lambda load, dump, **kw: NS(
            load_block_ids=load, dump_block_ids=dump, **kw
        ),
        "UCMConnectorMetadata": Metadata,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    obj = ns["UCMDirectConnector"]()
    obj._async_dump_req_ids = {"r", "other"}
    obj._pending_dump_tasks = []
    obj._rank_consistency = Mock()
    obj._poll_pending_load_tasks = Mock(return_value={"r"})
    obj._wait_pending_dump_task = Mock()
    obj.requests_meta = {}
    obj._pending_async_load_dispatches = {}
    obj._async_load_req_ids = set()
    return obj


class Metadata:
    def __init__(self, plans, preempted):
        self.request_meta = plans
        self.preempted_req_ids = preempted


def schedule(obj, new_dump=False):
    if new_dump:
        obj._pending_async_load_dispatches = {"r": NS(dump_block_ids=([b"key"], [1]))}
    return obj.build_connector_meta(
        NS(
            preempted_req_ids={"r"},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=[],
            finished_req_ids=set(),
        )
    )


@pytest.mark.parametrize("failed_save", [False, True])
@pytest.mark.parametrize("already_polled", [False, True])
def test_preempted_save_then_failed_load_has_only_receive_completion(
    failed_save, already_polled
):
    scheduler, worker = connector(), connector()
    old = NS(request_ids={"r", "other"})
    unrelated = NS(request_ids={"unrelated"})
    worker._pending_dump_tasks = ([old] if not already_polled else []) + [unrelated]

    def wait(task):
        assert "r" in worker._async_dump_req_ids  # Never retire before waiting.
        if failed_save:
            raise OSError("completed dump failed")

    worker._wait_pending_dump_task.side_effect = wait
    metadata = schedule(scheduler)
    worker.handle_preemptions(metadata)
    assert worker._pending_dump_tasks == [unrelated]
    assert worker._async_dump_req_ids == {"other"}
    worker._rank_consistency.finish_dump.assert_called_once_with({"r"})
    assert scheduler.request_finished(NS(request_id="r"), [1]) == (False, None)
    assert worker.get_finished({"r"}) == (None, {"r"})
    # A shared completed task must not erase another request's notification.
    assert scheduler.request_finished(NS(request_id="other"), [2]) == (True, None)


def test_new_save_after_preemption_keeps_its_notification():
    scheduler, worker = connector(), connector()
    metadata = schedule(scheduler, new_dump=True)
    worker.handle_preemptions(metadata)
    # wait_for_save registers this step's plans after handle_preemptions.
    worker._track_async_dump_requests(metadata.request_meta)
    worker._poll_pending_load_tasks.return_value = set()
    assert scheduler.request_finished(NS(request_id="r"), [1]) == (True, None)
    assert worker.get_finished({"r"}) == ({"r"}, None)
    assert worker.get_finished({"r"}) == (None, None)


def test_no_preemption_preserves_normal_save_notification():
    worker = connector()
    worker.handle_preemptions(set())
    worker._poll_pending_load_tasks.return_value = set()
    assert worker.get_finished({"r"}) == ({"r"}, None)


def consistency_types():
    """Execute complete production classes, omitting hardware import plumbing."""
    root = SOURCE.parents[2]
    ns = {"logger": Mock()}
    for path in (
        root / "store/pipeline/errors.py",
        SOURCE.with_name("rank_consistency.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        module = ast.parse("from __future__ import annotations")
        module.body.extend(n for n in tree.body if isinstance(n, ast.ClassDef))
        exec(compile(module, str(path), "exec"), ns)
    return ns


class Store:
    """Deterministic storage boundary; wait errors represent completed failures."""

    def __init__(self):
        self.error = None
        self.tasks = []

    def dump_data(self, keys, *args):
        task = NS(keys=set(keys), error=self.error, done=False, waits=0)
        self.tasks.append(task)
        return task

    def check(self, task):
        return task.done

    def wait(self, task):
        task.waits += 1
        task.done = True
        if task.error:
            raise task.error


def real_worker(enabled=True):
    types = consistency_types()
    w = connector()
    # Use the real wait/poll/accounting path instead of the original helper stub.
    del w._wait_pending_dump_task
    w._rank_consistency = types["RankConsistencyManager"](
        is_scheduler=False, use_consistency_manager=enabled
    )
    w.store = Store()
    w._async_dump_req_ids.clear()
    w.enable_event_sync = False
    w.device = None
    w.tp_rank = 0
    w.is_mla = False
    w.block_data_size = 1
    ptr = Mock()
    ptr.shape = (1, 1)
    ptr.reshape.return_value = ptr
    w.kv_cache_layout = NS(extract_block_addrs=lambda ids: ptr)
    w._get_rank_dump_blocks = lambda plan: plan.dump_block_ids
    w._get_event_handle = lambda: 0
    w._get_connector_metadata = lambda: w.metadata
    return w, types


def report(manager):
    meta = NS(missing_reqs=set(), missing_blocks=set(), dump_succeeded_blocks=set())
    manager.update_worker_meta(meta)
    return meta


def submit_plans(w, plans):
    w.metadata = Metadata(
        {
            rid: NS(dump_block_ids=(keys, list(range(len(keys)))))
            for rid, keys in plans.items()
        },
        set(),
    )
    w.wait_for_save()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "outcome", ["success", "io_error", "unhealthy", "submit_error"]
)
@pytest.mark.parametrize("polled", [False, True])
def test_real_consistency_preemption_preserves_other_request(enabled, outcome, polled):
    w, types = real_worker(enabled)
    if outcome == "io_error":
        w.store.error = OSError("completed save failed")
    elif outcome == "unhealthy":
        w.store.error = types["StoreUnhealthyError"]("store unhealthy")
    elif outcome == "submit_error":
        w.store.dump_data = Mock(side_effect=OSError("submission failed"))
    submit_plans(w, {"r": [b"r"], "other": [b"other"]})
    if polled:
        for task in w.store.tasks:
            task.done = True
        w._poll_pending_dump_tasks()
    w.handle_preemptions({"r"})
    assert w._async_dump_req_ids == {"other"}
    assert not w._pending_dump_tasks
    assert not w._rank_consistency._dump_task_contexts
    first = report(w._rank_consistency)
    success = enabled and outcome == "success"
    assert first.dump_succeeded_blocks == ({b"r"} if success else set())
    assert "r" not in w._rank_consistency._dump_blocks_by_request
    w._poll_pending_load_tasks.return_value = set()
    assert w.get_finished({"other"}) == ({"other"}, None)
    assert report(w._rank_consistency).dump_succeeded_blocks == (
        {b"other"} if success else set()
    )
    assert not w._rank_consistency._dump_blocks_by_request
    assert not w._rank_consistency._dump_failed_blocks_by_request
    assert all(task.waits == 1 for task in w.store.tasks)


def planned_step(s, saves, preempted=()):
    s.block_size = s.cp_world_size = 1
    requests = []
    for rid, key in saves.items():
        s.requests_meta[rid] = NS(
            hbm_hit_block_num=0,
            total_hit_block_num=0,
            ucm_block_ids=[key],
            vllm_block_ids=[],
            token_processed=0,
            num_token_ids=1,
        )
        requests.append(NS(req_id=rid, block_ids=([1],)))
    return s.build_connector_meta(
        NS(
            scheduled_new_reqs=requests,
            scheduled_cached_reqs=[],
            finished_req_ids=set(),
            preempted_req_ids=set(preempted),
            num_scheduled_tokens={rid: 1 for rid in saves},
        )
    )


@pytest.mark.parametrize("same_step_resume", [False, True])
def test_queued_old_steps_then_preemption_and_resumed_save(same_step_resume):
    s = connector()
    s._async_dump_req_ids.clear()
    w, _ = real_worker()
    # Scheduler gets ahead of Worker: construct every plan before executing any.
    steps = [
        planned_step(s, {"r": b"old1", "other": b"other"}),
        planned_step(s, {"r": b"old2"}),
    ]
    steps.append(planned_step(s, {"r": b"new"} if same_step_resume else {}, {"r"}))
    if not same_step_resume:
        steps.append(planned_step(s, {"r": b"new"}))
    w._poll_pending_load_tasks.return_value = set()
    for index, metadata in enumerate(steps):
        w.handle_preemptions(metadata)
        if index == 2:
            assert "r" not in w._async_dump_req_ids
            assert all(t.done for t in w.store.tasks)
            assert report(w._rank_consistency).dump_succeeded_blocks == {
                b"old1",
                b"old2",
            }
        w.metadata = metadata
        w.wait_for_save()
        assert w.get_finished(set()) == (None, None)
    assert s.request_finished(NS(request_id="r"), [1]) == (True, None)
    assert w.get_finished({"r"}) == ({"r"}, None)
    assert report(w._rank_consistency).dump_succeeded_blocks == {b"new"}
    assert s.request_finished(NS(request_id="other"), [1]) == (True, None)
    assert w.get_finished({"other"}) == ({"other"}, None)
    assert report(w._rank_consistency).dump_succeeded_blocks == {b"other"}
    assert all(t.waits == 1 for t in w.store.tasks)
    assert not w._pending_dump_tasks
    assert not w._async_dump_req_ids


def test_old_dump_success_does_not_clear_new_load_miss():
    w, types = real_worker()
    scheduler = types["RankConsistencyManager"](is_scheduler=True)
    submit_plans(w, {"r": [b"key"]})
    w.handle_preemptions({"r"})
    store = NS(
        load_data=lambda *a, **kw: object(),
        wait=Mock(side_effect=types["StoreNotFoundError"]("missing")),
    )
    task = w._rank_consistency.submit_load(store, {"r": [b"key"]}, [b"key"], [0], None)
    with pytest.raises(types["StoreNotFoundError"]):
        w._rank_consistency.wait_load(task)
    meta = report(w._rank_consistency)
    assert meta.dump_succeeded_blocks == meta.missing_blocks == {b"key"}
    scheduler.apply_worker_meta(meta)
    assert b"key" in scheduler._tracker
    assert report(w._rank_consistency).dump_succeeded_blocks == set()


def test_preemption_does_not_drain_unrelated_pending_save():
    w, _ = real_worker()
    submit_plans(w, {"r": [b"r"]})
    submit_plans(w, {"other": [b"other"]})
    first, unrelated = w.store.tasks
    w.handle_preemptions({"r"})
    assert first.waits == 1
    assert unrelated.waits == 0
    assert not unrelated.done
    assert [pending.task for pending in w._pending_dump_tasks] == [unrelated]
    assert w._async_dump_req_ids == {"other"}
    assert report(w._rank_consistency).dump_succeeded_blocks == {b"r"}
    w._poll_pending_load_tasks.return_value = set()
    assert w.get_finished({"other"}) == ({"other"}, None)
    assert unrelated.waits == 1
    assert report(w._rank_consistency).dump_succeeded_blocks == {b"other"}
