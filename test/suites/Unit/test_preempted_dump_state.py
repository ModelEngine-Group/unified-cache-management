"""CPU regressions for retiring preempted UCM dump notifications."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


SOURCE = Path(__file__).resolve().parents[3] / "ucm/integration/vllm/ucm_connector.py"


def connector():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "UCMDirectConnector")
    methods = {"build_connector_meta", "_track_async_dump_requests",
               "_flush_pending_dump_tasks", "handle_preemptions",
               "get_finished", "request_finished"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in methods]
    cls.bases = []
    cls.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                        names=[ast.alias(name="annotations")], level=0), cls],
                        type_ignores=[])
    ns = {"logger": Mock(), "UCMConnectorMetadata":
          lambda plans, preempted: NS(request_meta=plans, preempted_req_ids=preempted)}
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


def schedule(obj, new_dump=False):
    if new_dump:
        obj._pending_async_load_dispatches = {
            "r": NS(dump_block_ids=([b"key"], [1]))}
    return obj.build_connector_meta(NS(
        preempted_req_ids={"r"}, scheduled_new_reqs=[],
        scheduled_cached_reqs=[], finished_req_ids=set()))


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
