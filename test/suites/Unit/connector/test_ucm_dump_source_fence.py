"""CPU-only lifecycle tests; do not require vLLM, torch, or NPU initialization."""

import ast
import copy
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[4]


def load_symbols(path, classes):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    nodes = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            selected = copy.copy(node)
            selected.bases = []
            selected.keywords = []
            if classes[node.name] is not None:
                selected.body = [
                    method
                    for method in node.body
                    if getattr(method, "name", None) in classes[node.name]
                ]
            nodes.append(selected)
    namespace = {"__name__": __name__, "dataclass": dataclass, "logger": Mock()}
    exec(
        compile(ast.fix_missing_locations(ast.Module(nodes, [])), str(path), "exec"),
        namespace,
    )
    return namespace


HMA = load_symbols(
    "ucm/integration/vllm/hma_connector.py",
    {
        "FAWADumpTask": None,
        "UCMFAWAConnector": {
            "_poll_completed_dump_tasks",
            "_finish_dump_task",
            "_fence_pending_dump_tasks",
            "_drain_best_effort_dump_tasks",
            "handle_preemptions",
            "get_finished",
            "shutdown",
        },
    },
)
NPU = load_symbols(
    "ucm/integration/vllm/device.py",
    {
        "NpuDevice": {
            "wait_for_dump_source",
            "query_event_handle",
            "synchronize_event_handle",
        }
    },
)
WRAPPER = load_symbols(
    "ucm/integration/vllm/ucm_connector.py", {"UCMConnector": {"shutdown"}}
)


class DumpSourceFenceTests(unittest.TestCase):
    def setUp(self):
        self.connector = HMA["UCMFAWAConnector"]()
        self.connector.enable_async_dump_source_fence = True
        self.connector.device = Mock()
        self.connector.device.wait_for_dump_source.return_value = 77
        self.connector.device.query_event_handle.return_value = False
        self.connector._rank_consistency = Mock()
        self.connector._record_counter = Mock()
        self.fence = SimpleNamespace(event_handles=[101, 202])
        self.store = Mock()
        self.store.check.return_value = False
        self.store.get_dump_source_fence.return_value = self.fence
        self.task = HMA["FAWADumpTask"]("WA", self.store, object(), 2, 11)
        self.connector.tp_dump_tasks = {("request",): [self.task]}

    def test_cpu_returns_before_copy_completion_and_does_not_reinsert_waits(self):
        self.connector.handle_preemptions(None)
        self.connector._rank_consistency.wait_dump.assert_not_called()
        self.connector.device.wait_for_dump_source.assert_called_once_with([101, 202])
        self.assertIs(self.task.source_fence, self.fence)
        self.assertEqual(self.task.source_wait_handle, 77)
        self.connector.handle_preemptions(None)
        self.store.get_dump_source_fence.assert_called_once_with(self.task.task)
        self.connector.device.wait_for_dump_source.assert_called_once()

    def test_completed_copy_retains_fence_until_consumer_acknowledges_waits(self):
        self.connector.handle_preemptions(None)
        self.store.check.return_value = True
        self.connector._poll_completed_dump_tasks()
        self.assertIs(self.task.source_fence, self.fence)
        self.connector._rank_consistency.wait_dump.assert_not_called()
        self.connector.device.destroy_event_handle.assert_not_called()
        self.connector.device.query_event_handle.return_value = True
        self.connector._poll_completed_dump_tasks()
        self.assertEqual(self.connector.tp_dump_tasks, {})
        self.connector._rank_consistency.wait_dump.assert_called_once_with(
            self.task.task
        )
        self.assertEqual(
            self.connector.device.destroy_event_handle.call_args_list,
            [call(77), call(11)],
        )
        self.assertIsNone(self.task.source_fence)

    def test_finished_requests_drain_before_reporting_success(self):
        self.connector.handle_preemptions(None)
        self.assertEqual(self.connector.get_finished({"request"}), ({"request"}, None))
        self.connector.device.synchronize_event_handle.assert_called_once_with(77)
        self.connector._rank_consistency.finish_dump.assert_called_once_with(
            {"request"}
        )
        self.assertEqual(self.connector.tp_dump_tasks, {})

    def test_fallback_for_unsupported_store(self):
        self.store.get_dump_source_fence.return_value = None
        self.connector.handle_preemptions(None)
        self.connector._rank_consistency.wait_dump.assert_called_once_with(
            self.task.task
        )
        self.connector.device.wait_for_dump_source.assert_not_called()
        self.assertEqual(self.connector.tp_dump_tasks, {})

    def test_fallback_for_unsupported_device(self):
        self.connector.device.wait_for_dump_source.return_value = None
        self.connector.handle_preemptions(None)
        self.connector._rank_consistency.wait_dump.assert_called_once_with(
            self.task.task
        )
        self.assertEqual(self.connector.tp_dump_tasks, {})

    def test_fallback_for_fence_publication_error(self):
        self.store.get_dump_source_fence.side_effect = RuntimeError(
            "publication failed"
        )
        self.connector.handle_preemptions(None)
        self.connector._rank_consistency.wait_dump.assert_called_once_with(
            self.task.task
        )

    def test_device_wait_failure_stops_before_releasing_source(self):
        self.connector.device.wait_for_dump_source.side_effect = RuntimeError(
            "wait failed"
        )
        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            self.connector.handle_preemptions(None)
        self.connector._rank_consistency.wait_dump.assert_not_called()
        self.connector.device.destroy_event_handle.assert_not_called()
        self.assertIn(("request",), self.connector.tp_dump_tasks)

    def test_default_disabled_path_keeps_original_host_wait(self):
        self.connector.enable_async_dump_source_fence = False
        self.connector.handle_preemptions(None)
        self.connector._rank_consistency.wait_dump.assert_called_once_with(
            self.task.task
        )
        self.store.get_dump_source_fence.assert_not_called()

    def test_dump_failure_still_records_failure_and_releases_events(self):
        self.connector.handle_preemptions(None)
        self.connector._rank_consistency.wait_dump.side_effect = RuntimeError(
            "copy failed"
        )
        self.connector.get_finished({"request"})
        self.connector._record_counter.assert_called_once_with(
            "connector_dump_wait_errors_total"
        )
        self.assertIsNone(self.task.source_fence)
        self.assertEqual(self.connector.tp_dump_tasks, {})

    def test_empty_fence_needs_no_consumer_ack(self):
        self.fence.event_handles = []
        self.connector.device.wait_for_dump_source.return_value = 0
        self.connector.handle_preemptions(None)
        self.store.check.return_value = True
        self.connector._poll_completed_dump_tasks()
        self.connector.device.query_event_handle.assert_not_called()
        self.connector.device.synchronize_event_handle.assert_not_called()
        self.assertEqual(self.connector.tp_dump_tasks, {})

    def test_consumer_ack_error_does_not_free_native_fence(self):
        self.connector.handle_preemptions(None)
        self.store.check.return_value = True
        self.connector.device.query_event_handle.side_effect = RuntimeError(
            "query failed"
        )
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self.connector._poll_completed_dump_tasks()
        self.assertIs(self.task.source_fence, self.fence)
        self.connector.device.destroy_event_handle.assert_not_called()

    def test_shutdown_drains_pending_fences(self):
        self.connector.handle_preemptions(None)
        self.connector.shutdown()
        self.connector._rank_consistency.wait_dump.assert_called_once_with(
            self.task.task
        )
        self.connector.device.synchronize_event_handle.assert_called_once_with(77)
        self.assertEqual(self.connector.tp_dump_tasks, {})

    def test_outer_connector_forwards_upstream_shutdown_hook(self):
        wrapper = WRAPPER["UCMConnector"]()
        wrapper.connector = self.connector
        self.connector.handle_preemptions(None)
        wrapper.shutdown()
        self.assertEqual(self.connector.tp_dump_tasks, {})
        wrapper.connector = object()
        wrapper.shutdown()


class NpuDumpSourceWaitTests(unittest.TestCase):
    def setUp(self):
        self.device = NPU["NpuDevice"]()
        self.device.events = {}
        self.device.destroy_event_handle = Mock()
        self.acl = Mock()
        self.acl.rt.create_event.return_value = (77, 0)
        self.acl.rt.stream_wait_event.return_value = 0
        self.acl.rt.record_event.return_value = 0
        self.acl.rt.query_event_status.return_value = (1, 0)
        self.acl.rt.synchronize_event.return_value = 0
        self.stream = SimpleNamespace(npu_stream=88, synchronize=Mock())
        self.torch_npu = Mock()
        self.torch_npu.npu.current_stream.return_value = self.stream
        self.modules = patch.dict(
            "sys.modules", {"acl": self.acl, "torch_npu": self.torch_npu}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_waits_all_copy_streams_before_recording_ack(self):
        self.assertEqual(self.device.wait_for_dump_source([101, 202]), 77)
        self.assertEqual(
            self.acl.rt.method_calls,
            [
                call.create_event(),
                call.stream_wait_event(88, 101),
                call.stream_wait_event(88, 202),
                call.record_event(77, 88),
            ],
        )
        self.stream.synchronize.assert_not_called()
        self.assertIn(77, self.device.events)

    def test_empty_fence_does_not_create_an_event(self):
        self.assertEqual(self.device.wait_for_dump_source([]), 0)
        self.acl.rt.create_event.assert_not_called()

    def test_partial_wait_failure_drains_before_releasing_ack(self):
        self.acl.rt.stream_wait_event.side_effect = [0, 1]
        with self.assertRaisesRegex(RuntimeError, "Wait for dump source event"):
            self.device.wait_for_dump_source([101, 202])
        self.stream.synchronize.assert_called_once()
        self.device.destroy_event_handle.assert_called_once_with(77)

    def test_query_distinguishes_not_ready_and_errors(self):
        self.assertTrue(self.device.query_event_handle(77))
        self.acl.rt.query_event_status.return_value = (0, 0)
        self.assertFalse(self.device.query_event_handle(77))
        self.acl.rt.query_event_status.return_value = (0, 1)
        with self.assertRaises(RuntimeError):
            self.device.query_event_handle(77)


if __name__ == "__main__":
    unittest.main()
