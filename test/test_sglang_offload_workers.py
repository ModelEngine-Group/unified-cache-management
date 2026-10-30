"""Single-worker dump polling, ordered publication and shutdown tests."""

import collections
import queue
import threading
import time
import types
import unittest

from test_sglang_direct_preparation import linker_module

linker_module.deque = collections.deque
linker_module.Empty = queue.Empty
linker_module.time = time


class OffloadWorkerTest(unittest.TestCase):
    def setUp(self):
        self.complete_all = False
        self.complete = {key: threading.Event() for key in ("A", "B", "C")}
        self.submitted = {key: threading.Event() for key in self.complete}
        self.waited = {key: threading.Event() for key in self.complete}
        self.submit_fail, self.check_fail, self.wait_fail = set(), set(), set()
        self.reaped, self.checked_after_wait = set(), []
        self.events = {}
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._tp_rank, obj._closed = 0, False
        obj._offload_queue, obj._offload_results = queue.Queue(), queue.Queue()
        obj._load_queue, obj._completed_loads = queue.Queue(), queue.Queue()
        obj._load_results, obj._lookup_progress, obj._pending_loads = {}, {}, {}
        obj._lookup_progress_lock, obj._pending_lock = (
            threading.Lock(),
            threading.Lock(),
        )
        obj.layer_done_counter = types.SimpleNamespace(reset=lambda: None)
        obj._page_transfers = lambda transfers, **kwargs: transfers

        def submit(transfers, prerequisite_handle=0):
            key = transfers[0]
            if key in self.submit_fail:
                raise RuntimeError("test submit failure")
            self.events[key] = prerequisite_handle
            self.submitted[key].set()
            return [key]

        def check(key):
            if key in self.reaped:
                self.checked_after_wait.append(key)
                raise RuntimeError("task already reaped")
            if key in self.check_fail:
                raise RuntimeError("test check failure")
            return self.complete_all or self.complete[key].is_set()

        def wait(tasks):
            for key in tasks:
                self.reaped.add(key)
                self.waited[key].set()
                if key in self.wait_fail:
                    raise RuntimeError("test wait failure")

        obj.adapter = types.SimpleNamespace(
            supports_dump_prerequisite=True,
            submit_offload=submit,
            check=check,
            wait=wait,
            close=lambda: None,
        )
        obj._load_thread = threading.Thread(target=obj._load_worker, daemon=True)
        obj._offload_thread = threading.Thread(target=obj._offload_worker, daemon=True)
        self.obj = obj
        self.addCleanup(self.stop)
        obj._load_thread.start()
        obj._offload_thread.start()

    def stop(self):
        self.complete_all = True
        closer = threading.Thread(target=self.obj.close, daemon=True)
        closer.start()
        closer.join(3)
        self.assertFalse(closer.is_alive(), "close() did not drain dump tasks")
        self.assertFalse(self.obj._offload_thread.is_alive())
        self.assertFalse(self.obj._load_thread.is_alive())
        self.assertEqual(self.obj._offload_queue.unfinished_tasks, 0)
        self.assertEqual(self.checked_after_wait, [])

    def enqueue(self, key):
        index_event = types.SimpleNamespace(synchronize=lambda: None)
        kv_event = types.SimpleNamespace(
            cuda_event=77,
            synchronize=lambda: self.fail("KV event should be forwarded to Store"),
        )
        self.obj._offload_queue.put(([key], [index_event, kv_event]))

    def test_pending_tasks_do_not_block_new_submissions(self):
        self.enqueue("A")
        self.enqueue("B")
        self.assertTrue(self.submitted["B"].wait(2))
        self.assertFalse(self.waited["A"].is_set())
        self.assertFalse(self.waited["B"].is_set())
        self.assertEqual(self.events, {"A": 77, "B": 77})

    def test_reap_out_of_order_but_publish_in_submission_order(self):
        self.enqueue("A")
        self.enqueue("B")
        self.complete["B"].set()
        self.assertTrue(self.waited["B"].wait(2))
        self.assertTrue(self.obj._offload_results.empty())
        self.assertEqual(self.obj._offload_queue.unfinished_tasks, 2)
        # Different statuses make the publication order observable.
        with self.assertLogs(linker_module.logger, level="ERROR"):
            self.check_fail.add("A")
            self.assertFalse(self.obj._offload_results.get(timeout=2))
        self.assertTrue(self.obj._offload_results.get(timeout=2))
        self.assertTrue(self.waited["A"].is_set())

    def test_submit_failure_does_not_block_following_node(self):
        self.submit_fail.add("A")
        with self.assertLogs(linker_module.logger, level="ERROR"):
            self.enqueue("A")
            self.assertFalse(self.obj._offload_results.get(timeout=2))
        self.enqueue("B")
        self.complete["B"].set()
        self.assertTrue(self.obj._offload_results.get(timeout=2))

    def test_wait_failure_is_published_and_next_node_completes(self):
        self.wait_fail.add("A")
        self.complete["A"].set()
        with self.assertLogs(linker_module.logger, level="ERROR"):
            self.enqueue("A")
            self.assertFalse(self.obj._offload_results.get(timeout=2))
        self.enqueue("B")
        self.complete["B"].set()
        self.assertTrue(self.obj._offload_results.get(timeout=2))

    def test_shutdown_drains_pending_nodes(self):
        self.enqueue("A")
        self.enqueue("B")
        self.assertTrue(self.submitted["B"].wait(2))
        # Cleanup completes the tasks, calls reset/close and checks both joins.


if __name__ == "__main__":
    unittest.main()
