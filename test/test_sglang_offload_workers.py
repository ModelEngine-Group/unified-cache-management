"""Deterministic thread tests for dump submission and ordered reclamation."""

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


class ReadyEvent:
    def __init__(self, gate=None, entered=None, handle=0):
        self.gate = gate
        self.entered = entered
        self.cuda_event = handle
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None and not self.gate.wait(3):
            raise RuntimeError("test event timeout")


class OffloadWorkersTest(unittest.TestCase):
    def setUp(self):
        self.gates = []
        self.complete_all = False
        self.complete = {key: threading.Event() for key in ("A", "B", "C")}
        self.submitted = {key: threading.Event() for key in self.complete}
        self.waited = {key: threading.Event() for key in self.complete}
        self.events = {}
        self.submit_fail = set()
        self.check_fail = set()
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._tp_rank = 0
        obj._closed = False
        obj._offload_queue = queue.Queue()
        obj._offload_submitted_queue = queue.Queue()
        obj._offload_results = queue.Queue()
        obj._load_queue = queue.Queue()
        obj._completed_loads = queue.Queue()
        obj._load_results = {}
        obj._lookup_progress = {}
        obj._lookup_progress_lock = threading.Lock()
        obj._pending_loads = {}
        obj._pending_lock = threading.Lock()
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
            if key in self.check_fail:
                raise RuntimeError("test check failure")
            return self.complete_all or self.complete[key].is_set()

        def wait(tasks):
            for key in tasks:
                self.waited[key].set()

        obj.adapter = types.SimpleNamespace(
            supports_dump_prerequisite=True,
            submit_offload=submit,
            check=check,
            wait=wait,
            close=lambda: None,
        )
        obj._load_thread = threading.Thread(target=obj._load_worker, daemon=True)
        obj._offload_thread = threading.Thread(target=obj._offload_worker, daemon=True)
        obj._offload_submit_thread = threading.Thread(
            target=obj._offload_submit_worker, daemon=True
        )
        self.obj = obj
        self.addCleanup(self.stop)
        obj._load_thread.start()
        obj._offload_thread.start()
        obj._offload_submit_thread.start()

    def stop(self):
        self.complete_all = True
        for gate in self.gates:
            gate.set()
        closer = threading.Thread(target=self.obj.close, daemon=True)
        closer.start()
        closer.join(3)
        self.assertFalse(closer.is_alive(), "close() did not drain the queues")
        for worker in (self.obj._offload_thread, self.obj._offload_submit_thread):
            self.assertFalse(worker.is_alive())
        self.assertEqual(self.obj._offload_queue.unfinished_tasks, 0)
        self.assertEqual(self.obj._offload_submitted_queue.unfinished_tasks, 0)

    def enqueue(self, key, event=None):
        kv_event = ReadyEvent(handle=77)
        self.obj._offload_queue.put(([key], [event or ReadyEvent(), kv_event]))
        return kv_event

    def test_reap_while_next_node_waits_for_gpu_event(self):
        self.enqueue("A")
        self.assertTrue(self.submitted["A"].wait(2))
        gate, entered = threading.Event(), threading.Event()
        self.gates.append(gate)
        kv_event = self.enqueue("B", ReadyEvent(gate, entered))
        self.assertTrue(entered.wait(2))
        self.complete["A"].set()
        self.assertTrue(self.obj._offload_results.get(timeout=2))
        self.assertFalse(self.submitted["B"].is_set())
        gate.set()
        self.assertTrue(self.submitted["B"].wait(2))
        self.assertEqual(self.events["B"], 77)
        self.assertFalse(kv_event.synchronized)

    def test_reap_while_next_node_builds_addresses(self):
        gate, entered = threading.Event(), threading.Event()
        self.gates.append(gate)

        def prepare(transfers, **kwargs):
            if transfers == ["B"]:
                entered.set()
                if not gate.wait(3):
                    raise RuntimeError("test preparation timeout")
            return transfers

        self.obj._page_transfers = prepare
        self.enqueue("A")
        self.assertTrue(self.submitted["A"].wait(2))
        self.enqueue("B")
        self.assertTrue(entered.wait(2))
        self.complete["A"].set()
        self.assertTrue(self.obj._offload_results.get(timeout=2))
        self.assertFalse(self.submitted["B"].is_set())

    def test_out_of_order_completion_publishes_in_order(self):
        self.check_fail.add("A")
        # Hold A's check failure until B has completed, so order is observable.
        original_check = self.obj.adapter.check
        self.obj.adapter.check = lambda key: (
            False
            if key == "A" and not self.complete["A"].is_set()
            else original_check(key)
        )
        self.enqueue("A")
        self.enqueue("B")
        self.complete["B"].set()
        self.assertTrue(self.waited["B"].wait(2))
        self.assertTrue(self.obj._offload_results.empty())
        with self.assertLogs(linker_module.logger, level="ERROR"):
            self.complete["A"].set()
            self.assertFalse(self.obj._offload_results.get(timeout=2))
        self.assertTrue(self.obj._offload_results.get(timeout=2))
        self.assertTrue(self.waited["A"].is_set())

    def test_submission_failure_still_publishes_and_drains(self):
        self.submit_fail.add("A")
        with self.assertLogs(linker_module.logger, level="ERROR"):
            self.enqueue("A")
            self.assertFalse(self.obj._offload_results.get(timeout=2))
        self.enqueue("B")
        self.complete["B"].set()
        self.assertTrue(self.obj._offload_results.get(timeout=2))


if __name__ == "__main__":
    unittest.main()
