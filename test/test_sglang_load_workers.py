"""Layer submission, completion ordering and failure draining."""

import queue
import types
import unittest

from common.sglang_utils import linker_module


class LoadWorkerTest(unittest.TestCase):
    def run_worker(self, *, fail_submit=None, fail_wait=None, fail_check=None):
        events = []
        submitted = []
        reaped = set()
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj.num_layers = 4
        obj._tp_rank = 0
        obj._requires_load_finish = True
        obj._load_queue = queue.Queue()
        obj._load_results = {
            0: types.SimpleNamespace(
                set_result=lambda ok: events.append(("result", ok))
            )
        }
        obj._page_transfers = lambda transfers, **kwargs: transfers
        obj.layer_done_counter = types.SimpleNamespace(
            complete=lambda batch, layer: events.append(("complete", layer)),
            fail=lambda batch, error: events.append(("fail", str(error))),
        )

        def submit(prepared, layer):
            events.append(("submit", layer))
            if layer == fail_submit:
                raise RuntimeError("submit failed")
            submitted.append(layer)
            return [layer]

        def check(layer):
            self.assertNotIn(layer, reaped)
            if layer == fail_check:
                raise RuntimeError("check failed")
            # First layer finishes while later layers are still being submitted.
            return layer == 0 and len(submitted) >= 2

        def wait(tasks):
            for layer in tasks:
                self.assertNotIn(layer, reaped)
                reaped.add(layer)
                events.append(("wait", layer))
                if layer == fail_wait:
                    raise RuntimeError("wait failed")

        obj.adapter = types.SimpleNamespace(
            prepare_load=lambda transfers: transfers,
            submit_prepared_load_layer=submit,
            check=check,
            wait=wait,
        )
        obj._load_queue.put(
            (0, {"req": [object()]}, types.SimpleNamespace(synchronize=lambda: None))
        )
        obj._load_queue.put(None)
        obj._load_worker()
        self.assertEqual(obj._load_queue.unfinished_tasks, 0)
        self.assertEqual(reaped, set(submitted))
        return events

    def test_submission_continues_and_ready_layers_publish_early(self):
        events = self.run_worker()
        self.assertLess(events.index(("submit", 1)), events.index(("wait", 0)))
        self.assertLess(events.index(("complete", 0)), events.index(("submit", 2)))
        self.assertLess(events.index(("submit", 3)), events.index(("wait", 1)))
        self.assertEqual([e[1] for e in events if e[0] == "complete"], list(range(4)))
        self.assertEqual(events[-1], ("result", True))

    def test_failures_drain_submitted_layers(self):
        for kwargs in ({"fail_submit": 2}, {"fail_wait": 1}, {"fail_check": 0}):
            with self.subTest(**kwargs):
                events = self.run_worker(**kwargs)
                self.assertEqual(events[-1], ("result", False))
                failure = next(i for i, e in enumerate(events) if e[0] == "fail")
                self.assertTrue(
                    all(i < failure for i, e in enumerate(events) if e[0] == "wait")
                )
