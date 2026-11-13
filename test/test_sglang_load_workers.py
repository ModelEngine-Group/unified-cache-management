"""Layer submission, completion ordering and failure draining."""

from concurrent.futures import Future
import queue
import types
import unittest
from unittest.mock import patch

from common.sglang_utils import linker_module


class LoadWorkerTest(unittest.TestCase):
    def run_worker(
        self,
        *,
        fail_submit=None,
        fail_wait=None,
        fail_check=None,
        forward_event=False,
        supports_prerequisite=False,
        handle=42,
        prefetch=False,
        cancelled=False,
    ):
        events = []
        submitted = []
        reaped = set()
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj.num_layers = 4
        obj._tp_rank = 0
        obj._dp_rank = obj._attn_dp_rank = 0
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

        def submit(prepared, layer, prerequisite_handle=0):
            self.assertIn(("indices_ready", None), events)
            if forward_event:
                if supports_prerequisite and handle:
                    self.assertNotIn(("forward_done", None), events)
                    self.assertEqual(prerequisite_handle, handle)
                else:
                    self.assertIn(("forward_done", None), events)
                    self.assertEqual(prerequisite_handle, 0)
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
            supports_transfer_prerequisite=supports_prerequisite,
            prepare_load=lambda transfers: transfers,
            submit_prepared_load_layer=submit,
            check=check,
            wait=wait,
        )
        ready_events = [
            types.SimpleNamespace(
                synchronize=lambda: events.append(("indices_ready", None))
            )
        ]
        if forward_event:
            ready_events.append(
                types.SimpleNamespace(
                    cuda_event=handle,
                    synchronize=lambda: events.append(("forward_done", None)),
                )
            )
        if prefetch:
            future = Future()
            future.add_done_callback(lambda f: events.append(
                ("result", "cancelled" if f.cancelled() else f.result())
            ))
            if cancelled:
                future.cancel()
            obj._load_queue.put(linker_module._PrefetchLoad(
                [object()], ready_events, future
            ))
        else:
            obj._load_queue.put((0, {"req": [object()]}, ready_events))
        obj._load_queue.put(None)
        obj._load_worker()
        self.assertEqual(obj._load_queue.unfinished_tasks, 0)
        self.assertEqual(reaped, set(submitted))
        return events

    def test_prefetch_completes_after_all_layers_without_layer_notifications(self):
        with patch.object(linker_module, "Future", Future, create=True):
            events = self.run_worker(prefetch=True)
        self.assertEqual(events[-1], ("result", True))
        self.assertEqual(len([e for e in events if e[0] == "wait"]), 4)
        self.assertFalse(any(e[0] in ("complete", "fail") for e in events))

    def test_prefetch_failures_drain_before_future_completion(self):
        for failure in ({"fail_submit": 2}, {"fail_wait": 1}, {"fail_check": 0}):
            with self.subTest(**failure):
                events = self.run_worker(prefetch=True, **failure)
                self.assertEqual(events[-1], ("result", False))
                self.assertFalse(any(e[0] in ("complete", "fail") for e in events))

    def test_cancelled_prefetch_never_touches_device(self):
        events = self.run_worker(prefetch=True, cancelled=True)
        self.assertEqual(events, [("result", "cancelled")])

    def test_prefetch_entry_records_dependencies_and_rejects_unavailable_worker(self):
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._closed = False
        obj._load_thread = types.SimpleNamespace(is_alive=lambda: True)
        obj._load_queue = queue.Queue()
        obj._expand = lambda transfers, **kwargs: transfers
        obj._freeze_gc_once = lambda: None
        ready = [object()]
        obj._record_transfer_events = lambda: ready
        transfers = [object()]
        with patch.object(linker_module, "Future", Future, create=True):
            future = obj.prefetch(transfers)
            task = obj._load_queue.get_nowait()
            self.assertIs(task.future, future)
            self.assertIs(task.ready_events, ready)
            self.assertIs(task.transfers, transfers)
            self.assertFalse(future.done())
            self.assertIsNone(obj.prefetch([]))
            obj._closed = True
            self.assertIsNone(obj.prefetch(transfers))
            obj._closed = False
            obj._load_thread = types.SimpleNamespace(is_alive=lambda: False)
            self.assertIsNone(obj.prefetch(transfers))
        self.assertTrue(obj._load_queue.empty())

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

    def test_forward_event_is_passed_to_every_layer(self):
        events = self.run_worker(forward_event=True, supports_prerequisite=True)
        self.assertEqual(events[-1], ("result", True))

    def test_forward_event_falls_back_to_cpu_wait(self):
        for supports, handle in ((False, 42), (True, 0)):
            with self.subTest(supports=supports, handle=handle):
                events = self.run_worker(
                    forward_event=True,
                    supports_prerequisite=supports,
                    handle=handle,
                )
                self.assertLess(
                    events.index(("indices_ready", None)),
                    events.index(("forward_done", None)),
                )
                self.assertEqual(events[-1], ("result", True))

    def test_start_records_indices_and_forward_on_their_streams(self):
        records = []
        forward_stream = object()

        def event():
            index = len(records)
            records.append(None)
            return types.SimpleNamespace(
                record=lambda stream=None: records.__setitem__(index, stream)
            )

        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._pending_loads = {"req": [object()]}
        obj.forward_stream = forward_stream
        obj._requires_load_finish = False
        obj._freeze_gc_once = lambda: None
        obj.layer_done_counter = types.SimpleNamespace(update_producer=lambda: 0)
        obj._load_queue = queue.Queue()
        with patch.object(
            linker_module, "device_module", types.SimpleNamespace(Event=event)
        ):
            self.assertEqual(obj.start_layer_wise_loading(), 0)
        self.assertEqual(records, [None, forward_stream])
        self.assertEqual(len(obj._load_queue.get()[2]), 2)

    def test_native_transfer_ready_is_reused_for_load_and_dump(self):
        obj = object.__new__(linker_module.UcmDirectLinker)
        ready_events = [object(), object()]
        obj.forward_stream = object()
        obj._pending_loads = {"req": [object()]}
        obj._requires_load_finish = False
        obj._freeze_gc_once = lambda: None
        obj.layer_done_counter = types.SimpleNamespace(update_producer=lambda: 0)
        obj._load_queue = queue.Queue()
        obj._offload_queue = queue.Queue()
        obj._dump_ranks = (0,)
        obj._tp_rank = 0
        obj._expand = lambda transfers, **kwargs: transfers
        streams = []

        def record_ready(linker):
            streams.append(linker.forward_stream)
            return types.SimpleNamespace(events=ready_events)

        with patch.object(
            linker_module.UcmDirectLinker,
            "record_transfer_ready",
            record_ready,
            create=True,
        ):
            obj.start_layer_wise_loading()
            self.assertTrue(obj.offload([types.SimpleNamespace(keys=("a",))]))
        self.assertIs(obj._load_queue.get()[2], ready_events)
        self.assertIs(obj._offload_queue.get()[1], ready_events)
        self.assertEqual(streams, [obj.forward_stream, obj.forward_stream])

    def test_legacy_transfer_ready_without_forward_stream(self):
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj.forward_stream = None
        recorded = []
        event = types.SimpleNamespace(record=lambda: recorded.append(True))
        with patch.object(
            linker_module, "device_module", types.SimpleNamespace(Event=lambda: event)
        ):
            self.assertEqual(obj._record_transfer_events(), [event])
        self.assertEqual(recorded, [True])
