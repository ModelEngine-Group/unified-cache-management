"""Exercise the real linker executors with a controllable Store, without GPUs."""

import importlib.util
import sys
import types
import unittest
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock
from unittest.mock import Mock, patch


def load_linker():
    names = [
        "torch",
        "torch.distributed",
        "sglang",
        "sglang.srt",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.hybrid_cache",
        "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler",
        "sglang.srt.mem_cache.unified_cache",
        "sglang.srt.mem_cache.unified_cache.unified_cache_linker",
        "sglang.srt.runtime_context",
        "v2_offload_test",
        "v2_offload_test.device_layout",
        "v2_offload_test.runtime",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    modules["torch"].distributed = modules["torch.distributed"]
    group = types.SimpleNamespace(rank_replicated=False)
    modules[names[6]].resolve_hybrid_device_pool_group = lambda **kwargs: group
    modules[names[8]].LayerWiseLoadCounter = lambda layers: Mock()
    modules[names[8]].UnifiedCacheLinker = object
    modules[names[9]].get_model = lambda: types.SimpleNamespace(model_path="test")
    layout = Mock(num_layers=1)
    layout.expand.side_effect = lambda transfers: transfers
    layout.block_ids.return_value = (b"x" * 16,)
    layout.prepare.return_value = types.SimpleNamespace(
        keys=("page",), payloads=lambda: [object()]
    )
    modules["v2_offload_test.device_layout"].DsaLayout = lambda *args: layout
    modules["v2_offload_test.runtime"].Runtime = lambda config, **kwargs: config["runtime"]
    modules["v2_offload_test.runtime"].proxy = lambda: types.SimpleNamespace(
        RequestContext=types.SimpleNamespace,
        IoContext=lambda: types.SimpleNamespace(prerequisite_handle=0),
    )
    path = Path(__file__).resolve().parents[1] / "ucm/integration/sglang/v2/linker.py"
    spec = importlib.util.spec_from_file_location("v2_offload_test.linker", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class Store:
    def __init__(self):
        self.submissions = Queue()
        self.completions = Queue()
        self.receipts = Queue()
        self.lock = Lock()
        self.batches = deque()
        self.commits = []
        self.finished_requests = []
        self.active = set()
        self.rejected = set()
        self.prepare_error = False
        self.receipt_error = False
        self.auto_finish = False

    def on_new_request(self, context):
        pass

    def prepare_put(self, rid, keys):
        return types.SimpleNamespace(
            keys=[types.SimpleNamespace(to_bytes=lambda: keys)]
        )

    def complete_put(self, rid, success):
        self.commits.append((rid, success))

    def on_request_finished(self, rid):
        self.finished_requests.append(rid)

    def submit_put(self, job_id, payloads, token, context):
        self._submit("put", job_id, None)

    def submit_get(self, job_id, payloads, token, context):
        self._submit("get", job_id, None)

    def add_plans(self, *plans):
        with self.lock:
            self.batches.append(plans)

    def prepare_offload(self):
        if self.prepare_error:
            raise RuntimeError("prepare failed")
        with self.lock:
            return self.batches.popleft() if self.batches else ()

    def submit_offload(self, job_id, plan):
        if plan in self.rejected:
            raise RuntimeError("not accepted")
        self._submit("disk", job_id, plan)

    def _submit(self, kind, job_id, plan):
        with self.lock:
            self.active.add(job_id)
        self.submissions.put((kind, job_id, plan))

    def finish(self, job_id, success=True):
        self.completions.put(types.SimpleNamespace(job_id=job_id, success=success))

    def get_finished(self):
        results = []
        with self.lock:
            if self.auto_finish:
                results = [
                    types.SimpleNamespace(job_id=job_id, success=True)
                    for job_id in self.active
                ]
                self.active.clear()
                return results
            while True:
                try:
                    result = self.completions.get_nowait()
                except Empty:
                    break
                self.active.remove(result.job_id)
                results.append(result)
        return results

    def complete_offload(self, plan):
        if self.receipt_error:
            raise RuntimeError("receipt failed")
        self.receipts.put(plan)


class OffloadTest(unittest.TestCase):
    def setUp(self):
        self.module = load_linker()
        self.store = Store()
        params = types.SimpleNamespace(
            page_size=1,
            token_to_kv_pool_allocator=Mock(),
            attn_tp_cache_group=None,
            tp_cache_group=None,
        )
        self.linker = self.module.UcmDirectLinker(
            None, params, components=(), extra_config={"runtime": self.store}
        )
        self.linker.record_transfer_ready = lambda: types.SimpleNamespace(
            events=[types.SimpleNamespace(synchronize=lambda: None)]
        )
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.store.prepare_error = False
        self.store.receipt_error = False
        self.store.auto_finish = True
        self.linker.close()

    def next_job(self, kind):
        actual, job_id, plan = self.store.submissions.get(timeout=2)
        self.assertEqual(actual, kind)
        return job_id, plan

    def put(self):
        self.assertTrue(self.linker.offload((object(),)))
        job_id, _ = self.next_job("put")
        node = self.linker._offload_results[-1]
        return job_id, node

    def test_slow_foreign_plan_does_not_delay_or_fail_nodes(self):
        self.store.add_plans("other-node/other-branch")
        put_id, first = self.put()
        self.store.finish(put_id)
        self.assertTrue(first.result(timeout=2))
        disk_id, plan = self.next_job("disk")
        put_id, second = self.put()
        self.store.finish(put_id)
        self.assertTrue(second.result(timeout=2))
        self.assertEqual(self.linker.num_completed_offloads(), 2)
        self.assertTrue(self.linker.pop_completed_offload())
        self.assertTrue(self.linker.pop_completed_offload())
        self.assertTrue(self.store.receipts.empty())
        with self.assertLogs(self.module.logger, level="ERROR"):
            self.store.finish(disk_id, False)
            self.assertEqual(self.store.receipts.get(timeout=2), plan)
            self.linker._io_executor.submit(lambda: None).result(timeout=2)
        self.assertTrue(first.result())
        self.assertTrue(second.result())
        self.assertEqual(len(self.store.finished_requests), 2)

    def test_scheduler_tick_drains_without_node_put_and_bounds_batches(self):
        self.store.add_plans("first")
        self.store.add_plans("second")
        self.assertEqual(self.linker.num_completed_offloads(), 0)
        first, _ = self.next_job("disk")
        self.linker.num_completed_offloads()
        self.linker._io_executor.submit(lambda: None).result(timeout=2)
        self.assertTrue(self.store.submissions.empty())
        self.store.finish(first)
        self.assertEqual(self.store.receipts.get(timeout=2), "first")
        second, _ = self.next_job("disk")
        self.store.finish(second)
        self.assertEqual(self.store.receipts.get(timeout=2), "second")

    def test_rejected_plan_is_receipted_without_replay(self):
        self.store.rejected.add("rejected")
        self.store.add_plans("rejected", "accepted")
        with self.assertLogs(self.module.logger, level="ERROR"):
            self.linker.num_completed_offloads()
            self.assertEqual(self.store.receipts.get(timeout=2), "rejected")
            job_id, _ = self.next_job("disk")
        self.store.finish(job_id)
        self.assertEqual(self.store.receipts.get(timeout=2), "accepted")
        self.linker.reset()
        self.assertTrue(self.store.submissions.empty())
        self.assertTrue(self.store.receipts.empty())

    def test_prepare_failure_does_not_change_put_result(self):
        self.store.prepare_error = True
        job_id, node = self.put()
        with self.assertLogs(self.module.logger, level="ERROR"):
            self.store.finish(job_id)
            self.assertTrue(node.result(timeout=2))
            self.linker.num_completed_offloads()
            self.linker._control_executor.submit(lambda: None).result(timeout=2)

    def test_slow_put_commit_does_not_block_get_completion(self):
        entered = Event()
        release = Event()
        complete_put = self.store.complete_put

        def slow_commit(rid, success):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release Put commit")
            complete_put(rid, success)

        self.store.complete_put = slow_commit
        put_id, node = self.put()
        self.store.finish(put_id)
        try:
            self.assertTrue(entered.wait(timeout=2))
            get_result = Future()
            self.linker._io_executor.submit(
                self.linker._submit_job,
                self.store.submit_get,
                get_result,
                [],
                object(),
                object(),
            ).result(timeout=2)
            get_id, _ = self.next_job("get")
            self.store.finish(get_id)
            self.assertTrue(get_result.result(timeout=2))
            self.assertFalse(node.done())
        finally:
            release.set()
        self.assertTrue(node.result(timeout=2))

    def test_put_failure_is_not_reported_as_success(self):
        job_id, node = self.put()
        self.store.finish(job_id, False)
        self.assertFalse(node.result(timeout=2))
        self.assertFalse(self.linker.pop_completed_offload())
        self.assertFalse(self.store.commits[-1][1])

    def test_out_of_order_puts_keep_node_publication_order(self):
        first_id, first = self.put()
        second_id, second = self.put()
        self.store.finish(second_id)
        self.assertTrue(second.result(timeout=2))
        self.assertFalse(first.done())
        self.assertEqual(self.linker.num_completed_offloads(), 0)
        self.store.finish(first_id)
        self.assertTrue(first.result(timeout=2))
        self.assertEqual(self.linker.num_completed_offloads(), 2)

    def test_get_and_store_offload_share_completion_stream_without_crosstalk(self):
        self.store.add_plans("disk")
        self.linker.num_completed_offloads()
        disk_id, _ = self.next_job("disk")
        get_result = Future()
        self.linker._io_executor.submit(
            self.linker._submit_job,
            self.store.submit_get,
            get_result,
            [],
            object(),
            object(),
        ).result(timeout=2)
        get_id, _ = self.next_job("get")
        self.assertNotEqual(get_id, disk_id)
        self.store.finish(get_id, False)
        self.assertFalse(get_result.result(timeout=2))
        self.assertTrue(self.store.receipts.empty())
        self.store.finish(disk_id)
        self.assertEqual(self.store.receipts.get(timeout=2), "disk")

    def test_close_waits_for_independent_disk_job_and_receipt(self):
        self.store.add_plans("disk")
        self.linker.num_completed_offloads()
        job_id, _ = self.next_job("disk")
        entered = Event()
        original = self.linker._flush_store_offloads

        def flush():
            entered.set()
            original()

        self.linker._flush_store_offloads = flush
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing = executor.submit(self.linker.close)
            try:
                self.assertTrue(entered.wait(timeout=2))
                self.assertFalse(closing.done())
                self.assertTrue(self.store.receipts.empty())
                with self.assertRaisesRegex(RuntimeError, "closing"):
                    self.linker.offload((object(),))
            finally:
                self.store.finish(job_id)
            closing.result(timeout=2)
        self.assertEqual(self.store.receipts.get(timeout=2), "disk")
        self.assertTrue(self.linker._closed)

    def test_reset_drains_pending_plans_without_scheduler_ticks(self):
        self.store.add_plans("pending")
        with ThreadPoolExecutor(max_workers=1) as executor:
            resetting = executor.submit(self.linker.reset)
            job_id, _ = self.next_job("disk")
            self.assertFalse(resetting.done())
            self.store.finish(job_id)
            resetting.result(timeout=2)
        self.assertEqual(self.store.receipts.get(timeout=2), "pending")
        self.assertFalse(self.linker._closing)

    def test_close_continues_after_entire_batch_is_rejected(self):
        self.store.add_plans("rejected")
        self.store.add_plans("next-batch")
        self.store.rejected.add("rejected")
        with self.assertLogs(self.module.logger, level="ERROR"):
            with ThreadPoolExecutor(max_workers=1) as executor:
                closing = executor.submit(self.linker.close)
                job_id, plan = self.next_job("disk")
                self.assertEqual(plan, "next-batch")
                self.store.finish(job_id)
                closing.result(timeout=2)
        self.assertEqual(self.store.receipts.get_nowait(), "rejected")
        self.assertEqual(self.store.receipts.get_nowait(), "next-batch")

    def test_failed_receipt_prevents_successful_close(self):
        self.store.add_plans("unreleased")
        self.store.rejected.add("unreleased")
        self.store.receipt_error = True
        with self.assertLogs(self.module.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "receipt failed"):
                self.linker.close()
        self.assertFalse(self.linker._closed)
        self.assertTrue(self.store.receipts.empty())
        # Explicitly model operator recovery of the failed receipt for cleanup.
        self.store.receipt_error = False
        self.store.complete_offload("unreleased")
        self.linker._store_offload_error = None


if __name__ == "__main__":
    unittest.main()
