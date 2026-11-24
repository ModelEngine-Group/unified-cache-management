"""Exercise the real linker executors with a controllable Store, without GPUs."""

import hashlib
import importlib.util
import logging
import sys
import types
import unittest
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock
from unittest.mock import Mock, patch


def load_linker(*, mamba=False):
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
        "sglang.srt.utils",
        "sglang.srt.mem_cache.hicache_storage",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    modules["ucm.logger"] = types.SimpleNamespace(init_logger=logging.getLogger)
    modules["torch"].distributed = modules["torch.distributed"]
    modules["torch.distributed"].is_initialized = lambda: False
    device_module = Mock()
    modules["sglang.srt.utils"].get_device_module = lambda: device_module
    group = types.SimpleNamespace(rank_replicated=False)
    modules[names[6]].resolve_hybrid_device_pool_group = lambda **kwargs: group
    class Counter(Mock):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.consumer_index = 2

        def wait_until(self, threshold):
            pass

        def reset(self):
            pass

        def finish(self, index):
            return True

    modules[names[8]].LayerWiseLoadCounter = Counter
    modules[names[8]].UnifiedCacheLinker = object
    modules[names[9]].get_model = lambda: types.SimpleNamespace(model_path="test")
    modules[names[9]].get_parallel = lambda: types.SimpleNamespace(
        dp_rank=None, attn_dp_rank=0
    )
    layout = Mock(
        num_layers=1,
        identity="test-layout",
        section_sizes={"FA": 8192},
        device_id=3,
        has_shared_shards=False,
    )
    modules[names[-1]].PoolName = types.SimpleNamespace(MAMBA="mamba")
    layout.entries = (types.SimpleNamespace(name="mamba"),) if mamba else ()
    layout.expand.side_effect = lambda transfers, **kwargs: (
        [(layout, transfers)] if transfers else []
    )
    layout.block_ids.return_value = (b"x" * 16,)
    layout.prepare.return_value = types.SimpleNamespace(
        keys=("page",), payloads=lambda: [object()]
    )
    modules["v2_offload_test.device_layout"].DeviceLayout = (
        lambda *args, **kwargs: layout
    )
    modules["v2_offload_test.device_layout"].fingerprint = lambda _: "a" * 64
    modules["ucm.integration.sglang.patch.request_hooks"] = types.SimpleNamespace(
        install_request_hooks=lambda linker_type, **kwargs: None
    )
    modules["v2_offload_test.runtime"].Runtime = lambda config, **kwargs: config[
        "runtime"
    ]
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
        self.put_tokens = {}
        self.finished_requests = []
        self.active = set()
        self.rejected = set()
        self.prepare_error = False
        self.receipt_error = False
        self.auto_finish = False

    def on_new_request(self, context):
        pass

    def lookup(self, rid, keys):
        return [True], False

    def lookup_on_prefix(self, rid, keys):
        return 0, False

    def prepare_put(self, rid, keys):
        token = types.SimpleNamespace(
            keys=[types.SimpleNamespace(to_bytes=lambda: keys)]
        )
        self.put_tokens[rid] = token
        return token

    def complete_put(self, rid, token, success):
        if self.put_tokens.pop(rid) is not token:
            raise AssertionError("CompletePut must use the PreparePut token")
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


@dataclass
class PartitionTransfer:
    name: str
    keys: list
    host_indices: object
    hit_policy: str = "all_pages"


class PartitionIndices:
    """Minimal list-like stand-in for a flattened device index tensor."""

    def __init__(self, values):
        self.values = list(values)

    def reshape(self, *shape):
        del shape
        return self

    def __getitem__(self, item):
        return self.values[item]


class OffloadTest(unittest.TestCase):
    def setUp(self):
        self.module = load_linker()
        self.store = Store()
        params = types.SimpleNamespace(
            page_size=1,
            token_to_kv_pool_allocator=Mock(),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            req_to_token_pool=Mock(),
        )
        self.params = params
        self.linker = self.module.UcmDirectLinker(
            None, params, components=(), extra_config={"runtime": self.store}
        )
        self.linker.record_transfer_ready = lambda: types.SimpleNamespace(
            events=[types.SimpleNamespace(synchronize=lambda: None)]
        )
        self.addCleanup(self.cleanup)

    def test_counter_aggregates_short_waits_once_per_batch(self):
        counter = self.module._MeasuredLayerWiseLoadCounter(3)
        with (
            patch.object(self.module, "perf_counter", side_effect=[1, 1.01, 2, 2.01, 3, 3.01]),
            patch.object(self.module.LayerWiseLoadCounter, "wait_until") as wait,
            patch.object(self.module.logger, "info") as info,
        ):
            counter.wait_until(0)
            counter.wait_until(1)
            info.assert_not_called()
            counter.wait_until(2)
            counter.finish(2)
        self.assertEqual(wait.call_count, 3)
        info.assert_called_once()
        self.assertEqual(info.call_args.args[1:3], (2, 3))
        self.assertAlmostEqual(info.call_args.args[3], 30.0)
        self.assertAlmostEqual(info.call_args.args[4], 10.0)

    def test_counter_does_not_log_fast_wait(self):
        with (
            patch.object(self.module, "perf_counter", side_effect=[1.0, 1.001]),
            patch.object(self.module.logger, "info") as info,
        ):
            self.linker.layer_done_counter.wait_until(0)
        info.assert_not_called()

    def test_load_summary_separates_queue_and_event_time(self):
        clock = [100, 100, 100.005, 100.005, 100.006,
                 100.006, 100.006, 100.006, 100.010, 100.010]
        event = types.SimpleNamespace(synchronize=lambda: None)
        with (
            patch.object(self.module, "perf_counter", side_effect=clock),
            patch.object(self.module.logger, "info") as info,
            patch.object(self.linker, "_load_layers"),
        ):
            self.assertTrue(self.linker._load_worker(0, {}, [event], queued_at=99))
        values = info.call_args.args
        self.assertAlmostEqual(values[3], 1010.0)
        self.assertAlmostEqual(values[4], 1000.0)
        self.assertAlmostEqual(values[5], 5.0)

    def test_load_metrics_count_bytes_and_publish_only_after_completion(self):
        metrics = dict(payload_ms=0.0, payload_cpu_ms=0.0,
                       addr_ms=0.0, tolist_ms=0.0, pack_ms=0.0,
                       io_queue_ms=0.0, submit_ms=0.0,
                       job_latency_sum_ms=0.0, publish_lag_ms=0.0,
                       bytes=0, segments=0, jobs=0)
        payload = types.SimpleNamespace(sizes=[16, 32])
        pages = types.SimpleNamespace(
            payloads=lambda layer, timings=None: [payload, payload]
        )
        self.linker.layer_done_counter.complete = Mock()
        with ThreadPoolExecutor(max_workers=1) as executor:
            loading = executor.submit(self.linker._load_layers, 0,
                                      [(pages, object())], object(), metrics)
            job_id, _ = self.next_job("get")
            self.linker.layer_done_counter.complete.assert_not_called()
            self.store.finish(job_id)
            loading.result(timeout=2)
        self.linker.layer_done_counter.complete.assert_called_once_with(0, 0)
        self.assertEqual(metrics["bytes"], 96)
        self.assertEqual(metrics["segments"], 4)
        self.assertEqual(metrics["jobs"], 1)
        self.assertGreaterEqual(metrics["job_latency_sum_ms"], metrics["submit_ms"])
        self.assertGreaterEqual(metrics["publish_lag_ms"], 0)

    def test_load_completion_measurement_uses_observed_time(self):
        result = Future()
        result.load_metrics = {"job_latency_sum_ms": 0.0}
        result.submitted_at = 10.0
        with patch.object(self.module, "perf_counter", return_value=10.025):
            self.linker._measure_load_completion(result)
        self.assertAlmostEqual(result.load_metrics["job_latency_sum_ms"], 25.0)
        self.assertFalse(result.done())

    def test_mamba_pool_uses_the_linkers_layer_counter(self):
        self.params.req_to_token_pool.register_layer_transfer_counter.assert_not_called()
        module = load_linker(mamba=True)
        linker = module.UcmDirectLinker(
            None, self.params, components=(), extra_config={"runtime": self.store}
        )
        self.addCleanup(linker.close)
        self.params.req_to_token_pool.register_layer_transfer_counter.assert_called_once_with(
            linker.layer_done_counter
        )

    def test_cache_domain_is_shared_within_a_dp_replica(self):
        domains = []

        def make_runtime(config, *, unique_id, **kwargs):
            domains.append(unique_id)
            return config["runtime"]

        with (
            patch.object(self.module, "Runtime", side_effect=make_runtime),
            patch.object(
                self.module,
                "fingerprint",
                side_effect=lambda value: hashlib.sha256(
                    repr(value).encode()
                ).hexdigest(),
            ),
            patch.object(self.module, "get_parallel") as get_parallel,
        ):
            for dp_rank, attn_dp_rank, tp_rank in (
                (None, 0, 0),
                (0, 0, 1),
                (1, 0, 0),
                (None, 1, 0),
            ):
                get_parallel.return_value = types.SimpleNamespace(
                    dp_rank=dp_rank, attn_dp_rank=attn_dp_rank, tp_rank=tp_rank
                )
                linker = self.module.UcmDirectLinker(
                    None, self.params, components=(), extra_config={"runtime": Store()}
                )
                self.addCleanup(linker.close)

        self.assertEqual(domains[0], domains[1])
        self.assertEqual(len(set(domains)), 3)

    def test_single_host_guard_checks_replica_groups(self):
        params = types.SimpleNamespace(
            tp_cache_group="tp",
            attn_tp_cache_group="tp",
            attn_cp_cache_group="cp",
            pp_cache_group="pp",
        )
        for affected_group in (None, "tp", "cp", "pp"):
            for peer_host in ("other-host", None):
                with self.subTest(group=affected_group, peer_host=peer_host):
                    visited = []

                    def gather(output, local_hosts, *, group):
                        visited.append(group)
                        peer = peer_host if group == affected_group else "local-host"
                        output[:] = [local_hosts, {peer}]

                    distributed = Mock()
                    distributed.is_initialized.return_value = True
                    distributed.get_world_size.return_value = 2
                    distributed.all_gather_object.side_effect = gather
                    with (
                        patch.object(self.module, "dist", distributed),
                        patch("builtins.open", return_value=StringIO("local-host")),
                    ):
                        if affected_group is None:
                            self.linker._require_single_host(params)
                        else:
                            with self.assertRaisesRegex(ValueError, "each DP replica"):
                                self.linker._require_single_host(params)
                    self.assertEqual(visited, ["tp", "cp", "pp"])

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
        self.assertTrue(self.linker.offload((types.SimpleNamespace(keys=("page",)),)))
        job_id, _ = self.next_job("put")
        node = self.linker._offload_results[-1]
        return job_id, node

    def test_sharded_put_waits_for_group_verdict(self):
        self.linker._coordinated_puts = True
        for group_success in (True, False):
            job_id, node = self.put()
            self.store.completions.put(
                types.SimpleNamespace(job_id=job_id, success=True)
            )
            self.assertTrue(node.result(timeout=2))
            self.assertTrue(self.linker.pop_completed_offload())
            self.assertEqual(len(self.store.commits), int(not group_success))
            self.assertEqual(
                self.linker.commit_completed_puts([group_success]), [group_success]
            )
            self.assertEqual(self.store.commits[-1][1], group_success)

    def test_sharded_put_reset_aborts_without_group_verdict(self):
        self.linker._coordinated_puts = True
        job_id, node = self.put()
        self.store.completions.put(types.SimpleNamespace(job_id=job_id, success=True))
        self.assertTrue(node.result(timeout=2))
        self.linker.reset()
        self.assertEqual([success for _, success in self.store.commits], [False])

    def test_non_owner_lookup_registers_request_without_store_query(self):
        self.linker._lookup_owner = False
        transfers = (types.SimpleNamespace(keys=("page",), hit_policy="all_pages"),)
        with patch.object(self.store, "lookup", side_effect=AssertionError("queried")):
            self.assertEqual(self.linker.lookup("r", transfers), [1])
        self.assertIn("r", self.linker._registered_requests)
        self.assertEqual(self.linker._page_requests["page"], {"r"})
        self.linker.finish_request("r")
        self.assertIn("r", self.store.finished_requests)

    def test_dump_uses_each_lookup_request_and_holds_its_route_until_commit(self):
        transfers = (types.SimpleNamespace(keys=("page",), hit_policy="all_pages"),)
        for rid in ("a", "b"):
            self.assertEqual(self.linker.lookup(rid, transfers), [1])
        first, node = self.put()
        second, _ = self.next_job("put")
        self.linker.finish_request("a")
        self.linker.finish_request("b")
        self.assertEqual(self.store.finished_requests, [])
        self.store.finish(first)
        self.store.finish(second)
        self.assertTrue(node.result(timeout=2))
        self.assertCountEqual(self.store.commits, [("a", True), ("b", True)])
        self.assertCountEqual(self.store.finished_requests, ["a", "b"])

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

        def slow_commit(rid, token, success):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release Put commit")
            complete_put(rid, token, success)

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

    def _fa_section(self, pages):
        entries = [
            types.SimpleNamespace(page_size=2),
            types.SimpleNamespace(page_size=2),
        ]
        indices = PartitionIndices(range(pages * 2))
        transfers = tuple(
            PartitionTransfer(
                name=name,
                keys=[f"page-{i}" for i in range(pages)],
                host_indices=indices,
            )
            for name in ("kv", "indexer")
        )
        return types.SimpleNamespace(name="FA", entries=entries), transfers

    def _swa_section(self, pages=2):
        return (
            types.SimpleNamespace(
                name="SWA", entries=[types.SimpleNamespace(page_size=2)]
            ),
            (
                PartitionTransfer(
                    name="swa",
                    keys=[f"tail-{i}" for i in range(pages)],
                    host_indices=PartitionIndices(range(pages * 2)),
                    hit_policy="trailing_pages",
                ),
            ),
        )

    def test_tail_owner_keeps_trailing_window_whole_and_agrees_across_ranks(self):
        fa_section, fa_transfers = self._fa_section(8)
        swa_section, swa_transfers = self._swa_section()
        sections = [(fa_section, fa_transfers), (swa_section, swa_transfers)]
        owners = []
        for rank in range(4):
            self.linker._dump_rank = rank
            self.linker._dump_size = 4
            owner = self.linker._tail_owner(sections)
            owners.append(owner)
            selected = self.linker._partition_offload(swa_section, swa_transfers, owner)
            if rank == owner:
                self.assertIs(selected, swa_transfers)
            else:
                self.assertEqual(selected, ())
        self.assertEqual(len(set(owners)), 1)
        # Prefers the FA tail key, so the owner is the hash of "page-7".
        expected = int.from_bytes(hashlib.sha256(b"page-7").digest()[:8], "little") % 4
        self.assertEqual(owners[0], expected)

    def test_fa_pages_partitioned_disjoint_complete_and_balanced(self):
        for pages in (1, 2, 3, 5, 8, 11):
            all_keys = []
            slice_lengths = []
            for rank in range(4):
                self.linker._dump_rank = rank
                self.linker._dump_size = 4
                fa_section, fa_transfers = self._fa_section(pages)
                selected = self.linker._partition_offload(fa_section, fa_transfers, 0)
                start = pages * rank // 4
                end = pages * (rank + 1) // 4
                if start == end:
                    self.assertEqual(selected, ())
                    slice_lengths.append(0)
                    continue
                expected_keys = [f"page-{i}" for i in range(start, end)]
                self.assertEqual([t.keys for t in selected], [expected_keys] * 2)
                self.assertIs(selected[0].host_indices, selected[1].host_indices)
                slice_lengths.append(end - start)
                all_keys.extend(selected[0].keys)
            self.assertEqual(all_keys, [f"page-{i}" for i in range(pages)])
            self.assertLessEqual(max(slice_lengths) - min(slice_lengths), 1)

    def test_single_rank_partition_keeps_transfers_unchanged(self):
        fa_section, fa_transfers = self._fa_section(5)
        swa_section, swa_transfers = self._swa_section()
        self.linker._dump_rank = 0
        self.linker._dump_size = 1
        self.assertIs(
            self.linker._partition_offload(fa_section, fa_transfers, 0), fa_transfers
        )
        self.assertIs(
            self.linker._partition_offload(swa_section, swa_transfers, 0),
            swa_transfers,
        )


if __name__ == "__main__":
    unittest.main()
