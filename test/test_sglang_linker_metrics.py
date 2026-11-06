"""CPU-only tests of actual adapter/worker paths with deterministic metric clocks."""

import collections
import json
import queue
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_sglang_direct_preparation as preparation
import yaml
from common.sglang_utils import (
    adapter_module,
    linker_module,
    metrics_module,
)


class LinkerMetricsTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.samples = []
        for module in (metrics_module, adapter_module, linker_module):
            self.enterContext(
                patch.object(module, "record_metrics", self.samples.append)
            )
        self.enterContext(
            patch.object(
                metrics_module,
                "time",
                types.SimpleNamespace(perf_counter=lambda: self.now),
            )
        )
        self.enterContext(
            patch.object(
                linker_module,
                "time",
                types.SimpleNamespace(perf_counter=lambda: self.now),
                create=True,
            )
        )
        self.fixture = preparation.LayoutPreparationTest()
        self.adapter = self.fixture.adapter()
        self.adapter.metrics_enabled = True
        self.store = types.SimpleNamespace(
            load_data=Mock(side_effect=[1, 2, 3, 4]),
            dump_data=Mock(side_effect=[1, 2, 3, 4]),
            wait=Mock(),
            check=Mock(return_value=True),
        )
        self.adapter._ensure_store_schema = lambda *a, **kw: self.store

    def total(self, name):
        return sum(sample.get(name, 0) for sample in self.samples)

    def test_prefetched_layers_record_at_wait_with_shard_bytes(self):
        prepared = self.adapter.prepare_load([self.fixture.transfer()])
        first = self.adapter.submit_prepared_load_layer(prepared, 0)
        self.now = 11
        second = self.adapter.submit_prepared_load_layer(prepared, 2)
        self.assertEqual(self.samples, [])
        self.now = 13
        self.adapter.wait(second)
        self.now = 15
        self.adapter.wait(first)
        self.assertEqual(self.total("sglang_connector_load_requests_total"), 2)
        self.assertEqual(self.total("sglang_connector_load_bytes_total"), 32)
        self.assertEqual(self.total("sglang_connector_load_pages"), 4)
        self.assertEqual(self.total("sglang_connector_load_duration_ms"), 7000)
        first[0].observation.finish()
        self.assertEqual(self.total("sglang_connector_load_requests_total"), 2)

    def test_distributed_load_options_preserve_completion_metrics(self):
        self.store.supports_distributed_backend_load = True
        for replicated in (False, True):
            for event in (0, 43):
                with self.subTest(replicated=replicated, event=event):
                    self.samples.clear()
                    self.adapter._identity["rank_replicated"] = replicated
                    prepared = self.adapter.prepare_load([self.fixture.transfer()])
                    tasks = self.adapter.submit_prepared_load_layer(
                        prepared, 0, prerequisite_handle=event
                    )
                    expected = {"distribute_backend_load": replicated}
                    if event:
                        expected["prerequisite_handle"] = event
                    self.assertEqual(self.store.load_data.call_args.kwargs, expected)
                    self.assertEqual(self.samples, [])
                    self.adapter.wait(tasks)
                    self.assertEqual(
                        self.total("sglang_connector_load_requests_total"), 1
                    )
                    self.assertEqual(
                        self.total("sglang_connector_load_bytes_total"), 16
                    )

    def test_distributed_load_submit_failure_drains_and_records_metrics(self):
        self.store.supports_distributed_backend_load = True
        self.adapter._identity["rank_replicated"] = True
        self.store.load_data.side_effect = [1, RuntimeError("submit")]
        prepared = self.adapter.prepare_load(
            [self.fixture.transfer(), self.fixture.transfer()]
        )
        with self.assertLogs(adapter_module.logger, "ERROR"):
            with self.assertRaisesRegex(RuntimeError, "submit"):
                self.adapter.submit_prepared_load_layer(
                    prepared, 0, prerequisite_handle=43
                )
        for call in self.store.load_data.call_args_list:
            self.assertEqual(
                call.kwargs,
                {"prerequisite_handle": 43, "distribute_backend_load": True},
            )
        self.store.wait.assert_called_once_with(1)
        self.assertEqual(self.total("sglang_connector_load_requests_total"), 2)
        self.assertEqual(self.total("sglang_connector_load_submit_errors_total"), 1)
        self.assertEqual(self.total("sglang_connector_load_bytes_total"), 16)

    def test_partial_submit_failure_drains_success_and_records_failed_attempt(self):
        self.store.dump_data.side_effect = [1, RuntimeError("submit")]
        with self.assertLogs(adapter_module.logger, "ERROR"):
            with self.assertRaisesRegex(RuntimeError, "submit"):
                self.adapter.submit_offload(
                    [self.fixture.transfer(), self.fixture.transfer()]
                )
        self.store.wait.assert_called_once_with(1)
        self.assertEqual(self.total("sglang_connector_dump_requests_total"), 2)
        self.assertEqual(self.total("sglang_connector_dump_submit_errors_total"), 1)
        self.assertEqual(self.total("sglang_connector_dump_bytes_total"), 32)

    def test_wait_failure_does_not_skip_remaining_tasks(self):
        tasks = self.adapter.submit_offload(
            [self.fixture.transfer(), self.fixture.transfer()]
        )
        self.store.wait.side_effect = [RuntimeError("wait"), None]
        with self.assertRaisesRegex(RuntimeError, "wait"):
            self.adapter.wait(tasks)
        self.assertEqual(self.store.wait.call_count, 2)
        self.assertEqual(self.total("sglang_connector_dump_wait_errors_total"), 1)
        self.assertEqual(self.total("sglang_connector_dump_bytes_total"), 32)

    def test_check_failure_is_not_overwritten_by_successful_cleanup(self):
        tasks = self.adapter.submit_offload(
            [self.fixture.transfer(), self.fixture.transfer()]
        )
        self.store.check.side_effect = RuntimeError("check")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "check"):
                self.adapter.check(tasks[0])
        self.assertEqual(self.total("sglang_linker_dump_check_errors_total"), 1)
        self.assertEqual(self.total("sglang_connector_dump_requests_total"), 0)
        self.adapter.wait(tasks)
        self.assertEqual(self.total("sglang_connector_dump_wait_errors_total"), 1)
        self.assertEqual(self.total("sglang_connector_dump_bytes_total"), 32)

    def test_disabled_and_empty_observations_do_not_read_clock(self):
        self.adapter.metrics_enabled = False
        with patch.object(
            metrics_module.time, "perf_counter", side_effect=AssertionError("clock")
        ):
            tasks = self.adapter.submit_offload(
                [self.fixture.transfer(), self.fixture.transfer()]
            )
            self.adapter.wait(tasks)
            self.assertIsNone(
                metrics_module.AsyncIoObservation.begin(True, False, 0, 8)
            )
            self.assertIsNone(metrics_module.LinkerOperation.begin(False, "load"))
        self.assertEqual(self.samples, [])

    def test_operation_queue_and_completion_have_separate_boundaries(self):
        obs = metrics_module.LinkerOperation.begin(True, "offload")
        self.now = 12
        obs.start()
        self.assertEqual(self.total("sglang_linker_offload_completed_total"), 0)
        self.now = 17
        obs.finish(False)
        obs.finish(True)
        self.assertEqual(self.total("sglang_linker_offload_queue_duration_ms"), 2000)
        self.assertEqual(self.total("sglang_linker_offload_duration_ms"), 7000)
        self.assertEqual(self.total("sglang_linker_offload_errors_total"), 1)
        self.assertEqual(self.total("sglang_linker_offload_completed_total"), 1)

    def lookup_linker(self):
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._metrics_enabled = True
        obj._lookup_owner = True
        obj._lookup_progress = collections.OrderedDict()
        obj._expand = lambda transfers, **kw: transfers
        obj._page_transfers = lambda transfers, **kw: [
            self.fixture.transfer(addresses=None)
        ]
        obj.adapter = types.SimpleNamespace(lookup=Mock(return_value=[2, 4]))
        self.enterContext(
            patch.object(
                linker_module, "PoolName", types.SimpleNamespace(KV="kv"), create=True
            )
        )
        return obj, [types.SimpleNamespace(name="kv", keys=["a", "b", "c", "d"])]

    def test_lookup_sparse_boundaries_resume_and_non_owner(self):
        obj, transfers = self.lookup_linker()
        self.assertEqual(obj.lookup("r", transfers), [2, 4])
        obj.lookup("r", transfers)
        self.assertEqual(self.total("sglang_linker_lookup_query_pages_total"), 8)
        self.assertEqual(self.total("sglang_linker_lookup_hit_pages_total"), 8)
        self.assertEqual(self.total("sglang_linker_lookup_resumed_total"), 1)
        obj._lookup_owner = False
        self.assertEqual(obj.lookup("r", transfers), [1, 2, 3, 4])
        self.assertEqual(self.total("sglang_linker_lookup_hit_pages_total"), 8)
        self.assertEqual(obj.adapter.lookup.call_count, 2)
        self.assertEqual(self.total("sglang_linker_lookup_skipped_total"), 1)

    def test_lookup_error_records_no_successful_query_or_hit_pages(self):
        obj, transfers = self.lookup_linker()
        obj.adapter.lookup.side_effect = RuntimeError("lookup")
        with self.assertRaisesRegex(RuntimeError, "lookup"):
            obj.lookup("r", transfers)
        self.assertEqual(self.total("sglang_linker_lookup_errors_total"), 1)
        self.assertEqual(self.total("sglang_linker_lookup_requests_total"), 1)
        self.assertEqual(self.total("sglang_linker_lookup_query_pages_total"), 0)
        self.assertEqual(len(self.samples), 1)
        self.assertNotIn("r", obj._lookup_progress)

    def test_lookup_records_one_batch_without_including_export_time(self):
        obj, transfers = self.lookup_linker()

        def lookup(*args, **kwargs):
            self.now += 0.25
            return [2, 4]

        def record(values):
            self.samples.append(values)
            self.now += 1

        obj.adapter.lookup.side_effect = lookup
        with patch.object(linker_module, "record_metrics", record):
            self.assertEqual(obj.lookup("r", transfers), [2, 4])
        self.assertEqual(len(self.samples), 1)
        self.assertEqual(self.total("sglang_linker_lookup_duration_ms"), 250)
        self.assertEqual(self.total("sglang_linker_lookup_query_pages_total"), 4)
        self.assertEqual(self.total("sglang_linker_lookup_hit_pages_total"), 4)
        self.assertEqual(obj._lookup_progress["r"].hit_pages, 4)

    def test_disabled_lookup_preserves_result_without_clock_or_samples(self):
        obj, transfers = self.lookup_linker()
        obj._metrics_enabled = False
        with patch.object(
            linker_module.time, "perf_counter", side_effect=AssertionError("clock")
        ):
            self.assertEqual(obj.lookup("r", transfers), [2, 4])
            obj.adapter.lookup.side_effect = RuntimeError("lookup")
            with self.assertRaisesRegex(RuntimeError, "lookup"):
                obj.lookup("failed", transfers)
        self.assertEqual(self.samples, [])
        self.assertNotIn("failed", obj._lookup_progress)

    def test_resume_fallback_records_actual_probe(self):
        self.store.lookup_on_prefix = Mock(side_effect=[-1, 1])
        result = self.adapter.lookup(
            [self.fixture.transfer(addresses=None)],
            ["a", "b", "c"],
            resume_from_pages=3,
        )
        self.assertEqual(result, [1, 2])
        self.assertEqual(self.total("sglang_linker_lookup_fallback_total"), 1)

    def test_load_worker_counts_failure_and_drains_prefetched_tasks(self):
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._metrics_enabled, obj._tp_rank = True, 0
        obj._requires_load_finish = False
        obj._load_queue, obj._completed_loads = queue.Queue(), queue.Queue()
        obj._page_transfers = lambda *a, **kw: [self.fixture.transfer()]
        obj.num_layers = 3
        self.store.check.return_value = False
        obj.layer_done_counter = types.SimpleNamespace(complete=Mock(), fail=Mock())
        obj.adapter = self.adapter
        self.store.wait.side_effect = [RuntimeError("load"), None]
        observation = metrics_module.LinkerOperation.begin(True, "load")
        obj._load_queue.put(
            (
                1,
                {"rid": []},
                [types.SimpleNamespace(synchronize=lambda: None)],
                observation,
            )
        )
        obj._load_queue.put(None)
        with self.assertLogs(linker_module.logger, "ERROR"):
            obj._load_worker()
        self.assertEqual(self.store.wait.call_count, 2)
        self.assertEqual(self.total("sglang_linker_load_errors_total"), 1)
        self.assertEqual(self.total("sglang_linker_load_completed_total"), 1)
        self.assertEqual(self.total("sglang_connector_load_bytes_total"), 16)
        self.assertEqual(obj._load_queue.unfinished_tasks, 0)

    def test_config_overrides_and_disabled_default_adapter(self):
        document = {
            "enable_metrics": True,
            "metrics_config_path": "old.yaml",
            "ucm_connectors": [
                {
                    "ucm_connector_name": "test",
                    "ucm_connector_config": {},
                }
            ],
        }
        self.enterContext(patch.object(adapter_module, "yaml", yaml, create=True))
        self.enterContext(
            patch.object(
                Path,
                "open",
                unittest.mock.mock_open(read_data=yaml.safe_dump(document)),
            )
        )
        adapter = adapter_module.UcmDirectStoreAdapter.from_config(
            {
                "UCM_CONFIG_FILE": "config.yaml",
                "enable_metrics": False,
                "metrics_config_path": "new.yaml",
            },
            {},
        )
        self.assertEqual(
            adapter.launch_config,
            {"enable_metrics": False, "metrics_config_path": "new.yaml"},
        )
        self.assertFalse(adapter.metrics_enabled)
        self.assertNotIn("enable_metrics", adapter._store_config)

    def test_linker_logger_identity_config_and_disable(self):
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._dp_rank, obj._attn_dp_rank, obj._tp_rank = 0, 3, 1
        obj.adapter = types.SimpleNamespace(
            launch_config={"metrics_config_path": "file.yaml"}
        )
        params = types.SimpleNamespace(enable_metrics=True, pp_rank=2, attn_cp_rank=0)
        self.enterContext(
            patch.object(
                linker_module,
                "get_model",
                lambda: types.SimpleNamespace(model_path="model"),
                create=True,
            )
        )
        self.enterContext(
            patch.object(
                linker_module,
                "get_parallel",
                lambda: types.SimpleNamespace(enable_dp_attention=True),
                create=True,
            )
        )
        factory = self.enterContext(
            patch.object(
                linker_module,
                "create_metrics_logger",
                return_value=types.SimpleNamespace(config={"counter": []}),
            )
        )
        obj._init_metrics(params, {"metrics_config_path": "override.yaml"})
        self.assertTrue(obj.adapter.metrics_enabled)
        self.assertEqual(factory.call_args.kwargs["ranks"]["dp_rank"], 3)
        self.assertEqual(
            factory.call_args.kwargs["launch_config"]["metrics_config_path"],
            "override.yaml",
        )
        factory.return_value = None
        params.enable_metrics = False
        obj._init_metrics(params, {})
        self.assertFalse(obj.adapter.metrics_enabled)

    def test_shutdown_stops_poller_before_final_export(self):
        events = []
        dispatcher = types.SimpleNamespace(
            drain_to_consumers=lambda: events.append("drain"),
            get_stats_and_clear=lambda name: ({"last": 1}, {}, {}),
        )
        logger = types.SimpleNamespace(
            shutdown=lambda: events.append("stop"),
            metrics_dispatcher=dispatcher,
            update_stats=lambda *stats: events.append(stats[0]),
        )
        metrics_module.shutdown_metrics_logger(logger)
        self.assertEqual(events, ["stop", "drain", {"last": 1}])

    def test_registration_precedes_store_setup_and_failure_closes_logger(self):
        events = []
        group = types.SimpleNamespace(entry_map={}, num_layers=0, rank_replicated=False)
        parallel = types.SimpleNamespace(tp_size=1, dp_rank=0, attn_dp_rank=0)
        params = types.SimpleNamespace(
            page_size=4,
            token_to_kv_pool_allocator=types.SimpleNamespace(
                get_kvcache=lambda: object()
            ),
            attn_tp_cache_group=None,
            tp_cache_group=None,
            attn_cp_size=1,
            attn_cp_rank=0,
            pp_rank=0,
            enable_metrics=True,
        )
        logger = types.SimpleNamespace(config={"counter": []})

        def create(**kwargs):
            events.append("logger")
            return logger

        def register(pools):
            events.append("store")
            raise RuntimeError("Store setup failed")

        adapter = types.SimpleNamespace(
            launch_config={},
            register_device_pools=register,
            close=lambda: events.append("store.close"),
        )
        for name, value in {
            "get_parallel": lambda: parallel,
            "get_model": lambda: types.SimpleNamespace(model_path="model"),
            "resolve_hybrid_device_pool_group": lambda **kwargs: group,
            "torch": types.SimpleNamespace(
                distributed=types.SimpleNamespace(is_available=lambda: False)
            ),
            "Queue": queue.Queue,
            "OrderedDict": collections.OrderedDict,
            "create_metrics_logger": create,
            "shutdown_metrics_logger": lambda instance: events.append("logger.close"),
        }.items():
            self.enterContext(patch.object(linker_module, name, value, create=True))
        cls = linker_module.UcmDirectLinker
        self.enterContext(patch.object(cls, "_build_pool_bundles", return_value=()))
        self.enterContext(
            patch.object(cls, "_build_layer_pointer_layouts", return_value={})
        )
        self.enterContext(patch.object(cls, "_log_pool_topology"))
        with self.assertLogs(linker_module.logger, "ERROR"):
            with self.assertRaisesRegex(RuntimeError, "Store setup failed"):
                cls(None, params, components=[], adapter=adapter)
        self.assertEqual(events, ["logger", "store", "store.close", "logger.close"])

    def test_non_owner_offload_and_pending_cancel_never_create_io_samples(self):
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._metrics_enabled = True
        obj._partition_offload = lambda transfers: []
        obj._offload_queue = queue.Queue()
        obj._offload_results = queue.Queue()
        obj._expand = lambda transfers, **kwargs: transfers
        obj._freeze_gc_once = lambda: None
        obj._pending_loads, obj._lookup_progress = {}, {}
        self.assertTrue(obj.offload([object()]))
        self.assertEqual(obj._offload_queue.get_nowait(), ([], [], None))
        self.assertEqual(obj.num_completed_offloads(), 0)
        self.assertTrue(obj.load("request", [object()]))
        self.assertTrue(obj.cancel_queued_load("request"))
        self.assertFalse(obj.cancel_queued_load("request"))
        self.assertEqual(self.total("sglang_linker_offload_skipped_total"), 1)
        self.assertEqual(self.total("sglang_linker_load_cancelled_requests_total"), 1)
        self.assertFalse(
            any(
                key.startswith("sglang_connector_")
                for sample in self.samples
                for key in sample
            )
        )


class DashboardTest(unittest.TestCase):
    def test_dashboard_metric_names_and_panel_ids(self):
        root = Path(__file__).resolve().parents[1] / "examples/metrics"
        config = yaml.safe_load(
            (root / "metrics_configs_sglang.yaml").read_text(encoding="utf-8")
        )
        names = {
            item["name"]
            for kind in ("counter", "gauge", "histogram")
            for item in config[kind]
        }
        for item in config["histogram"]:
            names.update(
                item["name"] + suffix for suffix in ("_sum", "_count", "_bucket")
            )
        board = json.loads(
            (root / "sglang-linker-dashboard.json").read_text(encoding="utf-8")
        )
        ids = [panel["id"] for panel in board["panels"]]
        self.assertEqual(len(ids), len(set(ids)))
        import re

        queried = set()
        for panel in board["panels"]:
            for target in panel.get("targets", []):
                expr = target["expr"]
                self.assertNotIn("or vector(0)", expr)
                self.assertIn('worker_id=~"$worker_id"', expr)
                for name in re.findall(r"\$\{ucm_prefix\}(sglang_\w+)\{", expr):
                    self.assertIn(name, names)
                    queried.add(name)
        for item in config["counter"]:
            if item["name"].startswith("sglang_linker_"):
                self.assertIn(item["name"], queried)


if __name__ == "__main__":
    unittest.main()
