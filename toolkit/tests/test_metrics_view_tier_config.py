from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.tools.metrics_view.terminal_view_metrics.config import (
    load_config,
    metric_names_for_scrape,
)
from ucm_toolkit.tools.metrics_view.terminal_view_metrics.parser import (
    parse_prometheus_text,
)
from ucm_toolkit.tools.metrics_view.terminal_view_metrics.query import QueryEngine
from ucm_toolkit.tools.metrics_view.terminal_view_metrics.storage import MetricsStore


class MetricsViewTierConfigTest(unittest.TestCase):
    def test_tier_hit_rates_use_load_source_counters(self):
        config = load_config("metrics_lite")
        metrics = {metric["name"]: metric for metric in config["metrics"]}
        names = metric_names_for_scrape(config)

        self.assertNotIn("params", config)
        self.assertTrue(
            {
                "hbm_hit_rate",
                "yuanrong_dram_hit_rate",
                "yuanrong_ssd_hit_rate",
                "cache_hit_rate",
                "posix_hit_rate",
            }.issubset(metrics)
        )
        self.assertTrue(
            {
                "vllm:prefix_cache_hits_total",
                "vllm:prefix_cache_queries_total",
                "ucm:cache_load_shards_total",
                "ucm:cache_load_wait_shards_total",
                "ucm:yuanrong_load_success_shards_total",
                "ucm:yuanrong_lookup_miss_posix_load_success_shards_total",
                "ucm:yuanrong_load_fallback_posix_load_success_shards_total",
                "ucm:yuanrong_local_dram_load_hits_total",
                "ucm:yuanrong_remote_load_hits_total",
                "ucm:yuanrong_local_ssd_load_hits_total",
            }.issubset(names)
        )
        all_expressions = " ".join(
            metric.get("expr", "") for metric in config["metrics"]
        )
        self.assertNotIn("tp_size", all_expressions)
        self.assertNotIn("cache_load_backend_shards_total", all_expressions)
        self.assertNotIn("yuanrong_l2_hits_per_second", metrics)
        self.assertNotIn("ucm:yuanrong_l2_load_hits_total", names)

    def test_capacity_metrics_use_node_max_and_posix_average(self):
        config = load_config("metrics_lite")
        metrics = {metric["name"]: metric for metric in config["metrics"]}
        for name in (
            "yuanrong_dram_used_bytes",
            "yuanrong_dram_capacity_bytes",
            "yuanrong_dram_usage_ratio",
            "yuanrong_ssd_used_bytes",
            "yuanrong_ssd_capacity_bytes",
            "yuanrong_ssd_usage_ratio",
        ):
            self.assertEqual(metrics[name]["aggregate"], "max")
        for name in (
            "posix_store_used_bytes",
            "posix_store_capacity_bytes",
            "posix_store_usage_ratio",
        ):
            self.assertEqual(metrics[name]["aggregate"], "avg")

    def test_yuanrong_tier_formula_values(self):
        initial = """
vllm:prefix_cache_hits_total 10
vllm:prefix_cache_queries_total 20
vllm:external_prefix_cache_hits_total 5
vllm:external_prefix_cache_queries_total 10
ucm:yuanrong_load_success_shards_total 10
ucm:yuanrong_lookup_miss_posix_load_success_shards_total 5
ucm:yuanrong_load_fallback_posix_load_success_shards_total 5
ucm:yuanrong_local_dram_load_hits_total 10
ucm:yuanrong_remote_load_hits_total 5
ucm:yuanrong_local_ssd_load_hits_total 5
"""
        final = """
vllm:prefix_cache_hits_total 30
vllm:prefix_cache_queries_total 120
vllm:external_prefix_cache_hits_total 45
vllm:external_prefix_cache_queries_total 110
ucm:yuanrong_load_success_shards_total 70
ucm:yuanrong_lookup_miss_posix_load_success_shards_total 25
ucm:yuanrong_load_fallback_posix_load_success_shards_total 25
ucm:yuanrong_local_dram_load_hits_total 40
ucm:yuanrong_remote_load_hits_total 15
ucm:yuanrong_local_ssd_load_hits_total 25
"""

        values = self._query_values(initial, final)

        self.assertAlmostEqual(values["hbm_hit_rate"], 0.2)
        self.assertAlmostEqual(values["yuanrong_dram_hit_rate"], 0.128)
        self.assertAlmostEqual(values["yuanrong_ssd_hit_rate"], 0.064)
        self.assertAlmostEqual(values["posix_hit_rate"], 0.128)

    def test_cache_posix_formula_ignores_registered_zero_yuanrong_series(self):
        initial = """
vllm:prefix_cache_hits_total 10
vllm:prefix_cache_queries_total 20
vllm:external_prefix_cache_hits_total 5
vllm:external_prefix_cache_queries_total 10
ucm:cache_load_shards_total 15
ucm:cache_load_wait_shards_total 5
ucm:yuanrong_load_success_shards_total 0
ucm:yuanrong_lookup_miss_posix_load_success_shards_total 0
ucm:yuanrong_load_fallback_posix_load_success_shards_total 0
"""
        final = """
vllm:prefix_cache_hits_total 30
vllm:prefix_cache_queries_total 120
vllm:external_prefix_cache_hits_total 45
vllm:external_prefix_cache_queries_total 110
ucm:cache_load_shards_total 115
ucm:cache_load_wait_shards_total 30
ucm:yuanrong_load_success_shards_total 0
ucm:yuanrong_lookup_miss_posix_load_success_shards_total 0
ucm:yuanrong_load_fallback_posix_load_success_shards_total 0
"""

        values = self._query_values(initial, final)

        self.assertAlmostEqual(values["hbm_hit_rate"], 0.2)
        self.assertAlmostEqual(values["cache_hit_rate"], 0.24)
        self.assertAlmostEqual(values["posix_hit_rate"], 0.08)

    def _query_values(self, initial, final):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with MetricsStore(Path(temporary_dir) / "metrics.db") as store:
                store.write_samples(parse_prometheus_text(initial), 0)
                store.write_samples(parse_prometheus_text(final), 10_000)
                rows = QueryEngine(store).query_config(
                    load_config("metrics_lite"),
                    10,
                    start_ms=0,
                )
        return {
            row.metric: next(iter(row.values.values())) for row in rows if row.values
        }


if __name__ == "__main__":
    unittest.main()
