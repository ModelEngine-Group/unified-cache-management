from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
ROOT = TOOLKIT_ROOT / "ucm_toolkit" / "tools" / "metrics_view"
sys.path.insert(0, str(ROOT))

from terminal_view_metrics.cli import DEFAULT_DB, main
from terminal_view_metrics.config import (
    list_preset_configs,
    load_config,
    metric_names_for_scrape,
    metric_specs,
    parse_time_ms,
)
from terminal_view_metrics.parser import parse_prometheus_text
from terminal_view_metrics.query import QueryEngine
from terminal_view_metrics.render import render_json, render_table
from terminal_view_metrics.storage import MetricsStore


class TerminalViewMetricsTest(unittest.TestCase):
    def test_tool_lives_under_ucm_toolkit_tools_directory(self):
        self.assertEqual(ROOT, TOOLKIT_ROOT / "ucm_toolkit" / "tools" / "metrics_view")

    def test_parse_prometheus_text_handles_labels_and_special_values(self):
        text = """
# HELP ucm:load_duration Time to load
# TYPE ucm:load_duration histogram
ucm:load_duration_bucket{worker_id="0",model_name="qwen",le="0.5"} 3
ucm:load_duration_bucket{worker_id="0",model_name="qwen",le="+Inf"} 5
ucm:load_duration_sum{worker_id="0",model_name="qwen"} 2.5 1700000000000
ucm:cache_lookup_hit_rate{worker_id="0",note="escaped\\\"quote"} NaN
"""
        samples = parse_prometheus_text(text)

        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0].name, "ucm:load_duration_bucket")
        self.assertEqual(samples[0].labels["le"], "0.5")
        self.assertEqual(samples[0].value, 3.0)
        self.assertEqual(samples[2].timestamp_ms, 1700000000000)
        self.assertTrue(math.isnan(samples[3].value))
        self.assertEqual(samples[3].labels["note"], 'escaped"quote')

    def test_query_counter_rate_and_histogram_quantiles_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                t0 = 1_700_000_000_000
                t1 = t0 + 60_000

                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:load_bytes_total{worker_id="0"} 1000
ucm:load_duration_bucket{worker_id="0",le="0.1"} 10
ucm:load_duration_bucket{worker_id="0",le="0.5"} 20
ucm:load_duration_bucket{worker_id="0",le="1.0"} 30
ucm:load_duration_bucket{worker_id="0",le="+Inf"} 30
ucm:load_duration_sum{worker_id="0"} 12
ucm:load_duration_count{worker_id="0"} 30
"""
                    ),
                    t0,
                )
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:load_bytes_total{worker_id="0"} 2500
ucm:load_duration_bucket{worker_id="0",le="0.1"} 20
ucm:load_duration_bucket{worker_id="0",le="0.5"} 60
ucm:load_duration_bucket{worker_id="0",le="1.0"} 100
ucm:load_duration_bucket{worker_id="0",le="+Inf"} 100
ucm:load_duration_sum{worker_id="0"} 47
ucm:load_duration_count{worker_id="0"} 100
"""
                    ),
                    t1,
                )

                config = {
                    "metrics": [
                        {
                            "name": "ucm:load_bytes_total",
                            "type": "counter",
                            "op": "rate",
                            "aggregate": "sum",
                        },
                        {
                            "name": "ucm:load_duration",
                            "type": "histogram",
                            "quantiles": [0.5, 0.9, 0.99],
                            "avg": True,
                            "aggregate": "sum",
                        },
                    ]
                }

                rows = QueryEngine(store).query_config(config, 60, end_ms=t1)

                by_metric = {row.metric: row for row in rows}
                self.assertAlmostEqual(
                    by_metric["ucm:load_bytes_total"].values["rate"], 25.0
                )
                hist_values = by_metric["ucm:load_duration"].values
                self.assertEqual(list(hist_values), ["p50", "p90", "p99", "avg"])
                self.assertAlmostEqual(hist_values["avg"], 0.5)
                self.assertAlmostEqual(hist_values["p50"], 0.4333333333)
                self.assertAlmostEqual(hist_values["p90"], 0.8833333333)
                self.assertAlmostEqual(hist_values["p99"], 0.9883333333)

    def test_promql_expr_computes_grafana_style_hit_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                t0 = parse_time_ms("2026-06-25T10:00:00")
                t1 = t0 + 60_000
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:cache_lookup_hit_blocks_total{worker_id="0"} 100
ucm:cache_lookup_miss_blocks_total{worker_id="0"} 50
"""
                    ),
                    t0,
                )
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:cache_lookup_hit_blocks_total{worker_id="0"} 160
ucm:cache_lookup_miss_blocks_total{worker_id="0"} 70
"""
                    ),
                    t1,
                )

                config = {
                    "metrics": [
                        {
                            "name": "Cache Lookup Hit Rate",
                            "type": "promql",
                            "expr": 'sum by (${perWorker:raw}) (rate(ucm:cache_lookup_hit_blocks_total{model_name="$model_name", worker_id=~"$worker_id"}[$__rate_interval])) / (sum by (${perWorker:raw}) (rate(ucm:cache_lookup_hit_blocks_total{model_name="$model_name", worker_id=~"$worker_id"}[$__rate_interval])) + sum by (${perWorker:raw}) (rate(ucm:cache_lookup_miss_blocks_total{model_name="$model_name", worker_id=~"$worker_id"}[$__rate_interval])))',
                            "value": "hit_rate",
                            "aggregate": "avg",
                        }
                    ]
                }

                rows = QueryEngine(store).query_config(
                    config,
                    60,
                    start_ms=t0,
                    aggr_by_seconds=60,
                )

                self.assertEqual(rows[0].metric, "Cache Lookup Hit Rate")
                self.assertEqual(rows[0].group, {})
                self.assertAlmostEqual(rows[0].values["hit_rate"], 0.75)

    def test_promql_expr_computes_grafana_style_histogram_quantile(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                t0 = parse_time_ms("2026-06-25T10:00:00")
                t1 = t0 + 60_000
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:load_duration_bucket{worker_id="0",le="0.1"} 10
ucm:load_duration_bucket{worker_id="0",le="0.5"} 20
ucm:load_duration_bucket{worker_id="0",le="1.0"} 30
ucm:load_duration_bucket{worker_id="0",le="+Inf"} 30
"""
                    ),
                    t0,
                )
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:load_duration_bucket{worker_id="0",le="0.1"} 20
ucm:load_duration_bucket{worker_id="0",le="0.5"} 60
ucm:load_duration_bucket{worker_id="0",le="1.0"} 100
ucm:load_duration_bucket{worker_id="0",le="+Inf"} 100
"""
                    ),
                    t1,
                )

                config = {
                    "metrics": [
                        {
                            "name": "Load Duration p90",
                            "type": "promql",
                            "expr": 'histogram_quantile(0.9, sum by (le, ${perWorker:raw}) (rate(ucm:load_duration_bucket{model_name="$model_name", worker_id=~"$worker_id"}[$__rate_interval])))',
                            "value": "p90",
                            "unit": "ms",
                            "aggregate": "avg",
                        }
                    ]
                }

                rows = QueryEngine(store).query_config(
                    config,
                    60,
                    start_ms=t0,
                    aggr_by_seconds=60,
                )

                self.assertEqual(rows[0].metric, "Load Duration p90")
                self.assertAlmostEqual(rows[0].values["p90"], 0.8833333333)

    def test_config_loading_and_table_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "metrics.json"
            config_path.write_text(
                json.dumps(
                    {
                        "title": "Smoke",
                        "metrics": [
                            {
                                "name": "ucm:cache_lookup_hit_rate",
                                "type": "gauge",
                                "op": "last",
                                "aggregate": "avg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            self.assertEqual(config["title"], "Smoke")
            table = render_table(
                [
                    QueryEngine.Row(
                        metric="ucm:cache_lookup_hit_rate",
                        group={"worker_id": "0"},
                        values={"last": 0.75},
                        unit="",
                    )
                ]
            )

            self.assertIn("ucm:cache_lookup_hit_rate", table)
            self.assertNotIn("group", table.splitlines()[0])
            self.assertNotIn("worker_id=0", table)
            self.assertIn("0.750", table)

    def test_table_rendering_uses_short_bucket_time_and_hides_group(self):
        start_ms = parse_time_ms("2026-06-25T10:00:00")
        table = render_table(
            [
                QueryEngine.Row(
                    metric="ucm:load_duration",
                    group={"worker_id": "0"},
                    values={
                        "avg": 0.5,
                        "p50": 0.4333333333,
                        "p90": 0.8833333333,
                        "p99": 0.9883333333,
                    },
                    unit="ms",
                    start_ms=start_ms,
                    end_ms=start_ms + 60_000,
                )
            ]
        )

        self.assertNotIn("group", table.splitlines()[0])
        self.assertNotIn("worker_id=0", table)
        self.assertIn("06-25 10:00:00..06-25 10:01:00", table)
        self.assertNotIn("2026-", table)
        self.assertIn("p50=0.433 p90=0.883 p99=0.988 avg=0.500", table)

    def test_bucketed_table_groups_rows_by_metric_with_separator(self):
        start_ms = parse_time_ms("2026-06-25T10:00:00")
        table = render_table(
            [
                QueryEngine.Row(
                    "metric_a",
                    {},
                    {"rate": 1.0},
                    "",
                    start_ms,
                    start_ms + 60_000,
                ),
                QueryEngine.Row(
                    "metric_b",
                    {},
                    {"rate": 10.0},
                    "",
                    start_ms,
                    start_ms + 60_000,
                ),
                QueryEngine.Row(
                    "metric_a",
                    {},
                    {"rate": 2.0},
                    "",
                    start_ms + 60_000,
                    start_ms + 120_000,
                ),
                QueryEngine.Row(
                    "metric_b",
                    {},
                    {"rate": 20.0},
                    "",
                    start_ms + 60_000,
                    start_ms + 120_000,
                ),
            ]
        )

        metric_lines = [
            line
            for line in table.splitlines()
            if "metric_a" in line or "metric_b" in line or line.startswith("=")
        ]
        line_kinds = [
            (
                "separator"
                if line.startswith("=")
                else "metric_a" if "metric_a" in line else "metric_b"
            )
            for line in metric_lines
        ]
        self.assertEqual(
            line_kinds,
            ["metric_a", "metric_a", "separator", "metric_b", "metric_b"],
        )
        self.assertRegex(metric_lines[2], r"^={10,}$")

    def test_json_rendering_hides_group(self):
        output = render_json(
            [
                QueryEngine.Row(
                    metric="ucm:load_duration",
                    group={"worker_id": "0"},
                    values={"p50": 0.4, "p90": 0.8, "p99": 0.9, "avg": 0.5},
                    unit="ms",
                )
            ]
        )

        parsed = json.loads(output)
        self.assertNotIn("group", parsed[0])
        self.assertNotIn("worker_id", output)
        self.assertLess(output.index('"p50"'), output.index('"avg"'))

    def test_histogram_metric_normalization_uses_standard_values(self):
        specs = metric_specs(
            {
                "metrics": [
                    {
                        "name": "ucm:load_duration",
                        "type": "histogram",
                        "quantiles": [0.95],
                        "avg": False,
                        "aggregate": "avg",
                    }
                ]
            }
        )

        self.assertEqual(specs[0]["quantiles"], [0.5, 0.9, 0.99])
        self.assertTrue(specs[0]["avg"])

    def test_cli_query_outputs_table_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "metrics.db"
            config_path = tmp_path / "metrics.json"
            config_path.write_text(
                json.dumps(
                    {
                        "metrics": [
                            {
                                "name": "ucm:save_bytes_total",
                                "type": "counter",
                                "op": "rate",
                                "aggregate": "sum",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with MetricsStore(db_path) as store:
                store.write_samples(
                    parse_prometheus_text('ucm:save_bytes_total{worker_id="1"} 100'),
                    1_700_000_000_000,
                )
                store.write_samples(
                    parse_prometheus_text('ucm:save_bytes_total{worker_id="1"} 700'),
                    1_700_000_060_000,
                )

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "query",
                        "--db",
                        str(db_path),
                        "--config",
                        str(config_path),
                        "--window",
                        "60s",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertIn("ucm:save_bytes_total", output.getvalue())
            self.assertIn("rate=10.000", output.getvalue())

    def test_cli_check_scrapes_once_and_outputs_snapshot_aggregates(self):
        text = """
vllm:requests_total{engine="e0"} 12
vllm:requests_total{engine="e1"} 18
vllm:hits_total{engine="e0"} 6
vllm:hits_total{engine="e1"} 9
ucm:latency_bucket{engine="e0",le="0.1"} 2
ucm:latency_bucket{engine="e0",le="0.5"} 8
ucm:latency_bucket{engine="e0",le="1.0"} 10
ucm:latency_bucket{engine="e0",le="+Inf"} 10
ucm:latency_sum{engine="e0"} 4
ucm:latency_count{engine="e0"} 10
ucm:latency_bucket{engine="e1",le="0.1"} 1
ucm:latency_bucket{engine="e1",le="0.5"} 5
ucm:latency_bucket{engine="e1",le="1.0"} 10
ucm:latency_bucket{engine="e1",le="+Inf"} 10
ucm:latency_sum{engine="e1"} 6
ucm:latency_count{engine="e1"} 10
"""
        server = _MetricsServer(text)
        server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "metrics.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "metrics": [
                                {
                                    "name": "requests",
                                    "type": "promql",
                                    "expr": (
                                        "sum by () "
                                        "(increase(vllm:requests_total[$__rate_interval]))"
                                    ),
                                    "value": "requests",
                                    "aggregate": "sum",
                                },
                                {
                                    "name": "hit_rate",
                                    "type": "promql",
                                    "expr": (
                                        "sum by () "
                                        "(rate(vllm:hits_total[$__rate_interval])) / "
                                        "sum by () "
                                        "(rate(vllm:requests_total[$__rate_interval]))"
                                    ),
                                    "value": "hit_rate",
                                    "aggregate": "avg",
                                },
                                {
                                    "name": "latency",
                                    "source": "ucm:latency",
                                    "type": "histogram",
                                    "quantiles": [0.9],
                                    "avg": True,
                                    "aggregate": "avg",
                                },
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

                output = io.StringIO()
                with redirect_stdout(output):
                    result = main(
                        [
                            "check",
                            "--url",
                            server.url,
                            "--config",
                            str(config_path),
                            "--format",
                            "json",
                        ]
                    )

                self.assertEqual(result, 0)
                rows = {row["metric"]: row for row in json.loads(output.getvalue())}
                self.assertEqual(rows["requests"]["values"]["requests"], 30.0)
                self.assertAlmostEqual(rows["hit_rate"]["values"]["hit_rate"], 0.5)
                self.assertEqual(
                    list(rows["latency"]["values"]), ["p50", "p90", "p99", "avg"]
                )
                self.assertAlmostEqual(rows["latency"]["values"]["p50"], 0.38)
                self.assertAlmostEqual(rows["latency"]["values"]["p90"], 0.8571428571)
                self.assertAlmostEqual(rows["latency"]["values"]["p99"], 0.9857142857)
                self.assertAlmostEqual(rows["latency"]["values"]["avg"], 0.5)

                tagged_output = io.StringIO()
                with redirect_stdout(tagged_output):
                    result = main(
                        [
                            "check",
                            "--url",
                            server.url,
                            "--config",
                            str(config_path),
                            "--format",
                            "json",
                            "--tag",
                            "engine=e0",
                        ]
                    )

                self.assertEqual(result, 0)
                tagged_rows = {
                    row["metric"]: row for row in json.loads(tagged_output.getvalue())
                }
                self.assertEqual(tagged_rows["requests"]["values"]["requests"], 12.0)
                self.assertAlmostEqual(tagged_rows["latency"]["values"]["avg"], 0.4)
        finally:
            server.stop()

    def test_cli_query_defaults_to_metrics_lite_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "metrics.db"
            start_ms = parse_time_ms("2026-06-25T10:00:00")
            with MetricsStore(db_path) as store:
                store.write_samples(
                    parse_prometheus_text("vllm:time_to_first_token_seconds_count 5"),
                    start_ms,
                )
                store.write_samples(
                    parse_prometheus_text("vllm:time_to_first_token_seconds_count 9"),
                    start_ms + 10_000,
                )

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "query",
                        "--db",
                        str(db_path),
                        "--window",
                        "10s",
                        "--start-time",
                        "2026-06-25T10:00:00",
                        "--aggr-by",
                        "10s",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertIn("total_requests", output.getvalue())
            self.assertIn("requests=4.000", output.getvalue())

    def test_cli_query_filters_rows_by_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "metrics.db"
            config_path = tmp_path / "metrics.json"
            config_path.write_text(
                json.dumps(
                    {
                        "metrics": [
                            {
                                "name": "ucm:save_bytes_total",
                                "type": "counter",
                                "op": "rate",
                                "aggregate": "sum",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with MetricsStore(db_path) as store:
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:save_bytes_total{worker_id="0"} 100
ucm:save_bytes_total{worker_id="1"} 100
"""
                    ),
                    1_700_000_000_000,
                )
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:save_bytes_total{worker_id="0"} 700
ucm:save_bytes_total{worker_id="1"} 1300
"""
                    ),
                    1_700_000_060_000,
                )

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "query",
                        "--db",
                        str(db_path),
                        "--config",
                        str(config_path),
                        "--window",
                        "60s",
                        "--tag",
                        "worker_id=1",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertIn("rate=20.000", output.getvalue())
            self.assertNotIn("rate=10.000", output.getvalue())

    def test_cli_clean_clears_sqlite_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "metrics.db"
            with MetricsStore(db_path) as store:
                store.write_samples(
                    parse_prometheus_text('ucm:save_bytes_total{worker_id="1"} 100'),
                    1_700_000_000_000,
                )
                self.assertEqual(len(store.list_series("ucm:save_bytes_total")), 1)

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(["clean", "--db", str(db_path)])

            self.assertEqual(result, 0)
            self.assertIn(str(db_path), output.getvalue())
            with MetricsStore(db_path) as store:
                self.assertEqual(store.latest_ts_ms(), None)
                self.assertEqual(store.list_series("ucm:save_bytes_total"), [])

    def test_query_tag_filter_applies_before_promql_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                t0 = parse_time_ms("2026-06-25T10:00:00")
                t1 = t0 + 60_000
                store.write_samples(
                    parse_prometheus_text(
                        """
vllm:prompt_tokens_total{model_name="qwen"} 100
vllm:prompt_tokens_total{model_name="llama"} 100
"""
                    ),
                    t0,
                )
                store.write_samples(
                    parse_prometheus_text(
                        """
vllm:prompt_tokens_total{model_name="qwen"} 700
vllm:prompt_tokens_total{model_name="llama"} 1900
"""
                    ),
                    t1,
                )

                rows = QueryEngine(store).query_config(
                    {
                        "metrics": [
                            {
                                "name": "Prompt Tokens",
                                "type": "promql",
                                "expr": "sum by () (rate(vllm:prompt_tokens_total[$__rate_interval]))",
                                "value": "tokens_per_s",
                                "aggregate": "sum",
                            }
                        ]
                    },
                    60,
                    start_ms=t0,
                    tag_filters={"model_name": "qwen"},
                )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].values["tokens_per_s"], 10.0)

    def test_collect_without_config_stores_all_scraped_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metrics_path = tmp_path / "metrics.txt"
            db_path = tmp_path / "metrics.db"
            metrics_path.write_text(
                "\n".join(
                    [
                        'ucm:load_bytes_total{worker_id="0"} 100',
                        'external_metric_total{worker_id="0"} 200',
                    ]
                ),
                encoding="utf-8",
            )

            result = main(
                [
                    "collect",
                    "--once",
                    "--url",
                    metrics_path.as_uri(),
                    "--db",
                    str(db_path),
                ]
            )

            with MetricsStore(db_path) as store:
                self.assertEqual(result, 0)
                self.assertEqual(len(store.list_series("ucm:load_bytes_total")), 1)
                self.assertEqual(len(store.list_series("external_metric_total")), 1)

    def test_store_allows_different_metrics_with_same_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                store.write_samples(
                    parse_prometheus_text(
                        """
metric_a_total{worker_id="0"} 1
metric_b_total{worker_id="0"} 2
"""
                    ),
                    1_700_000_000_000,
                )

                self.assertEqual(len(store.list_series("metric_a_total")), 1)
                self.assertEqual(len(store.list_series("metric_b_total")), 1)

    def test_config_requires_explicit_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                with self.assertRaisesRegex(ValueError, "requires aggregate"):
                    QueryEngine(store).query_config(
                        {
                            "metrics": [
                                {
                                    "name": "ucm:load_bytes_total",
                                    "type": "counter",
                                    "op": "rate",
                                }
                            ]
                        },
                        60,
                    )

    def test_counter_aggregate_sum_and_avg_without_group_by(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                t0 = 1_700_000_000_000
                t1 = t0 + 60_000
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:load_bytes_total{worker_id="0"} 100
ucm:load_bytes_total{worker_id="1"} 100
"""
                    ),
                    t0,
                )
                store.write_samples(
                    parse_prometheus_text(
                        """
ucm:load_bytes_total{worker_id="0"} 700
ucm:load_bytes_total{worker_id="1"} 1300
"""
                    ),
                    t1,
                )

                sum_rows = QueryEngine(store).query_config(
                    {
                        "metrics": [
                            {
                                "name": "ucm:load_bytes_total",
                                "type": "counter",
                                "op": "rate",
                                "aggregate": "sum",
                            }
                        ]
                    },
                    60,
                    start_ms=t0,
                )
                avg_rows = QueryEngine(store).query_config(
                    {
                        "metrics": [
                            {
                                "name": "ucm:load_bytes_total",
                                "type": "counter",
                                "op": "rate",
                                "aggregate": "avg",
                            }
                        ]
                    },
                    60,
                    start_ms=t0,
                )

        self.assertEqual(sum_rows[0].group, {})
        self.assertAlmostEqual(sum_rows[0].values["rate"], 30.0)
        self.assertEqual(avg_rows[0].group, {})
        self.assertAlmostEqual(avg_rows[0].values["rate"], 15.0)

    def test_promql_aggregate_without_group_by(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                t0 = 1_700_000_000_000
                t1 = t0 + 60_000
                store.write_samples(
                    parse_prometheus_text(
                        """
vllm:num_requests_running{worker_id="0"} 2
vllm:num_requests_running{worker_id="1"} 4
"""
                    ),
                    t0,
                )
                store.write_samples(
                    parse_prometheus_text(
                        """
vllm:num_requests_running{worker_id="0"} 3
vllm:num_requests_running{worker_id="1"} 5
"""
                    ),
                    t1,
                )

                rows = QueryEngine(store).query_config(
                    {
                        "metrics": [
                            {
                                "name": "Running Requests",
                                "type": "promql",
                                "expr": "vllm:num_requests_running",
                                "value": "requests",
                                "aggregate": "sum",
                            }
                        ]
                    },
                    60,
                    start_ms=t0,
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].group, {})
        self.assertAlmostEqual(rows[0].values["requests"], 8.0)

    def test_scrape_filter_config_includes_metric_names_from_promql_expr(self):
        config = {
            "metrics": [
                {
                    "name": "Cache Lookup Hit Rate",
                    "type": "promql",
                    "expr": "sum by (worker_id) (rate(ucm:cache_lookup_hit_blocks_total[1m])) / sum by (worker_id) (rate(ucm:cache_lookup_miss_blocks_total[1m]))",
                    "aggregate": "avg",
                }
            ]
        }

        self.assertEqual(
            metric_names_for_scrape(config),
            {
                "ucm:cache_lookup_hit_blocks_total",
                "ucm:cache_lookup_miss_blocks_total",
            },
        )

    def test_metrics_lite_preset_tracks_remote_tool_datapoints(self):
        config = load_config("metrics_lite")
        metric_names = {metric["name"] for metric in config["metrics"]}
        specs = metric_specs(config)

        self.assertEqual(config["title"], "Metrics Lite")
        for metric in config["metrics"]:
            self.assertIn("aggregate", metric)
            self.assertNotIn("group_by", metric)
        removed = {
            "prompt_tokens_per_s",
            "generation_tokens_per_s",
            "total_generated_tokens",
            "decode_tps",
            "overall_tps",
            "interval_lookup_hit_rate",
            "posix_store_load_ratio",
            "cache_store_load_hit_fraction",
            "request_queue_time_s",
            "request_inference_time_s",
            "request_prefill_time_s",
            "request_decode_time_s",
        }
        self.assertFalse(removed & metric_names)
        self.assertEqual(config["metrics"][0]["name"], "total_requests")
        self.assertEqual(
            metric_names,
            {
                "total_requests",
                "e2e_request_latency_s",
                "ttft_s",
                "tpot_s",
                "layerwise_wait_blocking_ms",
                "cache_store_load_bandwidth_gbps",
                "cache_store_dump_bandwidth_gbps",
                "posix_store_load_bandwidth_gbps",
                "posix_store_dump_bandwidth_gbps",
                "prefix_cache_hit_rate",
                "external_prefix_cache_hit_rate",
                "posix_store_load_fraction",
            },
        )
        histograms = {
            metric["name"]: metric for metric in specs if metric["type"] == "histogram"
        }
        self.assertEqual(
            {name: metric["source"] for name, metric in histograms.items()},
            {
                "e2e_request_latency_s": "vllm:e2e_request_latency_seconds",
                "ttft_s": "vllm:time_to_first_token_seconds",
                "tpot_s": "vllm:request_time_per_output_token_seconds",
                "layerwise_wait_blocking_ms": "ucm:layerwise_wait_blocking_ms",
            },
        )
        for metric in histograms.values():
            self.assertEqual(metric["quantiles"], [0.5, 0.9, 0.99])
            self.assertTrue(metric["avg"])
        self.assertTrue(
            {
                "vllm:e2e_request_latency_seconds_bucket",
                "vllm:e2e_request_latency_seconds_sum",
                "vllm:e2e_request_latency_seconds_count",
                "vllm:time_to_first_token_seconds_bucket",
                "vllm:time_to_first_token_seconds_sum",
                "vllm:time_to_first_token_seconds_count",
                "vllm:request_time_per_output_token_seconds_bucket",
                "vllm:request_time_per_output_token_seconds_sum",
                "vllm:request_time_per_output_token_seconds_count",
                "ucm:layerwise_wait_blocking_ms_bucket",
                "ucm:layerwise_wait_blocking_ms_sum",
                "ucm:layerwise_wait_blocking_ms_count",
                "ucm:cache_load_bytes_total",
                "ucm:cache_dump_bytes_total",
                "ucm:posix_s2h_bytes_total",
                "ucm:posix_h2s_bytes_total",
                "vllm:prefix_cache_hits_total",
                "vllm:prefix_cache_queries_total",
                "vllm:external_prefix_cache_hits_total",
                "ucm:cache_load_backend_shards_total",
                "ucm:cache_load_shards_total",
            }.issubset(metric_names_for_scrape(config))
        )
        self.assertFalse(
            {
                "vllm:prompt_tokens_total",
                "vllm:generation_tokens_total",
                "vllm:request_generation_tokens_sum",
                "ucm:interval_lookup_hit_rates_sum",
                "ucm:interval_lookup_hit_rates_count",
                "e2e_request_latency_s_bucket",
                "vllm:request_queue_time_seconds_bucket",
                "vllm:request_queue_time_seconds_sum",
                "vllm:request_queue_time_seconds_count",
                "vllm:request_inference_time_seconds_bucket",
                "vllm:request_inference_time_seconds_sum",
                "vllm:request_inference_time_seconds_count",
                "vllm:request_prefill_time_seconds_bucket",
                "vllm:request_prefill_time_seconds_sum",
                "vllm:request_prefill_time_seconds_count",
                "vllm:request_decode_time_seconds_bucket",
                "vllm:request_decode_time_seconds_sum",
                "vllm:request_decode_time_seconds_count",
                "vllm:external_prefix_cache_queries_total",
            }
            & metric_names_for_scrape(config)
        )

    def test_preset_configs_use_explicit_aggregate_without_group_by(self):
        self.assertEqual(
            [path.name for path in list_preset_configs()], ["metrics_lite.json"]
        )
        for path in list_preset_configs():
            config = load_config(path)
            for metric in config.get("metrics", []):
                self.assertIn("aggregate", metric, path.name)
                self.assertIn(metric["aggregate"], {"sum", "avg"}, path.name)
                self.assertNotIn("group_by", metric, path.name)
                if metric.get("type") == "histogram":
                    self.assertTrue(metric.get("avg"), path.name)
                    self.assertEqual(metric.get("quantiles"), [0.5, 0.9, 0.99])

    def test_metrics_lite_preset_computes_remote_tool_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MetricsStore(Path(tmp) / "metrics.db") as store:
                t0 = parse_time_ms("2026-06-25T10:00:00")
                t1 = t0 + 10_000
                store.write_samples(
                    parse_prometheus_text(
                        """
vllm:e2e_request_latency_seconds_sum 2
vllm:e2e_request_latency_seconds_count 5
vllm:e2e_request_latency_seconds_bucket{le="0.1"} 1
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 3
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 5
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 5
vllm:time_to_first_token_seconds_sum 2
vllm:time_to_first_token_seconds_count 5
vllm:request_time_per_output_token_seconds_sum 1
vllm:request_time_per_output_token_seconds_count 5
vllm:prefix_cache_hits_total 40
vllm:prefix_cache_queries_total 50
vllm:external_prefix_cache_hits_total 10
vllm:external_prefix_cache_queries_total 25
ucm:layerwise_wait_blocking_ms_bucket{le="1"} 1
ucm:layerwise_wait_blocking_ms_bucket{le="5"} 3
ucm:layerwise_wait_blocking_ms_bucket{le="10"} 5
ucm:layerwise_wait_blocking_ms_bucket{le="+Inf"} 5
ucm:layerwise_wait_blocking_ms_sum 20
ucm:layerwise_wait_blocking_ms_count 5
ucm:cache_load_bytes_total 1000000000
ucm:cache_dump_bytes_total 2000000000
ucm:posix_s2h_bytes_total 3000000000
ucm:posix_h2s_bytes_total 4000000000
vllm:request_queue_time_seconds_sum 2.5
vllm:request_queue_time_seconds_count 5
vllm:request_inference_time_seconds_sum 5
vllm:request_inference_time_seconds_count 5
vllm:request_prefill_time_seconds_sum 1
vllm:request_prefill_time_seconds_count 5
vllm:request_decode_time_seconds_sum 4
vllm:request_decode_time_seconds_count 5
ucm:cache_load_backend_shards_total 6
ucm:cache_load_shards_total 10
"""
                    ),
                    t0,
                )
                store.write_samples(
                    parse_prometheus_text(
                        """
vllm:e2e_request_latency_seconds_sum 4
vllm:e2e_request_latency_seconds_count 9
vllm:e2e_request_latency_seconds_bucket{le="0.1"} 2
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 7
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 9
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 9
vllm:time_to_first_token_seconds_sum 3.2
vllm:time_to_first_token_seconds_count 9
vllm:request_time_per_output_token_seconds_sum 1.8
vllm:request_time_per_output_token_seconds_count 9
vllm:prefix_cache_hits_total 70
vllm:prefix_cache_queries_total 100
vllm:external_prefix_cache_hits_total 20
vllm:external_prefix_cache_queries_total 50
ucm:layerwise_wait_blocking_ms_bucket{le="1"} 2
ucm:layerwise_wait_blocking_ms_bucket{le="5"} 6
ucm:layerwise_wait_blocking_ms_bucket{le="10"} 9
ucm:layerwise_wait_blocking_ms_bucket{le="+Inf"} 9
ucm:layerwise_wait_blocking_ms_sum 40
ucm:layerwise_wait_blocking_ms_count 9
ucm:cache_load_bytes_total 6000000000
ucm:cache_dump_bytes_total 10000000000
ucm:posix_s2h_bytes_total 15000000000
ucm:posix_h2s_bytes_total 24000000000
vllm:request_queue_time_seconds_sum 4.5
vllm:request_queue_time_seconds_count 9
vllm:request_inference_time_seconds_sum 13
vllm:request_inference_time_seconds_count 9
vllm:request_prefill_time_seconds_sum 6
vllm:request_prefill_time_seconds_count 9
vllm:request_decode_time_seconds_sum 14
vllm:request_decode_time_seconds_count 9
ucm:cache_load_backend_shards_total 16
ucm:cache_load_shards_total 30
"""
                    ),
                    t1,
                )

                rows = QueryEngine(store).query_config(
                    load_config("metrics_lite"),
                    10,
                    start_ms=t0,
                    aggr_by_seconds=10,
                )
                values = {row.metric: row.values for row in rows}

        self.assertEqual(
            list(values["e2e_request_latency_s"]), ["p50", "p90", "p99", "avg"]
        )
        self.assertAlmostEqual(values["e2e_request_latency_s"]["p50"], 0.2333333333)
        self.assertAlmostEqual(values["e2e_request_latency_s"]["p90"], 0.4466666667)
        self.assertAlmostEqual(values["e2e_request_latency_s"]["p99"], 0.4946666667)
        self.assertAlmostEqual(values["e2e_request_latency_s"]["avg"], 0.5)
        self.assertEqual(
            list(values["layerwise_wait_blocking_ms"]), ["p50", "p90", "p99", "avg"]
        )
        self.assertAlmostEqual(values["layerwise_wait_blocking_ms"]["p50"], 3.0)
        self.assertAlmostEqual(values["layerwise_wait_blocking_ms"]["p90"], 8.0)
        self.assertAlmostEqual(values["layerwise_wait_blocking_ms"]["p99"], 9.8)
        self.assertAlmostEqual(values["layerwise_wait_blocking_ms"]["avg"], 5.0)
        self.assertAlmostEqual(values["cache_store_load_bandwidth_gbps"]["gbps"], 0.5)
        self.assertAlmostEqual(values["cache_store_dump_bandwidth_gbps"]["gbps"], 0.8)
        self.assertAlmostEqual(values["posix_store_load_bandwidth_gbps"]["gbps"], 1.2)
        self.assertAlmostEqual(values["posix_store_dump_bandwidth_gbps"]["gbps"], 2.0)
        self.assertAlmostEqual(values["prefix_cache_hit_rate"]["hit_rate"], 0.6)
        self.assertAlmostEqual(
            values["external_prefix_cache_hit_rate"]["hit_rate"], 0.2
        )
        self.assertAlmostEqual(values["posix_store_load_fraction"]["fraction"], 0.5)
        self.assertAlmostEqual(values["total_requests"]["requests"], 4.0)
        for removed in (
            "prompt_tokens_per_s",
            "generation_tokens_per_s",
            "total_generated_tokens",
            "decode_tps",
            "overall_tps",
            "interval_lookup_hit_rate",
            "posix_store_load_ratio",
            "cache_store_load_hit_fraction",
            "request_queue_time_s",
            "request_inference_time_s",
            "request_prefill_time_s",
            "request_decode_time_s",
        ):
            self.assertNotIn(removed, values)

    def test_default_db_uses_tmp_ucm_metrics_db(self):
        self.assertEqual(DEFAULT_DB, "/tmp/ucm_metrics.db")

    def test_cli_help_describes_start_time_as_local_iso_time(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                main(["query", "--help"])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("local ISO time", output.getvalue())

    def test_cli_help_does_not_offer_top_command(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        removed_command = "".join(["t", "op"])
        self.assertNotIn(f"query,{removed_command}", output.getvalue())
        self.assertNotIn("Refresh a terminal table", output.getvalue())

    def test_cli_query_can_anchor_window_by_start_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "metrics.db"
            config_path = tmp_path / "metrics.json"
            config_path.write_text(
                json.dumps(
                    {
                        "metrics": [
                            {
                                "name": "ucm:load_bytes_total",
                                "type": "counter",
                                "op": "rate",
                                "aggregate": "sum",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            start_ms = parse_time_ms("2026-06-25T10:00:00")
            with MetricsStore(db_path) as store:
                store.write_samples(
                    parse_prometheus_text('ucm:load_bytes_total{worker_id="0"} 100'),
                    start_ms,
                )
                store.write_samples(
                    parse_prometheus_text('ucm:load_bytes_total{worker_id="0"} 700'),
                    start_ms + 60_000,
                )
                store.write_samples(
                    parse_prometheus_text('ucm:load_bytes_total{worker_id="0"} 4300'),
                    start_ms + 120_000,
                )

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "query",
                        "--db",
                        str(db_path),
                        "--config",
                        str(config_path),
                        "--window",
                        "60s",
                        "--start-time",
                        "2026-06-25T10:01:00",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertIn("rate=60.000", output.getvalue())

    def test_cli_query_can_group_window_by_small_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "metrics.db"
            config_path = tmp_path / "metrics.json"
            config_path.write_text(
                json.dumps(
                    {
                        "metrics": [
                            {
                                "name": "ucm:load_bytes_total",
                                "type": "counter",
                                "op": "rate",
                                "aggregate": "sum",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            start_ms = parse_time_ms("2026-06-25T10:00:00")
            with MetricsStore(db_path) as store:
                for offset_ms, value in (
                    (0, 100),
                    (60_000, 700),
                    (120_000, 2500),
                ):
                    store.write_samples(
                        parse_prometheus_text(
                            f'ucm:load_bytes_total{{worker_id="0"}} {value}'
                        ),
                        start_ms + offset_ms,
                    )

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "query",
                        "--db",
                        str(db_path),
                        "--config",
                        str(config_path),
                        "--start-time",
                        "2026-06-25T10:00:00",
                        "--window",
                        "2m",
                        "--aggr-by",
                        "1m",
                    ]
                )

            table = output.getvalue()
            self.assertEqual(result, 0)
            self.assertIn("bucket", table)
            self.assertIn("rate=10.000", table)
            self.assertIn("rate=30.000", table)


class _MetricsServer:
    def __init__(self, text: str):
        self.text = text.encode("utf-8")
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(outer.text)))
                self.end_headers()
                self.wfile.write(outer.text)

            def log_message(self, _format, *_args):
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/metrics"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


if __name__ == "__main__":
    unittest.main()
