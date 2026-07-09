import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks import auto_trace_cache_analysis as analyzer


def _write_log(path, text):
    path.write_text(text.strip() + "\n", encoding="utf-8")


class AutoTraceCacheAnalysisTest(unittest.TestCase):
    def test_service_url_adds_http_prefix_only_when_missing(self):
        self.assertEqual(
            analyzer.metrics_url_from_service_url("127.0.0.1:8000"),
            "http://127.0.0.1:8000/metrics",
        )
        self.assertEqual(
            analyzer.metrics_url_from_service_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/metrics",
        )
        self.assertEqual(
            analyzer.metrics_url_from_service_url("https://example.com/metrics"),
            "https://example.com/metrics",
        )

    def test_builds_required_analysis_from_logs_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            log_dir = base / "logs"
            log_dir.mkdir()
            _write_log(
                log_dir / "worker0.log",
                """
                [2026-05-27 08:30:14.964321][UC][I] available kv cache memory: 1073741824 bytes
                [2026-05-27 08:30:14.964325][UC][I] timestamp: 1.0, request_id: req0, input_length: 10, output_length: 1, ucm_block_ids: ['a', 'b'] [2,2][ucm_connector.py:1,get_num_new_matched_tokens]
                [2026-05-27 08:30:15.000000][UC][I] timestamp: 2.0, input_length: 10, output_length: 1, ucm_block_ids: ['a', 'c'] [1,1][ucm_connector.py:1,get_num_new_matched_tokens]
                [2026-05-27 08:30:16.000000][UC][I] timestamp: 3.0, input_length: 10, output_length: 1, ucm_block_ids: ['a', 'b'] [2,2][ucm_connector.py:1,get_num_new_matched_tokens]
                """,
            )
            metrics = base / "metrics.txt"
            metrics.write_text(
                "\n".join(
                    [
                        'ucm:gpu_hbm_hit_tokens_total{model_name="m"} 10',
                        'ucm:ucm_hit_tokens_total{model_name="m"} 5',
                        'ucm:total_prefix_query_tokens_total{model_name="m"} 30',
                    ]
                ),
                encoding="utf-8",
            )

            args = analyzer.build_arg_parser().parse_args(
                [
                    "--log-dir",
                    str(log_dir),
                    "--block-kv-cache-size",
                    str(1024**3),
                    "--is-mla",
                    "true",
                    "--dram-pool-size-gb",
                    "2",
                    "--fs-pool-size-gb",
                    "3",
                    "--service-url",
                    metrics.as_uri(),
                ]
            )
            result = analyzer.build_analysis(args)

        analysis = result["analysis"]
        self.assertEqual(analysis["total_request_count"], 3)
        self.assertEqual(analysis["total_request_token_count"], 30)
        self.assertEqual(analysis["average_request_token_count"], 10.0)
        self.assertEqual(analysis["theoretical_max_kv_cache_hit_rate_percent"], 50.0)
        self.assertEqual(analysis["service_actual_kv_cache_hit_rate_percent"], 50.0)
        self.assertAlmostEqual(
            analysis["hbm_dram_pool_theoretical_hit_rate_percent"],
            100 / 3,
        )
        self.assertEqual(
            analysis["hbm_dram_fs_pool_theoretical_hit_rate_percent"],
            50.0,
        )

    def test_cli_writes_required_output_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            log_dir = base / "logs"
            log_dir.mkdir()
            _write_log(
                log_dir / "worker.log",
                """
                [2026-05-27 08:30:14.964321][UC][I] available kv cache memory: 1073741824 bytes
                [2026-05-27 08:30:14.964325][UC][I] timestamp: 1.0, input_length: 10, output_length: 1, ucm_block_ids: ['a', 'b'] [2,2][ucm_connector.py:1,get_num_new_matched_tokens]
                """,
            )
            metrics = base / "metrics.txt"
            metrics.write_text(
                "\n".join(
                    [
                        "ucm:gpu_hbm_hit_tokens_total 1",
                        "ucm:ucm_hit_tokens_total 1",
                        "ucm:total_prefix_query_tokens_total 4",
                    ]
                ),
                encoding="utf-8",
            )
            output = base / "analysis.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "benchmarks" / "auto_trace_cache_analysis.py"),
                    "--log-dir",
                    str(log_dir),
                    "--block-kv-cache-size",
                    str(1024**3),
                    "--is-mla",
                    "true",
                    "--dram-pool-size-gb",
                    "1",
                    "--fs-pool-size-gb",
                    "1",
                    "--service-url",
                    metrics.as_uri(),
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))

        self.assertIn("total request count: 1", completed.stdout)
        self.assertIn("average tokens per request: 10.00", completed.stdout)
        self.assertEqual(data["analysis"]["total_request_count"], 1)
        self.assertEqual(
            data["analysis"]["service_actual_kv_cache_hit_rate_percent"],
            50.0,
        )

    def test_missing_trace_or_metrics_data_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            log_dir = base / "logs"
            log_dir.mkdir()
            _write_log(
                log_dir / "worker.log",
                """
                [2026-05-27 08:30:14.964321][UC][I] available kv cache memory: 1073741824 bytes
                """,
            )
            metrics = base / "metrics.txt"
            metrics.write_text(
                "ucm:total_prefix_query_tokens_total 1\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "benchmarks" / "auto_trace_cache_analysis.py"),
                    "--log-dir",
                    str(log_dir),
                    "--block-kv-cache-size",
                    str(1024**3),
                    "--is-mla",
                    "true",
                    "--dram-pool-size-gb",
                    "1",
                    "--fs-pool-size-gb",
                    "1",
                    "--service-url",
                    metrics.as_uri(),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("no trace records found", completed.stderr)


if __name__ == "__main__":
    unittest.main()
