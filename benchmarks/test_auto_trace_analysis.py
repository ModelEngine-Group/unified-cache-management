import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from benchmarks import auto_trace_analysis


class AutoTraceAnalysisTest(unittest.TestCase):
    def test_collect_log_facts_accepts_direct_file_with_any_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "trace.txt"
            log_file.write_text(
                "available kv cache memory: 1024 bytes\n"
                "tensor_parallel_size=2\n"
                "timestamp: 1.5, request_id: request-1, input_length: 16, "
                "output_length: 1, ucm_block_ids: ['block-1']\n",
                encoding="utf-8",
            )

            output = io.StringIO()
            with redirect_stdout(output):
                facts = auto_trace_analysis.collect_log_facts(log_file)

            self.assertEqual(facts.log_files, [str(log_file)])
            self.assertEqual(len(facts.records), 1)
            self.assertEqual(facts.records[0].hash_ids, ["block-1"])
            self.assertEqual(facts.parse_errors, [])
            self.assertIn("Log files to parse (1):", output.getvalue())
            self.assertIn(str(log_file), output.getvalue())

    def test_parse_error_is_reported_after_remaining_lines_are_parsed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "broken.log"
            bad_line = (
                "timestamp: 1, input_length: 16, output_length: 1, "
                "ucm_block_ids: [not-a-literal]"
            )
            log_file.write_text(
                "available kv cache memory: 1024 bytes\n"
                f"{bad_line}\n"
                "timestamp: 2, input_length: 16, output_length: 1, "
                "ucm_block_ids: ['valid-block']\n"
                "tensor_parallel_size=2\n",
                encoding="utf-8",
            )

            errors = io.StringIO()
            with redirect_stderr(errors):
                facts = auto_trace_analysis.collect_log_facts(log_file)

            self.assertEqual(len(facts.records), 1)
            self.assertEqual(len(facts.parse_errors), 1)
            message = errors.getvalue()
            self.assertIn("Log line parse failures: 1", message)
            self.assertIn("Parse failure 1:", message)
            self.assertIn("stage: trace.ucm_block_ids.literal_eval", message)
            self.assertIn(f"source: {log_file}:2", message)
            self.assertIn("column:", message)
            self.assertIn(f"line excerpt: {bad_line!r}", message)

    def test_incomplete_trace_reports_pattern_match_stage(self):
        line = (
            "timestamp: 1, input_length: 16, output_length: 1, "
            "ucm_block_ids: ['missing-list-terminator'"
        )

        with self.assertRaises(ValueError) as raised:
            auto_trace_analysis.parse_trace_line(line, "server.log", 42)

        message = str(raised.exception)
        self.assertIn("stage: trace.pattern_match", message)
        self.assertIn("source: server.log:42", message)
        self.assertIn(f"line excerpt: {line!r}", message)

    def test_main_succeeds_when_valid_trace_follows_parse_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "mixed.log"
            log_file.write_text(
                "available kv cache memory: 1024 bytes\n"
                "tensor_parallel_size=1\n"
                "timestamp: 1, input_length: 16, output_length: 1, "
                "ucm_block_ids: [invalid]\n"
                "timestamp: 2, input_length: 16, output_length: 1, "
                "ucm_block_ids: ['valid-block']\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            errors = io.StringIO()

            with redirect_stdout(output), redirect_stderr(errors):
                exit_code = auto_trace_analysis.main(
                    [
                        "--log-dir",
                        str(log_file),
                        "--block-kv-cache-size",
                        "1024",
                        "--is-mla",
                        "false",
                        "--dram-pool-size-gb",
                        "0",
                        "--fs-pool-size-gb",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Trace cache hit rate analysis", output.getvalue())
            self.assertIn("Log line parse failures: 1", errors.getvalue())

    def test_directory_scan_keeps_log_file_patterns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            expected = [log_dir / "a.log", log_dir / "b.log.1"]
            for path in [*expected, log_dir / "ignored.txt"]:
                path.touch()

            self.assertEqual(auto_trace_analysis.iter_log_files(log_dir), expected)


if __name__ == "__main__":
    unittest.main()
