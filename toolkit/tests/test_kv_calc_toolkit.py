from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit import registry  # noqa: E402
from ucm_toolkit.cli import main  # noqa: E402


class KvCalcToolkitTest(unittest.TestCase):
    """Toolkit-level tests for kv-calc (registration, dispatch, doctor)."""

    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_kv_calc_is_registered_top_level_tool(self):
        registry.init_builtin_tools()

        tool = registry.get("kv-calc")

        self.assertEqual(tool.name, "kv-calc")
        self.assertIn("kv_calc", tool.aliases)
        self.assertFalse(tool.buildable)

    def test_cli_list_shows_kv_calc(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            main(["list"])

        self.assertIn("kv-calc", output.getvalue())

    def test_cli_can_run_kv_calc(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(
                [
                    "run",
                    "kv-calc",
                    "--model",
                    "qwen3-32b",
                    "--input-len",
                    "4096",
                    "--num-requests",
                    "1",
                    "--tp",
                    "1",
                ]
            )

        self.assertEqual(result, 0)
        # Parity with docs/source/_static/calculator.js: 1.0 GiB per seq.
        self.assertIn("1.0000 GiB", output.getvalue())
        self.assertIn("tokens", output.getvalue())

    def test_doctor_reports_no_environment_checks(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["doctor", "kv-calc"])

        self.assertEqual(result, 0)
        self.assertIn("kv-calc: no environment checks", output.getvalue())

    def test_invalid_tp_returns_nonzero(self):
        # RFC acceptance: clear error for tp=0.
        registry.init_builtin_tools()

        err = io.StringIO()
        with redirect_stderr(err):
            result = main(["run", "kv-calc", "--model", "qwen3-32b", "--tp", "0"])

        self.assertNotEqual(result, 0)
        self.assertIn("tp", err.getvalue().lower())

    def test_invalid_num_requests_returns_nonzero(self):
        # RFC acceptance: clear error for num_requests=0.
        registry.init_builtin_tools()

        err = io.StringIO()
        with redirect_stderr(err):
            result = main(
                ["run", "kv-calc", "--model", "qwen3-32b", "--num-requests", "0"]
            )

        self.assertNotEqual(result, 0)
        self.assertIn("num-requests", err.getvalue().lower())

    def test_kv_calc_readme_documents_flags(self):
        # Convention (matches develop's per-tool README pattern): each tool
        # ships a README.md documenting its flags, kept in sync with --help.
        standalone_readme = (
            ROOT / "ucm_toolkit" / "tools" / "kv_calc" / "README.md"
        )

        self.assertTrue(standalone_readme.exists())
        text = standalone_readme.read_text(encoding="utf-8")
        for flag in ("--model", "--input-len", "--num-requests", "--tp", "--dp"):
            self.assertIn(flag, text)


if __name__ == "__main__":
    unittest.main()
