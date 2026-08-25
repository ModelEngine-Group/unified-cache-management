"""Toolkit-level tests for precheck (registration, dispatch, doctor)."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit import registry  # noqa: E402
from ucm_toolkit.cli import main  # noqa: E402


class PrecheckToolkitTest(unittest.TestCase):
    """Toolkit integration: precheck is registered, listed, and runnable."""

    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_precheck_is_registered_top_level_tool(self):
        registry.init_builtin_tools()

        tool = registry.get("precheck")

        self.assertEqual(tool.name, "precheck")
        self.assertIn("pre_check", tool.aliases)
        self.assertFalse(tool.buildable)

    def test_cli_list_shows_precheck(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            main(["list"])

        self.assertIn("precheck", output.getvalue())

    def test_cli_can_dispatch_precheck_help(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["run", "precheck", "--help"])

        self.assertEqual(result, 0)
        self.assertIn("precheck", output.getvalue())

    def test_doctor_advertises_precheck(self):
        registry.init_builtin_tools()

        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["doctor", "precheck"])

        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("precheck", text)

    def test_precheck_has_readme(self):
        readme = ROOT / "ucm_toolkit" / "tools" / "precheck" / "README.md"
        self.assertTrue(readme.exists())


if __name__ == "__main__":
    unittest.main()
