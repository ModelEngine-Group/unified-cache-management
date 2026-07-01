from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit import registry
from ucm_toolkit.cli import main


class MetricsViewToolkitTest(unittest.TestCase):
    def setUp(self):
        registry._TOOLS.clear()
        registry._ALIASES.clear()

    def test_metrics_view_is_registered_top_level_tool(self):
        registry.init_builtin_tools()

        tool = registry.get("metrics-view")

        self.assertEqual(tool.name, "metrics-view")
        self.assertIn("metrics_view", tool.aliases)
        self.assertFalse(tool.buildable)

    def test_cli_can_run_metrics_view_list_configs(self):
        output = io.StringIO()

        with redirect_stdout(output):
            result = main(["run", "metrics-view", "list-configs"])

        self.assertEqual(result, 0)
        self.assertIn("metrics_lite", output.getvalue())

    def test_doctor_does_not_report_metrics_view_environment_checks(self):
        output = io.StringIO()

        with redirect_stdout(output):
            result = main(["doctor", "metrics-view"])

        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("metrics-view: no environment checks", text)
        self.assertNotIn("sqlite3", text)
        self.assertNotIn("PyYAML", text)

    def test_metrics_view_uses_toolkit_readme_only(self):
        standalone_readme = (
            ROOT / "ucm_toolkit" / "tools" / "metrics_view" / "README.md"
        )

        self.assertFalse(standalone_readme.exists())


if __name__ == "__main__":
    unittest.main()
