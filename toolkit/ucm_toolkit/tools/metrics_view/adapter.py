"""metrics-view adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from ...registry import ToolAdapter


class MetricsViewTool(ToolAdapter):
    """Adapter for terminal metrics collection and query."""

    name = "metrics-view"
    aliases = ("metrics_view", "terminal-metrics", "terminal_metrics")
    description = "Collect and query Prometheus metrics from a terminal."
    buildable = False

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "args", nargs="*", help="Arguments forwarded to metrics-view"
        )

    def run(self, tool_args: list[str]) -> int:
        metrics_main = _load_metrics_main()
        try:
            return int(metrics_main(tool_args))
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return exc.code
            return 0 if exc.code is None else 1

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        print(f"{self.name}: no environment checks")
        return 0


def _load_metrics_main() -> Callable[[list[str] | None], int]:
    tool_dir = Path(__file__).resolve().parent
    tool_dir_text = str(tool_dir)
    if tool_dir_text not in sys.path:
        sys.path.insert(0, tool_dir_text)
    from terminal_view_metrics.cli import main

    return main
