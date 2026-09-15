"""kv-calc tool adapter."""

from __future__ import annotations

import argparse

from ...registry import ToolAdapter


class KvCalcTool(ToolAdapter):
    """Adapter for the KV cache capacity estimator (RFC #1217)."""

    name = "kv-calc"
    aliases = ("kv_calc",)
    description = "Estimate KV cache capacity for a model and DP/TP deployment."
    buildable = False

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        """Register kv-calc run arguments (documentation-only; run() parses)."""
        parser.add_argument("args", nargs="*", help="Arguments forwarded to kv-calc")

    def run(self, tool_args: list[str]) -> int:
        """Run the kv-calc estimator with raw tool arguments."""
        from .cli import main

        try:
            return int(main(tool_args))
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return exc.code
            return 0 if exc.code is None else 1

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """kv-calc is pure computation; no environment checks needed."""
        print(f"{self.name}: no environment checks")
        return 0
