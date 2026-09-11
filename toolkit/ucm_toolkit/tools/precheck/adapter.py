"""precheck tool adapter."""

from __future__ import annotations

import argparse

from ...registry import ToolAdapter


class PrecheckTool(ToolAdapter):
    """Adapter for the UCM environment pre-check (RFC #1208)."""

    name = "precheck"
    aliases = ("pre_check",)
    description = (
        "Run UCM environment pre-checks before deploying UCM: serving-stack "
        "and uc-manager versions, accelerator driver (CUDA compute capability "
        "or Ascend HDK), kernel version, memory & /dev/shm size, and the "
        "posix-store bandwidth benchmark. Reports PASS/WARN/FAIL per item "
        "with remediation advice for failures."
    )
    buildable = False

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        """Register precheck run arguments (documentation-only; run() parses)."""
        parser.add_argument("args", nargs="*", help="Arguments forwarded to precheck")

    def run(self, tool_args: list[str]) -> int:
        """Run the pre-check with raw tool arguments."""
        from .cli import main

        try:
            return int(main(tool_args))
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return exc.code
            return 0 if exc.code is None else 1

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """precheck is itself an environment check; no separate doctor step."""
        print(f"{self.name}: run `ucm-toolkit run precheck` to check the env")
        return 0
