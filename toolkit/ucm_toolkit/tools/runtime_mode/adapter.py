"""runtime-mode tool adapter."""

from __future__ import annotations

import argparse

from ...registry import ToolAdapter


class RuntimeModeTool(ToolAdapter):
    """Adapter for online UCM runtime-mode control."""

    name = "runtime-mode"
    aliases = ("runtime_mode",)
    description = (
        "Switch the online runtime mode of UCM-enabled engines "
        "(enabled | lite | disabled) without restarting them, via their "
        "runtime-mode control files."
    )
    buildable = False

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "args", nargs="*", help="Arguments forwarded to runtime-mode"
        )

    def run(self, tool_args: list[str]) -> int:
        from .cli import main

        try:
            return int(main(tool_args))
        except SystemExit as exc:
            if isinstance(exc.code, int):
                return exc.code
            return 0 if exc.code is None else 1

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        import tempfile

        print(f"{self.name}: no environment checks")
        print(f"  default control directory: {tempfile.gettempdir()}")
        return 0
