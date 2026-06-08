"""POSIX AIO adapter interface."""

from __future__ import annotations

import argparse
import os
import sys

from ... import registry
from ...errors import ScriptNotFoundError
from ...registry import ToolAdapter
from ...runner import run_command


class PosixAioTool(ToolAdapter):
    """Adapter for ucm/store/test/e2e/posixstore_aio_test.py."""

    name = "posix-aio"
    aliases = ("posix_aio",)
    description = "Run the POSIX AIO store test script."
    buildable = False
    script_path = "ucm/store/test/e2e/posixstore_aio_test.py"

    def add_run_args(self, parser: argparse.ArgumentParser) -> None:
        """Register POSIX AIO run arguments."""
        parser.add_argument("--worker-number", type=int)
        parser.add_argument("--shard-size", type=int)
        parser.add_argument("--shard-number", type=int)
        parser.add_argument("--block-number", type=int)
        parser.add_argument("--dump-epoch-number", type=int)
        parser.add_argument("--load-epoch-number", type=int)
        parser.add_argument("--storage-backend", action="append")

    def run(self, tool_args: list[str]) -> int:
        """Run the POSIX AIO test script."""
        script = registry.resolve_repo_path(self.script_path or "")
        if not script.exists():
            raise ScriptNotFoundError(str(script))
        env = os.environ.copy()
        repo_root = str(registry.repo_root())
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = repo_root if not pythonpath else os.pathsep.join([repo_root, pythonpath])
        return run_command([sys.executable, str(script), *tool_args], env=env)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Inspect POSIX AIO script availability."""
        script = registry.resolve_repo_path(self.script_path or "")
        status = "OK" if script.exists() else "MISSING"
        print(f"{self.name}: {script} {status}")
        return 0 if script.exists() else 1
