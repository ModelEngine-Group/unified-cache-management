"""POSIX AIO adapter interface."""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import sys
from pathlib import Path

from ... import registry
from ...errors import ScriptNotFoundError
from ...registry import ToolAdapter
from ...runner import run_command


IMPORT_MODE_ENV = "UCM_TOOLKIT_POSIX_AIO_IMPORT"


def _path_resolves_to_repo_root(path: str, repo_root: Path) -> bool:
    """Return whether a sys.path entry points at the repository root."""
    try:
        candidate = Path(path or os.getcwd()).resolve()
    except OSError:
        return False
    return candidate == repo_root


def _ucm_is_available_without_repo_root(repo_root: Path) -> bool:
    """Return whether ucm can be imported without the source tree root."""
    search_path = [
        path
        for path in sys.path
        if not _path_resolves_to_repo_root(path, repo_root)
    ]
    return importlib.machinery.PathFinder.find_spec("ucm", search_path) is not None


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    """Prepend a path to PYTHONPATH in a child process environment."""
    value = str(path)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = value if not pythonpath else os.pathsep.join([value, pythonpath])


def _drop_repo_root_from_pythonpath(env: dict[str, str], repo_root: Path) -> None:
    """Remove repository-root entries from child PYTHONPATH."""
    pythonpath = env.get("PYTHONPATH")
    if not pythonpath:
        return

    entries = [
        entry
        for entry in pythonpath.split(os.pathsep)
        if not _path_resolves_to_repo_root(entry, repo_root)
    ]
    if entries:
        env["PYTHONPATH"] = os.pathsep.join(entries)
    else:
        env.pop("PYTHONPATH", None)


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
        repo_root = registry.repo_root()
        import_mode = env.get(IMPORT_MODE_ENV, "auto").strip().lower()
        if import_mode == "source":
            _prepend_pythonpath(env, repo_root)
        elif import_mode == "installed" or _ucm_is_available_without_repo_root(repo_root):
            _drop_repo_root_from_pythonpath(env, repo_root)
        else:
            _prepend_pythonpath(env, repo_root)
        return run_command([sys.executable, str(script), *tool_args], env=env)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Inspect POSIX AIO script availability."""
        script = registry.resolve_repo_path(self.script_path or "")
        status = "OK" if script.exists() else "MISSING"
        print(f"{self.name}: {script} {status}")
        return 0 if script.exists() else 1
