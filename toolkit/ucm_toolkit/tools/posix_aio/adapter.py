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
        path for path in sys.path if not _path_resolves_to_repo_root(path, repo_root)
    ]
    return importlib.machinery.PathFinder.find_spec("ucm", search_path) is not None


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    """Prepend a path to PYTHONPATH in a child process environment."""
    value = str(path)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        value if not pythonpath else os.pathsep.join([value, pythonpath])
    )


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
        parser.add_argument(
            "-w",
            "--worker-number",
            type=int,
            help="worker number: number of worker processes to start concurrently.",
        )
        parser.add_argument(
            "-s",
            "--shard-size",
            type=int,
            help=(
                "shard size: POSIX store I/O size. In layerwise mode, this is "
                "the K/V tensor size for one layer of one block. In non-layerwise "
                "mode, this is the K/V tensor size for all layers of one block."
            ),
        )
        parser.add_argument(
            "-n",
            "--shard-number",
            type=int,
            help="shard number: number of layers in layerwise mode; use 1 in non-layerwise mode.",
        )
        parser.add_argument(
            "-b",
            "--block-number",
            type=int,
            help="block number: total number of blocks.",
        )
        parser.add_argument(
            "-d",
            "--dump-epoch-number",
            type=int,
            help="dump epoch number: number of dump epochs.",
        )
        parser.add_argument(
            "-l",
            "--load-epoch-number",
            type=int,
            help="load epoch number: number of load epochs.",
        )
        parser.add_argument(
            "-o",
            "--storage-backend",
            action="append",
            help="storage backend: storage backend path; may be repeated.",
        )

    def _build_run_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="ucm-toolkit run posix-aio",
            description="Run the POSIX AIO store benchmark.",
        )
        self.add_run_args(parser)
        return parser

    @staticmethod
    def _forward_args(args: argparse.Namespace) -> list[str]:
        forwarded: list[str] = []
        option_names = (
            "worker_number",
            "shard_size",
            "shard_number",
            "block_number",
            "dump_epoch_number",
            "load_epoch_number",
        )
        for option_name in option_names:
            value = getattr(args, option_name)
            if value is not None:
                forwarded.extend([f"--{option_name.replace('_', '-')}", str(value)])
        if args.storage_backend is not None:
            for path in args.storage_backend:
                forwarded.extend(["--storage-backend", path])
        return forwarded

    def run(self, tool_args: list[str]) -> int:
        """Run the POSIX AIO test script."""
        parser = self._build_run_parser()
        args = parser.parse_args(tool_args)
        forwarded_args = self._forward_args(args)
        script = registry.resolve_repo_path(self.script_path or "")
        if not script.exists():
            raise ScriptNotFoundError(str(script))
        env = os.environ.copy()
        repo_root = registry.repo_root()
        import_mode = env.get(IMPORT_MODE_ENV, "auto").strip().lower()
        if import_mode == "source":
            _prepend_pythonpath(env, repo_root)
        elif import_mode == "installed" or _ucm_is_available_without_repo_root(
            repo_root
        ):
            _drop_repo_root_from_pythonpath(env, repo_root)
        else:
            _prepend_pythonpath(env, repo_root)
        return run_command([sys.executable, str(script), *forwarded_args], env=env)

    def doctor(self, args: argparse.Namespace | None = None) -> int:
        """Inspect POSIX AIO script availability."""
        script = registry.resolve_repo_path(self.script_path or "")
        status = "OK" if script.exists() else "MISSING"
        print(f"{self.name}: {script} {status}")
        return 0 if script.exists() else 1
