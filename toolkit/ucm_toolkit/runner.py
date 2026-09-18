"""Subprocess execution interfaces for ucm-toolkit."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

from .errors import CommandFailedError


def command_exists(command: str) -> bool:
    """Return whether an external command is available."""
    return shutil.which(command) is not None


def format_command(cmd: list[str]) -> str:
    """Format a command for display."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a command and return its exit code."""
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=False,
    )
    return completed.returncode


def check_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command and raise on failure."""
    returncode = run_command(cmd, cwd=cwd, env=env)
    if returncode != 0:
        raise CommandFailedError(format_command(cmd), returncode)
