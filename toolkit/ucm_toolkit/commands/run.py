"""Run command interface."""

from __future__ import annotations

from .. import registry
from ..errors import ToolkitError


class RunUsageError(ToolkitError):
    """Raised when run command usage is invalid."""


def handle(argv: list[str]) -> int:
    """Execute the run command with raw tool arguments."""
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: ucm-toolkit run TOOL [tool args...]")
        print()
        print("Run a top-level toolkit tool. Tool-specific arguments are forwarded.")
        print("Use `ucm-toolkit list` to show available top-level tools.")
        return 0
    tool_name = argv[0]
    tool_args = argv[1:]
    tool = registry.get(tool_name)
    return tool.run(tool_args)
