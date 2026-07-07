"""Build command interface."""

from __future__ import annotations

import argparse

from .. import registry
from ..errors import ToolkitError, ToolNotBuildableError


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the build command parser."""
    parser = subparsers.add_parser("build", help="Build a toolkit tool")
    parser.set_defaults(handler=handle)
    return parser


def handle(args: argparse.Namespace) -> int:
    """Execute the build command."""
    raw_args = getattr(args, "tool_args", None)
    if raw_args is None:
        raw_args = []
    if not raw_args:
        raise ToolkitError(
            "missing build tool\nusage: ucm-toolkit build TOOL [tool build args...]"
        )

    tool_name = raw_args[0]
    tool = registry.get(tool_name)
    if not tool.buildable:
        raise ToolNotBuildableError(tool.name)

    parser = argparse.ArgumentParser(prog=f"ucm-toolkit build {tool_name}")
    tool.add_build_args(parser)
    tool_args = parser.parse_args(raw_args[1:])
    tool_args.tool = tool_name
    return tool.build(tool_args)
