"""Command-line entry point for ucm-toolkit."""

from __future__ import annotations

import argparse
import sys

from . import registry
from .commands import build as build_cmd
from .commands import clean as clean_cmd
from .commands import doctor as doctor_cmd
from .commands import list as list_cmd
from .commands import run as run_cmd
from .errors import ToolkitError


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="ucm-toolkit")
    subparsers = parser.add_subparsers(dest="command")
    list_cmd.add_parser(subparsers)
    doctor_cmd.add_parser(subparsers)
    build_cmd.add_parser(subparsers)
    clean_cmd.add_parser(subparsers)
    subparsers.add_parser("run", help="Run a toolkit tool")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ucm-toolkit CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        registry.init_builtin_tools()
        if argv and argv[0] == "build":
            args = argparse.Namespace(command="build", tool_args=argv[1:])
            return build_cmd.handle(args)
        if argv and argv[0] == "run":
            return run_cmd.handle(argv[1:])

        parser = build_parser()
        args = parser.parse_args(argv)
        if not hasattr(args, "handler"):
            parser.print_help()
            return 0
        return args.handler(args)
    except ToolkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
