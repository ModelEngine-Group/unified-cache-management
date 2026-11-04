"""Doctor command interface."""

from __future__ import annotations

import argparse

from .. import registry


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the doctor command parser."""
    parser = subparsers.add_parser("doctor", help="Inspect toolkit tool setup")
    parser.add_argument("tool", nargs="?", help="Top-level tool name")
    parser.set_defaults(handler=handle)
    return parser


def handle(args: argparse.Namespace) -> int:
    """Execute the doctor command."""
    tools = [registry.get(args.tool)] if args.tool else registry.list_tools()
    exit_code = 0
    for tool in tools:
        code = tool.doctor(args)
        if code != 0:
            exit_code = code
    return exit_code
