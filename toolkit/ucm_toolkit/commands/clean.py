"""Clean command interface."""

from __future__ import annotations

import argparse

from .. import registry


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the clean command parser."""
    parser = subparsers.add_parser("clean", help="Clean a toolkit tool")
    parser.add_argument("tool", help="Top-level tool name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be cleaned"
    )
    parser.set_defaults(handler=handle)
    return parser


def handle(args: argparse.Namespace) -> int:
    """Execute the clean command."""
    tool = registry.get(args.tool)
    return tool.clean(args)
