"""List command interface."""

from __future__ import annotations

import argparse

from .. import registry


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the list command parser."""
    parser = subparsers.add_parser("list", help="List top-level toolkit tools")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show extra fields"
    )
    parser.set_defaults(handler=handle)
    return parser


def handle(args: argparse.Namespace) -> int:
    """Execute the list command."""
    for tool in registry.list_tools():
        kind = "buildable" if tool.buildable else "runnable"
        if args.verbose:
            aliases = ",".join(tool.aliases) if tool.aliases else "-"
            print(f"{tool.name:<14} {kind:<9} aliases={aliases:<16} {tool.description}")
        else:
            print(f"{tool.name:<14} {kind:<9} {tool.description}")
    return 0
