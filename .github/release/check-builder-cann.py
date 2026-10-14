#!/usr/bin/env python3
"""Inspect the active CANN installation and verify a source Builder version.

Inspection uses only the standard library so raw Builders need no extra tools.
Version arbitration and verification belong to the Runtime module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path


def inspect_cann(environ: Mapping[str, str]) -> dict[str, str]:
    """Read declared versions and known version files in the active Toolkit."""
    facts = {}
    if environ.get("CANN_VERSION"):
        facts["env:CANN_VERSION"] = environ["CANN_VERSION"]
    roots = set()
    for name in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        if environ.get(name):
            root = Path(environ[name]).resolve()
            roots.add(root)
            facts[f"env:{name}"] = str(root)
    if not roots:
        root = Path("/usr/local/Ascend/ascend-toolkit/latest").resolve()
        roots.add(root)
        facts["active-toolkit"] = str(root)
    for root in sorted(roots):
        paths = [
            root / "share/info/runtime/version.info",
            root / "version.info",
            root / "version.cfg",
            *sorted(root.glob("*-linux/ascend_toolkit_install.info")),
        ]
        for path in paths:
            if not path.is_file():
                continue
            match = re.search(
                r"^version\s*=\s*(.+?)\s*$",
                path.read_text(encoding="utf-8"),
                re.IGNORECASE | re.MULTILINE,
            )
            if match:
                facts[f"file:{path}"] = match.group(1).strip("\"'")
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect")
    version = commands.add_parser("version")
    version.add_argument("--facts", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--expected", required=True)
    verify.add_argument("--facts", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_cann(os.environ), sort_keys=True))
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ucm_release import runtime

    facts = (
        json.loads(args.facts.read_text(encoding="utf-8"))
        if args.facts is not None
        else inspect_cann(os.environ)
    )
    if args.command == "version":
        print(runtime.cann_installation_version(facts))
        return 0
    version = runtime.verify_cann_installation(facts, args.expected)
    print(f"Verified source Builder CANN {version}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
