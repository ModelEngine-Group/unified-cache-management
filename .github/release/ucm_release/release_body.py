"""Update the pipeline-owned section while preserving GitHub Release notes.

Keep this entry point standard-library-only so early failure jobs can use it
before installing the release build dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BEGIN = "<!-- ucm-release:begin -->"
END = "<!-- ucm-release:end -->"


def merge_body(body: str | None, notes: str) -> str:
    """Append once, then replace only the marked pipeline section on reruns."""
    body = body or ""
    section = f"{BEGIN}\n{notes.rstrip()}\n{END}"
    if BEGIN in body or END in body:
        if (
            body.count(BEGIN) != 1
            or body.count(END) != 1
            or body.index(BEGIN) > body.index(END)
        ):
            raise ValueError("Release body has ambiguous UCM section markers")
        start = body.index(BEGIN)
        end = body.index(END) + len(END)
        merged = body[:start] + section + body[end:]
    else:
        separator = ""
        if body and not body.endswith("\n\n"):
            separator = "\n" if body.endswith("\n") else "\n\n"
        merged = body + separator + section
    # Match the workflow's body readback after shell command substitution.
    return merged.rstrip("\n") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    args = parser.parse_args()
    release = json.loads(args.release.read_text(encoding="utf-8"))
    merged = merge_body(release.get("body"), args.notes.read_text(encoding="utf-8"))
    args.notes.write_text(merged, encoding="utf-8")


if __name__ == "__main__":
    main()
