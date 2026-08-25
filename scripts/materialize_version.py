#!/usr/bin/env python3
"""Materialize the source version used by UCM package and Chart builds."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "version.ini"
FORMAL_TAG = re.compile(
    r"v(?P<version>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)(?:rc(?:0|[1-9][0-9]*))?)",
    re.ASCII,
)
DRAFT_TAG = re.compile(
    r"draft/v(?P<base>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*))(?:-(?P<number>[1-9][0-9]*))?",
    re.ASCII,
)


def canonical_version(value: str) -> str:
    """Return *value* only when it is already canonical PEP 440."""
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"invalid PEP 440 version: {value!r}") from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError(
            f"version must use canonical PEP 440 spelling: {value!r} != {canonical!r}"
        )
    return canonical


def version_from_tag(tag: str) -> str:
    """Translate a formal or draft release tag to its Wheel version."""
    draft = DRAFT_TAG.fullmatch(tag)
    if draft is not None:
        number = draft.group("number") or "0"
        return canonical_version(f"{draft.group('base')}.dev{number}")
    formal = FORMAL_TAG.fullmatch(tag)
    if formal is not None:
        return canonical_version(formal.group("version"))
    raise ValueError(f"unsupported UCM release tag: {tag!r}")


def classify_tag(tag: str) -> dict[str, object]:
    """Return the immutable artifact coordinates and GitHub Release mode for *tag*."""
    version = version_from_tag(tag)
    draft = DRAFT_TAG.fullmatch(tag)
    if draft is not None:
        number = int(draft.group("number") or "0")
        return {
            "git_tag": tag,
            "release_kind": "draft",
            "version": version,
            "chart_version": f"{draft.group('base')}-draft.{number}",
            "image_version": version,
            "is_prerelease": True,
        }

    parsed = Version(version)
    chart_version = parsed.base_version
    if parsed.pre is not None:
        label, number = parsed.pre
        if label != "rc":
            raise ValueError(f"unsupported formal prerelease tag: {tag!r}")
        chart_version = f"{parsed.base_version}-rc.{number}"
    return {
        "git_tag": tag,
        "release_kind": "publish",
        "version": version,
        "chart_version": chart_version,
        "image_version": version,
        "is_prerelease": parsed.is_prerelease,
    }


def materialize_version(version: str, output: Path = DEFAULT_OUTPUT) -> str:
    """Atomically write one canonical VLLM_UC_VERSION assignment."""
    canonical = canonical_version(version)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o644
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as temporary:
            temporary.write(f"VLLM_UC_VERSION={canonical}\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return canonical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tag", help="formal v* or draft/v* tag")
    source.add_argument("--version", help="canonical PEP 440 version")
    parser.add_argument(
        "--classify",
        action="store_true",
        help="print Tag classification as JSON without writing version.ini",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        if arguments.classify:
            if arguments.tag is None:
                raise ValueError("--classify requires --tag")
            print(json.dumps(classify_tag(arguments.tag), sort_keys=True))
            return 0
        version = (
            version_from_tag(arguments.tag)
            if arguments.tag is not None
            else canonical_version(arguments.version)
        )
        print(materialize_version(version, arguments.output))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
