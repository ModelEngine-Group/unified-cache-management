#!/usr/bin/env python3
"""Materialize the source version used by UCM package and Chart builds."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from packaging.version import InvalidVersion, Version

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "version.ini"
VERSION_TRIPLE_PATTERN = (
    r"(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)"
)
FORMAL_TAG = re.compile(
    r"v(?P<version>" + VERSION_TRIPLE_PATTERN + r"(?:rc(?:0|[1-9][0-9]*))?)",
    re.ASCII,
)
STABLE_TAG = re.compile(
    r"v(?P<version>" + VERSION_TRIPLE_PATTERN + r")",
    re.ASCII,
)
DRAFT_TAG = re.compile(
    r"draft/v(?P<base>" + VERSION_TRIPLE_PATTERN + r")(?:-(?P<number>[1-9][0-9]*))?",
    re.ASCII,
)
NIGHTLY_TAG = re.compile(
    r"nightly/v(?P<base>"
    + VERSION_TRIPLE_PATTERN
    + r")-(?P<date>[0-9]{8})-(?P<number>[1-9][0-9]*)",
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


def _validate_nightly_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise ValueError(f"invalid Nightly date: {value!r}") from error
    if parsed.strftime("%Y%m%d") != value:
        raise ValueError(f"invalid Nightly date: {value!r}")
    return value


def version_from_tag(tag: str) -> str:
    """Translate an exact supported release Tag to its Wheel version."""
    draft = DRAFT_TAG.fullmatch(tag)
    if draft is not None:
        number = draft.group("number") or "0"
        return canonical_version(f"{draft.group('base')}.dev{number}")
    nightly = NIGHTLY_TAG.fullmatch(tag)
    if nightly is not None:
        release_date = _validate_nightly_date(nightly.group("date"))
        number = int(nightly.group("number"))
        return canonical_version(
            f"{nightly.group('base')}.dev{release_date}{number:03d}"
        )
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
            "release_type": "draft",
            "release_kind": "draft",
            "version": version,
            "chart_version": f"{draft.group('base')}-draft.{number}",
            "image_version": version,
            "is_prerelease": True,
        }

    nightly = NIGHTLY_TAG.fullmatch(tag)
    if nightly is not None:
        release_date = _validate_nightly_date(nightly.group("date"))
        number = int(nightly.group("number"))
        return {
            "git_tag": tag,
            "release_type": "nightly",
            "release_kind": "publish",
            "version": version,
            "chart_version": (
                f"{nightly.group('base')}-nightly.{release_date}.{number}"
            ),
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
        "release_type": "prerelease" if parsed.is_prerelease else "stable",
        "release_kind": "publish",
        "version": version,
        "chart_version": chart_version,
        "image_version": version,
        "is_prerelease": parsed.is_prerelease,
    }


def next_patch_version(tags: Iterable[str]) -> str:
    """Return the patch after the highest exact ``vX.Y.Z`` Tag."""
    versions = [
        Version(match.group("version"))
        for tag in tags
        if (match := STABLE_TAG.fullmatch(tag)) is not None
    ]
    if not versions:
        raise ValueError("no strict Stable vX.Y.Z Tag found")
    highest = max(versions)
    return f"{highest.major}.{highest.minor}.{highest.micro + 1}"


def next_nightly_sequence(
    tags: Iterable[str], *, base_version: str, release_date: str
) -> int:
    """Return the next sequence for one exact Nightly base and date."""
    if STABLE_TAG.fullmatch(f"v{base_version}") is None:
        raise ValueError(f"invalid Nightly base version: {base_version!r}")
    _validate_nightly_date(release_date)
    sequences = [
        int(match.group("number"))
        for tag in tags
        if (match := NIGHTLY_TAG.fullmatch(tag)) is not None
        and match.group("base") == base_version
        and match.group("date") == release_date
    ]
    return max(sequences, default=0) + 1


def next_nightly_classification(
    tags: Iterable[str], *, release_date: str
) -> dict[str, object]:
    """Classify the next Nightly derived only from existing Tag names."""
    known_tags = tuple(tags)
    base_version = next_patch_version(known_tags)
    sequence = next_nightly_sequence(
        known_tags, base_version=base_version, release_date=release_date
    )
    return classify_tag(f"nightly/v{base_version}-{release_date}-{sequence}")


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
    source.add_argument("--tag", help="formal, Draft, or Nightly Tag")
    source.add_argument("--version", help="canonical PEP 440 version")
    source.add_argument(
        "--next-nightly",
        action="store_true",
        help="classify the next Nightly from a newline-delimited Tag file",
    )
    parser.add_argument("--tags-file", type=Path)
    parser.add_argument("--date", help="Nightly date in YYYYMMDD form")
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
        if arguments.next_nightly:
            if arguments.classify:
                raise ValueError("--next-nightly already returns classification JSON")
            if arguments.tags_file is None or arguments.date is None:
                raise ValueError("--next-nightly requires --tags-file and --date")
            tags = [
                line.strip()
                for line in arguments.tags_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            print(
                json.dumps(
                    next_nightly_classification(tags, release_date=arguments.date),
                    sort_keys=True,
                )
            )
            return 0
        if arguments.tags_file is not None or arguments.date is not None:
            raise ValueError("--tags-file and --date require --next-nightly")
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
