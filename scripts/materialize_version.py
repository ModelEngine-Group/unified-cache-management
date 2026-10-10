#!/usr/bin/env python3
"""Materialize the source version used by UCM package and Chart builds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "version.ini"
RELEASE_ROOT = REPOSITORY_ROOT / ".github" / "release"
sys.path.insert(0, str(RELEASE_ROOT))

from ucm_release import version_config  # noqa: E402


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
        "--version-config",
        type=Path,
        help="validate or derive the Tag from this version.ini authority",
    )
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
            if arguments.version_config is None:
                raise ValueError("--next-nightly requires --version-config")
            config = version_config.load(arguments.version_config)
            print(
                json.dumps(
                    version_config.next_nightly_classification(
                        tags,
                        base_version=config["ucm_base_version"],
                        release_date=arguments.date,
                    ),
                    sort_keys=True,
                )
            )
            return 0
        if arguments.tags_file is not None or arguments.date is not None:
            raise ValueError("--tags-file and --date require --next-nightly")
        if arguments.classify:
            if arguments.tag is None:
                raise ValueError("--classify requires --tag")
            classification = (
                version_config.validate_tag_against_config(
                    arguments.tag, arguments.version_config
                )
                if arguments.version_config is not None
                else version_config.classify_tag(arguments.tag)
            )
            print(json.dumps(classification, sort_keys=True))
            return 0
        version = (
            version_config.version_from_tag(arguments.tag)
            if arguments.tag is not None
            else version_config.canonical_version(arguments.version)
        )
        print(version_config.materialize_version(version, arguments.output))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
