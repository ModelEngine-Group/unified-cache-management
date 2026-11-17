#!/usr/bin/env python3
"""Require matching English and Chinese paths for every current Markdown page."""
from __future__ import annotations

import argparse
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"


def missing_pairs(docs_root: Path = DOCS_ROOT) -> list[tuple[str, str]]:
    pages = {
        language: {
            p.relative_to(docs_root / language).as_posix()
            for p in (docs_root / language).rglob("*.md")
        }
        for language in ("en", "zh")
    }
    return sorted(
        (f"{language}/{path}", f"{other}/{path}")
        for language, other in (("en", "zh"), ("zh", "en"))
        for path in pages[language] - pages[other]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, default=DOCS_ROOT)
    args = parser.parse_args()
    missing = missing_pairs(args.docs_root)
    for page, counterpart in missing:
        print(f"{page} is missing its bilingual counterpart: {counterpart}")
    if missing:
        print(f"Bilingual documentation check failed: {len(missing)} missing files.")
        return 1
    print(
        "Bilingual documentation check passed: all Markdown paths have both languages."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
