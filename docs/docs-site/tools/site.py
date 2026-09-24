#!/usr/bin/env python3
"""Build and preview UCM documentation."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGS = ("en", "zh")


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _mkdocs(*args: str) -> int:
    return _run([sys.executable, "-m", "mkdocs", *args])


def serve(args: argparse.Namespace) -> int:
    """Serve the site locally with live reload (both languages)."""
    return _mkdocs("serve", "--dev-addr", args.dev_addr)


def build(args: argparse.Namespace) -> int:
    from site_build import build_site

    build_site(
        language=args.lang,
        site_dir=args.site_dir or ROOT / "site" / args.lang,
        site_url=args.site_url
        or f"http://127.0.0.1:8000/{'zh-cn' if args.lang == 'zh' else 'en'}/latest/",
        strict=args.strict,
        repository=args.repository,
        ref=args.ref,
        manifest_path=args.manifest,
    )
    return 0


def validate(args: argparse.Namespace) -> int:
    from site_build import build_site

    for language, slug in (("en", "en"), ("zh", "zh-cn")):
        build_site(
            language=language,
            site_dir=ROOT / "site" / language,
            site_url=f"https://docs.example.invalid/{slug}/latest/",
            strict=True,
        )
    return 0


def rtd(args: argparse.Namespace) -> int:
    from site_build import build_readthedocs

    build_readthedocs()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="site.py",
        description="UCM MkDocs documentation site entry point.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Serve the site locally.")
    p_serve.add_argument(
        "--dev-addr", default="127.0.0.1:8000", help="Dev server address."
    )
    p_serve.set_defaults(func=serve)

    p_build = sub.add_parser("build", help="Build the site.")
    p_build.add_argument(
        "--lang", default="en", choices=LANGS, help="Build only this language."
    )
    p_build.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors."
    )
    p_build.add_argument("--site-dir", type=Path)
    p_build.add_argument("--site-url")
    p_build.add_argument(
        "--repository", help="Read completed installation releases from OWNER/REPO."
    )
    p_build.add_argument("--ref", help="Git ref used for source/edit links.")
    p_build.add_argument(
        "--manifest", type=Path, help="Use a local validated release manifest."
    )
    p_build.set_defaults(func=build)

    p_rtd = sub.add_parser(
        "rtd", help="Build the language and version selected by Read the Docs."
    )
    p_rtd.set_defaults(func=rtd)

    p_validate = sub.add_parser("validate", help="Strict build across all languages.")
    p_validate.set_defaults(func=validate)

    args = parser.parse_args(argv)
    from release_manifest import ManifestError, ReleasePending

    try:
        return args.func(args)
    except ReleasePending as error:
        print(f"[docs] {error}; waiting for release completion", file=sys.stderr)
        return 183 if args.command == "rtd" else 2
    except (ManifestError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    raise SystemExit(main())
