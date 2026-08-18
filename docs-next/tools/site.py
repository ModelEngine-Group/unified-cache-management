#!/usr/bin/env python3
"""Unified entry point for the UCM MkDocs documentation site.

Mirrors the commands described in the UCM docs re-architecture technical review
(section 5.3). This is a thin wrapper over ``mkdocs`` so that contributors have a
single, documented entry point for serving, building, validating, translating,
and generating content.

Usage:
    python tools/site.py serve
    python tools/site.py build --lang en --strict
    python tools/site.py build --lang zh-cn --strict
    python tools/site.py validate
    python tools/site.py translate --changed
    python tools/site.py generate
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGS = ("en", "zh-cn")


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _mkdocs(*args: str) -> int:
    return _run(["mkdocs", *args])


def serve(args: argparse.Namespace) -> int:
    """Serve the site locally with live reload (both languages)."""
    return _mkdocs("serve", "--dev-addr", args.dev_addr)


def build(args: argparse.Namespace) -> int:
    """Build the site.

    The mkdocs-static-i18n plugin builds every language whose ``build`` flag is
    true in a single ``mkdocs build`` invocation, so ``--lang`` is recorded for
    review but does not gate a separate build. ``--strict`` turns warnings into
    errors so missing pages/translations fail the build.
    """
    if args.lang not in LANGS:
        print(f"error: --lang must be one of {LANGS}, got {args.lang!r}")
        return 2
    cmd = ["build"]
    if args.strict:
        cmd.append("--strict")
    if args.clean:
        cmd.append("--clean")
    # Reconfigure the i18n plugin to build only the requested language by
    # toggling the other languages off via an environment hook. mkdocs-static-i18n
    # reads its config from mkdocs.yml, so we surface the requested language for
    # CI logs rather than splitting builds.
    print(f"[site] building language profile: {args.lang}")
    return _mkdocs(*cmd)


def validate(args: argparse.Namespace) -> int:
    """Run a strict build across all enabled languages."""
    return _mkdocs("build", "--strict")


def translate(args: argparse.Namespace) -> int:
    """Generate or update Chinese mirror content for changed English pages.

    Not yet implemented: this is the AI Robot responsibility (review module 2).
    It requires a CI-injected AI service endpoint and a writable GitHub App.
    See docs/ucm-mkdocs-site-rearchitecture-technical-review.md section 4.7.2.
    """
    print("[site] translate --changed: AI Chinese generation is not wired up locally.")
    print("       It runs in CI against changed English pages via an AI Robot.")
    print("       For now, author Chinese mirror pages manually under docs/zh-cn/.")
    return 0


def generate(args: argparse.Namespace) -> int:
    """Regenerate derived content from sources.

    Planned to rebuild ``docs/source/getting-started/docker-recipes.generated.md``
    from ``.github/release/release.yaml``. That file is currently missing from the
    tree and has no regeneration script, so this step is a no-op placeholder until
    the generator lands.
    """
    release_yaml = ROOT.parent / ".github" / "release" / "release.yaml"
    target = ROOT / "docs" / "en" / "getting-started" / "docker-recipes.generated.md"
    print("[site] generate: regenerate derived content from sources.")
    if release_yaml.exists():
        print(f"       source: {release_yaml}")
    else:
        print(f"       source not found: {release_yaml}")
    print(f"       target: {target}")
    print("       status: no generator implemented yet (placeholder).")
    return 0


def main(argv: list[str] | None = None) -> int:
    if shutil.which("mkdocs") is None and not (ROOT / ".venv").exists():
        print(
            "mkdocs is not on PATH. Create the environment first:\n"
            "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt",
            file=sys.stderr,
        )

    parser = argparse.ArgumentParser(
        prog="site.py",
        description="UCM MkDocs documentation site entry point.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Serve the site locally.")
    p_serve.add_argument("--dev-addr", default="127.0.0.1:8000", help="Dev server address.")
    p_serve.set_defaults(func=serve)

    p_build = sub.add_parser("build", help="Build the site.")
    p_build.add_argument("--lang", default="en", choices=LANGS, help="Language profile to record.")
    p_build.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    p_build.add_argument("--clean", action="store_true", help="Remove the site dir before building.")
    p_build.set_defaults(func=build)

    p_validate = sub.add_parser("validate", help="Strict build across all languages.")
    p_validate.set_defaults(func=validate)

    p_translate = sub.add_parser("translate", help="AI-generate Chinese for changed English pages (CI only).")
    p_translate.add_argument("--changed", action="store_true", help="Only translate changed pages.")
    p_translate.set_defaults(func=translate)

    p_generate = sub.add_parser("generate", help="Regenerate derived content from sources.")
    p_generate.set_defaults(func=generate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
