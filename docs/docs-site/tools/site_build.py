"""Build one language and version without modifying documentation sources."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

from release_manifest import (
    RELEASE_MANIFEST_FILENAME,
    ManifestError,
    load_manifest,
    repository_name,
    resolve_manifest,
)

DOCS_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = {"en": "en", "zh": "zh-cn"}


def current_repository() -> str:
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    clone_url = os.environ.get("READTHEDOCS_GIT_CLONE_URL")
    if not clone_url:
        clone_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=DOCS_ROOT, text=True
        ).strip()
    return repository_name(clone_url)


def current_ref() -> str:
    if os.environ.get("READTHEDOCS_VERSION_TYPE") == "external":
        return (
            os.environ.get("READTHEDOCS_GIT_COMMIT_HASH")
            or subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=DOCS_ROOT, text=True
            ).strip()
        )
    return (
        os.environ.get("READTHEDOCS_GIT_IDENTIFIER")
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=DOCS_ROOT, text=True
        ).strip()
    )


def build_site(
    *,
    language: str,
    site_dir: Path,
    site_url: str,
    strict: bool = True,
    repository: str | None = None,
    ref: str | None = None,
    release_tag: str | None = None,
    manifest_path: Path | None = None,
    docs_root: Path = DOCS_ROOT,
) -> None:
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    if language not in LANGUAGES:
        raise ValueError(f"Unsupported documentation language: {language}")
    manifest = (
        load_manifest(manifest_path)
        if manifest_path
        else (resolve_manifest(repository, tag=release_tag) if repository else None)
    )
    config = load_config(
        config_file=str(docs_root / "mkdocs.yml"),
        docs_dir=str(docs_root / "docs"),
        site_dir=str(site_dir.resolve()),
        site_url=site_url,
        strict=strict,
    )
    i18n = config.plugins["i18n"]
    settings = dict(i18n.config)
    settings["languages"] = [dict(item) for item in i18n.config.languages]
    settings["build_only_locale"] = language
    errors, warnings = i18n.load_config(settings)
    if errors or warnings:
        raise ValueError(f"Invalid language configuration: {errors or warnings}")
    if repository:
        config.repo_url = f"https://github.com/{repository}"
        config.repo_name = repository
        config.edit_uri = (
            f"edit/{quote(ref or current_ref(), safe='/')}/docs/docs-site/docs/"
        )
    config.plugins.on_startup(command="build", dirty=False)
    try:
        build(config)
    finally:
        config.plugins.on_shutdown()
    if manifest is not None:
        (site_dir / RELEASE_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[docs] Installation artifacts: {manifest['release']['url']}")
    else:
        print(
            "[docs] No completed Schema 9 release; installation links to source builds."
        )


def build_readthedocs() -> None:
    language = {"en": "en", "zh-cn": "zh"}[os.environ["READTHEDOCS_LANGUAGE"]]
    ref = current_ref()
    tag = ref if os.environ.get("READTHEDOCS_VERSION_TYPE") == "tag" else None
    if tag:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=DOCS_ROOT, text=True
        ).strip()
        tagged = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            cwd=DOCS_ROOT,
            text=True,
        ).strip()
        if head != tagged:
            raise ManifestError("RTD checkout does not match the release tag")
    build_site(
        language=language,
        site_dir=Path(os.environ["READTHEDOCS_OUTPUT"]) / "html",
        site_url=os.environ["READTHEDOCS_CANONICAL_URL"],
        repository=current_repository(),
        ref=ref,
        release_tag=tag,
    )
