"""Fetch the current public installation manifest for a documentation build."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".github" / "release"))
from ucm_release.manifest import (
    RELEASE_MANIFEST_FILENAME,
    RELEASE_MANIFEST_KIND,
    RELEASE_MANIFEST_SCHEMA_VERSION,
    SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS,
    ManifestError,
)
from ucm_release.manifest import load_manifest as load_manifest
from ucm_release.manifest import (
    require_stable_manifest,
    validate_manifest,
)


class ReleasePending(ManifestError):
    """The tag exists before its publication has completed."""


def fetch_json(url: str) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "ucm-docs"}
    if urlparse(url).netloc == "api.github.com" and os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            return json.load(response)
    except HTTPError:
        raise
    except (OSError, ValueError) as error:
        raise ManifestError(f"Unable to read {url}: {error}") from error


def _release_manifest(
    repository: str, release: Mapping[str, Any], *, skip_unsupported: bool = False
) -> dict[str, Any] | None:
    tag = release["tag_name"]
    assets = [
        a
        for a in release.get("assets", [])
        if a.get("name") == RELEASE_MANIFEST_FILENAME
    ]
    if not assets:
        return None
    if len(assets) != 1:
        raise ManifestError(f"Release {tag} has multiple installation manifests")
    value = fetch_json(assets[0]["browser_download_url"])
    if not isinstance(value, dict):
        raise ManifestError(f"Release {tag} manifest must be an object")
    if skip_unsupported and (
        value.get("kind") != RELEASE_MANIFEST_KIND
        or value.get("schema_version") not in SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS
    ):
        return None
    manifest = validate_manifest(value)
    if manifest["release"]["tag"] != tag:
        raise ManifestError(f"Release {tag} has a manifest for another tag")
    if tag != f"v{manifest['release']['version']}":
        raise ManifestError(f"Release {tag} tag and package version differ")
    expected_type = "prerelease" if release.get("prerelease") else "stable"
    if manifest["release"]["type"] != expected_type:
        raise ManifestError(f"Release {tag} publication type differs from its manifest")
    expected = f"https://github.com/{repository}/releases/tag/{quote(tag, safe='')}"
    if manifest["release"]["url"].casefold() != expected.casefold():
        raise ManifestError(f"Release {tag} manifest belongs to another repository")
    published_assets = {a.get("name") for a in release.get("assets", [])}
    if set(manifest["github_release_assets"]) != published_assets:
        raise ManifestError(f"Release {tag} assets differ from its completed manifest")
    urls = {a["name"]: a["browser_download_url"] for a in release["assets"]}
    artifacts = list(manifest["wheels"])
    if manifest["chart"] is not None:
        artifacts.append(manifest["chart"])
    if manifest.get("toolkit") is not None:
        artifacts.append(manifest["toolkit"])
    for artifact in artifacts:
        if artifact["url"] != urls[artifact["filename"]]:
            raise ManifestError(
                f"Release {tag} download URL differs for {artifact['filename']}"
            )
    return manifest


def resolve_manifest(
    repository: str, *, tag: str | None = None
) -> dict[str, Any] | None:
    """Resolve an exact release, or prefer a completed stable release over prereleases."""
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ManifestError("repository must be OWNER/REPO")
    base = f"https://api.github.com/repos/{repository}/releases"
    if tag:
        try:
            release = fetch_json(f"{base}/tags/{quote(tag, safe='')}")
        except HTTPError as error:
            if error.code == 404:
                raise ReleasePending(f"Release {tag} is not published yet") from error
            raise
        if release.get("draft"):
            raise ReleasePending(f"Release {tag} is still a draft")
        manifest = _release_manifest(repository, release)
        if manifest is None:
            raise ReleasePending(f"Release {tag} has no completed supported manifest")
        return manifest

    stable_releases = []
    prereleases = []
    page = 1
    while True:
        batch = fetch_json(f"{base}?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise ManifestError("GitHub releases response must be an array")
        for release in batch:
            if release.get("draft"):
                continue
            try:
                version = Version(release["tag_name"].removeprefix("v"))
            except (InvalidVersion, KeyError):
                continue
            if version.is_devrelease or version.local:
                continue
            if release.get("prerelease"):
                prereleases.append((version, release))
            elif not version.is_prerelease:
                stable_releases.append((version, release))
        if len(batch) < 100:
            break
        page += 1
    # Development docs still need installation data before the first stable release.
    for candidates in (stable_releases, prereleases):
        for _, release in sorted(candidates, key=lambda item: item[0], reverse=True):
            manifest = _release_manifest(repository, release, skip_unsupported=True)
            if manifest is not None:
                if not release.get("prerelease"):
                    require_stable_manifest(manifest)
                return manifest
    return None


def repository_name(url: str) -> str:
    match = re.fullmatch(
        r"(?:https://github.com/|git@github.com:)([\w.-]+/[\w.-]+?)(?:\.git)?/?", url
    )
    if not match:
        raise ManifestError("Documentation repository must be a GitHub OWNER/REPO URL")
    return match[1]
