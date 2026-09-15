"""Notify Read the Docs only after a release manifest has been published and read back."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from release_manifest import (
    ManifestError,
    load_manifest,
    repository_name,
    resolve_manifest,
)

API = "https://app.readthedocs.org/api/v3"


def request(path: str, *, method: str = "GET", data=None):
    token = os.environ.get("RTD_API_TOKEN")
    if not token:
        raise ValueError("RTD_API_TOKEN is required when RTD publication is configured")
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    with urlopen(
        Request(API + path, data=body, headers=headers, method=method), timeout=30
    ) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def notify_release(
    repository: str, manifest_path: Path, projects: list[str]
) -> list[dict]:
    manifest = load_manifest(manifest_path)
    release = manifest["release"]
    tag = release["tag"]
    expected_url = f"https://github.com/{repository}/releases/tag/{quote(tag, safe='')}"
    if release["url"].casefold() != expected_url.casefold():
        raise ManifestError(
            "RTD release notification must use this repository's manifest"
        )
    if release["type"] not in {"stable", "prerelease"}:
        raise ManifestError("RTD release notification requires a published release")
    triggered = []
    for index, project in enumerate(projects):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project):
            raise ValueError("RTD project slug is missing or invalid")
        metadata = request(f"/projects/{project}/")
        if (
            repository_name(metadata["repository"]["url"]).casefold()
            != repository.casefold()
        ):
            raise ValueError(f"RTD project {project} is bound to another repository")
        if metadata["language"]["code"] != ("en", "zh-cn")[index]:
            raise ValueError(
                f"RTD project {project} has the wrong documentation language"
            )
    translations = request(f"/projects/{projects[0]}/translations/")["results"]
    if projects[1] not in {item["slug"] for item in translations}:
        raise ValueError(
            "Chinese RTD project is not a translation of the English project"
        )
    for index, project in enumerate(projects):
        prefix = f"/projects/{project}"
        language = ("en", "zh-cn")[index]
        request(prefix + "/sync-versions/", method="POST")
        version_path = prefix + f"/versions/{quote(tag, safe='')}/"
        # RTD synchronizes discovered refs asynchronously. Poll only this
        # documented transition, not failed builds or arbitrary API errors.
        for attempt in range(6):
            try:
                version = request(version_path)
                break
            except HTTPError as error:
                if error.code != 404 or attempt == 5:
                    raise
                time.sleep(5)
        if not version["active"]:
            previous = request(prefix + "/builds/?limit=20")["results"]
            last_id = max((build["id"] for build in previous), default=0)
            request(
                version_path, method="PATCH", data={"active": True, "hidden": False}
            )
            # Activation itself queues a build; capture that build instead of
            # creating a second build for the same version.
            for attempt in range(24):
                builds = request(prefix + "/builds/?limit=20")["results"]
                matching = [
                    build
                    for build in builds
                    if build["id"] > last_id and build["version"] == tag
                ]
                if matching:
                    build = min(matching, key=lambda item: item["id"])
                    break
                if attempt == 23:
                    raise RuntimeError(
                        f"RTD did not create the activated build: {project}/{tag}"
                    )
                time.sleep(5)
        else:
            build = request(version_path + "builds/", method="POST")["build"]
        triggered.append(
            {
                "project": project,
                "language": language,
                "version": tag,
                "build_id": build["id"],
            }
        )
        latest = request(prefix + "/versions/latest/builds/", method="POST")["build"]
        triggered.append(
            {
                "project": project,
                "language": language,
                "version": "latest",
                "build_id": latest["id"],
            }
        )
        if release["type"] == "stable":
            stable = request(prefix + "/versions/stable/")
            if stable["active"] and stable.get("ref") == tag:
                build = request(prefix + "/versions/stable/builds/", method="POST")[
                    "build"
                ]
                triggered.append(
                    {
                        "project": project,
                        "language": language,
                        "version": "stable",
                        "build_id": build["id"],
                    }
                )
    return triggered


def wait_for_builds(
    builds: list[dict], source_sha: str, timeout: int = 1800
) -> list[dict]:
    deadline = time.monotonic() + timeout
    pending = list(builds)
    completed = []
    while pending:
        for item in pending[:]:
            project, build_id = item["project"], item["build_id"]
            build = request(f"/projects/{project}/builds/{build_id}/")
            state = build["state"]
            state = state["code"] if isinstance(state, dict) else state
            if state not in {"finished", "cancelled"}:
                continue
            if state != "finished" or build["success"] is not True:
                raise RuntimeError(
                    f"RTD build failed: {project}/{build_id}: {build.get('error')}"
                )
            if (
                build["version"] != item["version"]
                or re.fullmatch(r"[0-9a-f]{40}", build["commit"]) is None
                or (item["version"] != "latest" and build["commit"] != source_sha)
            ):
                raise RuntimeError(
                    f"RTD build source differs from the release: {project}/{build_id}"
                )
            version = request(
                f"/projects/{project}/versions/{quote(item['version'], safe='')}/"
            )
            completed.append(
                {
                    **item,
                    "commit": build["commit"],
                    "url": version["urls"]["documentation"],
                    "status": "complete",
                }
            )
            pending.remove(item)
            print(f"[rtd] Build complete: {project}/{build_id}", flush=True)
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("RTD builds did not finish before the deadline")
            time.sleep(15)
    return completed


def read_public(url: str) -> bytes:
    # Public readback deliberately sends no RTD API credentials.
    with urlopen(
        Request(
            url,
            headers={"Cache-Control": "no-cache", "User-Agent": "ucm-docs-readback/1"},
        ),
        timeout=30,
    ) as response:
        return response.read()


def verify_public_docs(builds: list[dict], manifest: dict, repository: str) -> None:
    latest_manifest = resolve_manifest(repository)
    for build in builds:
        root = build["url"].rstrip("/") + "/"
        expected = latest_manifest if build["version"] == "latest" else manifest
        if (
            "language" in build
            and f"/{build['language']}/{build['version']}/" not in root
        ):
            raise ManifestError(
                f"RTD public URL has the wrong language or version: {root}"
            )
        manifest_url = urljoin(root, "release-manifest.json")
        try:
            observed = json.loads(read_public(manifest_url))
        except HTTPError as error:
            if error.code != 404 or expected is not None:
                raise
            observed = None
        if observed != expected:
            raise ManifestError(
                f"public documentation has the wrong release manifest: {root}"
            )
        for path, marker in (
            ("", "<html"),
            ("user-guide/quick_start/", "data-quickstart-selector"),
            ("toolkit/", "data-toolkit-install"),
            ("user-guide/frameworks/kubernetes/deploy/", "data-chart-install"),
        ):
            page = read_public(urljoin(root, path)).decode("utf-8")
            if marker not in page:
                raise ManifestError(
                    f"public documentation page is incomplete: {root}{path}"
                )
        print(f"[rtd] Public documentation verified: {root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project", action="append", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.project) != 2 or len(set(args.project)) != 2:
        raise ValueError("English and Chinese RTD projects must both be configured")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        raise ValueError("RTD validation requires the exact release source SHA")
    builds = notify_release(args.repository, args.manifest, args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"status": "pending", "builds": builds}, indent=2)
    )
    completed = wait_for_builds(builds, args.source_sha)
    verify_public_docs(completed, load_manifest(args.manifest), args.repository)
    args.output.write_text(
        json.dumps(
            {"status": "complete", "source_sha": args.source_sha, "builds": completed},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
