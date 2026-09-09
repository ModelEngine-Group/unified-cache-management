"""Retain and remove one UCM Tag release from its public schema-v6 manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

MANIFEST_KIND = "ucm-release-manifest"
MANIFEST_SCHEMA_VERSION = 6
MANIFEST_FILENAME = "release-manifest.json"
RELEASE_TYPES = frozenset({"stable", "prerelease", "draft", "nightly"})
RETRY_DELAYS_SECONDS = (0.0, 5.0, 15.0)

_MANIFEST_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "tag",
        "release_type",
        "actions_run_id",
        "chart_oci",
        "runtime_images",
        "github_release_assets",
    }
)
_RUNTIME_CHANNELS = ("ghcr", "dockerhub")
_RUNTIME_IMAGE_KEYS = frozenset({"members", "indexes"})
_OCI_REFERENCE = re.compile(
    r"(?P<repository>(?:ghcr\.io|docker\.io)/"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+)"
    r":(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})"
)
_REPOSITORY = re.compile(
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<repo>[A-Za-z0-9_.-]+)"
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MISSING_MARKERS = (
    "404",
    "manifest unknown",
    "manifest_unknown",
    "name unknown",
    "name_unknown",
    "not found",
)
_TRANSPORT_MARKERS = (
    "connection reset",
    "connection refused",
    "context deadline exceeded",
    "i/o timeout",
    "network is unreachable",
    "temporary failure",
    "timed out",
    "timeout",
    "tls handshake timeout",
)


class CleanupError(ValueError):
    """A local contract or permanent remote cleanup error."""


class RemoteError(CleanupError):
    """A structured remote failure used by the retry policy."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def is_missing(self) -> bool:
        return self.status == 404

    @property
    def is_retryable(self) -> bool:
        return (
            self.status is None
            or self.status in {409, 429}
            or (self.status is not None and self.status >= 500)
        )


class UnsafePackageVersion(RemoteError):
    """A GHCR package version has Tags outside the requested resource."""

    def __init__(self, reference: str, tags: Sequence[str]) -> None:
        rendered = ", ".join(sorted(tags))
        super().__init__(
            f"refusing to delete {reference}: package version also has Tags [{rendered}]",
            status=422,
        )


@dataclass(frozen=True)
class Resource:
    kind: str
    reference: str
    identifier: str | int | tuple[str, ...] | None = None
    holds_manifest: bool = False


@dataclass(frozen=True)
class ResourceFailure:
    resource: Resource
    attempts: int
    final_error: str


@dataclass(frozen=True)
class CleanupReport:
    tag: str
    completed: bool
    stopped_phase: int | None
    failures: tuple[ResourceFailure, ...]


@dataclass(frozen=True)
class ManifestRecord:
    manifest: dict[str, Any]
    created_at: str
    release_id: int
    draft: bool
    prerelease: bool


@dataclass(frozen=True)
class RetentionSelection:
    candidates: tuple[ManifestRecord, ...]
    skipped_reason: str | None = None


class CleanupRemote(Protocol):
    repository: str

    def probe(self, resource: Resource) -> object | None: ...

    def delete(self, resource: Resource, state: object) -> None: ...

    def release_resources(self, tag: str) -> list[Resource]: ...


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CleanupError(f"{context} must be an object")
    return value


def _array(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise CleanupError(f"{context} must be an array")
    return value


def _exact_keys(value: dict[str, Any], expected: frozenset[str], context: str) -> None:
    if set(value) != set(expected):
        missing = sorted(expected - value.keys())
        extra = sorted(value.keys() - expected)
        raise CleanupError(
            f"{context} fields must be exact; missing={missing}, extra={extra}"
        )


def _tagged_oci_reference(value: object, context: str, *, registry: str) -> str:
    if not isinstance(value, str):
        raise CleanupError(f"{context} must be a tagged OCI reference")
    match = _OCI_REFERENCE.fullmatch(value)
    if match is None or not match.group("repository").startswith(registry + "/"):
        raise CleanupError(f"{context} must be a tagged {registry} reference")
    return value


def validate_manifest(
    value: object, *, expected_tag: str | None = None
) -> dict[str, Any]:
    """Validate and return the exact public cleanup manifest contract."""
    manifest = _mapping(value, "release manifest")
    _exact_keys(manifest, _MANIFEST_KEYS, "release manifest")
    if manifest["kind"] != MANIFEST_KIND:
        raise CleanupError("release manifest kind is invalid")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise CleanupError("release manifest must use schema version 6")
    tag = manifest["tag"]
    if not isinstance(tag, str) or not tag or tag.strip() != tag:
        raise CleanupError("release manifest Tag must be a non-empty exact string")
    if expected_tag is not None and tag != expected_tag:
        raise CleanupError("release manifest Tag differs from the requested Tag")
    if (
        not isinstance(manifest["release_type"], str)
        or manifest["release_type"] not in RELEASE_TYPES
    ):
        raise CleanupError("release manifest release type is invalid")
    run_id = manifest["actions_run_id"]
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id < 1:
        raise CleanupError("release manifest Actions run ID must be a positive integer")

    chart = manifest["chart_oci"]
    if chart is not None:
        _tagged_oci_reference(chart, "release manifest Chart OCI", registry="ghcr.io")

    runtime_images = _mapping(
        manifest["runtime_images"], "release manifest Runtime Images"
    )
    _exact_keys(
        runtime_images, frozenset(_RUNTIME_CHANNELS), "release manifest Runtime Images"
    )
    for channel in _RUNTIME_CHANNELS:
        channel_value = _mapping(
            runtime_images[channel], f"release manifest {channel} images"
        )
        _exact_keys(
            channel_value, _RUNTIME_IMAGE_KEYS, f"release manifest {channel} images"
        )
        registry = "ghcr.io" if channel == "ghcr" else "docker.io"
        seen: set[str] = set()
        for image_kind in ("members", "indexes"):
            references = _array(
                channel_value[image_kind],
                f"release manifest {channel} {image_kind}",
            )
            for index, reference in enumerate(references):
                normalized = _tagged_oci_reference(
                    reference,
                    f"release manifest {channel} {image_kind}[{index}]",
                    registry=registry,
                )
                if normalized in seen:
                    raise CleanupError(
                        f"release manifest {channel} image references must be unique"
                    )
                seen.add(normalized)

    assets = _array(
        manifest["github_release_assets"], "release manifest GitHub Release assets"
    )
    seen_assets: set[str] = set()
    for asset in assets:
        if (
            not isinstance(asset, str)
            or not asset
            or asset in {".", ".."}
            or Path(asset).name != asset
        ):
            raise CleanupError(
                "release manifest has an invalid GitHub Release asset name"
            )
        if asset in seen_assets:
            raise CleanupError("release manifest GitHub Release assets must be unique")
        seen_assets.add(asset)
    if MANIFEST_FILENAME not in seen_assets:
        raise CleanupError(
            "release manifest must list itself as a GitHub Release asset"
        )
    return manifest


def registry_resources(manifest: object) -> list[Resource]:
    """Project phase-one resources in the required deletion order."""
    validated = validate_manifest(manifest)
    ghcr_resources: list[tuple[str, str]] = []
    if validated["chart_oci"] is not None:
        ghcr_resources.append(("chart-oci", validated["chart_oci"]))
    images = validated["runtime_images"]
    ghcr_resources.extend(("ghcr-index", ref) for ref in images["ghcr"]["indexes"])
    ghcr_resources.extend(("ghcr-member", ref) for ref in images["ghcr"]["members"])

    allowed_tags_by_package: dict[str, set[str]] = {}
    for _, reference in ghcr_resources:
        match = _OCI_REFERENCE.fullmatch(reference)
        if match is None:
            raise AssertionError("validated GHCR reference no longer parses")
        allowed_tags_by_package.setdefault(match.group("repository"), set()).add(
            match.group("tag")
        )
    result = []
    for kind, reference in ghcr_resources:
        match = _OCI_REFERENCE.fullmatch(reference)
        if match is None:
            raise AssertionError("validated GHCR reference no longer parses")
        allowed_tags = tuple(sorted(allowed_tags_by_package[match.group("repository")]))
        result.append(Resource(kind, reference, allowed_tags))

    result.extend(
        Resource("dockerhub-index", ref) for ref in images["dockerhub"]["indexes"]
    )
    result.extend(
        Resource("dockerhub-member", ref) for ref in images["dockerhub"]["members"]
    )
    return result


def select_retention_candidates(
    records: Sequence[ManifestRecord],
    *,
    current_tag: str,
    release_type: str,
    max_count: int,
    pypi_enabled: bool,
) -> RetentionSelection:
    """Select the oldest excess same-type Tags without guessing old manifests."""
    if not isinstance(current_tag, str) or not current_tag:
        raise CleanupError("current Tag must be non-empty")
    if release_type not in RELEASE_TYPES:
        raise CleanupError("retention release type is invalid")
    if (
        not isinstance(max_count, int)
        or isinstance(max_count, bool)
        or max_count == 0
        or max_count < -1
    ):
        raise CleanupError("max_count must be -1 or an integer >= 1")
    if not isinstance(pypi_enabled, bool):
        raise CleanupError("pypi_enabled must be boolean")
    if max_count == -1:
        return RetentionSelection((), "retention skipped: max_count is unlimited")
    if pypi_enabled:
        return RetentionSelection(
            (),
            "retention skipped: PyPI is enabled for this release type",
        )

    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        try:
            manifest = validate_manifest(record.manifest)
        except CleanupError:
            continue
        if manifest["release_type"] != release_type or manifest["tag"] == current_tag:
            continue
        if (
            not isinstance(record.created_at, str)
            or not record.created_at
            or not isinstance(record.release_id, int)
            or isinstance(record.release_id, bool)
            or record.release_id < 1
            or not isinstance(record.draft, bool)
            or not isinstance(record.prerelease, bool)
        ):
            continue
        expected_visibility = {
            "stable": (False, False),
            "prerelease": (False, True),
            "draft": (True, True),
            "nightly": (False, True),
        }[release_type]
        if (record.draft, record.prerelease) != expected_visibility:
            continue
        grouped.setdefault(manifest["tag"], []).append(record)

    unique_records: list[ManifestRecord] = []
    for tag_records in grouped.values():
        first = tag_records[0].manifest
        if any(record.manifest != first for record in tag_records[1:]):
            continue
        unique_records.append(
            min(tag_records, key=lambda item: (item.created_at, item.release_id))
        )
    unique_records.sort(
        key=lambda item: (item.created_at, item.release_id, item.manifest["tag"])
    )
    allowed_other_tags = max_count - 1
    excess = max(0, len(unique_records) - allowed_other_tags)
    return RetentionSelection(tuple(unique_records[:excess]))


def _failure(
    resource: Resource, attempts: int, error: BaseException
) -> ResourceFailure:
    return ResourceFailure(resource, attempts, str(error) or type(error).__name__)


def delete_resource_with_retry(
    remote: CleanupRemote,
    resource: Resource,
    *,
    sleeper=time.sleep,
    fail_resource: str | None = None,
) -> ResourceFailure | None:
    """Probe and delete one resource with exactly three independent attempts."""
    for attempt, delay in enumerate(RETRY_DELAYS_SECONDS, start=1):
        print(
            f"cleanup resource_type={resource.kind} reference={resource.reference} "
            f"attempt={attempt}/{len(RETRY_DELAYS_SECONDS)} delay={int(delay)}s",
            flush=True,
        )
        if delay:
            sleeper(delay)
        try:
            state = remote.probe(resource)
            if state is None:
                return None
            if fail_resource is not None and resource.reference == fail_resource:
                raise RemoteError(
                    f"synthetic HTTP 503 for {resource.reference}", status=503
                )
            remote.delete(resource, state)
            return None
        except RemoteError as error:
            if error.is_missing:
                return None
            if error.is_retryable and attempt < len(RETRY_DELAYS_SECONDS):
                continue
            return _failure(resource, attempt, error)
        except CleanupError as error:
            return _failure(resource, attempt, error)
    raise AssertionError("resource retry loop exhausted without a result")


def _run_phase(
    remote: CleanupRemote,
    resources: Sequence[Resource],
    *,
    sleeper,
    fail_resource: str | None,
) -> list[ResourceFailure]:
    failures: list[ResourceFailure] = []
    for resource in resources:
        failure = delete_resource_with_retry(
            remote,
            resource,
            sleeper=sleeper,
            fail_resource=fail_resource,
        )
        if failure is not None:
            failures.append(failure)
    return failures


def _release_resources_with_retry(
    remote: CleanupRemote, tag: str, *, sleeper
) -> tuple[list[Resource], ResourceFailure | None]:
    collection = Resource("github-releases", tag)
    for attempt, delay in enumerate(RETRY_DELAYS_SECONDS, start=1):
        print(
            f"cleanup resource_type={collection.kind} reference={collection.reference} "
            f"attempt={attempt}/{len(RETRY_DELAYS_SECONDS)} delay={int(delay)}s",
            flush=True,
        )
        if delay:
            sleeper(delay)
        try:
            return remote.release_resources(tag), None
        except RemoteError as error:
            if error.is_missing:
                return [], None
            if error.is_retryable and attempt < len(RETRY_DELAYS_SECONDS):
                continue
            return [], _failure(collection, attempt, error)
        except CleanupError as error:
            return [], _failure(collection, attempt, error)
    raise AssertionError("Release discovery retry loop exhausted without a result")


def cleanup_manifest(
    manifest: object,
    remote: CleanupRemote,
    *,
    sleeper=time.sleep,
    fail_resource: str | None = None,
) -> CleanupReport:
    """Delete one Tag through the four recovery-preserving phases."""
    validated = validate_manifest(manifest)
    tag = validated["tag"]
    phase_one_resources = registry_resources(validated)

    failures = _run_phase(
        remote,
        phase_one_resources,
        sleeper=sleeper,
        fail_resource=fail_resource,
    )
    if failures:
        return CleanupReport(tag, False, 1, tuple(failures))

    run_id = validated["actions_run_id"]
    actions = Resource(
        "actions-run",
        f"https://github.com/{remote.repository}/actions/runs/{run_id}",
        run_id,
    )
    failures = _run_phase(
        remote, [actions], sleeper=sleeper, fail_resource=fail_resource
    )
    if failures:
        return CleanupReport(tag, False, 2, tuple(failures))

    git_tag = Resource("git-tag", tag, tag)
    failures = _run_phase(
        remote, [git_tag], sleeper=sleeper, fail_resource=fail_resource
    )
    if failures:
        return CleanupReport(tag, False, 3, tuple(failures))

    releases, discovery_failure = _release_resources_with_retry(
        remote, tag, sleeper=sleeper
    )
    if discovery_failure is not None:
        return CleanupReport(tag, False, 4, (discovery_failure,))
    unbacked = [resource for resource in releases if not resource.holds_manifest]
    failures = _run_phase(
        remote, unbacked, sleeper=sleeper, fail_resource=fail_resource
    )
    if failures:
        return CleanupReport(tag, False, 4, tuple(failures))
    backed = [resource for resource in releases if resource.holds_manifest]
    failures = _run_phase(remote, backed, sleeper=sleeper, fail_resource=fail_resource)
    return CleanupReport(tag, not failures, 4 if failures else None, tuple(failures))


def render_failure_summary(failures: Sequence[ResourceFailure]) -> str:
    """Render only final failures; successful attempts intentionally stay out."""
    if not failures:
        return ""

    def cell(value: object) -> str:
        return " ".join(str(value).split()).replace("|", "\\|")

    lines = [
        "## UCM release cleanup final failures",
        "",
        "| Resource type | Reference | Final error |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {cell(item.resource.kind)} | {cell(item.resource.reference)} | "
        f"{cell(item.final_error)} |"
        for item in failures
    )
    return "\n".join(lines) + "\n"


def append_failure_summary(
    path: Path | None, failures: Sequence[ResourceFailure]
) -> None:
    summary = render_failure_summary(failures)
    if path is None or not summary:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(summary)


class ProductionRemote:
    """GitHub REST and Crane adapter for the cleanup domain Interface."""

    def __init__(
        self,
        repository: str,
        token: str,
        *,
        crane: str = "crane",
        api_base: str = "https://api.github.com",
        opener: Any | None = None,
    ) -> None:
        match = _REPOSITORY.fullmatch(repository)
        if match is None:
            raise CleanupError("repository must use owner/name form")
        if not token:
            raise CleanupError("GH_TOKEN or GITHUB_TOKEN is required")
        self.repository = repository
        self.owner = match.group("owner")
        self.token = token
        self.crane = crane
        self.api_base = api_base.rstrip("/")
        self._opener = opener or urllib.request.build_opener()
        self._package_owner_prefix: str | None = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        accept: str = "application/vnd.github+json",
    ) -> bytes:
        url = path if path.startswith("https://") else self.api_base + path
        request = urllib.request.Request(
            url,
            method=method,
            headers={
                "Accept": accept,
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "ucm-release-cleanup",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with self._opener.open(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace").strip()
            except OSError:
                detail = ""
            raise RemoteError(
                f"GitHub API HTTP {error.code}: {detail or error.reason}",
                status=error.code,
            ) from error
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            raise RemoteError(f"GitHub API transport error: {error}") from error

    def _github_json(self, method: str, path: str) -> Any:
        raw = self._request(method, path)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CleanupError("GitHub API returned malformed JSON") from error

    def _all_pages(self, path: str) -> list[dict[str, Any]]:
        separator = "&" if "?" in path else "?"
        values: list[dict[str, Any]] = []
        for page in range(1, 1001):
            value = self._github_json(
                "GET", f"{path}{separator}per_page=100&page={page}"
            )
            if not isinstance(value, list) or any(
                not isinstance(item, dict) for item in value
            ):
                raise CleanupError("GitHub API paginated response must be an array")
            values.extend(value)
            if len(value) < 100:
                return values
        raise CleanupError("GitHub API pagination exceeded 1000 pages")

    def list_releases(self) -> list[dict[str, Any]]:
        owner_repo = urllib.parse.quote(self.repository, safe="/")
        return self._all_pages(f"/repos/{owner_repo}/releases")

    @staticmethod
    def _manifest_asset(release: dict[str, Any]) -> dict[str, Any] | None:
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise CleanupError("GitHub Release assets must be an array")
        matches = [
            asset
            for asset in assets
            if isinstance(asset, dict) and asset.get("name") == MANIFEST_FILENAME
        ]
        if len(matches) > 1:
            raise CleanupError("GitHub Release has duplicate release manifest assets")
        return matches[0] if matches else None

    def _download_manifest(
        self, release: dict[str, Any], *, expected_tag: str
    ) -> dict[str, Any] | None:
        asset = self._manifest_asset(release)
        if asset is None:
            return None
        url = asset.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise CleanupError("release manifest asset has no API URL")
        raw = self._request("GET", url, accept="application/octet-stream")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CleanupError("release manifest asset is not valid JSON") from error
        return validate_manifest(value, expected_tag=expected_tag)

    def load_manifest_for_tag(self, tag: str) -> dict[str, Any]:
        releases = [
            release
            for release in self.list_releases()
            if release.get("tag_name") == tag
        ]
        manifests = [
            manifest
            for release in releases
            if (manifest := self._download_manifest(release, expected_tag=tag))
            is not None
        ]
        if not manifests:
            raise CleanupError(f"Tag {tag} has no exact schema-v6 release manifest")
        if any(manifest != manifests[0] for manifest in manifests[1:]):
            raise CleanupError(f"Tag {tag} has conflicting release manifests")
        return manifests[0]

    def list_manifest_records(self) -> list[ManifestRecord]:
        records: list[ManifestRecord] = []
        for release in self.list_releases():
            tag = release.get("tag_name")
            if not isinstance(tag, str) or not tag:
                continue
            try:
                manifest = self._download_manifest(release, expected_tag=tag)
            except RemoteError:
                raise
            except CleanupError:
                continue
            if manifest is None:
                continue
            created_at = release.get("created_at")
            release_id = release.get("id")
            draft = release.get("draft")
            prerelease = release.get("prerelease")
            if (
                not isinstance(created_at, str)
                or not created_at
                or not isinstance(release_id, int)
                or isinstance(release_id, bool)
                or release_id < 1
                or not isinstance(draft, bool)
                or not isinstance(prerelease, bool)
            ):
                continue
            records.append(
                ManifestRecord(
                    manifest,
                    created_at,
                    release_id,
                    draft,
                    prerelease,
                )
            )
        return records

    def _owner_package_prefix(self) -> str:
        if self._package_owner_prefix is not None:
            return self._package_owner_prefix
        owner = urllib.parse.quote(self.owner, safe="")
        value = self._github_json("GET", f"/users/{owner}")
        if not isinstance(value, dict) or value.get("type") not in {
            "Organization",
            "User",
        }:
            raise CleanupError("GitHub package owner type is invalid")
        prefix = "orgs" if value["type"] == "Organization" else "users"
        self._package_owner_prefix = f"/{prefix}/{owner}"
        return self._package_owner_prefix

    def _ghcr_version_state(
        self, reference: str, *, allowed_tags: Sequence[str]
    ) -> str | None:
        match = _OCI_REFERENCE.fullmatch(reference)
        if match is None or not reference.startswith("ghcr.io/"):
            raise CleanupError("GHCR resource reference is invalid")
        repository = match.group("repository").removeprefix("ghcr.io/")
        parts = repository.split("/")
        if len(parts) < 2 or parts[0].casefold() != self.owner.casefold():
            raise CleanupError("GHCR resource does not belong to the repository owner")
        package = urllib.parse.quote("/".join(parts[1:]), safe="")
        base = f"{self._owner_package_prefix()}/packages/container/{package}/versions"
        try:
            versions = self._all_pages(base)
        except RemoteError as error:
            if error.is_missing:
                return None
            raise
        target_tag = match.group("tag")
        matches: list[tuple[int, list[str]]] = []
        for version in versions:
            version_id = version.get("id")
            metadata = version.get("metadata")
            container = (
                metadata.get("container") if isinstance(metadata, dict) else None
            )
            tags = container.get("tags") if isinstance(container, dict) else None
            if not isinstance(tags, list) or any(
                not isinstance(tag, str) for tag in tags
            ):
                raise CleanupError("GHCR package version Tags are malformed")
            if target_tag in tags:
                if not isinstance(version_id, int) or isinstance(version_id, bool):
                    raise CleanupError("GHCR package version ID is malformed")
                matches.append((version_id, tags))
        if not matches:
            return None
        if len(matches) != 1:
            raise RemoteError(
                f"GHCR Tag {reference} resolves to multiple package versions",
                status=422,
            )
        version_id, tags = matches[0]
        allowed = set(allowed_tags)
        if target_tag not in allowed:
            raise CleanupError("GHCR resource allowed Tag set omits its target Tag")
        other_tags = sorted(set(tags) - allowed)
        if other_tags:
            raise UnsafePackageVersion(reference, other_tags)
        return f"{base}/{version_id}"

    @staticmethod
    def _crane_error(detail: str) -> RemoteError:
        normalized = " ".join(detail.casefold().split())
        if any(marker in normalized for marker in _MISSING_MARKERS):
            return RemoteError(f"Crane resource is absent: {detail}", status=404)
        if any(marker in normalized for marker in _TRANSPORT_MARKERS):
            return RemoteError(f"Crane transport error: {detail}")
        status_match = re.search(
            r"(?:http|status(?: code)?|response status)\D{0,8}([45][0-9]{2})",
            normalized,
        )
        if status_match is not None:
            status = int(status_match.group(1))
            return RemoteError(f"Crane HTTP {status}: {detail}", status=status)
        if "too many requests" in normalized:
            return RemoteError(f"Crane HTTP 429: {detail}", status=429)
        if "unauthorized" in normalized or "authentication required" in normalized:
            return RemoteError(f"Crane HTTP 401: {detail}", status=401)
        if "denied" in normalized or "forbidden" in normalized:
            return RemoteError(f"Crane HTTP 403: {detail}", status=403)
        return RemoteError(f"Crane permanent error: {detail}", status=422)

    def _run_crane(self, operation: str, reference: str) -> str:
        try:
            result = subprocess.run(
                [self.crane, operation, reference],
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired as error:
            raise RemoteError(f"Crane {operation} timed out for {reference}") from error
        except OSError as error:
            raise RemoteError(f"Crane {operation} transport error: {error}") from error
        if result.returncode != 0:
            detail = (
                result.stderr.strip() or result.stdout.strip() or str(result.returncode)
            )
            raise self._crane_error(detail)
        return result.stdout.strip()

    def _github_path_state(self, resource: Resource) -> str | None:
        if resource.kind == "actions-run":
            path = f"/repos/{self.repository}/actions/runs/{resource.identifier}"
        elif resource.kind == "git-tag":
            ref = urllib.parse.quote(f"tags/{resource.identifier}", safe="/")
            path = f"/repos/{self.repository}/git/ref/{ref}"
        elif resource.kind == "github-release":
            path = f"/repos/{self.repository}/releases/{resource.identifier}"
        else:
            raise CleanupError(f"unsupported GitHub resource kind: {resource.kind}")
        try:
            value = self._github_json("GET", path)
        except RemoteError as error:
            if error.is_missing:
                return None
            raise
        if resource.kind == "github-release" and (
            not isinstance(value, dict)
            or value.get("tag_name") != str(resource.reference).split("#", 1)[0]
        ):
            return None
        return path

    def probe(self, resource: Resource) -> object | None:
        if resource.kind in {"chart-oci", "ghcr-index", "ghcr-member"}:
            if not isinstance(resource.identifier, tuple) or any(
                not isinstance(tag, str) for tag in resource.identifier
            ):
                raise CleanupError("GHCR resource allowed Tag set is invalid")
            return self._ghcr_version_state(
                resource.reference, allowed_tags=resource.identifier
            )
        if resource.kind in {"dockerhub-index", "dockerhub-member"}:
            return self._run_crane("digest", resource.reference)
        return self._github_path_state(resource)

    def delete(self, resource: Resource, state: object) -> None:
        if resource.kind in {"chart-oci", "ghcr-index", "ghcr-member"}:
            if not isinstance(state, str):
                raise CleanupError("GHCR deletion state is invalid")
            self._github_json("DELETE", state)
            return
        if resource.kind in {"dockerhub-index", "dockerhub-member"}:
            if not isinstance(state, str) or _DIGEST.fullmatch(state) is None:
                raise CleanupError("DockerHub deletion state is not a manifest digest")
            match = _OCI_REFERENCE.fullmatch(resource.reference)
            if match is None:
                raise CleanupError("DockerHub deletion reference is invalid")
            self._run_crane("delete", f"{match.group('repository')}@{state}")
            return
        if not isinstance(state, str):
            raise CleanupError("GitHub deletion state is invalid")
        delete_path = state
        if resource.kind == "git-tag":
            ref = urllib.parse.quote(f"tags/{resource.identifier}", safe="/")
            delete_path = f"/repos/{self.repository}/git/refs/{ref}"
        self._github_json("DELETE", delete_path)

    def release_resources(self, tag: str) -> list[Resource]:
        resources: list[Resource] = []
        for release in self.list_releases():
            if release.get("tag_name") != tag:
                continue
            release_id = release.get("id")
            if not isinstance(release_id, int) or isinstance(release_id, bool):
                raise CleanupError("GitHub Release ID is malformed")
            resources.append(
                Resource(
                    "github-release",
                    f"{tag}#{release_id}",
                    release_id,
                    holds_manifest=self._manifest_asset(release) is not None,
                )
            )
        return sorted(resources, key=lambda item: int(item.identifier or 0))


def _boolean(value: str) -> bool:
    normalized = value.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _summary_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _add_remote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub owner/repository; defaults to GITHUB_REPOSITORY",
    )
    parser.add_argument("--crane", default="crane")
    parser.add_argument(
        "--api-base", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    parser.add_argument("--fail-resource")
    parser.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    tag = commands.add_parser("tag", help="clean one exact Tag")
    tag.add_argument("--tag", required=True)
    _add_remote_arguments(tag)

    retention = commands.add_parser(
        "retention", help="clean oldest excess same-type Tags"
    )
    retention.add_argument("--current-tag", required=True)
    retention.add_argument(
        "--release-type", choices=sorted(RELEASE_TYPES), required=True
    )
    retention.add_argument("--max-count", type=int, required=True)
    retention.add_argument("--pypi-enabled", type=_boolean, required=True)
    _add_remote_arguments(retention)
    return parser


def _production_remote(arguments: argparse.Namespace) -> ProductionRemote:
    repository = arguments.repository
    if not isinstance(repository, str) or not repository:
        raise CleanupError("--repository or GITHUB_REPOSITORY is required")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    return ProductionRemote(
        repository,
        token,
        crane=arguments.crane,
        api_base=arguments.api_base,
    )


def _run_tag(
    arguments: argparse.Namespace, remote: ProductionRemote
) -> list[ResourceFailure]:
    manifest = remote.load_manifest_for_tag(arguments.tag)
    report = cleanup_manifest(
        manifest,
        remote,
        fail_resource=arguments.fail_resource,
    )
    if report.completed:
        print(f"cleanup complete: {report.tag}")
    return list(report.failures)


def _run_retention(
    arguments: argparse.Namespace, remote: ProductionRemote
) -> list[ResourceFailure]:
    skip = select_retention_candidates(
        [],
        current_tag=arguments.current_tag,
        release_type=arguments.release_type,
        max_count=arguments.max_count,
        pypi_enabled=arguments.pypi_enabled,
    )
    if skip.skipped_reason is not None:
        print(skip.skipped_reason)
        return []
    selection = select_retention_candidates(
        remote.list_manifest_records(),
        current_tag=arguments.current_tag,
        release_type=arguments.release_type,
        max_count=arguments.max_count,
        pypi_enabled=arguments.pypi_enabled,
    )
    failures: list[ResourceFailure] = []
    for candidate in selection.candidates:
        report = cleanup_manifest(
            candidate.manifest,
            remote,
            fail_resource=arguments.fail_resource,
        )
        failures.extend(report.failures)
        if report.completed:
            print(f"retention cleanup complete: {report.tag}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        remote = _production_remote(arguments)
        failures = (
            _run_tag(arguments, remote)
            if arguments.command == "tag"
            else _run_retention(arguments, remote)
        )
        append_failure_summary(_summary_path(arguments.summary), failures)
        if failures:
            for failure in failures:
                print(
                    f"{failure.resource.kind} {failure.resource.reference}: "
                    f"{failure.final_error}",
                    file=sys.stderr,
                )
            return 1
        return 0
    except CleanupError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
