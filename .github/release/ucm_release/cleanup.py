"""Retain and remove current UCM releases using the public manifest."""

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

if __package__:
    from . import nightly_cleanup
    from .manifest import RELEASE_MANIFEST_FILENAME as MANIFEST_FILENAME
    from .manifest import ManifestError as CleanupError
    from .manifest import validate_manifest
else:
    # Filename entry points also need the package parent for manifest imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ucm_release import nightly_cleanup
    from ucm_release.manifest import RELEASE_MANIFEST_FILENAME as MANIFEST_FILENAME
    from ucm_release.manifest import ManifestError as CleanupError
    from ucm_release.manifest import validate_manifest

RELEASE_TYPES = frozenset({"stable", "prerelease", "draft", "nightly"})
RETRY_DELAYS_SECONDS = (0.0, 5.0, 15.0)
_OCI_REFERENCE = re.compile(
    r"(?P<repository>(?:ghcr\.io|docker\.io)/"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+)"
    r":(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})"
)
_REPOSITORY = re.compile(
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<repo>[A-Za-z0-9_.-]+)"
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PATH_COMPONENT = re.compile(r"[a-z0-9][a-z0-9.+-]*")
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


class _GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Signed artifact redirects must not forward the GitHub credential."""

    def redirect_request(self, request, response, code, message, headers, new_url):
        if urllib.parse.urlsplit(new_url).scheme != "https":
            raise CleanupError("GitHub download redirected to a non-HTTPS URL")
        redirected = super().redirect_request(
            request, response, code, message, headers, new_url
        )
        if redirected is not None and (
            urllib.parse.urlsplit(request.full_url).netloc
            != urllib.parse.urlsplit(new_url).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


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

    def is_absent(self, resource: Resource, state: object) -> bool: ...

    def recovery_release_ids(self, record: dict[str, Any]) -> set[int]: ...

    def release_resources(self, tag: str) -> list[Resource]: ...


def registry_resources(manifest: object) -> list[Resource]:
    """Project phase-one resources in the required deletion order."""
    validated = validate_manifest(manifest)
    images = {
        channel: {"indexes": set(), "members": set()}
        for channel in ("ghcr", "dockerhub")
    }
    for image in validated["images"]:
        for channel, publication in image["publications"].items():
            if publication is None:
                continue
            kind = "indexes" if publication["multi_arch"] else "members"
            images[channel][kind].add(publication["pull"])
            images[channel]["members"].update(
                member["reference"] for member in publication["members"]
            )
    ghcr_resources: list[tuple[str, str]] = []
    if validated["chart"] is not None and validated["chart"]["oci"] is not None:
        ghcr_resources.append(("chart-oci", validated["chart"]["oci"]))
    ghcr_resources.extend(
        ("ghcr-index", ref) for ref in sorted(images["ghcr"]["indexes"])
    )
    ghcr_resources.extend(
        ("ghcr-member", ref) for ref in sorted(images["ghcr"]["members"])
    )

    references = ghcr_resources + [
        (f"dockerhub-{kind}", reference)
        for plural, kind in (("indexes", "index"), ("members", "member"))
        for reference in sorted(images["dockerhub"][plural])
    ]
    return _registry_resources_from_references(references)


def _registry_resources_from_references(
    references: Sequence[tuple[str, str]],
) -> list[Resource]:
    """Bind exact references to the allowed tags of each GHCR package."""
    allowed_tags_by_package: dict[str, set[str]] = {}
    for kind, reference in references:
        if kind.startswith("dockerhub-"):
            continue
        match = _OCI_REFERENCE.fullmatch(reference)
        if match is None:
            raise AssertionError("validated GHCR reference no longer parses")
        allowed_tags_by_package.setdefault(match.group("repository"), set()).add(
            match.group("tag")
        )
    result = []
    for kind, reference in references:
        if kind.startswith("dockerhub-"):
            result.append(Resource(kind, reference))
            continue
        match = _OCI_REFERENCE.fullmatch(reference)
        if match is None:
            raise AssertionError("validated GHCR reference no longer parses")
        allowed_tags = tuple(sorted(allowed_tags_by_package[match.group("repository")]))
        result.append(Resource(kind, reference, allowed_tags))

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
    normalized_by_record: dict[int, dict[str, Any]] = {}
    for record in records:
        try:
            manifest = validate_manifest(record.manifest)
        except CleanupError:
            continue
        if (
            manifest["release"]["type"] != release_type
            or manifest["release"]["tag"] == current_tag
        ):
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
        normalized_by_record[id(record)] = manifest
        grouped.setdefault(manifest["release"]["tag"], []).append(record)

    unique_records: list[ManifestRecord] = []
    for tag_records in grouped.values():
        first = tag_records[0].manifest
        if any(record.manifest != first for record in tag_records[1:]):
            continue
        unique_records.append(
            min(tag_records, key=lambda item: (item.created_at, item.release_id))
        )
    unique_records.sort(
        key=lambda item: (
            item.created_at,
            item.release_id,
            normalized_by_record[id(item)]["release"]["tag"],
        )
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
    """Probe and delete a resource, confirming Release deletion by ID readback."""
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
            try:
                remote.delete(resource, state)
            except RemoteError as error:
                if not error.is_missing:
                    raise
            if not remote.is_absent(resource, state):
                raise RemoteError(
                    f"{resource.kind} still exists after deletion", status=409
                )
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


def _cleanup_phases(
    tag: str,
    remote: CleanupRemote,
    registry: Sequence[Resource],
    actions: Sequence[Resource],
    releases: Sequence[Resource],
    *,
    sleeper,
    fail_resource: str | None,
) -> CleanupReport:
    """Remove one target while retaining its last recovery record until the end."""
    early_actions = [item for item in actions if not item.holds_manifest]
    for phase, resources in enumerate(
        (registry, early_actions, [Resource("git-tag", tag, tag)]), start=1
    ):
        failures = _run_phase(
            remote, resources, sleeper=sleeper, fail_resource=fail_resource
        )
        if failures:
            return CleanupReport(tag, False, phase, tuple(failures))
    unbacked = [item for item in releases if not item.holds_manifest]
    failures = _run_phase(
        remote, unbacked, sleeper=sleeper, fail_resource=fail_resource
    )
    if failures:
        return CleanupReport(tag, False, 4, tuple(failures))
    holders = [item for item in releases if item.holds_manifest]
    holders.extend(item for item in actions if item.holds_manifest)
    failures = _run_phase(remote, holders, sleeper=sleeper, fail_resource=fail_resource)
    return CleanupReport(tag, not failures, 4 if failures else None, tuple(failures))


def cleanup_manifest(
    manifest: object,
    remote: CleanupRemote,
    *,
    release_id: int | None = None,
    sleeper=time.sleep,
    fail_resource: str | None = None,
) -> CleanupReport:
    """Resolve stable IDs, then remove a published release in four phases."""
    validated = validate_manifest(manifest)
    tag = validated["release"]["tag"]
    releases, failure = _release_resources_with_retry(remote, tag, sleeper=sleeper)
    if failure is not None:
        return CleanupReport(tag, False, 4, (failure,))
    if release_id is not None:
        releases = [item for item in releases if item.identifier != release_id]
        releases.append(
            Resource("github-release", f"{tag}#{release_id}", release_id, True)
        )
    if not releases:
        failure = _failure(
            Resource("github-releases", tag),
            1,
            CleanupError("cannot resolve GitHub Release IDs before deleting the Tag"),
        )
        return CleanupReport(tag, False, 4, (failure,))
    run_id = validated["release"]["actions_run_id"]
    actions = [
        Resource(
            "actions-run",
            f"https://github.com/{remote.repository}/actions/runs/{run_id}",
            run_id,
        )
    ]
    return _cleanup_phases(
        tag,
        remote,
        registry_resources(validated),
        actions,
        releases,
        sleeper=sleeper,
        fail_resource=fail_resource,
    )


def cleanup_record(
    record: object,
    remote: CleanupRemote,
    *,
    sleeper=time.sleep,
    fail_resource: str | None = None,
) -> CleanupReport:
    """Delete a proven Nightly target without requiring a successful Manifest."""
    validated = nightly_cleanup.validate_record(record, remote.repository)
    tag = validated["tag"]
    registry = _registry_resources_from_references(
        [(item["kind"], item["reference"]) for item in validated["resources"]]
    )
    holders = remote.recovery_release_ids(validated)
    releases = [
        Resource(
            "github-release", f"{tag}#{identifier}", identifier, identifier in holders
        )
        for identifier in validated["release_ids"]
    ]
    # A failed attachment update can leave the full inventory only in the run.
    # Keep that run until the Tag and unbacked Releases are successfully removed.
    actions = [
        Resource(
            "actions-run",
            f"https://github.com/{remote.repository}/actions/runs/{identifier}",
            identifier,
            holds_manifest=not bool(holders),
        )
        for identifier in sorted({run["id"] for run in validated["runs"]})
    ]
    return _cleanup_phases(
        tag,
        remote,
        registry,
        actions,
        releases,
        sleeper=sleeper,
        fail_resource=fail_resource,
    )


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
        self._opener = opener or urllib.request.build_opener(_GitHubRedirectHandler())
        self._package_owner_prefix: str | None = None
        self._package_versions: dict[str, list[dict[str, Any]]] = {}
        self._package_denied: dict[str, RemoteError] = {}

    def _request(
        self,
        method: str,
        path: str,
        *,
        accept: str = "application/vnd.github+json",
    ) -> bytes:
        url = path if path.startswith("https://") else self.api_base + path
        if (
            urllib.parse.urlsplit(url).netloc
            != urllib.parse.urlsplit(self.api_base).netloc
        ):
            raise CleanupError("GitHub API URL is outside the configured API host")
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
            raise CleanupError(f"Tag {tag} has no exact schema 9 manifest")
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
        if base in self._package_denied:
            raise self._package_denied[base]
        try:
            if base not in self._package_versions:
                self._package_versions[base] = self._all_pages(base)
            versions = self._package_versions[base]
        except RemoteError as error:
            if error.is_missing:
                return None
            if error.status in {401, 403}:
                self._package_denied[base] = error
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
        # The package listing is shared by hundreds of historical references;
        # verify the selected version's current tags before every deletion.
        path = f"{base}/{version_id}"
        try:
            current = self._github_json("GET", path)
        except RemoteError as error:
            if error.is_missing:
                self._package_versions[base] = [
                    item for item in versions if item.get("id") != version_id
                ]
                return None
            raise
        if not isinstance(current, dict) or current.get("id") != version_id:
            raise CleanupError("GHCR package version response does not match its ID")
        metadata = current.get("metadata", {})
        container = metadata.get("container", {}) if isinstance(metadata, dict) else {}
        tags = container.get("tags") if isinstance(container, dict) else None
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise CleanupError("GHCR package version Tags are malformed")
        if target_tag not in tags:
            self._package_versions.pop(base, None)
            raise RemoteError(
                "GHCR Tag changed during cleanup; inventory must be refreshed",
                status=409,
            )
        allowed = set(allowed_tags)
        if target_tag not in allowed:
            raise CleanupError("GHCR resource allowed Tag set omits its target Tag")
        other_tags = sorted(set(tags) - allowed)
        if other_tags:
            raise UnsafePackageVersion(reference, other_tags)
        return path

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
            not isinstance(value, dict) or value.get("id") != resource.identifier
        ):
            raise CleanupError("GitHub Release response does not match its ID")
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

    def is_absent(self, resource: Resource, state: object) -> bool:
        """Read back the exact ID/digest used by DELETE; only 404 means absent."""
        try:
            if resource.kind in {"chart-oci", "ghcr-index", "ghcr-member"}:
                if not isinstance(state, str):
                    raise CleanupError("GHCR readback state is invalid")
                self._github_json("GET", state)
                return False
            return self.probe(resource) is None
        except RemoteError as error:
            if error.is_missing:
                return True
            raise

    def recovery_release_ids(self, record: dict[str, Any]) -> set[int]:
        """Find live Release attachments covering the complete recovered target."""
        expected_resources = {
            (item["kind"], item["reference"]) for item in record["resources"]
        }
        expected_runs = {(item["id"], item["attempt"]) for item in record["runs"]}
        holders: set[int] = set()
        for identifier in record["release_ids"]:
            try:
                release = self._github_json(
                    "GET", f"/repos/{self.repository}/releases/{identifier}"
                )
            except RemoteError as error:
                if error.is_missing:
                    continue
                raise
            if not isinstance(release, dict) or release.get("id") != identifier:
                raise CleanupError("recovery Release response does not match its ID")
            assets = release.get("assets")
            if not isinstance(assets, list):
                raise CleanupError("recovery Release assets are malformed")
            for asset in assets:
                if not isinstance(asset, dict) or asset.get("name") not in {
                    nightly_cleanup.CLEANUP_FILENAME,
                    MANIFEST_FILENAME,
                }:
                    continue
                asset_id = asset.get("id")
                if type(asset_id) is not int or asset_id < 1:
                    raise CleanupError("recovery asset ID is malformed")
                raw = self._request(
                    "GET",
                    f"/repos/{self.repository}/releases/assets/{asset_id}",
                    accept="application/octet-stream",
                )
                try:
                    value = json.loads(raw)
                    saved = (
                        nightly_cleanup.validate_record(value, self.repository)
                        if asset["name"] == nightly_cleanup.CLEANUP_FILENAME
                        else nightly_cleanup.record_from_manifest(
                            value, self.repository, identifier, record["source_sha"]
                        )
                    )
                except (UnicodeError, json.JSONDecodeError, CleanupError) as error:
                    raise CleanupError(
                        f"recovery asset cannot be validated: {error}"
                    ) from error
                saved_resources = {
                    (item["kind"], item["reference"]) for item in saved["resources"]
                }
                saved_runs = {(item["id"], item["attempt"]) for item in saved["runs"]}
                if (
                    saved["tag"] == record["tag"]
                    and saved["source_sha"] == record["source_sha"]
                    and identifier in saved["release_ids"]
                    and expected_resources <= saved_resources
                    and expected_runs <= saved_runs
                ):
                    holders.add(identifier)
        return holders

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
    parser.add_argument(
        "--dry-run", action="store_true", help="read-only cleanup preview"
    )
    parser.add_argument(
        "--inventory", type=Path, help="reviewed historical Nightly inventory"
    )
    parser.add_argument("--report", type=Path, help="write per-version JSON results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser(
        "record", help="persist an unfinished Nightly cleanup target"
    )
    record.add_argument("--repository", required=True)
    record.add_argument("--tag", required=True)
    record.add_argument("--source-sha", required=True)
    record.add_argument("--run-id", type=int, required=True)
    record.add_argument("--run-attempt", type=int, required=True)
    record.add_argument("--release-id", type=int)
    record.add_argument("--previous", type=Path)
    record.add_argument("--plan", type=Path)
    record.add_argument("--output", type=Path, required=True)

    tag = commands.add_parser("tag", help="clean one exact Tag")
    tag.add_argument("--tag", required=True)
    _add_remote_arguments(tag)

    retention = commands.add_parser(
        "retention", help="clean oldest excess same-type Tags"
    )
    retention.add_argument("--current-tag", help="required for non-Nightly retention")
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


def _read_document(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanupError(f"cannot read cleanup input {path}: {error}") from error


def _write_document(path: Path | None, value: object) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run_record(arguments: argparse.Namespace) -> None:
    record = nightly_cleanup.record_for_tag(
        arguments.repository,
        arguments.tag,
        arguments.source_sha,
        arguments.run_id,
        arguments.run_attempt,
        arguments.release_id,
    )
    if arguments.previous is not None:
        previous = nightly_cleanup.validate_record(
            _read_document(arguments.previous), arguments.repository
        )
        if previous["tag"] != arguments.tag:
            raise CleanupError("previous cleanup record belongs to another Tag")
        record = nightly_cleanup.merge_records(
            [previous, record], arguments.repository
        )[0]
    if arguments.plan is not None:
        record = nightly_cleanup.record_from_plan(
            record, _read_document(arguments.plan)
        )
    _write_document(arguments.output, record)
    print(json.dumps(record, sort_keys=True))


def _nightly_catalog(arguments: argparse.Namespace, remote: ProductionRemote):
    if __package__:
        from .nightly_catalog import collect_catalog
    else:
        from ucm_release.nightly_catalog import collect_catalog
    path = getattr(arguments, "inventory", None)
    history = _read_document(path) if path is not None else None
    return collect_catalog(remote, history)


def _record_preview(record: dict[str, Any], remote: CleanupRemote) -> None:
    for resource in _registry_resources_from_references(
        [(item["kind"], item["reference"]) for item in record["resources"]]
    ):
        remote.probe(resource)


def _record_result(record: dict[str, Any], status: str, reason: str | None = None):
    result = {
        "tag": record["tag"],
        "status": status,
        "release_ids": record["release_ids"],
        "run_ids": sorted({item["id"] for item in record["runs"]}),
        "registry_references": len(record["resources"]),
    }
    if reason is not None:
        result["reason"] = reason
    return result


def _run_nightly_targets(
    arguments: argparse.Namespace,
    remote: ProductionRemote,
    targets: Sequence[dict[str, Any]],
    *,
    blocked: dict[str, str],
    kept: Sequence[dict[str, Any]] = (),
    deferred: Sequence[dict[str, Any]] = (),
) -> list[ResourceFailure]:
    failures: list[ResourceFailure] = []
    results = [_record_result(item, "kept", blocked.get(item["tag"])) for item in kept]
    for item in deferred:
        reason = blocked.get(item["tag"])
        if reason is not None:
            failures.append(
                _failure(
                    Resource("nightly-version", item["tag"]), 1, CleanupError(reason)
                )
            )
        results.append(
            _record_result(
                item,
                "blocked" if reason else "deferred",
                reason or "publication run is active or queued",
            )
        )
    dry_run = getattr(arguments, "dry_run", False)
    for record in targets:
        reason = blocked.get(record["tag"])
        if reason is not None:
            failure = _failure(
                Resource("nightly-version", record["tag"]), 1, CleanupError(reason)
            )
            failures.append(failure)
            results.append(_record_result(record, "blocked", reason))
            continue
        if dry_run:
            try:
                _record_preview(record, remote)
            except CleanupError as error:
                failures.append(
                    _failure(Resource("nightly-version", record["tag"]), 1, error)
                )
                results.append(_record_result(record, "blocked", str(error)))
            else:
                results.append(_record_result(record, "would-delete"))
            continue
        try:
            report = cleanup_record(
                record, remote, fail_resource=arguments.fail_resource
            )
        except CleanupError as error:
            failures.append(
                _failure(Resource("nightly-version", record["tag"]), 1, error)
            )
            results.append(_record_result(record, "blocked", str(error)))
            continue
        failures.extend(report.failures)
        results.append(
            _record_result(record, "deleted" if report.completed else "blocked")
        )
        if report.failures:
            results[-1]["failures"] = [
                {
                    "kind": item.resource.kind,
                    "reference": item.resource.reference,
                    "attempts": item.attempts,
                    "error": item.final_error,
                }
                for item in report.failures
            ]
    summary = {
        "repository": remote.repository,
        "max_count": getattr(arguments, "max_count", None),
        "status": (
            "failed"
            if failures
            else "dry-run" if dry_run else "deferred" if deferred else "complete"
        ),
        "results": results,
    }
    _write_document(getattr(arguments, "report", None), summary)
    for result in results:
        print(
            f"nightly {result['status']}: {result['tag']}"
            + (f" - {result['reason']}" if "reason" in result else "")
        )
    return failures


def _run_nightly_retention(
    arguments: argparse.Namespace, remote: ProductionRemote
) -> list[ResourceFailure]:
    if arguments.max_count == -1 or arguments.pypi_enabled:
        print(
            "Nightly retention skipped: unlimited count or immutable PyPI publication"
        )
        _write_document(
            getattr(arguments, "report", None),
            {"repository": remote.repository, "status": "skipped", "results": []},
        )
        return []
    nightly_cleanup.select_retention([], arguments.max_count, set())
    catalog = _nightly_catalog(arguments, remote)
    selection = nightly_cleanup.select_retention(
        catalog.records, arguments.max_count, catalog.active_tags
    )
    orphan_candidates = [
        item for item in catalog.orphans if item["tag"] not in catalog.active_tags
    ]
    orphan_deferred = [
        item for item in catalog.orphans if item["tag"] in catalog.active_tags
    ]
    return _run_nightly_targets(
        arguments,
        remote,
        (*selection.candidates, *orphan_candidates),
        blocked=catalog.blocked,
        kept=selection.kept,
        deferred=(*selection.deferred, *orphan_deferred),
    )


def _run_tag(
    arguments: argparse.Namespace, remote: ProductionRemote
) -> list[ResourceFailure]:
    if arguments.tag.startswith("nightly/"):
        catalog = _nightly_catalog(arguments, remote)
        targets = [
            item
            for item in (*catalog.records, *catalog.orphans)
            if item["tag"] == arguments.tag
        ]
        if not targets:
            raise CleanupError(f"Nightly {arguments.tag} has no proven cleanup target")
        if arguments.tag in catalog.active_tags:
            return _run_nightly_targets(
                arguments, remote, [], blocked=catalog.blocked, deferred=targets
            )
        return _run_nightly_targets(arguments, remote, targets, blocked=catalog.blocked)
    manifest = remote.load_manifest_for_tag(arguments.tag)
    if getattr(arguments, "dry_run", False):
        print(f"would-delete: {arguments.tag}")
        return []
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
    if arguments.release_type == "nightly":
        return _run_nightly_retention(arguments, remote)
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
        if getattr(arguments, "dry_run", False):
            print(f"would-delete: {candidate.manifest['release']['tag']}")
            continue
        report = cleanup_manifest(
            candidate.manifest,
            remote,
            release_id=candidate.release_id,
            fail_resource=arguments.fail_resource,
        )
        failures.extend(report.failures)
        if report.completed:
            print(f"retention cleanup complete: {report.tag}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "record":
            _run_record(arguments)
            return 0
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
        _write_document(
            getattr(arguments, "report", None),
            {"status": "failed", "error": str(error)},
        )
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
