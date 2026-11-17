"""GitHub and registry operations used by cleanup discovery and execution."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from .cleanup_records import (
    OCI_REFERENCE,
    RELEASE_WORKFLOWS,
    ActiveRelease,
    CleanupError,
    Resource,
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


class ProductionRemote:
    """Own remote API behavior, resource mutation and exact deletion readback."""

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

    def request(
        self,
        method: str,
        path: str,
        *,
        accept: str = "application/vnd.github+json",
        data: bytes | None = None,
        content_type: str = "application/json",
    ) -> bytes:
        url = path if path.startswith("https://") else self.api_base + path
        parsed_url = urllib.parse.urlsplit(url)
        api_host = urllib.parse.urlsplit(self.api_base).netloc
        is_upload = (
            method == "POST"
            and data is not None
            and api_host == "api.github.com"
            and parsed_url.scheme == "https"
            and parsed_url.netloc == "uploads.github.com"
            and re.fullmatch(
                r"/repos/"
                + re.escape(self.repository)
                + r"/releases/[1-9][0-9]*/assets",
                parsed_url.path,
            )
            is not None
        )
        if parsed_url.netloc != api_host and not is_upload:
            raise CleanupError("GitHub API URL is outside the configured API host")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": accept,
                "Content-Type": content_type,
                "User-Agent": "ucm-release-cleanup",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        # Release assets and Actions archives redirect to blob storage.
        request.add_unredirected_header("Authorization", f"Bearer {self.token}")
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

    def json_request(self, method: str, path: str) -> Any:
        raw = self.request(method, path)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CleanupError("GitHub API returned malformed JSON") from error

    def list_pages(self, path: str, *, key: str | None = None) -> list[dict[str, Any]]:
        separator = "&" if "?" in path else "?"
        values: list[dict[str, Any]] = []
        for page in range(1, 1001):
            value = self.json_request(
                "GET", f"{path}{separator}per_page=100&page={page}"
            )
            if key is not None and isinstance(value, dict):
                value = value.get(key)
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
        return self.list_pages(f"/repos/{owner_repo}/releases")

    def _owner_package_prefix(self) -> str:
        if self._package_owner_prefix is not None:
            return self._package_owner_prefix
        owner = urllib.parse.quote(self.owner, safe="")
        value = self.json_request("GET", f"/users/{owner}")
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
        match = OCI_REFERENCE.fullmatch(reference)
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
                self._package_versions[base] = self.list_pages(base)
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
            current = self.json_request("GET", path)
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
            value = self.json_request("GET", path)
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
        """Return deletion state, or None when the exact resource is absent."""
        if resource.kind in {"chart-oci", "ghcr-index", "ghcr-member"}:
            if not isinstance(resource.identifier, tuple) or any(
                not isinstance(tag, str) for tag in resource.identifier
            ):
                raise CleanupError("GHCR resource allowed Tag set is invalid")
            return self._ghcr_version_state(
                resource.reference, allowed_tags=resource.identifier
            )
        if resource.kind in {"dockerhub-index", "dockerhub-member"}:
            try:
                return self._run_crane("digest", resource.reference)
            except RemoteError as error:
                if error.is_missing:
                    return None
                raise
        return self._github_path_state(resource)

    def delete(self, resource: Resource, state: object) -> None:
        if resource.kind in {"chart-oci", "ghcr-index", "ghcr-member"}:
            if not isinstance(state, str):
                raise CleanupError("GHCR deletion state is invalid")
            self.json_request("DELETE", state)
            return
        if resource.kind in {"dockerhub-index", "dockerhub-member"}:
            if not isinstance(state, str) or _DIGEST.fullmatch(state) is None:
                raise CleanupError("DockerHub deletion state is not a manifest digest")
            match = OCI_REFERENCE.fullmatch(resource.reference)
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
        self.json_request("DELETE", delete_path)

    def is_absent(self, resource: Resource, state: object) -> bool:
        """Read back the exact ID/digest used by DELETE; only 404 means absent."""
        try:
            if resource.kind in {"chart-oci", "ghcr-index", "ghcr-member"}:
                if not isinstance(state, str):
                    raise CleanupError("GHCR readback state is invalid")
                self.json_request("GET", state)
                return False
            if resource.kind in {"dockerhub-index", "dockerhub-member"}:
                if not isinstance(state, str) or _DIGEST.fullmatch(state) is None:
                    raise CleanupError(
                        "DockerHub readback state is not a manifest digest"
                    )
                match = OCI_REFERENCE.fullmatch(resource.reference)
                if match is None:
                    raise CleanupError("DockerHub readback reference is invalid")
                self._run_crane("digest", f"{match.group('repository')}@{state}")
                return False
            return self.probe(resource) is None
        except RemoteError as error:
            if error.is_missing:
                return True
            raise

    def read_release_run(self, run_id: int) -> dict[str, Any] | None:
        """Read an exact Actions run and verify UCM publication ownership."""
        if type(run_id) is not int or run_id < 1:
            raise CleanupError("cleanup Actions run ID must be positive")
        try:
            run = self.json_request(
                "GET", f"/repos/{self.repository}/actions/runs/{run_id}"
            )
        except RemoteError as error:
            if error.is_missing:
                return None
            raise
        if not isinstance(run, dict) or run.get("id") != run_id:
            raise CleanupError("Actions run response does not match its ID")
        repository = run.get("repository")
        path = run.get("path")
        if (
            not isinstance(repository, dict)
            or not isinstance(repository.get("full_name"), str)
            or repository["full_name"].casefold() != self.repository.casefold()
            or not isinstance(path, str)
            or path.split("@", 1)[0] not in RELEASE_WORKFLOWS
        ):
            raise CleanupError("Actions run is not owned by a UCM release workflow")
        return run

    def ensure_idle(self, run_ids: Sequence[int]) -> None:
        """Read publication state again before any destructive phase."""
        for run_id in sorted(set(run_ids)):
            run = self.read_release_run(run_id)
            if run is not None and run.get("status") != "completed":
                raise ActiveRelease(f"Tag is still used by Actions run {run_id}")
