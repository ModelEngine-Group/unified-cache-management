"""Discover every existing Nightly and bind trusted cleanup evidence to it.

The catalog includes unfinished Releases and Tag-only publications. Missing
evidence blocks deletion of a version instead of making it invisible to quota
selection. Registry transport and all mutations stay in the cleanup entrypoint.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from . import nightly_cleanup as domain
from . import version_config

_SOURCE_WORKFLOWS = {
    ".github/workflows/release-nightly.yml": {"schedule", "workflow_dispatch"},
    ".github/workflows/release-tag.yml": {"push"},
}
_ARTIFACT = re.compile(
    r"ucm-nightly-cleanup(?:-(?:open|plan))?-run-(?P<run>[1-9][0-9]*)"
    r"-attempt-(?P<attempt>[1-9][0-9]*)"
)
_SHA = re.compile(r"[0-9a-f]{40}")
_OWNED_STATUS = re.compile(
    r"^Status: `(release-open|artifacts-ready|artifacts-failed|images-failed|publication-failed)`",
    re.MULTILINE,
)


class CatalogRemote(Protocol):
    repository: str
    api_base: str

    def _github_json(self, method: str, path: str) -> Any: ...

    def _all_pages(self, path: str) -> list[dict[str, Any]]: ...

    def _request(
        self, method: str, path: str, *, accept: str = "application/vnd.github+json"
    ) -> bytes: ...


@dataclass(frozen=True)
class NightlyCatalog:
    records: tuple[dict[str, Any], ...]
    orphans: tuple[dict[str, Any], ...]
    active_tags: frozenset[str]
    blocked: dict[str, str]


def _tag(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or version_config.NIGHTLY_TAG.fullmatch(value) is None
    ):
        return None
    try:
        version_config.classify_tag(value)
    except ValueError:
        return None
    return value


def _positive(value: object, context: str) -> int:
    if type(value) is not int or value < 1:
        raise domain.CleanupRecordError(f"{context} must be a positive integer")
    return value


def _missing(error: Exception) -> bool:
    return getattr(error, "status", getattr(error, "code", None)) == 404


def _owned_release(release: dict[str, Any]) -> bool:
    body = release.get("body") or ""
    return isinstance(body, str) and (
        "<!-- ucm-release:begin -->" in body or _OWNED_STATUS.search(body) is not None
    )


def _release_tag(release: dict[str, Any]) -> str | None:
    direct = _tag(release.get("tag_name"))
    if direct is not None:
        return direct
    if (
        release.get("draft") is True
        and isinstance(release.get("tag_name"), str)
        and release["tag_name"].startswith("untagged-")
        and _owned_release(release)
    ):
        return _tag(release.get("name"))
    return None


class _Collector:
    def __init__(self, remote: CatalogRemote) -> None:
        self.remote = remote
        self.repository = remote.repository
        self.prefix = "/repos/" + quote(self.repository, safe="/")
        self.records: dict[str, dict[str, Any]] = {}
        self.existing: set[str] = set()
        self.proven: set[str] = set()
        self.historical: set[str] = set()
        self.active: set[str] = set()
        self.blocked: dict[str, str] = {}
        self.releases: dict[int, dict[str, Any]] = {}
        self.runs: dict[int, dict[str, Any] | None] = {}

    def block(self, tag: str, reason: object) -> None:
        message = str(reason)
        previous = self.blocked.get(tag)
        if previous is None:
            self.blocked[tag] = message
        elif message not in previous:
            self.blocked[tag] = previous + "; " + message

    def add(self, record: object, *, proven: bool = False) -> None:
        validated = domain.validate_record(record, self.repository)
        tag = validated["tag"]
        previous = self.records.get(tag)
        try:
            merged = domain.merge_records(
                [validated] if previous is None else [previous, validated],
                self.repository,
            )[0]
        except domain.CleanupRecordError as error:
            self.block(tag, error)
            return
        self.records[tag] = merged
        if proven:
            self.proven.add(tag)

    def basic(
        self, tag: str, *, release_id: int | None = None, source_sha: str | None = None
    ) -> None:
        self.add(
            {
                "kind": domain.CLEANUP_KIND,
                "schema_version": domain.CLEANUP_SCHEMA_VERSION,
                "repository": self.repository,
                "tag": tag,
                "source_sha": source_sha,
                "release_ids": [] if release_id is None else [release_id],
                "runs": [],
                "resources": [],
            }
        )

    def envelope(self, path: str, key: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        separator = "&" if "?" in path else "?"
        for page in range(1, 1001):
            value = self.remote._github_json(
                "GET", f"{path}{separator}per_page=100&page={page}"
            )
            items = value.get(key) if isinstance(value, dict) else None
            if not isinstance(items, list) or any(
                not isinstance(item, dict) for item in items
            ):
                raise domain.CleanupRecordError(
                    f"GitHub {key} response must contain an array"
                )
            result.extend(items)
            if len(items) < 100:
                return result
        raise domain.CleanupRecordError(f"GitHub {key} pagination exceeded 1000 pages")

    def asset(self, asset: dict[str, Any]) -> object:
        asset_id = _positive(asset.get("id"), "Release asset ID")
        raw = self.remote._request(
            "GET",
            f"{self.prefix}/releases/assets/{asset_id}",
            accept="application/octet-stream",
        )
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise domain.CleanupRecordError(
                "cleanup evidence asset is not valid JSON"
            ) from error

    def discover_releases(self) -> None:
        releases = self.remote._all_pages(f"{self.prefix}/releases")
        for release in releases:
            release_id = _positive(release.get("id"), "Release ID")
            self.releases[release_id] = release
        for release_id, release in self.releases.items():
            direct = _tag(release.get("tag_name"))
            inferred = _release_tag(release)
            orphan = (
                release.get("draft") is True
                and isinstance(release.get("tag_name"), str)
                and release["tag_name"].startswith("untagged-")
            )
            if direct is None and not orphan:
                continue
            assets = release.get("assets")
            if not isinstance(assets, list) or any(
                not isinstance(asset, dict) for asset in assets
            ):
                if inferred:
                    self.existing.add(inferred)
                    self.basic(inferred, release_id=release_id)
                    self.block(inferred, "Release assets are not an array")
                continue
            evidence: list[dict[str, Any]] = []
            errors: list[str] = []
            for name in (domain.CLEANUP_FILENAME, "release-manifest.json"):
                matches = [asset for asset in assets if asset.get("name") == name]
                if len(matches) > 1:
                    errors.append(f"duplicate {name} assets")
                    continue
                if not matches:
                    continue
                try:
                    value = self.asset(matches[0])
                    if name == domain.CLEANUP_FILENAME:
                        record = domain.validate_record(value, self.repository)
                        if release_id not in record["release_ids"]:
                            raise domain.CleanupRecordError(
                                "cleanup asset does not bind its live Release ID"
                            )
                        if record["source_sha"] is None or not record["runs"]:
                            raise domain.CleanupRecordError(
                                "producer cleanup asset has no source SHA or run binding"
                            )
                    else:
                        record = domain.record_from_manifest(
                            value, self.repository, release_id
                        )
                    evidence.append(record)
                except (domain.CleanupRecordError, OSError) as error:
                    errors.append(str(error))
            identities = {record["tag"] for record in evidence}
            if inferred:
                identities.add(inferred)
            if direct:
                identities.add(direct)
            name_tag = _tag(release.get("name"))
            if name_tag and (direct or evidence or inferred):
                identities.add(name_tag)
            if not identities:
                # Ordinary user Drafts are not reclassified by a Nightly-like name.
                continue
            if inferred:
                tag = inferred
            elif len(identities) == 1:
                tag = next(iter(identities))
            else:
                raise domain.CleanupRecordError(
                    f"Release ID {release_id} has no unambiguous original Nightly identity"
                )
            self.existing.add(tag)
            self.basic(tag, release_id=release_id)
            if len(identities) > 1:
                self.block(
                    tag,
                    "Release Tag, name, and cleanup evidence have conflicting identities",
                )
            for error in errors:
                self.block(tag, error)
            if len(identities) == 1:
                for record in evidence:
                    self.add(record, proven=True)

    def discover_refs(self) -> None:
        refs = self.remote._github_json(
            "GET", f"{self.prefix}/git/matching-refs/tags/nightly/"
        )
        if not isinstance(refs, list) or any(not isinstance(ref, dict) for ref in refs):
            raise domain.CleanupRecordError("GitHub matching refs must be an array")
        for ref in refs:
            raw_name = ref.get("ref")
            tag = _tag(
                raw_name.removeprefix("refs/tags/")
                if isinstance(raw_name, str)
                else None
            )
            if tag is None:
                continue
            self.existing.add(tag)
            self.basic(tag)
            try:
                target = ref.get("object")
                seen: set[str] = set()
                while isinstance(target, dict) and target.get("type") == "tag":
                    sha = target.get("sha")
                    if (
                        not isinstance(sha, str)
                        or _SHA.fullmatch(sha) is None
                        or sha in seen
                        or len(seen) >= 5
                    ):
                        raise domain.CleanupRecordError(
                            "annotated Nightly Tag cannot resolve to one commit"
                        )
                    seen.add(sha)
                    resolved = self.remote._github_json(
                        "GET", f"{self.prefix}/git/tags/{sha}"
                    )
                    target = (
                        resolved.get("object") if isinstance(resolved, dict) else None
                    )
                if (
                    not isinstance(target, dict)
                    or target.get("type") != "commit"
                    or not isinstance(target.get("sha"), str)
                    or _SHA.fullmatch(target["sha"]) is None
                ):
                    raise domain.CleanupRecordError(
                        "Nightly Tag does not point to a commit"
                    )
                self.basic(tag, source_sha=target["sha"])
            except (domain.CleanupRecordError, OSError) as error:
                self.block(tag, error)

    def history(self, document: object) -> None:
        if document is None:
            return
        if (
            not isinstance(document, dict)
            or set(document)
            != {
                "kind",
                "schema_version",
                "repository",
                "targets",
                "evidence",
                "blocked",
            }
            or document["kind"] != "ucm-nightly-cleanup-inventory"
            or type(document["schema_version"]) is not int
            or document["schema_version"] != 1
            or not isinstance(document["repository"], str)
            or document["repository"].casefold() != self.repository.casefold()
            or not isinstance(document["targets"], list)
            or not isinstance(document["evidence"], dict)
            or not isinstance(document["blocked"], dict)
        ):
            raise domain.CleanupRecordError(
                "historical cleanup inventory identity or fields are invalid"
            )
        for tag, reason in document["blocked"].items():
            if _tag(tag) is None or not isinstance(reason, str) or not reason:
                raise domain.CleanupRecordError(
                    "historical cleanup blocker must bind one Nightly"
                )
            self.basic(tag)
            self.block(tag, reason)
        for record in document["targets"]:
            self.historical.add(domain.validate_record(record, self.repository)["tag"])
            self.add(record, proven=True)

    def trusted_run(self, run: object) -> bool:
        if not isinstance(run, dict):
            return False
        path = run.get("path")
        if not isinstance(path, str):
            return False
        path = path.split("@", 1)[0]
        if (
            path not in _SOURCE_WORKFLOWS
            or run.get("event") not in _SOURCE_WORKFLOWS[path]
        ):
            return False
        head_repository = run.get("head_repository")
        if (
            not isinstance(head_repository, dict)
            or not isinstance(head_repository.get("full_name"), str)
            or head_repository["full_name"].casefold() != self.repository.casefold()
        ):
            return False
        repository = run.get("repository")
        if repository is not None and (
            not isinstance(repository, dict)
            or not isinstance(repository.get("full_name"), str)
            or repository["full_name"].casefold() != self.repository.casefold()
        ):
            return False
        return (
            isinstance(run.get("head_sha"), str)
            and _SHA.fullmatch(run["head_sha"]) is not None
        )

    def artifact(self, artifact: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
        match = _ARTIFACT.fullmatch(artifact.get("name", ""))
        assert match is not None
        run_id = _positive(run.get("id"), "source run ID")
        attempt = int(match.group("attempt"))
        if int(match.group("run")) != run_id or attempt > _positive(
            run.get("run_attempt"), "source run attempt"
        ):
            raise domain.CleanupRecordError(
                "cleanup artifact name differs from its source run"
            )
        artifact_run = artifact.get("workflow_run")
        if artifact_run is not None and (
            not isinstance(artifact_run, dict) or artifact_run.get("id") != run_id
        ):
            raise domain.CleanupRecordError(
                "cleanup artifact metadata differs from its source run"
            )
        artifact_id = _positive(artifact.get("id"), "cleanup artifact ID")
        raw = self.remote._request(
            "GET",
            f"{self.prefix}/actions/artifacts/{artifact_id}/zip",
            accept="application/octet-stream",
        )
        if len(raw) > 8 * 1024 * 1024:
            raise domain.CleanupRecordError(
                "cleanup artifact exceeds its inventory size limit"
            )
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                if (
                    len(entries) != 1
                    or entries[0].filename != domain.CLEANUP_FILENAME
                    or entries[0].file_size > 2 * 1024 * 1024
                ):
                    raise domain.CleanupRecordError(
                        "cleanup artifact must contain only release-cleanup.json"
                    )
                value = json.loads(archive.read(entries[0]))
        except (
            zipfile.BadZipFile,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RuntimeError,
        ) as error:
            raise domain.CleanupRecordError(
                "cleanup artifact is not a readable inventory ZIP"
            ) from error
        record = domain.validate_record(value, self.repository)
        if (
            record["source_sha"] != run["head_sha"]
            or {"id": run_id, "attempt": attempt} not in record["runs"]
        ):
            raise domain.CleanupRecordError(
                "cleanup artifact does not bind its source SHA and attempt"
            )
        if any(
            item["attempt"] > run["run_attempt"]
            for item in record["runs"]
            if item["id"] == run_id
        ):
            raise domain.CleanupRecordError(
                "cleanup artifact contains a future source run attempt"
            )
        if run["path"].split("@", 1)[
            0
        ] == ".github/workflows/release-tag.yml" and record["tag"] != _tag(
            run.get("head_branch")
        ):
            raise domain.CleanupRecordError(
                "cleanup artifact Tag differs from the Tag caller"
            )
        return record

    def discover_runs(self) -> None:
        for path in _SOURCE_WORKFLOWS:
            filename = path.rsplit("/", 1)[-1]
            try:
                runs = self.envelope(
                    f"{self.prefix}/actions/workflows/{filename}/runs", "workflow_runs"
                )
            except (domain.CleanupRecordError, OSError) as error:
                if _missing(error):
                    continue
                raise
            for run in runs:
                if not self.trusted_run(run) or run["path"].split("@", 1)[0] != path:
                    continue
                run_id = _positive(run.get("id"), "source run ID")
                self.runs[run_id] = run
                tag = (
                    _tag(run.get("head_branch"))
                    if filename == "release-tag.yml"
                    else None
                )
                if filename == "release-tag.yml" and tag is None:
                    continue
                active = run.get("status") != "completed"
                if active:
                    if tag:
                        self.active.add(tag)
                    self.active.update(
                        record["tag"]
                        for record in self.records.values()
                        if record["source_sha"] == run["head_sha"]
                    )
                # A complete live Release attachment already merges all attempts;
                # artifacts recover Tag-only and interrupted attachment updates.
                try:
                    artifacts = self.envelope(
                        f"{self.prefix}/actions/runs/{run_id}/artifacts", "artifacts"
                    )
                    for artifact in artifacts:
                        name = artifact.get("name")
                        if (
                            not isinstance(name, str)
                            or _ARTIFACT.fullmatch(name) is None
                            or artifact.get("expired") is True
                        ):
                            continue
                        try:
                            record = self.artifact(artifact, run)
                            self.add(record, proven=True)
                            if active:
                                self.active.add(record["tag"])
                        except (domain.CleanupRecordError, OSError) as error:
                            affected = (
                                {tag}
                                if tag
                                else {
                                    record["tag"]
                                    for record in self.records.values()
                                    if record["source_sha"] == run["head_sha"]
                                }
                            )
                            for affected_tag in affected:
                                if affected_tag:
                                    self.block(affected_tag, error)
                except (domain.CleanupRecordError, OSError) as error:
                    if _missing(error):
                        continue
                    affected = (
                        {tag}
                        if tag
                        else {
                            record["tag"]
                            for record in self.records.values()
                            if record["source_sha"] == run["head_sha"]
                        }
                    )
                    if not affected:
                        raise
                    for affected_tag in affected:
                        if affected_tag:
                            self.block(affected_tag, error)

    def bind_live_objects(self) -> None:
        for tag, record in tuple(self.records.items()):
            for release_id in record["release_ids"]:
                release = self.releases.get(release_id)
                if release is None:
                    continue
                direct = _release_tag(release)
                orphan_name = (
                    _tag(release.get("name"))
                    if release.get("draft") is True
                    and str(release.get("tag_name", "")).startswith("untagged-")
                    else None
                )
                if direct != tag and orphan_name != tag:
                    self.block(
                        tag, f"Release ID {release_id} belongs to a different identity"
                    )
            for binding in record["runs"]:
                run_id = binding["id"]
                if run_id not in self.runs:
                    try:
                        self.runs[run_id] = self.remote._github_json(
                            "GET", f"{self.prefix}/actions/runs/{run_id}"
                        )
                    except (domain.CleanupRecordError, OSError) as error:
                        if _missing(error):
                            self.runs[run_id] = None
                        else:
                            self.block(tag, error)
                            self.active.add(tag)
                            continue
                run = self.runs[run_id]
                if run is None:
                    continue
                if not self.trusted_run(run) or run.get("id") != run_id:
                    self.block(
                        tag,
                        f"source run {run_id} is not a trusted repository publication",
                    )
                    continue
                if (
                    record["source_sha"] is not None
                    and record["source_sha"] != run["head_sha"]
                ):
                    self.block(
                        tag, f"source run {run_id} SHA differs from cleanup binding"
                    )
                else:
                    self.basic(tag, source_sha=run["head_sha"])
                if binding["attempt"] > _positive(
                    run.get("run_attempt"), "source run attempt"
                ):
                    self.block(
                        tag, f"source run {run_id} attempt differs from cleanup binding"
                    )
                if (
                    run["path"].split("@", 1)[0] == ".github/workflows/release-tag.yml"
                    and _tag(run.get("head_branch")) != tag
                ):
                    self.block(
                        tag, f"source run {run_id} Tag differs from cleanup binding"
                    )
                if run.get("status") != "completed":
                    self.active.add(tag)

    def finish(self) -> NightlyCatalog:
        for tag in self.existing:
            if tag not in self.proven:
                self.block(tag, "no trusted complete cleanup inventory was recovered")
        records = domain.merge_records(
            [record for tag, record in self.records.items() if tag in self.existing],
            self.repository,
        )
        orphans = domain.merge_records(
            [
                record
                for tag, record in self.records.items()
                if tag not in self.existing
                and (
                    record["resources"]
                    or tag in self.blocked
                    or (tag in self.historical and record["runs"])
                )
            ],
            self.repository,
        )
        return NightlyCatalog(
            records, orphans, frozenset(self.active), dict(self.blocked)
        )


def collect_catalog(
    remote: CatalogRemote, history_document: object = None
) -> NightlyCatalog:
    """Read Releases, Nightly refs, and trusted caller evidence without mutations.

    Global listing/authentication failures propagate: cleanup must not run against
    a partial catalog. Individual unproven or conflicting versions remain visible
    in ``records`` and receive an explicit deletion blocker.
    """
    collector = _Collector(remote)
    collector.discover_releases()
    collector.discover_refs()
    collector.history(history_document)
    collector.discover_runs()
    collector.bind_live_objects()
    return collector.finish()
