"""Discover release versions and manage their exact cleanup and recovery evidence.

Resolution performs reads only. Saving new attempt snapshots or recovered plans
is explicit, and must complete before deleting the objects holding that evidence.
"""

from __future__ import annotations

import io
import json
import re
import urllib.parse
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import cleanup_records as domain
from . import version_config
from .cleanup_records import (
    ActiveRelease,
    CleanupError,
    CleanupResolution,
    CleanupTarget,
    PendingRecovery,
    Resource,
    plan_resources,
    registry_resources_from_references,
)
from .cleanup_remote import ProductionRemote, RemoteError
from .manifest import RELEASE_MANIFEST_FILENAME as MANIFEST_FILENAME

CLEANUP_RECORD_PREFIX = "release-cleanup-"
_SOURCE_WORKFLOWS = domain.RELEASE_WORKFLOW_EVENTS
_ARTIFACT = re.compile(
    r"ucm-(?:release|nightly)-cleanup(?:-(?:open|plan))?-run-(?P<run>[1-9][0-9]*)"
    r"-attempt-(?P<attempt>[1-9][0-9]*)"
)
_SHA = re.compile(r"[0-9a-f]{40}")
_OWNED_STATUS = re.compile(
    r"^Status: `(release-open|artifacts-ready|artifacts-failed|images-failed|publication-failed)`",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ReleaseCatalog:
    records: tuple[dict[str, Any], ...]
    orphans: tuple[dict[str, Any], ...]
    active_tags: frozenset[str]
    blocked: dict[str, str]


def _tag(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        version_config.classify_tag(value)
    except ValueError:
        return None
    return value


def _positive(value: object, context: str) -> int:
    if type(value) is not int or value < 1:
        raise CleanupError(f"{context} must be a positive integer")
    return value


def _missing(error: Exception) -> bool:
    return getattr(error, "status", getattr(error, "code", None)) == 404


def _owned_release(release: dict[str, Any]) -> bool:
    body = release.get("body") or ""
    return isinstance(body, str) and (
        "<!-- ucm-release:begin -->" in body or _OWNED_STATUS.search(body) is not None
    )


def release_tag(release: dict[str, Any]) -> str | None:
    direct = _tag(release.get("tag_name"))
    if direct is not None:
        return direct
    if (
        release.get("draft") is True
        and str(release.get("tag_name", "")).startswith("untagged-")
        and (
            _owned_release(release)
            or any(
                isinstance(asset, dict)
                and (
                    asset.get("name") in {domain.CLEANUP_FILENAME, MANIFEST_FILENAME}
                    or str(asset.get("name", "")).startswith(CLEANUP_RECORD_PREFIX)
                )
                for asset in release.get("assets", [])
            )
        )
    ):
        return _tag(release.get("name"))
    return None


class ReleaseInventory:
    """Own evidence discovery, format adapters, and explicit snapshot persistence."""

    def __init__(self, remote: ProductionRemote) -> None:
        self.remote = remote
        self.repository = remote.repository
        self.owner = self.repository.split("/", 1)[0].lower()
        self.api_base = getattr(remote, "api_base", "https://api.github.com")

    def _run(self, run_id: int) -> dict[str, Any] | None:
        return self.remote.read_release_run(run_id)

    def _tag_commit(self, tag: str) -> str | None:
        ref = urllib.parse.quote(f"tags/{tag}", safe="/")
        try:
            value = self.remote.json_request(
                "GET", f"/repos/{self.repository}/git/ref/{ref}"
            )
        except RemoteError as error:
            if error.is_missing:
                return None
            raise
        if not isinstance(value, dict):
            raise CleanupError("Git Tag reference response is malformed")
        target = value.get("object")
        while isinstance(target, dict):
            sha = target.get("sha")
            if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
                raise CleanupError("Git Tag object SHA is malformed")
            if target.get("type") == "commit":
                return sha
            if target.get("type") != "tag":
                raise CleanupError("Git Tag does not resolve to a commit")
            value = self.remote.json_request(
                "GET", f"/repos/{self.repository}/git/tags/{sha}"
            )
            target = value.get("object") if isinstance(value, dict) else None
        raise CleanupError("Git Tag object response is malformed")

    def _matching_releases(
        self, tag: str, *, release_id: int | None = None
    ) -> list[dict[str, Any]]:
        releases = [
            release
            for release in self.remote.list_releases()
            if release_tag(release) == tag
        ]
        if release_id is not None and not any(
            release.get("id") == release_id for release in releases
        ):
            try:
                selected = self.remote.json_request(
                    "GET", f"/repos/{self.repository}/releases/{release_id}"
                )
            except RemoteError as error:
                if not error.is_missing:
                    raise
            else:
                if (
                    not isinstance(selected, dict)
                    or selected.get("id") != release_id
                    or release_tag(selected) != tag
                ):
                    raise CleanupError("selected Release does not match its Tag and ID")
                releases.append(selected)
        for release in releases:
            identifier = release.get("id")
            if (
                not isinstance(identifier, int)
                or isinstance(identifier, bool)
                or identifier < 1
                or not isinstance(release.get("assets"), list)
                or any(not isinstance(asset, dict) for asset in release["assets"])
            ):
                raise CleanupError("GitHub Release identity or assets are malformed")
        return sorted(releases, key=lambda release: release["id"])

    def _asset_json(self, asset: dict[str, Any]) -> dict[str, Any]:
        identifier = asset.get("id")
        if (
            not isinstance(identifier, int)
            or isinstance(identifier, bool)
            or identifier < 1
        ):
            raise CleanupError("GitHub recovery asset ID is malformed")
        # Use our own repository endpoint, never an asset-supplied foreign URL.
        raw = self.remote.request(
            "GET",
            f"/repos/{self.repository}/releases/assets/{identifier}",
            accept="application/octet-stream",
        )
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CleanupError("GitHub recovery asset is not valid JSON") from error
        if not isinstance(document, dict):
            raise CleanupError("GitHub recovery asset must be an object")
        return document

    def _validate_run_binding(
        self, run: dict[str, Any], *, tag: str, source_sha: str
    ) -> None:
        if (
            not isinstance(source_sha, str)
            or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None
        ):
            raise CleanupError("publication source SHA is malformed")
        path = run["path"].split("@", 1)[0]
        if path == ".github/workflows/release-tag.yml":
            if (
                run.get("event") != "push"
                or run.get("head_branch") != tag
                or run.get("head_sha") != source_sha
            ):
                raise CleanupError("release run does not match the exact Tag and SHA")
        elif version_config.classify_tag(tag)["release_type"] != "nightly" or run.get(
            "event"
        ) not in {"schedule", "workflow_dispatch"}:
            raise CleanupError("Nightly run does not match the Tag provenance")

    def _publication_runs(self, tag: str, source_sha: str) -> list[int]:
        """Find reruns that may precede their persistent ownership snapshot."""
        branch = urllib.parse.quote(tag, safe="")
        paths = [
            (
                f"/repos/{self.repository}/actions/workflows/release-tag.yml/runs"
                f"?branch={branch}&event=push"
            )
        ]
        if version_config.classify_tag(tag)["release_type"] == "nightly":
            paths.append(
                f"/repos/{self.repository}/actions/workflows/release-nightly.yml/runs"
                f"?head_sha={source_sha}"
            )
        runs: set[int] = set()
        for path in paths:
            for run in self.remote.list_pages(path, key="workflow_runs"):
                run_id = run.get("id")
                if (
                    not isinstance(run_id, int)
                    or isinstance(run_id, bool)
                    or run_id < 1
                ):
                    raise CleanupError("legacy publication run ID is malformed")
                runs.add(run_id)
        return sorted(runs)

    def _failed_before_publication(self, run_id: int) -> bool:
        """Prove an unrecorded run never reached a resource publisher.

        Include every rerun attempt: a later planning failure must not hide an
        earlier partial publication. Require each known publisher's skipped job
        record rather than interpreting missing jobs as evidence of absence.
        """
        jobs = self.remote.list_pages(
            f"/repos/{self.repository}/actions/runs/{run_id}/jobs?filter=all",
            key="jobs",
        )
        publishers = (
            "Tag Image · ",
            "Verify directly published image members",
            "Publish index · ",
            "Publish · Chart OCI",
            "PyPI · ",
            "Release · Publish backend Wheels, Chart, and Config",
            "Release · Finalize all channels",
        )
        preparatory = (
            "Release · Validate version, Tag, and publication targets",
            "Select Runtime Registry Tags",
            "Release · Open ",
            "Inspect selected Runtime manifests",
            "Capability check · ",
            "Resolve Runtime capabilities and Builders",
            "Prepare immutable Wheel Builders",
            "Release · Report planning failure",
            "Wheel · ",
            "Validate Wheel Runtime · ",
            "Chart · Helm package",
            "Verify PyPI · ",
            "Verify · Published bilingual documentation",
            "Verify · Published Helm package and image tags",
            "Verify · Complete RC delivery",
            "Cleanup · Retain the configured same-type Releases",
        )
        callers = {
            "Run Release core",
            "Run official Release core",
            "Run fork Release core",
        }
        seen: set[str] = set()
        planned = False
        for job in jobs:
            name = job.get("name")
            if (
                not isinstance(name, str)
                or job.get("run_id") != run_id
                or job.get("status") != "completed"
            ):
                return False
            components = name.split(" / ")
            if len(components) == 1:
                if name == "Classify immutable Release Tag":
                    continue
                if name in callers and job.get("conclusion") == "skipped":
                    continue
                return False
            if components[0] not in callers:
                return False
            label = components[1]
            if label == "Plan Wheels, Images, and Chart":
                if job.get("conclusion") not in {"failure", "skipped"}:
                    return False
                planned = True
                continue
            publisher = next((p for p in publishers if label.startswith(p)), None)
            if publisher is not None:
                if job.get("conclusion") != "skipped":
                    return False
                seen.add(publisher)
            elif not label.startswith(preparatory):
                return False
        return planned and seen == set(publishers)

    def _legacy_plans(
        self, tag: str, source_sha: str, *, recorded_runs: Sequence[int] = ()
    ) -> list[tuple[int, dict[str, Any] | None]]:
        """Recover exact plans, including attempts older than the first snapshot."""
        plans = []
        recorded = set(recorded_runs)
        for run_id in self._publication_runs(tag, source_sha):
            run = self._run(run_id)
            if run is None:
                continue
            self._validate_run_binding(run, tag=tag, source_sha=run.get("head_sha"))
            if run.get("status") != "completed":
                raise ActiveRelease(f"Tag may still be used by Actions run {run_id}")
            if run_id in recorded:
                continue
            artifacts = self.remote.list_pages(
                f"/repos/{self.repository}/actions/runs/{run_id}/artifacts",
                key="artifacts",
            )
            matches = [
                artifact
                for artifact in artifacts
                if artifact.get("name") == f"ucm-release-plan-run-{run_id}"
            ]
            if not matches:
                if run["path"].split("@", 1)[0] == ".github/workflows/release-tag.yml":
                    if self._failed_before_publication(run_id):
                        plans.append((run_id, None))
                        continue
                    raise CleanupError(
                        f"publication run {run_id} has no recoverable release plan"
                    )
                continue
            if len(matches) != 1 or matches[0].get("expired") is not False:
                raise CleanupError(
                    f"publication run {run_id} release plan is ambiguous or expired"
                )
            artifact = matches[0]
            identifier = artifact.get("id")
            if (
                not isinstance(identifier, int)
                or isinstance(identifier, bool)
                or identifier < 1
            ):
                raise CleanupError("legacy release plan artifact ID is malformed")
            owner = artifact.get("workflow_run")
            if (
                not isinstance(owner, dict)
                or owner.get("id") != run_id
                or owner.get("head_sha") != run.get("head_sha")
            ):
                raise CleanupError(
                    "legacy release plan artifact belongs to another run"
                )
            raw = self.remote.request(
                "GET", f"/repos/{self.repository}/actions/artifacts/{identifier}/zip"
            )
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    members = [
                        name
                        for name in archive.namelist()
                        if name.rsplit("/", 1)[-1] == "release-plan.json"
                    ]
                    if len(members) != 1:
                        raise CleanupError(
                            "legacy artifact has no unique release-plan.json"
                        )
                    plan = json.loads(archive.read(members[0]))
            except (
                zipfile.BadZipFile,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as error:
                raise CleanupError(
                    "legacy release plan archive is malformed"
                ) from error
            if not isinstance(plan, dict):
                raise CleanupError("legacy release plan must be an object")
            if plan.get("git_tag") != tag:
                if run["path"].split("@", 1)[0] == ".github/workflows/release-tag.yml":
                    raise CleanupError("legacy release plan belongs to another Tag")
                continue
            plan_resources(plan, repository=self.repository, tag=tag)
            plans.append((run_id, plan))
        return plans

    def save_record(
        self,
        tag: str,
        run_id: int,
        source_sha: str,
        plan: dict[str, Any] | None = None,
        *,
        run_attempt: int | None = None,
        release_id: int | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Save one immutable publication attempt for any release type."""
        if self._tag_commit(tag) != source_sha:
            raise CleanupError("cleanup source SHA does not match the Git Tag")
        run = self._run(run_id)
        if run is None:
            raise CleanupError("cannot record a missing publication run")
        self._validate_run_binding(run, tag=tag, source_sha=source_sha)
        if run_attempt is not None and run.get("run_attempt") != run_attempt:
            raise CleanupError(
                "cleanup snapshot attempt differs from the publication run"
            )
        return self._persist_record(
            tag, run, source_sha, plan, release_id=release_id, output_dir=output_dir
        )

    def _persist_record(
        self,
        tag: str,
        run: dict[str, Any],
        source_sha: str,
        plan: dict[str, Any] | None = None,
        *,
        release_id: int | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        self._validate_run_binding(run, tag=tag, source_sha=source_sha)
        attempt = _positive(run.get("run_attempt"), "publication run attempt")
        document = domain.new_record(
            self.repository, tag, source_sha, run["id"], attempt, release_id, plan
        )
        name = domain.record_basename(document, attempt, planned=plan is not None)
        content = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / name).write_bytes(content)
        releases = self._matching_releases(tag, release_id=release_id)
        if len(releases) != 1:
            raise CleanupError("recording cleanup requires one exact Tag Release")
        release = releases[0]
        if release_id is not None and release["id"] != release_id:
            raise CleanupError("cleanup recovery Release changed after resolution")
        assets = [asset for asset in release["assets"] if asset.get("name") == name]
        if assets:
            if len(assets) != 1 or self._asset_json(assets[0]) != document:
                raise CleanupError(
                    "cleanup attempt snapshot conflicts with its saved asset"
                )
            return document
        upload_url = release.get("upload_url")
        if not isinstance(upload_url, str):
            raise CleanupError("GitHub Release has no upload URL")
        upload_url = upload_url.split("{", 1)[0]
        parsed = urllib.parse.urlparse(upload_url)
        api_host = urllib.parse.urlparse(self.api_base).netloc
        allowed_hosts = {api_host}
        if api_host == "api.github.com":
            allowed_hosts.add("uploads.github.com")
        if (
            parsed.scheme != "https"
            or parsed.netloc not in allowed_hosts
            or parsed.path
            != f"/repos/{self.repository}/releases/{release['id']}/assets"
            or parsed.query
            or parsed.fragment
        ):
            raise CleanupError("GitHub Release upload URL is outside this repository")
        raw = self.remote.request(
            "POST",
            upload_url + "?" + urllib.parse.urlencode({"name": name}),
            data=content,
            content_type="application/json",
        )
        try:
            asset = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CleanupError(
                "cleanup asset upload returned malformed JSON"
            ) from error
        if (
            not isinstance(asset, dict)
            or asset.get("name") != name
            or self._asset_json(asset) != document
        ):
            raise CleanupError(
                "cleanup asset upload readback differs from its snapshot"
            )
        return document

    def decode_asset(self, asset: dict[str, Any], release_id: int) -> dict[str, Any]:
        """Bind a supported asset format to its actual owning Release."""
        document = self._asset_json(asset)
        name = asset.get("name")
        if name == domain.CLEANUP_FILENAME:
            record = domain.decode_legacy_record(document, self.repository)
            if (
                release_id not in record["release_ids"]
                or not record["source_sha"]
                or not record["runs"]
            ):
                raise CleanupError(
                    "cleanup asset does not bind its live Release ID and source run"
                )
        elif name == MANIFEST_FILENAME:
            record = domain.record_from_manifest(document, self.repository, release_id)
        else:
            record = domain.decode_snapshot(document, self.repository, name, release_id)
        for binding in record["runs"]:
            run = self._run(binding["id"])
            if run is not None:
                if name == domain.CLEANUP_FILENAME and record["source_sha"] != run.get(
                    "head_sha"
                ):
                    raise CleanupError(
                        "legacy cleanup source SHA differs from its publication run"
                    )
                sources = [
                    s["source_sha"]
                    for s in record.get("attempt_sources", ())
                    if s["id"] == binding["id"] and s["attempt"] == binding["attempt"]
                ]
                self._validate_run_binding(
                    run,
                    tag=record["tag"],
                    source_sha=sources[0] if sources else run.get("head_sha"),
                )
                if binding["attempt"] > _positive(
                    run.get("run_attempt"), "publication run attempt"
                ):
                    raise CleanupError(
                        "cleanup snapshot contains a future publication attempt"
                    )
        return record

    def resolve_releases(self, tag: str) -> list[Resource]:
        return [
            Resource(
                "github-release",
                f"{tag}#{release['id']}",
                release["id"],
                holds_recovery_data=any(
                    a.get("name") == MANIFEST_FILENAME
                    or str(a.get("name", "")).startswith(CLEANUP_RECORD_PREFIX)
                    for a in release["assets"]
                ),
            )
            for release in self._matching_releases(tag)
        ]

    def resolve_record(self, record: object) -> CleanupResolution:
        """Resolve every source format to the same frozen deletion target."""
        record = domain.validate_record(record, self.repository)
        tag = record["tag"]
        references = [(r["kind"], r["reference"]) for r in record["resources"]]
        runs = {r["id"] for r in record["runs"]}
        releases = self._matching_releases(tag)
        for release_id in record["release_ids"]:
            if any(r["id"] == release_id for r in releases):
                continue
            try:
                release = self.remote.json_request(
                    "GET", f"/repos/{self.repository}/releases/{release_id}"
                )
            except RemoteError as error:
                if error.is_missing:
                    continue
                raise
            if (
                not isinstance(release, dict)
                or release.get("id") != release_id
                or release_tag(release) != tag
            ):
                raise CleanupError(
                    "recovery Release response does not match its Tag and ID"
                )
            releases.append(release)
        pending = []
        commit = self._tag_commit(tag)
        if commit is not None:
            for run_id, plan in self._legacy_plans(
                tag, commit, recorded_runs=sorted(runs)
            ):
                run = self._run(run_id)
                if run is None:
                    raise CleanupError(
                        "legacy publication run disappeared before recovery"
                    )
                source_sha = run["head_sha"]
                snapshot = domain.new_record(
                    self.repository,
                    tag,
                    source_sha,
                    run_id,
                    run["run_attempt"],
                    plan=plan,
                )
                references.extend(
                    (r["kind"], r["reference"]) for r in snapshot["resources"]
                )
                runs.add(run_id)
                if len(releases) != 1:
                    raise CleanupError(
                        "recording cleanup requires one exact Tag Release"
                    )
                pending.append(
                    PendingRecovery(tag, run, source_sha, plan, releases[0]["id"])
                )
        expected = set(references)
        expected_runs = {
            (binding["id"], binding["attempt"]) for binding in record["runs"]
        }
        expected_runs.update((p.run["id"], p.run["run_attempt"]) for p in pending)
        holders = set()
        for release in releases:
            saved_references = set()
            saved_runs = set()
            for asset in release["assets"]:
                name = asset.get("name", "")
                if name not in {domain.CLEANUP_FILENAME, MANIFEST_FILENAME} and not str(
                    name
                ).startswith(CLEANUP_RECORD_PREFIX):
                    continue
                saved = self.decode_asset(asset, release["id"])
                if saved["tag"] != tag:
                    raise CleanupError("recovery asset belongs to another Tag")
                saved_references.update(
                    (r["kind"], r["reference"]) for r in saved["resources"]
                )
                saved_runs.update((r["id"], r["attempt"]) for r in saved["runs"])
            for recovered in pending:
                if recovered.release_id != release["id"]:
                    continue
                snapshot = domain.new_record(
                    self.repository,
                    recovered.tag,
                    recovered.source_sha,
                    recovered.run["id"],
                    recovered.run["run_attempt"],
                    plan=recovered.plan,
                )
                saved_references.update(
                    (r["kind"], r["reference"]) for r in snapshot["resources"]
                )
                saved_runs.add((recovered.run["id"], recovered.run["run_attempt"]))
            if expected <= saved_references and expected_runs <= saved_runs:
                holders.add(release["id"])
        self.remote.ensure_idle(sorted(runs))
        target = CleanupTarget(
            tag,
            tuple(registry_resources_from_references(references)),
            tuple(
                Resource(
                    "actions-run",
                    f"https://github.com/{self.repository}/actions/runs/{rid}",
                    rid,
                    holds_recovery_data=not bool(holders),
                )
                for rid in sorted(runs)
            ),
            tuple(
                Resource(
                    "github-release",
                    f"{tag}#{r['id']}",
                    r["id"],
                    holds_recovery_data=r["id"] in holders,
                )
                for r in sorted(releases, key=lambda r: r["id"])
            ),
        )
        return CleanupResolution(target, tuple(pending))

    def persist_recovery(self, resolution: CleanupResolution) -> None:
        """Save and read back recovered attempt evidence before destructive work."""
        if resolution.pending_recovery:
            self.remote.ensure_idle(resolution.target.run_ids)
        for pending in resolution.pending_recovery:
            self._persist_record(
                pending.tag,
                pending.run,
                pending.source_sha,
                pending.plan,
                release_id=pending.release_id,
                output_dir=Path("out/cleanup"),
            )


class _Collector:
    def __init__(
        self, remote: ProductionRemote, release_type: str | None = None
    ) -> None:
        self.remote = remote
        self.inventory = ReleaseInventory(remote)
        self.release_type = release_type
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
        if (
            self.release_type is not None
            and version_config.classify_tag(tag)["release_type"] != self.release_type
        ):
            return
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

    def discover_releases(self) -> None:
        releases = self.remote.list_releases()
        for release in releases:
            release_id = _positive(release.get("id"), "Release ID")
            self.releases[release_id] = release
        for release_id, release in self.releases.items():
            direct = _tag(release.get("tag_name"))
            inferred = release_tag(release)
            candidate = direct or inferred or _tag(release.get("name"))
            if (
                self.release_type is not None
                and candidate is not None
                and version_config.classify_tag(candidate)["release_type"]
                != self.release_type
            ):
                continue
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
            names = [
                asset.get("name")
                for asset in assets
                if asset.get("name") in {domain.CLEANUP_FILENAME, MANIFEST_FILENAME}
                or str(asset.get("name", "")).startswith(CLEANUP_RECORD_PREFIX)
            ]
            for name in dict.fromkeys(names):
                matches = [asset for asset in assets if asset.get("name") == name]
                if len(matches) > 1:
                    errors.append(f"duplicate {name} assets")
                    continue
                if not matches:
                    continue
                try:
                    record = self.inventory.decode_asset(matches[0], release_id)
                    evidence.append(record)
                except (domain.CleanupRecordError, OSError) as error:
                    errors.append(str(error))
                    if (
                        not _missing(error)
                        and getattr(error, "status", None) is not None
                        and inferred
                    ):
                        self.active.add(inferred)
            if orphan and not _owned_release(release) and not evidence:
                inferred = None
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
                    f"Release ID {release_id} has no unambiguous original release identity"
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
        refs = self.remote.json_request("GET", f"{self.prefix}/git/matching-refs/tags/")
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
                            "annotated release Tag cannot resolve to one commit"
                        )
                    seen.add(sha)
                    resolved = self.remote.json_request(
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
                        "release Tag does not point to a commit"
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
            decoded = domain.decode_legacy_record(record, self.repository)
            self.historical.add(decoded["tag"])
            self.add(decoded, proven=True)

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
        # This REST endpoint redirects to the ZIP and requires GitHub's API media type.
        raw = self.remote.request(
            "GET",
            f"{self.prefix}/actions/artifacts/{artifact_id}/zip",
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
                    or entries[0].file_size > 2 * 1024 * 1024
                    or entries[0].filename != entries[0].filename.rsplit("/", 1)[-1]
                ):
                    raise domain.CleanupRecordError(
                        "cleanup artifact must contain only one cleanup record"
                    )
                value = json.loads(archive.read(entries[0]))
                record = domain.decode_record(
                    value, self.repository, asset_name=entries[0].filename
                )
        except (
            zipfile.BadZipFile,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RuntimeError,
        ) as error:
            raise domain.CleanupRecordError(
                "cleanup artifact is not a readable inventory ZIP"
            ) from error
        expected_source = next(
            (
                item["source_sha"]
                for item in record["attempt_sources"]
                if item["id"] == run_id and item["attempt"] == attempt
            ),
            record["source_sha"],
        )
        if value.get("kind") == domain.CLEANUP_RECORD_KIND:
            self.inventory._validate_run_binding(
                run, tag=record["tag"], source_sha=expected_source
            )
        elif expected_source != run["head_sha"]:
            raise CleanupError(
                "cleanup artifact source SHA differs from its source run"
            )
        if {"id": run_id, "attempt": attempt} not in record["runs"]:
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
            if (
                self.release_type is not None
                and self.release_type != "nightly"
                and path.endswith("/release-nightly.yml")
            ):
                continue
            filename = path.rsplit("/", 1)[-1]
            try:
                runs = self.remote.list_pages(
                    f"{self.prefix}/actions/workflows/{filename}/runs",
                    key="workflow_runs",
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
                    else:
                        self.active.update(
                            record["tag"]
                            for record in self.records.values()
                            if record["source_sha"] == run["head_sha"]
                        )
                # A complete live Release attachment already merges all attempts;
                # artifacts recover Tag-only and interrupted attachment updates.
                try:
                    artifacts = self.remote.list_pages(
                        f"{self.prefix}/actions/runs/{run_id}/artifacts",
                        key="artifacts",
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
                direct = release_tag(release)
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
                        self.runs[run_id] = self.remote.json_request(
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
                expected_source = next(
                    (
                        item["source_sha"]
                        for item in record["attempt_sources"]
                        if item["id"] == run_id
                        and item["attempt"] == binding["attempt"]
                    ),
                    record["source_sha"],
                )
                if expected_source is not None:
                    try:
                        # A scheduled rerun can reuse a Tag pinned to an older
                        # commit. Tag-push runs must match their event SHA.
                        self.inventory._validate_run_binding(
                            run, tag=tag, source_sha=expected_source
                        )
                    except CleanupError as error:
                        self.block(
                            tag,
                            f"source run {run_id} SHA differs from cleanup binding: {error}",
                        )
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

    def recover_unrecorded_versions(self) -> None:
        """Keep manifest-less versions visible when trusted old plans prove ownership."""
        for tag, record in self.records.items():
            if tag in self.proven or tag in self.active or tag not in self.existing:
                continue
            source_sha = record["source_sha"]
            if source_sha is None:
                source_sha = self.inventory._tag_commit(tag)
            if source_sha is None:
                continue
            try:
                plans = self.inventory._legacy_plans(tag, source_sha)
            except ActiveRelease:
                self.active.add(tag)
            except CleanupError as error:
                self.block(tag, error)
            else:
                if plans:
                    self.proven.add(tag)

    def finish(self) -> ReleaseCatalog:
        existing = self.existing & self.records.keys()
        for tag in existing:
            if tag not in self.proven and tag not in self.active:
                self.block(tag, "no trusted complete cleanup inventory was recovered")
        records = domain.merge_records(
            [record for tag, record in self.records.items() if tag in existing],
            self.repository,
        )
        orphans = domain.merge_records(
            [
                record
                for tag, record in self.records.items()
                if tag not in existing
                and (
                    record["resources"]
                    or tag in self.blocked
                    or (tag in self.historical and record["runs"])
                )
            ],
            self.repository,
        )
        return ReleaseCatalog(
            records,
            orphans,
            frozenset(self.active & self.records.keys()),
            {
                tag: reason
                for tag, reason in self.blocked.items()
                if tag in self.records
            },
        )


def collect_catalog(
    remote: ProductionRemote,
    history_document: object = None,
    *,
    release_type: str | None = None,
) -> ReleaseCatalog:
    """Read all trusted version evidence; individual incomplete versions stay counted."""
    collector = _Collector(remote, release_type)
    collector.discover_releases()
    collector.discover_refs()
    collector.history(history_document)
    collector.discover_runs()
    collector.bind_live_objects()
    collector.recover_unrecorded_versions()
    return collector.finish()
