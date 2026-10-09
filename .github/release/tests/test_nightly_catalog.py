"""Catalog actual Nightlies without losing unfinished or orphaned publications."""

from __future__ import annotations

import importlib
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
catalog = importlib.import_module("ucm_release.nightly_catalog")
domain = importlib.import_module("ucm_release.nightly_cleanup")
REPOSITORY = "release-org/unified-cache-management"
PREFIX = f"/repos/{REPOSITORY}"
TAG = "nightly/v0.9.0-20261009-1"
SHA = "a" * 40


class RemoteError(domain.CleanupRecordError):
    def __init__(self, status):
        super().__init__(f"GitHub API HTTP {status}")
        self.status = status


class FakeRemote:
    repository = REPOSITORY
    api_base = "https://api.github.com"

    def __init__(self):
        self.releases = []
        self.refs = []
        self.tag_objects = {}
        self.workflows = {"release-nightly.yml": [], "release-tag.yml": []}
        self.runs = {}
        self.artifacts = {}
        self.downloads = {}
        self.errors = {}
        self.calls = []

    def _github_json(self, method, path):
        self.calls.append((method, path))
        assert method == "GET"
        path = urlparse(path).path
        if path in self.errors:
            raise self.errors[path]
        if path == PREFIX + "/git/matching-refs/tags/nightly/":
            return self.refs
        if "/git/tags/" in path:
            return self.tag_objects[path.rsplit("/", 1)[-1]]
        if "/actions/workflows/" in path:
            name = path.split("/actions/workflows/", 1)[1].split("/", 1)[0]
            return {"workflow_runs": self.workflows.get(name, [])}
        if path.endswith("/artifacts"):
            run_id = int(path.split("/runs/", 1)[1].split("/", 1)[0])
            return {"artifacts": self.artifacts.get(run_id, [])}
        if "/actions/runs/" in path:
            run_id = int(path.rsplit("/", 1)[-1])
            if run_id not in self.runs:
                raise RemoteError(404)
            return self.runs[run_id]
        raise AssertionError(path)

    def _all_pages(self, path):
        self.calls.append(("GET", path))
        if path in self.errors:
            raise self.errors[path]
        assert path == PREFIX + "/releases"
        return self.releases

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path))
        assert method == "GET"
        if path in self.errors:
            raise self.errors[path]
        return self.downloads[path]

    def add_run(self, run, *, listed=True):
        self.runs[run["id"]] = run
        if listed:
            self.workflows[run["path"].rsplit("/", 1)[-1]].append(run)

    def add_asset(self, release, value, *, asset_id=10, name="release-cleanup.json"):
        release["assets"].append(
            {"name": name, "id": asset_id, "url": "https://untrusted.test/unused"}
        )
        self.downloads[f"{PREFIX}/releases/assets/{asset_id}"] = json.dumps(
            value
        ).encode()

    def add_artifact(
        self, run, record, *, artifact_id=100, phase="", attempt=1, extra=False
    ):
        name = f"ucm-nightly-cleanup{phase}-run-{run['id']}-attempt-{attempt}"
        self.artifacts.setdefault(run["id"], []).append(
            {
                "id": artifact_id,
                "name": name,
                "expired": False,
                "workflow_run": {"id": run["id"]},
            }
        )
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("release-cleanup.json", json.dumps(record))
            if extra:
                archive.writestr("../untrusted.py", "code")
        self.downloads[f"{PREFIX}/actions/artifacts/{artifact_id}/zip"] = (
            output.getvalue()
        )


def _release(tag=TAG, *, release_id=1, draft=True, tag_name=None, owned=True):
    return {
        "id": release_id,
        "tag_name": tag_name or tag,
        "name": tag,
        "draft": draft,
        "prerelease": True,
        "assets": [],
        "body": "Status: `artifacts-failed`" if owned else "user notes",
    }


def _ref(tag=TAG, *, sha=SHA, kind="commit"):
    return {"ref": "refs/tags/" + tag, "object": {"type": kind, "sha": sha}}


def _run(
    *,
    run_id=123,
    attempt=1,
    status="completed",
    event="schedule",
    path="release-nightly.yml",
    branch="develop",
    sha=SHA,
    repository=REPOSITORY,
):
    return {
        "id": run_id,
        "run_attempt": attempt,
        "status": status,
        "event": event,
        "path": ".github/workflows/" + path,
        "head_branch": branch,
        "head_sha": sha,
        "head_repository": {"full_name": repository},
        "repository": {"full_name": REPOSITORY},
    }


def _record(tag=TAG, *, release_id=1, run_id=123, attempt=1, sha=SHA, resources=False):
    record = domain.record_for_tag(REPOSITORY, tag, sha, run_id, attempt, release_id)
    if resources:
        version = importlib.import_module("ucm_release.version_config").classify_tag(
            tag
        )["image_version"]
        record["resources"] = [
            {
                "kind": "ghcr-member",
                "reference": f"ghcr.io/release-org/vllm-openai:v1-ucm-{version}",
            }
        ]
    return record


def _legacy(tag=TAG, run_id=123):
    return {
        "kind": "ucm-release-manifest",
        "schema_version": 6,
        "tag": tag,
        "release_type": "nightly",
        "actions_run_id": run_id,
        "chart_oci": None,
        "github_release_assets": ["release-manifest.json"],
        "runtime_images": {
            "ghcr": {"indexes": [], "members": []},
            "dockerhub": {"indexes": [], "members": []},
        },
    }


def _history(*records, blocked=None):
    return {
        "kind": "ucm-nightly-cleanup-inventory",
        "schema_version": 1,
        "repository": REPOSITORY,
        "targets": list(records),
        "evidence": {},
        "blocked": blocked or {},
    }


def test_manifestless_failed_drafts_count_and_remain_explicitly_blocked():
    remote = FakeRemote()
    remote.releases = [_release(), _release("nightly/v0.9.0-20261008-1", release_id=2)]
    remote.refs = [_ref(), _ref("nightly/v0.9.0-20261008-1")]
    result = catalog.collect_catalog(remote)
    assert len(result.records) == 2
    assert set(result.blocked) == {item["tag"] for item in result.records}
    assert (
        domain.select_retention(result.records, 1)
        .candidates[0]["tag"]
        .endswith("20261008-1")
    )


def test_supported_legacy_manifest_recovers_published_and_untagged_drafts():
    remote = FakeRemote()
    published = _release(draft=False)
    orphan = _release(
        "nightly/v0.9.0-20261008-1", release_id=2, tag_name="untagged-abcd", owned=False
    )
    remote.releases = [published, orphan]
    remote.add_asset(published, _legacy(), name="release-manifest.json")
    remote.add_asset(
        orphan, _legacy(orphan["name"], 124), asset_id=11, name="release-manifest.json"
    )
    result = catalog.collect_catalog(remote)
    assert len(result.records) == 2
    assert result.blocked == {}
    assert result.orphans == ()  # GitHub orphan Drafts still occupy quota.


def test_untagged_name_requires_pipeline_evidence_and_ordinary_drafts_stay_out():
    remote = FakeRemote()
    remote.releases = [
        _release(tag_name="untagged-a"),
        _release(
            "nightly/v0.9.0-20261008-1",
            release_id=2,
            tag_name="untagged-b",
            owned=False,
        ),
        _release("draft/v0.9.0", release_id=3),
    ]
    result = catalog.collect_catalog(remote)
    assert [record["tag"] for record in result.records] == [TAG]
    assert TAG in result.blocked


def test_live_producer_record_binds_release_source_run_and_actual_tag():
    remote = FakeRemote()
    release = _release()
    remote.releases = [release]
    remote.refs = [_ref()]
    remote.add_asset(release, _record(resources=True))
    remote.add_run(_run(), listed=False)
    result = catalog.collect_catalog(remote)
    assert len(result.records[0]["resources"]) == 1
    assert result.blocked == {}
    assert all(method == "GET" for method, _ in remote.calls)
    assert not any("untrusted.test" in path for _, path in remote.calls)


@pytest.mark.parametrize(
    "change",
    ["release", "source", "repository", "caller_tag", "workflow", "future_attempt"],
)
def test_producer_record_binding_failures_block_the_version(change):
    remote = FakeRemote()
    release = _release()
    remote.releases = [release]
    remote.refs = [_ref()]
    record = _record(resources=True)
    run = _run()
    if change == "release":
        record["release_ids"] = [2]
    elif change == "source":
        run["head_sha"] = "b" * 40
    elif change == "repository":
        run["head_repository"]["full_name"] = "untrusted/repo"
    elif change == "caller_tag":
        run.update(
            path=".github/workflows/release-tag.yml",
            event="push",
            head_branch="nightly/v0.9.0-20261008-1",
        )
    elif change == "workflow":
        run["path"] = ".github/workflows/arbitrary.yml"
    else:
        record["runs"][0]["attempt"] = 2
    remote.add_asset(release, record)
    remote.runs[123] = run
    result = catalog.collect_catalog(remote)
    assert len(result.records) == 1
    assert TAG in result.blocked


def test_duplicate_release_records_merge_and_wrong_additional_release_id_is_blocked():
    remote = FakeRemote()
    first, second = _release(), _release(release_id=2)
    remote.releases = [first, second, _release("v0.9.0", release_id=3)]
    remote.add_asset(first, _record())
    remote.add_asset(second, _record(release_id=2), asset_id=11)
    result = catalog.collect_catalog(remote)
    assert len(result.records) == 1
    assert result.records[0]["release_ids"] == [1, 2]
    forged = _record()
    forged["release_ids"].append(3)
    remote.add_asset(
        first, forged, asset_id=12
    )  # Duplicate asset itself is also unsafe.
    result = catalog.collect_catalog(remote)
    assert TAG in result.blocked


@pytest.mark.parametrize("status", ["in_progress", "queued", "waiting"])
def test_active_tag_only_source_is_protected_and_counts_toward_quota(status):
    remote = FakeRemote()
    remote.refs = [_ref()]
    run = _run(status=status, path="release-tag.yml", event="push", branch=TAG)
    remote.add_run(run)
    remote.add_artifact(run, _record(release_id=None))
    result = catalog.collect_catalog(remote)
    assert len(result.records) == 1
    assert result.active_tags == {TAG}
    assert result.blocked == {}


def test_unknown_active_nightly_caller_conservatively_protects_matching_source_sha():
    remote = FakeRemote()
    remote.refs = [_ref(), _ref("nightly/v0.9.0-20261008-1")]
    remote.add_run(_run(status="queued"))
    result = catalog.collect_catalog(remote)
    assert result.active_tags == {TAG, "nightly/v0.9.0-20261008-1"}


def test_prepare_artifact_without_a_live_tag_or_release_does_not_create_phantom_quota():
    remote = FakeRemote()
    run = _run()
    remote.add_run(run)
    remote.add_artifact(run, _record(release_id=None))
    result = catalog.collect_catalog(remote)
    assert result.records == result.orphans == ()


def test_plan_artifact_and_prior_attempts_recover_exact_union_after_attachment_failure():
    remote = FakeRemote()
    remote.refs = [_ref()]
    run = _run(attempt=2)
    remote.add_run(run)
    remote.add_artifact(run, _record(release_id=None), artifact_id=100, attempt=1)
    remote.add_artifact(
        run,
        _record(attempt=2, resources=True),
        artifact_id=101,
        phase="-plan",
        attempt=2,
    )
    result = catalog.collect_catalog(remote)
    assert result.blocked == {}
    assert result.records[0]["runs"] == [
        {"id": 123, "attempt": 1},
        {"id": 123, "attempt": 2},
    ]
    assert len(result.records[0]["resources"]) == 1


@pytest.mark.parametrize("defect", ["sha", "attempt", "tag", "zip"])
def test_artifact_binding_and_zip_shape_fail_closed(defect):
    remote = FakeRemote()
    remote.refs = [_ref()]
    run = _run(path="release-tag.yml", event="push", branch=TAG)
    remote.add_run(run)
    record = _record(release_id=None)
    if defect == "sha":
        record["source_sha"] = "b" * 40
    if defect == "attempt":
        record["runs"][0]["attempt"] = 2
    if defect == "tag":
        record["tag"] = "nightly/v0.9.0-20261008-1"
    remote.add_artifact(run, record, extra=defect == "zip")
    result = catalog.collect_catalog(remote)
    assert TAG in result.blocked
    assert len(result.records) == 1


def test_untrusted_workflow_run_artifacts_are_never_downloaded():
    remote = FakeRemote()
    remote.refs = [_ref()]
    run = _run(repository="attacker/repo")
    remote.add_run(run)
    remote.add_artifact(run, _record(release_id=None))
    result = catalog.collect_catalog(remote)
    assert TAG in result.blocked
    assert not any(path.endswith("/zip") for _, path in remote.calls)


def test_annotated_tag_is_resolved_and_binds_source_sha():
    remote = FakeRemote()
    remote.refs = [_ref(sha="b" * 40, kind="tag")]
    remote.tag_objects["b" * 40] = {"object": {"type": "commit", "sha": SHA}}
    result = catalog.collect_catalog(remote, _history(_record(release_id=None)))
    assert result.records[0]["source_sha"] == SHA
    assert result.blocked == {}


def test_history_orphans_are_outside_quota_and_explicit_missing_evidence_is_preserved():
    remote = FakeRemote()
    live = _record()
    old = _record("nightly/v0.9.0-20261008-1", release_id=2, run_id=124, resources=True)
    blocked_tag = "nightly/v0.7.0-20260910-1"
    remote.releases = [_release()]
    result = catalog.collect_catalog(
        remote,
        _history(live, old, blocked={blocked_tag: "plan was removed before archival"}),
    )
    assert [record["tag"] for record in result.records] == [TAG]
    assert {record["tag"] for record in result.orphans} == {old["tag"], blocked_tag}
    assert result.blocked == {blocked_tag: "plan was removed before archival"}


def test_history_run_only_orphan_remains_cleanup_target_but_empty_snapshot_record_does_not():
    remote = FakeRemote()
    run_only = _record(release_id=None)
    empty = _record("nightly/v0.9.0-20261008-1", release_id=None)
    empty["runs"] = []
    result = catalog.collect_catalog(remote, _history(run_only, empty))
    assert result.records == ()
    assert [record["tag"] for record in result.orphans] == [TAG]


@pytest.mark.parametrize("mutation", ["repository", "schema", "fields"])
def test_historical_inventory_global_identity_is_strict(mutation):
    document = _history(_record())
    if mutation == "repository":
        document["repository"] = "another/repo"
    elif mutation == "schema":
        document["schema_version"] = 2
    else:
        document["extra"] = True
    with pytest.raises(domain.CleanupRecordError, match="historical cleanup inventory"):
        catalog.collect_catalog(FakeRemote(), document)


@pytest.mark.parametrize(
    "path",
    [
        "/releases",
        "/git/matching-refs/tags/nightly/",
        "/actions/workflows/release-nightly.yml/runs",
    ],
)
def test_global_403_prevents_cleanup_using_an_incomplete_catalog(path):
    remote = FakeRemote()
    remote.errors[PREFIX + path] = RemoteError(403)
    with pytest.raises(RemoteError, match="403"):
        catalog.collect_catalog(remote)


def test_per_version_asset_or_run_403_is_visible_and_other_versions_are_retained():
    remote = FakeRemote()
    first, second = _release(), _release("nightly/v0.9.0-20261008-1", release_id=2)
    remote.releases = [first, second]
    remote.add_asset(first, _record())
    remote.add_asset(
        second, _record(second["name"], release_id=2, run_id=124), asset_id=11
    )
    remote.errors[PREFIX + "/actions/runs/123"] = RemoteError(403)
    result = catalog.collect_catalog(remote)
    assert len(result.records) == 2
    assert TAG in result.blocked
    assert TAG in result.active_tags
    assert second["name"] not in result.blocked


def test_history_and_live_tag_source_conflict_is_not_silently_merged():
    remote = FakeRemote()
    remote.refs = [_ref(sha="b" * 40)]
    result = catalog.collect_catalog(remote, _history(_record(release_id=None)))
    assert TAG in result.blocked
    assert "conflicting source SHAs" in result.blocked[TAG]


def test_conflicting_asset_tag_cannot_invent_an_additional_quota_slot():
    remote = FakeRemote()
    release = _release()
    remote.releases = [release]
    remote.add_asset(release, _record("nightly/v0.9.0-20261008-1"))
    result = catalog.collect_catalog(remote)
    assert [record["tag"] for record in result.records] == [TAG]
    assert "conflicting identities" in result.blocked[TAG]
