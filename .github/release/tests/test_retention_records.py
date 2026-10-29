from __future__ import annotations

import copy
import importlib
import io
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
cleanup = importlib.import_module("ucm_release.cleanup")
inventory_module = importlib.import_module("ucm_release.cleanup_inventory")
execution = cleanup
resources_module = importlib.import_module("ucm_release.cleanup_records")
remote_module = importlib.import_module("ucm_release.cleanup_remote")

REPOSITORY = "release-org/unified-cache-management"
TAG = "draft/v0.8.0-1"
SOURCE_SHA = "a" * 40


def _plan(*, multi_arch: bool = False, dockerhub: bool = False, tag: str = TAG) -> dict:
    coordinates = resources_module.version_config.classify_tag(tag)
    reference = f"ghcr.io/release-org/vllm:v0.23.0-ucm-{coordinates['image_version']}"
    members = (
        [reference + "-amd64", reference + "-arm64"] if multi_arch else [reference]
    )
    return {
        "kind": "ucm-release-plan",
        "route": "release",
        "repository": REPOSITORY,
        "git_tag": tag,
        "release_type": "draft",
        "version": coordinates["version"],
        "publish": {
            "ghcr": {"enabled": True},
            "dockerhub": {"enabled": dockerhub, "namespace": "docker.io/release-org"},
            "chart_oci": {"enabled": True, "namespace": "ghcr.io/release-org/charts"},
        },
        "families": [
            {
                "create_index": multi_arch,
                "published_reference": reference,
                "members": [{"reference": member} for member in members],
            }
        ],
        "chart": {
            "name": "unified-cache-chart",
            "version": coordinates["chart_version"],
        },
    }


def _snapshot(*, run_id: int = 100, resources: tuple[dict, ...] = ()) -> dict:
    return {
        "kind": "ucm-release-targets",
        "schema_version": 1,
        "repository": REPOSITORY,
        "tag": TAG,
        "release_type": "draft",
        "run_id": run_id,
        "source_sha": SOURCE_SHA,
        "version": "0.8.0.dev1",
        "resources": list(resources),
    }


def _run_payload(
    run_id: int = 100, *, status: str = "completed", attempt: int = 1
) -> dict:
    return {
        "id": run_id,
        "repository": {"full_name": REPOSITORY},
        "head_repository": {"full_name": REPOSITORY},
        "path": ".github/workflows/release-tag.yml",
        "head_sha": SOURCE_SHA,
        "head_branch": TAG,
        "event": "push",
        "status": status,
        "run_attempt": attempt,
    }


def _early_failure_jobs(run_id: int = 100) -> list[dict]:
    labels = [
        "Tag Image · ${{ matrix.label }}",
        "Verify directly published image members",
        "Publish index · ${{ matrix.label }}",
        "Publish · Chart OCI",
        "PyPI · ${{ needs.plan.outputs.pypi_disposition }}",
        "Release · Publish backend Wheels, Chart, and Config",
        "Release · Finalize all channels",
    ]
    return [
        {
            "run_id": run_id,
            "name": "Run Release core / " + label,
            "status": "completed",
            "conclusion": conclusion,
        }
        for label, conclusion in [
            ("Plan Wheels, Images, and Chart", "failure"),
            *((label, "skipped") for label in labels),
        ]
    ]


class FakeRemote:
    """In-memory GitHub responses injected through the public I/O interface."""

    repository = REPOSITORY
    owner = "release-org"
    api_base = "https://api.github.com"

    def __init__(self):
        self.releases = []
        self.runs = {}
        self.tag_sha = SOURCE_SHA
        self.tag_object = None
        self.payloads = {}
        self.archives = {}
        self.artifacts = {}
        self.jobs = {}
        self.events = []
        self.present = set()
        self.upload_error = None
        self.corrupt_upload = False
        self.uploaded_ids = set()

    def list_releases(self):
        return self.releases

    def read_release_run(self, run_id):
        return cleanup.ProductionRemote.read_release_run(self, run_id)

    def json_request(self, method, path):
        assert method == "GET"
        self.events.append((method, path))
        if path.endswith("/git/matching-refs/tags/"):
            return [
                {
                    "ref": "refs/tags/" + r["tag_name"],
                    "object": {"type": "commit", "sha": self.tag_sha},
                }
                for r in self.releases
                if not r["tag_name"].startswith("untagged-")
                and self.tag_sha is not None
            ]
        if "/git/ref/tags/" in path:
            if self.tag_sha is None:
                raise execution.RemoteError("missing Tag", status=404)
            return {
                "object": {
                    "type": "tag" if self.tag_object else "commit",
                    "sha": self.tag_object or self.tag_sha,
                }
            }
        if "/git/tags/" in path:
            assert path.endswith("/" + self.tag_object)
            return {"object": {"type": "commit", "sha": self.tag_sha}}
        if "/actions/runs/" in path:
            run_id = int(path.rsplit("/", 1)[-1])
            if run_id not in self.runs:
                raise execution.RemoteError("missing run", status=404)
            return self.runs[run_id]
        if "/releases/" in path:
            release_id = int(path.rsplit("/", 1)[-1])
            matches = [
                release for release in self.releases if release["id"] == release_id
            ]
            if not matches:
                raise execution.RemoteError("missing Release", status=404)
            return matches[0]
        pytest.fail(f"unexpected GitHub JSON request: {path}")

    def request(self, method, path, *, data=None, **_):
        self.events.append((method, path))
        if method == "GET":
            if "/releases/assets/" in path:
                asset_id = int(path.rsplit("/", 1)[-1])
                document = self.payloads[asset_id]
                if self.corrupt_upload and asset_id in self.uploaded_ids:
                    document = dict(document, source_sha="b" * 40)
                if asset_id in self.uploaded_ids:
                    self.events.append(("record-readback", asset_id))
                return json.dumps(document).encode()
            match = re.search(r"/actions/artifacts/([0-9]+)/zip$", path)
            if match:
                return self.archives[int(match[1])]
            pytest.fail(f"unexpected GitHub binary request: {path}")
        assert method == "POST", "inventory must never delete resources"
        self.events.append(("record-persisted", path))
        if self.upload_error is not None:
            raise self.upload_error
        parsed = urllib.parse.urlsplit(path)
        release_id = int(parsed.path.split("/")[-2])
        release = next(item for item in self.releases if item["id"] == release_id)
        name = urllib.parse.parse_qs(parsed.query)["name"][0]
        asset_id = max(self.payloads, default=0) + 1
        asset = {"id": asset_id, "name": name}
        self.payloads[asset_id] = json.loads(data)
        self.uploaded_ids.add(asset_id)
        release["assets"].append(asset)
        return json.dumps(asset).encode()

    def list_pages(self, path, *, key=None):
        if path.endswith("/releases"):
            return self.releases
        if key == "workflow_runs":
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
            values = list(self.runs.values())
            workflow = path.split("/workflows/", 1)[1].split("/", 1)[0]
            return [
                r
                for r in values
                if r["path"].split("@", 1)[0].endswith("/" + workflow)
                and (not query.get("branch") or r["head_branch"] == query["branch"][0])
                and (not query.get("head_sha") or r["head_sha"] == query["head_sha"][0])
            ]
        match = re.search(r"/actions/runs/([0-9]+)/(artifacts|jobs)", path)
        assert match is not None and key == match[2]
        collection = self.artifacts if key == "artifacts" else self.jobs
        return collection.get(int(match[1]), [])

    def ensure_idle(self, run_ids):
        for run_id in sorted(set(run_ids)):
            if run_id in self.runs:
                run = self.json_request(
                    "GET", f"/repos/{REPOSITORY}/actions/runs/{run_id}"
                )
                if run["status"] != "completed":
                    raise execution.ActiveRelease(f"run {run_id} is in progress")

    def probe(self, resource):
        self.events.append(("probe", resource.kind))
        return resource.reference if resource.reference in self.present else None

    def delete(self, resource, state):
        assert state == resource.reference
        self.events.append(("delete", resource.kind))
        self.present.remove(resource.reference)

    def is_absent(self, resource, state):
        return state not in self.present


def _release(*, release_id=10, tag=TAG, assets=()):
    return {
        "id": release_id,
        "tag_name": tag,
        "assets": list(assets),
        "upload_url": f"https://uploads.github.com/repos/{REPOSITORY}/releases/{release_id}/assets{{?name,label}}",
    }


def _install_snapshots(remote, documents):
    assets = []
    for asset_id, (name, document) in enumerate(documents, start=1):
        assets.append({"id": asset_id, "name": name})
        remote.payloads[asset_id] = document
        run_id = (
            document.get("run_id")
            or document.get("actions_run_id")
            or document.get("release", {}).get("actions_run_id")
        )
        attempt = int(name.split("-")[-2]) if name.startswith("release-cleanup-") else 1
        previous_attempt = remote.runs.get(run_id, {}).get("run_attempt", 1)
        remote.runs[run_id] = _run_payload(
            run_id, attempt=max(attempt, previous_attempt)
        )
    remote.releases = [_release(assets=assets)]


def _legacy_remote(*, existing=None, source_sha=SOURCE_SHA):
    remote = FakeRemote()
    if existing is None:
        remote.releases = [_release()]
    else:
        _install_snapshots(remote, [existing])
    remote.runs[100] = dict(_run_payload(), head_sha=source_sha)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("release-plan.json", json.dumps(_plan()))
    remote.archives[20] = archive.getvalue()
    remote.artifacts[100] = [
        {
            "id": 20,
            "name": "ucm-release-plan-run-100",
            "expired": False,
            "workflow_run": {"id": 100, "head_sha": source_sha},
        }
    ]
    return remote


def _inventory(remote):
    return inventory_module.ReleaseInventory(remote)


def _resolve(inventory, *, tag=TAG, release_type="draft", release_id=10):
    catalog = inventory_module.collect_catalog(
        inventory.remote, release_type=release_type
    )
    if tag in catalog.blocked:
        raise execution.CleanupError(catalog.blocked[tag])
    selected = next(record for record in catalog.records if record["tag"] == tag)
    return inventory.resolve_record(selected)


def _make_present(remote, target):
    remote.present = {
        resource.reference
        for resource in target.registry + target.actions + target.releases
    } | {target.tag}


def test_catalog_counts_failed_and_schema6_even_when_evidence_is_invalid():
    remote = FakeRemote()
    nightly = "nightly/v0.8.0-20261009-1"
    remote.releases = [
        _release(release_id=3, assets=[{"id": 30, "name": "release-manifest.json"}]),
        _release(release_id=4, tag=nightly),
    ]
    remote.payloads[30] = {}
    catalog = inventory_module.collect_catalog(remote)
    assert {
        (record["tag"], tuple(record["release_ids"])) for record in catalog.records
    } == {
        (TAG, (3,)),
        (nightly, (4,)),
    }
    assert set(catalog.blocked) == {TAG, nightly}
    assert not any(event[0] in {"POST", "delete"} for event in remote.events)


def test_opened_record_cleans_an_early_failure_with_no_registry_targets():
    remote = FakeRemote()
    _install_snapshots(remote, [("release-cleanup-100-1-opened.json", _snapshot())])
    resolution = _resolve(_inventory(remote))
    assert resolution.pending_recovery == ()
    target = resolution.target
    assert target.registry == ()
    assert target.run_ids == (100,)
    assert [
        (resource.identifier, resource.holds_recovery_data)
        for resource in target.releases
    ] == [(10, True)]


def test_cleanup_target_unions_references_from_all_run_attempts():
    remote = FakeRemote()
    first = {
        "kind": "ghcr-member",
        "reference": "ghcr.io/release-org/vllm:v0.23.0-ucm-0.8.0.dev1",
    }
    second = {
        "kind": "ghcr-member",
        "reference": "ghcr.io/release-org/vllm:v0.23.1-ucm-0.8.0.dev1",
    }
    _install_snapshots(
        remote,
        [
            ("release-cleanup-100-1-opened.json", _snapshot()),
            ("release-cleanup-100-1-planned.json", _snapshot(resources=(first,))),
            (
                "release-cleanup-100-2-planned.json",
                _snapshot(resources=(first, second)),
            ),
            ("release-cleanup-101-1-opened.json", _snapshot(run_id=101)),
        ],
    )
    target = _resolve(_inventory(remote)).target
    assert {resource.reference for resource in target.registry} == {
        first["reference"],
        second["reference"],
    }
    assert target.run_ids == (100, 101)
    assert all(
        resource.identifier == ("v0.23.0-ucm-0.8.0.dev1", "v0.23.1-ucm-0.8.0.dev1")
        for resource in target.registry
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", "other-org/other-repo"),
        ("tag", "draft/v0.8.0-2"),
        (
            "resources",
            [
                {
                    "kind": "github-release",
                    "reference": "ghcr.io/release-org/vllm:0.8.0.dev1",
                }
            ],
        ),
    ],
)
def test_cleanup_record_rejects_foreign_identity_or_unsupported_resources(field, value):
    remote = FakeRemote()
    snapshot = _snapshot()
    snapshot[field] = value
    _install_snapshots(remote, [("release-cleanup-100-1-planned.json", snapshot)])
    with pytest.raises(execution.CleanupError):
        _resolve(_inventory(remote))
    assert not any(event[0] == "POST" for event in remote.events)


def test_save_record_keeps_snapshots_from_earlier_attempts():
    remote = FakeRemote()
    remote.releases = [_release()]
    remote.runs[100] = _run_payload(status="in_progress")
    inventory = _inventory(remote)
    inventory.save_record(TAG, 100, SOURCE_SHA, _plan())
    remote.runs[100]["run_attempt"] = 2
    inventory.save_record(TAG, 100, SOURCE_SHA, _plan(multi_arch=True))
    snapshots = {
        asset["name"]: remote.payloads[asset["id"]]
        for asset in remote.releases[0]["assets"]
    }
    assert set(snapshots) == {
        "release-cleanup-100-1-planned.json",
        "release-cleanup-100-2-planned.json",
    }
    assert len(snapshots["release-cleanup-100-1-planned.json"]["resources"]) == 2
    assert len(snapshots["release-cleanup-100-2-planned.json"]["resources"]) == 4


def test_nightly_opened_record_uses_frozen_tag_commit_after_workflow_start():
    remote = FakeRemote()
    nightly = "nightly/v0.8.0-20261009-1"
    remote.releases = [_release(tag=nightly)]
    remote.runs[100] = dict(
        _run_payload(status="in_progress"),
        path=".github/workflows/release-nightly.yml",
        event="schedule",
        head_branch="develop",
        head_sha="b" * 40,
    )
    inventory = _inventory(remote)
    document = inventory.save_record(nightly, 100, SOURCE_SHA)
    assert document["source_sha"] == SOURCE_SHA
    assert document["resources"] == []
    remote.runs[100]["status"] = "completed"
    target = _resolve(inventory, tag=nightly, release_type="nightly").target
    assert target.registry == ()
    assert target.run_ids == (100,)


def test_record_targets_cli_checks_tag_source_before_writing_remote_assets(
    tmp_path, monkeypatch, capsys
):
    remote = FakeRemote()
    monkeypatch.setattr(cleanup, "_production_remote", lambda arguments: remote)
    result = cleanup.main(
        [
            "record-targets",
            "--tag",
            "nightly/v0.8.0-20261009-1",
            "--release-type",
            "nightly",
            "--run-id",
            "100",
            "--source-sha",
            SOURCE_SHA,
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert result == 2
    assert "missing publication run" in capsys.readouterr().err.casefold()
    assert not any(event[0] == "POST" for event in remote.events)


def test_cleanup_source_commit_peels_an_annotated_tag():
    remote = FakeRemote()
    remote.tag_object = "b" * 40
    remote.releases = [_release()]
    remote.runs[100] = _run_payload()
    document = _inventory(remote).save_record(TAG, 100, SOURCE_SHA)
    assert document["source_sha"] == SOURCE_SHA
    assert [
        path for method, path in remote.events if method == "GET" and "/git/" in path
    ] == [
        f"/repos/{REPOSITORY}/git/ref/tags/{TAG}",
        f"/repos/{REPOSITORY}/git/tags/{remote.tag_object}",
    ]


def test_schema6_manifest_recovers_chart_and_all_image_channels():
    remote = FakeRemote()
    image_tag = "v0.23.0-ucm-0.8.0.dev1"
    manifest = {
        "kind": "ucm-release-manifest",
        "schema_version": 6,
        "tag": TAG,
        "release_type": "draft",
        "actions_run_id": 100,
        "chart_oci": "ghcr.io/release-org/charts/unified-cache-chart:0.8.0-draft.1",
        "runtime_images": {
            channel: {
                "indexes": [prefix + image_tag],
                "members": [
                    prefix + image_tag + "-amd64",
                    prefix + image_tag + "-arm64",
                ],
            }
            for channel, prefix in [
                ("ghcr", "ghcr.io/release-org/vllm:"),
                ("dockerhub", "docker.io/release-org/vllm:"),
            ]
        },
        "github_release_assets": ["release-manifest.json", "ucm.whl"],
    }
    _install_snapshots(remote, [("release-manifest.json", manifest)])
    target = _resolve(_inventory(remote)).target
    assert [resource.kind for resource in target.registry] == [
        "chart-oci",
        "ghcr-index",
        "ghcr-member",
        "ghcr-member",
        "dockerhub-index",
        "dockerhub-member",
        "dockerhub-member",
    ]
    assert target.run_ids == (100,)


def test_schema9_draft_display_url_accepts_untagged_coordinate_only_in_its_repository():
    remote = FakeRemote()
    manifest = json.loads(
        (RELEASE_ROOT / "tests/fixtures/release-manifest.json")
        .read_text()
        .replace("ghcr.io/example/", "ghcr.io/release-org/")
    )
    manifest["release"].update(
        tag=TAG,
        type="draft",
        actions_run_id=100,
        url=f"https://github.com/{REPOSITORY}/releases/tag/untagged-3b8a04ce1a6019d63222",
    )
    _install_snapshots(remote, [("release-manifest.json", manifest)])
    target = _resolve(_inventory(remote)).target
    assert target.run_ids == (100,)
    assert len(target.registry) == 4
    manifest["release"][
        "url"
    ] = "https://github.com/other-org/other-repo/releases/tag/untagged-3b8a04ce1a6019d63222"
    with pytest.raises(execution.CleanupError, match="another repository"):
        _resolve(_inventory(remote))


@pytest.mark.parametrize("expired", [False, True])
def test_missing_or_expired_legacy_plan_blocks_cleanup(expired):
    remote = FakeRemote()
    remote.releases = [_release()]
    remote.runs[100] = _run_payload()
    if expired:
        remote.artifacts[100] = [
            {"id": 20, "name": "ucm-release-plan-run-100", "expired": True}
        ]
    with pytest.raises(
        execution.CleanupError, match="expired|no recoverable release plan"
    ):
        _resolve(_inventory(remote))
    assert not any(event[0] == "POST" for event in remote.events)


def test_legacy_early_failure_is_persisted_without_losing_recorded_partial_targets(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    remote = FakeRemote()
    reference = "ghcr.io/release-org/vllm:v0.23.0-ucm-0.8.0.dev1"
    saved = _snapshot(
        run_id=101, resources=({"kind": "ghcr-member", "reference": reference},)
    )
    _install_snapshots(remote, [("release-cleanup-101-1-planned.json", saved)])
    remote.runs[100] = _run_payload()
    remote.jobs[100] = _early_failure_jobs()
    inventory = _inventory(remote)
    resolution = _resolve(inventory)
    target = resolution.target
    assert target.run_ids == (100, 101)
    assert [resource.reference for resource in target.registry] == [reference]
    assert len(resolution.pending_recovery) == 1
    assert not any(event[0] == "POST" for event in remote.events)
    inventory.persist_recovery(resolution)
    recovered = remote.payloads[max(remote.uploaded_ids)]
    assert recovered["run_id"] == 100 and recovered["resources"] == []
    assert _resolve(inventory).target == target
    _make_present(remote, target)
    assert execution.execute_cleanup(target, remote, sleeper=lambda _: None).completed
    event_names = [event[0] for event in remote.events]
    assert event_names.index("record-readback") < event_names.index("delete")
    assert remote.events[-1] == ("delete", "github-release")


@pytest.mark.parametrize(
    "evidence", ["attempted_publisher", "missing_publisher", "unknown_job"]
)
def test_legacy_without_plan_refuses_incomplete_or_attempted_publication_evidence(
    evidence,
):
    remote = FakeRemote()
    remote.releases = [_release()]
    remote.runs[100] = _run_payload()
    jobs = _early_failure_jobs()
    if evidence == "attempted_publisher":
        attempted = dict(
            jobs[1],
            conclusion="failure",
            name="Run Release core / Tag Image · Runtime / Build, verify, and publish",
        )
        jobs.append(attempted)
    elif evidence == "missing_publisher":
        jobs.pop()
    else:
        jobs.append(dict(jobs[0], name="Run Release core / Unknown publication stage"))
    remote.jobs[100] = jobs
    with pytest.raises(execution.CleanupError, match="no recoverable release plan"):
        _resolve(_inventory(remote))
    assert not any(event[0] == "POST" for event in remote.events)


@pytest.mark.parametrize("recovery_case", ["no_record", "opened_record", "moved_tag"])
def test_recovered_legacy_plan_is_persisted_before_actions_run_deletion(
    tmp_path, monkeypatch, recovery_case
):
    monkeypatch.chdir(tmp_path)
    saved = _snapshot(run_id=101)
    historical_sha = "b" * 40 if recovery_case == "moved_tag" else SOURCE_SHA
    stage = "planned" if recovery_case == "moved_tag" else "opened"
    if recovery_case == "moved_tag":
        saved["resources"] = [
            {
                "kind": "ghcr-member",
                "reference": "ghcr.io/release-org/vllm:v0.23.1-ucm-0.8.0.dev1",
            }
        ]
    existing = (
        None
        if recovery_case == "no_record"
        else (f"release-cleanup-101-1-{stage}.json", saved)
    )
    remote = _legacy_remote(existing=existing, source_sha=historical_sha)
    inventory = _inventory(remote)
    resolution = _resolve(inventory)
    target = resolution.target
    assert len(target.registry) == (3 if recovery_case == "moved_tag" else 2)
    assert target.run_ids == ((100,) if existing is None else (100, 101))
    assert len(resolution.pending_recovery) == 1
    assert resolution.pending_recovery[0].source_sha == historical_sha
    assert not any(event[0] == "POST" for event in remote.events)
    inventory.persist_recovery(resolution)
    assert remote.payloads[max(remote.uploaded_ids)]["source_sha"] == historical_sha
    assert _resolve(inventory).target == target
    _make_present(remote, target)
    assert execution.execute_cleanup(target, remote, sleeper=lambda _: None).completed
    event_names = [event[0] for event in remote.events]
    assert (
        event_names.index("record-persisted")
        < event_names.index("record-readback")
        < event_names.index("delete")
    )
    assert remote.events[-1] == ("delete", "github-release")


def test_unrecorded_active_publication_prevents_all_retention_deletes():
    remote = FakeRemote()
    _install_snapshots(remote, [("release-cleanup-100-1-opened.json", _snapshot())])
    remote.runs[101] = _run_payload(101, status="queued")
    arguments = SimpleNamespace(
        current_tag="draft/v0.8.0-2",
        release_type="draft",
        max_count=1,
        pypi_enabled=False,
        fail_resource=None,
    )
    assert cleanup._run_retention(arguments, remote) == []
    assert not any(event[0] in {"POST", "probe", "delete"} for event in remote.events)
    assert (
        TAG
        in inventory_module.collect_catalog(remote, release_type="draft").active_tags
    )


@pytest.mark.parametrize("multi_arch,dockerhub", [(False, False), (True, True)])
def test_plan_projection_covers_partial_publication_without_disabled_targets(
    multi_arch, dockerhub
):
    resources = resources_module.plan_resources(
        _plan(multi_arch=multi_arch, dockerhub=dockerhub),
        repository=REPOSITORY,
        tag=TAG,
    )
    coordinates = [(resource.kind, resource.reference) for resource in resources]
    assert coordinates[0] == (
        "chart-oci",
        "ghcr.io/release-org/charts/unified-cache-chart:0.8.0-draft.1",
    )
    reference = "ghcr.io/release-org/vllm:v0.23.0-ucm-0.8.0.dev1"
    if multi_arch:
        assert coordinates[1:] == [
            ("ghcr-index", reference),
            ("ghcr-member", reference + "-amd64"),
            ("ghcr-member", reference + "-arm64"),
            ("dockerhub-index", reference.replace("ghcr.io", "docker.io")),
            ("dockerhub-member", reference.replace("ghcr.io", "docker.io") + "-amd64"),
            ("dockerhub-member", reference.replace("ghcr.io", "docker.io") + "-arm64"),
        ]
    else:
        assert coordinates[1:] == [("ghcr-member", reference)]


def test_plan_projection_rejects_repository_and_tag_mismatch():
    for repository, tag in [
        ("other-org/other-repo", TAG),
        (REPOSITORY, "draft/v0.8.0-2"),
    ]:
        with pytest.raises(execution.CleanupError, match="identity differs"):
            resources_module.plan_resources(_plan(), repository=repository, tag=tag)


@pytest.fixture
def custom_draft_plan():
    return _plan(multi_arch=True, tag="draft/v0.8.0-test")


def test_custom_draft_cleanup_preserves_full_version_and_resource_coordinates(
    custom_draft_plan,
):
    plan = custom_draft_plan
    snapshot = resources_module.new_record(
        REPOSITORY, plan["git_tag"], SOURCE_SHA, 100, 1, plan=plan
    )
    record = resources_module.decode_snapshot(
        snapshot, REPOSITORY, "release-cleanup-100-1-planned.json"
    )
    assert snapshot["version"] == "0.8.0.dev0+draft.test"
    assert record["versions"] == [snapshot["version"]]
    family = plan["families"][0]
    assert {item["reference"] for item in record["resources"]} == {
        f"ghcr.io/release-org/charts/unified-cache-chart:{plan['chart']['version']}",
        family["published_reference"],
        *(member["reference"] for member in family["members"]),
    }
    snapshot["version"] = "0.8.0.dev0"
    with pytest.raises(resources_module.CleanupRecordError, match="version differs"):
        resources_module.decode_snapshot(
            snapshot, REPOSITORY, "release-cleanup-100-1-planned.json"
        )


@pytest.mark.parametrize("resource_index", [0, 1], ids=["chart", "image"])
def test_custom_draft_cleanup_record_cannot_claim_another_labels_resources(
    custom_draft_plan, resource_index
):
    plan = custom_draft_plan
    record = resources_module.record_from_plan(
        resources_module.record_for_tag(
            REPOSITORY, plan["git_tag"], SOURCE_SHA, 100, 1
        ),
        plan,
    )
    resource = record["resources"][resource_index]
    resource["reference"] = resource["reference"].replace("draft.test", "draft.other")
    with pytest.raises(resources_module.CleanupRecordError, match="not owned"):
        resources_module.validate_record(record, REPOSITORY)


def test_custom_draft_retention_uses_release_creation_order():
    tags = ["draft/v0.8.0-z-last-name", "draft/v0.8.0-a-first-name"]
    records = [
        resources_module.record_for_tag(REPOSITORY, tag, SOURCE_SHA, index, 1, index)
        for index, tag in enumerate(tags, 1)
    ]
    selection = resources_module.select_retention(records, 1, release_type="draft")
    assert [record["tag"] for record in selection.candidates] == tags[:1]


def test_retention_defers_active_oldest_without_deleting_a_retained_newer_tag():
    remote = FakeRemote()
    remote.releases = [
        _release(release_id=1),
        _release(release_id=2, tag="draft/v0.8.0-2"),
    ]
    remote.runs[100] = _run_payload(status="in_progress")
    arguments = SimpleNamespace(
        current_tag="draft/v0.8.0-3",
        release_type="draft",
        max_count=2,
        pypi_enabled=False,
        fail_resource=None,
    )
    assert cleanup._run_retention(arguments, remote) == []
    assert not any(event[0] in {"POST", "probe", "delete"} for event in remote.events)


def test_cleanup_rechecks_active_run_before_any_registry_delete():
    remote = FakeRemote()
    reference = "ghcr.io/release-org/vllm:v0.23.0-ucm-0.8.0.dev1"
    _install_snapshots(
        remote,
        [
            (
                "release-cleanup-100-1-planned.json",
                _snapshot(resources=({"kind": "ghcr-member", "reference": reference},)),
            )
        ],
    )
    target = _resolve(_inventory(remote)).target
    remote.runs[100]["status"] = "queued"
    remote.events.clear()
    with pytest.raises(execution.ActiveRelease):
        execution.execute_cleanup(target, remote, sleeper=lambda _: None)
    assert remote.events == [("GET", f"/repos/{REPOSITORY}/actions/runs/100")]


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", {"full_name": "other-org/other-repo"}),
        ("path", ".github/workflows/lint.yml"),
    ],
)
def test_remote_rejects_unowned_completed_runs_before_deletion(
    monkeypatch, field, value
):
    remote = cleanup.ProductionRemote(REPOSITORY, "test-token")
    run = _run_payload()
    run[field] = value
    requests = []

    def json_request(method, path):
        requests.append((method, path))
        return run

    monkeypatch.setattr(remote, "json_request", json_request)
    monkeypatch.setattr(
        remote, "probe", lambda resource: pytest.fail("unowned run resource was probed")
    )
    monkeypatch.setattr(
        remote,
        "delete",
        lambda resource, state: pytest.fail("unowned run resource was deleted"),
    )
    target = execution.CleanupTarget(
        TAG,
        (execution.Resource("ghcr-member", "ghcr.io/release-org/vllm:0.8.0.dev1"),),
        (
            execution.Resource(
                "actions-run", f"https://github.com/{REPOSITORY}/actions/runs/100", 100
            ),
        ),
        (),
    )
    with pytest.raises(execution.CleanupError, match="owned by a UCM release workflow"):
        execution.execute_cleanup(target, remote, sleeper=lambda _: None)
    assert requests == [("GET", f"/repos/{REPOSITORY}/actions/runs/100")]


@pytest.mark.parametrize("command", ["tag", "retention"])
def test_dry_run_resolves_legacy_targets_without_uploading_recovery(command, capsys):
    remote = _legacy_remote()
    arguments = SimpleNamespace(
        tag=TAG,
        current_tag="draft/v0.8.0-2",
        release_type="draft",
        max_count=1,
        pypi_enabled=False,
        fail_resource=None,
        dry_run=True,
    )
    runner = cleanup._run_tag if command == "tag" else cleanup._run_retention
    assert runner(arguments, remote) == []
    assert "would-delete" in capsys.readouterr().out
    assert remote.releases[0]["assets"] == []
    assert not any(event[0] in {"POST", "delete"} for event in remote.events)


@pytest.mark.parametrize("failure", ["upload", "readback"])
def test_recovery_failure_blocks_resource_deletion(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    remote = _legacy_remote()
    if failure == "upload":
        remote.upload_error = execution.RemoteError("upload forbidden", status=403)
    else:
        remote.corrupt_upload = True
    arguments = SimpleNamespace(tag=TAG, fail_resource=None, dry_run=False)
    assert cleanup._run_tag(arguments, remote)
    assert not any(event[0] in {"probe", "delete"} for event in remote.events)


def test_github_download_does_not_forward_token_to_redirected_storage():
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b"asset"

    class Opener:
        def open(self, request, *, timeout):
            requests.append(request)
            assert timeout == 60
            return Response()

    remote = cleanup.ProductionRemote(REPOSITORY, "test-token", opener=Opener())
    assert remote.request("GET", f"/repos/{REPOSITORY}/releases/assets/10") == b"asset"
    source = requests[0]
    assert source.get_header("Authorization") == "Bearer test-token"
    redirected = urllib.request.HTTPRedirectHandler().redirect_request(
        source, None, 302, "Found", {}, "https://storage.example/release-asset"
    )
    assert redirected.get_header("Authorization") is None


TYPES = ["draft", "nightly"]


def _tag(kind, number):
    return (
        f"draft/v0.8.0-{number}"
        if kind == "draft"
        else f"nightly/v0.8.0-202610{number:02d}-1"
    )


def _world(kind, count=3):
    remote = FakeRemote()
    for number in range(1, count + 1):
        tag = _tag(kind, number)
        run_id = 100 + number
        remote.runs[run_id] = dict(_run_payload(run_id), head_branch=tag)
        document = resources_module.new_record(REPOSITORY, tag, SOURCE_SHA, run_id, 1)
        name = resources_module.record_basename(document, 1)
        remote.payloads[number] = document
        remote.releases.append(
            _release(release_id=number, tag=tag, assets=[{"id": number, "name": name}])
        )
        remote.present.update(
            {
                tag,
                f"{tag}#{number}",
                f"https://github.com/{REPOSITORY}/actions/runs/{run_id}",
            }
        )
    return remote


@pytest.mark.parametrize("kind", TYPES)
def test_local_record_cli_saves_a_bound_snapshot_without_credentials(tmp_path, kind):
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in {"GH_TOKEN", "GITHUB_TOKEN", "PYTHONPATH"}
    }
    command = [
        sys.executable,
        str(RELEASE_ROOT / "ucm_release/cleanup.py"),
        "record",
        "--repository",
        REPOSITORY,
        "--tag",
        _tag(kind, 1),
        "--source-sha",
        SOURCE_SHA,
        "--run-id",
        "101",
        "--run-attempt",
        "2",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "release-cleanup-101-2-opened.json"
    document = resources_module.decode_snapshot(
        json.loads(output.read_text()), REPOSITORY, output.name
    )
    assert document["runs"] == [{"id": 101, "attempt": 2}]
    assert document["resources"] == []


@pytest.mark.parametrize("kind", TYPES)
def test_retention_cli_defers_active_excess_versions(tmp_path, monkeypatch, kind):
    remote = _world(kind)
    remote.runs[101]["status"] = "queued"
    monkeypatch.setattr(cleanup, "_production_remote", lambda arguments: remote)
    report = tmp_path / "report.json"
    status = cleanup.main(
        [
            "retention",
            "--repository",
            REPOSITORY,
            "--release-type",
            kind,
            "--max-count",
            "2",
            "--pypi-enabled",
            "false",
            "--report",
            str(report),
        ]
    )
    assert status == 0
    value = json.loads(report.read_text())
    assert value["status"] == "deferred"
    assert sorted(item["status"] for item in value["results"]) == [
        "deferred",
        "kept",
        "kept",
    ]
    assert not any(event[0] in {"POST", "delete"} for event in remote.events)


@pytest.mark.parametrize("kind", TYPES)
def test_one_blocked_version_does_not_prevent_independent_cleanup(
    tmp_path, monkeypatch, kind
):
    remote = _world(kind)
    remote.payloads[1] = dict(remote.payloads[1], repository="foreign/repo")
    monkeypatch.setattr(cleanup, "_production_remote", lambda arguments: remote)
    report = tmp_path / "report.json"
    status = cleanup.main(
        [
            "retention",
            "--repository",
            REPOSITORY,
            "--release-type",
            kind,
            "--max-count",
            "1",
            "--pypi-enabled",
            "false",
            "--report",
            str(report),
        ]
    )
    assert status == 1
    assert _tag(kind, 1) in remote.present
    assert _tag(kind, 2) not in remote.present
    assert {item["status"] for item in json.loads(report.read_text())["results"]} == {
        "kept",
        "blocked",
        "deleted",
    }


@pytest.mark.parametrize("kind", TYPES)
def test_tag_preview_uses_the_same_target_without_remote_mutation(
    tmp_path, monkeypatch, kind
):
    remote = _world(kind, 1)
    original = remote.present.copy()
    monkeypatch.setattr(cleanup, "_production_remote", lambda arguments: remote)
    report = tmp_path / "report.json"
    status = cleanup.main(
        [
            "tag",
            "--repository",
            REPOSITORY,
            "--tag",
            _tag(kind, 1),
            "--dry-run",
            "--report",
            str(report),
        ]
    )
    assert status == 0
    assert remote.present == original
    assert not any(event[0] in {"POST", "delete"} for event in remote.events)
    assert json.loads(report.read_text())["results"][0]["status"] == "would-delete"


@pytest.mark.parametrize("kind", TYPES)
def test_incomplete_release_snapshot_keeps_the_artifact_run_until_last(kind):
    remote = _world(kind, 1)
    saved = resources_module.decode_snapshot(
        remote.payloads[1], REPOSITORY, "release-cleanup-101-1-opened.json", 1
    )
    remote.runs[101]["run_attempt"] = 2
    latest = resources_module.record_for_tag(
        REPOSITORY, _tag(kind, 1), SOURCE_SHA, 101, 2, 1
    )
    target = (
        inventory_module.ReleaseInventory(remote)
        .resolve_record(resources_module.merge_records([saved, latest], REPOSITORY)[0])
        .target
    )
    assert target.actions[0].holds_recovery_data is True
    assert target.releases[0].holds_recovery_data is False
    remote.payloads[2] = resources_module.new_record(
        REPOSITORY, _tag(kind, 1), SOURCE_SHA, 101, 2
    )
    remote.releases[0]["assets"].append(
        {"id": 2, "name": "release-cleanup-101-2-opened.json"}
    )
    complete = (
        inventory_module.ReleaseInventory(remote)
        .resolve_record(resources_module.merge_records([saved, latest], REPOSITORY)[0])
        .target
    )
    assert complete.actions[0].holds_recovery_data is False
    assert complete.releases[0].holds_recovery_data is True


def test_signed_download_redirect_does_not_forward_the_credential():
    request = urllib.request.Request(
        "https://api.github.com/repos/release-org/repo/actions/artifacts/1/zip",
        headers={
            "Authorization": "Bearer test-only",
            "Accept": "application/octet-stream",
        },
    )
    handler = remote_module._GitHubRedirectHandler()
    redirected = handler.redirect_request(
        request, None, 302, "Found", {}, "https://blob.example.test/signed.zip"
    )
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Accept") == "application/octet-stream"
    with pytest.raises(cleanup.CleanupError):
        handler.redirect_request(
            request, None, 302, "Found", {}, "http://blob.example.test/file"
        )


@pytest.mark.parametrize("kind", TYPES)
def test_unowned_untagged_draft_with_the_same_name_is_not_a_cleanup_target(kind):
    remote = _world(kind, 1)
    manual = _release(release_id=99, tag="untagged-user")
    manual.update(name=_tag(kind, 1), draft=True, body="user notes")
    remote.releases.append(manual)
    record = resources_module.decode_snapshot(
        remote.payloads[1], REPOSITORY, "release-cleanup-101-1-opened.json", 1
    )
    target = inventory_module.ReleaseInventory(remote).resolve_record(record).target
    assert [release.identifier for release in target.releases] == [1]
    assert not any(event[0] in {"POST", "delete"} for event in remote.events)


def test_pending_plan_does_not_cover_another_attempt_recovered_only_from_artifacts():
    remote = _legacy_remote()
    remote.runs[101] = _run_payload(101)
    snapshot = resources_module.new_record(REPOSITORY, TAG, SOURCE_SHA, 101, 1)
    snapshot["resources"] = [
        {
            "kind": "ghcr-member",
            "reference": "ghcr.io/release-org/vllm:v0.23.1-ucm-0.8.0.dev1",
        }
    ]
    record = resources_module.decode_snapshot(
        snapshot, REPOSITORY, "release-cleanup-101-1-planned.json", 10
    )
    resolution = inventory_module.ReleaseInventory(remote).resolve_record(record)
    assert [item.run["id"] for item in resolution.pending_recovery] == [100]
    assert all(action.holds_recovery_data for action in resolution.target.actions)
    assert not resolution.target.releases[0].holds_recovery_data


def test_scheduled_artifact_preserves_the_frozen_tag_commit():
    remote = _world("nightly", 1)
    run = remote.runs[101]
    run.update(
        path=".github/workflows/release-nightly.yml",
        event="schedule",
        head_branch="develop",
        head_sha="b" * 40,
    )
    remote.releases[0]["assets"] = []
    snapshot = remote.payloads[1]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("release-cleanup-101-1-opened.json", json.dumps(snapshot))
    remote.archives[50] = archive.getvalue()
    remote.artifacts[101] = [
        {
            "id": 50,
            "name": "ucm-release-cleanup-run-101-attempt-1",
            "expired": False,
            "workflow_run": {"id": 101},
        }
    ]
    catalog = inventory_module.collect_catalog(remote, release_type="nightly")
    assert catalog.blocked == {}
    assert catalog.records[0]["source_sha"] == SOURCE_SHA
    assert catalog.records[0]["runs"] == [{"id": 101, "attempt": 1}]


def test_legacy_preview_report_uses_the_resolved_resources(tmp_path, monkeypatch):
    remote = _legacy_remote()
    monkeypatch.setattr(cleanup, "_production_remote", lambda arguments: remote)
    report = tmp_path / "report.json"
    assert (
        cleanup.main(
            [
                "tag",
                "--repository",
                REPOSITORY,
                "--tag",
                TAG,
                "--dry-run",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    result = json.loads(report.read_text())["results"][0]
    assert result["registry_references"] == 2
    assert result["run_ids"] == [100]
    assert result["release_ids"] == [10]


SNAPSHOT_NIGHTLY_TAG = "nightly/v0.9.0-20261009-1"
SNAPSHOT_NIGHTLY_VERSION = "0.9.0.dev20261009001"


class SnapshotRemote:
    """Keep GitHub assets in memory; require local recovery before upload."""

    repository = REPOSITORY
    api_base = "https://api.github.com"

    def __init__(self, release_type, tag, version, *, output_dir):
        self.release_type = release_type
        self.tag = tag
        self.version = version
        self.output_dir = output_dir
        self.events = []
        self.payloads = {}
        self.failure = None
        self.run = {
            "id": 42,
            "repository": {"full_name": REPOSITORY},
            "path": f".github/workflows/release-{'nightly' if release_type == 'nightly' else 'tag'}.yml",
            "head_sha": SOURCE_SHA,
            "head_branch": tag,
            "event": "schedule" if release_type == "nightly" else "push",
            "status": "in_progress",
            "run_attempt": 1,
        }
        self.release = {
            "id": 10,
            "tag_name": tag,
            "draft": True,
            "name": tag,
            "upload_url": (
                f"https://uploads.github.com/repos/{REPOSITORY}"
                "/releases/10/assets{?name,label}"
            ),
            "assets": [],
        }

    def json_request(self, method, path):
        self.events.append((method, path))
        assert method == "GET"
        if path == f"/repos/{REPOSITORY}/git/ref/tags/{self.tag}":
            return {"object": {"type": "commit", "sha": SOURCE_SHA}}
        assert path == f"/repos/{REPOSITORY}/releases/10"
        return copy.deepcopy(self.release)

    def read_release_run(self, run_id):
        assert run_id == self.run["id"]
        self.events.append(("GET", f"/repos/{REPOSITORY}/actions/runs/{run_id}"))
        return copy.deepcopy(self.run)

    def list_releases(self):
        self.events.append(("GET", f"/repos/{REPOSITORY}/releases"))
        return [copy.deepcopy(self.release)]

    def request(self, method, path, *, accept=None, data=None, content_type=None):
        self.events.append((method, path))
        if method == "POST":
            parsed = urllib.parse.urlsplit(path)
            assert parsed.path == f"/repos/{REPOSITORY}/releases/10/assets"
            name = urllib.parse.parse_qs(parsed.query)["name"][0]
            local_record = self.output_dir / name
            assert local_record.exists(), "local recovery must precede upload"
            assert data == local_record.read_bytes()
            assert content_type == "application/json"
            if self.failure == "upload":
                raise remote_module.RemoteError("synthetic upload failure", status=503)
            asset = {"id": 51 + len(self.payloads), "name": name}
            self.payloads[asset["id"]] = data
            self.release["assets"].append(asset)
            return json.dumps(asset).encode()
        assert method == "GET", "snapshot persistence must never delete assets"
        assert accept == "application/octet-stream"
        assert path.startswith(f"/repos/{REPOSITORY}/releases/assets/")
        if self.failure == "readback":
            raise remote_module.RemoteError("synthetic readback failure", status=503)
        asset_id = int(path.rsplit("/", 1)[-1])
        if self.failure == "mismatch":
            document = json.loads(self.payloads[asset_id])
            return json.dumps(dict(document, source_sha="b" * 40)).encode()
        return self.payloads[asset_id]


@pytest.fixture(
    params=[
        ("nightly", SNAPSHOT_NIGHTLY_TAG, SNAPSHOT_NIGHTLY_VERSION),
        ("draft", "draft/v0.9.0-1", "0.9.0.dev1"),
    ],
    ids=["nightly", "draft"],
)
def snapshot_remote(request, tmp_path):
    return SnapshotRemote(*request.param, output_dir=tmp_path / "out/cleanup")


def _snapshot_reference(snapshot_remote, family="v1"):
    return f"ghcr.io/release-org/vllm-openai:{family}-ucm-{snapshot_remote.version}"


def _snapshot_plan(snapshot_remote, *, family="v1"):
    reference = _snapshot_reference(snapshot_remote, family)
    return {
        "kind": "ucm-release-plan",
        "repository": REPOSITORY,
        "route": "release",
        "release_type": snapshot_remote.release_type,
        "git_tag": snapshot_remote.tag,
        "version": snapshot_remote.version,
        "publish": {
            "ghcr": {"enabled": True},
            "dockerhub": {"enabled": False},
            "chart_oci": {"enabled": False},
        },
        "families": [
            {
                "create_index": False,
                "published_reference": reference,
                "members": [{"cpu_arch": "amd64", "reference": reference}],
            }
        ],
    }


def _save_snapshot(snapshot_remote, plan=None):
    return inventory_module.ReleaseInventory(snapshot_remote).save_record(
        snapshot_remote.tag,
        snapshot_remote.run["id"],
        SOURCE_SHA,
        plan,
        run_attempt=snapshot_remote.run["run_attempt"],
        release_id=10,
        output_dir=snapshot_remote.output_dir,
    )


@pytest.mark.parametrize("planned", [False, True], ids=["opened", "planned"])
def test_snapshot_is_saved_locally_uploaded_and_read_back(snapshot_remote, planned):
    document = _save_snapshot(
        snapshot_remote, _snapshot_plan(snapshot_remote) if planned else None
    )
    name = f"release-cleanup-42-1-{'planned' if planned else 'opened'}.json"
    assert document == {
        "kind": "ucm-release-targets",
        "schema_version": 1,
        "repository": REPOSITORY,
        "tag": snapshot_remote.tag,
        "release_type": snapshot_remote.release_type,
        "run_id": 42,
        "source_sha": SOURCE_SHA,
        "version": snapshot_remote.version,
        "resources": (
            [{"kind": "ghcr-member", "reference": _snapshot_reference(snapshot_remote)}]
            if planned
            else []
        ),
    }
    assert json.loads((snapshot_remote.output_dir / name).read_bytes()) == document
    assert snapshot_remote.release["assets"] == [{"id": 51, "name": name}]
    assert [
        (method, path)
        for method, path in snapshot_remote.events
        if method != "GET" or "/releases/assets/" in path
    ] == [
        (
            "POST",
            f"https://uploads.github.com/repos/{REPOSITORY}/releases/10/assets?name={name}",
        ),
        ("GET", f"/repos/{REPOSITORY}/releases/assets/51"),
    ]


@pytest.mark.parametrize("planned", [False, True], ids=["opened", "planned"])
def test_repeating_same_attempt_stage_reads_its_existing_asset(
    snapshot_remote, planned
):
    plan = _snapshot_plan(snapshot_remote) if planned else None
    expected = _save_snapshot(snapshot_remote, plan)
    saved_payloads = dict(snapshot_remote.payloads)
    snapshot_remote.events.clear()
    assert _save_snapshot(snapshot_remote, plan) == expected
    assert snapshot_remote.payloads == saved_payloads
    assert len(snapshot_remote.release["assets"]) == 1
    assert all(method == "GET" for method, _ in snapshot_remote.events)
    assert snapshot_remote.events[-1] == (
        "GET",
        f"/repos/{REPOSITORY}/releases/assets/51",
    )


def test_conflicting_same_attempt_stage_cannot_replace_the_saved_snapshot(
    snapshot_remote,
):
    _save_snapshot(snapshot_remote, _snapshot_plan(snapshot_remote))
    saved_payloads = dict(snapshot_remote.payloads)
    snapshot_remote.events.clear()
    with pytest.raises(resources_module.CleanupError, match="snapshot conflicts"):
        _save_snapshot(snapshot_remote, _snapshot_plan(snapshot_remote, family="v2"))
    assert snapshot_remote.payloads == saved_payloads
    assert len(snapshot_remote.release["assets"]) == 1
    assert all(method == "GET" for method, _ in snapshot_remote.events)


def test_all_attempts_and_stages_remain_available_for_inventory_union(snapshot_remote):
    _save_snapshot(snapshot_remote)
    _save_snapshot(snapshot_remote, _snapshot_plan(snapshot_remote))
    first_attempt_payloads = dict(snapshot_remote.payloads)
    snapshot_remote.run["run_attempt"] = 2
    _save_snapshot(snapshot_remote)
    _save_snapshot(snapshot_remote, _snapshot_plan(snapshot_remote, family="v2"))
    assert all(
        snapshot_remote.payloads[asset_id] == payload
        for asset_id, payload in first_attempt_payloads.items()
    )
    expected_names = {
        f"release-cleanup-42-{attempt}-{stage}.json"
        for attempt in (1, 2)
        for stage in ("opened", "planned")
    }
    assert {
        asset["name"] for asset in snapshot_remote.release["assets"]
    } == expected_names
    assert {
        path.name for path in snapshot_remote.output_dir.iterdir()
    } == expected_names
    inventory = inventory_module.ReleaseInventory(snapshot_remote)
    records = [
        inventory.decode_asset(asset, 10) for asset in snapshot_remote.release["assets"]
    ]
    merged = resources_module.merge_records(records, REPOSITORY)[0]
    assert merged["release_ids"] == [10]
    assert merged["runs"] == [{"id": 42, "attempt": 1}, {"id": 42, "attempt": 2}]
    assert merged["resources"] == [
        {
            "kind": "ghcr-member",
            "reference": _snapshot_reference(snapshot_remote, family),
        }
        for family in ("v1", "v2")
    ]
    assert merged["attempt_sources"] == [
        {"id": 42, "attempt": attempt, "source_sha": SOURCE_SHA} for attempt in (1, 2)
    ]
    assert not any(method == "DELETE" for method, _ in snapshot_remote.events)


@pytest.mark.parametrize("failure", ["upload", "readback", "mismatch"])
def test_failed_upload_or_readback_leaves_a_local_recovery_snapshot(
    snapshot_remote, failure
):
    snapshot_remote.failure = failure
    with pytest.raises(resources_module.CleanupError, match="failure|readback differs"):
        _save_snapshot(snapshot_remote, _snapshot_plan(snapshot_remote))
    output = snapshot_remote.output_dir / "release-cleanup-42-1-planned.json"
    document = json.loads(output.read_bytes())
    recovered = resources_module.decode_snapshot(document, REPOSITORY, output.name)
    assert recovered["tag"] == snapshot_remote.tag
    assert recovered["runs"] == [{"id": 42, "attempt": 1}]
    assert recovered["resources"] == [
        {"kind": "ghcr-member", "reference": _snapshot_reference(snapshot_remote)}
    ]
    assert any(method == "POST" for method, _ in snapshot_remote.events)
    assert not any(method == "DELETE" for method, _ in snapshot_remote.events)


@pytest.mark.parametrize("field,value", [("tag_name", "other-tag"), ("id", 11)])
def test_selected_release_must_match_its_tag_and_id_before_upload(
    snapshot_remote, field, value
):
    snapshot_remote.release[field] = value
    if field == "tag_name":
        snapshot_remote.release["name"] = "other-tag"
    with pytest.raises(
        resources_module.CleanupError, match="does not match its Tag and ID"
    ):
        _save_snapshot(snapshot_remote)
    assert snapshot_remote.release["assets"] == []
    assert all(method == "GET" for method, _ in snapshot_remote.events)


def test_historical_nightly_aggregate_remains_readable_without_rewriting(tmp_path):
    snapshot_remote = SnapshotRemote(
        "nightly",
        SNAPSHOT_NIGHTLY_TAG,
        SNAPSHOT_NIGHTLY_VERSION,
        output_dir=tmp_path / "out/cleanup",
    )
    snapshot_remote.run["run_attempt"] = 2
    aggregate = {
        "kind": "ucm-release-cleanup",
        "schema_version": 1,
        "repository": REPOSITORY,
        "tag": SNAPSHOT_NIGHTLY_TAG,
        "source_sha": SOURCE_SHA,
        "release_ids": [9, 10],
        "runs": [{"id": 42, "attempt": 1}, {"id": 42, "attempt": 2}],
        "resources": [
            {"kind": "ghcr-member", "reference": _snapshot_reference(snapshot_remote)}
        ],
    }
    snapshot_remote.payloads[50] = json.dumps(aggregate).encode()
    asset = {"id": 50, "name": "release-cleanup.json"}
    decoded = inventory_module.ReleaseInventory(snapshot_remote).decode_asset(asset, 10)
    assert decoded["release_ids"] == aggregate["release_ids"]
    assert decoded["runs"] == aggregate["runs"]
    assert decoded["resources"] == aggregate["resources"]
    assert json.loads(snapshot_remote.payloads[50]) == aggregate
    assert all(method == "GET" for method, _ in snapshot_remote.events)
    assert not snapshot_remote.output_dir.exists()
