from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
cleanup = importlib.import_module("ucm_release.cleanup")
records_ops = importlib.import_module("ucm_release.cleanup_records")
remote_ops = importlib.import_module("ucm_release.cleanup_remote")
manifest_ops = importlib.import_module("ucm_release.manifest")
inventory_ops = importlib.import_module("ucm_release.cleanup_inventory")


@pytest.mark.parametrize("script", ["cleanup.py", "release.py"])
def test_release_entrypoints_run_by_filename_without_pythonpath(tmp_path, script):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(RELEASE_ROOT / "ucm_release" / script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def _manifest(
    tag="draft/v0.8.0-3",
    *,
    release_type="draft",
    chart="ghcr.io/release-org/charts/unified-cache-chart:0.8.0-draft.3",
    ghcr_members=None,
    ghcr_indexes=None,
    dockerhub_members=None,
    dockerhub_indexes=None,
):
    manifest = json.loads(
        (RELEASE_ROOT / "tests/fixtures/release-manifest.json").read_text()
    )
    manifest["release"].update(tag=tag, type=release_type, actions_run_id=12345)
    manifest["chart"]["oci"] = chart
    template = manifest["images"][0]
    manifest["images"] = []
    channels = {
        "ghcr": (
            (
                ghcr_members
                if ghcr_members is not None
                else [
                    "ghcr.io/release-org/vllm-openai:v0.23.0-amd64",
                    "ghcr.io/release-org/vllm-openai:v0.23.0-arm64",
                ]
            ),
            (
                ghcr_indexes
                if ghcr_indexes is not None
                else ["ghcr.io/release-org/vllm-openai:v0.23.0"]
            ),
        ),
        "dockerhub": (dockerhub_members or [], dockerhub_indexes or []),
    }
    for channel, (members, indexes) in channels.items():
        for reference in indexes or members:
            image = json.loads(json.dumps(template))
            image["id"] = f"{channel}-{len(manifest['images'])}"
            image["publications"] = {"ghcr": None, "dockerhub": None}
            selected = members if indexes else [reference]
            image["publications"][channel] = {
                "pull": reference,
                "multi_arch": bool(indexes),
                "members": [
                    {"architecture": architecture, "reference": member}
                    for architecture, member in zip(("amd64", "arm64"), selected)
                ],
            }
            manifest["images"].append(image)
    return manifest


def _record(manifest: dict[str, object], release_id: int) -> dict[str, object]:
    return records_ops.record_for_tag(
        "release-org/unified-cache-management",
        manifest["release"]["tag"],
        "a" * 40,
        manifest["release"]["actions_run_id"],
        1,
        release_id,
    )


def _target(manifest: dict[str, object], releases) -> records_ops.CleanupTarget:
    tag = manifest["release"]["tag"]
    run_id = manifest["release"]["actions_run_id"]
    run = cleanup.Resource(
        "actions-run",
        f"https://github.com/release-org/unified-cache-management/actions/runs/{run_id}",
        run_id,
    )
    return cleanup.CleanupTarget(
        tag, tuple(records_ops.registry_resources(manifest)), (run,), tuple(releases)
    )


class FakeRemote:
    repository = "release-org/unified-cache-management"
    owner = "release-org"
    api_base = "https://api.github.com"

    def __init__(
        self,
        *,
        present: set[str] | None = None,
        releases: list[cleanup.Resource] | None = None,
    ) -> None:
        self.present = set(present or set())
        self.releases = list(releases or [])
        self.probe_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.release_calls: list[str] = []
        self.probe_errors: dict[str, list[BaseException]] = defaultdict(list)
        self.delete_errors: dict[str, list[BaseException]] = defaultdict(list)
        self.release_errors: list[BaseException] = []

    read_release_run = remote_ops.ProductionRemote.read_release_run

    def ensure_idle(self, run_ids) -> None:
        pass

    def probe(self, resource: cleanup.Resource) -> object | None:
        self.probe_calls.append(resource.reference)
        if self.probe_errors[resource.reference]:
            raise self.probe_errors[resource.reference].pop(0)
        return resource.reference if resource.reference in self.present else None

    def delete(self, resource: cleanup.Resource, state: object) -> None:
        assert state == resource.reference
        self.delete_calls.append(resource.reference)
        if self.delete_errors[resource.reference]:
            raise self.delete_errors[resource.reference].pop(0)
        self.present.discard(resource.reference)

    def is_absent(self, resource: cleanup.Resource, state: object) -> bool:
        return resource.reference not in self.present

    def list_releases(self):
        tag = self.releases[0].reference.split("#", 1)[0] if self.releases else ""
        self.release_calls.append(tag)
        if self.release_errors:
            raise self.release_errors.pop(0)
        return [
            {
                "id": r.identifier,
                "tag_name": r.reference.split("#", 1)[0],
                "assets": (
                    [{"name": "release-manifest.json"}] if r.holds_recovery_data else []
                ),
            }
            for r in self.releases
        ]


def _control_references(manifest: dict[str, object]) -> tuple[str, str]:
    run = "https://github.com/release-org/unified-cache-management/actions/runs/" + str(
        manifest["release"]["actions_run_id"]
    )
    return run, str(manifest["release"]["tag"])


def test_rich_single_arch_pull_and_member_are_deleted_once() -> None:
    resources = records_ops.registry_resources(
        _manifest(
            ghcr_members=["ghcr.io/release-org/vllm-openai:v0.23.0-amd64"],
            ghcr_indexes=[],
        )
    )

    assert [
        (item.kind, item.reference)
        for item in resources
        if item.kind.startswith("ghcr-")
    ] == [
        (
            "ghcr-member",
            "ghcr.io/release-org/vllm-openai:v0.23.0-amd64",
        )
    ]


def test_manifest_rejects_duplicate_or_wrong_registry_references() -> None:
    duplicate = _manifest(
        ghcr_members=["ghcr.io/release-org/vllm:v1"],
        ghcr_indexes=["ghcr.io/release-org/vllm:v1"],
    )
    with pytest.raises(cleanup.CleanupError, match="must be unique"):
        manifest_ops.validate_manifest(duplicate)

    wrong_registry = _manifest(dockerhub_members=["ghcr.io/release-org/vllm:v1"])
    with pytest.raises(cleanup.CleanupError, match="docker.io"):
        manifest_ops.validate_manifest(wrong_registry)


def test_release_discovery_preserves_all_exact_tag_release_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    releases = [
        {
            "id": 8,
            "tag_name": manifest["release"]["tag"],
            "assets": [
                {
                    "name": "release-manifest.json",
                    "url": "https://api.github.test/assets/8",
                }
            ],
        },
        {"id": 9, "tag_name": manifest["release"]["tag"], "assets": []},
        {"id": 10, "tag_name": "draft/v0.8.0-2", "assets": []},
    ]
    monkeypatch.setattr(remote, "list_releases", lambda: releases)
    resources = inventory_ops.ReleaseInventory(remote).resolve_releases(
        str(manifest["release"]["tag"])
    )
    assert [(item.identifier, item.holds_recovery_data) for item in resources] == [
        (8, True),
        (9, False),
    ]


def test_retention_counts_each_tag_once_including_old_schemas() -> None:
    records = [
        _record(_manifest("draft/v0.8.0-1"), 1),
        _record(_manifest("draft/v0.8.0-1"), 2),
        _record(_manifest("draft/v0.8.0-2"), 3),
        _record(_manifest("draft/v0.8.0-3"), 4),
        _record(_manifest("draft/v0.8.0-4"), 7),
        _record(
            _manifest("v0.8.0rc1", release_type="prerelease"),
            5,
        ),
    ]
    old_schema = _manifest("draft/v0.7.0-1")
    old_schema["schema_version"] = 5
    records.append(_record(old_schema, 6))

    selection = records_ops.select_retention(
        records,
        current_tag="draft/v0.8.0-4",
        release_type="draft",
        max_count=3,
        pypi_enabled=False,
    )

    assert [record["tag"] for record in selection.candidates] == [
        "draft/v0.8.0-1",
        "draft/v0.8.0-2",
    ]
    assert selection.skipped_reason is None


def test_retention_counts_failed_nightly_without_a_manifest() -> None:
    failed = _record(_manifest("nightly/v0.8.1-20260825-1", release_type="nightly"), 1)
    successful = _record(
        _manifest("nightly/v0.8.1-20260825-2", release_type="nightly"), 2
    )
    latest = _record(_manifest("nightly/v0.8.1-20260826-1", release_type="nightly"), 3)
    selection = records_ops.select_retention(
        [failed, successful, latest],
        release_type="nightly",
        max_count=2,
        pypi_enabled=False,
    )
    assert selection.candidates == (failed,)


@pytest.mark.parametrize(
    ("max_count", "pypi_enabled"),
    [(-1, False), (7, True)],
)
def test_retention_skip_does_not_query_remote_manifests(
    max_count: int, pypi_enabled: bool
) -> None:
    class NoReadRemote:
        def list_release_records(self):
            raise AssertionError("retention skip must not query Releases")

    arguments = SimpleNamespace(
        current_tag="draft/v0.8.0-3",
        release_type="draft",
        max_count=max_count,
        pypi_enabled=pypi_enabled,
        fail_resource=None,
    )

    assert cleanup._run_retention(arguments, NoReadRemote()) == []


def test_retry_reprobes_and_waits_zero_five_fifteen_before_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    resource = cleanup.Resource("ghcr-member", "ghcr.io/release-org/vllm:v1")
    remote = FakeRemote(present={resource.reference})
    remote.delete_errors[resource.reference] = [
        cleanup.RemoteError("HTTP 503 one", status=503),
        cleanup.RemoteError("HTTP 429 two", status=429),
    ]
    sleeps: list[float] = []

    failure = cleanup.delete_resource_with_retry(
        remote, resource, sleeper=sleeps.append
    )

    assert failure is None
    assert remote.probe_calls == [resource.reference] * 3
    assert remote.delete_calls == [resource.reference] * 3
    assert sleeps == [5.0, 15.0]
    log = capsys.readouterr().out
    assert f"reference={resource.reference}" in log
    assert "attempt=1/3 delay=0s" in log
    assert "attempt=2/3 delay=5s" in log
    assert "attempt=3/3 delay=15s" in log


def test_404_is_idempotent_and_permanent_errors_do_not_retry() -> None:
    missing = cleanup.Resource("git-tag", "draft/v0.8.0-3")
    missing_remote = FakeRemote()
    missing_remote.probe_errors[missing.reference] = [
        cleanup.RemoteError("not found", status=404)
    ]

    assert (
        cleanup.delete_resource_with_retry(
            missing_remote, missing, sleeper=lambda _: None
        )
        is None
    )
    assert missing_remote.delete_calls == []

    for status in (400, 401, 403, 422):
        resource = cleanup.Resource("github-release", f"tag#{status}")
        remote = FakeRemote(present={resource.reference})
        remote.delete_errors[resource.reference] = [
            cleanup.RemoteError(f"HTTP {status}", status=status)
        ]
        sleeps: list[float] = []

        failure = cleanup.delete_resource_with_retry(
            remote, resource, sleeper=sleeps.append
        )

        assert failure is not None
        assert failure.attempts == 1
        assert remote.probe_calls == [resource.reference]
        assert sleeps == []


def test_dockerhub_503_attempts_three_times_and_stage_one_continues_then_blocks() -> (
    None
):
    manifest = _manifest(dockerhub_members=["docker.io/release-org/vllm:v1"])
    phase_one = records_ops.registry_resources(manifest)
    run, tag = _control_references(manifest)
    release = cleanup.Resource("github-release", f"{tag}#99", 99)
    present = {item.reference for item in phase_one} | {run, tag, release.reference}
    remote = FakeRemote(present=present, releases=[release])
    target = next(
        item.reference for item in phase_one if item.kind.startswith("dockerhub-")
    )
    sleeps: list[float] = []

    report = cleanup.execute_cleanup(
        _target(manifest, remote.releases),
        remote,
        sleeper=sleeps.append,
        fail_resource=target,
    )

    assert report.completed is False
    assert report.stopped_phase == 1
    assert report.failures[0].resource.reference == target
    assert report.failures[0].attempts == 3
    # DockerHub retries retain the first resolved digest instead of probing the Tag again.
    assert remote.probe_calls.count(target) == 1
    assert target not in remote.delete_calls
    assert all(
        item.reference in remote.delete_calls
        for item in phase_one
        if item.reference != target
    )
    assert run not in remote.probe_calls
    assert tag in remote.present and release.reference in remote.present
    assert sleeps == [5.0, 15.0]


def _cleanup_cli_fixture(monkeypatch, tag="draft/v0.8.0-3"):
    registry = (
        cleanup.Resource("ghcr-member", "ghcr.io/release-org/vllm:v1", ("v1",)),
        cleanup.Resource("dockerhub-member", "docker.io/release-org/vllm:v1"),
    )
    run = cleanup.Resource(
        "actions-run",
        "https://github.com/release-org/unified-cache-management/actions/runs/12345",
        12345,
    )
    release = cleanup.Resource(
        "github-release", f"{tag}#99", 99, holds_recovery_data=True
    )
    unbacked = cleanup.Resource("github-release", f"{tag}#100", 100)
    target = cleanup.CleanupTarget(tag, registry, (run,), (release, unbacked))
    remote = FakeRemote(
        present={tag, run.reference, release.reference, unbacked.reference}
        | {item.reference for item in registry}
    )
    record = records_ops.record_for_tag(remote.repository, tag, "a" * 40, 12345, 1, 99)
    saved = []

    class SelectedInventory:
        def __init__(self, client):
            assert client is remote

        def resolve_record(self, selected):
            assert selected == record
            return inventory_ops.CleanupResolution(target)

        def persist_recovery(self, resolution):
            saved.append(resolution)

    monkeypatch.setattr(cleanup, "_production_remote", lambda _: remote)
    monkeypatch.setattr(cleanup, "ReleaseInventory", SelectedInventory)
    monkeypatch.setattr(
        cleanup,
        "collect_catalog",
        lambda *args, **kwargs: SimpleNamespace(
            records=[record], orphans=[], active_tags=set(), blocked={}
        ),
    )
    return target, remote, saved


@pytest.mark.parametrize("tag", ["draft/v0.8.0-3", "nightly/v0.8.0-20261010-1"])
def test_ghcr_skip_succeeds_preserves_recovery_and_can_retry(
    tmp_path, monkeypatch, tag
):
    target, remote, _ = _cleanup_cli_fixture(monkeypatch, tag)
    ghcr = target.registry[0]
    remote.delete_errors[ghcr.reference] = [
        cleanup.RemoteError("HTTP 400 last tagged version", status=400)
    ]
    report_path, summary_path = tmp_path / "report.json", tmp_path / "summary.md"
    arguments = [
        "tag",
        "--tag",
        tag,
        "--report",
        str(report_path),
        "--summary",
        str(summary_path),
    ]

    assert cleanup.main(arguments) == 0

    report = json.loads(report_path.read_text())
    assert report["status"] == report["results"][0]["status"] == "partial"
    assert report["results"][0]["skipped"] == [
        {
            "kind": ghcr.kind,
            "reference": ghcr.reference,
            "attempts": 1,
            "error": "HTTP 400 last tagged version",
        }
    ]
    assert remote.present == {ghcr.reference, target.releases[0].reference}
    assert "skipped GHCR resources" in summary_path.read_text()

    assert cleanup.main(arguments) == 0
    assert json.loads(report_path.read_text())["status"] == "complete"
    assert not remote.present


def test_dockerhub_failure_remains_blocking_when_ghcr_is_skipped(tmp_path, monkeypatch):
    target, remote, _ = _cleanup_cli_fixture(monkeypatch)
    ghcr, dockerhub = target.registry
    before = remote.present.copy()
    for resource in (ghcr, dockerhub):
        remote.delete_errors[resource.reference] = [
            cleanup.RemoteError("HTTP 403 denied", status=403)
        ]
    report_path = tmp_path / "report.json"

    assert cleanup.main(["tag", "--tag", target.tag, "--report", str(report_path)]) == 1

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    result = report["results"][0]
    assert result["status"] == "blocked"
    assert [issue["reference"] for issue in result["failures"]] == [dockerhub.reference]
    assert [issue["reference"] for issue in result["skipped"]] == [ghcr.reference]
    assert remote.present == before


def test_ghcr_probe_error_in_dry_run_is_nonfatal_and_does_not_write(
    tmp_path, monkeypatch
):
    target, remote, saved = _cleanup_cli_fixture(monkeypatch)
    ghcr = target.registry[0]
    before = remote.present.copy()
    remote.probe_errors[ghcr.reference] = [
        cleanup.RemoteError("HTTP 403 denied", status=403)
    ]
    report_path = tmp_path / "report.json"

    assert (
        cleanup.main(
            ["tag", "--tag", target.tag, "--dry-run", "--report", str(report_path)]
        )
        == 0
    )

    report = json.loads(report_path.read_text())
    assert report["status"] == "dry-run"
    assert report["results"][0]["skipped"][0]["reference"] == ghcr.reference
    assert remote.present == before and not remote.delete_calls and not saved
    assert remote.probe_calls == [item.reference for item in target.registry]


def test_actions_failure_blocks_tag_and_releases() -> None:
    manifest = _manifest(
        chart=None, ghcr_members=[], ghcr_indexes=[], dockerhub_members=[]
    )
    run, tag = _control_references(manifest)
    release = cleanup.Resource("github-release", f"{tag}#99", 99)
    remote = FakeRemote(present={run, tag, release.reference}, releases=[release])
    remote.delete_errors[run] = [
        cleanup.RemoteError("conflict", status=409),
        cleanup.RemoteError("conflict", status=409),
        cleanup.RemoteError("conflict", status=409),
    ]

    report = cleanup.execute_cleanup(
        _target(manifest, remote.releases), remote, sleeper=lambda _: None
    )

    assert report.stopped_phase == 2
    assert remote.probe_calls.count(run) == 3
    assert tag not in remote.probe_calls
    assert remote.release_calls == []


def test_tag_failure_keeps_release_manifest_and_rerun_recovers_from_404() -> None:
    manifest = _manifest(chart=None, ghcr_members=[], ghcr_indexes=[])
    run, tag = _control_references(manifest)
    release = cleanup.Resource("github-release", f"{tag}#99", 99)
    remote = FakeRemote(present={run, tag, release.reference}, releases=[release])
    remote.delete_errors[tag] = [
        cleanup.RemoteError("server", status=503),
        cleanup.RemoteError("server", status=503),
        cleanup.RemoteError("server", status=503),
    ]

    first = cleanup.execute_cleanup(
        _target(manifest, remote.releases), remote, sleeper=lambda _: None
    )

    assert first.stopped_phase == 3
    assert run not in remote.present
    assert tag in remote.present
    assert release.reference in remote.present
    assert remote.release_calls == []

    second = cleanup.execute_cleanup(
        _target(manifest, remote.releases), remote, sleeper=lambda _: None
    )

    assert second.completed is True
    assert tag not in remote.present
    assert release.reference not in remote.present
    assert remote.probe_calls.count(run) == 2


def test_release_ids_are_frozen_before_tag_deletion() -> None:
    manifest = _manifest(chart=None, ghcr_members=[], ghcr_indexes=[])
    run, tag = _control_references(manifest)
    release = cleanup.Resource("github-release", f"{tag}#99", 99)

    class TagDependentRemote(FakeRemote):
        def list_releases(self):
            releases = super().list_releases()
            return [
                release for release in releases if release["tag_name"] in self.present
            ]

    remote = TagDependentRemote(
        present={run, tag, release.reference}, releases=[release]
    )

    report = cleanup.execute_cleanup(
        _target(manifest, remote.releases), remote, sleeper=lambda _: None
    )

    assert report.completed is True
    assert release.reference in remote.delete_calls
    assert release.reference not in remote.present
    assert remote.release_calls == []


def test_executor_uses_known_release_id_without_tag_discovery() -> None:
    tag = "nightly/v0.8.0-20260825-1"
    run = cleanup.Resource(
        "actions-run",
        "https://github.com/release-org/unified-cache-management/actions/runs/12345",
        12345,
        holds_recovery_data=True,
    )
    release = cleanup.Resource("github-release", f"{tag}#99", 99)
    remote = FakeRemote(present={run.reference, tag, release.reference})
    target = cleanup.CleanupTarget(tag, (), (run,), (release,))

    report = cleanup.execute_cleanup(target, remote, sleeper=lambda _: None)

    assert report.completed is True
    assert release.reference in remote.delete_calls
    assert remote.delete_calls == [tag, release.reference, run.reference]
    assert remote.release_calls == []


def test_release_probe_uses_id_after_tag_name_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    release = cleanup.Resource("github-release", "nightly/v0.8.0-20260825-1#99", 99)
    path = "/repos/release-org/unified-cache-management/releases/99"
    calls: list[tuple[str, str]] = []

    def github_json(method: str, request_path: str):
        calls.append((method, request_path))
        return {"id": 99, "tag_name": "untagged-example"}

    monkeypatch.setattr(remote, "json_request", github_json)

    assert remote.probe(release) == path
    assert calls == [("GET", path)]


def test_release_probe_rejects_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    release = cleanup.Resource("github-release", "nightly/v0.8.0-20260825-1#99", 99)
    monkeypatch.setattr(remote, "json_request", lambda method, path: [])

    with pytest.raises(cleanup.CleanupError):
        remote.probe(release)


@pytest.mark.parametrize("delete_status", [204, 404])
def test_release_deletion_requires_readback_to_confirm_absence(
    monkeypatch: pytest.MonkeyPatch, delete_status: int
) -> None:
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    release = cleanup.Resource("github-release", "nightly/v0.8.0-20260825-1#99", 99)
    path = "/repos/release-org/unified-cache-management/releases/99"
    calls: list[tuple[str, str]] = []

    def github_json(method: str, request_path: str):
        calls.append((method, request_path))
        if method == "DELETE":
            if delete_status == 404:
                raise cleanup.RemoteError("not found", status=404)
            return None
        return {"id": 99, "tag_name": "nightly/v0.8.0-20260825-1"}

    monkeypatch.setattr(remote, "json_request", github_json)

    failure = cleanup.delete_resource_with_retry(
        remote, release, sleeper=lambda _: None
    )

    assert failure is not None
    assert failure.attempts == 3
    assert calls == [("GET", path), ("DELETE", path), ("GET", path)] * 3


def test_release_deletion_succeeds_after_readback_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    release = cleanup.Resource("github-release", "nightly/v0.8.0-20260825-1#99", 99)
    calls: list[str] = []

    def github_json(method: str, path: str):
        calls.append(method)
        if method == "DELETE":
            return None
        if calls == ["GET"]:
            return {"id": 99, "tag_name": "nightly/v0.8.0-20260825-1"}
        raise cleanup.RemoteError("not found", status=404)

    monkeypatch.setattr(remote, "json_request", github_json)

    assert (
        cleanup.delete_resource_with_retry(remote, release, sleeper=lambda _: None)
        is None
    )
    assert calls == ["GET", "DELETE", "GET"]


def test_release_get_404_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    release = cleanup.Resource("github-release", "nightly/v0.8.0-20260825-1#99", 99)
    calls: list[str] = []

    def github_json(method: str, path: str):
        calls.append(method)
        raise cleanup.RemoteError("not found", status=404)

    monkeypatch.setattr(remote, "json_request", github_json)

    assert (
        cleanup.delete_resource_with_retry(remote, release, sleeper=lambda _: None)
        is None
    )
    assert calls == ["GET"]


def test_release_phase_attempts_every_exact_release_independently() -> None:
    manifest = _manifest(chart=None, ghcr_members=[], ghcr_indexes=[])
    run, tag = _control_references(manifest)
    first = cleanup.Resource("github-release", f"{tag}#1", 1)
    second = cleanup.Resource("github-release", f"{tag}#2", 2)
    remote = FakeRemote(
        present={run, tag, first.reference, second.reference},
        releases=[first, second],
    )
    remote.delete_errors[first.reference] = [
        cleanup.RemoteError("forbidden", status=403)
    ]

    report = cleanup.execute_cleanup(
        _target(manifest, remote.releases), remote, sleeper=lambda _: None
    )

    assert report.stopped_phase == 4
    assert report.failures[0].resource == first
    assert first.reference in remote.present
    assert second.reference not in remote.present
    assert second.reference in remote.delete_calls


def test_unbacked_release_failure_preserves_manifest_holder_for_retry() -> None:
    manifest = _manifest(chart=None, ghcr_members=[], ghcr_indexes=[])
    run, tag = _control_references(manifest)
    unbacked = cleanup.Resource("github-release", f"{tag}#1", 1)
    holder = cleanup.Resource("github-release", f"{tag}#2", 2, holds_recovery_data=True)
    remote = FakeRemote(
        present={run, tag, unbacked.reference, holder.reference},
        releases=[unbacked, holder],
    )
    remote.delete_errors[unbacked.reference] = [
        cleanup.RemoteError("forbidden", status=403)
    ]

    report = cleanup.execute_cleanup(
        _target(manifest, remote.releases), remote, sleeper=lambda _: None
    )

    assert report.stopped_phase == 4
    assert unbacked.reference in remote.present
    assert holder.reference in remote.present
    assert holder.reference not in remote.delete_calls


def test_ghcr_package_version_with_another_tag_is_permanent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    monkeypatch.setattr(remote, "_owner_package_prefix", lambda: "/users/release-org")
    monkeypatch.setattr(
        remote,
        "list_pages",
        lambda path: [
            {
                "id": 77,
                "metadata": {"container": {"tags": ["v0.23.0"]}},
            }
        ],
    )
    monkeypatch.setattr(
        remote,
        "json_request",
        lambda method, path: {
            "id": 77,
            "metadata": {"container": {"tags": ["v0.23.0", "shared-latest"]}},
        },
    )
    resource = cleanup.Resource(
        "ghcr-index",
        "ghcr.io/release-org/vllm-openai:v0.23.0",
        ("v0.23.0",),
    )

    with pytest.raises(remote_ops.UnsafePackageVersion, match="shared-latest"):
        remote.probe(resource)


def test_ghcr_allows_other_target_tags_from_the_same_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(
        chart=None,
        ghcr_indexes=["ghcr.io/release-org/vllm-openai:v0.23.0"],
        ghcr_members=[
            "ghcr.io/release-org/vllm-openai:v0.23.0-amd64",
            "ghcr.io/release-org/vllm-openai:v0.23.0-arm64",
        ],
    )
    resources = records_ops.registry_resources(manifest)
    assert resources[0].identifier == ("v0.23.0", "v0.23.0-amd64", "v0.23.0-arm64")
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    monkeypatch.setattr(remote, "_owner_package_prefix", lambda: "/users/release-org")
    monkeypatch.setattr(
        remote,
        "list_pages",
        lambda path: [
            {
                "id": 77,
                "metadata": {"container": {"tags": ["v0.23.0", "v0.23.0-amd64"]}},
            }
        ],
    )

    monkeypatch.setattr(
        remote,
        "json_request",
        lambda method, path: {
            "id": 77,
            "metadata": {"container": {"tags": ["v0.23.0", "v0.23.0-amd64"]}},
        },
    )
    assert remote.probe(resources[0]).endswith("/77")


@pytest.mark.parametrize(
    ("detail", "status", "retryable"),
    [
        ("MANIFEST_UNKNOWN", 404, False),
        ("context deadline exceeded", None, True),
        ("HTTP 429 too many requests", 429, True),
        ("unauthorized", 401, False),
    ],
)
def test_crane_errors_are_structurally_classified(
    detail: str, status: int | None, retryable: bool
) -> None:
    error = cleanup.ProductionRemote._crane_error(detail)

    assert error.status == status
    assert error.is_retryable is retryable


@pytest.mark.parametrize(
    ("detail", "status"), [("MANIFEST_UNKNOWN", 404), ("HTTP 403 forbidden", 403)]
)
def test_dockerhub_probe_and_preview_distinguish_absence_from_denied_access(
    tmp_path, monkeypatch, detail, status
):
    target, remote, saved = _cleanup_cli_fixture(monkeypatch)
    before = remote.present.copy()
    dockerhub = target.registry[1]
    client = cleanup.ProductionRemote(remote.repository, "token")

    def crane(operation, reference):
        assert (operation, reference) == ("digest", dockerhub.reference)
        raise client._crane_error(detail)

    monkeypatch.setattr(client, "_run_crane", crane)
    if status == 404:
        assert client.probe(dockerhub) is None
    else:
        with pytest.raises(cleanup.RemoteError) as caught:
            client.probe(dockerhub)
        assert caught.value.status == status

    def probe(resource):
        if resource == dockerhub:
            return client.probe(resource)
        return FakeRemote.probe(remote, resource)

    monkeypatch.setattr(remote, "probe", probe)
    report_path = tmp_path / "report.json"
    result = cleanup.main(
        ["tag", "--tag", target.tag, "--dry-run", "--report", str(report_path)]
    )
    report = json.loads(report_path.read_text())
    assert result == (0 if status == 404 else 1)
    assert report["status"] == ("dry-run" if status == 404 else "failed")
    assert report["results"][0]["status"] == (
        "would-delete" if status == 404 else "blocked"
    )
    assert remote.present == before and not remote.delete_calls and not saved


def test_dockerhub_delete_uses_the_probed_manifest_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        remote,
        "_run_crane",
        lambda operation, reference: calls.append((operation, reference)) or "",
    )
    resource = cleanup.Resource(
        "dockerhub-member", "docker.io/release-org/vllm-openai:v0.23.0-amd64"
    )
    digest = "sha256:" + "a" * 64

    remote.delete(resource, digest)

    assert calls == [("delete", f"docker.io/release-org/vllm-openai@{digest}")]


def test_rerunning_an_old_tag_does_not_replace_the_latest_deletion_boundary() -> None:
    records = [_record(_manifest(f"draft/v0.8.0-{i}"), i) for i in range(1, 5)]
    selection = records_ops.select_retention(
        records,
        current_tag="draft/v0.8.0-1",
        release_type="draft",
        max_count=3,
        pypi_enabled=False,
    )
    assert selection.candidates == ()


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_record_cli_rejects_invalid_plan_before_remote_writes(
    tmp_path, monkeypatch, capsys, contents
):
    path = tmp_path / "invalid-plan.json"
    path.write_text(contents)
    remote = SimpleNamespace(
        save_record=lambda *args: pytest.fail(
            "invalid input must not reach publication"
        )
    )
    monkeypatch.setattr(cleanup, "_production_remote", lambda args: remote)
    status = cleanup.main(
        [
            "record-targets",
            "--tag",
            "draft/v0.8.0-1",
            "--release-type",
            "draft",
            "--run-id",
            "100",
            "--source-sha",
            "a" * 40,
            "--plan",
            str(path),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert status == 2
    assert "Traceback" not in capsys.readouterr().err


def test_retention_resolves_and_preserves_known_release_before_execution(
    monkeypatch,
) -> None:
    manifest = _manifest("draft/v0.8.0-1", chart=None, ghcr_members=[], ghcr_indexes=[])
    run, tag = _control_references(manifest)
    release = cleanup.Resource("github-release", f"{tag}#99", 99)
    record = _record(manifest, 99)
    saved = []

    class SelectedInventory:
        def __init__(self, remote):
            self.remote = remote

        def resolve_record(self, selected):
            assert selected == record
            target = _target(manifest, [release])
            return inventory_ops.CleanupResolution(target)

        def persist_recovery(self, resolution):
            assert self.remote.delete_calls == []
            saved.append(resolution)

    monkeypatch.setattr(cleanup, "ReleaseInventory", SelectedInventory)
    monkeypatch.setattr(
        cleanup,
        "collect_catalog",
        lambda *args, **kwargs: SimpleNamespace(
            records=[record], orphans=[], active_tags=set(), blocked={}
        ),
    )
    remote = FakeRemote(present={run, tag, release.reference})
    arguments = SimpleNamespace(
        current_tag="draft/v0.8.0-2",
        release_type="draft",
        max_count=1,
        pypi_enabled=False,
        fail_resource=None,
    )

    assert cleanup._run_retention(arguments, remote) == []
    assert len(saved) == 1
    assert release.reference in remote.delete_calls
    assert release.reference not in remote.present


@pytest.mark.parametrize("transient_failure", [None, "delete", "readback"])
def test_dockerhub_retry_and_readback_keep_the_original_digest(
    monkeypatch, transient_failure
):
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    repository = "docker.io/release-org/vllm-openai"
    resource = cleanup.Resource("dockerhub-member", f"{repository}:v0.23.0-amd64")
    original = "sha256:" + "a" * 64
    replacement = "sha256:" + "b" * 64
    present = {original, replacement}
    tag_digest = original
    calls = []
    failed_once = False

    def crane(operation, reference):
        nonlocal tag_digest, failed_once
        calls.append((operation, reference))
        if reference == resource.reference:
            return tag_digest
        assert reference == f"{repository}@{original}"
        if operation == "delete":
            tag_digest = replacement
            if transient_failure == "delete" and not failed_once:
                failed_once = True
                raise cleanup.RemoteError("HTTP 503", status=503)
            present.discard(original)
            return ""
        if transient_failure == "readback" and not failed_once:
            failed_once = True
            raise cleanup.RemoteError("readback timed out")
        if original not in present:
            raise cleanup.RemoteError("manifest unknown", status=404)
        return original

    monkeypatch.setattr(remote, "_run_crane", crane)
    sleeps = []
    assert (
        cleanup.delete_resource_with_retry(remote, resource, sleeper=sleeps.append)
        is None
    )
    assert calls.count(("digest", resource.reference)) == 1
    assert ("digest", f"{repository}@{original}") in calls
    assert all(reference != f"{repository}@{replacement}" for _, reference in calls)
    assert present == {replacement}
    assert sleeps == ([] if transient_failure is None else [5.0])


def test_dockerhub_probe_can_retry_until_a_digest_is_resolved(monkeypatch):
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    repository = "docker.io/release-org/vllm-openai"
    resource = cleanup.Resource("dockerhub-member", f"{repository}:v0.23.0-amd64")
    digest = "sha256:" + "a" * 64
    calls = []

    def crane(operation, reference):
        calls.append((operation, reference))
        if len(calls) == 1:
            raise cleanup.RemoteError("probe timed out")
        if reference == resource.reference:
            return digest
        assert reference == f"{repository}@{digest}"
        if operation == "delete":
            return ""
        raise cleanup.RemoteError("manifest unknown", status=404)

    monkeypatch.setattr(remote, "_run_crane", crane)
    sleeps = []
    assert (
        cleanup.delete_resource_with_retry(remote, resource, sleeper=sleeps.append)
        is None
    )
    assert calls == [
        ("digest", resource.reference),
        ("digest", resource.reference),
        ("delete", f"{repository}@{digest}"),
        ("digest", f"{repository}@{digest}"),
    ]
    assert sleeps == [5.0]


def test_local_record_and_event_validation_do_not_construct_a_remote(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cleanup,
        "_production_remote",
        lambda _: pytest.fail("local commands need no token"),
    )
    identity = [
        "--repository",
        "release-org/unified-cache-management",
        "--tag",
        "draft/v0.8.0-1",
        "--source-sha",
        "a" * 40,
        "--run-id",
        "100",
        "--run-attempt",
        "2",
    ]
    assert cleanup.main(["record", *identity, "--output-dir", str(tmp_path)]) == 0
    path = tmp_path / "release-cleanup-100-2-opened.json"
    assert path.is_file()
    assert cleanup.main(["validate-record", *identity, "--input", str(path)]) == 0


@pytest.mark.parametrize(
    ("option", "mismatched_value"),
    [("--run-id", "101"), ("--run-attempt", "3"), ("--source-sha", "b" * 40)],
)
def test_event_validation_rejects_a_different_publication_binding(
    tmp_path, monkeypatch, option, mismatched_value
):
    snapshot = records_ops.new_record(
        "release-org/unified-cache-management", "draft/v0.8.0-1", "a" * 40, 100, 2
    )
    path = tmp_path / records_ops.record_basename(snapshot, 2)
    path.write_text(json.dumps(snapshot))
    monkeypatch.setattr(
        cleanup, "_production_remote", lambda _: pytest.fail("validation is read-only")
    )
    identity = [
        "--repository",
        "release-org/unified-cache-management",
        "--tag",
        "draft/v0.8.0-1",
        "--source-sha",
        "a" * 40,
        "--run-id",
        "100",
        "--run-attempt",
        "2",
    ]
    identity[identity.index(option) + 1] = mismatched_value
    assert cleanup.main(["validate-record", *identity, "--input", str(path)]) == 2
    assert json.loads(path.read_text()) == snapshot


CATALOG_REPOSITORY = "release-org/unified-cache-management"
CATALOG_PREFIX = f"/repos/{CATALOG_REPOSITORY}"
CATALOG_TAG = "nightly/v0.9.0-20261009-1"
CATALOG_SHA = "a" * 40


class CatalogHttpError(remote_ops.RemoteError):
    def __init__(self, status):
        super().__init__(f"GitHub API HTTP {status}", status=status)


class CatalogRemote(remote_ops.ProductionRemote):
    repository = CATALOG_REPOSITORY

    def __init__(self):
        self.releases = []
        self.refs = []
        self.tag_objects = {}
        self.workflows = {"release-nightly.yml": [], "release-tag.yml": []}
        self.runs = {}
        self.artifacts = {}
        self.jobs = {}
        self.downloads = {}
        self.download_accepts = {}
        self.errors = {}
        self.calls = []

    def json_request(self, method, path):
        self.calls.append((method, path))
        assert method == "GET"
        parsed = urlparse(path)
        path = parsed.path
        page = int(parse_qs(parsed.query).get("page", ["1"])[0])
        start = (page - 1) * 100
        if path in self.errors:
            raise self.errors[path]
        if path == CATALOG_PREFIX + "/releases":
            return self.releases[start : start + 100]
        if path == CATALOG_PREFIX + "/git/matching-refs/tags/":
            return self.refs
        if path.startswith(CATALOG_PREFIX + "/git/ref/tags/"):
            tag = path.split("/git/ref/tags/", 1)[1]
            for ref in self.refs:
                if ref["ref"] == "refs/tags/" + tag:
                    return ref
            raise CatalogHttpError(404)
        if path.startswith(CATALOG_PREFIX + "/releases/"):
            release_id = int(path.rsplit("/", 1)[-1])
            for release in self.releases:
                if release["id"] == release_id:
                    return release
            raise CatalogHttpError(404)
        if "/git/tags/" in path:
            return self.tag_objects[path.rsplit("/", 1)[-1]]
        if "/actions/workflows/" in path:
            name = path.split("/actions/workflows/", 1)[1].split("/", 1)[0]
            runs = self.workflows.get(name, [])
            source_sha = parse_qs(parsed.query).get("head_sha", [None])[0]
            if source_sha is not None:
                runs = [run for run in runs if run["head_sha"] == source_sha]
            return {"workflow_runs": runs[start : start + 100]}
        if path.endswith("/artifacts"):
            run_id = int(path.split("/runs/", 1)[1].split("/", 1)[0])
            return {"artifacts": self.artifacts.get(run_id, [])[start : start + 100]}
        if "/actions/runs/" in path and path.endswith("/jobs"):
            run_id = int(path.split("/runs/", 1)[1].split("/", 1)[0])
            return {"jobs": self.jobs.get(run_id, [])[start : start + 100]}
        if "/actions/runs/" in path:
            run_id = int(path.rsplit("/", 1)[-1])
            if run_id not in self.runs:
                raise CatalogHttpError(404)
            return self.runs[run_id]
        raise AssertionError(path)

    def request(self, method, path, *, accept="application/vnd.github+json"):
        self.calls.append((method, path))
        self.download_accepts[path] = accept
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
        self.downloads[f"{CATALOG_PREFIX}/releases/assets/{asset_id}"] = json.dumps(
            value
        ).encode()

    def add_artifact(
        self,
        run,
        record,
        *,
        artifact_id=100,
        phase="",
        attempt=1,
        extra=False,
        filename="release-cleanup.json",
        family="nightly",
    ):
        name = f"ucm-{family}-cleanup{phase}-run-{run['id']}-attempt-{attempt}"
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
            archive.writestr(filename, json.dumps(record))
            if extra:
                archive.writestr("../untrusted.py", "code")
        self.downloads[f"{CATALOG_PREFIX}/actions/artifacts/{artifact_id}/zip"] = (
            output.getvalue()
        )


def _catalog_release(
    tag=CATALOG_TAG, *, release_id=1, draft=True, tag_name=None, owned=True
):
    return {
        "id": release_id,
        "tag_name": tag_name or tag,
        "name": tag,
        "draft": draft,
        "prerelease": True,
        "assets": [],
        "body": "Status: `artifacts-failed`" if owned else "user notes",
    }


def _catalog_ref(tag=CATALOG_TAG, *, sha=CATALOG_SHA, kind="commit"):
    return {"ref": "refs/tags/" + tag, "object": {"type": kind, "sha": sha}}


def _catalog_run(
    *,
    run_id=123,
    attempt=1,
    status="completed",
    event="schedule",
    path="release-nightly.yml",
    branch="develop",
    sha=CATALOG_SHA,
    repository=CATALOG_REPOSITORY,
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
        "repository": {"full_name": CATALOG_REPOSITORY},
    }


def _catalog_legacy_record(
    tag=CATALOG_TAG,
    *,
    release_id=1,
    run_id=123,
    attempt=1,
    sha=CATALOG_SHA,
    resources=False,
):
    # Legacy producer JSON has eight fields; internal normalization adds provenance.
    record = {
        "kind": "ucm-release-cleanup",
        "schema_version": 1,
        "repository": CATALOG_REPOSITORY,
        "tag": tag,
        "source_sha": sha,
        "release_ids": [] if release_id is None else [release_id],
        "runs": [{"id": run_id, "attempt": attempt}],
        "resources": [],
    }
    if resources:
        version = cleanup.version_config.classify_tag(tag)["image_version"]
        record["resources"] = [
            {
                "kind": "ghcr-member",
                "reference": f"ghcr.io/release-org/vllm-openai:v1-ucm-{version}",
            }
        ]
    return record


def _catalog_legacy_manifest(tag=CATALOG_TAG, run_id=123):
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


def _catalog_history(*records, blocked=None):
    return {
        "kind": "ucm-nightly-cleanup-inventory",
        "schema_version": 1,
        "repository": CATALOG_REPOSITORY,
        "targets": list(records),
        "evidence": {},
        "blocked": blocked or {},
    }


def _nightly_catalog(remote, history=None):
    return inventory_ops.collect_catalog(remote, history, release_type="nightly")


def _catalog_snapshot(tag, *, run_id=123, sha=CATALOG_SHA, resources=False):
    document = records_ops.new_record(CATALOG_REPOSITORY, tag, sha, run_id, 1)
    if resources:
        version = cleanup.version_config.classify_tag(tag)["image_version"]
        document["resources"] = [
            {
                "kind": "ghcr-member",
                "reference": f"ghcr.io/release-org/vllm-openai:v1-ucm-{version}",
            }
        ]
    return document, records_ops.record_basename(document, 1, planned=resources)


@pytest.mark.parametrize(
    "release_type,tag", [("nightly", CATALOG_TAG), ("draft", "draft/v0.9.0-1")]
)
@pytest.mark.parametrize("storage", ["asset", "artifact"])
def test_immutable_snapshot_evidence_binds_any_release_type(release_type, tag, storage):
    remote = CatalogRemote()
    release = _catalog_release(tag)
    remote.releases = [release]
    remote.refs = [_catalog_ref(tag)]
    run = _catalog_run(path="release-tag.yml", event="push", branch=tag)
    remote.add_run(run)
    document, name = _catalog_snapshot(tag, resources=True)
    if storage == "asset":
        remote.add_asset(release, document, name=name)
    else:
        remote.add_artifact(
            run, document, filename=name, family="release", phase="-plan"
        )

    result = inventory_ops.collect_catalog(remote, release_type=release_type)

    assert result.records == (
        records_ops.decode_snapshot(document, CATALOG_REPOSITORY, name, 1),
    )
    assert result.blocked == {}
    assert result.active_tags == set()
    assert all(method == "GET" for method, _ in remote.calls)


@pytest.mark.parametrize(
    "release_type,tag", [("nightly", CATALOG_TAG), ("draft", "draft/v0.9.0-1")]
)
@pytest.mark.parametrize("storage", ["asset", "artifact"])
def test_moved_tag_keeps_ordinary_attempt_sources_and_blocks_nightly_conflicts(
    release_type, tag, storage
):
    remote = CatalogRemote()
    release = _catalog_release(tag)
    remote.releases = [release]
    moved_sha = "b" * 40
    remote.refs = [_catalog_ref(tag, sha=moved_sha)]
    for index, sha in enumerate((CATALOG_SHA, moved_sha)):
        run_id = 123 + index
        run = _catalog_run(
            run_id=run_id, path="release-tag.yml", event="push", branch=tag, sha=sha
        )
        remote.add_run(run)
        document, name = _catalog_snapshot(tag, run_id=run_id, sha=sha, resources=True)
        document["resources"][0]["reference"] = document["resources"][0][
            "reference"
        ].replace("v1-", f"v{index + 1}-")
        if storage == "asset":
            remote.add_asset(release, document, name=name, asset_id=10 + index)
        else:
            remote.add_artifact(
                run,
                document,
                filename=name,
                family="release",
                phase="-plan",
                artifact_id=100 + index,
            )

    result = inventory_ops.collect_catalog(remote, release_type=release_type)

    assert len(result.records) == 1
    if release_type == "nightly":
        assert "conflicting source SHAs" in result.blocked[tag]
    else:
        assert result.blocked == {}
        record = result.records[0]
        assert record["source_sha"] is None
        assert record["attempt_sources"] == [
            {"id": 123, "attempt": 1, "source_sha": CATALOG_SHA},
            {"id": 124, "attempt": 1, "source_sha": moved_sha},
        ]
        assert len(record["resources"]) == 2


@pytest.mark.parametrize(
    "release_type,tag", [("nightly", CATALOG_TAG), ("draft", "draft/v0.9.0-1")]
)
@pytest.mark.parametrize("status", ["in_progress", "queued", "waiting"])
def test_snapshot_active_tag_only_source_is_protected(release_type, tag, status):
    remote = CatalogRemote()
    remote.refs = [_catalog_ref(tag)]
    run = _catalog_run(status=status, path="release-tag.yml", event="push", branch=tag)
    remote.add_run(run)
    document, name = _catalog_snapshot(tag)
    remote.add_artifact(run, document, filename=name, family="release")

    result = inventory_ops.collect_catalog(remote, release_type=release_type)

    assert len(result.records) == 1
    assert result.active_tags == {tag}
    assert result.blocked == {}


def test_manifestless_failed_drafts_count_and_remain_explicitly_blocked():
    remote = CatalogRemote()
    remote.releases = [
        _catalog_release(),
        _catalog_release("nightly/v0.9.0-20261008-1", release_id=2),
    ]
    remote.refs = [_catalog_ref(), _catalog_ref("nightly/v0.9.0-20261008-1")]
    result = _nightly_catalog(remote)
    assert len(result.records) == 2
    assert set(result.blocked) == {item["tag"] for item in result.records}
    assert (
        records_ops.select_retention(result.records, 1)
        .candidates[0]["tag"]
        .endswith("20261008-1")
    )


def test_catalog_collects_release_run_and_evidence_beyond_first_page():
    remote = CatalogRemote()
    remote.releases = [
        _catalog_release(f"v0.8.{number}", release_id=1000 + number)
        for number in range(100)
    ] + [_catalog_release()]
    remote.refs = [_catalog_ref()]
    remote.workflows["release-nightly.yml"] = [
        _catalog_run(run_id=1000 + number, repository="other/repository")
        for number in range(100)
    ]
    run = _catalog_run()
    remote.add_run(run)
    remote.artifacts[run["id"]] = [
        {"id": 1000 + number, "name": "unrelated-evidence"} for number in range(100)
    ]
    remote.add_artifact(run, _catalog_legacy_record(resources=True))

    result = _nightly_catalog(remote)

    assert result.records == (
        records_ops.validate_record(
            _catalog_legacy_record(resources=True), CATALOG_REPOSITORY
        ),
    )
    assert result.blocked == {}
    for path in (
        CATALOG_PREFIX + "/releases",
        CATALOG_PREFIX + "/actions/workflows/release-nightly.yml/runs",
        CATALOG_PREFIX + "/actions/runs/123/artifacts",
    ):
        assert ("GET", path + "?per_page=100&page=2") in remote.calls
    assert all(method == "GET" for method, _ in remote.calls)


def test_supported_legacy_manifest_recovers_published_and_untagged_drafts():
    remote = CatalogRemote()
    published = _catalog_release(draft=False)
    orphan = _catalog_release(
        "nightly/v0.9.0-20261008-1", release_id=2, tag_name="untagged-abcd", owned=False
    )
    remote.releases = [published, orphan]
    remote.add_asset(
        published, _catalog_legacy_manifest(), name="release-manifest.json"
    )
    remote.add_asset(
        orphan,
        _catalog_legacy_manifest(orphan["name"], 124),
        asset_id=11,
        name="release-manifest.json",
    )
    result = _nightly_catalog(remote)
    assert len(result.records) == 2
    assert result.blocked == {}
    assert result.orphans == ()  # GitHub orphan Drafts still occupy quota.


def test_untagged_name_requires_pipeline_evidence_and_ordinary_drafts_stay_out():
    remote = CatalogRemote()
    remote.releases = [
        _catalog_release(tag_name="untagged-a"),
        _catalog_release(
            "nightly/v0.9.0-20261008-1",
            release_id=2,
            tag_name="untagged-b",
            owned=False,
        ),
        _catalog_release("draft/v0.9.0", release_id=3),
    ]
    result = _nightly_catalog(remote)
    assert [record["tag"] for record in result.records] == [CATALOG_TAG]
    assert CATALOG_TAG in result.blocked


def test_live_producer_record_binds_release_source_run_and_actual_tag():
    remote = CatalogRemote()
    release = _catalog_release()
    remote.releases = [release]
    remote.refs = [_catalog_ref()]
    remote.add_asset(release, _catalog_legacy_record(resources=True))
    remote.add_run(_catalog_run(), listed=False)
    result = _nightly_catalog(remote)
    assert len(result.records[0]["resources"]) == 1
    assert result.blocked == {}
    assert all(method == "GET" for method, _ in remote.calls)
    assert not any("untrusted.test" in path for _, path in remote.calls)


def test_actions_zip_download_uses_rest_media_type_and_release_assets_use_octet_stream():
    remote = CatalogRemote()
    release = _catalog_release()
    remote.releases = [release]
    remote.add_asset(release, _catalog_legacy_record())
    run = _catalog_run()
    remote.add_run(run)
    remote.add_artifact(run, _catalog_legacy_record(release_id=None))

    result = _nightly_catalog(remote)

    assert result.blocked == {}
    assert remote.download_accepts[CATALOG_PREFIX + "/actions/artifacts/100/zip"] == (
        "application/vnd.github+json"
    )
    assert remote.download_accepts[CATALOG_PREFIX + "/releases/assets/10"] == (
        "application/octet-stream"
    )


@pytest.mark.parametrize(
    "change",
    ["release", "source", "repository", "caller_tag", "workflow", "future_attempt"],
)
def test_producer_record_binding_failures_block_the_version(change):
    remote = CatalogRemote()
    release = _catalog_release()
    remote.releases = [release]
    remote.refs = [_catalog_ref()]
    record = _catalog_legacy_record(resources=True)
    run = _catalog_run()
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
    result = _nightly_catalog(remote)
    assert len(result.records) == 1
    assert CATALOG_TAG in result.blocked


def test_duplicate_release_records_merge_and_wrong_additional_release_id_is_blocked():
    remote = CatalogRemote()
    first, second = _catalog_release(), _catalog_release(release_id=2)
    remote.releases = [first, second, _catalog_release("v0.9.0", release_id=3)]
    remote.add_asset(first, _catalog_legacy_record())
    remote.add_asset(second, _catalog_legacy_record(release_id=2), asset_id=11)
    result = _nightly_catalog(remote)
    assert len(result.records) == 1
    assert result.records[0]["release_ids"] == [1, 2]
    forged = _catalog_legacy_record()
    forged["release_ids"].append(3)
    remote.add_asset(
        first, forged, asset_id=12
    )  # Duplicate asset itself is also unsafe.
    result = _nightly_catalog(remote)
    assert CATALOG_TAG in result.blocked


@pytest.mark.parametrize("status", ["in_progress", "queued", "waiting"])
def test_active_tag_only_source_is_protected_and_counts_toward_quota(status):
    remote = CatalogRemote()
    remote.refs = [_catalog_ref()]
    run = _catalog_run(
        status=status, path="release-tag.yml", event="push", branch=CATALOG_TAG
    )
    remote.add_run(run)
    remote.add_artifact(run, _catalog_legacy_record(release_id=None))
    result = _nightly_catalog(remote)
    assert len(result.records) == 1
    assert result.active_tags == {CATALOG_TAG}
    assert result.blocked == {}


def test_unknown_active_nightly_caller_conservatively_protects_matching_source_sha():
    remote = CatalogRemote()
    remote.refs = [_catalog_ref(), _catalog_ref("nightly/v0.9.0-20261008-1")]
    remote.add_run(_catalog_run(status="queued"))
    result = _nightly_catalog(remote)
    assert result.active_tags == {CATALOG_TAG, "nightly/v0.9.0-20261008-1"}


def test_prepare_artifact_without_a_live_tag_or_release_does_not_create_phantom_quota():
    remote = CatalogRemote()
    run = _catalog_run()
    remote.add_run(run)
    remote.add_artifact(run, _catalog_legacy_record(release_id=None))
    result = _nightly_catalog(remote)
    assert result.records == result.orphans == ()


def test_plan_artifact_and_prior_attempts_recover_exact_union_after_attachment_failure():
    remote = CatalogRemote()
    remote.refs = [_catalog_ref()]
    run = _catalog_run(attempt=2)
    remote.add_run(run)
    remote.add_artifact(
        run, _catalog_legacy_record(release_id=None), artifact_id=100, attempt=1
    )
    remote.add_artifact(
        run,
        _catalog_legacy_record(attempt=2, resources=True),
        artifact_id=101,
        phase="-plan",
        attempt=2,
    )
    result = _nightly_catalog(remote)
    assert result.blocked == {}
    assert result.records[0]["runs"] == [
        {"id": 123, "attempt": 1},
        {"id": 123, "attempt": 2},
    ]
    assert len(result.records[0]["resources"]) == 1


@pytest.mark.parametrize("defect", ["sha", "attempt", "tag", "zip"])
def test_artifact_binding_and_zip_shape_fail_closed(defect):
    remote = CatalogRemote()
    remote.refs = [_catalog_ref()]
    run = _catalog_run(path="release-tag.yml", event="push", branch=CATALOG_TAG)
    remote.add_run(run)
    record = _catalog_legacy_record(release_id=None)
    if defect == "sha":
        record["source_sha"] = "b" * 40
    if defect == "attempt":
        record["runs"][0]["attempt"] = 2
    if defect == "tag":
        record["tag"] = "nightly/v0.9.0-20261008-1"
    remote.add_artifact(run, record, extra=defect == "zip")
    result = _nightly_catalog(remote)
    assert CATALOG_TAG in result.blocked
    assert len(result.records) == 1


def test_untrusted_workflow_run_artifacts_are_never_downloaded():
    remote = CatalogRemote()
    remote.refs = [_catalog_ref()]
    run = _catalog_run(repository="attacker/repo")
    remote.add_run(run)
    remote.add_artifact(run, _catalog_legacy_record(release_id=None))
    result = _nightly_catalog(remote)
    assert CATALOG_TAG in result.blocked
    assert not any(path.endswith("/zip") for _, path in remote.calls)


def test_annotated_tag_is_resolved_and_binds_source_sha():
    remote = CatalogRemote()
    remote.refs = [_catalog_ref(sha="b" * 40, kind="tag")]
    remote.tag_objects["b" * 40] = {"object": {"type": "commit", "sha": CATALOG_SHA}}
    result = _nightly_catalog(
        remote, _catalog_history(_catalog_legacy_record(release_id=None))
    )
    assert result.records[0]["source_sha"] == CATALOG_SHA
    assert result.blocked == {}


def test_history_orphans_are_outside_quota_and_explicit_missing_evidence_is_preserved():
    remote = CatalogRemote()
    live = _catalog_legacy_record()
    old = _catalog_legacy_record(
        "nightly/v0.9.0-20261008-1", release_id=2, run_id=124, resources=True
    )
    blocked_tag = "nightly/v0.7.0-20260910-1"
    remote.releases = [_catalog_release()]
    result = _nightly_catalog(
        remote,
        _catalog_history(
            live, old, blocked={blocked_tag: "plan was removed before archival"}
        ),
    )
    assert [record["tag"] for record in result.records] == [CATALOG_TAG]
    assert {record["tag"] for record in result.orphans} == {old["tag"], blocked_tag}
    assert result.blocked == {blocked_tag: "plan was removed before archival"}


def test_history_run_only_orphan_remains_cleanup_target_but_empty_snapshot_record_does_not():
    remote = CatalogRemote()
    run_only = _catalog_legacy_record(release_id=None)
    empty = _catalog_legacy_record("nightly/v0.9.0-20261008-1", release_id=None)
    empty["runs"] = []
    result = _nightly_catalog(remote, _catalog_history(run_only, empty))
    assert result.records == ()
    assert [record["tag"] for record in result.orphans] == [CATALOG_TAG]


@pytest.mark.parametrize("mutation", ["repository", "schema", "fields"])
def test_historical_inventory_global_identity_is_strict(mutation):
    document = _catalog_history(_catalog_legacy_record())
    if mutation == "repository":
        document["repository"] = "another/repo"
    elif mutation == "schema":
        document["schema_version"] = 2
    else:
        document["extra"] = True
    with pytest.raises(
        records_ops.CleanupRecordError, match="historical cleanup inventory"
    ):
        _nightly_catalog(CatalogRemote(), document)


@pytest.mark.parametrize(
    "path",
    [
        "/releases",
        "/git/matching-refs/tags/",
        "/actions/workflows/release-nightly.yml/runs",
    ],
)
def test_global_403_prevents_cleanup_using_an_incomplete_catalog(path):
    remote = CatalogRemote()
    remote.errors[CATALOG_PREFIX + path] = CatalogHttpError(403)
    with pytest.raises(CatalogHttpError, match="403"):
        _nightly_catalog(remote)


def test_per_version_asset_or_run_403_is_visible_and_other_versions_are_retained():
    remote = CatalogRemote()
    first, second = _catalog_release(), _catalog_release(
        "nightly/v0.9.0-20261008-1", release_id=2
    )
    remote.releases = [first, second]
    remote.add_asset(first, _catalog_legacy_record())
    remote.add_asset(
        second,
        _catalog_legacy_record(second["name"], release_id=2, run_id=124),
        asset_id=11,
    )
    remote.errors[CATALOG_PREFIX + "/actions/runs/123"] = CatalogHttpError(403)
    result = _nightly_catalog(remote)
    assert len(result.records) == 2
    assert CATALOG_TAG in result.blocked
    assert CATALOG_TAG in result.active_tags
    assert second["name"] not in result.blocked


def test_history_and_live_tag_source_conflict_is_not_silently_merged():
    remote = CatalogRemote()
    remote.refs = [_catalog_ref(sha="b" * 40)]
    result = _nightly_catalog(
        remote, _catalog_history(_catalog_legacy_record(release_id=None))
    )
    assert CATALOG_TAG in result.blocked
    assert "conflicting source SHAs" in result.blocked[CATALOG_TAG]


def test_conflicting_asset_tag_cannot_invent_an_additional_quota_slot():
    remote = CatalogRemote()
    release = _catalog_release()
    remote.releases = [release]
    remote.add_asset(release, _catalog_legacy_record("nightly/v0.9.0-20261008-1"))
    result = _nightly_catalog(remote)
    assert [record["tag"] for record in result.records] == [CATALOG_TAG]
    assert "conflicting identities" in result.blocked[CATALOG_TAG]
