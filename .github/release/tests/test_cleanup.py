from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
cleanup = importlib.import_module("ucm_release.cleanup")


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


def _record(
    manifest: dict[str, object], created_at: str, release_id: int
) -> cleanup.ManifestRecord:
    release_type = manifest["release"]["type"]
    draft, prerelease = {
        "stable": (False, False),
        "prerelease": (False, True),
        "draft": (True, True),
        "nightly": (False, True),
    }[str(release_type)]
    return cleanup.ManifestRecord(manifest, created_at, release_id, draft, prerelease)


class FakeRemote:
    repository = "release-org/unified-cache-management"

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

    def release_resources(self, tag: str) -> list[cleanup.Resource]:
        self.release_calls.append(tag)
        if self.release_errors:
            raise self.release_errors.pop(0)
        return list(self.releases)


def _control_references(manifest: dict[str, object]) -> tuple[str, str]:
    run = "https://github.com/release-org/unified-cache-management/actions/runs/" + str(
        manifest["release"]["actions_run_id"]
    )
    return run, str(manifest["release"]["tag"])


def test_rich_single_arch_pull_and_member_are_deleted_once() -> None:
    resources = cleanup.registry_resources(
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
        cleanup.validate_manifest(duplicate)

    wrong_registry = _manifest(dockerhub_members=["ghcr.io/release-org/vllm:v1"])
    with pytest.raises(cleanup.CleanupError, match="docker.io"):
        cleanup.validate_manifest(wrong_registry)


def test_manual_manifest_is_downloaded_from_all_exact_tag_releases(
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
    monkeypatch.setattr(
        remote,
        "_request",
        lambda method, path, accept="application/vnd.github+json": json.dumps(
            manifest
        ).encode(),
    )

    assert remote.load_manifest_for_tag(str(manifest["release"]["tag"])) == manifest
    resources = remote.release_resources(str(manifest["release"]["tag"]))
    assert [(item.identifier, item.holds_manifest) for item in resources] == [
        (8, True),
        (9, False),
    ]


def test_retention_selects_oldest_same_type_schema_v6_tags_and_reserves_current() -> (
    None
):
    records = [
        _record(_manifest("draft/v0.8.0-1"), "2026-08-20T00:00:00Z", 1),
        _record(_manifest("draft/v0.8.0-1"), "2026-08-20T00:01:00Z", 2),
        _record(_manifest("draft/v0.8.0-2"), "2026-08-21T00:00:00Z", 3),
        _record(_manifest("draft/v0.8.0-3"), "2026-08-22T00:00:00Z", 4),
        _record(_manifest("draft/v0.8.0-4"), "2026-08-01T00:00:00Z", 7),
        _record(
            _manifest("v0.8.0rc1", release_type="prerelease"),
            "2026-08-19T00:00:00Z",
            5,
        ),
    ]
    old_schema = _manifest("draft/v0.7.0-1")
    old_schema["schema_version"] = 5
    records.append(_record(old_schema, "2026-08-01T00:00:00Z", 6))

    selection = cleanup.select_retention_candidates(
        records,
        current_tag="draft/v0.8.0-4",
        release_type="draft",
        max_count=3,
        pypi_enabled=False,
    )

    assert [record.manifest["release"]["tag"] for record in selection.candidates] == [
        "draft/v0.8.0-1"
    ]
    assert selection.skipped_reason is None


def test_retention_excludes_failed_draft_nightly_with_a_manifest() -> None:
    failed_manifest = _manifest("nightly/v0.8.1-20260825-1", release_type="nightly")
    failed = cleanup.ManifestRecord(
        failed_manifest,
        "2026-08-25T18:00:00Z",
        1,
        True,
        True,
    )
    successful = _record(
        _manifest("nightly/v0.8.1-20260825-2", release_type="nightly"),
        "2026-08-25T18:01:00Z",
        2,
    )

    failed_only = cleanup.select_retention_candidates(
        [failed],
        current_tag="nightly/v0.8.1-20260826-1",
        release_type="nightly",
        max_count=1,
        pypi_enabled=False,
    )
    successful_only = cleanup.select_retention_candidates(
        [successful],
        current_tag="nightly/v0.8.1-20260826-1",
        release_type="nightly",
        max_count=1,
        pypi_enabled=False,
    )

    assert failed_only.candidates == ()
    assert successful_only.candidates == (successful,)


@pytest.mark.parametrize(
    ("max_count", "pypi_enabled"),
    [(-1, False), (7, True)],
)
def test_retention_skip_does_not_query_remote_manifests(
    max_count: int, pypi_enabled: bool
) -> None:
    class NoReadRemote:
        def list_manifest_records(self):
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


def test_synthetic_503_attempts_three_times_and_stage_one_continues_then_blocks() -> (
    None
):
    manifest = _manifest(dockerhub_members=["docker.io/release-org/vllm:v1"])
    phase_one = cleanup.registry_resources(manifest)
    run, tag = _control_references(manifest)
    release = cleanup.Resource("github-release", f"{tag}#99", 99)
    present = {item.reference for item in phase_one} | {run, tag, release.reference}
    remote = FakeRemote(present=present, releases=[release])
    target = phase_one[0].reference
    sleeps: list[float] = []

    report = cleanup.cleanup_manifest(
        manifest,
        remote,
        sleeper=sleeps.append,
        fail_resource=target,
    )

    assert report.completed is False
    assert report.stopped_phase == 1
    assert report.failures[0].resource.reference == target
    assert report.failures[0].attempts == 3
    assert remote.probe_calls.count(target) == 3
    assert target not in remote.delete_calls
    assert all(
        item.reference in remote.delete_calls
        for item in phase_one
        if item.reference != target
    )
    assert run not in remote.probe_calls
    assert tag in remote.present and release.reference in remote.present
    assert sleeps == [5.0, 15.0]


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

    report = cleanup.cleanup_manifest(manifest, remote, sleeper=lambda _: None)

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

    first = cleanup.cleanup_manifest(manifest, remote, sleeper=lambda _: None)

    assert first.stopped_phase == 3
    assert run not in remote.present
    assert tag in remote.present
    assert release.reference in remote.present
    assert remote.release_calls == []

    second = cleanup.cleanup_manifest(manifest, remote, sleeper=lambda _: None)

    assert second.completed is True
    assert tag not in remote.present
    assert release.reference not in remote.present
    assert remote.probe_calls.count(run) == 2


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

    report = cleanup.cleanup_manifest(manifest, remote, sleeper=lambda _: None)

    assert report.stopped_phase == 4
    assert report.failures[0].resource == first
    assert first.reference in remote.present
    assert second.reference not in remote.present
    assert second.reference in remote.delete_calls


def test_unbacked_release_failure_preserves_manifest_holder_for_retry() -> None:
    manifest = _manifest(chart=None, ghcr_members=[], ghcr_indexes=[])
    run, tag = _control_references(manifest)
    unbacked = cleanup.Resource("github-release", f"{tag}#1", 1)
    holder = cleanup.Resource("github-release", f"{tag}#2", 2, holds_manifest=True)
    remote = FakeRemote(
        present={run, tag, unbacked.reference, holder.reference},
        releases=[unbacked, holder],
    )
    remote.delete_errors[unbacked.reference] = [
        cleanup.RemoteError("forbidden", status=403)
    ]

    report = cleanup.cleanup_manifest(manifest, remote, sleeper=lambda _: None)

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
        "_all_pages",
        lambda path: [
            {
                "id": 77,
                "metadata": {"container": {"tags": ["v0.23.0", "shared-latest"]}},
            }
        ],
    )
    resource = cleanup.Resource(
        "ghcr-index",
        "ghcr.io/release-org/vllm-openai:v0.23.0",
        ("v0.23.0",),
    )

    with pytest.raises(cleanup.UnsafePackageVersion, match="shared-latest"):
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
    resources = cleanup.registry_resources(manifest)
    assert resources[0].identifier == ("v0.23.0", "v0.23.0-amd64", "v0.23.0-arm64")
    remote = cleanup.ProductionRemote("release-org/unified-cache-management", "token")
    monkeypatch.setattr(remote, "_owner_package_prefix", lambda: "/users/release-org")
    monkeypatch.setattr(
        remote,
        "_all_pages",
        lambda path: [
            {
                "id": 77,
                "metadata": {"container": {"tags": ["v0.23.0", "v0.23.0-amd64"]}},
            }
        ],
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
