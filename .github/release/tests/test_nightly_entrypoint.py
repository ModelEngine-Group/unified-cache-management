from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_cleanup import FakeRemote, cleanup

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "release-org/unified-cache-management"


def record(day: int, release_id: int | None = None):
    return cleanup.nightly_cleanup.record_for_tag(
        REPOSITORY,
        f"nightly/v0.8.0-202610{day:02d}-1",
        "a" * 40,
        1000 + day,
        1,
        release_id,
    )


def world(records):
    present = set()
    for item in records:
        present.add(item["tag"])
        present.update(f"{item['tag']}#{value}" for value in item["release_ids"])
        present.update(
            f"https://github.com/{REPOSITORY}/actions/runs/{run['id']}"
            for run in item["runs"]
        )
    return FakeRemote(present=present)


def arguments(tmp_path, *, dry_run=False):
    return SimpleNamespace(
        dry_run=dry_run,
        fail_resource=None,
        report=tmp_path / "report.json",
        max_count=2,
        pypi_enabled=False,
        inventory=None,
    )


def test_record_cli_merges_attempts_without_requiring_a_token(tmp_path):
    previous = record(1, 99)
    source = tmp_path / "previous.json"
    source.write_text(json.dumps(previous))
    output = tmp_path / "record.json"
    environment = os.environ.copy()
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "PYTHONPATH"):
        environment.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ucm_release/cleanup.py"),
            "record",
            "--repository",
            REPOSITORY,
            "--tag",
            previous["tag"],
            "--source-sha",
            "a" * 40,
            "--run-id",
            "1001",
            "--run-attempt",
            "2",
            "--release-id",
            "99",
            "--previous",
            str(source),
            "--output",
            str(output),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["runs"] == [
        {"id": 1001, "attempt": 1},
        {"id": 1001, "attempt": 2},
    ]


def test_nightly_retention_uses_actual_objects_and_protects_active_targets(
    tmp_path, monkeypatch
):
    records = [record(day, 90 + day) for day in (1, 2, 3)]
    remote = world(records)
    catalog = SimpleNamespace(
        records=records, orphans=(), active_tags={records[0]["tag"]}, blocked={}
    )
    monkeypatch.setattr(cleanup, "_nightly_catalog", lambda args, client: catalog)
    args = arguments(tmp_path)
    assert cleanup._run_nightly_retention(args, remote) == []
    report = json.loads(args.report.read_text())
    assert report["status"] == "deferred"
    assert [item["status"] for item in report["results"]] == [
        "kept",
        "kept",
        "deferred",
    ]
    assert remote.delete_calls == []


def test_blocked_target_does_not_prevent_other_independent_cleanup(tmp_path):
    blocked, clean = record(1, 91), record(2, 92)
    remote = world([blocked, clean])
    args = arguments(tmp_path)
    failures = cleanup._run_nightly_targets(
        args, remote, [blocked, clean], blocked={blocked["tag"]: "missing plan"}
    )
    assert len(failures) == 1
    assert blocked["tag"] in remote.present
    assert clean["tag"] not in remote.present
    report = json.loads(args.report.read_text())
    assert report["status"] == "failed"
    assert [item["status"] for item in report["results"]] == ["blocked", "deleted"]


def test_recovery_metadata_403_blocks_only_its_version(tmp_path):
    targets = [record(1, 91), record(2, 92)]

    class DeniedRemote(FakeRemote):
        def recovery_release_ids(self, target):
            if target["tag"] == targets[0]["tag"]:
                raise cleanup.RemoteError("asset metadata forbidden", status=403)
            return super().recovery_release_ids(target)

    remote = DeniedRemote(present=world(targets).present)
    args = arguments(tmp_path)
    failures = cleanup._run_nightly_targets(args, remote, targets, blocked={})
    assert len(failures) == 1
    assert targets[0]["tag"] in remote.present
    assert targets[1]["tag"] not in remote.present
    assert json.loads(args.report.read_text())["status"] == "failed"


def test_dry_run_only_reads_and_writes_its_local_report(tmp_path):
    target = record(1, 91)
    remote = world([target])
    original = remote.present.copy()
    args = arguments(tmp_path, dry_run=True)
    assert cleanup._run_nightly_targets(args, remote, [target], blocked={}) == []
    assert remote.present == original
    assert remote.delete_calls == []
    assert json.loads(args.report.read_text())["results"][0]["status"] == "would-delete"


@pytest.mark.parametrize("release_id", [None, 91])
def test_missing_release_attachment_keeps_artifact_run_when_tag_delete_fails(
    release_id,
):
    target = record(1, release_id)
    remote = world([target])
    remote.delete_errors[target["tag"]] = [cleanup.RemoteError("forbidden", status=403)]
    report = cleanup.cleanup_record(target, remote, sleeper=lambda _: None)
    run = f"https://github.com/{REPOSITORY}/actions/runs/1001"
    assert report.completed is False
    assert report.stopped_phase == 3
    assert run in remote.present
    assert run not in remote.delete_calls


def test_recovery_asset_must_cover_the_latest_attempt_and_resources(monkeypatch):
    target = record(1, 91)
    saved = json.loads(json.dumps(target))
    target["runs"].append({"id": 1001, "attempt": 2})
    remote = cleanup.ProductionRemote(REPOSITORY, "token")
    monkeypatch.setattr(
        remote,
        "_github_json",
        lambda method, path: {
            "id": 91,
            "assets": [{"id": 700, "name": "release-cleanup.json"}],
        },
    )
    monkeypatch.setattr(
        remote, "_request", lambda *args, **kwargs: json.dumps(saved).encode()
    )
    assert remote.recovery_release_ids(target) == set()
    saved["runs"] = target["runs"]
    assert remote.recovery_release_ids(target) == {91}


def test_signed_download_redirect_does_not_forward_the_credential():
    request = urllib.request.Request(
        "https://api.github.com/repos/release-org/repo/actions/artifacts/1/zip",
        headers={
            "Authorization": "Bearer test-only",
            "Accept": "application/octet-stream",
        },
    )
    handler = cleanup._GitHubRedirectHandler()
    redirected = handler.redirect_request(
        request, None, 302, "Found", {}, "https://blob.example.test/signed.zip"
    )
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("Accept") == "application/octet-stream"
    with pytest.raises(cleanup.CleanupError):
        handler.redirect_request(
            request, None, 302, "Found", {}, "http://blob.example.test/file"
        )
