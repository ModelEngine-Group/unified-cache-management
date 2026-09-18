from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import trigger_rtd
import yaml
from manifest_fixtures import manifest_fixture

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "scope,projects,token,enabled",
    [
        ("fork", ("", ""), "", False),
        ("fork", ("docs-en", "docs-zh"), "test-token", True),
        ("official", ("docs-en", "docs-zh"), "test-token", True),
        ("official", ("", ""), "", None),
        ("fork", ("docs-en", ""), "test-token", None),
        ("fork", ("docs-en", "docs-zh"), "", None),
    ],
)
def test_release_docs_configuration(tmp_path, scope, projects, token, enabled):
    workflow = yaml.safe_load((ROOT / ".github/workflows/release-ucm.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["release-preflight"]["steps"]
        if "RTD_PROJECT_EN" in step.get("env", {})
    )
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={
            **os.environ,
            "PUBLICATION_SCOPE": scope,
            "RTD_PROJECT_EN": projects[0],
            "RTD_PROJECT_ZH": projects[1],
            "RTD_API_TOKEN": token,
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    if enabled is None:
        assert result.returncode != 0
        assert "::error::Configure RTD_PROJECT_EN" in result.stdout
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text().strip() == f"enabled={str(enabled).lower()}"


@pytest.mark.parametrize("publish_docs", [False, True])
def test_release_acceptance_only_requires_enabled_docs(tmp_path, publish_docs):
    workflow = yaml.safe_load((ROOT / ".github/workflows/release-ucm.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["verify-release-delivery"]["steps"]
        if "run" in step
    )
    # Execute the workflow's actual receipt logic without contacting GitHub or RTD.
    script = step["run"].split("python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    source_sha = "a" * 40
    documents = {
        "input/github-release.json": {
            "tagName": "v0.8.0rc1",
            "isDraft": False,
            "isPrerelease": True,
        },
        "input/final/release-state.json": {
            "release": {"status": "complete"},
            "pypi": {},
            "toolkit_package": {},
        },
        "input/chart/chart-delivery.json": {"status": "complete"},
    }
    if publish_docs:
        documents["input/docs/docs-receipt.json"] = {
            "status": "complete",
            "source_sha": source_sha,
        }
    for name, document in documents.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
    (tmp_path / "out").mkdir()
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={
            **os.environ,
            "GH_REPO": "example/ucm",
            "SOURCE_SHA": source_sha,
            "RELEASE_TAG": "v0.8.0rc1",
            "EXPECTED_PRERELEASE": "true",
            "GITHUB_RUN_ID": "123",
            "PUBLISH_DOCS": str(publish_docs).lower(),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    acceptance = json.loads((tmp_path / "out/acceptance.json").read_text())
    assert acceptance["status"] == "complete"
    if publish_docs:
        assert acceptance["docs"] == documents["input/docs/docs-receipt.json"]
    else:
        assert acceptance["docs"]["status"] == "skipped"


@pytest.mark.parametrize("current_stable", ["v0.9.0", "v0.9.1"])
def test_release_rebuild_does_not_move_stable_alias(
    monkeypatch, tmp_path, current_stable
):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_fixture("0.9.0")))
    calls = []

    def request(path, *, method="GET", data=None):
        calls.append((path, method, data))
        if path.endswith("/projects/docs-en/") or path.endswith("/projects/docs-zh/"):
            return {
                "repository": {"url": "https://github.com/example/ucm.git"},
                "language": {"code": "zh-cn" if "docs-zh" in path else "en"},
            }
        if path.endswith("/translations/"):
            return {"results": [{"slug": "docs-zh"}]}
        if method == "GET":
            return {"active": True, "ref": current_stable}
        return {"build": {"id": len(calls)}}

    monkeypatch.setattr(trigger_rtd, "request", request)
    triggered = trigger_rtd.notify_release(
        "example/ucm", manifest_path, ["docs-en", "docs-zh"]
    )
    assert not any(method == "PATCH" for _, method, _ in calls)
    triggered = {f"{item['project']}/{item['version']}" for item in triggered}
    for project in ("docs-en", "docs-zh"):
        assert f"{project}/v0.9.0" in triggered
        assert f"{project}/latest" in triggered
        assert (f"{project}/stable" in triggered) == (current_stable == "v0.9.0")


def test_rtd_project_must_belong_to_the_release_repository(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_fixture("0.9.0")))
    calls = []

    def request(path, **kwargs):
        calls.append((path, kwargs))
        return {"repository": {"url": "https://github.com/other/ucm.git"}}

    monkeypatch.setattr(trigger_rtd, "request", request)
    with pytest.raises(ValueError, match="another repository"):
        trigger_rtd.notify_release("example/ucm", manifest_path, ["docs-en"])
    assert len(calls) == 1


def test_workflow_notifies_rtd_only_after_manifest_readback():
    workflow = yaml.safe_load((ROOT / ".github/workflows/release-ucm.yml").read_text())
    job = workflow["jobs"]["verify-release-docs"]
    assert "update-release-images" in job["needs"]
    steps = job["steps"]
    readback = next(
        i
        for i, step in enumerate(steps)
        if "gh release download" in step.get("run", "")
    )
    notification = next(
        i for i, step in enumerate(steps) if "trigger_rtd.py" in step.get("run", "")
    )
    assert notification > readback
    assert steps[notification]["env"]["GH_TOKEN"] == "${{ github.token }}"
    run = steps[notification]["run"]
    assert "--source-sha" in run and "--output out/docs-receipt.json" in run
    assert "RTD_PROJECT_EN" in run and "RTD_PROJECT_ZH" in run


@pytest.mark.parametrize("success,commit", [(False, "a" * 40), (True, "b" * 40)])
def test_rtd_finished_build_must_succeed_at_release_commit(
    monkeypatch, success, commit
):
    monkeypatch.setattr(
        trigger_rtd,
        "request",
        lambda path: {
            "state": {"code": "finished"},
            "success": success,
            "commit": commit,
            "version": "v0.9.0rc1",
            "error": "build error",
        },
    )
    with pytest.raises(RuntimeError):
        trigger_rtd.wait_for_builds(
            [{"project": "docs-en", "version": "v0.9.0rc1", "build_id": 12}], "a" * 40
        )


def test_rtd_wait_records_successful_build_and_public_url(monkeypatch):
    responses = iter(
        [
            {"state": {"code": "building"}},
            {
                "state": {"code": "finished"},
                "success": True,
                "commit": "a" * 40,
                "version": "v0.9.0rc1",
            },
            {"urls": {"documentation": "https://docs.example/en/v0.9.0rc1/"}},
        ]
    )
    monkeypatch.setattr(trigger_rtd, "request", lambda path: next(responses))
    monkeypatch.setattr(trigger_rtd.time, "sleep", lambda seconds: None)
    result = trigger_rtd.wait_for_builds(
        [{"project": "docs-en", "version": "v0.9.0rc1", "build_id": 12}], "a" * 40
    )
    assert result[0]["status"] == "complete"
    assert result[0]["build_id"] == 12
    assert result[0]["url"] == "https://docs.example/en/v0.9.0rc1/"


def test_public_doc_readback_rejects_another_release(monkeypatch):
    monkeypatch.setattr(trigger_rtd, "resolve_manifest", lambda repository: None)
    monkeypatch.setattr(trigger_rtd, "read_public", lambda url: b'{"wrong": "release"}')
    with pytest.raises(trigger_rtd.ManifestError, match="wrong release manifest"):
        trigger_rtd.verify_public_docs(
            [{"version": "v0.9.0rc1", "url": "https://docs.example/en/v0.9.0rc1/"}],
            {"release": {"version": "0.9.0rc1"}},
            "example/ucm",
        )


def test_latest_build_tracks_default_branch_without_requiring_tag_sha(monkeypatch):
    responses = iter(
        [
            {
                "state": {"code": "finished"},
                "success": True,
                "commit": "b" * 40,
                "version": "latest",
            },
            {"urls": {"documentation": "https://docs.example/en/latest/"}},
        ]
    )
    monkeypatch.setattr(trigger_rtd, "request", lambda path: next(responses))
    result = trigger_rtd.wait_for_builds(
        [{"project": "docs-en", "version": "latest", "build_id": 13}], "a" * 40
    )
    assert result[0]["commit"] == "b" * 40


def test_public_readback_identifies_client_without_sending_api_credentials(monkeypatch):
    monkeypatch.setenv("RTD_API_TOKEN", "private-rtd-token")
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"public documentation"

    def open_public(request, timeout):
        captured.append(request)
        return Response()

    monkeypatch.setattr(trigger_rtd, "urlopen", open_public)
    assert (
        trigger_rtd.read_public("https://docs.example/en/latest/")
        == b"public documentation"
    )
    headers = dict(captured[0].header_items())
    assert headers["User-agent"] == "ucm-docs-readback/1"
    assert "Authorization" not in headers
