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
@pytest.mark.parametrize(
    "workflow_name,job_name",
    [("release-ucm.yml", "release-preflight"), ("docs-check.yml", "publish-latest")],
)
def test_docs_configuration(
    tmp_path, scope, projects, token, enabled, workflow_name, job_name
):
    workflow = yaml.safe_load((ROOT / ".github/workflows" / workflow_name).read_text())
    step = next(
        step
        for step in workflow["jobs"][job_name]["steps"]
        if "RTD_PROJECT_EN" in step.get("env", {})
    )
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={
            **os.environ,
            "PUBLICATION_SCOPE": scope,
            "GITHUB_REPOSITORY": (
                "ModelEngine-Group/unified-cache-management"
                if scope == "official"
                else "example/ucm"
            ),
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
                "default_branch": "develop",
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


@pytest.mark.parametrize(
    "invalid", [None, "repository", "language", "branch", "translation"]
)
def test_latest_validates_both_projects_before_triggering(monkeypatch, invalid):
    calls = []

    def request(path, *, method="GET", data=None):
        calls.append((path, method))
        if method == "POST":
            return {"build": {"id": len(calls)}}
        if path.endswith("/translations/"):
            return {
                "results": [] if invalid == "translation" else [{"slug": "docs-zh"}]
            }
        chinese = "docs-zh" in path
        return {
            "repository": {
                "url": (
                    "https://github.com/other/ucm"
                    if chinese and invalid == "repository"
                    else "https://github.com/example/ucm"
                )
            },
            "language": {
                "code": "en" if not chinese or invalid == "language" else "zh-cn"
            },
            "default_branch": (
                "old-branch" if chinese and invalid == "branch" else "develop"
            ),
        }

    monkeypatch.setattr(trigger_rtd, "request", request)
    if invalid:
        with pytest.raises(ValueError):
            trigger_rtd.notify_latest("example/ucm", ["docs-en", "docs-zh"])
        assert not any(method == "POST" for _, method in calls)
    else:
        builds = trigger_rtd.notify_latest("example/ucm", ["docs-en", "docs-zh"])
        assert [
            (item["project"], item["language"], item["version"]) for item in builds
        ] == [
            ("docs-en", "en", "latest"),
            ("docs-zh", "zh-cn", "latest"),
        ]
        assert [path for path, method in calls if method == "POST"] == [
            "/projects/docs-en/versions/latest/builds/",
            "/projects/docs-zh/versions/latest/builds/",
        ]


@pytest.mark.parametrize("state", ["building", "cancelled"])
def test_latest_does_not_succeed_on_timeout_or_cancellation(monkeypatch, state):
    monkeypatch.setattr(trigger_rtd, "request", lambda path: {"state": {"code": state}})
    with pytest.raises(TimeoutError if state == "building" else RuntimeError):
        trigger_rtd.wait_for_builds(
            [{"project": "docs-en", "version": "latest", "build_id": 1}],
            "a" * 40,
            timeout=0,
        )


def test_latest_cli_writes_verified_build_receipt(monkeypatch, tmp_path):
    output = tmp_path / "receipt.json"
    builds = [{"project": "docs-en", "version": "latest", "build_id": 1}]
    completed = [{**builds[0], "commit": "b" * 40, "status": "complete"}]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trigger_rtd.py",
            "--latest",
            "--repository",
            "example/ucm",
            "--project",
            "docs-en",
            "--project",
            "docs-zh",
            "--source-sha",
            "a" * 40,
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(trigger_rtd, "notify_latest", lambda *args: builds)
    monkeypatch.setattr(trigger_rtd, "wait_for_builds", lambda *args: completed)
    verified = []
    monkeypatch.setattr(
        trigger_rtd, "verify_public_docs", lambda *args: verified.append(args)
    )
    trigger_rtd.main()
    assert verified == [(completed, None, "example/ucm")]
    assert json.loads(output.read_text()) == {
        "status": "complete",
        "source_sha": "a" * 40,
        "builds": completed,
    }
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--manifest", "release.json"])
    with pytest.raises(SystemExit) as error:
        trigger_rtd.main()
    assert error.value.code == 2


def test_latest_publication_waits_for_checks_and_keeps_secrets_out_of_pr_jobs():
    workflow = yaml.safe_load((ROOT / ".github/workflows/docs-check.yml").read_text())
    job = workflow["jobs"]["publish-latest"]
    assert set(job["needs"]) == {"release", "check"}
    assert job["if"].strip() == (
        "${{ github.ref == 'refs/heads/develop' &&\n"
        "    (github.event_name == 'push' || github.event_name == 'workflow_dispatch') }}"
    )
    for name in ("release", "check"):
        assert "RTD_API_TOKEN" not in json.dumps(workflow["jobs"][name])
    assert any(
        "trigger_rtd.py --latest" in step.get("run", "") for step in job["steps"]
    )
