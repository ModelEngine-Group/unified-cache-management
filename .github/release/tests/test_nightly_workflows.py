"""Nightly completion, publication evidence, and privileged workflow contracts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"


def _load(name: str) -> dict:
    value = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    if True in value:
        value["on"] = value.pop(True)
    return value


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_nightly_cleanup_covers_all_completed_outcomes_and_uses_trusted_code() -> None:
    workflow = _load("nightly-cleanup.yml")
    assert workflow["on"]["workflow_run"] == {
        "workflows": [
            "UCM Nightly Release · Shanghai 02:00",
            "UCM Tag Release · Stable, Prerelease, Draft, and Nightly",
        ],
        "types": ["completed"],
    }
    assert workflow["on"]["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True
    job = workflow["jobs"]["cleanup"]
    assert "head_repository.full_name == github.repository" in job["if"]
    assert "workflow_run.name" not in job["if"]
    assert "workflow_run.path" in job["if"]
    assert "conclusion" not in job["if"]
    checkout = next(
        step for step in job["steps"] if "checkout@" in step.get("uses", "")
    )
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    retention = next(step for step in job["steps"] if step.get("id") == "retention")
    assert "--current-tag" not in retention["run"]
    assert "--release-type nightly" in retention["run"]
    assert "--inventory" in retention["run"] and "--dry-run" in retention["run"]
    assert "--report out/cleanup/report.json" in retention["run"]
    assert "continue-on-error" not in retention
    assert job["steps"][-1]["if"] == "${{ always() }}"


def test_nightly_mutations_share_an_outer_lock_separate_from_the_core_tag_lock() -> (
    None
):
    for name in (
        "release-nightly.yml",
        "release-tag.yml",
        "nightly-cleanup.yml",
        "cleanup-ucm-release.yml",
    ):
        concurrency = _load(name)["concurrency"]
        assert "ucm-nightly-" in concurrency["group"]
        assert concurrency["queue"] == "max"
        assert concurrency["cancel-in-progress"] is False
    core = _load("release-ucm.yml")
    assert (
        core["concurrency"]["group"]
        == "ucm-release-${{ github.repository_id }}-${{ inputs.git_tag }}"
    )
    retention = next(
        step
        for step in core["jobs"]["update-release-images"]["steps"]
        if step.get("id") == "retention"
    )
    assert retention["if"] == "${{ inputs.release_type != 'nightly' }}"


def test_nightly_ownership_precedes_creation_and_exact_targets_gate_publishers() -> (
    None
):
    prepare = _load("release-nightly.yml")["jobs"]["prepare-nightly"]
    script = next(
        step["run"] for step in prepare["steps"] if step.get("id") == "prepare"
    )
    assert script.index("cleanup.py record") < script.index("gh api --method POST")
    assert prepare["steps"][-1]["if"].startswith("${{ always()")
    jobs = _load("release-ucm.yml")["jobs"]
    owner = _step(
        jobs["open-release"], "Persist Nightly ownership on the exact Release ID"
    )
    targets = _step(jobs["plan"], "Persist exact Nightly targets before publication")
    for step in (owner, targets):
        assert step["if"] == "${{ inputs.release_type == 'nightly' }}"
        assert "--previous" in step["run"]
        assert "/releases/${RELEASE_ID}" in step["run"]
        assert "cmp " in step["run"]
    assert "--plan out/plan/release-plan.json" in targets["run"]
    for name in ("build-images", "publish-chart-oci", "publish-image-indexes"):
        assert "plan" in jobs[name]["needs"]


@pytest.mark.parametrize(
    "field",
    [
        "repository",
        "source_sha",
        "runs",
        "kind",
        "tag",
        "valid",
        "prior_attempt",
        "no_binding",
        "custom_name",
        "custom_nightly_name",
        "impostor_path",
    ],
)
def test_completed_tag_authorization_uses_immutable_binding_and_accepts_retries(
    tmp_path: Path, field: str
) -> None:
    if shutil.which("jq") is None:
        pytest.skip(
            "jq is needed to execute the actual Actions source-verification step"
        )
    repository = "release-org/unified-cache-management"
    run_id, attempt = 42, 2
    source_sha = "a" * 40
    event = {
        "workflow_run": {
            "head_repository": {"full_name": repository},
            "status": "completed",
            "id": run_id,
            "run_attempt": attempt,
            "head_sha": source_sha,
            "name": "UCM Tag Release · Stable, Prerelease, Draft, and Nightly",
            "path": ".github/workflows/release-tag.yml",
            "event": "push",
            "head_branch": "nightly/v0.9.0-20261009-1",
        }
    }
    binding = {
        "kind": "ucm-release-cleanup",
        "schema_version": 1,
        "repository": repository,
        "tag": "nightly/v0.9.0-20261009-1",
        "source_sha": source_sha,
        "release_ids": [],
        "runs": [{"id": run_id, "attempt": attempt}],
        "resources": [],
    }
    invalid = {
        "repository": "other/repository",
        "source_sha": "b" * 40,
        "runs": [{"id": run_id, "attempt": 1}],
        "kind": "untrusted-record",
        "tag": "nightly/v0.9.0-20261008-1",
    }
    if field in invalid:
        binding[field] = invalid[field]
    if field in {"custom_name", "custom_nightly_name"}:
        event["workflow_run"]["name"] = "Nightly cleanup fixture - failure"
    if field == "custom_nightly_name":
        event["workflow_run"].update(
            path=".github/workflows/release-nightly.yml@develop",
            event="workflow_dispatch",
            head_branch="develop",
        )
    if field == "impostor_path":
        event["workflow_run"]["path"] = ".github/workflows/impostor.yml"
    binding_attempt = 1 if field == "prior_attempt" else attempt
    if field == "prior_attempt":
        binding["runs"] = [{"id": run_id, "attempt": binding_attempt}]
    archive = tmp_path / "binding.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("release-cleanup.json", json.dumps(binding))
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    python = bin_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if sys.argv[1].endswith('cleanup.py'):\n"
        "    pathlib.Path('record-calls.json').write_text(json.dumps(sys.argv[2:]))\n"
        "else:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    (tmp_path / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    gh = bin_path / "gh"
    artifact_entries = (
        []
        if field == "no_binding"
        else [
            {
                "name": f"ucm-nightly-cleanup-run-{run_id}-attempt-{binding_attempt}",
                "id": 9,
                "expired": False,
            }
        ]
    )
    response = json.dumps([{"artifacts": artifact_entries}])
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "if any('/artifacts?' in arg for arg in sys.argv):\n"
        f"    print({response!r})\n"
        "elif any('/artifacts/9/zip' in arg for arg in sys.argv):\n"
        "    sys.stdout.buffer.write(pathlib.Path(os.environ['MOCK_ARCHIVE']).read_bytes())\n"
        "else:\n"
        "    raise SystemExit('Unexpected API call')\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    output_path = tmp_path / "output"
    job = _load("nightly-cleanup.yml")["jobs"]["cleanup"]
    source = next(step for step in job["steps"] if step.get("id") == "source")
    result = subprocess.run(
        ["bash", "-c", source["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_path}{os.pathsep}{os.environ['PATH']}",
            "GH_REPO": repository,
            "GITHUB_EVENT_NAME": "workflow_run",
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_OUTPUT": str(output_path),
            "MOCK_ARCHIVE": str(archive),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if field in invalid or field == "impostor_path":
        assert result.returncode != 0
        assert (
            not output_path.exists() or "eligible=true" not in output_path.read_text()
        )
        assert not (tmp_path / "record-calls.json").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "eligible=true" in output_path.read_text()
        if field not in {"no_binding", "custom_nightly_name"}:
            arguments = json.loads((tmp_path / "record-calls.json").read_text())
            assert arguments[arguments.index("--run-attempt") + 1] == str(attempt)
