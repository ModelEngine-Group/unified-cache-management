from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".github" / "release"))
release_body = importlib.import_module("ucm_release.release_body")


@pytest.mark.parametrize("body", [None, "", "## 更新说明\n\n修复缓存问题。\n"])
def test_appends_pipeline_notes_and_is_idempotent(body: str | None) -> None:
    notes = "Status: `release-open`\n\nRelease artifacts are building.\n"
    merged = release_body.merge_body(body, notes)
    assert merged.startswith(body or "")
    assert notes in merged
    assert merged.count(release_body.BEGIN) == 1
    assert release_body.merge_body(merged.rstrip("\n"), notes) == merged


def test_stage_updates_preserve_manual_notes_on_both_sides() -> None:
    prefix = "## Changes\n\n- Fix cache eviction.\n\n"
    suffix = "\n\n## Upgrade instructions\n\nKeep the existing config.\n"
    body = release_body.merge_body(prefix, "Status: `release-open`").rstrip("\n")
    body += suffix
    for status in ("artifacts-ready", "images-failed", "complete"):
        body = release_body.merge_body(body, f"Status: `{status}`")
        assert body == (
            prefix
            + f"{release_body.BEGIN}\nStatus: `{status}`\n{release_body.END}"
            + suffix
        )


def test_unmarked_legacy_status_is_preserved() -> None:
    body = "Status: `artifacts-ready`\n\nManual addition below the old status.\n"
    assert release_body.merge_body(body, "Status: `complete`").startswith(body)


@pytest.mark.parametrize(
    "body",
    [
        release_body.BEGIN,
        release_body.END + release_body.BEGIN,
        release_body.BEGIN * 2 + release_body.END,
    ],
)
def test_ambiguous_markers_are_rejected_without_guessing_what_to_delete(
    body: str,
) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        release_body.merge_body(body, "Status: `complete`")


def test_workflow_open_failure_and_rerun_preserve_release_body(tmp_path: Path) -> None:
    """Execute the real shell steps against a local gh stub, with no GitHub writes."""
    release_path = tmp_path / "release.json"
    original = "## 更新说明\n\n保留人工发布记录。\n"
    release_path.write_text(
        json.dumps(
            {
                "id": 42,
                "tag_name": "v1.0.0",
                "name": "v1.0.0",
                "body": original,
                "draft": False,
                "prerelease": False,
                "assets": [],
            }
        ),
        encoding="utf-8",
    )
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['TEST_RELEASE_JSON'])\n"
        "release = json.loads(path.read_text())\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['api', '--paginate']:\n"
        "    print(json.dumps([[release]]))\n"
        "elif args[:3] == ['api', '--method', 'PATCH']:\n"
        "    for index, arg in enumerate(args):\n"
        "        if arg in ('-f', '-F'):\n"
        "            key, value = args[index + 1].split('=', 1)\n"
        "            release[key] = json.loads(value) if arg == '-F' else value\n"
        "    path.write_text(json.dumps(release))\n"
        "else:\n"
        "    raise SystemExit(f'Unexpected gh call: {args}')\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "TEST_RELEASE_JSON": str(release_path),
        "GH_REPO": "example/ucm",
        "GITHUB_REPOSITORY": "example/ucm",
        "RELEASE_TAG": "v1.0.0",
        "RELEASE_TYPE": "stable",
        "IS_PRERELEASE": "false",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        **{
            f"{stage}_OUTCOME": "failure"
            for stage in ("CANDIDATE", "INSPECT", "PROBE", "RESOLVE", "BUILDER", "PLAN")
        },
    }
    jobs = yaml.safe_load(
        (ROOT / ".github/workflows/release-ucm.yml").read_text(encoding="utf-8")
    )["jobs"]
    for job, status in (
        ("open-release", "release-open"),
        ("report-planning-failure", "artifacts-failed"),
        ("open-release", "release-open"),
    ):
        run = next(step["run"] for step in jobs[job]["steps"] if "run" in step)
        subprocess.run(
            ["bash", "-e", "-c", run],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        document = json.loads(release_path.read_text(encoding="utf-8"))
        body = document["body"]
        assert body.startswith(original)
        assert body.count("Status:") == 1
        assert f"Status: `{status}`" in body
        # A rerun must recognize the marked section after manual release notes,
        # even when a previous attempt already uploaded an artifact.
        document["assets"] = [{"name": "backend.whl"}]
        release_path.write_text(json.dumps(document), encoding="utf-8")
