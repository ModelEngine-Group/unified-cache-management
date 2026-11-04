"""Release lookup must distinguish absence, ambiguity and lookup failure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/release/find-github-release.sh"
REPOSITORY = "release-org/unified-cache-management"
TAG = "draft/v0.8.0-1"


def _run_lookup(tmp_path: Path, response: str, api_status: int = 0):
    response_path = tmp_path / "response.json"
    response_path.write_text(response)
    calls = tmp_path / "gh.json"
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['TEST_GH_CALLS']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.stdout.write(Path(os.environ['TEST_GH_RESPONSE']).read_text())\n"
        "raise SystemExit(int(os.environ['TEST_GH_STATUS']))\n"
    )
    gh.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPT), REPOSITORY, TAG],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "TEST_GH_CALLS": str(calls),
            "TEST_GH_RESPONSE": str(response_path),
            "TEST_GH_STATUS": str(api_status),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert json.loads(calls.read_text()) == [
        "api",
        "--paginate",
        "--slurp",
        f"/repos/{REPOSITORY}/releases?per_page=100",
    ]
    return result


@pytest.mark.parametrize("match", [None, "exact-tag", "draft-name"])
def test_lookup_matches_exact_identity_across_pages(tmp_path, match):
    unrelated = [
        {"id": 1, "tag_name": "another-tag", "name": TAG, "draft": True},
        {"id": 2, "tag_name": "untagged-public", "name": TAG, "draft": False},
        {"id": 3, "tag_name": "untagged-other", "name": "another-tag", "draft": True},
    ]
    expected = []
    if match:
        expected = [
            {
                "id": 4,
                "tag_name": TAG if match == "exact-tag" else "untagged-draft",
                "name": "Display name" if match == "exact-tag" else TAG,
                "draft": match == "draft-name",
            }
        ]
    result = _run_lookup(tmp_path, json.dumps([unrelated, expected]))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_ambiguous_lookup_returns_three_without_selecting_a_release(tmp_path):
    response = [
        [{"id": 1, "tag_name": TAG, "draft": False}],
        [{"id": 2, "tag_name": "untagged-draft", "name": TAG, "draft": True}],
    ]
    result = _run_lookup(tmp_path, json.dumps(response))
    assert result.returncode == 3
    assert not result.stdout
    assert "refusing ambiguity" in result.stderr


@pytest.mark.parametrize(
    "response,api_status", [("[[]]", 1), ("invalid JSON", 0), ("", 0)]
)
def test_api_and_parse_failures_do_not_look_like_absence(
    tmp_path, response, api_status
):
    result = _run_lookup(tmp_path, response, api_status)
    assert result.returncode == 1
    assert not result.stdout
