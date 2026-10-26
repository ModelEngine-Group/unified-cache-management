"""Exercise the shared installer without downloading binaries."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
ACTION = ROOT / ".github/actions/setup-crane"


def _archive() -> bytes:
    buffer = io.BytesIO()
    binary = b"#!/bin/sh\nprintf 'test-crane\\n'\n"
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("crane")
        member.mode = 0o755
        member.size = len(binary)
        archive.addfile(member, io.BytesIO(binary))
    return buffer.getvalue()


def _run_action(tmp_path: Path, payload: bytes, curl_status: int = 0):
    response = tmp_path / "archive.tar.gz"
    response.write_bytes(payload)
    curl = tmp_path / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "sys.stdout.buffer.write(Path(os.environ['TEST_CURL_RESPONSE']).read_bytes())\n"
        "raise SystemExit(int(os.environ['TEST_CURL_STATUS']))\n"
    )
    curl.chmod(0o755)
    path_file = tmp_path / "github-path"
    path_file.write_text("/existing/path\n")
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "RUNNER_TEMP": str(tmp_path / "runner temp"),
        "GITHUB_ACTION_PATH": str(ACTION),
        "GITHUB_PATH": str(path_file),
        "TEST_CURL_RESPONSE": str(response),
        "TEST_CURL_STATUS": str(curl_status),
    }
    action = yaml.safe_load((ACTION / "action.yml").read_text())
    assert action["runs"]["using"] == "composite"
    step = action["runs"]["steps"][0]
    assert step["shell"] == "bash"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, path_file, Path(env["RUNNER_TEMP"]) / "crane"


def test_composite_installs_downloaded_binary_and_appends_its_directory(tmp_path):
    result, path_file, directory = _run_action(tmp_path, _archive())
    assert result.returncode == 0, result.stderr
    assert path_file.read_text() == f"/existing/path\n{directory}\n"
    assert subprocess.check_output([str(directory / "crane")], text=True) == (
        "test-crane\n"
    )


@pytest.mark.parametrize("failure", ["download", "extract"])
def test_failed_install_does_not_append_github_path(tmp_path, failure):
    # A successful tar must not hide a failed curl at the other end of the pipe.
    payload, status = (_archive(), 22) if failure == "download" else (b"invalid", 0)
    result, path_file, _ = _run_action(tmp_path, payload, status)
    assert result.returncode != 0
    assert path_file.read_text() == "/existing/path\n"
