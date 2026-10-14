from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
INSTALL = RELEASE_ROOT / "install-validate-wheel-runtime.sh"


def _run_install(
    tmp_path: Path, *, status: int = 0, distribution: str = "uc-manager-ascend-a2"
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()
    wheel = wheel_directory / "uc_manager_ascend_a2-0.2.0-cp312-cp312-linux_aarch64.whl"
    wheel.write_bytes(b"local Wheel")
    (wheel_directory / "wheel-result.json").write_text(
        json.dumps(
            {
                "distribution": distribution,
                "version": "0.2.0",
                "external_libraries": ["libruntime.so", "libascend_hal.so"],
                "deferred_external_libraries": ["libascend_hal.so"],
                "dependencies": ["wrapt==1.17.2"],
                "glibc_floor": "GLIBC_2.34",
            }
        ),
        encoding="utf-8",
    )
    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    docker = binary_directory / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as calls:\n"
        "    calls.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(int(os.environ['DOCKER_RUN_STATUS']) if sys.argv[1] == 'run' else 0)\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    calls = tmp_path / "docker-calls.jsonl"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{binary_directory}{os.pathsep}{environment['PATH']}",
            "DOCKER_CALLS": str(calls),
            "DOCKER_RUN_STATUS": str(status),
            "RUNTIME_IMAGE": "quay.io/ascend/vllm-ascend@sha256:" + "a" * 64,
            "RUNTIME_PLATFORM": "linux/arm64",
            "BACKEND_DISTRIBUTION": "uc-manager-ascend-a2",
            "PLATFORM_ARG": "ascend-a2",
        }
    )
    completed = subprocess.run(
        ["bash", str(INSTALL), str(wheel_directory)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    recorded = (
        [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
        if calls.exists()
        else []
    )
    return completed, recorded, wheel


@pytest.mark.parametrize("status", [0, 53], ids=["pass", "runtime-failure"])
def test_local_wheel_validation_uses_native_pinned_runtime_and_propagates_failure(
    tmp_path: Path, status: int
) -> None:
    completed, calls, wheel = _run_install(tmp_path, status=status)

    assert completed.returncode == status, completed.stderr
    assert len(calls) == 2  # Validation failure never retries another Runtime or Builder.
    pinned_runtime = "quay.io/ascend/vllm-ascend@sha256:" + "a" * 64
    assert calls[0] == ["pull", "--platform", "linux/arm64", pinned_runtime]
    run = calls[1]
    assert run[:6] == ["run", "--rm", "--platform", "linux/arm64", "--entrypoint", "bash"]
    assert f"{wheel}:/tmp/{wheel.name}:ro" in run
    assert run[-3:-1] == [pinned_runtime, "-c"]
    assert "EXPECTED_BACKEND_DISTRIBUTION=uc-manager-ascend-a2" in run
    assert "EXPECTED_UCM_VERSION=0.2.0" in run
    assert "EXPECTED_WHEEL_GLIBC_FLOOR=GLIBC_2.34" in run
    assert 'EXPECTED_RUNTIME_REQUIREMENTS=["wrapt==1.17.2"]' in run
    assert 'DEFERRED_EXTERNAL_LIBRARIES=["libascend_hal.so"]' in run


def test_mismatched_wheel_result_is_rejected_before_runtime_installation(
    tmp_path: Path,
) -> None:
    completed, calls, _ = _run_install(tmp_path, distribution="uc-manager-cuda")

    assert completed.returncode != 0
    assert calls == []
