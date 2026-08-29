from __future__ import annotations

import os
import subprocess
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
RETRY = RELEASE_ROOT / "retry-registry-command.sh"
FAKE_COMMAND = r"""
count_file="$1"
failures="$2"
message="$3"
failure_status="$4"
count=0
if [ -f "${count_file}" ]; then
  count="$(<"${count_file}")"
fi
count=$((count + 1))
printf '%s\n' "${count}" >"${count_file}"
if [ "${count}" -le "${failures}" ]; then
  printf '%s\n' "${message}" >&2
  exit "${failure_status}"
fi
printf 'success\n'
"""


def _run_retry(
    tmp_path: Path, *, failures: int, message: str, failure_status: int = 75
) -> tuple[subprocess.CompletedProcess[str], int]:
    count_file = tmp_path / "attempt-count"
    environment = os.environ.copy()
    environment["UCM_REGISTRY_RETRY_DELAYS"] = "0 0 0 0"
    completed = subprocess.run(
        [
            "bash",
            str(RETRY),
            str(tmp_path / "command.log"),
            "bash",
            "-c",
            FAKE_COMMAND,
            "fake-command",
            str(count_file),
            str(failures),
            message,
            str(failure_status),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed, int(count_file.read_text(encoding="utf-8").strip())


def test_registry_rate_limit_retries_until_the_command_succeeds(
    tmp_path: Path,
) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=2,
        message="toomanyrequests: retry-after: 100ms",
    )

    assert completed.returncode == 0
    assert attempts == 3
    assert completed.stderr.count("rate-limited; retrying") == 2


def test_non_rate_limit_failure_is_not_retried(tmp_path: Path) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="manifest unknown",
        failure_status=42,
    )

    assert completed.returncode == 42
    assert attempts == 1
    assert "non-retryable error" in completed.stderr


def test_registry_rate_limit_stops_at_the_bounded_attempt_count(
    tmp_path: Path,
) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="unexpected status code 429",
    )

    assert completed.returncode == 75
    assert attempts == 5
    assert "failed after 5 attempts" in completed.stderr


def test_unrelated_number_containing_429_is_not_a_rate_limit(tmp_path: Path) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="blob 1429 is unavailable",
        failure_status=42,
    )

    assert completed.returncode == 42
    assert attempts == 1
    assert "non-retryable error" in completed.stderr


def test_retry_fails_when_the_command_log_cannot_be_written(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["bash", str(RETRY), str(tmp_path), "bash", "-c", "printf 'success\\n'"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "retry log could not be written" in completed.stderr
