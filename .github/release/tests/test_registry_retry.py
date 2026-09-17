from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
RETRY = RELEASE_ROOT / "retry-registry-command.sh"
QUAY_BLOB_URL = (
    "https://quay.io/v2/ascend/vllm-ascend/blobs/sha256:"
    "43fb28be3dcbd7fb3ad28be98625645ad928161bc2a501869b02d688bc51646b"
)
QUAY_BLOB_UNAUTHORIZED = (
    f"unexpected status from GET request to {QUAY_BLOB_URL}: 401 Unauthorized"
)
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
    tmp_path: Path,
    *,
    failures: int,
    message: str,
    failure_status: int = 75,
    retry_transport: bool = False,
    retry_quay_blob: bool = False,
    rate_limit_marker: Path | None = None,
    rate_limit_scope: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], int]:
    count_file = tmp_path / "attempt-count"
    environment = os.environ.copy()
    environment["UCM_REGISTRY_RETRY_DELAYS"] = "0 0 0 0"
    command = ["bash", str(RETRY), str(tmp_path / "command.log")]
    if retry_transport:
        command.append("--retry-transport")
    if retry_quay_blob:
        command.append("--retry-quay-blob")
    if rate_limit_marker is not None:
        command.extend(["--rate-limit-marker", str(rate_limit_marker)])
    if rate_limit_scope is not None:
        command.extend(["--rate-limit-scope", rate_limit_scope])
    command.extend(
        [
            "bash",
            "-c",
            FAKE_COMMAND,
            "fake-command",
            str(count_file),
            str(failures),
            message,
            str(failure_status),
        ]
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed, int(count_file.read_text(encoding="utf-8").strip())


def test_registry_rate_limit_retries_until_the_command_succeeds(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "rate-limit.marker"
    completed, attempts = _run_retry(
        tmp_path,
        failures=2,
        message="toomanyrequests: retry-after: 100ms",
        rate_limit_marker=marker,
    )

    assert completed.returncode == 0
    assert attempts == 3
    assert completed.stderr.count("rate-limited; retrying") == 2
    assert not marker.exists()


def test_registry_connection_reset_retries_until_the_command_succeeds(
    tmp_path: Path,
) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=2,
        message=(
            "read tcp 10.1.1.185:45928->185.199.108.154:443: "
            "read: connection reset by peer"
        ),
        retry_transport=True,
    )

    assert completed.returncode == 0
    assert attempts == 3
    assert completed.stderr.count("transient transport error; retrying") == 2


def test_quay_blob_unauthorized_retries_until_the_command_succeeds(
    tmp_path: Path,
) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=2,
        message=f"#9 ERROR: failed to copy: {QUAY_BLOB_UNAUTHORIZED}",
        retry_quay_blob=True,
    )

    assert completed.returncode == 0
    assert attempts == 3
    assert completed.stderr.count("retrying") == 2


def test_quay_blob_unauthorized_stops_at_the_bounded_attempt_count(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "rate-limit.marker"
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message=QUAY_BLOB_UNAUTHORIZED,
        failure_status=71,
        retry_transport=True,
        retry_quay_blob=True,
        rate_limit_marker=marker,
    )

    assert completed.returncode == 71
    assert attempts == 5
    assert "failed after 5 attempts" in completed.stderr
    assert not marker.exists()


def test_quay_blob_unauthorized_requires_explicit_retry(tmp_path: Path) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message=QUAY_BLOB_UNAUTHORIZED,
        failure_status=42,
        retry_transport=True,
    )

    assert completed.returncode == 42
    assert attempts == 1
    assert "non-retryable error" in completed.stderr


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            QUAY_BLOB_UNAUTHORIZED.replace(
                QUAY_BLOB_URL,
                "https://quay.io/v2/ascend/vllm-ascend/manifests/v0.24.0rc",
            ),
            id="manifest",
        ),
        pytest.param(
            QUAY_BLOB_UNAUTHORIZED.replace(QUAY_BLOB_URL, "https://quay.io/v2/auth"),
            id="token",
        ),
        pytest.param("docker login quay.io: 401 Unauthorized", id="login"),
        pytest.param(
            QUAY_BLOB_UNAUTHORIZED.replace("GET request", "PUT request"),
            id="push",
        ),
        pytest.param(
            QUAY_BLOB_UNAUTHORIZED.replace("quay.io", "ghcr.io"),
            id="other-registry",
        ),
        pytest.param(
            QUAY_BLOB_UNAUTHORIZED.replace("401 Unauthorized", "403 Forbidden"),
            id="forbidden",
        ),
        pytest.param(
            QUAY_BLOB_UNAUTHORIZED.replace("401 Unauthorized", "400 Bad Request"),
            id="bad-request",
        ),
        pytest.param(
            QUAY_BLOB_UNAUTHORIZED.replace(QUAY_BLOB_URL, QUAY_BLOB_URL[:-1]),
            id="invalid-digest",
        ),
    ],
)
def test_quay_blob_retry_does_not_retry_other_registry_failures(
    tmp_path: Path,
    message: str,
) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message=message,
        failure_status=42,
        retry_transport=True,
        retry_quay_blob=True,
    )

    assert completed.returncode == 42
    assert attempts == 1
    assert "non-retryable error" in completed.stderr


@pytest.mark.parametrize("message", ["EOF", "unexpected EOF", "retrying EOF"])
def test_registry_eof_retries_until_the_command_succeeds(
    tmp_path: Path,
    message: str,
) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=1,
        message=message,
        retry_transport=True,
    )

    assert completed.returncode == 0
    assert attempts == 2
    assert "transient transport error; retrying" in completed.stderr


def test_registry_transport_failure_stops_at_the_bounded_attempt_count(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "rate-limit.marker"
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="unexpected EOF",
        failure_status=71,
        retry_transport=True,
        rate_limit_marker=marker,
    )

    assert completed.returncode == 71
    assert attempts == 5
    assert "failed after 5 attempts" in completed.stderr
    assert not marker.exists()


def test_registry_eof_requires_explicit_transport_retry(tmp_path: Path) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="Dockerfile parse error: unexpected EOF",
        failure_status=42,
    )

    assert completed.returncode == 42
    assert attempts == 1
    assert "non-retryable error" in completed.stderr


def test_transport_retry_does_not_retry_permanent_registry_error(
    tmp_path: Path,
) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="manifest unknown",
        failure_status=42,
        retry_transport=True,
    )

    assert completed.returncode == 42
    assert attempts == 1
    assert "non-retryable error" in completed.stderr


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
    marker = tmp_path / "rate-limit.marker"
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="unexpected status code 429",
        rate_limit_marker=marker,
    )

    assert completed.returncode == 75
    assert attempts == 5
    assert "failed after 5 attempts" in completed.stderr
    assert marker.read_text(encoding="utf-8") == "rate-limit-exhausted\n"


def test_registry_marker_requires_the_rate_limit_to_match_its_scope(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "rate-limit.marker"
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="GET https://pypi.org/packages/example: 429 Too Many Requests",
        rate_limit_marker=marker,
        rate_limit_scope="example/ucm-builder",
    )

    assert completed.returncode == 75
    assert attempts == 5
    assert not marker.exists()


def test_registry_marker_accepts_a_rate_limit_in_its_scope(tmp_path: Path) -> None:
    marker = tmp_path / "rate-limit.marker"
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message=(
            "GET https://ghcr.io/v2/example/ucm-builder/blobs/sha256:abc: "
            "429 Too Many Requests"
        ),
        rate_limit_marker=marker,
        rate_limit_scope="example/ucm-builder",
    )

    assert completed.returncode == 75
    assert attempts == 5
    assert marker.read_text(encoding="utf-8") == "rate-limit-exhausted\n"


def test_layer_size_containing_429_is_not_a_rate_limit(tmp_path: Path) -> None:
    completed, attempts = _run_retry(
        tmp_path,
        failures=10,
        message="builder layer: 429B / 1.2MB",
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
