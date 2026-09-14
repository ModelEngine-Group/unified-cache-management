"""Loopback registry integration contract for the slim release package.

Only the real registry:2.8.3 push/pull integration test and the minimal
subprocess-environment safety check are retained.  The hundreds of fixture
evidence, JSON-shape, and reconciliation change-detector tests were removed
per the slimming plan -- they exercised the verify.py fixture loop that D1
deletes rather than observable behaviour.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = REPO_ROOT / ".github" / "release"


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _builder_resolver_transport(
    registry, monkeypatch: pytest.MonkeyPatch, responses: list[str]
) -> list[list[str]]:
    calls: list[list[str]] = []
    payloads = iter(responses)
    monkeypatch.setattr(registry, "resolve_pinned_crane", lambda: "crane")

    def run(binary: str, arguments: list[str]):
        assert binary == "crane"
        calls.append(arguments)
        return subprocess.CompletedProcess(
            [binary, *arguments], 0, stdout=next(payloads), stderr=""
        )

    monkeypatch.setattr(registry, "_run_registry_tool", run)
    return calls


def _modules():
    sys.path.insert(0, str(RELEASE_ROOT))
    return (
        importlib.import_module("ucm_release.registry"),
        None,
    )


def test_registry_subprocess_environment_is_minimal_and_keeps_login_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry subprocesses inherit only proxy/transport env, never secrets."""
    registry, _ = _modules()
    values = {
        "HOME": "/runner/home",
        "DOCKER_CONFIG": "/runner/docker-config",
        "SSL_CERT_FILE": "/etc/certs.pem",
        "HTTPS_PROXY": "http://proxy.internal:8080",
        "NO_PROXY": "127.0.0.1",
        "GITHUB_TOKEN": "must-not-leak",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "must-not-leak",
        "PATH": "/attacker/path",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    environment = registry._minimal_registry_environment()

    assert environment == {
        "HOME": "/runner/home",
        "DOCKER_CONFIG": "/runner/docker-config",
        "SSL_CERT_FILE": "/etc/certs.pem",
        "HTTPS_PROXY": "http://proxy.internal:8080",
        "NO_PROXY": "127.0.0.1",
    }

    crane = tmp_path / "crane"
    crane.write_text("not executed", encoding="utf-8")
    invocation: dict[str, object] = {}

    def run(
        arguments: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        invocation.update({"arguments": arguments, **options})
        return subprocess.CompletedProcess(
            arguments, 0, stdout="sha256:" + "1" * 64 + "\n", stderr=""
        )

    monkeypatch.setattr(registry.subprocess, "run", run)
    reference = "docker.io/vllm/vllm-openai:v0.10.2"
    assert registry._crane(str(crane), "digest", reference) == "sha256:" + "1" * 64
    assert invocation["env"] == environment
    assert invocation["arguments"] == [str(crane), "digest", reference]


@pytest.mark.parametrize(
    ("platforms", "message"),
    [
        (
            [{"os": "linux", "architecture": "arm64", "digest": _digest("2")}],
            "no linux/amd64 member",
        ),
        (
            [
                {"os": "linux", "architecture": "amd64", "digest": _digest("2")},
                {"os": "linux", "architecture": "amd64", "digest": _digest("3")},
            ],
            "multiple linux/amd64 members",
        ),
    ],
)
def test_builder_index_requires_exactly_one_requested_architecture(
    monkeypatch: pytest.MonkeyPatch,
    platforms: list[dict[str, str]],
    message: str,
) -> None:
    registry, _ = _modules()
    index = {
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {
                "digest": item["digest"],
                "platform": {"os": item["os"], "architecture": item["architecture"]},
            }
            for item in platforms
        ],
    }
    _builder_resolver_transport(
        registry, monkeypatch, [_digest("1"), json.dumps(index)]
    )

    with pytest.raises(ValueError, match=message):
        registry.resolve_builder_root(
            "ghcr.io/release-org/ucm-builder-vllm",
            "cuda13.0-cp312-manylinux2_28-amd64-r1",
            architecture="amd64",
        )


def test_single_manifest_builder_config_proves_requested_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _modules()
    manifest = {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"digest": _digest("2")},
    }
    calls = _builder_resolver_transport(
        registry,
        monkeypatch,
        [
            _digest("1"),
            json.dumps(manifest),
            json.dumps({"os": "linux", "architecture": "amd64"}),
        ],
    )

    result = registry.resolve_builder_root(
        "ghcr.io/release-org/ucm-builder-vllm",
        "cuda13.0-cp312-manylinux2_28-amd64-r1",
        architecture="amd64",
    )

    immutable_reference = "ghcr.io/release-org/ucm-builder-vllm@" + _digest("1")
    assert calls[-1] == ["config", immutable_reference]
    assert result == {
        "index_digest": _digest("1"),
        "manifest_digest": _digest("1"),
        "config_digest": _digest("2"),
        "operations": [
            {
                "type": "crane-digest",
                "capability": "read",
                "reference": "ghcr.io/release-org/ucm-builder-vllm:cuda13.0-cp312-manylinux2_28-amd64-r1",
            },
            {
                "type": "crane-manifest",
                "capability": "read",
                "reference": immutable_reference,
            },
            {
                "type": "crane-config",
                "capability": "read",
                "reference": immutable_reference,
            },
        ],
    }


@pytest.mark.parametrize(
    "config",
    [
        {"os": "linux", "architecture": "arm64"},
        {"os": "windows", "architecture": "amd64"},
    ],
)
def test_single_manifest_builder_rejects_config_platform_mismatch(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, str]
) -> None:
    registry, _ = _modules()
    manifest = {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"digest": _digest("2")},
    }
    _builder_resolver_transport(
        registry,
        monkeypatch,
        [_digest("1"), json.dumps(manifest), json.dumps(config)],
    )

    with pytest.raises(ValueError, match="does not match requested linux/amd64"):
        registry.resolve_builder_root(
            "ghcr.io/release-org/ucm-builder-vllm",
            "cuda13.0-cp312-manylinux2_28-amd64-r1",
            architecture="amd64",
        )


def test_single_manifest_builder_rejects_malformed_config_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _modules()
    manifest = {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"digest": _digest("2")},
    }
    _builder_resolver_transport(
        registry, monkeypatch, [_digest("1"), json.dumps(manifest), "not-json"]
    )

    with pytest.raises(json.JSONDecodeError):
        registry.resolve_builder_root(
            "ghcr.io/release-org/ucm-builder-vllm",
            "cuda13.0-cp312-manylinux2_28-amd64-r1",
            architecture="amd64",
        )
