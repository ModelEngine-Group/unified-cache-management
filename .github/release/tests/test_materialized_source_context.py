from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
sys.path.insert(0, str(RELEASE_ROOT))

core = importlib.import_module("ucm_release.core")
wheel = importlib.import_module("ucm_release.wheel")

DIGEST = "sha256:" + "1" * 64
SOURCE_VERSION = "0.6.0.dev13"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir()
    with tarfile.open(archive_path, "r:") as archive:
        archive.extractall(destination, filter="data")


def _expected_materialized_version() -> str:
    source = _git("show", "HEAD:version.ini")
    if "UCM_SUPPORTED_VLLM_VERSIONS=" not in source:
        return f"VLLM_UC_VERSION={SOURCE_VERSION}\n"
    return (
        "\n".join(
            (
                f"VLLM_UC_VERSION={SOURCE_VERSION}"
                if line.startswith("VLLM_UC_VERSION=")
                else line
            )
            for line in source.splitlines()
        )
        + "\n"
    )


def _production_task(source_sha: str) -> dict[str, object]:
    task: dict[str, object] = {
        "kind": "ucm-production-wheel-build-task",
        "schema_version": 1,
        "spec_id": "cann900-a2-amd64",
        "profile_id": "cann900-a2",
        "distribution": "uc-manager-cann-a2",
        "build_platform": "ascend",
        "cpu_arch": "amd64",
        "platform": "linux/amd64",
        "runner": "ubuntu-24.04",
        "python_version": "3.12",
        "python_abi": "cp312",
        "wheel_platform": "linux",
        "base_version": "0.6.0",
        "stage": "draft",
        "wheel_version": SOURCE_VERSION,
        "source_sha": source_sha,
        "source_identity_sha256": "2" * 64,
        "builder": {
            "repository": "registry.example/ucm-builder",
            "tag": "v1",
            "index_digest": DIGEST,
            "manifest_digest": DIGEST,
            "config_digest": DIGEST,
        },
        "runtime": {
            "repository": "registry.example/ucm-runtime",
            "tag": "v1",
            "index_digest": DIGEST,
            "manifest_digest": DIGEST,
            "config_digest": DIGEST,
        },
        "required_native": ["ucmtrans", "mooncakestore"],
        "forbidden_native": ["ds3fsstore"],
        "dependency_lock_sha256": DIGEST,
        "runtime_requirements": ["wrapt==1.17.2"],
        "write_authority": [],
    }
    task["sha256"] = hashlib.sha256(core.canonical_bytes(task)).hexdigest()
    return task


def test_materialized_source_context_is_deterministic_and_keeps_original_identity(
    tmp_path: Path,
) -> None:
    source_sha = _git("rev-parse", "HEAD")
    first = wheel.prepare_source_context(tmp_path / "first", source_sha, SOURCE_VERSION)
    second = wheel.prepare_source_context(
        tmp_path / "second", source_sha, SOURCE_VERSION
    )

    assert first["source_tree"] == _git("rev-parse", "HEAD^{tree}")
    assert first["materialized_tree"] != first["source_tree"]
    assert first["materialized_tree"] == second["materialized_tree"]
    assert (tmp_path / "first/ucm-source.tar").read_bytes() == (
        tmp_path / "second/ucm-source.tar"
    ).read_bytes()
    assert (tmp_path / "first/source-context.json").read_bytes() == (
        tmp_path / "second/source-context.json"
    ).read_bytes()
    assert (ROOT / "version.ini").read_text(encoding="utf-8") != (
        _expected_materialized_version()
    )

    extracted = tmp_path / "extracted"
    _extract(tmp_path / "first/ucm-source.tar", extracted)
    verified = wheel.verify_source_context(
        tmp_path / "first/ucm-source.tar",
        tmp_path / "first/source-context.json",
        extracted,
        tmp_path / "first/source-commit.payload",
        source_sha,
        SOURCE_VERSION,
    )
    assert verified["source_version"] == SOURCE_VERSION
    assert (extracted / "version.ini").read_text(
        encoding="utf-8"
    ) == _expected_materialized_version()


def test_production_authority_binds_materialized_tree_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    source_sha = _git("rev-parse", "HEAD")
    prepared = wheel.prepare_source_context(
        tmp_path / "context", source_sha, SOURCE_VERSION
    )
    manifest = core.load_json(tmp_path / "context/source-context.json")
    task = _production_task(source_sha)
    tools = {f"tool-{index}.whl": DIGEST for index in range(7)}

    authority = wheel.build_production_authority(
        task,
        manifest,
        1_700_000_000,
        tools,
        output=tmp_path / "authority.json",
    )

    assert authority["schema_version"] == 3
    assert authority["source_tree"] == prepared["source_tree"]
    assert authority["materialized_tree"] == prepared["materialized_tree"]
    assert authority["source_version"] == SOURCE_VERSION

    tampered = json.loads(json.dumps(manifest))
    tampered["source_version"] = "0.6.0.dev14"
    with pytest.raises(ValueError, match="source context"):
        wheel.build_production_authority(task, tampered, 1_700_000_000, tools)

    extracted = tmp_path / "tampered-source"
    _extract(tmp_path / "context/ucm-source.tar", extracted)
    (extracted / "version.ini").write_text(
        "VLLM_UC_VERSION=0.6.0.dev14\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        wheel.verify_source_context(
            tmp_path / "context/ucm-source.tar",
            tmp_path / "context/source-context.json",
            extracted,
            tmp_path / "context/source-commit.payload",
            source_sha,
            SOURCE_VERSION,
        )
