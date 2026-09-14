from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
sys.path.insert(0, str(RELEASE_ROOT))

core = importlib.import_module("ucm_release.core")
wheel = importlib.import_module("ucm_release.wheel")

DIGEST = "sha256:" + "1" * 64
RUNTIME_REQUIREMENTS = ["wrapt==1.17.2"]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _extended_task(
    *, profile: str = "cann900-a2", cpu_arch: str = "amd64"
) -> dict[str, Any]:
    profiles = {
        "cuda130": {
            "accelerator": "cuda",
            "build_arg": "cuda",
            "distribution": "uc-manager-cuda",
            "npu_arch": "na",
            "wheel_platform": "manylinux_2_28",
        },
        "cann900-a2": {
            "accelerator": "ascend",
            "build_arg": "ascend",
            "distribution": "uc-manager-cann-a2",
            "npu_arch": "a2",
            "wheel_platform": "linux",
        },
        "cann900-a3": {
            "accelerator": "ascend",
            "build_arg": "ascend-a3",
            "distribution": "uc-manager-cann-a3",
            "npu_arch": "a3",
            "wheel_platform": "linux",
        },
    }
    selected = profiles[profile]
    declaration = {
        "spec_id": f"{profile}-{cpu_arch}",
        "profile_id": profile,
        "accelerator": selected["accelerator"],
        "accelerator_runtime": ("cuda-13.0" if profile == "cuda130" else "cann-9.0.0"),
        "npu_arch_or_na": selected["npu_arch"],
        "os": "ubuntu-22.04",
        "cpu_arch": cpu_arch,
        "python_version": "3.12",
        "python_abi": "cp312",
        "wheel_version": "0.6.0+" + profile.replace("-", "."),
        "wheel_platform": selected["wheel_platform"],
        "binary_profile_id": "release-" + profile,
        "dist_name": selected["distribution"],
        "validation_targets": [selected["npu_arch"]],
        "required_native": (
            ["ucmtrans"] if profile == "cuda130" else ["ucmtrans", "mooncakestore"]
        ),
        "forbidden_native": (
            ["mooncakestore", "ds3fsstore"] if profile == "cuda130" else ["ds3fsstore"]
        ),
        "allowed_dt_needed": ["libc.so.6"],
        "external_required_dependencies": [],
    }
    dependency_lock = {
        "build_tools": [{"filename": "build.whl", "sha256": DIGEST}],
        "runtime_dependencies": [],
    }
    task: dict[str, Any] = {
        "task_id": "wheel-" + "2" * 64,
        **declaration,
        "declaration_sha256": core.sha256_value(declaration),
        "runner": "ubuntu-24.04",
        "platform": f"linux/{cpu_arch}",
        "builder": {
            "root": {
                "repository": "registry.example/ucm-builder",
                "manifest_digest": DIGEST,
                "config_digest": DIGEST,
            }
        },
        "builder_sha256": DIGEST,
        "build": {
            "docker_target": "wheel",
            "platform_arg": selected["build_arg"],
        },
        "dependency_lock_sha256": core.sha256_value(dependency_lock),
        "dependency_lock": dependency_lock,
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "write_authority": [],
        "build_eligible": True,
        "artifact_name": "ucm-wheel-test",
    }
    task["task_sha256"] = core.sha256_value(task)
    return task


def _extended_authority(task: dict[str, Any]) -> dict[str, Any]:
    root = task["builder"]["root"]
    return {
        "schema_version": 1,
        "kind": "ucm-native-build-authority",
        "task_id": task["task_id"],
        "spec_id": task["spec_id"],
        "profile_id": task["profile_id"],
        "cpu_arch": task["cpu_arch"],
        "platform": task["platform"],
        "build": task["build"],
        "python_version": task["python_version"],
        "python_abi": task["python_abi"],
        "wheel_version": task["wheel_version"],
        "wheel_platform": task["wheel_platform"],
        "source_sha": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "source_archive_sha256": DIGEST,
        "source_date_epoch": 1_700_000_000,
        "task_sha256": task["task_sha256"],
        "builder_coordinate": f"{root['repository']}@{root['manifest_digest']}",
        "builder_config_digest": root["config_digest"],
        "dependency_lock_sha256": task["dependency_lock_sha256"],
        "tool_wheels": {"build.whl": DIGEST},
        "required_native": task["required_native"],
        "forbidden_native": task["forbidden_native"],
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "build_context_sha256": DIGEST,
    }


def _write(path: Path, value: object) -> None:
    path.write_bytes(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )


def _production_task(*, stage: str = "rc") -> dict[str, Any]:
    task: dict[str, Any] = {
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
        "stage": stage,
        "wheel_version": "0.6.0rc1",
        "source_sha": "3" * 40,
        "source_identity_sha256": "4" * 64,
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
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "write_authority": [],
    }
    task["sha256"] = hashlib.sha256(
        json.dumps(
            task, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    return task


def _production_authority(task: dict[str, Any]) -> dict[str, Any]:
    builder = task["builder"]
    return {
        "schema_version": 2,
        "kind": "ucm-native-build-authority",
        "spec_id": task["spec_id"],
        "profile_id": task["profile_id"],
        "distribution": task["distribution"],
        "base_version": task["base_version"],
        "stage": task["stage"],
        "cpu_arch": task["cpu_arch"],
        "platform": task["platform"],
        "wheel_version": task["wheel_version"],
        "source_sha": task["source_sha"],
        "source_tree": "5" * 40,
        "source_archive_sha256": DIGEST,
        "source_date_epoch": 1_700_000_000,
        "task_sha256": "sha256:" + task["sha256"],
        "builder_coordinate": (f"{builder['repository']}@{builder['manifest_digest']}"),
        "builder_config_digest": builder["config_digest"],
        "dependency_lock_sha256": task["dependency_lock_sha256"],
        "tool_wheels": {f"tool-{index}.whl": DIGEST for index in range(7)},
        "required_native": task["required_native"],
        "forbidden_native": task["forbidden_native"],
        "build_context_sha256": DIGEST,
    }


def test_extended_v1_projection_writes_the_exact_canonical_wrapper(
    tmp_path: Path,
) -> None:
    task = _extended_task()
    authority = _extended_authority(task)
    task_path = tmp_path / "task.json"
    authority_path = tmp_path / "authority.json"
    output = tmp_path / "wheel-build.json"
    _write(task_path, task)
    _write(authority_path, authority)

    result = wheel.build_wheel_config(task_path, authority_path, output)

    expected = {
        "authority": authority,
        "distribution": "uc-manager-cann-a2",
        "kind": "ucm-wheel-build-config",
        "platform": "ascend",
        "python": {"abi": "cp312", "version": "3.12"},
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "schema_version": 1,
    }
    expected_bytes = (
        json.dumps(
            expected, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )
    assert result == expected
    assert output.read_bytes() == expected_bytes


def test_production_v2_projection_has_the_same_exact_top_level_contract(
    tmp_path: Path,
) -> None:
    task = _production_task()
    authority = _production_authority(task)
    task_path = tmp_path / "task.json"
    authority_path = tmp_path / "authority.json"
    output = tmp_path / "wheel-build.json"
    _write(task_path, task)
    _write(authority_path, authority)

    result = wheel.build_wheel_config(task_path, authority_path, output)

    assert result == {
        "authority": authority,
        "distribution": "uc-manager-cann-a2",
        "kind": "ucm-wheel-build-config",
        "platform": "ascend",
        "python": {"abi": "cp312", "version": "3.12"},
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "schema_version": 1,
    }
    assert (
        output.read_bytes()
        == json.dumps(
            result, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wheel_platform", "evil"),
        ("build_platform", "evil"),
        ("runtime_requirements", []),
    ],
)
def test_production_projection_names_profile_field_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    task = _production_task()
    task[field] = value
    payload = {key: item for key, item in task.items() if key != "sha256"}
    task["sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    authority = _production_authority(task)
    task_path = tmp_path / "task.json"
    authority_path = tmp_path / "authority.json"
    _write(task_path, task)
    _write(authority_path, authority)

    with pytest.raises(ValueError, match=field):
        wheel.build_wheel_config(
            task_path, authority_path, tmp_path / "wheel-build.json"
        )


@pytest.mark.parametrize(
    ("mutation", "rehash", "message"),
    [
        (lambda task: task.update(sha256="9" * 64), False, "hash"),
        (lambda task: task.update(cpu_arch="arm64"), True, "cpu_arch/spec/platform"),
        (
            lambda task: task.update(spec_id="cann900-a2-arm64"),
            True,
            "cpu_arch/spec/platform",
        ),
        (
            lambda task: task.update(platform="linux/arm64"),
            True,
            "cpu_arch/spec/platform",
        ),
        (
            lambda task: task.update(stage="stable"),
            True,
            "stage/version",
        ),
        (
            lambda task: task.update(source_sha="9" * 39),
            True,
            "source identity",
        ),
        (
            lambda task: task.update(source_identity_sha256="9" * 63),
            True,
            "source identity",
        ),
        (
            lambda task: task.update(dependency_lock_sha256="sha256:" + "9" * 63),
            True,
            "dependency_lock",
        ),
        (
            lambda task: task.update(write_authority=["publish"]),
            True,
            "write authority",
        ),
    ],
)
def test_production_task_validation_names_mutation_family(
    tmp_path: Path, mutation: object, rehash: bool, message: str
) -> None:
    original = _production_task()
    task = json.loads(json.dumps(original))
    mutation(task)
    if rehash:
        payload = {key: item for key, item in task.items() if key != "sha256"}
        task["sha256"] = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
    task_path = tmp_path / "task.json"
    authority_path = tmp_path / "authority.json"
    _write(task_path, task)
    _write(authority_path, _production_authority(original))

    with pytest.raises(ValueError, match=message):
        wheel.build_wheel_config(
            task_path, authority_path, tmp_path / "wheel-build.json"
        )


def test_load_accepts_only_the_exact_canonical_wrapper(tmp_path: Path) -> None:
    task = _extended_task()
    authority = _extended_authority(task)
    config = {
        "authority": authority,
        "distribution": "uc-manager-cann-a2",
        "kind": "ucm-wheel-build-config",
        "platform": "ascend",
        "python": {"abi": "cp312", "version": "3.12"},
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "schema_version": 1,
    }
    path = tmp_path / "wheel-build.json"
    _write(path, config)

    assert wheel.load_wheel_build_config(path) == config


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(task_id="wheel-" + "9" * 64),
        lambda value: value.update(spec_id="cann900-a2-arm64"),
        lambda value: value.update(profile_id="cann900-a3"),
        lambda value: value.update(cpu_arch="arm64", platform="linux/arm64"),
        lambda value: value.update(platform="linux/arm64"),
        lambda value: value.update(
            build={"docker_target": "wheel", "platform_arg": "ascend-a3"}
        ),
        lambda value: value.update(python_version="3.11"),
        lambda value: value.update(python_abi="cp311"),
        lambda value: value.update(wheel_version="0.6.0+cann900.a3"),
        lambda value: value.update(wheel_platform="manylinux_2_28"),
        lambda value: value.update(source_sha="9" * 40),
        lambda value: value.update(source_tree="9" * 40),
        lambda value: value.update(task_sha256="sha256:" + "9" * 64),
        lambda value: value.update(
            builder_coordinate="registry.example/other@" + DIGEST
        ),
        lambda value: value.update(builder_config_digest="sha256:" + "9" * 64),
        lambda value: value.update(dependency_lock_sha256="sha256:" + "9" * 64),
        lambda value: value.update(tool_wheels={"other.whl": DIGEST}),
        lambda value: value.update(required_native=["ucmtrans"]),
        lambda value: value.update(forbidden_native=["hash_retrieval_backend"]),
        lambda value: value.update(runtime_requirements=[]),
    ],
)
def test_projection_rejects_every_task_authority_overlap_drift(
    tmp_path: Path, mutation: object
) -> None:
    task = _extended_task()
    authority = _extended_authority(task)
    mutation(authority)
    task_path = tmp_path / "task.json"
    authority_path = tmp_path / "authority.json"
    output = tmp_path / "wheel-build.json"
    _write(task_path, task)
    _write(authority_path, authority)

    with pytest.raises(ValueError):
        wheel.build_wheel_config(task_path, authority_path, output)

    assert not output.exists()


def _valid_config(*, cpu_arch: str = "amd64") -> dict[str, Any]:
    task = _extended_task(cpu_arch=cpu_arch)
    return {
        "authority": _extended_authority(task),
        "distribution": "uc-manager-cann-a2",
        "kind": "ucm-wheel-build-config",
        "platform": "ascend",
        "python": {"abi": "cp312", "version": "3.12"},
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "schema_version": 1,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("distribution"),
        lambda value: value.update(extra=True),
        lambda value: value.update(kind="ucm-wheel-build"),
        lambda value: value.update(schema_version=2),
        lambda value: value.update(distribution="uc-manager-cann-a3"),
        lambda value: value.update(platform="ascend-a3"),
        lambda value: value.update(python={"abi": "cp311", "version": "3.11"}),
        lambda value: value.update(runtime_requirements=[]),
        lambda value: value["authority"].pop("task_id"),
        lambda value: value["authority"].update(unexpected=True),
    ],
)
def test_loader_rejects_missing_extra_and_drifted_fields(
    tmp_path: Path, mutation: object
) -> None:
    config = _valid_config()
    mutation(config)
    path = tmp_path / "wheel-build.json"
    _write(path, config)

    with pytest.raises(ValueError):
        wheel.load_wheel_build_config(path)


@pytest.mark.parametrize(
    "raw",
    [
        lambda value: json.dumps(value, indent=2).encode("utf-8") + b"\n",
        lambda value: json.dumps(value, sort_keys=True).encode("utf-8"),
        lambda value: json.dumps(value, sort_keys=True).encode("utf-8") + b"\n\n",
        lambda value: (
            b'{"authority":{},"authority":{},"distribution":"uc-manager-cann-a2",'
            b'"kind":"ucm-wheel-build-config","platform":"ascend",'
            b'"python":{"abi":"cp312","version":"3.12"},'
            b'"runtime_requirements":["wrapt==1.17.2"],'
            b'"schema_version":1}\n'
        ),
    ],
)
def test_loader_rejects_noncanonical_or_duplicate_json(
    tmp_path: Path, raw: object
) -> None:
    path = tmp_path / "wheel-build.json"
    path.write_bytes(raw(_valid_config()))

    with pytest.raises((ValueError, json.JSONDecodeError)):
        wheel.load_wheel_build_config(path)


def test_prepare_source_changes_only_the_project_name(tmp_path: Path) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    original = (
        b'# name = "uc-manager" fake\n'
        b'[build-system]\nrequires = ["setuptools"]\n\n'
        b'[project] # preserved\nname = "uc-manager" # preserved\n'
        b'description = "uc-manager remains here"\n\n'
        b'[tool.test]\nname = "uc-manager"\n'
    )
    (source_root / "pyproject.toml").write_bytes(original)

    result = wheel.prepare_wheel_source(config_path, source_root)

    expected = original.replace(
        b'name = "uc-manager" # preserved',
        b'name = "uc-manager-cann-a2" # preserved',
        1,
    )
    assert (source_root / "pyproject.toml").read_bytes() == expected
    assert result["distribution"] == "uc-manager-cann-a2"


@pytest.mark.parametrize(
    "project",
    [
        b'[build-system]\nrequires = ["setuptools"]\n',
        b'[project]\nname = "uc-manager"\n[project]\nversion = "1"\n',
        b'[project]\ndescription = "missing"\n',
        b'[project]\nname = "uc-manager"\nname = "uc-manager"\n',
        b'[project]\nname = "UC-Manager"\n',
    ],
)
def test_prepare_source_rejects_malformed_project_authority_without_editing(
    tmp_path: Path, project: bytes
) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    project_path = source_root / "pyproject.toml"
    project_path.write_bytes(project)

    with pytest.raises(ValueError):
        wheel.prepare_wheel_source(config_path, source_root)

    assert project_path.read_bytes() == project


@pytest.mark.parametrize(
    "original",
    [
        b"[project]\n\"name\" = 'uc-manager'\n",
        b'"project"."name" = "uc-manager"\n',
        b'[ "project" ]\n\'name\' = "uc-manager"\n',
        b'project.name = "uc-manager"\n',
        b'project = { name = "uc-manager" }\n',
    ],
)
def test_prepare_source_supports_valid_quoted_dotted_and_inline_project_names(
    tmp_path: Path, original: bytes
) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    project_path = source_root / "pyproject.toml"
    project_path.write_bytes(original)

    wheel.prepare_wheel_source(config_path, source_root)

    assert project_path.read_bytes() == original.replace(
        b"uc-manager", b"uc-manager-cann-a2", 1
    )


@pytest.mark.parametrize("quote", [b'"""', b"'''"])
def test_prepare_source_ignores_fake_assignments_inside_multiline_strings(
    tmp_path: Path, quote: bytes
) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    project_path = source_root / "pyproject.toml"
    original = (
        b"note = "
        + quote
        + b'\n[project]\nname = "uc-manager"\n'
        + quote
        + b'\n[project]\nname = "uc-manager" # real\n'
    )
    project_path.write_bytes(original)

    wheel.prepare_wheel_source(config_path, source_root)

    assert project_path.read_bytes() == original.replace(
        b'name = "uc-manager" # real',
        b'name = "uc-manager-cann-a2" # real',
    )


def test_prepare_source_preserves_crlf_and_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    project_path = source_root / "pyproject.toml"
    original = b'[project]\r\nname = "uc-manager"\r\n'
    project_path.write_bytes(original)
    project_path.chmod(0o640)

    wheel.prepare_wheel_source(config_path, source_root)

    assert project_path.read_bytes() == original.replace(
        b"uc-manager", b"uc-manager-cann-a2"
    )
    assert stat.S_IMODE(project_path.stat().st_mode) == 0o640


def test_prepare_source_is_idempotent_for_the_exact_target(tmp_path: Path) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    project_path = source_root / "pyproject.toml"
    original = (
        b'note = """name = "uc-manager"""\n' b'[project]\nname = "uc-manager-cann-a2"\n'
    )
    project_path.write_bytes(original)
    before_inode = project_path.stat().st_ino

    wheel.prepare_wheel_source(config_path, source_root)

    assert project_path.read_bytes() == original
    assert project_path.stat().st_ino == before_inode


def test_prepare_source_rejects_ambiguous_semantic_candidate_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    project_path = source_root / "pyproject.toml"
    original = b'# uc-manager\n[project]\nname = "uc-manager"\n'
    project_path.write_bytes(original)
    real_loads = wheel.tomllib.loads

    def ambiguous_loads(value: str) -> dict[str, object]:
        parsed = real_loads(value)
        if "uc-manager-cann-a2" in value:
            parsed["project"]["name"] = "uc-manager-cann-a2"
        return parsed

    monkeypatch.setattr(wheel.tomllib, "loads", ambiguous_loads)

    with pytest.raises(ValueError, match="ambiguous"):
        wheel.prepare_wheel_source(config_path, source_root)

    assert project_path.read_bytes() == original


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ucm_release", *arguments],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(RELEASE_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_build_config_writes_exact_stdout_and_output(tmp_path: Path) -> None:
    task = _extended_task()
    task_path = tmp_path / "task.json"
    authority_path = tmp_path / "authority.json"
    output = tmp_path / "out" / "wheel-build.json"
    _write(task_path, task)
    _write(authority_path, _extended_authority(task))

    result = _run_cli(
        "wheel",
        "build-config",
        "--task-file",
        str(task_path),
        "--authority-file",
        str(authority_path),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.encode("utf-8") == output.read_bytes()
    assert wheel.load_wheel_build_config(output)["platform"] == "ascend"


def test_cli_build_config_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    task = _extended_task()
    authority = _extended_authority(task)
    authority["profile_id"] = "cann900-a3"
    task_path = tmp_path / "task.json"
    authority_path = tmp_path / "authority.json"
    output = tmp_path / "wheel-build.json"
    _write(task_path, task)
    _write(authority_path, authority)

    result = _run_cli(
        "wheel",
        "build-config",
        "--task-file",
        str(task_path),
        "--authority-file",
        str(authority_path),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert not output.exists()


def test_build_config_atomic_failure_preserves_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _extended_task()
    task_path = tmp_path / "task.json"
    authority_path = tmp_path / "authority.json"
    output = tmp_path / "wheel-build.json"
    _write(task_path, task)
    _write(authority_path, _extended_authority(task))
    output.write_bytes(b"existing\n")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(wheel.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        wheel.build_wheel_config(task_path, authority_path, output)

    assert output.read_bytes() == b"existing\n"
    assert list(tmp_path.glob(".wheel-build.json.*")) == []


def test_prepare_source_atomic_failure_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    project_path = source_root / "pyproject.toml"
    original = b'[project]\nname = "uc-manager"\n'
    project_path.write_bytes(original)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(wheel.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        wheel.prepare_wheel_source(config_path, source_root)

    assert project_path.read_bytes() == original
    assert list(source_root.glob(".pyproject.toml.*")) == []


def test_cli_prepare_source_reports_canonical_success(tmp_path: Path) -> None:
    config_path = tmp_path / "wheel-build.json"
    _write(config_path, _valid_config())
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "pyproject.toml").write_text(
        '[project]\nname = "uc-manager"\n', encoding="utf-8"
    )

    result = _run_cli(
        "wheel",
        "prepare-source",
        "--build-config",
        str(config_path),
        "--source-root",
        str(source_root),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "distribution": "uc-manager-cann-a2",
        "kind": "ucm-wheel-source-preparation",
        "project_file": "pyproject.toml",
        "schema_version": 1,
    }
