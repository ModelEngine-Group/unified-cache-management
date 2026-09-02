"""Inspect a wheel and bind its bytes and metadata to one declared wheel spec."""

from __future__ import annotations

import ast
import base64
import copy
import csv
import email.parser
import functools
import hashlib
import io
import json
import os
import platform as host_platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from . import version_config
from .core import (
    DEFAULT_RELEASE,
    DEFAULT_SCHEMA_DIR,
    REPO_ROOT,
    canonical_bytes,
    cpu_toolchain_authority,
    host_cpu_toolchain_authority,
    load_catalog,
    python_runtime_requirements,
    sha256_value,
)

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
FIXTURE_MARKER = "ucm/_fixture_build.py"
COMPONENT_MANIFEST = "ucm/ucm-native-components.json"
AUTHORITY_KIND = "ucm-native-build-authority"
CLOSURE_KIND = "ucm-linux-dependency-closure"
SOURCE_CONTEXT_KIND = "ucm-canonical-source-context"
SOURCE_CONTEXT_PREFIX = b"ucm-build-context-v3\0"
LEGACY_UCM_VERSION_KEY = "VLLM_UC_VERSION"
HOST_PATH_MARKERS = (
    b"/Users/",
    b"/home/runner/",
    b"/private/var/",
    b"/var/folders/",
    b"/tmp/",
)
NATIVE_MEMBER_DIRECTORIES = {
    "ucmtrans": "ucm/shared/trans",
    "metrics": "ucm/shared/metrics",
    "ucmmetrics": "ucm/shared/metrics",
    "ucmlogger": "ucm/shared/infra",
    "ucmnfsstore": "ucm/store/nfsstore",
    "ucmpcstore": "ucm/store/pcstore",
    "posixstore": "ucm/store/posix",
    "compressor": "ucm/store/compress",
    "cachestore": "ucm/store/cache",
    "emptystore": "ucm/store/empty",
    "fakestore": "ucm/store/fake",
    "ucmpipelinestore": "ucm/store/pipeline",
    "mooncakestore": "ucm/store/mooncakestore",
    "ds3fsstore": "ucm/store/ds3fs",
}
SHARED_LIBRARY_COMPONENTS = {
    "metrics",
    "posixstore",
    "compressor",
    "cachestore",
    "emptystore",
    "fakestore",
    "mooncakestore",
    "ds3fsstore",
}
EXTERNAL_REQUIRED_FIELDS = {
    "dependency",
    "provider",
    "expected_mount_root",
    "relation",
    "required_at",
}
WHEEL_BUILD_CONFIG_KIND = "ucm-wheel-build-config"
WHEEL_BUILD_CONFIG_FIELDS = {
    "authority",
    "distribution",
    "kind",
    "platform",
    "python",
    "runtime_requirements",
    "schema_version",
}
EXTENDED_V1_AUTHORITY_FIELDS = {
    "schema_version",
    "kind",
    "task_id",
    "spec_id",
    "profile_id",
    "cpu_arch",
    "platform",
    "build",
    "python_version",
    "python_abi",
    "wheel_version",
    "wheel_platform",
    "source_sha",
    "source_tree",
    "source_archive_sha256",
    "source_date_epoch",
    "task_sha256",
    "builder_coordinate",
    "builder_config_digest",
    "dependency_lock_sha256",
    "tool_wheels",
    "required_native",
    "forbidden_native",
    "runtime_requirements",
    "build_context_sha256",
}
EXTENDED_V4_AUTHORITY_FIELDS = EXTENDED_V1_AUTHORITY_FIELDS | {
    "materialized_tree",
    "source_version",
}
PRODUCTION_V2_AUTHORITY_FIELDS = {
    "schema_version",
    "kind",
    "spec_id",
    "profile_id",
    "distribution",
    "base_version",
    "stage",
    "cpu_arch",
    "platform",
    "wheel_version",
    "source_sha",
    "source_tree",
    "source_archive_sha256",
    "source_date_epoch",
    "task_sha256",
    "builder_coordinate",
    "builder_config_digest",
    "dependency_lock_sha256",
    "tool_wheels",
    "required_native",
    "forbidden_native",
    "build_context_sha256",
}
PRODUCTION_V3_AUTHORITY_FIELDS = PRODUCTION_V2_AUTHORITY_FIELDS | {
    "materialized_tree",
    "source_version",
}
WHEEL_BUILD_PYTHON = {"abi": "cp312", "version": "3.12"}
WHEEL_BUILD_RUNTIME_REQUIREMENTS = ["wrapt==1.17.2"]
WHEEL_BUILD_PROFILES = {
    "cuda130": {
        "build_platform": "cuda",
        "distribution": "uc-manager-cuda",
        "python": WHEEL_BUILD_PYTHON,
        "runtime_requirements": WHEEL_BUILD_RUNTIME_REQUIREMENTS,
        "wheel_platform": "manylinux_2_28",
    },
    "cann900-a2": {
        "build_platform": "ascend",
        "distribution": "uc-manager-cann-a2",
        "python": WHEEL_BUILD_PYTHON,
        "runtime_requirements": WHEEL_BUILD_RUNTIME_REQUIREMENTS,
        "wheel_platform": "linux",
    },
    "cann900-a3": {
        "build_platform": "ascend-a3",
        "distribution": "uc-manager-cann-a3",
        "python": WHEEL_BUILD_PYTHON,
        "runtime_requirements": WHEEL_BUILD_RUNTIME_REQUIREMENTS,
        "wheel_platform": "linux",
    },
}
PRODUCTION_WHEEL_TASK_FIELDS = {
    "kind",
    "schema_version",
    "spec_id",
    "profile_id",
    "distribution",
    "build_platform",
    "cpu_arch",
    "platform",
    "runner",
    "python_version",
    "python_abi",
    "wheel_platform",
    "base_version",
    "stage",
    "wheel_version",
    "source_sha",
    "source_identity_sha256",
    "builder",
    "runtime",
    "required_native",
    "forbidden_native",
    "dependency_lock_sha256",
    "runtime_requirements",
    "write_authority",
    "sha256",
}


_WHEEL_DECLARATION_FIELDS = (
    "spec_id",
    "profile_id",
    "accelerator",
    "accelerator_runtime",
    "npu_arch_or_na",
    "os",
    "cpu_arch",
    "python_version",
    "python_abi",
    "wheel_version",
    "wheel_platform",
    "binary_profile_id",
    "dist_name",
    "validation_targets",
    "required_native",
    "forbidden_native",
    "allowed_dt_needed",
    "external_required_dependencies",
)


def _validate_wheel_task(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("selected wheel task must be an object")
    task = copy.deepcopy(value)
    task_payload = {key: item for key, item in task.items() if key != "task_sha256"}
    if re.fullmatch(
        r"wheel-[0-9a-f]{64}", str(task.get("task_id"))
    ) is None or task.get("task_sha256") != sha256_value(task_payload):
        raise ValueError("wheel task hash mismatch")
    missing = [field for field in _WHEEL_DECLARATION_FIELDS if field not in task]
    if missing:
        raise ValueError(f"wheel task declaration fields are missing: {missing}")
    declaration = {
        field: copy.deepcopy(task[field]) for field in _WHEEL_DECLARATION_FIELDS
    }
    if task.get("declaration_sha256") != sha256_value(declaration):
        raise ValueError("wheel task declaration hash mismatch")
    dependency_lock = task.get("dependency_lock")
    if not isinstance(dependency_lock, dict) or task.get(
        "dependency_lock_sha256"
    ) != sha256_value(dependency_lock):
        raise ValueError("wheel task dependency authority is invalid")
    return task


def _selected_wheel_task(
    spec_id: str,
    *,
    task: dict[str, Any] | None = None,
    task_path: Path,
) -> dict[str, Any]:
    if (task is None) == (task_path is None):
        raise ValueError(
            "real wheel operation requires exactly one selected wheel task"
        )
    selected = (
        _validate_wheel_task(task)
        if task is not None
        else _validate_wheel_task(_canonical_record(task_path, "selected wheel task"))
    )
    if selected["spec_id"] != spec_id:
        raise ValueError("selected wheel task spec differs from requested spec")
    return selected


def _wheel_spec_from_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        field: copy.deepcopy(task[field])
        for field in (*_WHEEL_DECLARATION_FIELDS, "declaration_sha256")
    } | {"build_eligible": task["build_eligible"]}


def _fixture_wheel_specs(release: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract wheel spec declarations from the catalog for fixture/inspection paths."""
    specs: list[dict[str, Any]] = []
    for item in release.get("wheel_profiles", []):
        spec = copy.deepcopy(item)
        if "spec_id" not in spec:
            spec["spec_id"] = spec.get("id", "")
        specs.append(spec)
    return specs


def check_build_environment(
    task: dict[str, Any], *, python_executable: Path
) -> dict[str, Any]:
    """Execute the typed immutable-builder checks declared by one wheel task."""
    if not isinstance(task, dict):
        raise ValueError("wheel task must be an object")
    task_id = task.get("task_id")
    if re.fullmatch(r"wheel-[0-9a-f]{64}", str(task_id)) is None:
        raise ValueError("wheel task ID is invalid")
    host_authority = host_cpu_toolchain_authority(host_platform.machine())
    if task.get("cpu_arch") != host_authority.cpu_arch:
        raise ValueError("wheel task CPU architecture differs from builder host")
    executable = Path(python_executable)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("declared Python executable is not executable")
    completed = subprocess.run(
        [
            str(executable),
            "-c",
            (
                "import json,sys,sysconfig;"
                "print(json.dumps({'version':f'{sys.version_info.major}."
                "{sys.version_info.minor}','abi':f'cp{sys.version_info.major}"
                "{sys.version_info.minor}','soabi':sysconfig.get_config_var('SOABI')}))"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("declared Python executable cannot report its identity")
    try:
        python_identity = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("declared Python identity is malformed") from error
    if python_identity.get("version") != task.get(
        "python_version"
    ) or python_identity.get("abi") != task.get("python_abi"):
        raise ValueError("declared Python executable differs from wheel task")
    builder = task.get("builder")
    checks = builder.get("checks") if isinstance(builder, dict) else None
    if not isinstance(checks, list) or not checks:
        raise ValueError("wheel task builder checks are missing")
    evidence: list[dict[str, str]] = []
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("wheel task builder check is malformed")
        kind = check.get("kind")
        target = ""
        if kind == "python":
            if (
                check.get("version") != python_identity["version"]
                or check.get("abi") != python_identity["abi"]
            ):
                raise ValueError("Python builder check differs from executable")
            target = str(executable)
        elif kind == "python-soabi":
            soabi = python_identity.get("soabi")
            if not isinstance(soabi, str) or not soabi.startswith(
                str(check.get("prefix", "")) + "-"
            ):
                raise ValueError("Python SOABI builder check failed")
            target = soabi
        elif kind in {"command", "command-version"}:
            command = shutil.which(str(check.get("name", "")))
            if command is None:
                raise ValueError(f"required command is missing: {check.get('name')}")
            target = command
            if kind == "command-version":
                version = subprocess.run(
                    [command, *check.get("arguments", [])],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                output = version.stdout + version.stderr
                if version.returncode != 0 or str(check.get("contains")) not in output:
                    raise ValueError(
                        f"required command version check failed: {check.get('name')}"
                    )
        elif kind in {
            "file",
            "directory",
            "library-cache",
            "shared-library-dependencies",
        }:
            path = Path(str(check.get("path", "")))
            present = (
                path.is_dir()
                if kind in {"directory", "library-cache"}
                else path.is_file()
            )
            if not present:
                raise ValueError(f"required {kind} is missing: {path}")
            if kind == "library-cache":
                ldconfig = shutil.which("ldconfig")
                if ldconfig is None:
                    raise ValueError("required command is missing: ldconfig")
                refreshed = subprocess.run(
                    [ldconfig, str(path)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if refreshed.returncode != 0:
                    raise ValueError(f"shared library cache refresh failed: {path}")
            elif kind == "shared-library-dependencies":
                preflight_dependencies(path, str(task.get("spec_id", "")), task=task)
            target = str(path)
        else:
            raise ValueError(f"unsupported builder check kind {kind!r}")
        evidence.append({"kind": str(kind), "target": target, "status": "passed"})
    return {
        "schema_version": 1,
        "kind": "ucm-builder-environment-check",
        "task_id": task_id,
        "task_sha256": task.get("task_sha256"),
        "cpu_arch": host_authority.cpu_arch,
        "python_executable": str(executable),
        "checks": evidence,
        "status": "passed",
    }


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, value: object) -> None:
    _atomic_write_bytes(path, canonical_bytes(value) + b"\n")


def _atomic_write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if mode is not None:
            os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _external_required_by_dependency(
    declarations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        if (
            not isinstance(declaration, dict)
            or set(declaration) != EXTERNAL_REQUIRED_FIELDS
        ):
            raise ValueError("external-required dependency declaration is invalid")
        dependency = declaration.get("dependency")
        if not isinstance(dependency, str) or dependency in result:
            raise ValueError("external-required dependency declarations are not unique")
        if (
            declaration.get("relation") != "transitive"
            or declaration.get("required_at") != "device-runtime"
            or not str(declaration.get("expected_mount_root", "")).startswith("/")
        ):
            raise ValueError("external-required dependency declaration is invalid")
        result[dependency] = declaration
    return result


def _external_required_resolution(declaration: dict[str, Any]) -> dict[str, Any]:
    return {
        **declaration,
        "direct": False,
        "kind": "external-required",
    }


def build_fixture_wheel(
    output_dir: Path,
    source_sha: str,
    profile_id: str,
    *,
    release_path: Path = DEFAULT_RELEASE,
    schema_dir: Path = DEFAULT_SCHEMA_DIR,
) -> dict[str, Any]:
    """Build one deterministic, source-bound wheel for the fork candidate lane."""
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("fixture wheel source SHA must be a full lowercase Git commit")
    release = load_catalog(release_path, schema_dir)
    specs = {item["spec_id"]: item for item in _fixture_wheel_specs(release)}
    if profile_id not in specs:
        raise ValueError(f"unknown fixture wheel profile: {profile_id}")
    spec = specs[profile_id]
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("fixture wheel output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    version = release["ucm_version"]
    platform = cpu_toolchain_authority(spec["cpu_arch"]).wheel_arch
    tag = f"{spec['python_abi']}-{spec['python_abi']}-linux_{platform}"
    filename = f"{_dist_filename_component(spec['dist_name'])}-{version}-{tag}.whl"  # fmt: skip  # noqa: E501
    dist_info = f"{_dist_filename_component(spec['dist_name'])}-{version}.dist-info"  # fmt: skip  # noqa: E501
    members = {
        "ucm/__init__.py": f"__version__ = {version!r}\n",
        "ucm/_fixture_build.py": (
            f"SOURCE_SHA = {source_sha!r}\nPROFILE_ID = {profile_id!r}\n"
        ),
        f"{dist_info}/METADATA": "\n".join(
            [
                "Metadata-Version: 2.1",
                f"Name: {spec['dist_name']}",
                f"Version: {version}",
                *(
                    f"Requires-Dist: {requirement}"
                    for requirement in python_runtime_requirements(release)
                ),
                "",
            ]
        ),
        f"{dist_info}/WHEEL": "\n".join(
            [
                "Wheel-Version: 1.0",
                "Generator: ucm-fork-fixture-only",
                "Root-Is-Purelib: false",
                f"Tag: {tag}",
                "",
            ]
        ),
    }
    record_rows: list[list[str]] = []
    for name, content in members.items():
        raw = content.encode("utf-8")
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
            .decode("ascii")
            .rstrip("=")
        )
        record_rows.append([name, f"sha256={digest}", str(len(raw))])
    record_name = f"{dist_info}/RECORD"
    record_rows.append([record_name, "", ""])
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(record_rows)
    members[record_name] = record_buffer.getvalue()

    wheel_path = output_dir / filename
    with zipfile.ZipFile(wheel_path, "w") as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name])
    wheel_sha256 = _sha256(wheel_path)
    inspection = inspect_wheel(
        wheel_path,
        profile_id,
        wheel_sha256,
        "fixture",
        release_path=release_path,
        schema_dir=schema_dir,
    )
    inspection_path = output_dir / "wheel-inspection.json"
    _write_canonical(inspection_path, inspection)
    inspection_sha256 = _sha256(inspection_path)
    build_record = {
        "schema_version": 1,
        "kind": "ucm-fixture-wheel-build",
        "fixture_only": True,
        "publication_status": "unpublished",
        "publication_eligible": False,
        "source_sha": source_sha,
        "profile_id": profile_id,
        "wheel_sha256": wheel_sha256,
        "inspection_sha256": inspection_sha256,
    }
    _write_canonical(output_dir / "fixture-build.json", build_record)
    (output_dir / "wheel.sha256").write_text(wheel_sha256 + "\n", encoding="utf-8")
    return {
        "wheel_path": str(wheel_path),
        "wheel_sha256": wheel_sha256,
        "inspection_sha256": inspection_sha256,
        "inspection": inspection,
        "build_record": build_record,
    }


def _safe_wheel_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and "\\" not in name
        and all(part not in {"", ".", ".."} for part in name.split("/"))
        and path.as_posix() == name
    )


def _verify_record(archive: zipfile.ZipFile, record_name: str) -> None:
    names = [item.filename for item in archive.infolist() if not item.is_dir()]
    if len(names) != len(set(names)) or any(
        not _safe_wheel_name(name) for name in names
    ):
        raise ValueError("wheel contains duplicate, unsafe, or noncanonical members")
    rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    if any(len(row) != 3 for row in rows):
        raise ValueError("wheel RECORD must contain exactly three columns")
    by_name = {row[0]: row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != set(names):
        raise ValueError("wheel RECORD does not exactly cover archive files")
    for name in names:
        _, encoded_digest, encoded_size = by_name[name]
        if name == record_name:
            if encoded_digest or encoded_size:
                raise ValueError(
                    "wheel RECORD self-entry must have empty digest and size"
                )
            continue
        if not encoded_digest.startswith("sha256="):
            raise ValueError(f"wheel RECORD entry lacks SHA256: {name}")
        encoded = encoded_digest.partition("=")[2]
        try:
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except ValueError as error:
            raise ValueError(f"wheel RECORD has invalid SHA256: {name}") from error
        content = archive.read(name)
        if hashlib.sha256(content).digest() != expected:
            raise ValueError(f"wheel RECORD SHA256 mismatch: {name}")
        if not encoded_size.isdecimal() or int(encoded_size) != len(content):
            raise ValueError(f"wheel RECORD size mismatch: {name}")


def _unique_json(data: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_record(path: Path, label: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = _unique_json(raw, label)
    if raw != canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} is noncanonical")
    return value


def _validate_native_lists(authority: dict[str, Any]) -> None:
    required = authority.get("required_native")
    forbidden = authority.get("forbidden_native")
    if (
        not isinstance(required, list)
        or not required
        or not isinstance(forbidden, list)
        or not forbidden
        or any(
            not isinstance(item, str) or re.fullmatch(r"[a-z0-9_]+", item) is None
            for item in [*required, *forbidden]
        )
        or len(required) != len(set(required))
        or len(forbidden) != len(set(forbidden))
        or set(required) & set(forbidden)
    ):
        raise ValueError("build authority native target lists are invalid")


def _validate_wheel_build_authority(authority: object) -> dict[str, Any]:
    if not isinstance(authority, dict):
        raise ValueError("wheel build authority must be an object")
    schema_version = authority.get("schema_version")
    expected_fields = {
        1: EXTENDED_V1_AUTHORITY_FIELDS,
        2: PRODUCTION_V2_AUTHORITY_FIELDS,
        3: PRODUCTION_V3_AUTHORITY_FIELDS,
        4: EXTENDED_V4_AUTHORITY_FIELDS,
    }.get(schema_version)
    if expected_fields is None or set(authority) != expected_fields:
        raise ValueError("wheel build authority schema is unsupported")
    cpu_arch = authority.get("cpu_arch")
    if (
        authority.get("kind") != AUTHORITY_KIND
        or cpu_arch not in {"amd64", "arm64"}
        or authority.get("platform") != f"linux/{cpu_arch}"
        or re.fullmatch(r"[0-9a-f]{40}", str(authority.get("source_sha"))) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(authority.get("source_tree"))) is None
        or (
            schema_version in {3, 4}
            and re.fullmatch(r"[0-9a-f]{40}", str(authority.get("materialized_tree")))
            is None
        )
        or any(
            DIGEST_RE.fullmatch(str(authority.get(field))) is None
            for field in (
                "source_archive_sha256",
                "task_sha256",
                "builder_config_digest",
                "dependency_lock_sha256",
                "build_context_sha256",
            )
        )
        or re.fullmatch(
            r"[^@ ]+@sha256:[0-9a-f]{64}",
            str(authority.get("builder_coordinate")),
        )
        is None
        or type(authority.get("source_date_epoch")) is not int
        or not 315532800 <= authority["source_date_epoch"] <= 4354819199
        or not isinstance(authority.get("wheel_version"), str)
        or not authority["wheel_version"]
    ):
        raise ValueError("wheel build authority identity is invalid")
    tools = authority.get("tool_wheels")
    if (
        not isinstance(tools, dict)
        or not tools
        or (schema_version in {2, 3} and len(tools) != 7)
        or any(
            not isinstance(name, str)
            or not name.endswith(".whl")
            or DIGEST_RE.fullmatch(str(digest)) is None
            for name, digest in tools.items()
        )
    ):
        raise ValueError("wheel build authority tool wheels are invalid")
    _validate_native_lists(authority)
    if schema_version in {1, 4}:
        build = authority.get("build")
        profile_settings = WHEEL_BUILD_PROFILES.get(authority.get("profile_id"))
        if (
            re.fullmatch(r"wheel-[0-9a-f]{64}", str(authority.get("task_id"))) is None
            or profile_settings is None
            or authority.get("spec_id") != f"{authority.get('profile_id')}-{cpu_arch}"
            or not isinstance(build, dict)
            or set(build) != {"docker_target", "platform_arg"}
            or not all(isinstance(value, str) and value for value in build.values())
            or build.get("platform_arg") != profile_settings["build_platform"]
            or authority.get("python_version") != profile_settings["python"]["version"]
            or authority.get("python_abi") != profile_settings["python"]["abi"]
            or authority.get("wheel_platform") != profile_settings["wheel_platform"]
            or authority.get("runtime_requirements")
            != profile_settings["runtime_requirements"]
            or (
                schema_version == 4
                and authority.get("source_version") != authority["wheel_version"]
            )
        ):
            raise ValueError("extended schema-v1 build authority is invalid")
    else:
        profile = authority.get("profile_id")
        expected = WHEEL_BUILD_PROFILES.get(profile)
        base_version = str(authority.get("base_version"))
        stage = authority.get("stage")
        stage_patterns = {
            "draft": re.escape(base_version) + r"\.dev[1-9][0-9]*",
            "rc": re.escape(base_version) + r"rc[1-9][0-9]*",
            "stable": re.escape(base_version),
            "hotfix": re.escape(base_version),
        }
        if (
            expected is None
            or authority.get("distribution") != expected["distribution"]
            or authority.get("spec_id") != f"{profile}-{cpu_arch}"
            or re.fullmatch(
                r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)",
                base_version,
            )
            is None
            or stage not in stage_patterns
            or re.fullmatch(stage_patterns[stage], str(authority["wheel_version"]))
            is None
            or (
                schema_version == 3
                and authority.get("source_version") != authority["wheel_version"]
            )
        ):
            raise ValueError("production build authority is invalid")
    return authority


def _validate_wheel_build_config_value(config: object) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != WHEEL_BUILD_CONFIG_FIELDS:
        raise ValueError("wheel build config fields are not exact")
    if (
        config.get("kind") != WHEEL_BUILD_CONFIG_KIND
        or config.get("schema_version") != 1
    ):
        raise ValueError("wheel build config contract is invalid")
    authority = _validate_wheel_build_authority(config.get("authority"))
    profile = authority["profile_id"]
    expected = WHEEL_BUILD_PROFILES.get(profile)
    if expected is None:
        raise ValueError("wheel build config profile is invalid")
    distribution = expected["distribution"]
    platform_arg = expected["build_platform"]
    if (
        config.get("python") != expected["python"]
        or config.get("runtime_requirements") != expected["runtime_requirements"]
    ):
        raise ValueError("wheel build config profile settings are invalid")
    if authority["schema_version"] in {1, 4}:
        platform_arg = authority["build"]["platform_arg"]
        if (
            authority["python_version"] != config["python"]["version"]
            or authority["python_abi"] != config["python"]["abi"]
            or authority["runtime_requirements"] != config["runtime_requirements"]
        ):
            raise ValueError("wheel build config differs from schema-v1 authority")
    if (
        config.get("distribution") != distribution
        or config.get("platform") != platform_arg
    ):
        raise ValueError("wheel build config profile projection is invalid")
    return config


def load_wheel_build_config(path: Path) -> dict[str, Any]:
    """Load the sole canonical setup release-mode input."""
    return _validate_wheel_build_config_value(
        _canonical_record(path, "wheel build config")
    )


def wheel_build_profile(profile_id: object) -> dict[str, Any]:
    """Project one stable canonical wheel profile by ID."""
    profile = (
        WHEEL_BUILD_PROFILES.get(profile_id) if isinstance(profile_id, str) else None
    )
    if profile is None:
        raise ValueError("wheel build profile_id is invalid")
    return {
        "id": profile_id,
        "distribution": profile["distribution"],
        "build_platform": profile["build_platform"],
        "wheel_platform": profile["wheel_platform"],
        "python_version": profile["python"]["version"],
        "python_abi": profile["python"]["abi"],
        "runtime_requirements": copy.deepcopy(profile["runtime_requirements"]),
    }


def prepare_wheel_source(config_path: Path, source_root: Path) -> dict[str, Any]:
    """Rewrite only the authorized PEP 621 project-name assignment."""
    config = load_wheel_build_config(config_path)
    project_path = Path(source_root) / "pyproject.toml"
    if not project_path.is_file() or project_path.is_symlink():
        raise ValueError("wheel source pyproject.toml must be one regular file")
    raw = project_path.read_bytes()
    try:
        text = raw.decode("utf-8")
        original = tomllib.loads(text)
    except UnicodeDecodeError as error:
        raise ValueError("wheel source pyproject.toml must be UTF-8") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(
            f"wheel source pyproject.toml is invalid TOML: {error}"
        ) from error
    project = original.get("project") if isinstance(original, dict) else None
    original_name = project.get("name") if isinstance(project, dict) else None
    target = str(config["distribution"])
    if original_name == target:
        return {
            "distribution": target,
            "kind": "ucm-wheel-source-preparation",
            "project_file": "pyproject.toml",
            "schema_version": 1,
        }
    if original_name != "uc-manager":
        raise ValueError(
            "wheel source semantic project.name must be uc-manager or the target"
        )
    expected = copy.deepcopy(original)
    expected["project"]["name"] = target
    source_bytes = b"uc-manager"
    target_bytes = target.encode("utf-8")
    candidates: list[bytes] = []
    offset = 0
    while True:
        start = raw.find(source_bytes, offset)
        if start < 0:
            break
        end = start + len(source_bytes)
        candidate = raw[:start] + target_bytes + raw[end:]
        try:
            parsed = tomllib.loads(candidate.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError):
            pass
        else:
            if parsed == expected:
                candidates.append(candidate)
        offset = start + 1
    if len(candidates) != 1:
        raise ValueError(
            "wheel source project.name textual authority is missing or ambiguous"
        )
    _atomic_write_bytes(
        project_path,
        candidates[0],
        mode=project_path.stat().st_mode & 0o7777,
    )
    return {
        "distribution": target,
        "kind": "ucm-wheel-source-preparation",
        "project_file": "pyproject.toml",
        "schema_version": 1,
    }


def _validate_production_wheel_task(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != PRODUCTION_WHEEL_TASK_FIELDS:
        raise ValueError("production wheel task fields are not exact")
    task = copy.deepcopy(value)
    payload = {key: item for key, item in task.items() if key != "sha256"}
    actual_hash = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    profile = task.get("profile_id")
    settings = WHEEL_BUILD_PROFILES.get(profile)
    if settings is None:
        raise ValueError("production wheel task profile_id is invalid")
    expected_profile_fields = {
        "distribution": settings["distribution"],
        "build_platform": settings["build_platform"],
        "python_version": settings["python"]["version"],
        "python_abi": settings["python"]["abi"],
        "wheel_platform": settings["wheel_platform"],
        "runtime_requirements": settings["runtime_requirements"],
    }
    for field, expected in expected_profile_fields.items():
        if task.get(field) != expected:
            raise ValueError(f"production wheel task {field} differs from profile")
    cpu_arch = task.get("cpu_arch")
    stage = task.get("stage")
    base_version = str(task.get("base_version"))
    wheel_version = str(task.get("wheel_version"))
    stage_patterns = {
        "draft": re.escape(base_version) + r"\.dev[1-9][0-9]*",
        "rc": re.escape(base_version) + r"rc[1-9][0-9]*",
        "stable": re.escape(base_version),
        "hotfix": re.escape(base_version),
    }
    if (
        task.get("kind") != "ucm-production-wheel-build-task"
        or task.get("schema_version") != 1
    ):
        raise ValueError("production wheel task identity is invalid")
    if task.get("sha256") != actual_hash:
        raise ValueError("production wheel task hash mismatch")
    if (
        cpu_arch not in {"amd64", "arm64"}
        or task.get("spec_id") != f"{profile}-{cpu_arch}"
        or task.get("platform") != f"linux/{cpu_arch}"
    ):
        raise ValueError("production wheel task cpu_arch/spec/platform is invalid")
    if (
        re.fullmatch(
            r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)",
            base_version,
        )
        is None
        or stage not in stage_patterns
        or re.fullmatch(stage_patterns[stage], wheel_version) is None
    ):
        raise ValueError("production wheel task stage/version is invalid")
    if (
        re.fullmatch(r"[0-9a-f]{40}", str(task.get("source_sha"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(task.get("source_identity_sha256")))
        is None
    ):
        raise ValueError("production wheel task source identity is invalid")
    if DIGEST_RE.fullmatch(str(task.get("dependency_lock_sha256"))) is None:
        raise ValueError("production wheel task dependency_lock is invalid")
    if task.get("write_authority") != []:
        raise ValueError("production wheel task write authority is invalid")
    _validate_native_lists(task)
    builder = task.get("builder")
    if (
        not isinstance(builder, dict)
        or not isinstance(builder.get("repository"), str)
        or not builder["repository"]
        or any(
            DIGEST_RE.fullmatch(str(builder.get(field))) is None
            for field in ("manifest_digest", "config_digest")
        )
    ):
        raise ValueError("production wheel task builder is invalid")
    return task


def _validate_production_task_authority(
    task: dict[str, Any], authority: dict[str, Any]
) -> None:
    builder = task["builder"]
    expected = {
        "schema_version": authority["schema_version"],
        "kind": AUTHORITY_KIND,
        "spec_id": task["spec_id"],
        "profile_id": task["profile_id"],
        "distribution": task["distribution"],
        "base_version": task["base_version"],
        "stage": task["stage"],
        "cpu_arch": task["cpu_arch"],
        "platform": task["platform"],
        "wheel_version": task["wheel_version"],
        "source_sha": task["source_sha"],
        "task_sha256": "sha256:" + task["sha256"],
        "builder_coordinate": (f"{builder['repository']}@{builder['manifest_digest']}"),
        "builder_config_digest": builder["config_digest"],
        "dependency_lock_sha256": task["dependency_lock_sha256"],
        "required_native": task["required_native"],
        "forbidden_native": task["forbidden_native"],
    }
    mismatches = [
        field for field, value in expected.items() if authority.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            f"production build authority differs from selected task: {mismatches}"
        )
    if (
        authority["schema_version"] == 3
        and authority.get("source_version") != task["wheel_version"]
    ):
        raise ValueError("production build authority source version differs from task")


def build_production_authority(
    task: dict[str, Any],
    source_context: dict[str, Any],
    source_date_epoch: int,
    tool_wheels: dict[str, str],
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    """Bind one production task to a version-materialized source context."""
    task = _validate_production_wheel_task(task)
    required_context = {
        "schema_version",
        "kind",
        "source_sha",
        "source_tree",
        "materialized_tree",
        "source_version",
        "source_archive_sha256",
        "build_context_sha256",
    }
    if (
        not isinstance(source_context, dict)
        or not required_context.issubset(source_context)
        or source_context["schema_version"] != 2
        or source_context["kind"] != SOURCE_CONTEXT_KIND
        or source_context["source_sha"] != task["source_sha"]
        or source_context["source_version"] != task["wheel_version"]
        or any(
            re.fullmatch(r"[0-9a-f]{40}", str(source_context.get(field))) is None
            for field in ("source_tree", "materialized_tree")
        )
        or any(
            DIGEST_RE.fullmatch(str(source_context.get(field))) is None
            for field in ("source_archive_sha256", "build_context_sha256")
        )
    ):
        raise ValueError("production source context differs from selected task")
    if (
        type(source_date_epoch) is not int
        or not 315532800 <= source_date_epoch <= 4354819199
    ):
        raise ValueError("production source date epoch is invalid")
    if (
        not isinstance(tool_wheels, dict)
        or len(tool_wheels) != 7
        or any(
            not isinstance(name, str)
            or not name.endswith(".whl")
            or DIGEST_RE.fullmatch(str(digest)) is None
            for name, digest in tool_wheels.items()
        )
    ):
        raise ValueError("production tool wheel authority is invalid")
    builder = task["builder"]
    authority = {
        "schema_version": 3,
        "kind": AUTHORITY_KIND,
        "spec_id": task["spec_id"],
        "profile_id": task["profile_id"],
        "distribution": task["distribution"],
        "base_version": task["base_version"],
        "stage": task["stage"],
        "cpu_arch": task["cpu_arch"],
        "platform": task["platform"],
        "wheel_version": task["wheel_version"],
        "source_sha": task["source_sha"],
        "source_tree": source_context["source_tree"],
        "materialized_tree": source_context["materialized_tree"],
        "source_version": source_context["source_version"],
        "source_archive_sha256": source_context["source_archive_sha256"],
        "source_date_epoch": source_date_epoch,
        "task_sha256": "sha256:" + task["sha256"],
        "builder_coordinate": (f"{builder['repository']}@{builder['manifest_digest']}"),
        "builder_config_digest": builder["config_digest"],
        "dependency_lock_sha256": task["dependency_lock_sha256"],
        "tool_wheels": dict(sorted(tool_wheels.items())),
        "required_native": task["required_native"],
        "forbidden_native": task["forbidden_native"],
        "build_context_sha256": source_context["build_context_sha256"],
    }
    _validate_wheel_build_authority(authority)
    _validate_production_task_authority(task, authority)
    if output is not None:
        _write_canonical(Path(output), authority)
    return authority


def build_wheel_config(
    task_path: Path, authority_path: Path, output: Path
) -> dict[str, Any]:
    """Project one canonical setup input from a selected wheel task."""
    authority = _canonical_record(authority_path, "build authority")
    _validate_wheel_build_authority(authority)
    task_value = _canonical_record(task_path, "selected wheel task")
    if authority["schema_version"] in {1, 4}:
        task = _validate_wheel_task(task_value)
        _validate_build_authority(authority, task["spec_id"], task)
        distribution = task["dist_name"]
        platform_arg = task["build"]["platform_arg"]
    else:
        task = _validate_production_wheel_task(task_value)
        _validate_production_task_authority(task, authority)
        distribution = task["distribution"]
        platform_arg = task["build_platform"]
    result = {
        "authority": authority,
        "distribution": distribution,
        "kind": WHEEL_BUILD_CONFIG_KIND,
        "platform": platform_arg,
        "python": {
            "abi": task["python_abi"],
            "version": task["python_version"],
        },
        "runtime_requirements": task["runtime_requirements"],
        "schema_version": 1,
    }
    _validate_wheel_build_config_value(result)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_canonical(output, result)
    return result


def _tool_wheel_authority(task: dict[str, Any]) -> dict[str, str]:
    records = task["dependency_lock"]["build_tools"]
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError("build tool wheel authority is invalid")
    wheels = {record["filename"]: record["sha256"] for record in records}
    if len(wheels) != len(records):
        raise ValueError("build tool wheel authority is ambiguous")
    return dict(sorted(wheels.items()))


def _git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _canonical_source_version(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source version must be a non-empty string")
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"source version is not valid PEP 440: {value!r}") from error
    if str(parsed) != value:
        raise ValueError(f"source version is not canonical PEP 440: {value!r}")
    return value


def _git_object_digest(kind: str, data: bytes) -> bytes:
    header = f"{kind} {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).digest()  # noqa: S324 - Git SHA-1 object ID


def _source_context_digest(archive: bytes, commit_payload: bytes) -> str:
    material = (
        SOURCE_CONTEXT_PREFIX
        + len(archive).to_bytes(8, byteorder="big")
        + archive
        + commit_payload
    )
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _commit_tree(commit_payload: bytes) -> str:
    """Strictly parse the one tree identity carried by a raw Git commit payload."""
    if b"\0" in commit_payload or b"\r" in commit_payload:
        raise ValueError("source commit payload contains invalid control bytes")
    header_block, separator, _ = commit_payload.partition(b"\n\n")
    if not separator or not header_block:
        raise ValueError("source commit payload has no complete header block")

    headers: list[tuple[bytes, bytes]] = []
    has_header = False
    for line in header_block.split(b"\n"):
        if line.startswith(b" "):
            if not has_header:
                raise ValueError("source commit payload has an orphan continuation")
            continue
        match = re.fullmatch(rb"([a-z][a-z0-9-]*) (.+)", line)
        if match is None:
            raise ValueError("source commit payload has a malformed header")
        headers.append((match.group(1), match.group(2)))
        has_header = True

    tree_headers = [value for name, value in headers if name == b"tree"]
    if not headers or headers[0][0] != b"tree" or len(tree_headers) != 1:
        raise ValueError(
            "source commit payload must have exactly one leading tree header"
        )
    tree = tree_headers[0]
    if re.fullmatch(rb"[0-9a-f]{40}", tree) is None:
        raise ValueError("source commit payload tree is invalid")
    return tree.decode("ascii")


def _git_tree_digest(archive: tarfile.TarFile) -> str:
    root: dict[str, Any] = {}
    directories: set[str] = set()
    seen: set[str] = set()
    for member in archive.getmembers():
        name = member.name.rstrip("/")
        if not _safe_wheel_name(name) or name in seen:
            raise ValueError("source archive contains duplicate or unsafe paths")
        seen.add(name)
        if member.isdir():
            directories.add(name)
            continue
        if not (member.isfile() or member.issym()):
            raise ValueError(f"source archive has unsupported member type: {name}")
        node = root
        parts = name.split("/")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError("source archive has a file/directory path collision")
            node = child
        leaf = parts[-1]
        if leaf in node:
            raise ValueError("source archive has a file/directory path collision")
        if member.issym():
            data = member.linkname.encode("utf-8")
            mode = "120000"
        else:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"source archive member cannot be read: {name}")
            data = stream.read()
            mode = "100755" if member.mode & 0o111 else "100644"
        node[leaf] = (mode, _git_object_digest("blob", data))

    derived_directories: set[str] = set()

    def digest_tree(node: dict[str, Any], prefix: str = "") -> bytes:
        entries: list[tuple[str, bool, str, bytes]] = []
        for name, value in node.items():
            if isinstance(value, dict):
                path = f"{prefix}/{name}" if prefix else name
                derived_directories.add(path)
                entries.append((name, True, "40000", digest_tree(value, path)))
            else:
                mode, digest = value
                entries.append((name, False, mode, digest))

        def compare(
            left: tuple[str, bool, str, bytes],
            right: tuple[str, bool, str, bytes],
        ) -> int:
            left_key = left[0].encode() + (b"/" if left[1] else b"\0")
            right_key = right[0].encode() + (b"/" if right[1] else b"\0")
            return (left_key > right_key) - (left_key < right_key)

        body = b"".join(
            mode.encode("ascii") + b" " + name.encode() + b"\0" + digest
            for name, _, mode, digest in sorted(
                entries, key=functools.cmp_to_key(compare)
            )
        )
        return _git_object_digest("tree", body)

    digest = digest_tree(root).hex()
    if directories != derived_directories:
        raise ValueError("source archive directory entries are not canonical")
    return digest


def _verify_source_root(archive: tarfile.TarFile, source_root: Path) -> None:
    expected = {member.name.rstrip("/"): member for member in archive.getmembers()}
    actual = {
        path.relative_to(source_root).as_posix(): path
        for path in source_root.rglob("*")
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(
            f"extracted source context path set differs: missing={missing}, extra={extra}"
        )
    for name, member in expected.items():
        path = actual[name]
        if member.isdir():
            if not path.is_dir() or path.is_symlink():
                raise ValueError(f"extracted source directory differs: {name}")
        elif member.issym():
            if not path.is_symlink() or os.readlink(path) != member.linkname:
                raise ValueError(f"extracted source symlink differs: {name}")
        else:
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"extracted source file differs: {name}")
            stream = archive.extractfile(member)
            if stream is None or path.read_bytes() != stream.read():
                raise ValueError(f"extracted source file bytes differ: {name}")
            if bool(path.stat().st_mode & 0o111) != bool(member.mode & 0o111):
                raise ValueError(f"extracted source executable mode differs: {name}")


def _materialized_source_version_bytes(
    text: str, source_version: str, *, source: str
) -> bytes:
    try:
        return version_config.materialize_bytes(text, source_version, source=source)
    except ValueError as error:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        legacy_prefix = f"{LEGACY_UCM_VERSION_KEY}="
        current_prefix = f"{version_config.UCM_VERSION_KEY}="
        legacy_indexes = [
            index for index, line in enumerate(lines) if line.startswith(legacy_prefix)
        ]
        if len(legacy_indexes) != 1:
            raise
        # Preserve the old key when rebuilding historical source archives while
        # keeping the current version.ini parser strict about UCM_VERSION.
        if len(lines) == 1:
            try:
                Version(lines[0].split("=", 1)[1])
            except InvalidVersion:
                raise error
            return f"{legacy_prefix}{source_version}\n".encode()
        migrated_lines = [
            (
                f"{current_prefix}{line.removeprefix(legacy_prefix)}"
                if index == legacy_indexes[0]
                else line
            )
            for index, line in enumerate(lines)
        ]
        try:
            materialized = version_config.materialize_bytes(
                "\n".join(migrated_lines) + "\n", source_version, source=source
            )
        except ValueError:
            raise error
        if not materialized.startswith(current_prefix.encode()):
            raise error
        return legacy_prefix.encode() + materialized[len(current_prefix) :]


def verify_source_context(
    archive_path: Path,
    manifest_path: Path,
    source_root: Path,
    commit_payload_path: Path | None = None,
    expected_source_sha: str | None = None,
    expected_source_version: str | None = None,
    repository_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Verify original Git identity and its version-materialized source archive."""
    if commit_payload_path is None or expected_source_sha is None:
        raise ValueError("trusted expected source SHA and commit payload are required")
    if re.fullmatch(r"[0-9a-f]{40}", expected_source_sha) is None:
        raise ValueError("trusted expected source SHA is invalid")
    manifest = _canonical_record(manifest_path, "source context manifest")
    fields = {
        "schema_version",
        "kind",
        "source_sha",
        "source_tree",
        "materialized_tree",
        "source_version",
        "source_object_type",
        "source_commit_payload_sha256",
        "source_archive_sha256",
        "build_context_sha256",
    }
    if set(manifest) != fields:
        raise ValueError("source context manifest fields are not exact")
    if manifest["schema_version"] != 2 or manifest["kind"] != SOURCE_CONTEXT_KIND:
        raise ValueError("source context manifest identity is invalid")
    source_version = _canonical_source_version(manifest["source_version"])
    if (
        expected_source_version is not None
        and source_version != _canonical_source_version(expected_source_version)
    ):
        raise ValueError("source context version differs from trusted expected version")
    raw = Path(archive_path).read_bytes()
    commit_payload = Path(commit_payload_path).read_bytes()
    archive_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    commit_payload_digest = "sha256:" + hashlib.sha256(commit_payload).hexdigest()
    context_digest = _source_context_digest(raw, commit_payload)
    if manifest["source_archive_sha256"] != archive_digest:
        raise ValueError("source context archive digest differs from actual bytes")
    if manifest["source_commit_payload_sha256"] != commit_payload_digest:
        raise ValueError("source commit payload digest differs from actual bytes")
    if manifest["build_context_sha256"] != context_digest:
        raise ValueError("source context digest differs from actual context bytes")
    if manifest["source_sha"] != expected_source_sha:
        raise ValueError("source context SHA differs from trusted expected source SHA")
    if manifest["source_object_type"] != "commit":
        raise ValueError("source context object type must be commit")
    actual_commit_sha = _git_object_digest("commit", commit_payload).hex()
    if actual_commit_sha != expected_source_sha:
        raise ValueError(
            "source commit object does not match trusted expected source SHA"
        )
    if re.fullmatch(r"[0-9a-f]{40}", str(manifest["source_tree"])) is None:
        raise ValueError("source context tree is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", str(manifest["materialized_tree"])) is None:
        raise ValueError("materialized source tree is invalid")
    commit_tree = _commit_tree(commit_payload)
    if manifest["source_tree"] != commit_tree:
        raise ValueError("source context tree does not match source commit")
    source_version_config = subprocess.run(
        [
            "git",
            "-C",
            str(Path(repository_root).resolve()),
            "show",
            f"{expected_source_sha}:version.ini",
        ],
        capture_output=True,
        check=False,
    )
    if source_version_config.returncode != 0:
        raise ValueError("source commit version.ini cannot be read")
    expected_version_bytes = _materialized_source_version_bytes(
        source_version_config.stdout.decode("utf-8"),
        source_version,
        source=f"{expected_source_sha}:version.ini",
    )
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        if archive.pax_headers:
            raise ValueError(
                "materialized source archive has unexpected global headers"
            )
        if _git_tree_digest(archive) != manifest["materialized_tree"]:
            raise ValueError("source archive tree does not match materialized tree")
        try:
            version_member = archive.getmember("version.ini")
            version_stream = archive.extractfile(version_member)
        except KeyError:
            version_stream = None
        if version_stream is None or version_stream.read() != expected_version_bytes:
            raise ValueError("materialized source version.ini differs")
        _verify_source_root(archive, Path(source_root))
    return manifest


def prepare_source_context(
    output_dir: Path,
    source_sha: str,
    source_version: str,
    *,
    repository_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Create a deterministic source tree with only version.ini materialized."""
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source context requires a full lowercase Git commit")
    source_version = _canonical_source_version(source_version)
    repository_root = Path(repository_root).resolve()

    def git(
        *arguments: str,
        environment: dict[str, str] | None = None,
        input_bytes: bytes | None = None,
    ) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            input=input_bytes,
            capture_output=True,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            raise ValueError("git source context command failed")
        return completed.stdout

    if git("rev-parse", "HEAD").decode().strip() != source_sha:
        raise ValueError("source context SHA does not match checked HEAD")
    if git("cat-file", "-t", source_sha).decode().strip() != "commit":
        raise ValueError("source context identity must be a commit")
    source_tree = git("rev-parse", f"{source_sha}^{{tree}}").decode().strip()
    source_epoch = git("show", "-s", "--format=%ct", source_sha).decode().strip()
    if re.fullmatch(r"[1-9][0-9]*", source_epoch) is None:
        raise ValueError("source context commit time is invalid")
    version_entry = git("ls-tree", source_sha, "--", "version.ini").decode().strip()
    match = re.fullmatch(
        r"(?P<mode>100644|100755) blob [0-9a-f]{40}\tversion\.ini", version_entry
    )
    if match is None:
        raise ValueError("source context version.ini must be one regular Git file")

    repository_objects = git("rev-parse", "--git-path", "objects").decode().strip()
    repository_objects_path = Path(repository_objects)
    if not repository_objects_path.is_absolute():
        repository_objects_path = (repository_root / repository_objects_path).resolve()
    source_version_bytes = git("show", f"{source_sha}:version.ini")
    version_bytes = _materialized_source_version_bytes(
        source_version_bytes.decode("utf-8"),
        source_version,
        source=f"{source_sha}:version.ini",
    )
    with tempfile.TemporaryDirectory(prefix="ucm-materialized-source-") as temporary:
        temporary_root = Path(temporary)
        object_directory = temporary_root / "objects"
        object_directory.mkdir()
        environment = {
            **os.environ,
            "GIT_INDEX_FILE": str(temporary_root / "index"),
            "GIT_OBJECT_DIRECTORY": str(object_directory),
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(repository_objects_path),
        }
        git("read-tree", source_sha, environment=environment)
        version_blob = (
            git(
                "hash-object",
                "-w",
                "--stdin",
                environment=environment,
                input_bytes=version_bytes,
            )
            .decode()
            .strip()
        )
        git(
            "update-index",
            "--add",
            "--cacheinfo",
            f"{match.group('mode')},{version_blob},version.ini",
            environment=environment,
        )
        materialized_tree = git("write-tree", environment=environment).decode().strip()
        archive_bytes = git(
            "archive",
            "--format=tar",
            f"--mtime=@{source_epoch}",
            materialized_tree,
            environment=environment,
        )

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("source context output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "ucm-source.tar"
    archive_path.write_bytes(archive_bytes)
    commit_payload = git("cat-file", "commit", source_sha)
    if _git_object_digest("commit", commit_payload).hex() != source_sha:
        raise ValueError("exported Git commit payload differs from source SHA")
    if _commit_tree(commit_payload) != source_tree:
        raise ValueError("exported Git commit tree differs from source tree")
    commit_payload_path = output_dir / "source-commit.payload"
    commit_payload_path.write_bytes(commit_payload)
    manifest = {
        "schema_version": 2,
        "kind": SOURCE_CONTEXT_KIND,
        "source_sha": source_sha,
        "source_tree": source_tree,
        "materialized_tree": materialized_tree,
        "source_version": source_version,
        "source_object_type": "commit",
        "source_commit_payload_sha256": "sha256:"
        + hashlib.sha256(commit_payload).hexdigest(),
        "source_archive_sha256": "sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        "build_context_sha256": _source_context_digest(archive_bytes, commit_payload),
    }
    manifest_path = output_dir / "source-context.json"
    _write_canonical(manifest_path, manifest)
    return {
        **manifest,
        "source_archive_path": str(archive_path),
        "source_commit_payload_path": str(commit_payload_path),
        "source_manifest_path": str(manifest_path),
    }


def _validate_build_authority(
    authority: dict[str, Any],
    spec_id: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    schema_version = authority.get("schema_version")
    expected_fields = {
        1: EXTENDED_V1_AUTHORITY_FIELDS,
        4: EXTENDED_V4_AUTHORITY_FIELDS,
    }.get(schema_version)
    if expected_fields is None or set(authority) != expected_fields:
        raise ValueError("build authority fields are not exact")
    if DIGEST_RE.fullmatch(str(authority["build_context_sha256"])) is None:
        raise ValueError("build authority context digest is invalid")
    if DIGEST_RE.fullmatch(str(authority["source_archive_sha256"])) is None:
        raise ValueError("build authority source archive digest is invalid")
    root = task["builder"]["root"]
    expected = {
        "schema_version": schema_version,
        "kind": AUTHORITY_KIND,
        "task_id": task["task_id"],
        "spec_id": spec_id,
        "profile_id": task["profile_id"],
        "cpu_arch": task["cpu_arch"],
        "platform": task["platform"],
        "build": task["build"],
        "python_version": task["python_version"],
        "python_abi": task["python_abi"],
        "wheel_version": task["wheel_version"],
        "wheel_platform": task["wheel_platform"],
        "task_sha256": task["task_sha256"],
        "builder_coordinate": f"{root['repository']}@{root['manifest_digest']}",
        "builder_config_digest": root["config_digest"],
        "dependency_lock_sha256": task["dependency_lock_sha256"],
        "tool_wheels": _tool_wheel_authority(task),
        "required_native": task["required_native"],
        "forbidden_native": task["forbidden_native"],
        "runtime_requirements": task["runtime_requirements"],
    }
    for name, value in expected.items():
        if authority[name] != value:
            raise ValueError(f"build authority {name} differs from reviewed task")
    source_sha = authority["source_sha"]
    if re.fullmatch(r"[0-9a-f]{40}", str(source_sha)) is None:
        raise ValueError("build source authority SHA is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", str(authority["source_tree"])) is None:
        raise ValueError("build source authority tree is invalid")
    if schema_version == 4 and (
        re.fullmatch(r"[0-9a-f]{40}", str(authority["materialized_tree"])) is None
        or authority["source_version"] != task["wheel_version"]
    ):
        raise ValueError("materialized build source authority is invalid")
    checked_head = _git_value("rev-parse", "HEAD")
    if checked_head is not None:
        if checked_head != source_sha:
            raise ValueError(
                "build source authority does not match checked source HEAD"
            )
        if (
            _git_value("rev-parse", f"{source_sha}^{{tree}}")
            != authority["source_tree"]
        ):
            raise ValueError(
                "build source authority tree does not match checked source"
            )
    _zip_timestamp(authority["source_date_epoch"])
    return authority


def _validate_dependency_closure(
    closure: dict[str, Any],
    raw_wheel_sha256: str,
    authority: dict[str, Any],
    native: dict[str, Any],
    archive: zipfile.ZipFile,
    *,
    external_required_dependencies: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "kind",
        "spec_id",
        "raw_wheel_sha256",
        "build_context_sha256",
        "native_members",
        "unresolved_dependencies",
        "closure_sha256",
    }
    if set(closure) != fields:
        raise ValueError("dependency closure fields are not exact")
    digest_input = dict(closure)
    closure_digest = digest_input.pop("closure_sha256")
    if (
        closure_digest
        != "sha256:" + hashlib.sha256(canonical_bytes(digest_input)).hexdigest()
    ):
        raise ValueError("dependency closure digest is invalid")
    if (
        closure["schema_version"] != 1
        or closure["kind"] != CLOSURE_KIND
        or closure["spec_id"] != authority["spec_id"]
        or closure["raw_wheel_sha256"] != raw_wheel_sha256
        or closure["build_context_sha256"] != authority["build_context_sha256"]
    ):
        raise ValueError("dependency closure authority binding is invalid")
    if closure["unresolved_dependencies"] != []:
        raise ValueError(
            f"dependency closure has unresolved dependencies: {closure['unresolved_dependencies']}"
        )
    expected_names = set(native["native_artifacts"])
    records = closure["native_members"]
    if not isinstance(records, dict) or set(records) != expected_names:
        raise ValueError("dependency closure native member set is not exact")
    member_digests = {
        name: "sha256:" + hashlib.sha256(archive.read(name)).hexdigest()
        for name in expected_names
    }
    member_by_basename = {PurePosixPath(name).name: name for name in expected_names}
    declared_external = _external_required_by_dependency(external_required_dependencies)
    observed_external: set[str] = set()
    for name in sorted(expected_names):
        record = records[name]
        if not isinstance(record, dict) or set(record) != {
            "dt_needed",
            "resolved_dependencies",
            "unresolved_dependencies",
        }:
            raise ValueError(f"dependency closure record is invalid: {name}")
        if record["unresolved_dependencies"] != []:
            raise ValueError(
                f"dependency closure has unresolved dependencies for {name}"
            )
        needed = native["dt_needed"][name]
        if record["dt_needed"] != needed:
            missing = sorted(set(needed) - set(record.get("dt_needed", [])))
            if missing:
                raise ValueError(
                    f"dependency closure has unresolved entries for {name}: {missing}"
                )
            raise ValueError(f"dependency closure DT_NEEDED differs for {name}")
        resolutions = record["resolved_dependencies"]
        if not isinstance(resolutions, list) or not all(
            isinstance(resolution, dict) for resolution in resolutions
        ):
            raise ValueError(f"dependency closure resolutions are invalid: {name}")
        dependencies = [resolution.get("dependency") for resolution in resolutions]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"dependency closure has duplicate resolutions: {name}")
        direct_resolutions = {
            resolution.get("dependency")
            for resolution in resolutions
            if resolution.get("direct") is True
        }
        if direct_resolutions != set(needed):
            missing = sorted(set(needed) - direct_resolutions)
            raise ValueError(f"dependency closure has unresolved entries: {missing}")
        for resolution in resolutions:
            dependency = resolution.get("dependency")
            if not isinstance(dependency, str) or resolution.get("direct") not in {
                True,
                False,
            }:
                raise ValueError(
                    f"dependency closure resolution identity is invalid: {name}"
                )
            internal_member = member_by_basename.get(PurePosixPath(dependency).name)
            if internal_member is not None:
                expected_resolution = {
                    "dependency": dependency,
                    "direct": dependency in needed,
                    "kind": "wheel-member",
                    "member": internal_member,
                    "sha256": member_digests[internal_member],
                }
                if resolution != expected_resolution:
                    raise ValueError(
                        f"dependency closure must resolve internal {dependency} "
                        "from the exact wheel member"
                    )
            elif resolution.get("kind") == "external-required":
                declaration = declared_external.get(str(dependency))
                if declaration is None or resolution != _external_required_resolution(
                    declaration
                ):
                    raise ValueError(
                        "dependency closure external-required resolution is invalid: "
                        f"{dependency}"
                    )
                observed_external.add(str(dependency))
            elif resolution.get("kind") == "virtual":
                if resolution != {
                    "dependency": "linux-vdso.so.1",
                    "direct": dependency in needed,
                    "kind": "virtual",
                }:
                    raise ValueError(
                        f"dependency closure virtual resolution is invalid: {dependency}"
                    )
            elif (
                not isinstance(resolution, dict)
                or set(resolution) != {"dependency", "direct", "kind", "path", "sha256"}
                or resolution["kind"] != "external"
                or not isinstance(resolution["path"], str)
                or not resolution["path"].startswith("/")
                or DIGEST_RE.fullmatch(str(resolution["sha256"])) is None
            ):
                raise ValueError(
                    f"dependency closure external resolution is invalid: {dependency}"
                )
    if observed_external != set(declared_external):
        raise ValueError(
            "dependency closure external-required set differs from declaration: "
            f"expected={sorted(declared_external)}, observed={sorted(observed_external)}"
        )
    return closure


def _python_extension_suffix(spec: dict[str, Any], task: dict[str, Any]) -> str:
    architecture = cpu_toolchain_authority(spec["cpu_arch"]).wheel_arch
    checks = task.get("builder", {}).get("checks", [])
    soabi_checks = [item for item in checks if item.get("kind") == "python-soabi"]
    if len(soabi_checks) != 1:
        raise ValueError("wheel task requires one Python SOABI authority")
    prefix = soabi_checks[0].get("prefix")
    version = task.get("python_version")
    abi = task.get("python_abi")
    if (
        not isinstance(prefix, str)
        or not isinstance(version, str)
        or not isinstance(abi, str)
        or abi != "cp" + version.replace(".", "")
        or abi != spec.get("python_abi")
    ):
        raise ValueError("wheel task Python ABI/SOABI authority is inconsistent")
    return f".{prefix}-{abi.removeprefix('cp')}-{architecture}-linux-gnu.so"


def _expected_native_members(
    spec: dict[str, Any], task: dict[str, Any]
) -> dict[str, str]:
    suffix = _python_extension_suffix(spec, task)
    known = set(spec["required_native"]) | set(spec["forbidden_native"])
    missing = sorted(set(spec["required_native"]) - set(NATIVE_MEMBER_DIRECTORIES))
    if missing:
        raise ValueError(f"native component archive paths are undeclared: {missing}")
    return {
        component: (
            f"{NATIVE_MEMBER_DIRECTORIES[component]}/lib{component}.so"
            if component in SHARED_LIBRARY_COMPONENTS
            else f"{NATIVE_MEMBER_DIRECTORIES[component]}/{component}{suffix}"
        )
        for component in known & set(NATIVE_MEMBER_DIRECTORIES)
    }


def _elf_string(data: bytes, offset: int, size: int, label: str) -> str:
    if offset < 0 or offset >= len(data):
        raise ValueError(f"ELF string offset is invalid in {label}")
    limit = min(len(data), offset + size)
    end = data.find(b"\0", offset, limit)
    if end < 0:
        raise ValueError(f"ELF string is unterminated in {label}")
    try:
        return data[offset:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"ELF dynamic string is not UTF-8 in {label}") from error


def _inspect_elf(data: bytes, label: str) -> dict[str, Any]:
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ValueError(f"native member is not ELF: {label}")
    if data[4] != 2 or data[5] != 1 or data[6] != 1:
        raise ValueError(f"ELF must be 64-bit little-endian version 1: {label}")
    try:
        header = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
    except struct.error as error:
        raise ValueError(f"ELF header is truncated: {label}") from error
    machine = header[1]
    program_offset = header[4]
    program_size = header[8]
    program_count = header[9]
    if program_size != 56 or program_count < 1 or program_count > 1024:
        raise ValueError(f"ELF program header table is invalid: {label}")
    if program_offset + program_size * program_count > len(data):
        raise ValueError(f"ELF program header table is truncated: {label}")
    loads: list[tuple[int, int, int]] = []
    dynamics: list[tuple[int, int]] = []
    for index in range(program_count):
        values = struct.unpack_from(
            "<IIQQQQQQ", data, program_offset + index * program_size
        )
        kind, _, file_offset, virtual_address, _, file_size, _, _ = values
        if file_offset + file_size > len(data):
            raise ValueError(f"ELF program segment is truncated: {label}")
        if kind == 1:
            loads.append((virtual_address, file_offset, file_size))
        elif kind == 2:
            dynamics.append((file_offset, file_size))
    if len(dynamics) != 1:
        raise ValueError(f"ELF requires exactly one PT_DYNAMIC segment: {label}")
    dynamic_offset, dynamic_size = dynamics[0]
    if dynamic_size % 16 or dynamic_size > 1024 * 1024:
        raise ValueError(f"ELF dynamic segment is invalid: {label}")
    entries: list[tuple[int, int]] = []
    for offset in range(dynamic_offset, dynamic_offset + dynamic_size, 16):
        tag, value = struct.unpack_from("<QQ", data, offset)
        entries.append((tag, value))
        if tag == 0:
            break
    if not entries or entries[-1][0] != 0:
        raise ValueError(f"ELF dynamic segment lacks DT_NULL: {label}")
    string_addresses = [value for tag, value in entries if tag == 5]
    string_sizes = [value for tag, value in entries if tag == 10]
    if len(string_addresses) != 1 or len(string_sizes) != 1:
        raise ValueError(f"ELF dynamic string table is ambiguous: {label}")
    string_address = string_addresses[0]
    string_size = string_sizes[0]
    string_offset: int | None = None
    for virtual_address, file_offset, file_size in loads:
        if virtual_address <= string_address < virtual_address + file_size:
            string_offset = file_offset + string_address - virtual_address
            break
    if string_offset is None or string_offset + string_size > len(data):
        raise ValueError(f"ELF dynamic string table is outside PT_LOAD: {label}")
    needed = sorted(
        _elf_string(data, string_offset + value, string_size - value, label)
        for tag, value in entries
        if tag == 1
    )
    runpaths = [
        _elf_string(data, string_offset + value, string_size - value, label)
        for tag, value in entries
        if tag in {15, 29}
    ]
    if runpaths:
        raise ValueError(f"ELF RPATH/RUNPATH is forbidden in {label}: {runpaths}")
    if any(marker in data for marker in HOST_PATH_MARKERS):
        raise ValueError(f"ELF source/path leakage detected in {label}")
    return {"machine": machine, "needed": needed}


def _verify_native_evidence(
    archive: zipfile.ZipFile,
    spec: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    required = spec["required_native"]
    forbidden = spec["forbidden_native"]
    expected_members = _expected_native_members(spec, task)
    extension_suffix = _python_extension_suffix(spec, task)
    components_by_member = {
        member: component for component, member in expected_members.items()
    }
    native_members: dict[str, str] = {}
    elf_evidence: dict[str, dict[str, Any]] = {}
    for item in archive.infolist():
        name = item.filename
        if item.is_dir() or not name.endswith(".so"):
            continue
        data = archive.read(name)
        if not data.startswith(b"\x7fELF"):
            continue
        component = components_by_member.get(name)
        if component is None:
            basename = PurePosixPath(name).name
            if any(
                basename == PurePosixPath(member).name
                or basename == component_name
                or basename == f"{component_name}.so"
                for component_name, member in expected_members.items()
            ):
                raise ValueError(f"native component archive path is not exact: {name}")
            component = next(
                (
                    forbidden_component
                    for forbidden_component in forbidden
                    if re.fullmatch(
                        rf"(?:lib)?{re.escape(forbidden_component)}(?:{re.escape(extension_suffix.removesuffix('.so'))})?\.so",
                        basename,
                    )
                ),
                None,
            )
        if component is None:
            raise ValueError(f"unclassified native wheel member: {name}")
        if component in native_members:
            raise ValueError(f"duplicate native component {component!r}")
        native_members[component] = name
        elf_evidence[name] = _inspect_elf(data, name)
    actual = set(native_members)
    missing = [item for item in required if item not in actual]
    present_forbidden = [item for item in forbidden if item in actual]
    extras = sorted(actual - set(required) - set(forbidden))
    if missing:
        raise ValueError(f"required native components are missing: {missing}")
    if present_forbidden:
        raise ValueError(
            f"forbidden native components are present: {present_forbidden}"
        )
    if extras or len(actual) != len(required):
        raise ValueError(f"native component set is not exact: extras={extras}")
    cpu_authority = cpu_toolchain_authority(spec["cpu_arch"])
    expected_machine = cpu_authority.elf_machine
    machine_name = cpu_authority.elf_machine_name
    wrong_machines = {
        name: evidence["machine"]
        for name, evidence in elf_evidence.items()
        if evidence["machine"] != expected_machine
    }
    if wrong_machines:
        raise ValueError(
            f"ELF machine does not match {spec['cpu_arch']}: {wrong_machines}"
        )
    allowed_needed = set(spec["allowed_dt_needed"])
    unapproved = {
        name: sorted(set(evidence["needed"]) - allowed_needed)
        for name, evidence in elf_evidence.items()
        if set(evidence["needed"]) - allowed_needed
    }
    if unapproved:
        raise ValueError(f"unapproved ELF DT_NEEDED entries: {unapproved}")
    ordered_members = {component: native_members[component] for component in required}
    return {
        "native_components": list(required),
        "native_members": ordered_members,
        "native_artifacts": list(ordered_members.values()),
        "elf_machines": [machine_name],
        "dt_needed": {
            name: elf_evidence[name]["needed"] for name in sorted(elf_evidence)
        },
    }


def _verify_component_manifest(
    archive: zipfile.ZipFile,
    spec: dict[str, Any],
    profile_id: str,
    source_sha: str,
    build_key: str,
) -> tuple[dict[str, Any], str]:
    names = [item.filename for item in archive.infolist() if not item.is_dir()]
    if names.count(COMPONENT_MANIFEST) != 1:
        raise ValueError("wheel requires exactly one native component manifest")
    raw = archive.read(COMPONENT_MANIFEST)
    manifest = _unique_json(raw, COMPONENT_MANIFEST)
    expected = {
        "schema_version": 1,
        "kind": "ucm-native-components",
        "profile_id": profile_id,
        "spec_id": spec["spec_id"],
        "source_sha": source_sha,
        "build_key": build_key,
        "version": spec["wheel_version"],
        "cpu_arch": spec["cpu_arch"],
        "required_native": spec["required_native"],
        "forbidden_native": spec["forbidden_native"],
        "installed_targets": spec["required_native"],
    }
    if manifest != expected:
        missing_native = sorted(
            set(spec["required_native"]) - set(manifest.get("installed_targets", []))
        )
        raise ValueError(
            "native component manifest does not match source/profile/architecture/"
            f"required native authority; missing={missing_native}"
        )
    if raw != canonical_bytes(manifest) + b"\n":
        raise ValueError("native component manifest is noncanonical")
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return manifest, digest


def build_authority_record(
    output: Path,
    spec_id: str,
    source_sha: str,
    source_date_epoch: int,
    builder_coordinate: str,
    wheelhouse: Path,
    source_archive: Path,
    source_commit_payload: Path,
    source_manifest: Path,
    source_root: Path,
    task_path: Path | None = None,
) -> dict[str, Any]:
    """Derive build authority from canonical source and locked tool bytes."""
    task = _selected_wheel_task(spec_id, task_path=task_path)
    context = verify_source_context(
        source_archive,
        source_manifest,
        source_root,
        source_commit_payload,
        source_sha,
        task["wheel_version"],
    )
    expected_tools = _tool_wheel_authority(task)
    wheelhouse = Path(wheelhouse)
    actual_names = sorted(item.name for item in wheelhouse.iterdir() if item.is_file())
    if actual_names != sorted(expected_tools):
        raise ValueError("build tool wheelhouse file set differs from reviewed lock")
    for filename, expected_digest in expected_tools.items():
        if _sha256(wheelhouse / filename) != expected_digest:
            raise ValueError(f"build tool wheel bytes differ from lock: {filename}")
    root = task["builder"]["root"]
    record: dict[str, Any] = {
        "schema_version": 4,
        "kind": AUTHORITY_KIND,
        "task_id": task["task_id"],
        "spec_id": spec_id,
        "profile_id": task["profile_id"],
        "cpu_arch": task["cpu_arch"],
        "platform": task["platform"],
        "build": task["build"],
        "python_version": task["python_version"],
        "python_abi": task["python_abi"],
        "wheel_version": task["wheel_version"],
        "wheel_platform": task["wheel_platform"],
        "source_sha": source_sha,
        "source_tree": context["source_tree"],
        "materialized_tree": context["materialized_tree"],
        "source_version": context["source_version"],
        "source_archive_sha256": context["source_archive_sha256"],
        "source_date_epoch": source_date_epoch,
        "task_sha256": task["task_sha256"],
        "builder_coordinate": builder_coordinate,
        "builder_config_digest": root["config_digest"],
        "dependency_lock_sha256": task["dependency_lock_sha256"],
        "tool_wheels": expected_tools,
        "required_native": task["required_native"],
        "forbidden_native": task["forbidden_native"],
        "runtime_requirements": task["runtime_requirements"],
    }
    record["build_context_sha256"] = context["build_context_sha256"]
    _validate_build_authority(record, spec_id, task)
    Path(output).write_bytes(canonical_bytes(record) + b"\n")
    return record


def _parse_ldd_output(
    label: str,
    direct_needed: list[str],
    output: str,
    *,
    external_required_dependencies: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Parse every ldd line; reject unresolved or unbound output."""
    direct = set(direct_needed)
    seen: set[str] = set()
    missing_dependencies: set[str] = set()
    resolved: list[dict[str, Any]] = []
    declared_external = _external_required_by_dependency(
        external_required_dependencies or []
    )
    missing = re.compile(r"^(\S+)\s+=>\s+not found$")
    arrow = re.compile(r"^(\S+)\s+=>\s+(\S+)(?:\s+\(0x[0-9a-fA-F]+\))?$")
    located = re.compile(r"^(\S+)\s+\(0x[0-9a-fA-F]+\)$")
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        missing_match = missing.fullmatch(line)
        if missing_match is not None:
            dependency = missing_match.group(1)
            missing_dependencies.add(dependency)
            continue
        arrow_match = arrow.fullmatch(line)
        if arrow_match is not None:
            dependency, location = arrow_match.groups()
            if dependency in seen:
                raise ValueError(f"ldd output has duplicate dependency for {label}")
            seen.add(dependency)
            if not location.startswith("/"):
                raise ValueError(
                    f"ldd resolution must use an absolute path for {label}: {line}"
                )
            resolved.append(
                {
                    "dependency": dependency,
                    "direct": dependency in direct,
                    "kind": "located",
                    "path": location,
                }
            )
            continue
        located_match = located.fullmatch(line)
        if located_match is None:
            raise ValueError(
                f"ldd output has a malformed extra line for {label}: {line}"
            )
        location = located_match.group(1)
        basename = PurePosixPath(location).name
        dependency = basename if basename in direct else location
        if dependency in seen:
            raise ValueError(f"ldd output has duplicate dependency for {label}")
        seen.add(dependency)
        if location == "linux-vdso.so.1":
            resolved.append(
                {
                    "dependency": location,
                    "direct": location in direct,
                    "kind": "virtual",
                }
            )
        elif location.startswith("/"):
            resolved.append(
                {
                    "dependency": dependency,
                    "direct": dependency in direct,
                    "kind": "located",
                    "path": location,
                }
            )
        else:
            raise ValueError(f"ldd output has an unbound line for {label}: {line}")
    unexpected = sorted(missing_dependencies - set(declared_external))
    if unexpected:
        relations = [
            f"{'direct' if dependency in direct else 'transitive'} {dependency}"
            for dependency in unexpected
        ]
        raise ValueError(
            f"unexpected unresolved dependencies for {label}: "
            f"{relations} are not found"
        )
    for dependency in sorted(missing_dependencies):
        declaration = declared_external[dependency]
        if dependency in direct or declaration["relation"] != "transitive":
            raise ValueError(
                f"direct dependency {dependency} cannot use a transitive "
                f"external-required declaration for {label}"
            )
        if dependency in seen:
            raise ValueError(
                f"ldd output both resolves and defers dependency for {label}: "
                f"{dependency}"
            )
        seen.add(dependency)
        resolved.append(_external_required_resolution(declaration))
    observed_direct = {
        item["dependency"] for item in resolved if item["dependency"] in direct
    }
    missing = sorted(direct - observed_direct)
    if missing:
        raise ValueError(f"direct dependencies are not found in ldd output: {missing}")
    return sorted(
        resolved,
        key=lambda item: (str(item["dependency"]), str(item.get("path", ""))),
    )


def validate_preflight_ldd(
    task: dict[str, Any],
    label: str,
    direct_needed: list[str],
    output: str,
) -> dict[str, Any]:
    """Apply one reviewed spec's exact external dependency boundary to ldd output."""
    task = _validate_wheel_task(task)
    spec_id = task["spec_id"]
    declarations = task["external_required_dependencies"]
    resolved = _parse_ldd_output(
        label,
        direct_needed,
        output,
        external_required_dependencies=declarations,
    )
    observed = {
        item["dependency"]
        for item in resolved
        if item.get("kind") == "external-required"
    }
    declared = {
        item["dependency"]
        for item in _external_required_by_dependency(declarations).values()
    }
    if observed != declared:
        raise ValueError(
            "dependency preflight external-required set differs from declaration: "
            f"expected={sorted(declared)}, observed={sorted(observed)}"
        )
    return {
        "schema_version": 1,
        "kind": "ucm-external-dependency-preflight",
        "spec_id": spec_id,
        "binary": label,
        "external_required_dependencies": declarations,
        "resolved_dependencies": resolved,
        "unexpected_unresolved": [],
        "status": "passed",
    }


def preflight_dependencies(
    binary: Path,
    spec_id: str,
    *,
    task: dict[str, Any] | None = None,
    task_path: Path | None = None,
) -> dict[str, Any]:
    """Inspect a builder library and fail on any undeclared unresolved dependency."""
    task = _selected_wheel_task(spec_id, task=task, task_path=task_path)
    path = Path(binary)
    if not path.is_file():
        raise ValueError(f"dependency preflight binary is missing: {path}")
    dynamic = subprocess.run(
        ["readelf", "-d", str(path)], text=True, capture_output=True, check=False
    )
    if dynamic.returncode != 0:
        raise ValueError(f"readelf failed for dependency preflight: {path}")
    needed_pattern = re.compile(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]$")
    direct_needed = sorted(
        match.group(1)
        for line in dynamic.stdout.splitlines()
        if (match := needed_pattern.search(line.strip())) is not None
    )
    if not direct_needed:
        raise ValueError(f"dependency preflight found no DT_NEEDED entries: {path}")
    linked = subprocess.run(
        ["ldd", str(path)], text=True, capture_output=True, check=False
    )
    if linked.returncode != 0:
        raise ValueError(f"ldd failed for dependency preflight: {path}")
    return validate_preflight_ldd(
        task,
        str(path),
        direct_needed,
        linked.stdout,
    )


def audit_dependency_closure(
    path: Path,
    output: Path,
    spec_id: str,
    authority_path: Path,
    *,
    task_path: Path,
) -> dict[str, Any]:
    """Resolve every DT_NEEDED entry under Linux with wheel directories visible."""
    task = _selected_wheel_task(spec_id, task_path=task_path)
    spec = _wheel_spec_from_task(task)
    if sys.platform != "linux":
        raise ValueError("dependency closure audit requires Linux")
    authority = _validate_build_authority(
        _canonical_record(authority_path, "build authority"),
        spec_id,
        task,
    )
    raw_digest = _sha256(Path(path))
    with zipfile.ZipFile(path) as archive:
        native = _verify_native_evidence(archive, spec, task)
        with tempfile.TemporaryDirectory(prefix="ucm-wheel-closure-") as temporary:
            root = Path(temporary)
            for name in archive.namelist():
                if not _safe_wheel_name(name):
                    raise ValueError("dependency closure wheel has unsafe members")
                archive.extract(name, root)
            loader_dirs = sorted(
                {str((root / name).parent) for name in native["native_artifacts"]}
            )
            environment = {
                **os.environ,
                "LD_LIBRARY_PATH": ":".join(
                    [*loader_dirs, os.environ.get("LD_LIBRARY_PATH", "")]
                ).rstrip(":"),
            }
            members: dict[str, Any] = {}
            extracted_by_name = {
                PurePosixPath(name).name: (name, root / name)
                for name in native["native_artifacts"]
            }
            for name in native["native_artifacts"]:
                library = root / name
                readelf = subprocess.run(
                    ["readelf", "-h", "-d", str(library)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if readelf.returncode != 0:
                    raise ValueError(f"readelf failed for dependency closure: {name}")
                linked = subprocess.run(
                    ["ldd", str(library)],
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if linked.returncode != 0:
                    raise ValueError(f"ldd failed for dependency closure: {name}")
                resolved_dependencies: list[dict[str, Any]] = []
                for parsed in _parse_ldd_output(
                    name,
                    native["dt_needed"][name],
                    linked.stdout,
                    external_required_dependencies=spec[
                        "external_required_dependencies"
                    ],
                ):
                    if parsed["kind"] in {"virtual", "external-required"}:
                        resolved_dependencies.append(parsed)
                        continue
                    dependency = parsed["dependency"]
                    resolved = Path(parsed["path"]).resolve()
                    internal = extracted_by_name.get(PurePosixPath(dependency).name)
                    if internal is not None and resolved == internal[1].resolve():
                        resolved_dependencies.append(
                            {
                                "dependency": dependency,
                                "direct": parsed["direct"],
                                "kind": "wheel-member",
                                "member": internal[0],
                                "sha256": _sha256(internal[1]),
                            }
                        )
                    elif resolved.is_file():
                        resolved_dependencies.append(
                            {
                                "dependency": dependency,
                                "direct": parsed["direct"],
                                "kind": "external",
                                "path": str(resolved),
                                "sha256": _sha256(resolved),
                            }
                        )
                    else:
                        raise ValueError(
                            f"ldd resolved dependency is not a file for {name}: "
                            f"{dependency} => {resolved}"
                        )
                members[name] = {
                    "dt_needed": native["dt_needed"][name],
                    "resolved_dependencies": resolved_dependencies,
                    "unresolved_dependencies": [],
                }
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": CLOSURE_KIND,
        "spec_id": spec_id,
        "raw_wheel_sha256": raw_digest,
        "build_context_sha256": authority["build_context_sha256"],
        "native_members": members,
        "unresolved_dependencies": [],
    }
    record["closure_sha256"] = (
        "sha256:" + hashlib.sha256(canonical_bytes(record)).hexdigest()
    )
    with zipfile.ZipFile(path) as archive:
        _validate_dependency_closure(
            record,
            raw_digest,
            authority,
            native,
            archive,
            external_required_dependencies=spec["external_required_dependencies"],
        )
    Path(output).write_bytes(canonical_bytes(record) + b"\n")
    return record


def _verify_builder_candidate_evidence(
    archive: zipfile.ZipFile,
    wheel_metadata: email.message.Message,
    spec: dict[str, Any],
    profile_id: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    names = [item.filename for item in archive.infolist() if not item.is_dir()]
    if FIXTURE_MARKER in names:
        raise ValueError("builder candidate must not contain a fixture binding marker")
    record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
    if len(record_names) != 1:
        raise ValueError("builder candidate requires exactly one RECORD")
    _verify_record(archive, record_names[0])
    if wheel_metadata.get("Root-Is-Purelib", "").lower() != "false":
        raise ValueError("builder candidate must declare Root-Is-Purelib: false")
    build_names = [name for name in names if name.endswith(".dist-info/ucm-build.json")]
    if len(build_names) != 1:
        raise ValueError(
            "builder candidate requires exactly one embedded ucm-build.json"
        )
    binding = _unique_json(archive.read(build_names[0]), build_names[0])
    authority_names = [
        name for name in names if name.endswith(".dist-info/ucm-build-authority.json")
    ]
    closure_names = [
        name
        for name in names
        if name.endswith(".dist-info/ucm-dependency-closure.json")
    ]
    if len(authority_names) != 1 or len(closure_names) != 1:
        raise ValueError(
            "builder candidate requires one build authority and dependency closure"
        )
    authority = _validate_build_authority(
        _unique_json(archive.read(authority_names[0]), authority_names[0]),
        spec["spec_id"],
        task,
    )
    if archive.read(authority_names[0]) != canonical_bytes(authority) + b"\n":
        raise ValueError("embedded build authority is noncanonical")
    required = {
        "schema_version",
        "task_id",
        "spec_id",
        "kind",
        "source_kind",
        "profile_id",
        "build",
        "source_sha",
        "build_key",
        "build_context_sha256",
        "source_tree",
        "source_archive_sha256",
        "builder_coordinate",
        "builder_config_digest",
        "dependency_lock_sha256",
        "tool_wheels",
        "source_date_epoch",
        "accelerator",
        "accelerator_runtime",
        "npu_arch_or_na",
        "os",
        "cpu_arch",
        "python_abi",
        "python_version",
        "binary_profile_id",
        "dist_name",
        "wheel_version",
        "wheel_platform",
        "required_native",
        "forbidden_native",
        "allowed_dt_needed",
        "external_required_dependencies",
        "native_members",
        "component_manifest_sha256",
        "dependency_closure_sha256",
        "build_authority_sha256",
    }
    if set(binding) != required:
        raise ValueError(
            "embedded build binding fields mismatch: "
            f"missing={sorted(required - set(binding))}, extra={sorted(set(binding) - required)}"
        )
    if (
        binding["schema_version"] != 1
        or binding["kind"] != "ucm-native-wheel-build"
        or binding["source_kind"] != "builder-candidate"
    ):
        raise ValueError("embedded build binding identity is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", str(binding["source_sha"])) is None:
        raise ValueError("embedded build binding requires immutable source_commit")
    if binding["build_key"] != task["task_sha256"]:
        raise ValueError("embedded build binding build_key is not canonical")
    if (
        not isinstance(binding["source_date_epoch"], int)
        or isinstance(binding["source_date_epoch"], bool)
        or binding["source_date_epoch"] < 315532800
    ):
        raise ValueError("embedded build binding SOURCE_DATE_EPOCH is invalid")
    bound_fields = (
        "spec_id",
        "accelerator",
        "accelerator_runtime",
        "npu_arch_or_na",
        "os",
        "cpu_arch",
        "python_abi",
        "binary_profile_id",
        "dist_name",
        "wheel_version",
        "wheel_platform",
        "required_native",
        "forbidden_native",
        "allowed_dt_needed",
        "external_required_dependencies",
    )
    for field in bound_fields:
        if binding[field] != spec[field]:
            raise ValueError(
                f"embedded build binding {field} does not match planned spec: "
                f"{binding[field]!r} != {spec[field]!r}"
            )
    if binding["profile_id"] != profile_id:
        raise ValueError(
            "embedded build binding profile_id does not match planned spec"
        )
    for field in ("task_id", "build", "python_version"):
        if binding[field] != task[field]:
            raise ValueError(f"embedded build binding {field} differs from wheel task")
    native = _verify_native_evidence(archive, spec, task)
    closure_raw = archive.read(closure_names[0])
    closure_value = _unique_json(closure_raw, closure_names[0])
    if closure_raw != canonical_bytes(closure_value) + b"\n":
        raise ValueError("embedded dependency closure is noncanonical")
    closure = _validate_dependency_closure(
        closure_value,
        closure_value["raw_wheel_sha256"],
        authority,
        native,
        archive,
        external_required_dependencies=spec["external_required_dependencies"],
    )
    _, component_digest = _verify_component_manifest(
        archive,
        spec,
        profile_id,
        binding["source_sha"],
        binding["build_key"],
    )
    if binding["component_manifest_sha256"] != component_digest:
        raise ValueError("embedded component manifest digest mismatch")
    if binding["native_members"] != native["native_members"]:
        raise ValueError("embedded native member map does not match wheel bytes")
    authority_digest = (
        "sha256:" + hashlib.sha256(archive.read(authority_names[0])).hexdigest()
    )
    bound_authority = {
        "source_sha": authority["source_sha"],
        "build_key": authority["task_sha256"],
        "build_context_sha256": authority["build_context_sha256"],
        "source_tree": authority["source_tree"],
        "source_archive_sha256": authority["source_archive_sha256"],
        "builder_coordinate": authority["builder_coordinate"],
        "builder_config_digest": authority["builder_config_digest"],
        "dependency_lock_sha256": authority["dependency_lock_sha256"],
        "tool_wheels": authority["tool_wheels"],
        "build_authority_sha256": authority_digest,
        "dependency_closure_sha256": closure["closure_sha256"],
    }
    for field, value in bound_authority.items():
        if binding[field] != value:
            raise ValueError(f"embedded build binding {field} differs from authority")
    if archive.read(build_names[0]) != canonical_bytes(binding) + b"\n":
        raise ValueError("embedded build binding is noncanonical")
    return {
        "source_commit": binding["source_sha"],
        "build_context_digest": binding["build_context_sha256"],
        "build_key": binding["build_key"],
        "source_date_epoch": binding["source_date_epoch"],
        **{**native, "unresolved_dependencies": closure["unresolved_dependencies"]},
        "record_status": "passed",
    }


def _dist_filename_component(dist_name: str) -> str:
    """PEP 427 wheel filename / dist-info directory component for a dist_name.

    The catalog grammar restricts dist_name to hyphen-separated lowercase tokens,
    so collapsing hyphens to underscores matches setuptools' safe_name output.
    """
    return dist_name.replace("-", "_")


def _canonical_metadata(dist_name: str, version: str, dependencies: list[str]) -> bytes:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {dist_name}",
        f"Version: {version}",
        "Summary: Unified Cache Management",
        "Requires-Python: >=3.10",
        *(f"Requires-Dist: {dependency}" for dependency in dependencies),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _canonical_wheel_metadata(tag: str) -> bytes:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: ucm-release-sealer-v1",
            "Root-Is-Purelib: false",
            f"Tag: {tag}",
            "",
        ]
    ).encode("utf-8")


def _record_bytes(members: dict[str, bytes], record_name: str) -> bytes:
    rows: list[list[str]] = []
    for name in sorted(members):
        content = members[name]
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .decode("ascii")
            .rstrip("=")
        )
        rows.append([name, f"sha256={digest}", str(len(content))])
    rows.append([record_name, "", ""])
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _zip_timestamp(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    if (
        not isinstance(source_date_epoch, int)
        or isinstance(source_date_epoch, bool)
        or not 315532800 <= source_date_epoch <= 4354819199
    ):
        raise ValueError("SOURCE_DATE_EPOCH must fit the canonical ZIP timestamp range")
    values = list(time.gmtime(source_date_epoch)[:6])
    values[5] -= values[5] % 2
    return tuple(values)  # type: ignore[return-value]


def _check_member_path_leakage(name: str, data: bytes) -> None:
    if any(marker in data for marker in HOST_PATH_MARKERS):
        raise ValueError(f"source/path leakage detected in wheel member: {name}")


def _expected_wheel_tag(spec: dict[str, Any]) -> str:
    architecture = cpu_toolchain_authority(spec["cpu_arch"]).wheel_arch
    return (
        f"{spec['python_abi']}-{spec['python_abi']}-"
        f"{spec['wheel_platform']}_{architecture}"
    )


def _verify_canonical_builder_archive(
    archive: zipfile.ZipFile,
    binding: dict[str, Any],
    metadata_name: str,
    wheel_name: str,
    record_name: str,
    dependencies: list[str],
) -> None:
    infos = archive.infolist()
    names = [item.filename for item in infos]
    if names != sorted(names):
        raise ValueError("sealed wheel member order is noncanonical")
    expected_timestamp = _zip_timestamp(binding["source_date_epoch"])
    for item in infos:
        if item.is_dir():
            raise ValueError("sealed wheel must not contain directory entries")
        if item.date_time != expected_timestamp:
            raise ValueError(f"sealed wheel timestamp is noncanonical: {item.filename}")
        if item.create_system != 3 or item.external_attr >> 16 != 0o644:
            raise ValueError(f"sealed wheel mode is noncanonical: {item.filename}")
        if item.compress_type != zipfile.ZIP_DEFLATED:
            raise ValueError(
                f"sealed wheel compression is noncanonical: {item.filename}"
            )
        if item.extra or item.comment:
            raise ValueError(
                f"sealed wheel ZIP metadata is noncanonical: {item.filename}"
            )
    if archive.comment:
        raise ValueError("sealed wheel ZIP comment is forbidden")
    tag = _expected_wheel_tag(binding)
    if archive.read(metadata_name) != _canonical_metadata(
        binding["dist_name"], binding["wheel_version"], dependencies
    ):
        raise ValueError("sealed wheel METADATA bytes are noncanonical")
    if archive.read(wheel_name) != _canonical_wheel_metadata(tag):
        raise ValueError("sealed wheel WHEEL bytes are noncanonical")
    expected_record = _record_bytes(
        {name: archive.read(name) for name in names if name != record_name},
        record_name,
    )
    if archive.read(record_name) != expected_record:
        raise ValueError("sealed wheel RECORD bytes are noncanonical")


def seal_wheel(
    path: Path,
    output_dir: Path,
    spec_id: str,
    source_sha: str,
    build_key: str,
    source_date_epoch: int,
    authority_path: Path,
    dependency_closure_path: Path,
    *,
    task_path: Path,
) -> dict[str, Any]:
    """Seal one native builder output into the sole deterministic candidate wheel."""
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("release wheel source SHA must be a full lowercase Git commit")
    if DIGEST_RE.fullmatch(build_key) is None:
        raise ValueError("release wheel build key must be sha256:<64 lowercase hex>")
    task = _selected_wheel_task(spec_id, task_path=task_path)
    spec = _wheel_spec_from_task(task)
    timestamp = _zip_timestamp(source_date_epoch)
    profile_id = task["profile_id"]
    authority = _validate_build_authority(
        _canonical_record(authority_path, "build authority"),
        spec_id,
        task,
    )
    if source_sha != authority["source_sha"]:
        raise ValueError("release wheel source SHA differs from source authority")
    if build_key != authority["task_sha256"]:
        raise ValueError(
            f"release wheel build key does not match exact task authority for {spec_id}"
        )
    if source_date_epoch != authority["source_date_epoch"]:
        raise ValueError("release wheel epoch differs from build authority")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("sealed wheel output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = Path(path).read_bytes()
    raw_wheel_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
    end_record = raw.rfind(b"PK\x05\x06")
    if end_record < 0 or end_record + 22 > len(raw):
        raise ValueError("input wheel ZIP end record is missing")
    comment_size = int.from_bytes(raw[end_record + 20 : end_record + 22], "little")
    if end_record + 22 + comment_size != len(raw):
        raise ValueError("input wheel contains trailing bytes")
    try:
        with zipfile.ZipFile(path) as integrity_archive:
            bad_member = integrity_archive.testzip()
    except (zipfile.BadZipFile, zlib.error) as error:
        raise ValueError(f"input wheel ZIP is corrupt: {error}") from error
    if bad_member is not None:
        raise ValueError(f"input wheel CRC is corrupt: {bad_member}")
    with zipfile.ZipFile(path) as archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or any(
            not _safe_wheel_name(name) for name in names
        ):
            raise ValueError("input wheel contains duplicate or unsafe members")
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ValueError("input wheel requires exactly one METADATA and WHEEL")
        metadata_name = metadata_names[0]
        dist_info = metadata_name.removesuffix("/METADATA")
        expected_dist_info = f"{_dist_filename_component(spec['dist_name'])}-{spec['wheel_version']}.dist-info"  # fmt: skip  # noqa: E501
        if dist_info != expected_dist_info or wheel_names[0] != f"{dist_info}/WHEEL":
            raise ValueError(
                "input wheel dist-info path does not match controlled version"
            )
        metadata = email.parser.Parser().parsestr(
            archive.read(metadata_name).decode("utf-8")
        )
        if canonicalize_name(metadata.get("Name", "")) != canonicalize_name(spec["dist_name"]):  # fmt: skip  # noqa: E501
            raise ValueError(f"input wheel distribution must be {spec['dist_name']}")  # fmt: skip  # noqa: E501
        if metadata.get("Version") != spec["wheel_version"]:
            raise ValueError("input wheel version is not the controlled local version")
        if metadata.get_all("Requires-Dist", []) != task["runtime_requirements"]:
            raise ValueError(
                "input wheel runtime dependencies do not match release.yaml"
            )
        _, component_digest = _verify_component_manifest(
            archive, spec, profile_id, source_sha, build_key
        )
        native = _verify_native_evidence(archive, spec, task)
        closure = _validate_dependency_closure(
            _canonical_record(dependency_closure_path, "Linux dependency closure"),
            raw_wheel_sha256,
            authority,
            native,
            archive,
            external_required_dependencies=spec["external_required_dependencies"],
        )
        members = {
            name: archive.read(name)
            for name in names
            if not name.endswith(".dist-info/METADATA")
            and not name.endswith(".dist-info/WHEEL")
            and not name.endswith(".dist-info/RECORD")
            and not name.endswith(".dist-info/RECORD.jws")
            and not name.endswith(".dist-info/RECORD.p7s")
            and not name.endswith(".dist-info/ucm-build.json")
        }
    for name, data in members.items():
        _check_member_path_leakage(name, data)
    tag = _expected_wheel_tag(spec)
    metadata_name = f"{expected_dist_info}/METADATA"
    wheel_name = f"{expected_dist_info}/WHEEL"
    build_name = f"{expected_dist_info}/ucm-build.json"
    authority_name = f"{expected_dist_info}/ucm-build-authority.json"
    closure_name = f"{expected_dist_info}/ucm-dependency-closure.json"
    record_name = f"{expected_dist_info}/RECORD"
    binding = {
        "schema_version": 1,
        "kind": "ucm-native-wheel-build",
        "source_kind": "builder-candidate",
        "task_id": task["task_id"],
        "profile_id": profile_id,
        "spec_id": spec_id,
        "build": task["build"],
        "source_sha": source_sha,
        "build_key": build_key,
        "build_context_sha256": authority["build_context_sha256"],
        "source_tree": authority["source_tree"],
        "source_archive_sha256": authority["source_archive_sha256"],
        "builder_coordinate": authority["builder_coordinate"],
        "builder_config_digest": authority["builder_config_digest"],
        "dependency_lock_sha256": authority["dependency_lock_sha256"],
        "tool_wheels": authority["tool_wheels"],
        "source_date_epoch": source_date_epoch,
        "accelerator": spec["accelerator"],
        "accelerator_runtime": spec["accelerator_runtime"],
        "npu_arch_or_na": spec["npu_arch_or_na"],
        "os": spec["os"],
        "cpu_arch": spec["cpu_arch"],
        "python_abi": spec["python_abi"],
        "python_version": task["python_version"],
        "binary_profile_id": spec["binary_profile_id"],
        "dist_name": spec["dist_name"],
        "wheel_version": spec["wheel_version"],
        "wheel_platform": spec["wheel_platform"],
        "required_native": spec["required_native"],
        "forbidden_native": spec["forbidden_native"],
        "allowed_dt_needed": spec["allowed_dt_needed"],
        "external_required_dependencies": spec["external_required_dependencies"],
        "native_members": native["native_members"],
        "component_manifest_sha256": component_digest,
        "dependency_closure_sha256": closure["closure_sha256"],
        "build_authority_sha256": "sha256:"
        + hashlib.sha256(canonical_bytes(authority) + b"\n").hexdigest(),
    }
    members[metadata_name] = _canonical_metadata(
        spec["dist_name"], spec["wheel_version"], task["runtime_requirements"]
    )
    members[wheel_name] = _canonical_wheel_metadata(tag)
    members[build_name] = canonical_bytes(binding) + b"\n"
    members[authority_name] = canonical_bytes(authority) + b"\n"
    members[closure_name] = canonical_bytes(closure) + b"\n"
    members[record_name] = _record_bytes(members, record_name)
    filename = f"{_dist_filename_component(spec['dist_name'])}-{spec['wheel_version']}-{tag}.whl"  # fmt: skip  # noqa: E501
    wheel_path = output_dir / filename
    with zipfile.ZipFile(
        wheel_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED)
    wheel_sha256 = _sha256(wheel_path)
    inspection = inspect_wheel(
        wheel_path,
        spec_id,
        wheel_sha256,
        "builder-candidate",
        task=task,
    )
    inspection_path = output_dir / "wheel-inspection.json"
    _write_canonical(inspection_path, inspection)
    result = {
        "schema_version": 1,
        "kind": "ucm-native-wheel-seal",
        "source_kind": "builder-candidate",
        "publication_status": "unpublished",
        "publication_eligible": False,
        "spec_id": spec_id,
        "source_sha": source_sha,
        "build_key": build_key,
        "wheel_path": str(wheel_path),
        "wheel_sha256": wheel_sha256,
        "inspection_path": str(inspection_path),
        "inspection_sha256": _sha256(inspection_path),
        "inspection": inspection,
    }
    _write_canonical(output_dir / "wheel-seal.json", result)
    return result


def _verify_fixture_binding(
    archive: zipfile.ZipFile, spec: dict[str, Any]
) -> dict[str, str]:
    """Parse the unique canonical fixture marker as literals without execution."""
    names = [item.filename for item in archive.infolist() if not item.is_dir()]
    if len(names) != len(set(names)) or any(
        not _safe_wheel_name(name) for name in names
    ):
        raise ValueError("fixture wheel contains duplicate or unsafe members")
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
    record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
    if len(metadata_names) != 1 or len(wheel_names) != 1 or len(record_names) != 1:
        raise ValueError("fixture wheel requires one METADATA, WHEEL, and RECORD")
    dist_info = record_names[0].removesuffix("/RECORD")
    expected_names = {
        "ucm/__init__.py",
        FIXTURE_MARKER,
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/RECORD",
    }
    if set(names) != expected_names or names.count(FIXTURE_MARKER) != 1:
        raise ValueError("fixture wheel requires exactly the canonical member set")
    _verify_record(archive, record_names[0])
    raw = archive.read(FIXTURE_MARKER)
    try:
        text = raw.decode("utf-8")
        module = ast.parse(text, filename=FIXTURE_MARKER, mode="exec")
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(f"fixture binding marker is invalid: {error}") from error
    values: dict[str, str] = {}
    for statement in module.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or not isinstance(statement.value, ast.Constant)
            or not isinstance(statement.value.value, str)
        ):
            raise ValueError("fixture binding accepts only literal assignments")
        name = statement.targets[0].id
        if name not in {"SOURCE_SHA", "PROFILE_ID"} or name in values:
            raise ValueError("fixture binding has duplicate or extra fields")
        values[name] = statement.value.value
    if set(values) != {"SOURCE_SHA", "PROFILE_ID"}:
        raise ValueError("fixture binding is missing required fields")
    source_sha = values["SOURCE_SHA"]
    profile_id = values["PROFILE_ID"]
    canonical = f"SOURCE_SHA = {source_sha!r}\nPROFILE_ID = {profile_id!r}\n"
    if text != canonical:
        raise ValueError("fixture binding marker bytes are noncanonical")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("fixture binding source SHA is invalid")
    if profile_id != spec["spec_id"]:
        raise ValueError("fixture binding profile does not match the planned spec")
    return {
        "source_commit": source_sha,
        "profile_id": profile_id,
        "marker_status": "passed",
    }


def inspect_wheel(
    path: Path,
    spec_id: str,
    expected_sha256: str,
    source_kind: str,
    *,
    task: dict[str, Any] | None = None,
    task_path: Path | None = None,
    release_path: Path = DEFAULT_RELEASE,
    schema_dir: Path = DEFAULT_SCHEMA_DIR,
) -> dict[str, Any]:
    if source_kind not in {"fixture", "builder-candidate"}:
        raise ValueError("source_kind must be fixture or builder-candidate")
    if DIGEST_RE.fullmatch(expected_sha256) is None:
        raise ValueError("expected SHA256 must be sha256:<64 lowercase hex>")
    release: dict[str, Any] | None = None
    selected_task: dict[str, Any] | None = None
    if source_kind == "builder-candidate":
        selected_task = _selected_wheel_task(spec_id, task=task, task_path=task_path)
        spec = _wheel_spec_from_task(selected_task)
        runtime_requirements = selected_task["runtime_requirements"]
        expected_version = selected_task["wheel_version"]
        profile_id = selected_task["profile_id"]
    else:
        if task is not None or task_path is not None:
            raise ValueError("fixture wheel inspection does not accept a real task")
        release = load_catalog(release_path, schema_dir)
        specs = {item["spec_id"]: item for item in _fixture_wheel_specs(release)}
        if spec_id not in specs:
            raise ValueError(f"unknown fixture wheel spec: {spec_id}")
        spec = specs[spec_id]
        runtime_requirements = python_runtime_requirements(release)
        expected_version = release["ucm_version"]
        profile_id = spec_id
    if source_kind == "builder-candidate" and not spec["build_eligible"]:
        raise ValueError(
            "builder candidate planned spec has unresolved locks or runner"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"wheel SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    raw_wheel = Path(path).read_bytes()
    end_record = raw_wheel.rfind(b"PK\x05\x06")
    if end_record < 0 or end_record + 22 > len(raw_wheel):
        raise ValueError("wheel ZIP end record is missing")
    comment_size = int.from_bytes(
        raw_wheel[end_record + 20 : end_record + 22], "little"
    )
    if end_record + 22 + comment_size != len(raw_wheel):
        raise ValueError("wheel contains trailing bytes after the ZIP end record")
    try:
        with zipfile.ZipFile(path) as integrity_archive:
            bad_member = integrity_archive.testzip()
    except (zipfile.BadZipFile, zlib.error) as error:
        raise ValueError(f"wheel ZIP is corrupt: {error}") from error
    if bad_member is not None:
        raise ValueError(f"wheel CRC is corrupt: {bad_member}")
    try:
        filename_name, filename_version, _, filename_tags = parse_wheel_filename(
            path.name
        )
    except Exception as error:
        raise ValueError(f"invalid wheel filename: {error}") from error
    builder_evidence: dict[str, Any] | None = None
    fixture_binding: dict[str, str] | None = None
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        wheel_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise ValueError(
                "wheel must contain exactly one METADATA and one WHEEL file"
            )
        parser = email.parser.Parser()
        metadata = parser.parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        wheel_metadata = parser.parsestr(archive.read(wheel_names[0]).decode("utf-8"))
        if source_kind == "builder-candidate":
            assert selected_task is not None
            builder_evidence = _verify_builder_candidate_evidence(
                archive, wheel_metadata, spec, profile_id, selected_task
            )
            build_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/ucm-build.json")
            )
            record_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/RECORD")
            )
            binding = _unique_json(archive.read(build_name), build_name)
            _verify_canonical_builder_archive(
                archive,
                binding,
                metadata_names[0],
                wheel_names[0],
                record_name,
                runtime_requirements,
            )
            for name in archive.namelist():
                if not name.endswith(".dist-info/RECORD"):
                    _check_member_path_leakage(name, archive.read(name))
        else:
            fixture_binding = _verify_fixture_binding(archive, spec)
    distribution = metadata.get("Name", "")
    version = metadata.get("Version", "")
    if canonicalize_name(distribution) != canonicalize_name(spec["dist_name"]):  # fmt: skip  # noqa: E501
        raise ValueError(f"unexpected wheel distribution: {distribution}")  # fmt: skip  # noqa: E501
    if canonicalize_name(distribution) != canonicalize_name(str(filename_name)):
        raise ValueError("METADATA distribution does not match wheel filename")
    if version != str(filename_version):
        raise ValueError("METADATA version does not match wheel filename")
    if version != expected_version:
        raise ValueError(
            f"wheel version {version} does not match planned version {expected_version}"
        )
    filename_tag_strings = {str(tag) for tag in filename_tags}
    metadata_tags = set(wheel_metadata.get_all("Tag", []))
    if not metadata_tags or metadata_tags != filename_tag_strings:
        raise ValueError("WHEEL tags do not match wheel filename tags")
    if source_kind == "builder-candidate":
        expected_tags = {_expected_wheel_tag(spec)}
    else:
        architecture = cpu_toolchain_authority(spec["cpu_arch"]).wheel_arch
        expected_tags = {
            f"{spec['python_abi']}-{spec['python_abi']}-linux_{architecture}"
        }
    if filename_tag_strings != expected_tags:
        raise ValueError(
            "wheel tags do not match declared Python ABI and CPU architecture"
        )
    requires_dist = metadata.get_all("Requires-Dist", [])
    if requires_dist != runtime_requirements:
        raise ValueError("wheel runtime dependencies do not match selected authority")
    result = {
        "schema_version": 1,
        "kind": "ucm-wheel-inspection",
        "source_kind": source_kind,
        "spec_id": spec_id,
        "filename": path.name,
        "sha256": actual_sha256,
        "size": path.stat().st_size,
        "distribution": distribution,
        "version": version,
        "tags": sorted(filename_tag_strings),
        "requires_dist": requires_dist,
        "python_abi": spec["python_abi"],
        "cpu_arch": spec["cpu_arch"],
        "declaration_sha256": spec["declaration_sha256"],
        "status": "fixture-only" if source_kind == "fixture" else "candidate-inspected",
        "trust_level": (
            "fixture-only"
            if source_kind == "fixture"
            else "unpublished-builder-candidate"
        ),
        "published": False,
        "publication_eligible": False,
    }
    if builder_evidence is not None:
        result["builder_evidence"] = builder_evidence
    if fixture_binding is not None:
        result["fixture_binding"] = fixture_binding
    return result
