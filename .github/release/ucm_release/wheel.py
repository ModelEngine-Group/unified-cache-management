"""Prepare current Wheel sources and record audited build results."""

from __future__ import annotations

import copy
import email.parser
import hashlib
import re
import tomllib
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename

from . import policy as release_policy
from . import wheel_audit

WHEEL_ARCHITECTURES = {"amd64": "x86_64", "arm64": "aarch64"}
AUDITWHEEL_REPORT = "auditwheel-show.txt"
_UCM_DISTRIBUTION = re.compile(r"(?:[a-z0-9]+-)*uc-manager(?:-[a-z0-9]+)*")


def prepare_wheel_source(source_root: Path, distribution: str) -> dict[str, str]:
    """Set the PEP 621 distribution name for one compact Wheel build."""
    if (
        len(distribution) > release_policy.MAX_PYPI_DISTRIBUTION_LENGTH
        or canonicalize_name(distribution, validate=True) != distribution
        or _UCM_DISTRIBUTION.fullmatch(distribution) is None
    ):
        raise ValueError("compact Wheel distribution name is invalid")
    path = source_root / "pyproject.toml"
    raw = path.read_text(encoding="utf-8")
    document = tomllib.loads(raw)
    if document.get("project", {}).get("name") != "uc-manager":
        raise ValueError("compact Wheel source must start with project.name=uc-manager")
    source = 'name = "uc-manager"'
    if raw.count(source) != 1:
        raise ValueError("compact Wheel project.name is missing or ambiguous")
    updated = raw.replace(source, f'name = "{distribution}"')
    if tomllib.loads(updated).get("project", {}).get("name") != distribution:
        raise ValueError("compact Wheel project.name update failed")
    path.write_text(updated, encoding="utf-8")
    return {"distribution": distribution, "project_file": "pyproject.toml"}


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected a mapping")
    return value


def _wheel_metadata_tags(wheel_path: Path) -> set[str]:
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            metadata_files = [
                name
                for name in wheel.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/WHEEL")
            ]
            if len(metadata_files) != 1:
                raise ValueError(
                    "built Wheel must contain exactly one WHEEL metadata file"
                )
            metadata = wheel.read(metadata_files[0]).decode("utf-8")
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise ValueError("built Wheel metadata cannot be read") from error

    tags: set[str] = set()
    for line in metadata.splitlines():
        if not line.startswith("Tag:"):
            continue
        raw_tag = line.removeprefix("Tag:").strip()
        try:
            tags.update(str(tag) for tag in parse_tag(raw_tag))
        except ValueError as error:
            raise ValueError("built Wheel metadata contains an invalid Tag") from error
    if not tags:
        raise ValueError("built Wheel metadata has no Tag")
    return tags


def _normalized_requirements(values: object, context: str) -> list[str]:
    if not isinstance(values, list) or any(
        not isinstance(item, str) for item in values
    ):
        raise ValueError(f"{context} must be a string list")
    normalized: list[str] = []
    for raw in values:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as error:
            raise ValueError(f"{context} contains an invalid requirement") from error
        rendered = str(requirement)
        normalized.append(
            canonicalize_name(requirement.name) + rendered[len(requirement.name) :]
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{context} contains duplicate requirements")
    return sorted(normalized)


def _wheel_dependencies(
    wheel_path: Path, *, distribution: str, version: str
) -> list[str]:
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            metadata_files = [
                name
                for name in wheel.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ValueError("built Wheel must contain exactly one METADATA file")
            metadata = email.parser.Parser().parsestr(
                wheel.read(metadata_files[0]).decode("utf-8")
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise ValueError("built Wheel METADATA cannot be read") from error

    metadata_distribution = metadata.get("Name", "")
    if canonicalize_name(metadata_distribution) != canonicalize_name(distribution):
        raise ValueError("built Wheel METADATA distribution does not match its task")
    if metadata.get("Version", "") != version:
        raise ValueError("built Wheel METADATA version does not match its task")
    return _normalized_requirements(
        metadata.get_all("Requires-Dist", []), "built Wheel METADATA dependencies"
    )


def _validate_wheel_platforms(
    platforms: set[str], architecture: str, target_platform: str
) -> str:
    try:
        wheel_architecture = WHEEL_ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(
            f"unsupported Wheel CPU architecture: {architecture}"
        ) from error
    if any(
        re.search(r"(?:^|_)(?:amd64|arm64)(?:_|$)", platform) for platform in platforms
    ):
        raise ValueError("built Wheel platform must use x86_64/aarch64, not OCI names")
    target_parts = _manylinux_parts(target_platform)
    if target_parts[2] != wheel_architecture:
        raise ValueError("Wheel target platform architecture does not match its task")
    if target_platform not in platforms:
        raise ValueError(f"built Wheel platform must include {target_platform}")
    for platform in platforms:
        major, minor, platform_architecture = _manylinux_parts(platform)
        if platform_architecture != wheel_architecture:
            raise ValueError("built Wheel contains a mismatched manylinux architecture")
        if (major, minor) > target_parts[:2]:
            raise ValueError("built Wheel contains a platform newer than its target")
    return wheel_architecture


def _auditwheel_result(
    wheel_path: Path,
    architecture: str,
    target_platform: str,
    expected_external_patterns: list[str],
    distribution: str,
    report_path: Path | None,
) -> dict[str, Any]:
    report = report_path or wheel_path.with_name(AUDITWHEEL_REPORT)
    if not report.is_file():
        raise ValueError(f"auditwheel report is missing: {report}")
    raw_bytes = report.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("auditwheel report is not UTF-8") from error
    if not text.strip():
        raise ValueError("auditwheel report is empty")

    platform_matches = re.findall(
        r"(?m)^([^\s]+\.whl)\s+is\s+consistent\s+with\s+the\s+following\s+"
        r'platform\s+tag:\s+"([^"]+)"',
        text,
    )
    if len(platform_matches) != 1:
        raise ValueError("auditwheel report has no compatible platform tag")
    reported_filename, compatible_platform = platform_matches[0]
    if reported_filename != wheel_path.name:
        raise ValueError("auditwheel report belongs to a different Wheel")
    wheel_architecture = WHEEL_ARCHITECTURES[architecture]
    if re.search(r"(?:^|_)(?:amd64|arm64)(?:_|$)", compatible_platform):
        raise ValueError("auditwheel platform must use x86_64/aarch64")
    if not compatible_platform.endswith(f"_{wheel_architecture}"):
        raise ValueError("auditwheel platform architecture does not match its task")

    constrained_matches = re.findall(
        r'This constrains the platform tag to "([^"]+)"', text
    )
    if len(constrained_matches) > 1:
        raise ValueError("auditwheel report has ambiguous ABI platform constraints")
    if constrained_matches:
        abi_compatible_platform = constrained_matches[0]
    elif compatible_platform.startswith("manylinux_"):
        abi_compatible_platform = compatible_platform
    else:
        raise ValueError("auditwheel report has no manylinux ABI compatibility tag")
    abi_parts = _manylinux_parts(abi_compatible_platform)
    target_parts = _manylinux_parts(target_platform)
    if abi_parts[2] != wheel_architecture or abi_parts[:2] > target_parts[:2]:
        raise ValueError("auditwheel ABI platform exceeds the Wheel target")

    glibc_versions = sorted(
        {
            f"GLIBC_{version}"
            for version in re.findall(r"\bGLIBC_(\d+(?:\.\d+)+)\b", text)
        },
        key=lambda value: tuple(
            int(part) for part in value.removeprefix("GLIBC_").split(".")
        ),
    )
    policy_deferred = wheel_audit.deferred_policy_for_distribution(distribution)
    external_closure = wheel_audit.validate_external_library_closure(
        text,
        expected_patterns=expected_external_patterns,
        deferred_sonames=policy_deferred,
    )

    return {
        "auditwheel_report": {
            "filename": report.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "text": text,
        },
        "auditwheel_platform_tag": compatible_platform,
        "abi_compatible_platform_tag": abi_compatible_platform,
        "glibc_versions": glibc_versions,
        "glibc_floor": glibc_versions[-1] if glibc_versions else None,
        **external_closure,
    }


def record_wheel_result(
    task: Mapping[str, Any],
    wheel_path: Path,
    auditwheel_report_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one built Wheel and record its auditwheel-visible coordinates."""
    if not wheel_path.is_file():
        raise ValueError(f"Wheel result is missing: {wheel_path}")
    try:
        distribution, version, _build, tags = parse_wheel_filename(wheel_path.name)
    except ValueError as error:
        raise ValueError(f"invalid Wheel filename {wheel_path.name!r}") from error
    if canonicalize_name(str(distribution)) != canonicalize_name(
        str(task["dist_name"])
    ):
        raise ValueError("built Wheel distribution does not match its task")
    if str(version) != str(task["wheel_version"]):
        raise ValueError("built Wheel version does not match its task")
    python_abi = str(task["python_abi"])
    abi_pairs = {(tag.interpreter, tag.abi) for tag in tags}
    if abi_pairs != {(python_abi, python_abi)}:
        raise ValueError("built Wheel ABI does not match its task")
    architecture = str(task["cpu_arch"])
    platform_tags = {tag.platform for tag in tags}
    target_platform = str(task["target_platform_tag"])
    _validate_wheel_platforms(platform_tags, architecture, target_platform)
    filename_tags = {str(tag) for tag in tags}
    if _wheel_metadata_tags(wheel_path) != filename_tags:
        raise ValueError("built Wheel filename and WHEEL metadata Tags do not match")
    dependencies = _wheel_dependencies(
        wheel_path,
        distribution=str(task["dist_name"]),
        version=str(task["wheel_version"]),
    )
    expected_dependencies = _normalized_requirements(
        task.get("runtime_requirements"), "Wheel task runtime requirements"
    )
    if dependencies != expected_dependencies:
        raise ValueError("built Wheel dependencies do not match its task")
    result = {
        "kind": "ucm-wheel-result",
        "schema_version": 5,
        "task_id": task["id"],
        "distribution": task["dist_name"],
        "version": task["wheel_version"],
        "python_abi": python_abi,
        "cpu_arch": architecture,
        "filename": wheel_path.name,
        "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
        "platform_tags": sorted(platform_tags),
        "repair": copy.deepcopy(task["repair"]),
        "dependencies": dependencies,
    }
    result.update(
        _auditwheel_result(
            wheel_path,
            architecture,
            target_platform,
            sorted(str(item) for item in task["external_runtime_exclude_patterns"]),
            str(task["dist_name"]),
            auditwheel_report_path,
        )
    )
    return result


def _manylinux_parts(platform: str) -> tuple[int, int, str]:
    match = re.fullmatch(
        r"manylinux_(?P<major>[0-9]+)_(?P<minor>[0-9]+)_(?P<arch>x86_64|aarch64)",
        platform,
    )
    if match is None:
        raise ValueError(f"invalid manylinux platform tag: {platform!r}")
    return int(match.group("major")), int(match.group("minor")), match.group("arch")
