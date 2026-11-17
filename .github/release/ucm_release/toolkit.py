"""Own the optional toolkit package identity and its built Wheel contract."""

from __future__ import annotations

import email.parser
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name, parse_wheel_filename

from . import policy


def package_identity(repository: str, version: str) -> dict[str, str]:
    return {
        "distribution": f"{policy.pypi_distribution_prefix(repository)}ucm-toolkit",
        "version": version,
    }


def planned_package(plan: dict[str, Any]) -> dict[str, str] | None:
    package = plan.get("toolkit_package")
    if package is None:
        return None
    expected = package_identity(plan["repository"], plan["version"])
    if package != expected:
        raise ValueError("toolkit package identity differs from the release plan")
    return expected


def prepare_source(plan: dict[str, Any], source_root: Path) -> dict[str, str]:
    package = planned_package(plan)
    if package is None:
        raise ValueError("release plan has no toolkit package")
    path = source_root / "toolkit/pyproject.toml"
    text = path.read_text()
    source = 'name = "ucm-toolkit"'
    if text.count(source) != 1:
        raise ValueError("toolkit source must start with project.name=ucm-toolkit")
    path.write_text(text.replace(source, f'name = "{package["distribution"]}"'))
    return package


def record_result(plan: dict[str, Any], wheel: Path) -> dict[str, Any]:
    package = planned_package(plan)
    if package is None:
        raise ValueError("release plan has no toolkit package")
    distribution, version, build, tags = parse_wheel_filename(wheel.name)
    if (
        canonicalize_name(distribution) != package["distribution"]
        or str(version) != package["version"]
        or build
        or {str(tag) for tag in tags} != {"py3-none-any"}
    ):
        raise ValueError("toolkit Wheel coordinates differ from the release plan")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError("toolkit Wheel must contain one METADATA file")
        metadata = email.parser.BytesParser().parsebytes(
            archive.read(metadata_names[0])
        )
        if (
            canonicalize_name(metadata["Name"]) != package["distribution"]
            or metadata["Version"] != package["version"]
            or metadata.get_all("Requires-Dist", [])
        ):
            raise ValueError(
                "toolkit Wheel metadata differs from its lightweight contract"
            )
        prefix = "ucm_toolkit/"
        required = {
            prefix + "_version.py",
            prefix + "_resources/nic_monitor_pro.sh",
            prefix + "_resources/posixstore_aio_test.py",
            prefix + "_resources/dev-sandbox/CMakeLists.txt",
            prefix + "_resources/dev-sandbox/vendor/fmt/CMakeLists.txt",
            prefix + "_resources/dev-sandbox/vendor/fmt/LICENSE",
            prefix + "tools/precheck/precheck.defaults.json",
            prefix + "tools/metrics_view/configs/connector.json",
        }
        if missing := required - names:
            raise ValueError(f"toolkit Wheel is missing resources: {sorted(missing)}")
        if any(re.search(r"\.(?:so|dylib|dll|o|a|pyc)$", name) for name in names):
            raise ValueError("toolkit Wheel must not contain compiled files")
        entrypoints = archive.read(
            metadata_names[0].replace("METADATA", "entry_points.txt")
        )
        if b"ucm-toolkit = ucm_toolkit.cli:main" not in entrypoints:
            raise ValueError("toolkit Wheel has no CLI entry point")
    return {
        "kind": "ucm-toolkit-result",
        "schema_version": 1,
        **package,
        "filename": wheel.name,
        "sha256": "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }


def load_artifact(
    plan: dict[str, Any], root: Path | None
) -> tuple[dict[str, Any] | None, Path | None]:
    package = planned_package(plan)
    if package is None:
        if root is not None:
            raise ValueError("toolkit artifact supplied without a planned package")
        return None, None
    if root is None:
        raise ValueError("planned toolkit package has no artifact directory")
    results = list(root.rglob("toolkit-result.json"))
    wheels = list(root.rglob("*.whl"))
    if len(results) != 1 or len(wheels) != 1:
        raise ValueError("toolkit artifact must contain one result and one Wheel")
    result = json.loads(results[0].read_text())
    if result != record_result(plan, wheels[0]):
        raise ValueError("toolkit result does not match its built Wheel")
    return result, wheels[0]


def publication_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": result["distribution"],
        "version": result["version"],
        "role": "toolkit",
        "files": [{"filename": result["filename"], "sha256": result["sha256"]}],
    }


def validate_result(package: dict[str, str], result: dict[str, Any] | None) -> None:
    if not isinstance(result, dict) or set(result) != {
        "kind",
        "schema_version",
        "distribution",
        "version",
        "filename",
        "sha256",
    }:
        raise ValueError("toolkit result has an invalid contract")
    if (
        result["kind"] != "ucm-toolkit-result"
        or result["schema_version"] != 1
        or {key: result[key] for key in package} != package
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(result["sha256"])) is None
    ):
        raise ValueError("toolkit result differs from the planned package")
    name, version, build, tags = parse_wheel_filename(result["filename"])
    if (
        canonicalize_name(name) != package["distribution"]
        or str(version) != package["version"]
        or build
        or {str(tag) for tag in tags} != {"py3-none-any"}
    ):
        raise ValueError("toolkit result has invalid Wheel coordinates")
