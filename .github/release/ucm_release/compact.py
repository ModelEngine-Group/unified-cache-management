"""Compact planning shared by formal Release and inspected PR runtimes."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tomllib
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from . import builders
from . import runtime as runtime_ops
from . import upstream

ROUTES = frozenset({"pr", "daily", "release"})
WHEEL_ARCHITECTURES = {"amd64": "x86_64", "arm64": "aarch64"}
AUDITWHEEL_REPORT = "auditwheel-show.txt"


def prepare_wheel_source(source_root: Path, distribution: str) -> dict[str, str]:
    """Set the PEP 621 distribution name for one compact Wheel build."""
    if re.fullmatch(r"uc-manager(?:-[a-z0-9]+)*", distribution) is None:
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


def _backend(catalog: Mapping[str, Any], backend: str) -> dict[str, Any]:
    backends = catalog.get("backends")
    value = backends.get(backend) if isinstance(backends, dict) else None
    if value is None:
        return {
            "status": "blocked",
            "reason": f"{backend} has no UCM native backend policy",
        }
    if not isinstance(value, dict):
        raise ValueError(f"upstream backend {backend!r} platform policy is malformed")
    return value


def _distribution(backend: Mapping[str, Any], runtime_variant: str) -> str:
    exact = backend.get("distribution")
    template = backend.get("distribution_template")
    if isinstance(exact, str) and exact:
        value = exact
    elif isinstance(template, str) and template.count("{runtime_variant}") == 1:
        value = template.replace("{runtime_variant}", runtime_variant)
    else:
        raise ValueError("backend policy has no valid distribution rule")
    if re.fullmatch(r"uc-manager(?:-[a-z0-9]+)*", value) is None:
        raise ValueError("backend policy generated an invalid distribution")
    return value


def _runtime_label(runtime: Mapping[str, Any]) -> str:
    accelerator = str(runtime["accelerator_runtime"])
    os_label = f"{runtime['os_id']} {runtime['os_version']}"
    product = "vLLM Ascend" if runtime["product_id"] == "vllm-ascend" else "vLLM"
    prefix = f"{product} {runtime['version']}"
    if accelerator.startswith("cuda-"):
        return f"{prefix} · CUDA {accelerator.removeprefix('cuda-')} · {os_label}"
    return (
        f"{prefix} · CANN {accelerator.removeprefix('cann-')} "
        f"{str(runtime['variant']).upper()} · {os_label}"
    )


def _builder_map(catalog: object) -> dict[str, dict[str, Any]]:
    validated = builders.validate_catalog(catalog)
    if validated.get("schema_version") != 4:
        raise ValueError("Registry-driven planning requires Builder Catalog schema 4")
    result: dict[str, dict[str, Any]] = {}
    for raw in validated["builders"]:  # type: ignore[index]
        item = _mapping(raw, "Builder Catalog item")
        result[str(item["id"])] = item
    return result


def _build_map(selection: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in selection["wheel_builds"]:  # type: ignore[index]
        item = _mapping(raw, "Wheel build")
        result[str(item["id"])] = item
    return result


def _validate_builder_matches_build(
    build: Mapping[str, Any], builder: Mapping[str, Any]
) -> None:
    for field in (
        "id",
        "build_group",
        "runtime_variant",
        "backend",
        "accelerator",
        "accelerator_runtime",
        "variant",
        "soc_version",
        "python_version",
        "python_abi",
        "manylinux",
        "cpu_arch",
        "source_image",
        "source_image_digest",
        "build_mode",
        "recipe_revision",
        "sync_mode",
    ):
        if builder.get(field) != build.get(field):
            raise ValueError(
                f"{build['id']}: Builder {field} does not match Wheel capability"
            )


def _wheel_task(
    catalog: Mapping[str, Any],
    build: Mapping[str, Any],
    builder: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> dict[str, Any]:
    architecture = str(build["cpu_arch"])
    requirements = _mapping(catalog.get("requirements"), "formal requirements")
    return {
        "id": str(build["id"]),
        "label": (f"{build['build_group']} · {build['python_abi']} · {architecture}"),
        "runner": catalog["runners"][architecture],  # type: ignore[index]
        "profile_id": str(build["build_group"]),
        "group_id": str(build["build_group"]),
        "backend": str(build["backend"]),
        "runtime_variant": str(build["runtime_variant"]),
        "cpu_arch": architecture,
        "platform": f"linux/{architecture}",
        "python_version": str(build["python_version"]),
        "python_abi": str(build["python_abi"]),
        "manylinux": str(build["manylinux"]),
        "wheel_version": catalog["ucm_version"],
        "dist_name": _distribution(backend, str(build["runtime_variant"])),
        "build": {"docker_target": "wheel", "platform_arg": backend["platform"]},
        "builder": {
            "repository": str(builder["target_repository"]),
            "tag": str(builder["target_tag"]),
            "digest": str(builder["target_digest"]),
            "source_image": str(builder["source_image"]),
            "source_image_digest": str(builder["source_image_digest"]),
            "manylinux": str(builder["manylinux"]),
            "recipe_revision": str(builder["recipe_revision"]),
        },
        "build_requirements": copy.deepcopy(requirements["wheel_build"]),
        "runtime_requirements": copy.deepcopy(requirements["wheel_runtime"]),
    }


def _selected_runtimes(
    selection: Mapping[str, object], pinned_upstreams: list[str] | None
) -> list[dict[str, Any]]:
    values = [
        _mapping(item, "upstream runtime")
        for item in selection["runtimes"]  # type: ignore[index]
    ]
    if not pinned_upstreams:
        return values
    pinned = set(pinned_upstreams)
    selected = [
        item
        for item in values
        if f"{item['runtime_repository']}:{item['runtime_tag']}" in pinned
    ]
    if len(
        {f"{item['runtime_repository']}:{item['runtime_tag']}" for item in selected}
    ) != len(pinned):
        raise ValueError("pinned runtime references are not in the inspected selection")
    return selected


def resolve_plan(
    catalog: dict[str, Any],
    *,
    builder_catalog: dict[str, Any],
    runtime_selection: dict[str, Any],
    route: str,
    pinned_upstreams: list[str] | None = None,
    git_tag: str | None = None,
    release_kind: str | None = None,
    is_prerelease: bool | None = None,
    chart_version: str | None = None,
) -> dict[str, Any]:
    """Generate Wheel/Image tasks from one normalized Runtime selection."""
    if route not in ROUTES:
        raise ValueError(f"unsupported release route: {route}")
    selection = upstream.validate_selection(runtime_selection)
    builds = _build_map(selection)
    builder_by_id = _builder_map(builder_catalog)
    wheels_by_id: dict[str, dict[str, Any]] = {}
    images: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []

    for runtime in _selected_runtimes(selection, pinned_upstreams):
        backend = _backend(catalog, str(runtime["backend"]))
        if backend.get("status") == "blocked":
            continue
        if backend.get("status") != "supported":
            raise ValueError(f"backend {runtime['backend']!r} has an invalid status")
        wheel_ids = _mapping(runtime["wheel_build_ids"], f"{runtime['id']} Wheel links")
        image_ids: list[str] = []
        member_records: list[dict[str, str]] = []
        creates_index = len(runtime["architectures"]) > 1
        for architecture in runtime["architectures"]:
            build_id = str(wheel_ids[str(architecture)])
            build = builds.get(build_id)
            builder = builder_by_id.get(build_id)
            if build is None or builder is None:
                raise ValueError(
                    f"{runtime['id']}: matching Wheel/Builder {build_id!r} is missing"
                )
            _validate_builder_matches_build(build, builder)
            if build_id not in wheels_by_id:
                wheels_by_id[build_id] = _wheel_task(catalog, build, builder, backend)
            image_id = f"{runtime['id']}-{architecture}"
            image_ids.append(image_id)
            member_reference = (
                f"{runtime['target_repository']}:{runtime['target_tag']}"
                + (f"-{architecture}" if creates_index else "")
            )
            member_records.append(
                {
                    "image_id": image_id,
                    "cpu_arch": str(architecture),
                    "reference": member_reference,
                }
            )
            images.append(
                {
                    "id": image_id,
                    "label": f"{_runtime_label(runtime)} · {architecture}",
                    "runner": catalog["runners"][architecture],
                    "wheel_id": build_id,
                    "family_id": str(runtime["id"]),
                    "profile_id": str(runtime["id"]),
                    "cpu_arch": str(architecture),
                    "platform": f"linux/{architecture}",
                    "runtime": {
                        "product_id": runtime["product_id"],
                        "repository": runtime["runtime_repository"],
                        "tag": runtime["runtime_tag"],
                        "digest": runtime["runtime_digest"],
                        "image_reference": runtime["member_references"][architecture],
                        "version": runtime["version"],
                        "channel": runtime["channel"],
                        "variant": runtime["variant"],
                        "accelerator_runtime": runtime["accelerator_runtime"],
                        "soc_version": runtime["soc_version"],
                        "python_version": runtime["python_version"],
                        "python_abi": runtime["python_abi"],
                        "os_id": runtime["os_id"],
                        "os_version": runtime["os_version"],
                        "glibc_version": runtime["glibc_version"],
                    },
                    "target_repository": runtime["target_repository"],
                    "target_tag": runtime["target_tag"],
                    "build_requirements": copy.deepcopy(
                        catalog["requirements"]["wheel_runtime"]
                    ),
                }
            )
        published_reference = (
            f"{runtime['target_repository']}:{runtime['target_tag']}"
            if creates_index
            else member_records[0]["reference"]
        )
        families.append(
            {
                "id": str(runtime["id"]),
                "label": _runtime_label(runtime),
                "product_id": runtime["product_id"],
                "variant": runtime["variant"],
                "runtime": {
                    "repository": runtime["runtime_repository"],
                    "tag": runtime["runtime_tag"],
                    "digest": runtime["runtime_digest"],
                    "version": runtime["version"],
                    "channel": runtime["channel"],
                    "accelerator_runtime": runtime["accelerator_runtime"],
                    "soc_version": runtime["soc_version"],
                    "python_abi": runtime["python_abi"],
                    "os_id": runtime["os_id"],
                    "os_version": runtime["os_version"],
                    "glibc_version": runtime["glibc_version"],
                },
                "target_repository": runtime["target_repository"],
                "target_tag": runtime["target_tag"],
                "image_ids": sorted(image_ids),
                "members": sorted(member_records, key=lambda item: item["cpu_arch"]),
                "create_index": creates_index,
                "published_reference": published_reference,
            }
        )

    wheels = sorted(wheels_by_id.values(), key=lambda item: item["id"])
    images.sort(key=lambda item: item["id"])
    families.sort(key=lambda item: item["id"])
    if not wheels or not images or not families:
        raise ValueError("release plan has no supported Wheel/runtime families")
    limits = catalog["matrix_limits"]
    for key, values, limit_key in (
        ("wheels", wheels, "max_wheel_tasks"),
        ("images", images, "max_image_tasks"),
        ("families", families, "max_family_tasks"),
    ):
        if len(values) > int(limits[limit_key]):
            raise ValueError(f"release plan {key} count exceeds configured limit")
    distributions = sorted({str(item["dist_name"]) for item in wheels})
    publish = copy.deepcopy(catalog["publish"])
    publish["pypi"]["distributions"] = distributions
    resolved_release_type = catalog.get("release_type")
    if resolved_release_type not in {"stable", "prerelease", "draft", "nightly"}:
        raise ValueError(f"unsupported release type: {resolved_release_type!r}")
    resolved_version = str(catalog["ucm_version"])
    resolved_git_tag = git_tag or str(catalog["release_tag"])
    resolved_release_kind = release_kind or (
        "publish" if route == "release" else "none"
    )
    if resolved_release_kind not in {"none", "publish", "draft"}:
        raise ValueError(f"unsupported release kind: {resolved_release_kind}")
    resolved_prerelease = (
        Version(resolved_version).is_prerelease
        if is_prerelease is None
        else is_prerelease
    )
    chart = copy.deepcopy(catalog["chart"])
    if chart_version is not None:
        if not chart_version:
            raise ValueError("Chart version override must not be empty")
        chart["version"] = chart_version
    return {
        "kind": "ucm-release-plan",
        "route": route,
        "release_type": resolved_release_type,
        "version": resolved_version,
        "image_version": resolved_version,
        "git_tag": resolved_git_tag,
        "release_kind": resolved_release_kind,
        "is_prerelease": resolved_prerelease,
        "publish": publish,
        "chart": chart,
        "wheels": wheels,
        "images": images,
        "families": families,
        "wheel_matrix": {
            "include": [
                {key: task[key] for key in ("id", "label", "runner")} for task in wheels
            ]
        },
        "image_matrix": {
            "include": [
                {key: task[key] for key in ("id", "label", "runner", "wheel_id")}
                for task in images
            ]
        },
    }


def retag_pr_plan(
    plan: Mapping[str, Any],
    *,
    pr_number: int | str,
    author: str,
    run_id: int | str,
) -> dict[str, Any]:
    """Project collision-resistant PR targets without changing build tasks."""
    result = copy.deepcopy(_mapping(plan, "release plan"))
    if result.get("kind") != "ucm-release-plan" or result.get("route") != "pr":
        raise ValueError("PR retagging requires a compact PR plan")
    families = {
        str(item["id"]): item
        for item in result.get("families", [])
        if isinstance(item, dict)
    }
    for family in families.values():
        base_tag = runtime_ops.project_pr_tag(
            str(family["runtime"]["tag"]),
            pr_number=pr_number,
            author=author,
            run_id=run_id,
        )
        family["target_tag"] = base_tag
        for member in family["members"]:
            member["reference"] = f"{family['target_repository']}:{base_tag}" + (
                f"-{member['cpu_arch']}" if family["create_index"] else ""
            )
        family["published_reference"] = (
            f"{family['target_repository']}:{base_tag}"
            if family["create_index"]
            else family["members"][0]["reference"]
        )
    for image in result.get("images", []):
        if not isinstance(image, dict):
            raise ValueError("release plan images must be mappings")
        family = families.get(str(image["family_id"]))
        if family is None:
            raise ValueError(f"image {image['id']} references an unknown family")
        image["target_tag"] = family["target_tag"]
    return result


def select_task(plan: Mapping[str, Any], kind: str, task_id: str) -> dict[str, Any]:
    collection = {"wheel": "wheels", "image": "images"}.get(kind)
    if collection is None:
        raise ValueError(f"unsupported task kind: {kind}")
    matches = [item for item in plan.get(collection, []) if item.get("id") == task_id]
    if len(matches) != 1:
        raise ValueError(f"{kind} task {task_id!r} does not resolve exactly once")
    return copy.deepcopy(matches[0])


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


def _validate_wheel_platforms(platforms: set[str], architecture: str) -> str:
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
    expected = f"linux_{wheel_architecture}"
    if platforms != {expected}:
        raise ValueError(f"built Wheel platform must be exactly {expected}")
    return wheel_architecture


def _auditwheel_result(
    wheel_path: Path, architecture: str, report_path: Path | None
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

    glibc_versions = sorted(
        {
            f"GLIBC_{version}"
            for version in re.findall(r"\bGLIBC_(\d+(?:\.\d+)+)\b", text)
        },
        key=lambda value: tuple(
            int(part) for part in value.removeprefix("GLIBC_").split(".")
        ),
    )
    external_libraries: list[str] | None = None
    marker = "The following external shared libraries are required by the wheel:"
    if marker in text:
        payload = text.split(marker, 1)[1].lstrip()
        try:
            libraries, _ = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError as error:
            raise ValueError("auditwheel external-library report is invalid") from error
        if not isinstance(libraries, dict):
            raise ValueError("auditwheel external-library report must be a mapping")
        external_libraries = sorted(str(name) for name in libraries)
    elif "The wheel requires no external shared libraries" in text:
        external_libraries = []

    return {
        "auditwheel_report": {
            "filename": report.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "text": text,
        },
        "auditwheel_platform_tag": compatible_platform,
        "glibc_versions": glibc_versions,
        "glibc_floor": glibc_versions[-1] if glibc_versions else None,
        "external_libraries": external_libraries,
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
    _validate_wheel_platforms(platform_tags, architecture)
    filename_tags = {str(tag) for tag in tags}
    if _wheel_metadata_tags(wheel_path) != filename_tags:
        raise ValueError("built Wheel filename and WHEEL metadata Tags do not match")
    result = {
        "kind": "ucm-wheel-result",
        "schema_version": 2,
        "task_id": task["id"],
        "distribution": task["dist_name"],
        "version": task["wheel_version"],
        "python_abi": python_abi,
        "cpu_arch": architecture,
        "filename": wheel_path.name,
        "platform_tags": sorted(platform_tags),
    }
    result.update(_auditwheel_result(wheel_path, architecture, auditwheel_report_path))
    return result
