"""Compact planning shared by formal Release and inspected PR runtimes."""

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
from packaging.version import Version

from . import builders
from . import policy as release_policy
from . import runtime as runtime_ops
from . import upstream, wheel_audit

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


def _target_platform_tag(manylinux: str, architecture: str) -> str:
    try:
        wheel_architecture = WHEEL_ARCHITECTURES[architecture]
    except KeyError as error:
        raise ValueError(
            f"unsupported Wheel CPU architecture: {architecture}"
        ) from error
    if re.fullmatch(r"manylinux_[0-9]+_[0-9]+", manylinux) is None:
        raise ValueError(f"invalid manylinux policy: {manylinux!r}")
    return f"{manylinux}_{wheel_architecture}"


def _manylinux_parts(platform: str) -> tuple[int, int, str]:
    match = re.fullmatch(
        r"manylinux_(?P<major>[0-9]+)_(?P<minor>[0-9]+)_(?P<arch>x86_64|aarch64)",
        platform,
    )
    if match is None:
        raise ValueError(f"invalid manylinux platform tag: {platform!r}")
    return int(match.group("major")), int(match.group("minor")), match.group("arch")


def _auditwheel_version(requirements: list[str]) -> str:
    matches = [
        requirement
        for raw in requirements
        if (requirement := Requirement(raw)).name.lower() == "auditwheel"
    ]
    if len(matches) != 1:
        raise ValueError("Wheel build requirements must pin auditwheel exactly once")
    specifiers = list(matches[0].specifier)
    if len(specifiers) != 1 or specifiers[0].operator != "==":
        raise ValueError("Wheel build auditwheel requirement must be an exact pin")
    return specifiers[0].version


def _external_runtime_exclude_patterns(
    backend: Mapping[str, Any], build: Mapping[str, Any]
) -> list[str]:
    templates = backend.get("external_runtime_exclude_patterns")
    if (
        not isinstance(templates, list)
        or not templates
        or any(not isinstance(item, str) or not item for item in templates)
    ):
        raise ValueError("supported backend has no external runtime exclude patterns")
    accelerator_runtime = str(build["accelerator_runtime"])
    match = re.fullmatch(r"cuda-(?P<major>[0-9]+)(?:\.[0-9]+)*", accelerator_runtime)
    major = match.group("major") if match is not None else None
    patterns: list[str] = []
    for template in templates:
        if "{accelerator_major}" in template:
            if major is None:
                raise ValueError(
                    "external runtime exclude pattern requires a CUDA accelerator major"
                )
            pattern = template.replace("{accelerator_major}", major)
        else:
            pattern = template
        wheel_audit.validate_exclude_pattern(pattern)
        patterns.append(pattern)
    if len(patterns) != len(set(patterns)):
        raise ValueError("external runtime exclude patterns contain duplicates")
    return sorted(patterns)


def _meta_package(wheels: list[Mapping[str, Any]], version: str) -> dict[str, Any]:
    extras: dict[str, str] = {}
    for wheel in wheels:
        extra = str(wheel["runtime_variant"])
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", extra) is None:
            raise ValueError(f"Wheel channel cannot be used as an extra: {extra!r}")
        requirement = f"{wheel['dist_name']}=={version}"
        existing = extras.setdefault(extra, requirement)
        if existing != requirement:
            raise ValueError(f"Wheel channel {extra!r} maps to multiple distributions")
    if not extras:
        raise ValueError("meta package requires at least one backend extra")
    return {
        "distribution": "uc-manager",
        "version": version,
        "extras": {key: extras[key] for key in sorted(extras)},
    }


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
    manylinux = str(build["manylinux"])
    requirements = _mapping(catalog.get("requirements"), "formal requirements")
    build_requirements = copy.deepcopy(requirements["wheel_build"])
    target_platform_tag = _target_platform_tag(manylinux, architecture)
    external_runtime_exclude_patterns = _external_runtime_exclude_patterns(
        backend, build
    )
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
        "manylinux": manylinux,
        "target_platform_tag": target_platform_tag,
        "external_runtime_exclude_patterns": external_runtime_exclude_patterns,
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
        "build_requirements": build_requirements,
        "repair": {
            "tool": "auditwheel",
            "version": _auditwheel_version(build_requirements),
            "target_platform": target_platform_tag,
            "excluded_patterns": external_runtime_exclude_patterns,
        },
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
    repository = str(catalog.get("repository", ""))
    publication_scope = catalog.get("publication_scope")
    runtime_image_tag_prefix = catalog.get("runtime_image_tag_prefix")
    expected_scope, expected_prefix = release_policy.publication_identity(repository)
    if publication_scope != expected_scope:
        raise ValueError("formal publication scope does not match repository identity")
    if not isinstance(runtime_image_tag_prefix, str):
        raise ValueError("formal runtime image tag prefix must be a string")
    if runtime_image_tag_prefix != expected_prefix:
        raise ValueError(
            "formal runtime image tag prefix does not match repository owner"
        )
    publication_policy = _mapping(catalog.get("publish"), "formal publication policy")
    release_profile = _mapping(catalog.get("release_profile"), "release profile")
    profile_requests = _mapping(
        release_profile.get("publish"), "release profile publication requests"
    )
    for channel in release_policy.PUBLISH_CHANNELS:
        requested = profile_requests.get(channel)
        channel_policy = _mapping(
            publication_policy.get(channel), f"formal {channel} publication policy"
        )
        if (
            not isinstance(requested, bool)
            or channel_policy.get("requested") is not requested
        ):
            raise ValueError(
                f"formal {channel} publication request does not match its Profile"
            )
        scope_skipped = (
            expected_scope == "fork" and channel in {"pypi", "dockerhub"} and requested
        )
        expected_enabled = requested and not scope_skipped
        expected_disposition = (
            "scope-skipped" if scope_skipped else "publish" if requested else "disabled"
        )
        if (
            channel_policy.get("enabled") is not expected_enabled
            or channel_policy.get("disposition") != expected_disposition
        ):
            raise ValueError(
                f"formal {channel} publication decision does not match repository scope"
            )
    selection = upstream.validate_selection(runtime_selection)
    builds = _build_map(selection)
    builder_by_id = _builder_map(builder_catalog)
    wheels_by_id: dict[str, dict[str, Any]] = {}
    images: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []

    for runtime in _selected_runtimes(selection, pinned_upstreams):
        if runtime_image_tag_prefix and not str(runtime["target_tag"]).startswith(
            runtime_image_tag_prefix
        ):
            raise ValueError(
                f"{runtime['id']}: Runtime target tag is missing repository-owner prefix"
            )
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
    meta_package = _meta_package(wheels, resolved_version)
    image_by_wheel: dict[str, dict[str, Any]] = {}
    for image in images:
        image_by_wheel.setdefault(str(image["wheel_id"]), image)
    pypi_test_matrix = {
        "include": [
            {
                "id": wheel["id"],
                "label": f"{wheel['runtime_variant']} · {wheel['cpu_arch']}",
                "runner": wheel["runner"],
                "cpu_arch": wheel["cpu_arch"],
                "extra": wheel["runtime_variant"],
                "distribution": wheel["dist_name"],
                "platform_arg": wheel["build"]["platform_arg"],
                "runtime_image": image_by_wheel[str(wheel["id"])]["runtime"][
                    "image_reference"
                ],
            }
            for wheel in wheels
        ]
    }
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
        "repository": repository,
        "publication_scope": publication_scope,
        "runtime_image_tag_prefix": runtime_image_tag_prefix,
        "route": route,
        "release_type": resolved_release_type,
        "version": resolved_version,
        "image_version": resolved_version,
        "git_tag": resolved_git_tag,
        "release_kind": resolved_release_kind,
        "is_prerelease": resolved_prerelease,
        "publish": publish,
        "meta_package": meta_package,
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
        "pypi_test_matrix": pypi_test_matrix,
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
            tag_prefix=str(result.get("runtime_image_tag_prefix", "")),
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
    external_closure = wheel_audit.validate_external_library_closure(
        text,
        expected_patterns=expected_external_patterns,
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
            auditwheel_report_path,
        )
    )
    return result
