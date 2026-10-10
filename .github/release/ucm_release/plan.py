"""Plan builds from current release policy and inspected runtimes."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from . import builders
from . import policy as release_policy
from . import runtime as runtime_ops
from . import toolkit, upstream, wheel_audit

ROUTES = frozenset({"pr", "release"})
WHEEL_ARCHITECTURES = {"amd64": "x86_64", "arm64": "aarch64"}
_UCM_DISTRIBUTION = re.compile(r"(?:[a-z0-9]+-)*uc-manager(?:-[a-z0-9]+)*")


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


def _namespaced_distribution(prefix: str, distribution: str) -> str:
    value = f"{prefix}{distribution}"
    if (
        len(value) > release_policy.MAX_PYPI_DISTRIBUTION_LENGTH
        or canonicalize_name(value, validate=True) != value
        or _UCM_DISTRIBUTION.fullmatch(value) is None
    ):
        raise ValueError("namespaced Python distribution is invalid")
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


def _meta_package(
    wheels: list[Mapping[str, Any]], version: str, distribution: str
) -> dict[str, Any]:
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
        "distribution": distribution,
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
    distribution_prefix: str,
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
        "dist_name": _namespaced_distribution(
            distribution_prefix,
            _distribution(backend, str(build["runtime_variant"])),
        ),
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


def resolve_plan(
    catalog: dict[str, Any],
    *,
    builder_catalog: dict[str, Any],
    runtime_selection: dict[str, Any],
    route: str,
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
        decision = (
            channel_policy.get("enabled"),
            channel_policy.get("disposition"),
        )
        allowed = (
            {(False, "disabled")}
            if not requested
            else (
                {(True, "publish"), (False, "scope-skipped")}
                if (expected_scope == "fork" and channel == "pypi")
                or (channel == "dockerhub" and route != "release")
                else {(True, "publish")}
            )
        )
        if decision not in allowed:
            raise ValueError(
                f"formal {channel} publication decision does not match repository scope"
            )
        if channel == "dockerhub":
            namespace = channel_policy.get("namespace")
            if channel_policy.get("enabled") != (
                isinstance(namespace, str) and bool(namespace)
            ):
                raise ValueError(
                    "formal Docker Hub namespace does not match publication decision"
                )
    pypi_policy = _mapping(publication_policy["pypi"], "formal PyPI policy")
    expected_pypi_target = "pypi" if expected_scope == "official" else "testpypi"
    if pypi_policy.get("target") != expected_pypi_target:
        raise ValueError("formal PyPI target does not match repository scope")
    distribution_prefix = pypi_policy.get("distribution_prefix")
    expected_distribution_prefix = release_policy.pypi_distribution_prefix(repository)
    if distribution_prefix != expected_distribution_prefix:
        raise ValueError(
            "formal PyPI distribution prefix does not match repository owner"
        )
    selection = upstream.validate_selection(runtime_selection)
    builds = _build_map(selection)
    builder_by_id = _builder_map(builder_catalog)
    wheels_by_id: dict[str, dict[str, Any]] = {}
    images: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []

    for runtime in selection["runtimes"]:
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
                wheels_by_id[build_id] = _wheel_task(
                    catalog,
                    build,
                    builder,
                    backend,
                    expected_distribution_prefix,
                )
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
    meta_package = _meta_package(
        wheels,
        resolved_version,
        _namespaced_distribution(expected_distribution_prefix, "uc-manager"),
    )
    toolkit_package = toolkit.package_identity(repository, resolved_version)
    meta_package["extras"][
        "toolkit"
    ] = f"{toolkit_package['distribution']}=={resolved_version}"
    publish["pypi"]["distributions"].append(toolkit_package["distribution"])
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
        "version_authority": {
            "ucm_base_version": catalog["ucm_base_version"],
            "sha256": catalog["version_authority_sha256"],
            "runtime_selectors": copy.deepcopy(catalog["runtime_selectors"]),
        },
        "publish": publish,
        "meta_package": meta_package,
        "toolkit_package": toolkit_package,
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
