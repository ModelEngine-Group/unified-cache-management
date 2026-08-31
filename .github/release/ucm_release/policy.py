"""Human-maintained schema-v6 Release and platform policy.

The two policy files and their exact requirement lists are the formal build
authorities. ``compatibility_projection`` keeps transitional schema-v3
consumers on the same policy without duplicating Release configuration.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from . import core
from . import runtime as runtime_ops
from . import version_config, wheel_audit

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = RELEASE_ROOT / "release.yaml"
DEFAULT_PLATFORMS = RELEASE_ROOT / "platforms.yaml"
DEFAULT_SCHEMA = RELEASE_ROOT / "schemas" / "config.schema.json"
DEFAULT_BUILD_REQUIREMENTS = RELEASE_ROOT / "requirements" / "wheel-build.txt"
DEFAULT_RUNTIME_REQUIREMENTS = RELEASE_ROOT / "requirements" / "wheel-runtime.txt"
OFFICIAL_REPOSITORY = "ModelEngine-Group/unified-cache-management"

_COMPATIBILITY_SOURCE = {
    "staging_repository": "ghcr.io/{owner}/ucm-release-staging",
    "default_branch": "develop",
    "protected_environment": "release-production",
}
_COMPATIBILITY_LANES = ["feature-candidate", "protected-tag"]
_MATRIX_LIMITS = {
    "max_wheel_tasks": 128,
    "max_image_tasks": 256,
    "max_family_tasks": 128,
}
_SCAN_LIMITS = {"max_tags_per_repository": 1024, "max_selected_upstreams": 64}
RELEASE_TYPES = ("stable", "prerelease", "draft", "nightly")
PUBLISH_CHANNELS = core.PUBLISH_CHANNELS
PYPI_TARGETS = {
    "pypi": {
        "index": "https://upload.pypi.org/legacy/",
        "simple_index": "https://pypi.org/simple/",
        "json_api": "https://pypi.org/pypi/",
        "dependency_index": "https://pypi.org/simple/",
    },
    "testpypi": {
        "index": "https://test.pypi.org/legacy/",
        "simple_index": "https://test.pypi.org/simple/",
        "json_api": "https://test.pypi.org/pypi/",
        "dependency_index": "https://pypi.org/simple/",
    },
}


def publication_identity(repository: str) -> tuple[str, str]:
    """Derive immutable publication scope and Runtime tag prefix."""

    parts = repository.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError("repository identity must be owner/name")
    if repository.casefold() == OFFICIAL_REPOSITORY.casefold():
        return "official", ""
    owner = parts[0]
    owner_component = runtime_ops.sanitize_oci_tag_component(
        owner.lower(), max_length=len(owner)
    )
    return "fork", f"{owner_component}-"


def _companion_path(release_path: Path, explicit: Path | None, default: Path) -> Path:
    if explicit is not None:
        return explicit
    sibling = release_path.parent / default.relative_to(RELEASE_ROOT)
    return sibling if sibling.is_file() else default


def _exact_requirements(path: Path) -> list[str]:
    requirements: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise ValueError(
                f"{path}:{line_number}: invalid requirement {line!r}"
            ) from error
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.marker is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ValueError(
                f"{path}:{line_number}: requirement must be one unconditional exact pin"
            )
        try:
            version = str(Version(specifiers[0].version))
        except InvalidVersion as error:
            raise ValueError(
                f"{path}:{line_number}: requirement version is invalid"
            ) from error
        name = canonicalize_name(requirement.name)
        normalized = f"{name}=={version}"
        if name in requirements:
            raise ValueError(f"{path}:{line_number}: duplicate requirement {name!r}")
        requirements[name] = normalized
    if not requirements:
        raise ValueError(f"{path}: requirements file must not be empty")
    return [requirements[name] for name in sorted(requirements)]


def _select_release_profile(
    release: dict[str, Any], release_type: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if release_type not in RELEASE_TYPES:
        raise ValueError(f"unsupported release type: {release_type!r}")
    profiles = release["release_profiles"]
    profile = copy.deepcopy(profiles[release_type])
    addresses = release["publish"]
    switches = profile["publish"]
    publish: dict[str, dict[str, Any]] = {}
    for channel in PUBLISH_CHANNELS:
        config = copy.deepcopy(addresses[channel])
        requested = switches[channel]
        config["requested"] = requested
        config["enabled"] = requested
        config["disposition"] = "publish" if requested else "disabled"
        publish[channel] = config
    return profile, publish


def _pypi_target(config: dict[str, Any], target: str) -> dict[str, Any]:
    endpoints = PYPI_TARGETS[target]
    if target == "pypi" and config.get("index") != endpoints["index"]:
        raise ValueError("official PyPI upload endpoint differs from policy")
    return {**copy.deepcopy(config), "target": target, **endpoints}


def _dockerhub_namespace(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"docker\.io/[a-z0-9][a-z0-9._-]{0,127}", value) is None
    ):
        raise ValueError("Fork Docker Hub namespace must be docker.io/<account-or-org>")
    return value


def _resolve_release_profile(
    release: dict[str, Any],
    release_type: str,
    publication_scope: str,
    *,
    fork_test_pypi: bool = False,
    fork_dockerhub_namespace: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    profile, publish = _select_release_profile(release, release_type)
    publish["pypi"] = _pypi_target(
        publish["pypi"], "testpypi" if publication_scope == "fork" else "pypi"
    )
    if publication_scope == "fork":
        if not isinstance(fork_test_pypi, bool):
            raise ValueError("Fork TestPyPI availability must be boolean")
        if fork_dockerhub_namespace is not None:
            publish["dockerhub"]["namespace"] = _dockerhub_namespace(
                fork_dockerhub_namespace
            )
        availability = {
            "pypi": fork_test_pypi,
            "dockerhub": fork_dockerhub_namespace is not None,
        }
        for channel, available in availability.items():
            requested = publish[channel]["requested"]
            publish[channel]["enabled"] = requested and available
            publish[channel]["disposition"] = (
                "publish"
                if requested and available
                else "scope-skipped" if requested else "disabled"
            )
    return profile, publish


def _validate_release_semantics(release: dict[str, Any]) -> None:
    products = release["products"]
    product_ids = [product["id"] for product in products]
    if set(product_ids) != {"vllm", "vllm-ascend"} or len(product_ids) != 2:
        raise ValueError("release policy requires exactly vllm and vllm-ascend")
    for release_type in RELEASE_TYPES:
        profile_publish = release["release_profiles"][release_type]["publish"]
        if (
            any(
                profile_publish[channel]
                for channel in ("pypi", "ghcr", "dockerhub", "chart_oci")
            )
            and not profile_publish["github_release"]
        ):
            raise ValueError(
                f"{release_type}: enabled public channels require the GitHub Release "
                "Draft barrier"
            )
        if profile_publish["dockerhub"] and not profile_publish["ghcr"]:
            raise ValueError(
                f"{release_type}: Docker Hub publication requires GHCR source "
                "publication"
            )


def _validate_platform_semantics(platforms: dict[str, Any]) -> None:
    for backend, config in platforms["backends"].items():
        has_distribution = "distribution" in config
        has_template = "distribution_template" in config
        if has_distribution == has_template:
            raise ValueError(
                f"platform backend {backend!r} requires exactly one distribution rule"
            )
        if config["status"] == "blocked" and "reason" not in config:
            raise ValueError(f"blocked platform backend {backend!r} requires a reason")
        if config["status"] == "supported" and "reason" in config:
            raise ValueError(
                f"supported platform backend {backend!r} cannot have a reason"
            )
        patterns = config.get("external_runtime_exclude_patterns")
        if config["status"] == "supported":
            if not isinstance(patterns, list) or not patterns:
                raise ValueError(
                    f"supported platform backend {backend!r} requires external runtime exclude patterns"
                )
            for pattern in patterns:
                if not isinstance(pattern, str):
                    raise ValueError(
                        f"platform backend {backend!r} has invalid external runtime exclude pattern"
                    )
                rendered = pattern.replace("{accelerator_major}", "1")
                if (
                    pattern.count("{accelerator_major}") > 1
                    or "{" in rendered
                    or "}" in rendered
                ):
                    raise ValueError(
                        f"platform backend {backend!r} has invalid external runtime exclude pattern"
                    )
                try:
                    wheel_audit.validate_exclude_pattern(rendered)
                except ValueError as error:
                    raise ValueError(
                        f"platform backend {backend!r} has invalid external runtime exclude pattern"
                    ) from error
            if backend != "cuda" and any(
                "{accelerator_major}" in pattern for pattern in patterns
            ):
                raise ValueError(
                    f"platform backend {backend!r} cannot use accelerator_major"
                )
        elif patterns is not None:
            raise ValueError(
                f"blocked platform backend {backend!r} cannot declare runtime library policy"
            )


def load(
    release_path: Path = DEFAULT_RELEASE,
    *,
    platforms_path: Path | None = None,
    schema_path: Path = DEFAULT_SCHEMA,
    build_requirements_path: Path | None = None,
    runtime_requirements_path: Path | None = None,
) -> dict[str, Any]:
    """Load only the schema-v6 formal policy and its direct authorities."""
    resolved_platforms = _companion_path(
        release_path, platforms_path, DEFAULT_PLATFORMS
    )
    resolved_build_requirements = _companion_path(
        release_path, build_requirements_path, DEFAULT_BUILD_REQUIREMENTS
    )
    resolved_runtime_requirements = _companion_path(
        release_path, runtime_requirements_path, DEFAULT_RUNTIME_REQUIREMENTS
    )
    schema = core.load_json(schema_path)
    release = core.load_yaml(release_path)
    platforms = core.load_yaml(resolved_platforms)
    core.validate_schema(
        release, schema["$defs"]["releasePolicy"], root=schema, path="$.release"
    )
    core.validate_schema(
        platforms, schema["$defs"]["platformPolicy"], root=schema, path="$.platforms"
    )
    _validate_release_semantics(release)
    _validate_platform_semantics(platforms)
    return {
        "release": release,
        "platforms": platforms,
        "requirements": {
            "wheel_build": _exact_requirements(resolved_build_requirements),
            "wheel_runtime": _exact_requirements(resolved_runtime_requirements),
        },
    }


def resolve(
    release_path: Path = DEFAULT_RELEASE,
    *,
    platforms_path: Path | None = None,
    repository_root: Path = core.REPO_ROOT,
    repository: str | None = None,
    version_override: str | None = None,
    release_type: str = "stable",
    fork_test_pypi: bool = False,
    fork_dockerhub_namespace: str | None = None,
) -> dict[str, Any]:
    """Resolve the two human policies into the formal runtime authority."""
    bundle = load(release_path, platforms_path=platforms_path)
    release = copy.deepcopy(bundle["release"])
    platforms = copy.deepcopy(bundle["platforms"])
    resolved_repository = core.resolve_repository(
        repository, repository_root=repository_root
    )
    owner, repo = resolved_repository.split("/", 1)
    publication_scope, runtime_image_tag_prefix = publication_identity(
        resolved_repository
    )

    def resolve_repository_templates(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("{owner}", owner.lower()).replace(
                "{repo}", repo.lower()
            )
        if isinstance(value, list):
            return [resolve_repository_templates(item) for item in value]
        if isinstance(value, dict):
            return {
                key: resolve_repository_templates(item) for key, item in value.items()
            }
        return value

    merged = resolve_repository_templates(
        {
            **release,
            "excluded_upstream_variants": platforms["excluded_upstream_variants"],
            "builder_families": platforms["builder_families"],
            "backends": platforms["backends"],
        }
    )
    selected_profile, normalized_publish = _resolve_release_profile(
        merged,
        release_type,
        publication_scope,
        fork_test_pypi=fork_test_pypi,
        fork_dockerhub_namespace=fork_dockerhub_namespace,
    )
    merged["publish"] = normalized_publish
    merged["release_type"] = release_type
    merged["release_profile"] = selected_profile
    version_authority = version_config.load(repository_root / "version.ini")
    version = version_override or str(version_authority["ucm_version"])
    selectors = version_authority["supported_runtimes"]
    for product in merged["products"]:
        product["runtime_selectors"] = copy.deepcopy(selectors[product["id"]])
    chart_document = core.load_yaml(
        repository_root / release["chart"]["source"] / "Chart.yaml"
    )
    chart_name = chart_document.get("name")
    if not isinstance(chart_name, str) or not chart_name:
        raise ValueError("Chart.yaml must declare a non-empty name")
    merged.update(
        {
            "repository": resolved_repository,
            "publication_scope": publication_scope,
            "runtime_image_tag_prefix": runtime_image_tag_prefix,
            "ucm_version": version,
            "ucm_base_version": version_authority["ucm_base_version"],
            "version_authority_sha256": version_authority["authority_sha256"],
            "runtime_selectors": copy.deepcopy(selectors),
            "release_tag": f"v{version}",
            "matrix_limits": copy.deepcopy(_MATRIX_LIMITS),
            "requirements": copy.deepcopy(bundle["requirements"]),
        }
    )
    merged["chart"].update(
        {
            "name": chart_name,
            "version": core.derive_chart_version(version),
            "app_version": version,
        }
    )
    return merged


def _backend_contracts(platforms: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for backend, source in platforms["backends"].items():
        contract: dict[str, Any] = {
            "status": source["status"],
            "platform_arg": source["platform"],
            "required_native": [],
            "forbidden_native": [],
            "allowed_dt_needed": [],
            "external_required_dependencies": [],
        }
        if "reason" in source:
            contract["reason"] = source["reason"]
        if "distribution" in source:
            contract["distribution"] = source["distribution"]
        else:
            template = source["distribution_template"]
            marker = "{runtime_variant}"
            if not template.endswith(marker):
                raise ValueError(
                    f"platform backend {backend!r}: unsupported distribution template"
                )
            contract["distribution_prefix"] = template.removesuffix(marker)
        contracts[backend] = contract
    return contracts


def compatibility_projection(
    policy: dict[str, Any],
    *,
    chart_name: str,
    release_type: str = "stable",
    repository: str = OFFICIAL_REPOSITORY,
) -> dict[str, Any]:
    """Project v5 policy into the transitional schema-v3 catalog interface."""
    release = policy["release"]
    platforms = policy["platforms"]
    smoke_values = release["chart"]["smoke_values"]
    upstream_products = []
    for product in release["products"]:
        projected = {
            "id": product["id"],
            "runtime_product": product["id"],
            "runtime_repository": product["runtime_repository"],
            "target_repository": product["target_repository"],
            "version_specifier": ">=0",
            "channels": ["stable", "rc", "nightly"],
            # Transitional only. Formal runtime Python comes from OCI probing.
            "integration_python_abi": "cp312",
        }
        upstream_products.append(projected)
    publication_scope, _ = publication_identity(repository)
    _, publish = _resolve_release_profile(release, release_type, publication_scope)
    builder_families = platforms["builder_families"]
    return {
        "kind": "release-config",
        "schema_version": 4,
        "image_revision": 1,
        "source": copy.deepcopy(_COMPATIBILITY_SOURCE),
        "lanes": copy.deepcopy(_COMPATIBILITY_LANES),
        "runner_map": copy.deepcopy(release["runners"]),
        "upstream_products": upstream_products,
        "chart": {
            "source": release["chart"]["source"],
            "name": chart_name,
            "validation_cases": [
                {
                    "name": "cuda",
                    "values": smoke_values["vllm"],
                    "product_id": "vllm",
                    "variant": "default",
                    "expected_resource": "nvidia.com/gpu",
                },
                {
                    "name": "a2",
                    "values": smoke_values["vllm-ascend"],
                    "product_id": "vllm-ascend",
                    "variant": "a2",
                    "expected_resource": "huawei.com/Ascend910",
                },
                {
                    "name": "a3",
                    "values": smoke_values["vllm-ascend"],
                    "product_id": "vllm-ascend",
                    "variant": "a3",
                    "expected_resource": "huawei.com/Ascend910",
                },
            ],
        },
        "publish": publish,
        "matrix_limits": copy.deepcopy(_MATRIX_LIMITS),
        "scan_limits": copy.deepcopy(_SCAN_LIMITS),
        "wheel_build_requirements": copy.deepcopy(
            policy["requirements"]["wheel_build"]
        ),
        "wheel_runtime_requirements": copy.deepcopy(
            policy["requirements"]["wheel_runtime"]
        ),
        "builder_checks": {
            "cuda": {
                "commands": copy.deepcopy(
                    builder_families["cuda"]["required_commands"]
                ),
            },
            "ascend": {
                "commands": copy.deepcopy(
                    builder_families["ascend"]["required_commands"]
                ),
                "required_files": copy.deepcopy(
                    builder_families["ascend"]["required_files"]
                ),
            },
        },
        "backend_contracts": _backend_contracts(platforms),
    }
