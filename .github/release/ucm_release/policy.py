"""Human-maintained schema-v6 Release and platform policy.

The two policy files, version.ini and exact requirements are the build authorities.
"""

from __future__ import annotations

import copy
import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidName, canonicalize_name
from packaging.version import InvalidVersion, Version

from . import runtime as runtime_ops
from . import serialization, version_config, wheel_audit

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = RELEASE_ROOT / "release.yaml"
DEFAULT_PLATFORMS = RELEASE_ROOT / "platforms.yaml"
DEFAULT_SCHEMA = RELEASE_ROOT / "schemas" / "config.schema.json"
DEFAULT_BUILD_REQUIREMENTS = REPO_ROOT / "requirements-build.txt"
DEFAULT_PROJECT = REPO_ROOT / "pyproject.toml"
OFFICIAL_REPOSITORY = "ModelEngine-Group/unified-cache-management"

_MATRIX_LIMITS = {
    "max_wheel_tasks": 128,
    "max_image_tasks": 256,
    "max_family_tasks": 128,
}
RELEASE_TYPES = ("stable", "prerelease", "draft", "nightly")
PUBLISH_CHANNELS = ("pypi", "ghcr", "dockerhub", "chart_oci", "github_release")
MAX_PYPI_DISTRIBUTION_LENGTH = 128
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
_PYPI_BACKEND_DISTRIBUTION = re.compile(
    r"(?:[a-z0-9]+-)*uc-manager-"
    r"(?:cuda(?:-[a-z0-9]+)*|cann(?:[0-9]+)?-a[0-9]+(?:-[a-z0-9]+)*)"
)


_REPOSITORY_IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )  # noqa: E501
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _origin_repository(remote_url: str | None) -> str | None:
    if remote_url is None:
        return None
    prefixes = ("https://github.com/", "git@github.com:")
    for prefix in prefixes:
        if remote_url.startswith(prefix):
            repository = remote_url.removeprefix(prefix).removesuffix(".git")
            if re.fullmatch(r"[^/]+/[^/]+", repository):
                return repository
    return None


def resolve_repository(
    repository: str | None = None,
    *,
    repository_root: Path = REPO_ROOT,
) -> str:
    if repository:
        candidate = repository.strip()
    else:
        candidate = os.environ.get("GITHUB_REPOSITORY", "").strip()
        if not candidate:
            remote = _git_output(repository_root, "remote", "get-url", "origin")
            candidate = _origin_repository(remote).strip() if remote else ""
    if not candidate or _REPOSITORY_IDENTITY_RE.fullmatch(candidate) is None:
        raise ValueError(
            "could not resolve the running repository; pass --repository or set GITHUB_REPOSITORY (local dev infers from the origin remote)"
        )  # noqa: E501
    return candidate


def publication_identity(repository: str) -> tuple[str, str]:
    """Derive immutable publication scope and Runtime tag prefix."""

    if (
        repository.count("/") != 1
        or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None
    ):
        raise ValueError("repository identity must be owner/name")
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


def pypi_distribution_prefix(repository: str) -> str:
    """Derive the lossless PEP 503 namespace for Fork Python distributions."""

    publication_scope, _ = publication_identity(repository)
    if publication_scope == "official":
        return ""
    owner = repository.split("/", 1)[0]
    if len(owner) > 39:
        raise ValueError("repository owner exceeds the GitHub username limit")
    try:
        normalized = str(canonicalize_name(owner, validate=True))
    except InvalidName as error:
        raise ValueError(
            "repository owner is not a valid Python project prefix"
        ) from error
    if normalized != owner.casefold():
        raise ValueError("repository owner requires lossy Python name normalization")
    if _PYPI_BACKEND_DISTRIBUTION.fullmatch(f"{normalized}-uc-manager"):
        raise ValueError("repository owner makes the UCM meta name ambiguous")
    return f"{normalized}-"


def _companion_path(release_path: Path, explicit: Path | None, default: Path) -> Path:
    if explicit is not None:
        return explicit
    sibling = release_path.parent / default.relative_to(RELEASE_ROOT)
    return sibling if sibling.is_file() else default


def _exact_requirements(lines: list[str], context: str) -> list[str]:
    requirements: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise ValueError(
                f"{context}:{line_number}: invalid requirement {line!r}"
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
                f"{context}:{line_number}: requirement must be one unconditional exact pin"
            )
        try:
            version = str(Version(specifiers[0].version))
        except InvalidVersion as error:
            raise ValueError(
                f"{context}:{line_number}: requirement version is invalid"
            ) from error
        name = canonicalize_name(requirement.name)
        normalized = f"{name}=={version}"
        if name in requirements:
            raise ValueError(f"{context}:{line_number}: duplicate requirement {name!r}")
        requirements[name] = normalized
    if not requirements:
        raise ValueError(f"{context}: requirements must not be empty")
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
        raise ValueError("Docker Hub namespace must be docker.io/<account-or-org>")
    return value


def _resolve_release_profile(
    release: dict[str, Any],
    release_type: str,
    publication_scope: str,
    *,
    fork_test_pypi: bool = False,
    dockerhub_namespace: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    profile, publish = _select_release_profile(release, release_type)
    publish["pypi"] = _pypi_target(
        publish["pypi"], "testpypi" if publication_scope == "fork" else "pypi"
    )
    dockerhub = publish["dockerhub"]
    dockerhub_requested = dockerhub["requested"]
    if dockerhub_requested and dockerhub_namespace is not None:
        dockerhub["namespace"] = _dockerhub_namespace(dockerhub_namespace)
        dockerhub["enabled"] = True
        dockerhub["disposition"] = "publish"
    elif dockerhub_requested:
        dockerhub["enabled"] = False
        dockerhub["disposition"] = "scope-skipped"
    else:
        dockerhub["enabled"] = False
        dockerhub["disposition"] = "disabled"
    if publication_scope == "fork":
        if not isinstance(fork_test_pypi, bool):
            raise ValueError("Fork TestPyPI availability must be boolean")
        requested = publish["pypi"]["requested"]
        publish["pypi"]["enabled"] = requested and fork_test_pypi
        publish["pypi"]["disposition"] = (
            "publish"
            if requested and fork_test_pypi
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
    project_path: Path = DEFAULT_PROJECT,
) -> dict[str, Any]:
    """Load only the schema-v6 formal policy and its direct authorities."""
    resolved_platforms = _companion_path(
        release_path, platforms_path, DEFAULT_PLATFORMS
    )
    resolved_build_requirements = build_requirements_path or DEFAULT_BUILD_REQUIREMENTS
    build_requirements = _exact_requirements(
        resolved_build_requirements.read_text(encoding="utf-8").splitlines(),
        str(resolved_build_requirements),
    )
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    runtime_requirements = _exact_requirements(
        project["project"]["dependencies"], f"{project_path}:project.dependencies"
    )
    locked_build = {
        canonicalize_name(requirement.name): next(iter(requirement.specifier)).version
        for requirement in map(Requirement, build_requirements)
    }
    for raw in project["build-system"]["requires"]:
        requirement = Requirement(raw)
        locked_version = locked_build.get(canonicalize_name(requirement.name))
        if locked_version is None or locked_version not in requirement.specifier:
            raise ValueError(
                f"{resolved_build_requirements}: Builder pin does not satisfy {raw!r} "
                f"from {project_path}"
            )
    schema = serialization.load_json(schema_path)
    release = serialization.load_yaml(release_path)
    platforms = serialization.load_yaml(resolved_platforms)
    serialization.validate_schema(
        release, schema["$defs"]["releasePolicy"], root=schema, path="$.release"
    )
    serialization.validate_schema(
        platforms, schema["$defs"]["platformPolicy"], root=schema, path="$.platforms"
    )
    _validate_release_semantics(release)
    _validate_platform_semantics(platforms)
    return {
        "release": release,
        "platforms": platforms,
        "requirements": {
            "wheel_build": build_requirements,
            "wheel_runtime": runtime_requirements,
        },
    }


def resolve(
    release_path: Path = DEFAULT_RELEASE,
    *,
    platforms_path: Path | None = None,
    repository_root: Path = REPO_ROOT,
    repository: str | None = None,
    version_override: str | None = None,
    release_type: str = "stable",
    fork_test_pypi: bool = False,
    dockerhub_namespace: str | None = None,
) -> dict[str, Any]:
    """Resolve the two human policies into the formal runtime authority."""
    bundle = load(
        release_path,
        platforms_path=platforms_path,
        project_path=repository_root / "pyproject.toml",
        build_requirements_path=repository_root / "requirements-build.txt",
    )
    release = copy.deepcopy(bundle["release"])
    platforms = copy.deepcopy(bundle["platforms"])
    resolved_repository = resolve_repository(
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
        dockerhub_namespace=dockerhub_namespace,
    )
    normalized_publish["pypi"]["distribution_prefix"] = pypi_distribution_prefix(
        resolved_repository
    )
    merged["publish"] = normalized_publish
    merged["release_type"] = release_type
    merged["release_profile"] = selected_profile
    version_authority = version_config.load(repository_root / "version.ini")
    version = version_override or str(version_authority["ucm_version"])
    selectors = version_authority["supported_runtimes"]
    for product in merged["products"]:
        product["runtime_selectors"] = copy.deepcopy(selectors[product["id"]])
    chart_document = serialization.load_yaml(
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
            "version": version_config.derive_chart_version(version),
            "app_version": version,
        }
    )
    return merged
