"""Build and inspect the empty ``uc-manager`` public PyPI meta package."""

from __future__ import annotations

import email.parser
import hashlib
import re
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import Tag
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

from . import policy as release_policy

META_BASE_DISTRIBUTION = "uc-manager"
META_TAG = "py3-none-any"
META_RESULT_KIND = "ucm-meta-result"
META_RESULT_SCHEMA_VERSION = 1
META_SOURCE_DATE_EPOCH = 315532800
_EXTRA = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _canonical_version(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty PEP 440 version")
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"{context} must be a valid PEP 440 version") from error
    if parsed.local is not None:
        raise ValueError(f"{context} must not contain a local version")
    normalized = str(parsed)
    if value != normalized:
        raise ValueError(
            f"{context} must use canonical PEP 440 spelling {normalized!r}"
        )
    return normalized


def _distribution_family(plan: Mapping[str, Any]) -> tuple[str, str]:
    repository = plan.get("repository")
    if not isinstance(repository, str):
        raise ValueError("release plan repository must be owner/name")
    expected_scope, _ = release_policy.publication_identity(repository)
    if plan.get("publication_scope") != expected_scope:
        raise ValueError("release plan publication scope differs from repository")
    publish = _mapping(plan.get("publish"), "release plan publish")
    pypi = _mapping(publish.get("pypi"), "release plan publish.pypi")
    prefix = pypi.get("distribution_prefix")
    expected_prefix = release_policy.pypi_distribution_prefix(repository)
    if prefix != expected_prefix:
        raise ValueError(
            "release plan Python distribution prefix differs from repository"
        )
    meta_distribution = f"{expected_prefix}{META_BASE_DISTRIBUTION}"
    if (
        len(meta_distribution) > release_policy.MAX_PYPI_DISTRIBUTION_LENGTH
        or canonicalize_name(meta_distribution, validate=True) != meta_distribution
    ):
        raise ValueError("release plan meta distribution is not canonical")
    return meta_distribution, f"{meta_distribution}-"


def _canonical_exact_requirement(
    value: object, context: str, backend_prefix: str
) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be one exact requirement")
    try:
        requirement = Requirement(value)
    except InvalidRequirement as error:
        raise ValueError(f"{context} must be one exact requirement") from error
    specifiers = list(requirement.specifier)
    if (
        requirement.url is not None
        or requirement.marker is not None
        or requirement.extras
        or len(specifiers) != 1
        or specifiers[0].operator != "=="
        or "*" in specifiers[0].version
    ):
        raise ValueError(f"{context} must pin one distribution with ==")
    name = canonicalize_name(requirement.name)
    suffix = name.removeprefix(backend_prefix)
    if (
        len(name) > release_policy.MAX_PYPI_DISTRIBUTION_LENGTH
        or not name.startswith(backend_prefix)
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", suffix) is None
    ):
        raise ValueError(f"{context} must reference a planned UCM backend family")
    version = _canonical_version(specifiers[0].version, f"{context} version")
    canonical = f"{name}=={version}"
    if value != canonical:
        raise ValueError(f"{context} must use canonical requirement {canonical!r}")
    return name, canonical


def validate_meta_package(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize ``plan.meta_package`` against planned Wheels."""
    plan = _mapping(plan, "release plan")
    if plan.get("kind") != "ucm-release-plan":
        raise ValueError("meta package requires a ucm-release-plan")
    expected_distribution, backend_prefix = _distribution_family(plan)
    plan_version = _canonical_version(plan.get("version"), "release plan version")
    meta = _mapping(plan.get("meta_package"), "release plan meta_package")
    expected_fields = {"distribution", "version", "extras"}
    if set(meta) != expected_fields:
        raise ValueError("release plan meta_package fields must be exact")
    if meta.get("distribution") != expected_distribution:
        raise ValueError(f"meta package distribution must be {expected_distribution!r}")
    version = _canonical_version(meta.get("version"), "meta package version")
    if version != plan_version:
        raise ValueError("meta package version must match release plan version")

    extras = _mapping(meta.get("extras"), "meta package extras")
    if not extras:
        raise ValueError("meta package extras must not be empty")
    normalized_extras: dict[str, str] = {}
    backend_names: set[str] = set()
    for raw_extra, raw_requirement in extras.items():
        if (
            not isinstance(raw_extra, str)
            or _EXTRA.fullmatch(raw_extra) is None
            or canonicalize_name(raw_extra) != raw_extra
        ):
            raise ValueError(f"invalid canonical meta package extra: {raw_extra!r}")
        backend_name, requirement = _canonical_exact_requirement(
            raw_requirement, f"meta package extra {raw_extra!r}", backend_prefix
        )
        if requirement.removeprefix(f"{backend_name}==") != version:
            raise ValueError(f"meta package extra {raw_extra!r} must pin {version}")
        if backend_name in backend_names:
            raise ValueError("meta package extras must reference unique backends")
        backend_names.add(backend_name)
        normalized_extras[raw_extra] = requirement

    wheels = plan.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise ValueError("release plan wheels must be a non-empty array")
    planned_backends: set[str] = set()
    planned_extras: dict[str, str] = {}
    for index, raw_wheel in enumerate(wheels):
        wheel = _mapping(raw_wheel, f"release plan wheels[{index}]")
        distribution = wheel.get("dist_name")
        if not isinstance(distribution, str):
            raise ValueError(f"release plan wheels[{index}].dist_name is invalid")
        canonical_distribution = canonicalize_name(distribution)
        if (
            distribution != canonical_distribution
            or len(distribution) > release_policy.MAX_PYPI_DISTRIBUTION_LENGTH
            or not distribution.startswith(backend_prefix)
            or re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*",
                distribution.removeprefix(backend_prefix),
            )
            is None
        ):
            raise ValueError(f"release plan wheels[{index}].dist_name is invalid")
        wheel_version = _canonical_version(
            wheel.get("wheel_version"), f"release plan wheels[{index}].wheel_version"
        )
        if wheel_version != version:
            raise ValueError(
                "planned backend Wheel version must match meta package version"
            )
        planned_backends.add(distribution)

        runtime_variant = wheel.get("runtime_variant")
        if (
            not isinstance(runtime_variant, str)
            or _EXTRA.fullmatch(runtime_variant) is None
            or canonicalize_name(runtime_variant) != runtime_variant
        ):
            raise ValueError(
                f"release plan wheels[{index}].runtime_variant is not a valid extra"
            )
        requirement = f"{distribution}=={version}"
        existing = planned_extras.setdefault(runtime_variant, requirement)
        if existing != requirement:
            raise ValueError(
                f"planned runtime variant {runtime_variant!r} maps to multiple backends"
            )
    expected_extras = {key: planned_extras[key] for key in sorted(planned_extras)}
    if normalized_extras != expected_extras or backend_names != planned_backends:
        missing = sorted(set(expected_extras.items()) - set(normalized_extras.items()))
        unexpected = sorted(
            set(normalized_extras.items()) - set(expected_extras.items())
        )
        raise ValueError(
            "meta package extras must exactly match planned runtime variants: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "distribution": expected_distribution,
        "version": version,
        "extras": {key: normalized_extras[key] for key in sorted(normalized_extras)},
    }


def _metadata_requirements(meta: Mapping[str, Any]) -> list[str]:
    return sorted(
        f'{requirement}; extra == "{extra}"'
        for extra, requirement in meta["extras"].items()
    )


def materialize_meta_source(plan: Mapping[str, Any], output_dir: Path) -> Path:
    """Write a deterministic, package-free PEP 621 source tree."""
    meta = validate_meta_package(plan)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("meta package source directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    optional_dependencies = "\n".join(
        f'"{extra}" = ["{requirement}"]'
        for extra, requirement in meta["extras"].items()
    )
    pyproject = "\n".join(
        (
            "[build-system]",
            'requires = ["setuptools==75.8.2", "wheel==0.45.1"]',
            'build-backend = "setuptools.build_meta"',
            "",
            "[project]",
            f'name = "{meta["distribution"]}"',
            f'version = "{meta["version"]}"',
            'description = "Unified Cache Management backend selector"',
            'requires-python = ">=3.10"',
            "dependencies = []",
            "",
            "[project.optional-dependencies]",
            optional_dependencies,
            "",
            "[tool.setuptools]",
            "packages = []",
            "",
        )
    )
    path = output_dir / "pyproject.toml"
    path.write_text(pyproject, encoding="utf-8")
    return path


def _single_header(message: Any, name: str, context: str) -> str:
    values = message.get_all(name, [])
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError(f"meta Wheel {context} must contain exactly one {name}")
    return values[0]


def _canonical_metadata_requirement(value: str, backend_prefix: str) -> str:
    try:
        requirement = Requirement(value)
    except InvalidRequirement as error:
        raise ValueError(
            "meta Wheel METADATA contains an invalid Requires-Dist"
        ) from error
    marker = requirement.marker
    requirement.marker = None
    _, canonical = _canonical_exact_requirement(
        str(requirement), "meta Wheel METADATA Requires-Dist", backend_prefix
    )
    if marker is None:
        raise ValueError("meta Wheel dependencies must be guarded by one extra")
    return f"{canonical}; {marker}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def record_meta_wheel(plan: Mapping[str, Any], wheel_path: Path) -> dict[str, Any]:
    """Validate one empty meta Wheel and return its deterministic result record."""
    meta = validate_meta_package(plan)
    wheel_path = Path(wheel_path)
    if not wheel_path.is_file():
        raise ValueError(f"meta Wheel is missing: {wheel_path}")
    try:
        distribution, version, build, tags = parse_wheel_filename(wheel_path.name)
    except InvalidWheelFilename as error:
        raise ValueError(f"invalid meta Wheel filename: {wheel_path.name}") from error
    if canonicalize_name(str(distribution)) != meta["distribution"]:
        raise ValueError("meta Wheel filename distribution does not match the plan")
    if str(version) != meta["version"]:
        raise ValueError("meta Wheel filename version does not match the plan")
    if build:
        raise ValueError("meta Wheel filename must not contain a build tag")
    if tags != frozenset({Tag("py3", "none", "any")}):
        raise ValueError(f"meta Wheel filename tag must be {META_TAG}")
    filename_distribution = meta["distribution"].replace("-", "_")
    expected_filename = f"{filename_distribution}-{meta['version']}-{META_TAG}.whl"
    if wheel_path.name != expected_filename:
        raise ValueError(f"meta Wheel filename must be exactly {expected_filename}")

    try:
        with zipfile.ZipFile(wheel_path) as integrity_archive:
            bad_member = integrity_archive.testzip()
    except zipfile.BadZipFile as error:
        raise ValueError("meta Wheel ZIP is corrupt") from error
    if bad_member is not None:
        raise ValueError(f"meta Wheel CRC is corrupt: {bad_member}")

    expected_dist_info = f"{filename_distribution}-{meta['version']}.dist-info/"
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("meta Wheel contains duplicate members")
        if any(not name.startswith(expected_dist_info) for name in names):
            raise ValueError("meta Wheel must not contain package payload files")

        metadata_name = expected_dist_info + "METADATA"
        wheel_metadata_name = expected_dist_info + "WHEEL"
        record_name = expected_dist_info + "RECORD"
        for required_name in (metadata_name, wheel_metadata_name, record_name):
            if names.count(required_name) != 1:
                raise ValueError(
                    f"meta Wheel requires exactly one {Path(required_name).name}"
                )
        try:
            parser = email.parser.Parser()
            metadata = parser.parsestr(archive.read(metadata_name).decode("utf-8"))
            wheel_metadata = parser.parsestr(
                archive.read(wheel_metadata_name).decode("utf-8")
            )
        except UnicodeDecodeError as error:
            raise ValueError("meta Wheel metadata must be UTF-8") from error

        metadata_distribution = _single_header(metadata, "Name", "METADATA")
        metadata_version = _single_header(metadata, "Version", "METADATA")
        requires_python = _single_header(metadata, "Requires-Python", "METADATA")
        if metadata_distribution != meta["distribution"]:
            raise ValueError("meta Wheel METADATA distribution does not match the plan")
        if metadata_version != meta["version"]:
            raise ValueError("meta Wheel METADATA version does not match the plan")
        if requires_python != ">=3.10":
            raise ValueError("meta Wheel Requires-Python must be >=3.10")

        raw_extras = metadata.get_all("Provides-Extra", [])
        actual_extras = [canonicalize_name(extra) for extra in raw_extras]
        expected_extras = list(meta["extras"])
        if (
            any(raw != normalized for raw, normalized in zip(raw_extras, actual_extras))
            or len(set(actual_extras)) != len(actual_extras)
            or sorted(actual_extras) != expected_extras
        ):
            raise ValueError("meta Wheel Provides-Extra does not match the plan")
        raw_requirements = metadata.get_all("Requires-Dist", [])
        actual_requirements = sorted(
            _canonical_metadata_requirement(requirement, f"{meta['distribution']}-")
            for requirement in raw_requirements
        )
        expected_requirements = _metadata_requirements(meta)
        if (
            len(set(actual_requirements)) != len(actual_requirements)
            or actual_requirements != expected_requirements
        ):
            raise ValueError("meta Wheel Requires-Dist does not match the plan")

        root_is_purelib = _single_header(wheel_metadata, "Root-Is-Purelib", "WHEEL")
        metadata_tags = wheel_metadata.get_all("Tag", [])
        if root_is_purelib.lower() != "true" or metadata_tags != [META_TAG]:
            raise ValueError(
                "meta Wheel WHEEL metadata must declare py3-none-any purelib"
            )

    return {
        "kind": META_RESULT_KIND,
        "schema_version": META_RESULT_SCHEMA_VERSION,
        "distribution": meta["distribution"],
        "version": meta["version"],
        "filename": wheel_path.name,
        "sha256": _sha256(wheel_path),
        "size": wheel_path.stat().st_size,
        "tags": [META_TAG],
        "extras": meta["extras"],
        "requires_dist": expected_requirements,
    }
