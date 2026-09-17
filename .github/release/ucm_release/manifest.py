"""Build and validate the current public UCM release manifest."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version
from ucm_release import toolkit as toolkit_ops

RELEASE_MANIFEST_FILENAME = "release-manifest.json"
RELEASE_MANIFEST_KIND = "ucm-release-manifest"
RELEASE_MANIFEST_SCHEMA_VERSION = 9
_PATH_COMPONENT = re.compile(r"[a-z0-9][a-z0-9.+-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """A published installation manifest is invalid or cannot be read."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ManifestError(
            f"{context} fields differ; missing={missing}, extra={extra}"
        )


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} must be an object")
    return value


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ManifestError(f"{context} must be a non-empty string")
    return value


def _string_array(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ManifestError(f"{context} must be an array of non-empty strings")
    return value


def _sorted_string_array(value: object, context: str) -> list[str]:
    items = _string_array(value, context)
    if items != sorted(set(items)):
        raise ManifestError(f"{context} must be sorted and unique")
    return items


def _https_url(value: object, context: str, *, host: str | None = None) -> str:
    url = _nonempty_string(value, context)
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or (host and parsed.netloc != host)
    ):
        raise ManifestError(f"{context} must be an HTTPS URL")
    return url


def _validate_wheel_filename(
    wheel: Mapping[str, Any], platform_tags: Sequence[str], context: str
) -> None:
    try:
        distribution, version, build, tags = parse_wheel_filename(wheel["filename"])
    except ValueError as error:
        raise ManifestError(f"{context} filename is not a valid Wheel") from error
    architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(
        wheel["architecture"], wheel["architecture"]
    )
    if (
        canonicalize_name(str(distribution)) != canonicalize_name(wheel["distribution"])
        or str(version) != wheel["version"]
        or build
        or {tag.interpreter for tag in tags} != {wheel["python_abi"]}
        or {tag.abi for tag in tags} != {wheel["python_abi"]}
        or {tag.platform for tag in tags} != set(platform_tags)
        or len(platform_tags) != 1
        or not platform_tags[0].endswith(f"_{architecture}")
    ):
        raise ManifestError(f"{context} filename and platform identity differ")


def _accelerator(value: object, context: str) -> dict[str, Any]:
    accelerator = _mapping(value, context)
    _exact_keys(accelerator, {"runtime", "variant", "soc_version"}, context)
    for field in ("runtime", "variant", "soc_version"):
        _nonempty_string(accelerator.get(field), f"{context} {field}")
    return accelerator


def _publication(value: object, context: str, channel: str) -> dict[str, Any] | None:
    if value is None:
        return None
    publication = _mapping(value, context)
    _exact_keys(publication, {"pull", "multi_arch", "members"}, context)
    pull = _tagged_oci_reference(
        publication.get("pull"),
        f"{context} pull",
        registry="ghcr.io" if channel == "ghcr" else "docker.io",
    )
    multi_arch = publication.get("multi_arch")
    if not isinstance(multi_arch, bool):
        raise ManifestError(f"{context} multi_arch must be a boolean")
    members = publication.get("members")
    if not isinstance(members, list) or not members:
        raise ManifestError(f"{context} members must be a non-empty array")
    architectures: set[str] = set()
    references: set[str] = set()
    for index, raw_member in enumerate(members):
        member_context = f"{context} members[{index}]"
        member = _mapping(raw_member, member_context)
        _exact_keys(member, {"architecture", "reference"}, member_context)
        architecture = _nonempty_string(
            member.get("architecture"), f"{member_context} architecture"
        )
        reference = _nonempty_string(
            member.get("reference"), f"{member_context} reference"
        )
        _tagged_oci_reference(
            reference,
            f"{member_context} reference",
            registry="ghcr.io" if channel == "ghcr" else "docker.io",
        )
        if architecture in architectures or reference in references:
            raise ManifestError(f"{context} members must be unique")
        architectures.add(architecture)
        references.add(reference)
    if multi_arch and pull in references:
        raise ManifestError(f"{context} index and members must be unique")
    if multi_arch and len(members) < 2:
        raise ManifestError(f"{context} multi_arch requires at least two members")
    if not multi_arch and (len(members) != 1 or members[0]["reference"] != pull):
        raise ManifestError(
            f"{context} single-architecture pull must equal its only member"
        )
    return publication


def validate_manifest(
    value: object, *, expected_tag: str | None = None
) -> dict[str, Any]:
    """Validate the exact public Schema 9 installation contract."""

    manifest = _mapping(value, "release manifest")
    if manifest.get("kind") != RELEASE_MANIFEST_KIND:
        raise ManifestError(f"release manifest kind must be {RELEASE_MANIFEST_KIND}")
    schema_version = manifest.get("schema_version")
    if schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            "release manifest schema_version must be "
            f"{RELEASE_MANIFEST_SCHEMA_VERSION}"
        )
    _exact_keys(
        manifest,
        {
            "kind",
            "schema_version",
            "release",
            "python",
            "wheels",
            "images",
            "chart",
            "github_release_assets",
        }
        | ({"toolkit"} if "toolkit" in manifest else set()),
        "release manifest",
    )

    release = _mapping(manifest.get("release"), "release manifest release")
    _exact_keys(
        release,
        {"tag", "type", "version", "url", "actions_run_id"},
        "release manifest release",
    )
    for field in ("tag", "type", "version", "url"):
        _nonempty_string(release.get(field), f"release manifest release {field}")
    if release["type"] not in {"stable", "prerelease", "draft", "nightly"}:
        raise ManifestError("release manifest release type is invalid")
    if _PATH_COMPONENT.fullmatch(release["version"]) is None:
        raise ManifestError("release manifest release version is not path-safe")
    if expected_tag is not None and release["tag"] != expected_tag:
        raise ManifestError("release manifest Tag differs from the requested Tag")
    actions_run_id = release.get("actions_run_id")
    if (
        not isinstance(actions_run_id, int)
        or isinstance(actions_run_id, bool)
        or actions_run_id < 1
    ):
        raise ManifestError(
            "release manifest actions_run_id must be a positive integer"
        )

    python_package = _mapping(manifest.get("python"), "release manifest python")
    _exact_keys(
        python_package,
        {
            "distribution",
            "version",
            "extras",
            "pypi",
        },
        "release manifest python",
    )
    for field in ("distribution", "version"):
        _nonempty_string(python_package.get(field), f"release manifest python {field}")
    meta_distribution = python_package["distribution"]
    if (
        re.fullmatch(r"(?:[a-z0-9]+-)*uc-manager", meta_distribution) is None
        or python_package["version"] != release["version"]
    ):
        raise ManifestError("release manifest Python package must match the release")
    raw_extras = _mapping(
        python_package.get("extras"), "release manifest Python extras"
    )
    if not raw_extras:
        raise ManifestError("release manifest Python extras must not be empty")
    python_extras: dict[str, str] = {}
    distributions: set[str] = set()
    for extra in sorted(raw_extras):
        distribution = _nonempty_string(
            raw_extras[extra], f"release manifest Python extra {extra}"
        )
        if (
            _PATH_COMPONENT.fullmatch(extra) is None
            or not (
                distribution.startswith(f"{meta_distribution}-")
                or (
                    extra == "toolkit"
                    and "toolkit" in manifest
                    and distribution
                    == meta_distribution.removesuffix("uc-manager") + "ucm-toolkit"
                )
            )
            or distribution in distributions
        ):
            raise ManifestError("release manifest Python extras are invalid")
        distributions.add(distribution)
        python_extras[extra] = distribution
    pypi = python_package.get("pypi")
    if pypi is not None:
        pypi = _mapping(pypi, "release manifest Python PyPI")
        _exact_keys(
            pypi,
            {"index_url", "project_url"},
            "release manifest Python PyPI",
        )
        _https_url(
            pypi.get("index_url"),
            "release manifest Python PyPI index",
        )
        pypi_host = urlparse(pypi["index_url"]).netloc
        if pypi_host not in {"pypi.org", "test.pypi.org"}:
            raise ManifestError("release manifest Python PyPI host is invalid")
        _https_url(
            pypi.get("project_url"),
            "release manifest Python PyPI project",
            host=pypi_host,
        )
        if (
            pypi["index_url"] != f"https://{pypi_host}/simple"
            or pypi["project_url"]
            != f"https://{pypi_host}/project/{meta_distribution}/"
            f"{quote(release['version'], safe='')}/"
        ):
            raise ManifestError("release manifest PyPI URLs differ from the release")

    wheels = manifest.get("wheels")
    if not isinstance(wheels, list):
        raise ManifestError("release manifest wheels must be an array")
    wheel_keys = {
        "id",
        "product",
        "extra",
        "accelerator",
        "distribution",
        "version",
        "python_abi",
        "architecture",
        "platform_tags",
        "filename",
        "url",
        "sha256",
        "dependencies",
    }
    wheel_ids: set[str] = set()
    wheel_filenames: set[str] = set()
    wheel_extras: set[str] = set()
    for index, raw_wheel in enumerate(wheels):
        context = f"release manifest wheels[{index}]"
        wheel = _mapping(raw_wheel, context)
        _exact_keys(wheel, wheel_keys, context)
        for field in wheel_keys - {"accelerator", "dependencies", "platform_tags"}:
            _nonempty_string(wheel.get(field), f"{context} {field}")
        wheel_id = wheel["id"]
        filename = wheel["filename"]
        if wheel_id in wheel_ids or filename in wheel_filenames:
            raise ManifestError(
                "release manifest Wheel IDs and filenames must be unique"
            )
        wheel_ids.add(wheel_id)
        wheel_filenames.add(filename)
        _accelerator(wheel.get("accelerator"), f"{context} accelerator")
        if _PATH_COMPONENT.fullmatch(wheel["extra"]) is None:
            raise ManifestError("Wheel extra is not path-safe")
        if (
            python_extras.get(wheel["extra"]) != wheel["distribution"]
            or wheel["version"] != manifest["python"]["version"]
        ):
            raise ManifestError("Wheel does not match its declared Python extra")
        platform_tags = _sorted_string_array(
            wheel.get("platform_tags"), f"{context} platform_tags"
        )
        if not platform_tags:
            raise ManifestError("Wheel platform tags must not be empty")
        _validate_wheel_filename(wheel, platform_tags, context)
        wheel_extras.add(wheel["extra"])
        if Path(filename).name != filename:
            raise ManifestError("Wheel filename must not contain a path")
        if _SHA256.fullmatch(wheel["sha256"]) is None:
            raise ManifestError("Wheel sha256 must contain 64 lowercase hex digits")
        dependencies = _string_array(
            wheel.get("dependencies"), f"{context} dependencies"
        )
        if dependencies != sorted(set(dependencies)):
            raise ManifestError(f"{context} dependencies must be sorted and unique")
    toolkit_package = manifest.get("toolkit")
    if toolkit_package is not None:
        _validate_toolkit_manifest(toolkit_package, manifest)
    if wheel_extras != set(python_extras) - (
        {"toolkit"} if toolkit_package is not None else set()
    ):
        raise ManifestError("release manifest Wheels must publish every Python extra")

    images = manifest.get("images")
    if not isinstance(images, list):
        raise ManifestError("release manifest images must be an array")
    image_ids: set[str] = set()
    for index, raw_image in enumerate(images):
        context = f"release manifest images[{index}]"
        image = _mapping(raw_image, context)
        _exact_keys(
            image,
            {"id", "product", "upstream", "accelerator", "os", "publications"},
            context,
        )
        image_id = _nonempty_string(image.get("id"), f"{context} id")
        _nonempty_string(image.get("product"), f"{context} product")
        if image_id in image_ids:
            raise ManifestError("release manifest Image IDs must be unique")
        image_ids.add(image_id)
        upstream = _mapping(image.get("upstream"), f"{context} upstream")
        _exact_keys(upstream, {"version", "channel"}, f"{context} upstream")
        for field in ("version", "channel"):
            _nonempty_string(upstream.get(field), f"{context} upstream {field}")
        _accelerator(image.get("accelerator"), f"{context} accelerator")
        operating_system = _mapping(image.get("os"), f"{context} os")
        _exact_keys(operating_system, {"id", "version"}, f"{context} os")
        for field in ("id", "version"):
            _nonempty_string(operating_system.get(field), f"{context} os {field}")
        publications = _mapping(image.get("publications"), f"{context} publications")
        _exact_keys(publications, {"ghcr", "dockerhub"}, f"{context} publications")
        published = [
            _publication(
                publications.get(channel), f"{context} publications {channel}", channel
            )
            for channel in ("ghcr", "dockerhub")
        ]
        if all(publication is None for publication in published):
            raise ManifestError(f"{context} must have at least one publication")

    chart = manifest["chart"]
    required_assets = set(wheel_filenames)
    if chart is not None:
        chart = _mapping(chart, "release manifest chart")
        _exact_keys(
            chart,
            {"name", "version", "filename", "url", "oci"},
            "release manifest chart",
        )
        for field in ("name", "version", "filename", "url"):
            _nonempty_string(chart.get(field), f"release manifest chart {field}")
        chart_filename = chart["filename"]
        if Path(chart_filename).name != chart_filename:
            raise ManifestError(
                "release manifest Chart filename must not contain a path"
            )
        if chart_filename in wheel_filenames:
            raise ManifestError("release manifest Chart and Wheel files must be unique")
        if chart.get("oci") is not None:
            _tagged_oci_reference(
                chart.get("oci"), "release manifest chart oci", registry="ghcr.io"
            )
        required_assets.add(chart_filename)

    assets = _string_array(
        manifest.get("github_release_assets"),
        "release manifest github_release_assets",
    )
    if len(assets) != len(set(assets)):
        raise ManifestError("release manifest github_release_assets must be unique")
    if RELEASE_MANIFEST_FILENAME not in assets:
        raise ManifestError(
            "release manifest must list itself as a GitHub Release asset"
        )
    if toolkit_package is not None:
        required_assets.add(toolkit_package["filename"])
    missing_assets = sorted(required_assets - set(assets))
    if missing_assets:
        raise ManifestError(f"release manifest assets are missing {missing_assets}")
    return manifest


def _validate_toolkit_manifest(value: object, manifest: dict[str, Any]) -> None:
    package = _mapping(value, "release manifest toolkit")
    _exact_keys(
        package,
        {"distribution", "version", "filename", "url", "sha256"},
        "release manifest toolkit",
    )
    expected = {
        "distribution": manifest["python"]["distribution"].removesuffix("uc-manager")
        + "ucm-toolkit",
        "version": manifest["release"]["version"],
    }
    result = {key: package[key] for key in ("distribution", "version", "filename")}
    result.update(
        kind="ucm-toolkit-result",
        schema_version=1,
        sha256=f"sha256:{package['sha256']}",
    )
    try:
        toolkit_ops.validate_result(expected, result)
    except (ValueError, TypeError) as error:
        raise ManifestError(str(error)) from error
    if manifest["python"]["extras"].get("toolkit") != expected["distribution"]:
        raise ManifestError("toolkit extra differs from the published toolkit")
    _https_url(package["url"], "toolkit URL", host="github.com")
    expected_url = (
        manifest["release"]["url"].replace("/releases/tag/", "/releases/download/")
        + "/"
        + quote(package["filename"], safe="")
    )
    if package["url"] != expected_url:
        raise ManifestError("toolkit download URL differs from its Release")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(
            f"unable to read release manifest {path}: {error}"
        ) from error
    return validate_manifest(value)


def require_stable_manifest(manifest: Mapping[str, Any]) -> None:
    release = _mapping(manifest.get("release"), "release manifest release")
    if release.get("type") != "stable":
        raise ManifestError("Stable Manifest release type must be stable")
    version_text = _nonempty_string(release.get("version"), "Stable version")
    tag = _nonempty_string(release.get("tag"), "Stable Tag")
    try:
        version = Version(version_text)
    except InvalidVersion as error:
        raise ManifestError(
            f"Stable Manifest has invalid PEP 440 version {version_text!r}"
        ) from error
    if tag != f"v{version_text}":
        raise ManifestError(
            "Stable Manifest Tag must equal 'v' plus the public version"
        )
    if version.is_prerelease or version.is_devrelease or version.local is not None:
        raise ManifestError(
            "Stable Manifest version must not contain pre/dev/local segments"
        )


_OCI_REFERENCE = re.compile(
    r"(?P<repository>(?:ghcr\.io|docker\.io)/"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+)"
    r":(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})"
)


def _tagged_oci_reference(value: object, context: str, *, registry: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{context} must be a tagged OCI reference")
    match = _OCI_REFERENCE.fullmatch(value)
    if match is None or not match.group("repository").startswith(registry + "/"):
        raise ManifestError(f"{context} must be a tagged {registry} reference")
    return value


def _list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _release_page_url(release_document: dict[str, Any]) -> str:
    url = release_document.get("html_url")
    parsed = urlparse(url) if isinstance(url, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or not parsed.path
    ):
        raise ValueError("GitHub Release has an invalid page URL")
    return url


def _publication_references(record: dict[str, Any], context: str) -> dict[str, str]:
    references: dict[str, str] = {}
    for index, raw_target in enumerate(
        _list(record.get("targets"), f"{context} publication targets")
    ):
        target = _mapping(raw_target, f"{context} publication targets[{index}]")
        channel = target.get("channel")
        reference = target.get("reference")
        if (
            not isinstance(channel, str)
            or channel not in {"ghcr", "dockerhub"}
            or channel in references
            or not isinstance(reference, str)
            or not reference
        ):
            raise ValueError(f"{context} has invalid publication references")
        _target_repository(reference, f"{context} {channel} reference")
        references[channel] = reference
    if not references:
        raise ValueError(f"{context} has no published references")
    return {channel: references[channel] for channel in sorted(references)}


def _wheel_capability(
    wheel_id: str, images: list[dict[str, Any]]
) -> tuple[str, dict[str, str]]:
    capabilities: set[tuple[str, str, str, str]] = set()
    for image in images:
        if image.get("wheel_id") != wheel_id:
            continue
        runtime = _mapping(image.get("runtime"), f"Wheel {wheel_id} Runtime")
        values = tuple(
            runtime.get(field)
            for field in (
                "product_id",
                "accelerator_runtime",
                "variant",
                "soc_version",
            )
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"Wheel {wheel_id} has an invalid Runtime capability")
        capabilities.add(values)  # type: ignore[arg-type]
    if not capabilities:
        raise ValueError(f"Wheel {wheel_id} is not linked to an Image family")
    if len(capabilities) != 1:
        raise ValueError(f"Wheel {wheel_id} maps to conflicting Runtime capabilities")
    product, accelerator_runtime, variant, soc_version = capabilities.pop()
    return product, {
        "runtime": accelerator_runtime,
        "variant": variant,
        "soc_version": soc_version,
    }


def _project_image_publications(
    family: dict[str, Any], members: list[dict[str, Any]], family_id: str
) -> dict[str, dict[str, Any] | None]:
    pull_references = _publication_references(family, f"release family {family_id}")
    create_index = family.get("create_index")
    if not isinstance(create_index, bool):
        raise ValueError(f"release family {family_id} has invalid index ownership")
    if create_index and len(members) < 2:
        raise ValueError(
            f"multi-architecture family {family_id} requires at least two members"
        )

    members_by_channel: dict[str, list[dict[str, str]]] = {
        "ghcr": [],
        "dockerhub": [],
    }
    for member in members:
        architecture = member.get("cpu_arch")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError(f"release family {family_id} has invalid architectures")
        references = _publication_references(
            member, f"release family {family_id} member {architecture}"
        )
        if set(references) != set(pull_references):
            raise ValueError(
                f"release family {family_id} has inconsistent publication channels"
            )
        for channel, reference in references.items():
            members_by_channel[channel].append(
                {"architecture": architecture, "reference": reference}
            )

    result: dict[str, dict[str, Any] | None] = {}
    for channel in ("ghcr", "dockerhub"):
        if channel not in pull_references:
            result[channel] = None
            continue
        channel_members = sorted(
            members_by_channel[channel], key=lambda item: item["architecture"]
        )
        if len({item["architecture"] for item in channel_members}) != len(
            channel_members
        ):
            raise ValueError(f"release family {family_id} has duplicate architectures")
        if not create_index and (
            len(channel_members) != 1
            or pull_references[channel] != channel_members[0]["reference"]
        ):
            raise ValueError(
                f"single-architecture family {family_id} has an ambiguous pull reference"
            )
        result[channel] = {
            "pull": pull_references[channel],
            "multi_arch": create_index,
            "members": channel_members,
        }
    return result


def _canonical_public_version(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a version string")
    try:
        parsed = Version(value)
    except InvalidVersion as error:
        raise ValueError(f"{context} is not a valid PEP 440 version") from error
    if parsed.local is not None or str(parsed) != value:
        raise ValueError(
            f"{context} must be canonical and must not use a local version"
        )
    return value


def _public_sha256(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} has an invalid SHA256")
    normalized = value.removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{context} has an invalid SHA256")
    return normalized


def _project_python_package(
    state: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    release = _mapping(state.get("release"), "release state release")
    version = _canonical_public_version(release.get("version"), "release state version")
    meta = _mapping(state.get("meta_package"), "release state meta package")
    meta_distribution = meta.get("distribution")
    if (
        not isinstance(meta_distribution, str)
        or re.fullmatch(r"(?:[a-z0-9]+-)*uc-manager", meta_distribution) is None
        or meta.get("version") != version
    ):
        raise ValueError("release state meta package identity is invalid")
    backend_extras: dict[str, str] = {}
    for index, raw_wheel in enumerate(
        _list(state.get("wheels"), "release state Wheels")
    ):
        wheel = _mapping(raw_wheel, f"release state Wheels[{index}]")
        extra = wheel.get("runtime_variant")
        distribution = wheel.get("distribution")
        wheel_version = _canonical_public_version(
            wheel.get("version"), f"release state Wheels[{index}] version"
        )
        if (
            not isinstance(extra, str)
            or not extra
            or not isinstance(distribution, str)
            or not distribution.startswith(f"{meta_distribution}-")
            or wheel_version != version
            or (extra in backend_extras and backend_extras[extra] != distribution)
        ):
            raise ValueError("release state Wheel extra mapping is invalid")
        backend_extras[extra] = distribution

    toolkit_result = state.get("toolkit_package")
    all_extras = dict(backend_extras)
    if toolkit_result is not None:
        expected_toolkit = {
            "distribution": meta_distribution.removesuffix("uc-manager")
            + "ucm-toolkit",
            "version": version,
        }
        toolkit_ops.validate_result(expected_toolkit, toolkit_result)
        all_extras["toolkit"] = toolkit_result["distribution"]
    planned_extras = _mapping(meta.get("extras"), "release state meta package extras")
    if set(planned_extras) != set(all_extras):
        raise ValueError("release state meta package extras differ from backend Wheels")
    for extra, distribution in all_extras.items():
        if planned_extras.get(extra) != f"{distribution}=={version}":
            raise ValueError("release state meta package requirement is invalid")

    publish = _mapping(state.get("publish"), "release state publication plan")
    pypi_plan = _mapping(publish.get("pypi"), "release state PyPI plan")
    pypi: dict[str, str] | None = None
    if pypi_plan.get("enabled") is True:
        receipt = validate_pypi_receipt(
            _mapping(state.get("pypi"), "release state PyPI receipt")
        )
        if (
            receipt.get("status") != "complete"
            or receipt.get("version") != version
            or receipt.get("target") != pypi_plan.get("target")
            or receipt.get("repository_url") != pypi_plan.get("index")
            or receipt.get("extras") != planned_extras
            or receipt.get("projects") != expected_pypi_projects(state)
        ):
            raise ValueError("release state has no complete matching PyPI receipt")
        simple_index = str(pypi_plan.get("simple_index", "")).rstrip("/")
        index_url = urlparse(simple_index)
        if index_url.scheme != "https" or index_url.netloc not in {
            "pypi.org",
            "test.pypi.org",
        }:
            raise ValueError("PyPI installation requires a valid simple index")
        pypi = {
            "index_url": simple_index,
            "project_url": (
                f"https://{index_url.netloc}/project/"
                f"{quote(meta_distribution, safe='')}/{quote(version, safe='')}/"
            ),
        }
    elif state.get("pypi") is not None:
        raise ValueError(
            "release state has a PyPI receipt while publication is disabled"
        )

    return (
        {
            "distribution": meta_distribution,
            "version": version,
            "extras": {extra: all_extras[extra] for extra in sorted(all_extras)},
            "pypi": pypi,
        },
        backend_extras,
    )


def build_manifest(
    state: dict[str, Any], release_document: dict[str, Any]
) -> dict[str, Any]:
    """Project one complete Final Release State into exact public schema 9."""
    if state.get("kind") != "ucm-release-state" or state.get("schema_version") != 3:
        raise ValueError(
            "public release manifest requires Final Release State schema 3"
        )
    release = _mapping(state.get("release"), "release state release")
    if release.get("status") != "complete":
        raise ValueError("public release manifest requires complete publication")
    tag = release.get("git_tag")
    if release_document.get("tag_name") != tag:
        raise ValueError("GitHub Release does not match the release state tag")
    release_type = release.get("release_type")
    if release_type not in {"stable", "prerelease", "draft", "nightly"}:
        raise ValueError("release state has an invalid release type")
    actions_run_id = release.get("actions_run_id")
    if (
        not isinstance(actions_run_id, int)
        or isinstance(actions_run_id, bool)
        or actions_run_id < 1
    ):
        raise ValueError("release state has an invalid Actions run ID")
    urls = asset_urls(state, release_document)
    release_url = _release_page_url(release_document)
    python_package, backend_extras = _project_python_package(state)
    images = [
        _mapping(item, "release state Image")
        for item in _list(state.get("images"), "release state Images")
    ]
    wheels: list[dict[str, Any]] = []
    for index, raw_wheel in enumerate(
        _list(state.get("wheels"), "release state Wheels")
    ):
        wheel = _mapping(raw_wheel, f"release state Wheels[{index}]")
        filename = wheel.get("filename")
        extra = wheel.get("runtime_variant")
        distribution = wheel.get("distribution")
        dependencies = wheel.get("dependencies")
        platform_tags = wheel.get("platform_tags")
        if (
            not isinstance(filename, str)
            or filename not in urls
            or not isinstance(extra, str)
            or backend_extras.get(extra) != distribution
            or not isinstance(dependencies, list)
            or any(not isinstance(value, str) or not value for value in dependencies)
            or not isinstance(platform_tags, list)
            or not platform_tags
            or any(not isinstance(value, str) or not value for value in platform_tags)
            or sorted(set(platform_tags)) != platform_tags
        ):
            raise ValueError("release state Wheel cannot be projected into schema 9")
        wheel_id = wheel.get("id")
        if not isinstance(wheel_id, str) or not wheel_id:
            raise ValueError("release state Wheel has no ID")
        architecture = wheel.get("cpu_arch")
        python_abi = wheel.get("python_abi")
        if (
            not isinstance(architecture, str)
            or not architecture
            or not isinstance(python_abi, str)
            or not python_abi
        ):
            raise ValueError("release state Wheel platform differs from its identity")
        product, accelerator = _wheel_capability(wheel_id, images)
        wheels.append(
            {
                "id": wheel_id,
                "product": product,
                "extra": extra,
                "accelerator": accelerator,
                "distribution": distribution,
                "version": python_package["version"],
                "python_abi": python_abi,
                "architecture": architecture,
                "platform_tags": copy.deepcopy(platform_tags),
                "filename": filename,
                "url": urls[filename],
                "sha256": _public_sha256(wheel.get("sha256"), f"Wheel {wheel_id}"),
                "dependencies": copy.deepcopy(dependencies),
            }
        )
    wheels.sort(
        key=lambda item: (
            str(item["extra"]),
            str(item["python_abi"]),
            str(item["architecture"]),
            str(item["filename"]),
        )
    )

    images_by_family: dict[str, list[dict[str, Any]]] = {}
    for image in images:
        family_id = image.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("release state Image has no family ID")
        images_by_family.setdefault(family_id, []).append(image)

    projected_images: list[dict[str, Any]] = []
    runtime_fields = (
        "product_id",
        "version",
        "channel",
        "accelerator_runtime",
        "variant",
        "soc_version",
        "os_id",
        "os_version",
    )
    for index, raw_family in enumerate(
        _list(state.get("families"), "release state families")
    ):
        family = _mapping(raw_family, f"release state families[{index}]")
        if family.get("status") != "published":
            continue
        family_id = family.get("id")
        members = images_by_family.get(str(family_id), [])
        if not isinstance(family_id, str) or not family_id or not members:
            raise ValueError("published release family has no Images")
        if any(member.get("status") != "published" for member in members):
            raise ValueError(f"published release family {family_id} has failed members")
        runtimes = [
            _mapping(member.get("runtime"), f"release family {family_id} Runtime")
            for member in members
        ]
        runtime = runtimes[0]
        for field in runtime_fields:
            value = runtime.get(field)
            if (
                not isinstance(value, str)
                or not value
                or any(other.get(field) != value for other in runtimes[1:])
            ):
                raise ValueError(
                    f"release family {family_id} has inconsistent Runtime {field}"
                )
        projected_images.append(
            {
                "id": family_id,
                "product": runtime["product_id"],
                "upstream": {
                    "version": runtime["version"],
                    "channel": runtime["channel"],
                },
                "accelerator": {
                    "runtime": runtime["accelerator_runtime"],
                    "variant": runtime["variant"],
                    "soc_version": runtime["soc_version"],
                },
                "os": {"id": runtime["os_id"], "version": runtime["os_version"]},
                "publications": _project_image_publications(family, members, family_id),
            }
        )
    projected_images.sort(key=lambda item: str(item["id"]))

    chart = state.get("chart")
    chart_document = None
    if chart is not None:
        chart = _mapping(chart, "release state Chart")
        chart_filename = chart.get("filename")
        chart_oci = chart.get("oci_reference")
        if (
            not isinstance(chart_filename, str)
            or chart_filename not in urls
            or (
                chart_oci is not None
                and (not isinstance(chart_oci, str) or not chart_oci)
            )
        ):
            raise ValueError("release state Chart cannot be projected into schema 9")
        chart_document = {
            "name": chart["name"],
            "version": chart["version"],
            "filename": chart_filename,
            "url": urls[chart_filename],
            "oci": chart_oci,
        }
    asset_names = {
        str(_mapping(asset, "GitHub Release asset").get("name", ""))
        for asset in _list(release_document.get("assets"), "GitHub Release assets")
    }
    if "" in asset_names:
        raise ValueError("GitHub Release contains an unnamed asset")

    asset_names.add(RELEASE_MANIFEST_FILENAME)
    document = {
        "kind": RELEASE_MANIFEST_KIND,
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "release": {
            "tag": tag,
            "type": release_type,
            "version": python_package["version"],
            "url": release_url,
            "actions_run_id": actions_run_id,
        },
        "python": python_package,
        "wheels": wheels,
        "images": projected_images,
        "chart": chart_document,
        "github_release_assets": sorted(asset_names),
    }

    if state.get("toolkit_package") is not None:
        result = state["toolkit_package"]
        document["toolkit"] = {
            key: result[key] for key in ("distribution", "version", "filename")
        }
        document["toolkit"].update(
            sha256=result["sha256"].removeprefix("sha256:"),
            url=urls[result["filename"]],
        )
    return validate_manifest(document)


def asset_urls(
    manifest: dict[str, Any], release_document: dict[str, Any]
) -> dict[str, str]:
    release = _mapping(manifest.get("release"), "release manifest release")
    if release_document.get("tag_name") != release.get("git_tag"):
        raise ValueError("GitHub Release does not match the release manifest tag")

    urls: dict[str, str] = {}
    for index, raw_asset in enumerate(
        _list(release_document.get("assets"), "GitHub Release assets")
    ):
        asset = _mapping(raw_asset, f"GitHub Release assets[{index}]")
        name = asset.get("name")
        url = asset.get("browser_download_url")
        parsed = urlparse(url) if isinstance(url, str) else None
        if (
            not isinstance(name, str)
            or not name
            or name in urls
            or parsed is None
            or parsed.scheme != "https"
            or parsed.netloc != "github.com"
        ):
            raise ValueError("GitHub Release assets contain an invalid entry")
        urls[name] = url

    required = {
        str(item["filename"])
        for item in _list(manifest.get("wheels"), "release manifest Wheels")
    }
    if manifest.get("chart") is not None:
        chart = _mapping(manifest["chart"], "release manifest Chart")
        required.add(str(chart.get("filename", "")))
    if manifest.get("toolkit_package") is not None:
        required.add(manifest["toolkit_package"]["filename"])
    missing = sorted(required - urls.keys())
    if missing:
        raise ValueError(f"GitHub Release is missing required assets: {missing}")
    return urls


def _target_repository(reference: str, context: str) -> str:
    repository, separator, tag = reference.rpartition(":")
    if not separator or not repository or not tag:
        raise ValueError(f"{context} is not a tagged OCI reference")
    return repository


def validate_pypi_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Validate the receipt envelope before exact state comparison."""
    if (
        set(receipt)
        != {
            "kind",
            "schema_version",
            "status",
            "version",
            "target",
            "repository_url",
            "projects",
            "extras",
        }
        or receipt.get("kind") != "ucm-pypi-receipt"
        or receipt.get("schema_version") != 2
    ):
        raise ValueError("PyPI publication receipt has an invalid contract")
    if receipt.get("status") != "complete":
        raise ValueError("PyPI publication receipt is not complete")
    return copy.deepcopy(receipt)


def expected_pypi_projects(state: dict[str, Any]) -> list[dict[str, Any]]:
    version = state["release"]["version"]
    grouped: dict[str, dict[str, Any]] = {}
    for raw_wheel in _list(state.get("wheels"), "release state Wheels"):
        wheel = _mapping(raw_wheel, "release state Wheel")
        project = wheel.get("distribution")
        filename = wheel.get("filename")
        digest = wheel.get("sha256")
        if (
            not isinstance(project, str)
            or not project
            or not isinstance(filename, str)
            or not filename
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("release state Wheel has invalid PyPI coordinates")
        record = grouped.setdefault(
            project,
            {
                "project": project,
                "version": version,
                "role": "backend",
                "files": [],
            },
        )
        record["files"].append({"filename": filename, "sha256": f"sha256:{digest}"})
    if not grouped:
        raise ValueError("release state has no backend Wheels for PyPI")
    backends = sorted(grouped.values(), key=lambda item: item["project"])
    for backend in backends:
        backend["files"].sort(key=lambda item: item["filename"])

    meta = _mapping(state.get("meta_package"), "release state meta package")
    meta_digest = meta.get("sha256")
    meta_distribution = meta.get("distribution")
    if (
        not isinstance(meta_distribution, str)
        or not meta_distribution
        or meta.get("version") != version
        or not isinstance(meta.get("filename"), str)
        or not isinstance(meta_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", meta_digest) is None
    ):
        raise ValueError("release state meta package has invalid PyPI coordinates")
    return [
        *backends,
        *(
            [toolkit_ops.publication_record(state["toolkit_package"])]
            if state.get("toolkit_package") is not None
            else []
        ),
        {
            "project": meta_distribution,
            "version": version,
            "role": "meta",
            "files": [{"filename": meta["filename"], "sha256": meta_digest}],
        },
    ]
