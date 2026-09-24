"""Project-level builder discovery, synchronization, and release selection."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Iterable

from . import registry, runtime, serialization

_BUILDER_LABEL_PREFIX = "io.ucm.builder."
_SHARED_BUILDER_LEGACY_PRODUCT_IDS = frozenset({"vllm", "vllm-ascend", "sglang"})
_BUILDER_METADATA_FIELDS = (
    "id",
    "product_id",
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
)


def _owner(explicit: str | None) -> str:
    if explicit:
        value = explicit
    else:
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        if "/" not in repository:
            raise ValueError(
                "builder target owner requires GITHUB_REPOSITORY or --owner"
            )
        value = repository.split("/", 1)[0]
    normalized = value.lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,38}", normalized) is None:
        raise ValueError(f"invalid builder target owner {value!r}")
    return normalized


def _expand_owner(value: str, owner: str) -> str:
    return value.replace("{owner}", owner)


def _require_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected a mapping")
    return value


def _require_string(mapping: dict[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value


def _validate_oci_repository(value: str, context: str) -> None:
    if runtime.OCI_REPOSITORY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context}: invalid OCI repository {value!r}")


def _validate_oci_image(value: str, context: str) -> None:
    separator = value.rfind(":")
    if separator <= value.rfind("/"):
        raise ValueError(f"{context}: OCI image must include a tag: {value!r}")
    repository, tag = value[:separator], value[separator + 1 :]
    _validate_oci_repository(repository, context)
    if runtime.OCI_TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"{context}: invalid OCI tag {tag!r}")


def _generated_target_tag(
    build_group: str,
    python_abi: str,
    manylinux: str,
    cpu_arch: str,
    recipe_revision: str,
) -> str:
    parts = [
        build_group,
        python_abi,
        manylinux.replace("manylinux_", "manylinux"),
        cpu_arch,
        f"r{recipe_revision}",
    ]
    tag = "-".join(parts)
    if runtime.OCI_TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"generated Builder tag is invalid: {tag!r}")
    return tag


def catalog_from_builds(
    builds: Sequence[Mapping[str, object]],
    config_path: Path | None = None,
    *,
    owner: str | None = None,
    formal_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project one exact Runtime selection into Builder synchronization tasks."""
    from . import policy

    resolved_owner = _owner(owner)
    if formal_policy is None:
        bundle = policy.load(
            platforms_path=(
                config_path
                if config_path is not None and config_path.name == "platforms.yaml"
                else None
            )
        )
        platform_policy = bundle["platforms"]
    else:
        platform_policy = _require_mapping(formal_policy, "formal platform policy")
        if "platforms" in platform_policy:
            platform_policy = _require_mapping(
                platform_policy["platforms"], "formal platform policy.platforms"
            )
    families = _require_mapping(
        platform_policy.get("builder_families"), "builder_families"
    )
    backend_policies = _require_mapping(
        platform_policy.get("backends"), "platform backends"
    )
    items: list[dict[str, object]] = []
    for raw_group in builds:  # type: ignore[index]
        group = _require_mapping(raw_group, "Wheel build")
        _require_string(group, "product_id", "Wheel build")
        family_id = "cuda" if group.get("accelerator") == "cuda" else "ascend"
        family = _require_mapping(
            families.get(family_id), f"builder family {family_id}"
        )
        target_repository = _expand_owner(
            _require_string(family, "target_repository", f"builder family {family_id}"),
            resolved_owner,
        )
        build_group = _require_string(group, "build_group", "Wheel build")
        python_abi = _require_string(group, "python_abi", "Wheel build")
        manylinux = _require_string(group, "manylinux", "Wheel build")
        cpu_arch = _require_string(group, "cpu_arch", "Wheel build")
        recipe_revision = _require_string(group, "recipe_revision", "Wheel build")
        raw_backend_policy = backend_policies.get(str(group["backend"]))
        backend_policy = (
            _require_mapping(raw_backend_policy, f"platform backend {group['backend']}")
            if raw_backend_policy is not None
            else {"status": "blocked"}
        )
        item = {
            **copy.deepcopy(group),
            "target_repository": target_repository,
            "target_tag": _generated_target_tag(
                build_group, python_abi, manylinux, cpu_arch, recipe_revision
            ),
            "checks": {
                "commands": copy.deepcopy(family["required_commands"]),
                "blocking": backend_policy.get("status") == "supported",
                "python_version": group["python_version"],
                "python_abi": python_abi,
                "accelerator_runtime": group["accelerator_runtime"],
                "soc_version": group["soc_version"],
                "variant": group["variant"],
                "required_files": required_files(  # noqa: SLF001
                    family, str(group["variant"])
                ),
            },
        }
        items.append(item)
    catalog = {
        "kind": "ucm-builder-catalog",
        "schema_version": 3,
        "builders": sorted(items, key=lambda item: str(item["id"])),
    }
    return validate_catalog(catalog)


def validate_catalog(catalog: object) -> dict[str, object]:
    mapping = _require_mapping(catalog, "builder catalog")
    if mapping.get("kind") != "ucm-builder-catalog":
        raise ValueError("builder catalog: kind must be ucm-builder-catalog")
    schema_version = mapping.get("schema_version")
    if schema_version not in {3, 4}:
        raise ValueError("builder catalog: schema_version must be 3 or 4")
    values = mapping.get("builders")
    if not isinstance(values, list):
        raise ValueError("builder catalog: builders must be a list")
    ids: set[str] = set()
    target_coordinates: set[tuple[str, str]] = set()
    required = {
        "id",
        "product_id",
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
        "target_repository",
        "target_tag",
        "build_mode",
        "recipe_revision",
        "sync_mode",
        "checks",
    }
    if schema_version == 4:
        required.add("target_digest")
    for index, raw_item in enumerate(values):
        context = f"builder catalog builders[{index}]"
        item = _require_mapping(raw_item, context)
        if set(item) != required:
            raise ValueError(f"{context}: fields must be exact")
        item_id = _require_string(item, "id", context)
        if item_id in ids:
            raise ValueError(f"duplicate Builder id {item_id!r}")
        ids.add(item_id)
        if item.get("cpu_arch") not in {"amd64", "arm64"}:
            raise ValueError(f"{context}: unsupported cpu_arch")
        if item.get("build_mode") != "mirror":
            raise ValueError(f"{context}: build_mode must be mirror")
        if item.get("sync_mode") != "mirror":
            raise ValueError(f"{context}: unsupported sync_mode")
        if not isinstance(item.get("checks"), dict):
            raise ValueError(f"{context}: checks must be a mapping")
        _validate_oci_image(str(item.get("source_image")), f"{context} source_image")
        if (
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(item.get("source_image_digest")))
            is None
        ):
            raise ValueError(f"{context}: source_image_digest is invalid")
        if re.fullmatch(r"[0-9a-f]{12}", str(item.get("recipe_revision"))) is None:
            raise ValueError(f"{context}: recipe_revision is invalid")
        if (
            schema_version == 4
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(item.get("target_digest")))
            is None
        ):
            raise ValueError(f"{context}: target_digest is invalid")
        _validate_oci_repository(
            str(item.get("target_repository")), f"{context} target_repository"
        )
        if runtime.OCI_TAG_PATTERN.fullmatch(str(item.get("target_tag"))) is None:
            raise ValueError(f"{context}: target_tag is invalid")
        coordinate = (str(item["target_repository"]), str(item["target_tag"]))
        if coordinate in target_coordinates:
            raise ValueError(f"duplicate Builder target {coordinate!r}")
        target_coordinates.add(coordinate)
    return mapping


def finalize_catalog(catalog: object, observations: object) -> dict[str, object]:
    """Bind checked OCI labels and immutable target digests to a desired catalog."""

    desired = validate_catalog(catalog)
    if desired.get("schema_version") != 3:
        raise ValueError("Builder finalization requires desired Catalog schema 3")
    observed = _require_mapping(observations, "Builder observations")
    desired_by_id = {
        str(item["id"]): item for item in desired["builders"]  # type: ignore[index]
    }
    if set(observed) != set(desired_by_id):
        raise ValueError("Builder observations must cover the desired Catalog exactly")

    finalized: list[dict[str, object]] = []
    for builder_id in sorted(desired_by_id):
        expected = desired_by_id[builder_id]
        observation = _require_mapping(
            observed[builder_id], f"Builder observation {builder_id}"
        )
        if set(observation) != {"target_digest", "config"}:
            raise ValueError(f"Builder observation {builder_id} fields must be exact")
        digest = observation["target_digest"]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"Builder observation {builder_id} digest is invalid")
        record = registry_builder_record(
            str(expected["target_repository"]),
            str(expected["target_tag"]),
            observation["config"],
        )
        if record is None:
            raise ValueError(
                f"Builder observation {builder_id} is not checked schema 2"
            )
        for field in _BUILDER_METADATA_FIELDS:
            observed_value = record.get(field)
            expected_value = expected.get(field)
            if (
                field == "product_id"
                and expected_value == "shared"
                and observed_value in _SHARED_BUILDER_LEGACY_PRODUCT_IDS
            ):
                continue
            if observed_value != expected_value:
                raise ValueError(
                    f"Builder observation {builder_id} label {field} differs"
                )
        finalized.append({**copy.deepcopy(expected), "target_digest": digest})
    return validate_catalog(
        {
            "kind": "ucm-builder-catalog",
            "schema_version": 4,
            "builders": finalized,
        }
    )


def bind_source_catalog(catalog: object) -> dict[str, object]:
    """Pin Wheel builds directly to their immutable upstream Builder images."""

    desired = validate_catalog(catalog)
    if desired.get("schema_version") != 3:
        raise ValueError("Source Builder binding requires desired Catalog schema 3")

    bound: list[dict[str, object]] = []
    for raw_item in desired["builders"]:  # type: ignore[index]
        item = _require_mapping(raw_item, "Builder source binding")
        source_image = _require_string(item, "source_image", "Builder source binding")
        separator = source_image.rfind(":")
        if separator <= source_image.rfind("/"):
            raise ValueError("Builder source image must include a tag")
        bound.append(
            {
                **copy.deepcopy(item),
                "target_repository": source_image[:separator],
                "target_digest": item["source_image_digest"],
            }
        )

    return validate_catalog(
        {
            "kind": "ucm-builder-catalog",
            "schema_version": 4,
            "builders": bound,
        }
    )


def compute_sync_plan(catalog: object, existing_tags: object) -> dict[str, object]:
    """Return only catalog entries whose exact target tag is absent."""
    validated = validate_catalog(catalog)
    if validated.get("schema_version") != 3:
        raise ValueError("Builder sync planning requires an unfinalized Catalog")
    existing = _require_mapping(existing_tags, "existing builder tags")
    normalized: dict[str, set[str]] = {}
    for repository, raw_tags in existing.items():
        if not isinstance(repository, str) or not repository:
            raise ValueError(
                "existing builder tags: repository names must be non-empty strings"
            )
        if not isinstance(raw_tags, list) or not all(
            isinstance(tag, str) for tag in raw_tags
        ):
            raise ValueError(
                f"existing builder tags {repository}: tags must be a string list"
            )
        normalized[repository] = set(raw_tags)
    sort_fields = ("id",)
    missing = sorted(
        (
            item
            for item in validated["builders"]  # type: ignore[union-attr]
            if item["target_tag"]
            not in normalized.get(item["target_repository"], set())
        ),
        key=lambda item: tuple(item[field] for field in sort_fields),
    )
    matrix = [
        {
            **item,
            "label": f"{item['build_group']} · {item['python_abi']} · {item['cpu_arch']}",
        }
        for item in validated["builders"]
    ]
    return {
        "kind": "ucm-builder-sync-plan",
        "schema_version": 1,
        "builders": missing,
        "matrix": {"include": matrix},
    }


def builder_labels(builder: Mapping[str, object]) -> dict[str, str]:
    """Project one catalog entry into OCI labels used by PR inventory scans."""
    item = _require_mapping(dict(builder), "Builder label source")
    missing = [field for field in _BUILDER_METADATA_FIELDS if field not in item]
    if missing:
        raise ValueError(f"Builder label source is missing {missing}")
    labels = {
        f"{_BUILDER_LABEL_PREFIX}schema": "2",
        f"{_BUILDER_LABEL_PREFIX}checked": "true",
        f"{_BUILDER_LABEL_PREFIX}target_tag": _require_string(
            item, "target_tag", "Builder label source"
        ),
    }
    for field in _BUILDER_METADATA_FIELDS:
        value = item[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Builder label field {field} must be a non-empty string")
        labels[f"{_BUILDER_LABEL_PREFIX}{field}"] = value
    return labels


def registry_builder_record(
    repository: str, tag: str, config: object
) -> dict[str, object] | None:
    """Read one checked final Builder record from its OCI config labels."""
    if runtime.OCI_REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise ValueError(f"invalid Builder repository {repository!r}")
    if runtime.OCI_TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"invalid Builder tag {tag!r}")
    document = _require_mapping(config, f"Builder config {repository}:{tag}")
    nested = document.get("config", {})
    nested = _require_mapping(nested, f"Builder config {repository}:{tag}.config")
    raw_labels = nested.get("Labels") or {}
    labels = _require_mapping(
        raw_labels, f"Builder config {repository}:{tag}.config.Labels"
    )
    if labels.get(f"{_BUILDER_LABEL_PREFIX}schema") != "2":
        return None
    if labels.get(f"{_BUILDER_LABEL_PREFIX}checked") != "true":
        return None
    if labels.get(f"{_BUILDER_LABEL_PREFIX}target_tag") != tag:
        return None
    metadata: dict[str, object] = {}
    for field in _BUILDER_METADATA_FIELDS:
        value = labels.get(f"{_BUILDER_LABEL_PREFIX}{field}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Builder {repository}:{tag} label {field} is missing")
        metadata[field] = value
    created = labels.get(f"{_BUILDER_LABEL_PREFIX}created", document.get("created"))
    if not isinstance(created, str) or not created:
        raise ValueError(f"Builder {repository}:{tag} created timestamp is missing")
    return {
        **metadata,
        "target_repository": repository,
        "target_tag": tag,
        "created": created,
        "checked": True,
        "checks": {},
    }


def _crane_output(operation: str, reference: str) -> str:
    completed = None
    last_error = ""
    for attempt in range(1, 4):
        try:
            completed = subprocess.run(
                ["crane", operation, reference],
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            completed = None
            last_error = "timed out after 60 seconds"
        if completed is None:
            if attempt < 3:
                time.sleep(2**attempt)
            continue
        if completed.returncode == 0:
            return completed.stdout
        last_error = completed.stderr.strip() or str(completed.returncode)
        if attempt < 3:
            time.sleep(2**attempt)
    raise ValueError(
        f"crane {operation} failed for {reference}: {last_error or 'unknown error'}"
    )


def _live_builder_config(reference: str) -> object:
    try:
        manifest = json.loads(_crane_output("manifest", reference))
    except json.JSONDecodeError as error:
        raise ValueError(f"Builder manifest is malformed for {reference}") from error
    descriptors = manifest.get("manifests") if isinstance(manifest, dict) else None
    if not isinstance(descriptors, list):
        return json.loads(_crane_output("config", reference))
    repository = reference.rpartition(":")[0]
    configs = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        platform = descriptor.get("platform")
        digest = descriptor.get("digest")
        if (
            isinstance(platform, dict)
            and platform.get("os") == "linux"
            and platform.get("architecture") in {"amd64", "arm64"}
            and isinstance(digest, str)
        ):
            configs.append(
                json.loads(_crane_output("config", f"{repository}@{digest}"))
            )
    if len(configs) != 1:
        raise ValueError(f"Builder {reference} must contain exactly one Linux member")
    return configs[0]


def scan_registry_builders(
    formal_policy: Mapping[str, object],
    *,
    tag_loader=None,
    config_loader=None,
) -> dict[str, object]:
    """Scan only labelled, functionally promoted final Builder tags."""
    policy_mapping = _require_mapping(dict(formal_policy), "formal policy")
    if "platforms" in policy_mapping:
        policy_mapping = _require_mapping(
            policy_mapping["platforms"], "formal policy.platforms"
        )
    families = _require_mapping(
        policy_mapping.get("builder_families"), "builder_families"
    )
    repositories = sorted(
        {
            _require_string(
                _require_mapping(item, "builder family"),
                "target_repository",
                "builder family",
            )
            for item in families.values()
        }
    )
    load_tags = tag_loader or (
        lambda repository: _crane_output("ls", repository).splitlines()
    )
    load_config = config_loader or _live_builder_config
    records: list[dict[str, object]] = []
    for repository in repositories:
        for tag in sorted(set(str(item) for item in load_tags(repository))):
            if re.search(r"-r[0-9a-f]{12}$", tag) is None:
                continue
            record = registry_builder_record(
                repository, tag, load_config(f"{repository}:{tag}")
            )
            if record is not None:
                records.append(record)
    return {
        "kind": "ucm-builder-registry",
        "schema_version": 1,
        "builders": sorted(
            records,
            key=lambda item: (
                str(item["id"]),
                str(item["created"]),
                str(item["target_tag"]),
            ),
        ),
    }


def catalog_from_registry_records(
    records: Iterable[Mapping[str, object]]
) -> dict[str, object]:
    """Reopen selected Registry records as the compact Builder Catalog."""
    items: list[dict[str, object]] = []
    for raw in records:
        record = dict(raw)
        items.append(
            {
                key: copy.deepcopy(value)
                for key, value in record.items()
                if key not in {"created", "checked"}
            }
        )
    return validate_catalog(
        {
            "kind": "ucm-builder-catalog",
            "schema_version": 3,
            "builders": sorted(items, key=lambda item: str(item["id"])),
        }
    )


def required_files(family: Mapping[str, object], variant: str) -> list[str]:
    common = family.get("required_files", [])
    variants = family.get("variant_required_files", {})
    if not isinstance(common, list) or not all(
        isinstance(item, str) and item for item in common
    ):
        raise ValueError("Builder family required_files is invalid")
    if not isinstance(variants, Mapping):
        raise ValueError("Builder family variant_required_files is invalid")
    specific = variants.get(variant, [])
    if not isinstance(specific, list) or not all(
        isinstance(item, str) and item for item in specific
    ):
        raise ValueError(f"Builder variant {variant} required_files is invalid")
    return sorted(set(common + specific))


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ARCHITECTURES = ("amd64", "arm64")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MANYLINUX = re.compile(r"manylinux_(\d+)_(\d+)")
_CUDA_BUILDER_TAG = re.compile(r"^cuda(?P<runtime>[0-9]+\.[0-9]+)$")
_ASCEND_BUILDER_TAG = re.compile(
    r"^(?P<runtime>[0-9]+\.[0-9]+\.[0-9]+)-"
    r"(?P<variant>310p|910b|a3|950)-"
    r"(?P<manylinux>manylinux_[0-9]+_[0-9]+)-"
    r"py(?P<python>[0-9]+\.[0-9]+)$"
)


def _manylinux_floor(value: str) -> tuple[int, int]:
    match = _MANYLINUX.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid manylinux policy {value!r}")
    return int(match.group(1)), int(match.group(2))


def _source_repositories(family: Mapping[str, object], context: str) -> dict[str, str]:
    values = _require_mapping(
        family.get("source_repositories"), f"{context}.source_repositories"
    )
    if set(values) != set(SUPPORTED_ARCHITECTURES):
        raise ValueError(f"{context}: source_repositories must define amd64 and arm64")
    return {
        architecture: _require_string(values, architecture, context)
        for architecture in SUPPORTED_ARCHITECTURES
    }


def _manifest_member_digest(
    reference: str,
    architecture: str,
    *,
    tag_fixture: Mapping[str, object] | None,
    manifest_loader: Callable[[str], object] | None,
    config_loader: Callable[[str], object] | None = None,
    digest_loader: Callable[[str], str] | None = None,
) -> str:
    if tag_fixture is not None:
        values = tag_fixture.get("source_image_members")
        if isinstance(values, Mapping) and reference in values:
            members = _require_mapping(
                values[reference], f"source image fixture {reference}"
            )
            digest = members.get(architecture)
            if isinstance(digest, str) and _DIGEST.fullmatch(digest):
                return digest
            raise ValueError(
                f"source image fixture {reference} has no valid "
                f"{architecture} digest"
            )
    separator = reference.rfind(":")
    if separator <= reference.rfind("/"):
        raise ValueError(f"raw Builder reference must include a tag: {reference}")
    repository = reference[:separator]
    root_digest = (
        digest_loader(reference)
        if digest_loader is not None
        else registry.read("digest", reference).strip()
    )
    if not isinstance(root_digest, str) or _DIGEST.fullmatch(root_digest) is None:
        raise ValueError(f"raw Builder {reference} has invalid digest")
    pinned_reference = f"{repository}@{root_digest}"
    raw_manifest = (
        manifest_loader(pinned_reference)
        if manifest_loader is not None
        else registry.read("manifest", pinned_reference)
    )
    if isinstance(raw_manifest, str):
        try:
            raw_manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"raw Builder manifest {reference} is malformed"
            ) from error
    manifest = _require_mapping(raw_manifest, f"raw Builder manifest {reference}")
    descriptors = manifest.get("manifests")
    if not isinstance(descriptors, list):
        media_type = manifest.get("mediaType")
        if media_type not in {
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        }:
            raise ValueError(
                f"raw Builder {reference} has unsupported manifest {media_type!r}"
            )
        raw_config = (
            config_loader(pinned_reference)
            if config_loader is not None
            else registry.read("config", pinned_reference)
        )
        if isinstance(raw_config, str):
            try:
                raw_config = json.loads(raw_config)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"raw Builder config {reference} is malformed"
                ) from error
        config = _require_mapping(raw_config, f"raw Builder config {reference}")
        if (
            config.get("os", "linux") != "linux"
            or config.get("architecture") != architecture
        ):
            raise ValueError(f"raw Builder {reference} is not linux/{architecture}")
        return root_digest
    matches: list[str] = []
    for raw_descriptor in descriptors:
        descriptor = _require_mapping(
            raw_descriptor, f"raw Builder manifest {reference} member"
        )
        platform = descriptor.get("platform")
        if not isinstance(platform, Mapping):
            continue
        if (
            platform.get("os") == "linux"
            and platform.get("architecture") == architecture
        ):
            digest = descriptor.get("digest")
            if isinstance(digest, str) and _DIGEST.fullmatch(digest):
                matches.append(digest)
    if len(matches) != 1:
        raise ValueError(
            f"raw Builder {reference} must have exactly one "
            f"linux/{architecture} member"
        )
    return matches[0]


def _raw_builder_candidates(
    release: Mapping[str, object],
    required: set[tuple[str, str, str, str, str]],
    *,
    tag_fixture: Mapping[str, object] | None,
    tag_loader: Callable[[str], Sequence[str]] | None,
    manifest_loader: Callable[[str], object] | None,
    config_loader: Callable[[str], object] | None = None,
    digest_loader: Callable[[str], str] | None = None,
) -> list[dict[str, str]]:
    families = _require_mapping(release.get("builder_families"), "Builder families")
    values: list[dict[str, str]] = []
    tag_cache: dict[str, list[str]] = {}
    digest_cache: dict[tuple[str, str], str] = {}

    def tags(repository: str) -> list[str]:
        if repository not in tag_cache:
            tag_cache[repository] = registry.repository_tags(
                repository, tag_fixture=tag_fixture, tag_loader=tag_loader
            )
        return tag_cache[repository]

    def member(reference: str, architecture: str) -> str:
        key = (reference, architecture)
        if key not in digest_cache:
            digest_cache[key] = _manifest_member_digest(
                reference,
                architecture,
                tag_fixture=tag_fixture,
                manifest_loader=manifest_loader,
                config_loader=config_loader,
                digest_loader=digest_loader,
            )
        return digest_cache[key]

    cuda = _require_mapping(families.get("cuda"), "CUDA Builder family")
    cuda_repositories = _source_repositories(cuda, "CUDA Builder family")
    cuda_manylinux = _require_string(cuda, "manylinux", "CUDA Builder family")
    _manylinux_floor(cuda_manylinux)
    for architecture, repository in cuda_repositories.items():
        for tag in tags(repository):
            match = _CUDA_BUILDER_TAG.fullmatch(tag)
            if match is None:
                continue
            runtime = f"cuda-{match.group('runtime')}"
            if not any(
                accelerator == "cuda"
                and accelerator_runtime == runtime
                and required_architecture == architecture
                for (
                    accelerator,
                    accelerator_runtime,
                    _variant_name,
                    _python_abi,
                    required_architecture,
                ) in required
            ):
                continue
            reference = f"{repository}:{tag}"
            values.append(
                {
                    "accelerator": "cuda",
                    "accelerator_runtime": runtime,
                    "variant": "default",
                    "python_abi": "*",
                    "manylinux": cuda_manylinux,
                    "cpu_arch": architecture,
                    "source_image": reference,
                    "source_image_digest": member(reference, architecture),
                }
            )

    ascend = _require_mapping(families.get("ascend"), "Ascend Builder family")
    ascend_repositories = _source_repositories(ascend, "Ascend Builder family")
    ascend_manylinux = _require_string(ascend, "manylinux", "Ascend Builder family")
    _manylinux_floor(ascend_manylinux)
    raw_manylinux_by_runtime = ascend.get("manylinux_by_runtime", {})
    manylinux_by_runtime = _require_mapping(
        raw_manylinux_by_runtime, "Ascend Builder family manylinux_by_runtime"
    )
    for configured_runtime, configured_manylinux in manylinux_by_runtime.items():
        if (
            not isinstance(configured_runtime, str)
            or re.fullmatch(r"cann-[0-9]+\.[0-9]+\.[0-9]+", configured_runtime) is None
        ):
            raise ValueError(
                "Ascend Builder family manylinux_by_runtime has invalid runtime"
            )
        if not isinstance(configured_manylinux, str):
            raise ValueError(
                f"Ascend Builder family manylinux_by_runtime[{configured_runtime!r}] "
                "must be a string"
            )
        _manylinux_floor(configured_manylinux)
    seen_ascend: set[tuple[str, str]] = set()
    variant_by_token = {"910b": "a2", "a3": "a3", "950": "a5"}
    for architecture, repository in ascend_repositories.items():
        for tag in tags(repository):
            match = _ASCEND_BUILDER_TAG.fullmatch(tag)
            if match is None or match.group("variant") == "310p":
                continue
            runtime = f"cann-{match.group('runtime')}"
            selected_manylinux = str(
                manylinux_by_runtime.get(runtime, ascend_manylinux)
            )
            if match.group("manylinux") != selected_manylinux:
                continue
            variant = variant_by_token[match.group("variant")]
            python_abi = "cp" + match.group("python").replace(".", "")
            if (
                "ascend",
                runtime,
                variant,
                python_abi,
                architecture,
            ) not in required:
                continue
            reference = f"{repository}:{tag}"
            key = (reference, architecture)
            if key in seen_ascend:
                continue
            seen_ascend.add(key)
            values.append(
                {
                    "accelerator": "ascend",
                    "accelerator_runtime": runtime,
                    "variant": variant,
                    "python_abi": python_abi,
                    "manylinux": match.group("manylinux"),
                    "cpu_arch": architecture,
                    "source_image": reference,
                    "source_image_digest": member(reference, architecture),
                }
            )
    return values


def _mirror_revision(
    build: Mapping[str, object], commands: Sequence[str], required_files: Sequence[str]
) -> str:
    dockerfile = RELEASE_ROOT / "docker" / "Dockerfile.builder-mirror"
    payload = {
        "source_image_digest": build["source_image_digest"],
        "backend": build["backend"],
        "accelerator_runtime": build["accelerator_runtime"],
        "variant": build["variant"],
        "python_abi": build["python_abi"],
        "manylinux": build["manylinux"],
        "cpu_arch": build["cpu_arch"],
        "required_commands": sorted(set(commands)),
        "required_files": sorted(set(required_files)),
        "mirror_dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(serialization.canonical_bytes(payload)).hexdigest()[:12]


def _build_for_probe(
    probe: Mapping[str, object],
    candidates: Sequence[Mapping[str, str]],
    release: Mapping[str, object],
) -> dict[str, object]:
    accelerator, variant, build_group = runtime.wheel_variant(probe)
    architecture = _require_string(probe, "cpu_arch", "runtime probe")
    runtime_value = _require_string(probe, "accelerator_runtime", "runtime probe")
    python_abi = _require_string(probe, "python_abi", "runtime probe")
    matches = [
        dict(candidate)
        for candidate in candidates
        if candidate["accelerator"] == accelerator
        and candidate["accelerator_runtime"] == runtime_value
        and candidate["variant"] == variant
        and candidate["cpu_arch"] == architecture
        and candidate["python_abi"] in {"*", python_abi}
    ]
    if matches:
        lowest_floor = min(_manylinux_floor(item["manylinux"]) for item in matches)
        matches = [
            item
            for item in matches
            if _manylinux_floor(item["manylinux"]) == lowest_floor
        ]
    if len(matches) != 1:
        detail = [
            f"{item['source_image']}@{item['source_image_digest']} "
            f"({item['manylinux']})"
            for item in matches
        ]
        raise ValueError(
            f"{probe.get('runtime_ref')} linux/{architecture}: expected one "
            f"compatible raw Builder for {runtime_value}/{variant}/{python_abi}, "
            f"found {len(matches)}: {detail}"
        )
    raw = matches[0]
    families = _require_mapping(release.get("builder_families"), "Builder families")
    family = _require_mapping(
        families.get(accelerator), f"Builder family {accelerator}"
    )
    commands = family.get("required_commands")
    if not isinstance(commands, list) or not all(
        isinstance(item, str) and item for item in commands
    ):
        raise ValueError(f"Builder family {accelerator}: required_commands is invalid")
    required = required_files(family, variant)
    build: dict[str, object] = {
        "id": f"{build_group}-{python_abi}-{architecture}",
        # A UCM Wheel is determined by its native capability, not the Runtime
        # product consuming it.  Keeping this value product-neutral lets vLLM
        # and SGLang share an identical CUDA/CANN build task.
        "product_id": "shared",
        "build_group": build_group,
        "backend": _require_string(probe, "backend", "runtime probe"),
        "accelerator": accelerator,
        "accelerator_runtime": runtime_value,
        "variant": variant,
        "soc_version": _require_string(probe, "soc_version", "runtime probe"),
        "runtime_variant": build_group,
        "python_version": _require_string(probe, "python_version", "runtime probe"),
        "python_abi": python_abi,
        "manylinux": raw["manylinux"],
        "cpu_arch": architecture,
        "source_image": raw["source_image"],
        "source_image_digest": raw["source_image_digest"],
        "build_mode": "mirror",
        "recipe_revision": "",
        "sync_mode": "mirror",
    }
    build["recipe_revision"] = _mirror_revision(build, commands, required)
    return build


def resolve_probe_builds(
    release: Mapping[str, object],
    probes: Sequence[Mapping[str, object]],
    *,
    tag_fixture: Mapping[str, object] | None = None,
    tag_loader: Callable[[str], Sequence[str]] | None = None,
    manifest_loader: Callable[[str], object] | None = None,
    config_loader: Callable[[str], object] | None = None,
    digest_loader: Callable[[str], str] | None = None,
) -> list[dict[str, object]]:
    """Resolve raw mirror Builders for already-probed Runtime members."""

    required = {
        (
            runtime.wheel_variant(probe)[0],
            _require_string(probe, "accelerator_runtime", "runtime probe"),
            runtime.wheel_variant(probe)[1],
            _require_string(probe, "python_abi", "runtime probe"),
            _require_string(probe, "cpu_arch", "runtime probe"),
        )
        for probe in probes
    }
    raw_candidates = _raw_builder_candidates(
        release,
        required,
        tag_fixture=tag_fixture,
        tag_loader=tag_loader,
        manifest_loader=manifest_loader,
        config_loader=config_loader,
        digest_loader=digest_loader,
    )
    builds: dict[str, dict[str, object]] = {}
    backends = _require_mapping(release.get("backends"), "platform backends")
    for raw_probe in probes:
        probe = _require_mapping(raw_probe, "runtime probe")
        backend = _require_string(probe, "backend", "runtime probe")
        backend_policy = _require_mapping(backends.get(backend), f"backend {backend}")
        if backend_policy.get("status") == "blocked":
            continue
        build = _build_for_probe(probe, raw_candidates, release)
        existing = builds.get(str(build["id"]))
        if existing is not None and existing != build:
            raise ValueError(f"{build['id']}: raw Builder capability identity drift")
        builds[str(build["id"])] = build
    return sorted(builds.values(), key=lambda item: str(item["id"]))
