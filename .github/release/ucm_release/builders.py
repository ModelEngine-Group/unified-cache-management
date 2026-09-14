"""Project-level builder discovery, synchronization, and release selection."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

from . import core

CATALOG_FIELDS = (
    "project",
    "accelerator",
    "accelerator_runtime",
    "variant",
    "python_abi",
    "manylinux",
    "cpu_arch",
    "source_image",
    "target_repository",
    "target_tag",
    "build_mode",
)
CAPABILITY_FIELDS = (
    "accelerator_runtime",
    "variant",
    "python_abi",
    "manylinux",
    "cpu_arch",
)

_BUILDER_LABEL_PREFIX = "io.ucm.builder."
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
    if core.OCI_REPOSITORY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context}: invalid OCI repository {value!r}")


def _validate_oci_image(value: str, context: str) -> None:
    separator = value.rfind(":")
    if separator <= value.rfind("/"):
        raise ValueError(f"{context}: OCI image must include a tag: {value!r}")
    repository, tag = value[:separator], value[separator + 1 :]
    _validate_oci_repository(repository, context)
    if core.OCI_TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"{context}: invalid OCI tag {tag!r}")


def _validate_catalog_item(item: object, context: str) -> dict[str, str]:
    mapping = _require_mapping(item, context)
    for field in CATALOG_FIELDS:
        _require_string(mapping, field, context)
    unknown = set(mapping) - set(CATALOG_FIELDS)
    if unknown:
        raise ValueError(f"{context}: unsupported fields {sorted(unknown)}")
    if mapping["cpu_arch"] not in {"amd64", "arm64"}:
        raise ValueError(f"{context}: unsupported cpu_arch {mapping['cpu_arch']!r}")
    if mapping["accelerator"] not in {"cuda", "ascend"}:
        raise ValueError(
            f"{context}: unsupported accelerator {mapping['accelerator']!r}"
        )
    runtime_pattern = (
        r"cuda-\d+\.\d+" if mapping["accelerator"] == "cuda" else r"cann-\d+\.\d+\.\d+"
    )
    if re.fullmatch(runtime_pattern, mapping["accelerator_runtime"]) is None:
        raise ValueError(
            f"{context}: malformed accelerator_runtime {mapping['accelerator_runtime']!r}"
        )
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", mapping["variant"]) is None:
        raise ValueError(f"{context}: malformed variant {mapping['variant']!r}")
    if re.fullmatch(r"cp\d+", mapping["python_abi"]) is None:
        raise ValueError(f"{context}: malformed python_abi {mapping['python_abi']!r}")
    if re.fullmatch(r"manylinux_\d+_\d+", mapping["manylinux"]) is None:
        raise ValueError(f"{context}: malformed manylinux {mapping['manylinux']!r}")
    if mapping["build_mode"] not in {"mirror", "extend", "copy"}:
        raise ValueError(f"{context}: unsupported build_mode {mapping['build_mode']!r}")
    _validate_oci_image(mapping["source_image"], f"{context} source_image")
    _validate_oci_repository(
        mapping["target_repository"], f"{context} target_repository"
    )
    return mapping  # type: ignore[return-value]


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
    if core.OCI_TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"generated Builder tag is invalid: {tag!r}")
    return tag


def catalog_from_selection(
    selection: object,
    config_path: Path | None = None,
    *,
    owner: str | None = None,
    formal_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project one exact Runtime selection into Builder synchronization tasks."""
    from . import policy, upstream

    resolved = upstream.validate_selection(selection)
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
    for raw_group in resolved["wheel_builds"]:  # type: ignore[index]
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
                "required_files": upstream._required_files(  # noqa: SLF001
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
    if schema_version not in {1, 3, 4}:
        raise ValueError("builder catalog: schema_version must be 1, 3, or 4")
    values = mapping.get("builders")
    if not isinstance(values, list):
        raise ValueError("builder catalog: builders must be a list")
    if schema_version == 1:
        for index, item in enumerate(values):
            _validate_catalog_item(item, f"builder catalog builders[{index}]")
        return mapping
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
        if item.get("sync_mode") not in {"mirror", "registry-only"}:
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
        if core.OCI_TAG_PATTERN.fullmatch(str(item.get("target_tag"))) is None:
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
            if record.get(field) != expected.get(field):
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
    if validated.get("schema_version") not in {1, 3}:
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
    sort_fields = ("id",) if validated["schema_version"] == 3 else CATALOG_FIELDS
    missing = sorted(
        (
            item
            for item in validated["builders"]  # type: ignore[union-attr]
            if item["target_tag"]
            not in normalized.get(item["target_repository"], set())
        ),
        key=lambda item: tuple(item[field] for field in sort_fields),
    )
    registry_only_missing = [
        item for item in missing if item.get("sync_mode") == "registry-only"
    ]
    if registry_only_missing:
        coordinates = sorted(
            f"{item['target_repository']}:{item['target_tag']}"
            for item in registry_only_missing
        )
        raise ValueError(
            "Registry-only Builders disappeared after capability matching: "
            f"{coordinates}"
        )
    matrix = []
    matrix_source = (
        validated["builders"] if validated["schema_version"] == 3 else missing
    )
    for item in matrix_source:  # type: ignore[union-attr]
        if validated["schema_version"] == 3:
            matrix.append(
                {
                    **item,
                    "label": (
                        f"{item['build_group']} · {item['python_abi']} · "
                        f"{item['cpu_arch']}"
                    ),
                }
            )
            continue
        runtime_name, _, runtime_version = item["accelerator_runtime"].partition("-")
        runtime_label = f"{runtime_name.upper()} {runtime_version}"
        variant = "" if item["variant"] == "default" else f" {item['variant'].upper()}"
        matrix.append(
            {
                **item,
                "id": item["target_tag"],
                "label": f"{runtime_label}{variant} · {item['cpu_arch']}",
            }
        )
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
    if core.OCI_REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise ValueError(f"invalid Builder repository {repository!r}")
    if core.OCI_TAG_PATTERN.fullmatch(tag) is None:
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


def _capability_text(capability: dict[str, str]) -> str:
    return ", ".join(f"{field}={capability[field]}" for field in CAPABILITY_FIELDS)


def _nearest_candidates(
    available: list[dict[str, str]], capability: dict[str, str], accelerator: str
) -> list[str]:
    candidates = [item for item in available if item["accelerator"] == accelerator]
    if not candidates:
        candidates = available
    ranked = sorted(
        candidates,
        key=lambda item: (
            sum(item[field] != capability[field] for field in CAPABILITY_FIELDS),
            item["target_tag"],
        ),
    )
    return [
        f"{item['target_repository']}:{item['target_tag']} "
        f"({_capability_text(item)})"
        for item in ranked[:5]
    ]


def select_builders(catalog: object, release: object) -> dict[str, object]:
    """Select one builder for every release wheel profile and architecture."""
    validated = validate_catalog(catalog)
    available = list(validated["builders"])  # type: ignore[arg-type]
    release_config = _require_mapping(release, "release config")
    profiles = release_config.get("wheel_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("release config: wheel_profiles must be a non-empty list")
    selected: list[dict[str, str]] = []
    profile_ids: set[str] = set()
    for index, raw_profile in enumerate(profiles):
        context = f"release config wheel_profiles[{index}]"
        profile = _require_mapping(raw_profile, context)
        profile_id = _require_string(profile, "id", context)
        if profile_id in profile_ids:
            raise ValueError(f"duplicate release profile id: {profile_id}")
        profile_ids.add(profile_id)
        accelerator = _require_string(profile, "accelerator", context)
        runtime = _require_string(profile, "accelerator_runtime", context)
        python_abi = _require_string(profile, "python_abi", context)
        builder_manylinux = _require_string(profile, "builder_manylinux", context)
        if re.fullmatch(r"manylinux_\d+_\d+", builder_manylinux) is None:
            raise ValueError(
                f"{context}: malformed builder_manylinux {builder_manylinux!r}"
            )
        arches = profile.get("cpu_arch")
        if (
            not isinstance(arches, list)
            or not arches
            or not all(arch in {"amd64", "arm64"} for arch in arches)
        ):
            raise ValueError(
                f"{context}: cpu_arch must be a non-empty amd64/arm64 list"
            )
        if len(set(arches)) != len(arches):
            raise ValueError(f"{context}: cpu_arch contains duplicates")
        if accelerator == "cuda":
            variant = "default"
        elif accelerator == "ascend":
            npu_arch = profile.get("npu_arch")
            if (
                not isinstance(npu_arch, list)
                or len(npu_arch) != 1
                or not isinstance(npu_arch[0], str)
            ):
                raise ValueError(f"{context}: Ascend selection requires one npu_arch")
            variant = npu_arch[0]
        else:
            raise ValueError(f"{context}: unsupported accelerator {accelerator!r}")
        for cpu_arch in arches:
            partial = {
                "accelerator_runtime": runtime,
                "variant": variant,
                "python_abi": python_abi,
                "cpu_arch": str(cpu_arch),
            }
            capability = {
                **partial,
                "manylinux": builder_manylinux,
            }
            matches = [
                item
                for item in available
                if item["accelerator"] == accelerator
                and all(item[field] == capability[field] for field in CAPABILITY_FIELDS)
            ]
            if len(matches) != 1:
                nearest = _nearest_candidates(available, capability, accelerator)
                reason = "missing" if not matches else f"multiple ({len(matches)})"
                raise ValueError(
                    f"release profile {profile_id}: {reason} builder for requested capability "
                    f"{_capability_text(capability)}; nearest candidates: {nearest}"
                )
            selected.append({"profile_id": profile_id, **matches[0]})
    return {
        "kind": "ucm-builder-selection",
        "schema_version": 1,
        "builders": selected,
        "matrix": {"include": selected},
    }


def bind_selection(catalog: object, selection: object) -> dict[str, object]:
    """Bind one selected project builder to every release profile architecture."""
    release = copy.deepcopy(_require_mapping(catalog, "release catalog"))
    profiles = release.get("wheel_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("release catalog: wheel_profiles must be a non-empty list")

    selected = _require_mapping(selection, "builder selection")
    if set(selected) != {"kind", "schema_version", "builders", "matrix"}:
        raise ValueError("builder selection: fields must be exact")
    if selected.get("kind") != "ucm-builder-selection":
        raise ValueError("builder selection: kind must be ucm-builder-selection")
    if selected.get("schema_version") != 1:
        raise ValueError("builder selection: schema_version must be 1")
    items = selected.get("builders")
    matrix = selected.get("matrix")
    if not isinstance(items, list):
        raise ValueError("builder selection: builders must be a list")
    if (
        not isinstance(matrix, dict)
        or set(matrix) != {"include"}
        or matrix["include"] != items
    ):
        raise ValueError(
            "builder selection: matrix.include must exactly match builders"
        )

    profiles_by_id: dict[str, dict[str, object]] = {}
    expected_coordinates: set[tuple[str, str]] = set()
    for index, raw_profile in enumerate(profiles):
        context = f"release catalog wheel_profiles[{index}]"
        profile = _require_mapping(raw_profile, context)
        profile_id = _require_string(profile, "id", context)
        if profile_id in profiles_by_id:
            raise ValueError(f"duplicate release profile id: {profile_id}")
        requirements = profile.get("builders")
        architectures = profile.get("cpu_arch")
        if not isinstance(requirements, dict) or not isinstance(architectures, list):
            raise ValueError(
                f"release profile {profile_id}: builder requirements are invalid"
            )
        if set(requirements) != set(architectures):
            raise ValueError(
                f"release profile {profile_id}: builder architectures do not match cpu_arch"
            )
        profiles_by_id[profile_id] = profile
        expected_coordinates.update((profile_id, str(arch)) for arch in architectures)

    seen: set[tuple[str, str]] = set()
    for index, raw_item in enumerate(items):
        context = f"builder selection builders[{index}]"
        item = _require_mapping(raw_item, context)
        profile_id = _require_string(item, "profile_id", context)
        profile = profiles_by_id.get(profile_id)
        if profile is None:
            raise ValueError(f"{context}: unknown release profile {profile_id!r}")
        architecture = _require_string(item, "cpu_arch", context)
        requirements = profile["builders"]
        if not isinstance(requirements, dict) or architecture not in requirements:
            raise ValueError(
                f"{context}: undeclared architecture {architecture!r} for release profile {profile_id!r}"
            )
        catalog_item = {
            key: value for key, value in item.items() if key != "profile_id"
        }
        validated_item = _validate_catalog_item(catalog_item, context)
        coordinate = (profile_id, architecture)
        if coordinate in seen:
            raise ValueError(
                f"duplicate builder selection for release profile {profile_id!r} architecture {architecture!r}"
            )
        seen.add(coordinate)
        npu_arch = profile.get("npu_arch")
        expected_variant = (
            "default"
            if profile.get("accelerator") == "cuda"
            else npu_arch[0] if isinstance(npu_arch, list) and npu_arch else None
        )
        expected_capability = {
            "accelerator": profile.get("accelerator"),
            "accelerator_runtime": profile.get("accelerator_runtime"),
            "variant": expected_variant,
            "python_abi": profile.get("python_abi"),
            "manylinux": profile.get("builder_manylinux"),
            "cpu_arch": architecture,
        }
        mismatches = {
            key: (expected, validated_item[key])
            for key, expected in expected_capability.items()
            if validated_item[key] != expected
        }
        if mismatches:
            raise ValueError(
                f"{context}: selected capability does not match release profile {profile_id!r}: {mismatches}"
            )
        requirement = requirements[architecture]
        if not isinstance(requirement, dict):
            raise ValueError(
                f"release profile {profile_id}: builder requirement {architecture!r} must be a mapping"
            )
        requirement["root"] = {
            "repository": validated_item["target_repository"],
            "tag": validated_item["target_tag"],
        }

    missing = sorted(expected_coordinates - seen)
    if missing:
        raise ValueError(
            f"missing builder selection for release profile architectures: {missing}"
        )
    extras = sorted(seen - expected_coordinates)
    if extras:
        raise ValueError(
            f"extra builder selection for release profile architectures: {extras}"
        )
    return release
