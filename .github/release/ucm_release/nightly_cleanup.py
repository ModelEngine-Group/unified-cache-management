"""Nightly identity, exact publication inventory, and count retention.

This internal record can describe an unfinished publication. The public release
manifest remains the contract for a complete installation. Transport and proof
that a record belongs to a trusted workflow are owned by the cleanup entrypoint.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from . import runtime, version_config
from .manifest import ManifestError, validate_manifest

CLEANUP_FILENAME = "release-cleanup.json"
CLEANUP_KIND = "ucm-release-cleanup"
CLEANUP_SCHEMA_VERSION = 1
CleanupRecordError = ManifestError
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]+")
_SHA = re.compile(r"[0-9a-f]{40}")
_RESOURCE_ORDER = {
    "chart-oci": 0,
    "ghcr-index": 1,
    "ghcr-member": 2,
    "dockerhub-index": 3,
    "dockerhub-member": 4,
}
_RECORD_KEYS = {
    "kind",
    "schema_version",
    "repository",
    "tag",
    "source_sha",
    "release_ids",
    "runs",
    "resources",
}


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CleanupRecordError(f"{context} must be an object")
    return value


def _keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise CleanupRecordError(f"{context} fields differ from the cleanup contract")


def _array(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise CleanupRecordError(f"{context} must be an array")
    return value


def _positive_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CleanupRecordError(f"{context} must be a positive integer")
    return value


def _classification(tag: object) -> dict[str, object]:
    if not isinstance(tag, str) or version_config.NIGHTLY_TAG.fullmatch(tag) is None:
        raise CleanupRecordError("cleanup Tag must be a canonical Nightly Tag")
    try:
        return version_config.classify_tag(tag)
    except ValueError as error:
        raise CleanupRecordError(str(error)) from error


def _sort_key(record: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    match = version_config.NIGHTLY_TAG.fullmatch(record["tag"])
    assert match is not None  # validate_record establishes this contract.
    return (
        int(match.group("date")),
        int(match.group("number")),
        tuple(int(part) for part in match.group("base").split(".")),
    )


def _resource(
    value: object, repository: str, classification: dict[str, object]
) -> dict[str, str]:
    item = _object(value, "cleanup resource")
    _keys(item, {"kind", "reference"}, "cleanup resource")
    kind, reference = item["kind"], item["reference"]
    if not isinstance(kind, str) or kind not in _RESOURCE_ORDER:
        raise CleanupRecordError("cleanup resource kind is unsupported")
    try:
        package, tag = runtime.parse_runtime_reference(reference)
    except ValueError as error:
        raise CleanupRecordError(str(error)) from error
    # parse_runtime_reference accepts surrounding whitespace for interactive use;
    # destructive inventories must preserve one exact coordinate.
    if reference != f"{package}:{tag}":
        raise CleanupRecordError("cleanup resource reference must be exact")
    owner = repository.split("/", 1)[0].lower()
    if kind == "chart-oci":
        expected = f"ghcr.io/{owner}/charts/unified-cache-chart"
        if package != expected or tag != classification["chart_version"]:
            raise CleanupRecordError("Chart reference is not owned by this Nightly")
    else:
        registry = "ghcr.io" if kind.startswith("ghcr-") else "docker.io"
        components = package.split("/")
        if (
            len(components) != 3
            or components[0] != registry
            or components[2] not in {"vllm-openai", "vllm-ascend"}
            or (registry == "ghcr.io" and components[1] != owner)
        ):
            raise CleanupRecordError(
                "Image repository is not a release publication target"
            )
        # The Docker Hub namespace is repository configuration, not necessarily
        # its GitHub owner. The transport layer binds it to trusted plan evidence.
        suffix = f"-ucm-{classification['image_version']}"
        endings = [suffix]
        if kind.endswith("-member"):
            endings += [f"{suffix}-{arch}" for arch in runtime.SUPPORTED_ARCHITECTURES]
        if not any(tag.endswith(ending) and tag != ending for ending in endings):
            raise CleanupRecordError(
                "Image reference is not owned by this Nightly version"
            )
    return {"kind": kind, "reference": reference}


def validate_record(value: object, repository: str) -> dict[str, Any]:
    """Validate and normalize an internal record without claiming remote ownership.

    A recovered historical record may have no source SHA or surviving run. The
    caller must establish its provenance before allowing any deletion.
    """
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        raise CleanupRecordError("cleanup repository must be owner/name")
    record = _object(value, "cleanup record")
    _keys(record, _RECORD_KEYS, "cleanup record")
    if (
        record["kind"] != CLEANUP_KIND
        or type(record["schema_version"]) is not int
        or record["schema_version"] != CLEANUP_SCHEMA_VERSION
    ):
        raise CleanupRecordError("unsupported cleanup record kind or schema_version")
    if (
        not isinstance(record["repository"], str)
        or record["repository"].casefold() != repository.casefold()
    ):
        raise CleanupRecordError(
            "cleanup record repository differs from requested repository"
        )
    classification = _classification(record["tag"])
    source_sha = record["source_sha"]
    if source_sha is not None and (
        not isinstance(source_sha, str) or _SHA.fullmatch(source_sha) is None
    ):
        raise CleanupRecordError("cleanup source_sha must be a lowercase 40-digit SHA")
    release_ids = {
        _positive_int(item, "cleanup Release ID")
        for item in _array(record["release_ids"], "cleanup release_ids")
    }
    runs: set[tuple[int, int]] = set()
    for raw_run in _array(record["runs"], "cleanup runs"):
        run = _object(raw_run, "cleanup run")
        _keys(run, {"id", "attempt"}, "cleanup run")
        runs.add(
            (
                _positive_int(run["id"], "cleanup run ID"),
                _positive_int(run["attempt"], "cleanup run attempt"),
            )
        )
    resources: dict[str, dict[str, str]] = {}
    for raw_resource in _array(record["resources"], "cleanup resources"):
        resource = _resource(raw_resource, repository, classification)
        previous = resources.setdefault(resource["reference"], resource)
        if previous != resource:
            raise CleanupRecordError(
                "one cleanup reference has conflicting resource kinds"
            )
    return {
        "kind": CLEANUP_KIND,
        "schema_version": CLEANUP_SCHEMA_VERSION,
        "repository": repository,
        "tag": record["tag"],
        "source_sha": source_sha,
        "release_ids": sorted(release_ids),
        "runs": [
            {"id": run_id, "attempt": attempt} for run_id, attempt in sorted(runs)
        ],
        "resources": sorted(
            resources.values(),
            key=lambda item: (_RESOURCE_ORDER[item["kind"]], item["reference"]),
        ),
    }


def merge_records(
    records: Iterable[object], repository: str
) -> tuple[dict[str, Any], ...]:
    """Union attempts and duplicate Releases by Tag; conflicting sources fail closed."""
    grouped: dict[str, dict[str, Any]] = {}
    for value in records:
        record = validate_record(value, repository)
        previous = grouped.get(record["tag"])
        if previous is None:
            grouped[record["tag"]] = record
            continue
        sources = {sha for sha in (previous["source_sha"], record["source_sha"]) if sha}
        if len(sources) > 1:
            raise CleanupRecordError(
                f"Nightly {record['tag']} has conflicting source SHAs"
            )
        previous["source_sha"] = next(iter(sources), None)
        for field in ("release_ids", "runs", "resources"):
            previous[field].extend(record[field])
        grouped[record["tag"]] = validate_record(previous, repository)
    return tuple(sorted(grouped.values(), key=_sort_key, reverse=True))


def record_for_tag(
    repository: str,
    tag: str,
    source_sha: str,
    run_id: int,
    run_attempt: int,
    release_id: int | None = None,
) -> dict[str, Any]:
    """Create the initial binding before planning or publication can fail."""
    if source_sha is None:
        raise CleanupRecordError("a new cleanup record requires its source SHA")
    return validate_record(
        {
            "kind": CLEANUP_KIND,
            "schema_version": CLEANUP_SCHEMA_VERSION,
            "repository": repository,
            "tag": tag,
            "source_sha": source_sha,
            "release_ids": [] if release_id is None else [release_id],
            "runs": [{"id": run_id, "attempt": run_attempt}],
            "resources": [],
        },
        repository,
    )


def record_from_plan(record: object, plan: object) -> dict[str, Any]:
    """Add all exact enabled release targets before the first publication.

    Builder images, base runtimes, and Python package coordinates are deliberately
    absent: only family publication targets and this release's Chart are owned.
    """
    binding = _object(record, "cleanup record")
    repository = binding.get("repository")
    result = validate_record(binding, repository)
    plan = _object(plan, "release plan")
    classification = _classification(result["tag"])
    if (
        plan.get("kind") != "ucm-release-plan"
        or plan.get("repository") != repository
        or plan.get("route") != "release"
        or plan.get("release_type") != "nightly"
        or plan.get("git_tag") != result["tag"]
        or plan.get("version") != classification["version"]
        or plan.get("image_version") != classification["image_version"]
    ):
        raise CleanupRecordError("release plan identity differs from cleanup binding")
    publish = _object(plan.get("publish"), "release plan publish")
    for channel in ("ghcr", "dockerhub", "chart_oci"):
        if not isinstance(_object(publish.get(channel), channel).get("enabled"), bool):
            raise CleanupRecordError(f"{channel} publication enabled must be boolean")
    for value in _array(plan.get("families"), "release plan families"):
        family = _object(value, "release family")
        create_index = family.get("create_index")
        if not isinstance(create_index, bool):
            raise CleanupRecordError("release family create_index must be boolean")
        members = _array(family.get("members"), "release family members")
        references: set[str] = set()
        architectures: set[str] = set()
        for value in members:
            member = _object(value, "release family member")
            architecture = member.get("cpu_arch")
            reference = member.get("reference")
            if (
                architecture not in runtime.SUPPORTED_ARCHITECTURES
                or architecture in architectures
            ):
                raise CleanupRecordError(
                    "release family member architecture is invalid or duplicated"
                )
            _resource(
                {"kind": "ghcr-member", "reference": reference},
                repository,
                classification,
            )
            if reference in references:
                raise CleanupRecordError(
                    "release family member reference is duplicated"
                )
            architectures.add(architecture)
            references.add(reference)
            for channel, target in runtime.image_publication_targets(
                plan, reference
            ).items():
                result["resources"].append(
                    {"kind": f"{channel}-member", "reference": target}
                )
        published = family.get("published_reference")
        if (
            not members
            or (create_index and (len(members) < 2 or published in references))
            or (not create_index and (len(members) != 1 or published not in references))
        ):
            raise CleanupRecordError("release family index and members disagree")
        if create_index:
            for channel, target in runtime.image_publication_targets(
                plan, published
            ).items():
                result["resources"].append(
                    {"kind": f"{channel}-index", "reference": target}
                )
    if publish["chart_oci"]["enabled"]:
        chart = _object(plan.get("chart"), "release plan Chart")
        namespace = publish["chart_oci"].get("namespace")
        reference = f"{namespace}/{chart.get('name')}:{chart.get('version')}"
        result["resources"].append({"kind": "chart-oci", "reference": reference})
    return validate_record(result, repository)


def record_from_manifest(
    manifest: object,
    repository: str,
    release_id: int | None = None,
    source_sha: str | None = None,
) -> dict[str, Any]:
    """Project supported Schema 6/9 manifests solely for internal cleanup."""
    manifest = _object(manifest, "release manifest")
    resources: list[dict[str, str]] = []
    if manifest.get("schema_version") == 9:
        validated = validate_manifest(manifest)
        release = validated["release"]
        if release["type"] != "nightly":
            raise CleanupRecordError("cleanup manifest must describe a Nightly")
        tag, run_id = release["tag"], release["actions_run_id"]
        if release["version"] != _classification(tag)["version"]:
            raise CleanupRecordError(
                "release Manifest version differs from its Nightly Tag"
            )
        if validated["chart"] is not None and validated["chart"]["oci"] is not None:
            resources.append(
                {"kind": "chart-oci", "reference": validated["chart"]["oci"]}
            )
        for image in validated["images"]:
            for channel, publication in image["publications"].items():
                if publication is None:
                    continue
                kind = "index" if publication["multi_arch"] else "member"
                resources.append(
                    {"kind": f"{channel}-{kind}", "reference": publication["pull"]}
                )
                resources.extend(
                    {"kind": f"{channel}-member", "reference": member["reference"]}
                    for member in publication["members"]
                )
    elif (
        type(manifest.get("schema_version")) is int and manifest["schema_version"] == 6
    ):
        _keys(
            manifest,
            {
                "kind",
                "schema_version",
                "tag",
                "release_type",
                "actions_run_id",
                "chart_oci",
                "github_release_assets",
                "runtime_images",
            },
            "Schema 6 cleanup manifest",
        )
        if (
            manifest["kind"] != "ucm-release-manifest"
            or manifest["release_type"] != "nightly"
        ):
            raise CleanupRecordError(
                "Schema 6 cleanup manifest must describe a Nightly"
            )
        tag, run_id = manifest["tag"], manifest["actions_run_id"]
        assets = _array(manifest["github_release_assets"], "Schema 6 Release assets")
        if (
            any(not isinstance(item, str) or not item or "/" in item for item in assets)
            or "release-manifest.json" not in assets
        ):
            raise CleanupRecordError("Schema 6 Release assets are invalid")
        if manifest["chart_oci"] is not None:
            resources.append({"kind": "chart-oci", "reference": manifest["chart_oci"]})
        images = _object(manifest["runtime_images"], "Schema 6 runtime_images")
        _keys(images, {"ghcr", "dockerhub"}, "Schema 6 runtime_images")
        for channel, value in images.items():
            groups = _object(value, "Schema 6 image channel")
            _keys(groups, {"indexes", "members"}, "Schema 6 image channel")
            for plural, kind in (("indexes", "index"), ("members", "member")):
                references = _array(groups[plural], "Schema 6 image references")
                resources.extend(
                    {"kind": f"{channel}-{kind}", "reference": reference}
                    for reference in references
                )
    else:
        raise CleanupRecordError("cleanup supports only release Manifest Schema 6 or 9")
    return validate_record(
        {
            "kind": CLEANUP_KIND,
            "schema_version": CLEANUP_SCHEMA_VERSION,
            "repository": repository,
            "tag": tag,
            "source_sha": source_sha,
            "release_ids": [] if release_id is None else [release_id],
            "runs": [{"id": run_id, "attempt": 1}],
            "resources": resources,
        },
        repository,
    )


@dataclass(frozen=True)
class NightlyRetention:
    kept: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]
    deferred: tuple[dict[str, Any], ...]


def select_retention(
    records: Sequence[object],
    max_count: int,
    active_tags: Iterable[str] = (),
) -> NightlyRetention:
    """Count actual unique versions; protect active excess versions from deletion."""
    if type(max_count) is not int or max_count == 0 or max_count < -1:
        raise CleanupRecordError("max_count must be -1 or an integer >= 1")
    active = set(active_tags)
    for tag in active:
        _classification(tag)
    if not records:
        return NightlyRetention((), (), ())
    repository = _object(records[0], "cleanup record").get("repository")
    merged = merge_records(records, repository)
    if max_count == -1:
        return NightlyRetention(merged, (), ())
    excess = merged[max_count:]
    return NightlyRetention(
        merged[:max_count],
        tuple(record for record in excess if record["tag"] not in active),
        tuple(record for record in excess if record["tag"] in active),
    )
