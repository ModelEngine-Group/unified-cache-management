"""Release cleanup records, exact resource coordinates, and count retention.

Publication snapshots preserve immutable run/attempt/stage ownership. Historical
Nightly aggregates remain readable; both formats normalize to the same internal
record. Internal attempt sources and versions never become producer wire fields.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from packaging.version import Version

from . import runtime, version_config
from .manifest import (
    SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS,
    ManifestError,
    validate_manifest,
)

CleanupError = ManifestError
CleanupRecordError = ManifestError
RELEASE_TYPES = frozenset({"stable", "prerelease", "draft", "nightly"})
RETRY_DELAYS_SECONDS = (0.0, 5.0, 15.0)
RELEASE_WORKFLOW_EVENTS = {
    ".github/workflows/release-tag.yml": frozenset({"push"}),
    ".github/workflows/release-nightly.yml": frozenset(
        {"schedule", "workflow_dispatch"}
    ),
}
RELEASE_WORKFLOWS = frozenset(RELEASE_WORKFLOW_EVENTS)


@dataclass(frozen=True)
class Resource:
    kind: str
    reference: str
    identifier: str | int | tuple[str, ...] | None = None
    holds_recovery_data: bool = False


@dataclass(frozen=True)
class ResourceFailure:
    resource: Resource
    attempts: int
    final_error: str


@dataclass(frozen=True)
class CleanupReport:
    """Confirm full deletion, or distinguish fatal failures from nonfatal GHCR skips."""

    tag: str
    completed: bool
    stopped_phase: int | None
    failures: tuple[ResourceFailure, ...]
    skipped: tuple[ResourceFailure, ...] = ()


class ActiveRelease(CleanupError):
    """A publication run is still using the selected version's resources."""


@dataclass(frozen=True)
class CleanupTarget:
    """Exact resources and the objects preserving their recovery evidence."""

    tag: str
    registry: tuple[Resource, ...]
    actions: tuple[Resource, ...]
    releases: tuple[Resource, ...]

    @property
    def run_ids(self) -> tuple[int, ...]:
        return tuple(int(action.identifier) for action in self.actions)


@dataclass(frozen=True)
class PendingRecovery:
    tag: str
    run: dict[str, Any]
    source_sha: str
    plan: dict[str, Any] | None
    release_id: int


@dataclass(frozen=True)
class CleanupResolution:
    target: CleanupTarget
    pending_recovery: tuple[PendingRecovery, ...] = ()


RESOURCE_ORDER = {
    "chart-oci": 0,
    "ghcr-index": 1,
    "ghcr-member": 2,
    "dockerhub-index": 3,
    "dockerhub-member": 4,
}
OCI_REFERENCE = re.compile(
    r"(?P<repository>(?:ghcr\.io|docker\.io)/"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+)"
    r":(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})"
)


def registry_resources_from_references(
    references: Sequence[tuple[str, str]],
) -> list[Resource]:
    """Validate exact coordinates and bind each GHCR package's deletable Tags.

    A GHCR package version can carry multiple Tags. Its identifier includes all
    requested Tags in that package, so deletion cannot remove an unrelated Tag.
    """
    allowed_tags: dict[str, set[str]] = {}
    unique = sorted(
        set(references),
        key=lambda item: (RESOURCE_ORDER.get(item[0], len(RESOURCE_ORDER)), item[1]),
    )
    for kind, reference in unique:
        registry = "docker.io" if kind.startswith("dockerhub-") else "ghcr.io"
        match = OCI_REFERENCE.fullmatch(reference)
        if kind not in RESOURCE_ORDER or match is None:
            raise CleanupError("cleanup registry resource is invalid")
        if not match.group("repository").startswith(registry + "/"):
            raise CleanupError("cleanup resource registry does not match its kind")
        if registry == "ghcr.io":
            allowed_tags.setdefault(match.group("repository"), set()).add(
                match.group("tag")
            )
    result = []
    for kind, reference in unique:
        match = OCI_REFERENCE.fullmatch(reference)
        assert match is not None
        tags = allowed_tags.get(match.group("repository"))
        result.append(Resource(kind, reference, tuple(sorted(tags)) if tags else None))
    return result


def registry_resources(manifest: object) -> list[Resource]:
    """Project a validated public manifest in the registry deletion order."""
    validated = validate_manifest(manifest)
    references: list[tuple[str, str]] = []
    if validated["chart"] is not None and validated["chart"]["oci"] is not None:
        references.append(("chart-oci", validated["chart"]["oci"]))
    for image in validated["images"]:
        for channel, publication in image["publications"].items():
            if publication is None:
                continue
            kind = "index" if publication["multi_arch"] else "member"
            references.append((f"{channel}-{kind}", publication["pull"]))
            references.extend(
                (f"{channel}-member", member["reference"])
                for member in publication["members"]
            )
    return registry_resources_from_references(references)


def plan_references(plan: dict[str, Any]) -> list[tuple[str, str]]:
    """Project enabled targets after the caller validates plan identity and shape.

    Only family publication targets and this release's Chart are owned. Builder
    images, base runtimes, and Python package coordinates never enter cleanup.
    """
    references: list[tuple[str, str]] = []
    for family in plan["families"]:
        kind = "index" if family["create_index"] else "member"
        references.extend(
            (f"{channel}-{kind}", reference)
            for channel, reference in runtime.image_publication_targets(
                plan, family.get("published_reference")
            ).items()
        )
        for member in family["members"]:
            references.extend(
                (f"{channel}-member", reference)
                for channel, reference in runtime.image_publication_targets(
                    plan, member["reference"]
                ).items()
            )
    if plan["publish"]["chart_oci"]["enabled"]:
        chart = plan["chart"]
        namespace = plan["publish"]["chart_oci"].get("namespace")
        references.append(
            ("chart-oci", f"{namespace}/{chart.get('name')}:{chart.get('version')}")
        )
    return references


CLEANUP_FILENAME = "release-cleanup.json"
CLEANUP_KIND = "ucm-release-cleanup"
CLEANUP_SCHEMA_VERSION = 1
CLEANUP_RECORD_KIND = "ucm-release-targets"
CLEANUP_RECORD_PREFIX = "release-cleanup-"
_SNAPSHOT_NAME = re.compile(
    r"release-cleanup-(?P<run>[1-9][0-9]*)-(?P<attempt>[1-9][0-9]*)"
    r"-(?P<stage>opened|planned)\.json"
)
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]+")
_SHA = re.compile(r"[0-9a-f]{40}")
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
    if not isinstance(tag, str):
        raise CleanupRecordError("cleanup Tag must be a canonical release Tag")
    try:
        return version_config.classify_tag(tag)
    except ValueError as error:
        raise CleanupRecordError(str(error)) from error


def _version_matches_tag(value: object, classification: dict[str, object]) -> bool:
    """Bind labeled Drafts exactly while preserving historical local versions."""
    if not isinstance(value, str):
        return False
    try:
        parsed = Version(value)
    except ValueError:
        return False
    if str(parsed) != value:
        return False
    expected = str(classification["version"])
    if Version(expected).local is not None:
        return value == expected
    # Old plans recorded commit suffixes that were not part of their Git Tag.
    return parsed.public == expected


def _sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    match = version_config.NIGHTLY_TAG.fullmatch(record["tag"])
    if match is not None:
        return (
            int(match.group("date")),
            int(match.group("number")),
            tuple(int(part) for part in match.group("base").split(".")),
        )
    # Release IDs reflect creation order even when a Tag reuses an old commit.
    identifiers = record["release_ids"]
    return (min(identifiers, default=0), record["tag"])


def _registry_resource(value: object, repository: str) -> dict[str, str]:
    """Validate exact registry coordinates and the GitHub package owner."""
    item = _object(value, "cleanup resource")
    _keys(item, {"kind", "reference"}, "cleanup resource")
    kind, reference = item["kind"], item["reference"]
    if (
        not isinstance(kind, str)
        or kind not in RESOURCE_ORDER
        or not isinstance(reference, str)
    ):
        raise CleanupRecordError("cleanup resource kind or reference is invalid")
    registry_resources_from_references([(kind, reference)])
    match = OCI_REFERENCE.fullmatch(reference)
    assert match is not None
    package = match.group("repository")
    owner = repository.split("/", 1)[0].casefold()
    if package.startswith("ghcr.io/") and package.split("/")[1].casefold() != owner:
        raise CleanupRecordError("cleanup registry resource belongs to another owner")
    return {"kind": kind, "reference": reference}


def _resource(
    value: object,
    repository: str,
    classification: dict[str, object],
    versions: Sequence[str] = (),
) -> dict[str, str]:
    resource = _registry_resource(value, repository)
    kind, reference = resource["kind"], resource["reference"]
    match = OCI_REFERENCE.fullmatch(reference)
    assert match is not None
    tag = match.group("tag")
    if kind == "chart-oci":
        if tag != classification["chart_version"]:
            raise CleanupRecordError("Chart reference is not owned by this release")
    else:
        endings = []
        for version in versions or (str(classification["version"]),):
            suffix = "-ucm-" + runtime.oci_tag_version(version)
            endings.append(suffix)
            endings.extend(
                suffix + "-" + arch for arch in runtime.SUPPORTED_ARCHITECTURES
            )
        if not any(tag.endswith(ending) and tag != ending for ending in endings):
            raise CleanupRecordError(
                "Image reference is not owned by this release version"
            )
    return {"kind": kind, "reference": reference}


def validate_record(value: object, repository: str) -> dict[str, Any]:
    """Normalize internal evidence without claiming remote ownership.

    ``attempt_sources`` preserves each run binding when ordinary Tags move;
    ``versions`` preserves local version suffixes present in historical plans.
    Neither field is accepted by a producer wire decoder.
    """
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        raise CleanupRecordError("cleanup repository must be owner/name")
    record = _object(value, "cleanup record")
    if set(record) not in (
        _RECORD_KEYS,
        _RECORD_KEYS | {"attempt_sources", "versions"},
    ):
        raise CleanupRecordError(
            "cleanup record fields differ from the cleanup contract"
        )
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
    labeled_draft = Version(str(classification["version"])).local is not None
    source_sha = record["source_sha"]
    if source_sha is not None and (
        not isinstance(source_sha, str) or _SHA.fullmatch(source_sha) is None
    ):
        raise CleanupRecordError("cleanup source_sha must be a lowercase 40-digit SHA")
    release_ids = {
        _positive_int(item, "cleanup Release ID")
        for item in _array(record["release_ids"], "cleanup release_ids")
    }
    runs = set()
    for raw_binding in _array(record["runs"], "cleanup runs"):
        binding = _object(raw_binding, "cleanup run")
        _keys(binding, {"id", "attempt"}, "cleanup run")
        runs.add(
            (
                _positive_int(binding["id"], "cleanup run ID"),
                _positive_int(binding["attempt"], "cleanup run attempt"),
            )
        )
    versions = set()
    for version in record.get("versions", [str(classification["version"])]):
        if not isinstance(version, str):
            raise CleanupRecordError("cleanup version must be canonical")
        try:
            parsed = Version(version)
        except ValueError as error:
            raise CleanupRecordError("cleanup version must be canonical") from error
        if str(parsed) != version:
            raise CleanupRecordError("cleanup version must be canonical")
        if labeled_draft and not _version_matches_tag(version, classification):
            raise CleanupRecordError("cleanup version differs from its Tag")
        versions.add(version)
    if not versions:
        raise CleanupRecordError("cleanup record requires version coordinates")
    attempt_sources = {}
    for raw_binding in _array(
        record.get("attempt_sources", []), "cleanup attempt sources"
    ):
        binding = _object(raw_binding, "cleanup attempt source")
        _keys(binding, {"id", "attempt", "source_sha"}, "cleanup attempt source")
        key = (
            _positive_int(binding["id"], "cleanup source run ID"),
            _positive_int(binding["attempt"], "cleanup source run attempt"),
        )
        sha = binding["source_sha"]
        if key not in runs or not isinstance(sha, str) or _SHA.fullmatch(sha) is None:
            raise CleanupRecordError(
                "cleanup attempt source differs from its run binding"
            )
        previous = attempt_sources.setdefault(key, sha)
        if previous != sha:
            raise CleanupRecordError("one cleanup attempt has conflicting source SHAs")
    if source_sha is not None:
        for key in runs:
            previous = attempt_sources.setdefault(key, source_sha)
            if previous != source_sha:
                raise CleanupRecordError("cleanup record has conflicting source SHAs")
    sources = set(attempt_sources.values()) | ({source_sha} if source_sha else set())
    if classification["release_type"] == "nightly" and len(sources) > 1:
        raise CleanupRecordError(f"Nightly {record['tag']} has conflicting source SHAs")
    resources = {}
    for raw_resource in _array(record["resources"], "cleanup resources"):
        resource = (
            _resource(raw_resource, repository, classification, sorted(versions))
            if labeled_draft
            else _registry_resource(raw_resource, repository)
        )
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
        "source_sha": next(iter(sources)) if len(sources) == 1 else None,
        "release_ids": sorted(release_ids),
        "runs": [
            {"id": run_id, "attempt": attempt} for run_id, attempt in sorted(runs)
        ],
        "resources": sorted(
            resources.values(),
            key=lambda item: (RESOURCE_ORDER[item["kind"]], item["reference"]),
        ),
        "attempt_sources": [
            {"id": run_id, "attempt": attempt, "source_sha": sha}
            for (run_id, attempt), sha in sorted(attempt_sources.items())
        ],
        "versions": sorted(versions),
    }


def merge_records(
    records: Iterable[object], repository: str
) -> tuple[dict[str, Any], ...]:
    """Union attempts and Releases by Tag without losing moved-Tag provenance."""
    grouped = {}
    for value in records:
        record = validate_record(value, repository)
        previous = grouped.get(record["tag"])
        if previous is None:
            grouped[record["tag"]] = record
            continue
        sources = {
            binding["source_sha"]
            for item in (previous, record)
            for binding in item["attempt_sources"]
        } | {sha for sha in (previous["source_sha"], record["source_sha"]) if sha}
        if (
            _classification(record["tag"])["release_type"] == "nightly"
            and len(sources) > 1
        ):
            raise CleanupRecordError(
                f"Nightly {record['tag']} has conflicting source SHAs"
            )
        previous["source_sha"] = next(iter(sources)) if len(sources) == 1 else None
        for field in (
            "release_ids",
            "runs",
            "resources",
            "attempt_sources",
            "versions",
        ):
            previous[field].extend(record[field])
        grouped[record["tag"]] = validate_record(previous, repository)
    # Callers select one type. A mixed diagnostic catalog has a deterministic order.
    return tuple(
        sorted(
            grouped.values(),
            key=lambda record: (
                str(_classification(record["tag"])["release_type"]),
                _sort_key(record),
            ),
            reverse=True,
        )
    )


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
    """Add enabled publication coordinates; shared builder/base inputs stay absent."""
    binding = _object(record, "cleanup record")
    repository = binding.get("repository")
    result = validate_record(binding, repository)
    plan = _object(plan, "release plan")
    classification = _classification(result["tag"])
    version = plan.get("version")
    if (
        plan.get("kind") != "ucm-release-plan"
        or plan.get("repository") != repository
        or plan.get("route") != "release"
        or plan.get("release_type") != classification["release_type"]
        or plan.get("git_tag") != result["tag"]
        or not _version_matches_tag(version, classification)
        or (
            "image_version" in plan
            and plan["image_version"] != runtime.oci_tag_version(version)
        )
    ):
        raise CleanupRecordError("release plan identity differs from cleanup binding")
    for value in _array(plan.get("families"), "release plan families"):
        family = _object(value, "release family")
        if "published_reference" not in family:
            raise CleanupRecordError(
                "cleanup publication family must contain published_reference"
            )
    try:
        references = plan_references(plan)
    except (KeyError, TypeError, ValueError) as error:
        raise CleanupRecordError(f"invalid release plan structure: {error}") from error
    result["versions"].append(version)
    result["resources"].extend(
        _resource(
            {"kind": kind, "reference": reference},
            repository,
            classification,
            [version],
        )
        for kind, reference in references
    )
    return validate_record(result, repository)


def record_from_manifest(
    manifest: object,
    repository: str,
    release_id: int | None = None,
    source_sha: str | None = None,
) -> dict[str, Any]:
    """Project public manifests and historical Schema 6 for internal cleanup."""
    manifest = _object(manifest, "release manifest")
    schema_version = manifest.get("schema_version")
    resources: list[dict[str, str]] = []
    if (
        type(schema_version) is int
        and schema_version in SUPPORTED_RELEASE_MANIFEST_SCHEMA_VERSIONS
    ):
        validated = validate_manifest(manifest)
        release = validated["release"]
        tag, run_id = release["tag"], release["actions_run_id"]
        classification = _classification(tag)
        release_url = urlparse(release["url"])
        release_path = f"/{repository}/releases/tag/"
        if (
            release_url.scheme != "https"
            or release_url.netloc.casefold() != "github.com"
            or not release_url.path.casefold().startswith(release_path.casefold())
            or not release_url.path[len(release_path) :]
        ):
            raise CleanupRecordError(
                "release manifest URL belongs to another repository"
            )
        if (
            classification["release_type"] == "nightly"
            or Version(str(classification["version"])).local is not None
        ) and not _version_matches_tag(release["version"], classification):
            raise CleanupRecordError("release Manifest version differs from its Tag")
        resources = [
            {"kind": resource.kind, "reference": resource.reference}
            for resource in registry_resources(validated)
        ]
        version = release["version"]
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
            or manifest["release_type"]
            != _classification(manifest["tag"])["release_type"]
        ):
            raise CleanupRecordError("Schema 6 cleanup manifest identity is invalid")
        tag, run_id = manifest["tag"], manifest["actions_run_id"]
        version = str(_classification(tag)["version"])
        assets = _array(manifest["github_release_assets"], "Schema 6 Release assets")
        if (
            any(not isinstance(item, str) or not item for item in assets)
            or "release-manifest.json" not in assets
        ):
            raise CleanupRecordError("Schema 6 Release assets are invalid")
        # Historical producers used different asset-name contracts. Preserve each
        # reader's accepted shape without spreading those rules to new snapshots.
        if manifest["release_type"] == "nightly":
            invalid_assets = any("/" in item for item in assets)
        else:
            invalid_assets = len(set(assets)) != len(assets)
        if invalid_assets:
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
        raise CleanupRecordError(
            "cleanup requires a supported public release Manifest or Schema 6"
        )
    if manifest["schema_version"] == 6:
        resources = [
            _resource(resource, repository, _classification(tag), [version])
            for resource in resources
        ]
    record = validate_record(
        {
            "kind": CLEANUP_KIND,
            "schema_version": CLEANUP_SCHEMA_VERSION,
            "repository": repository,
            "tag": tag,
            "source_sha": source_sha,
            "release_ids": [] if release_id is None else [release_id],
            "runs": [{"id": run_id, "attempt": 1}],
            "resources": resources,
            "attempt_sources": (
                []
                if source_sha is None
                else [{"id": run_id, "attempt": 1, "source_sha": source_sha}]
            ),
            "versions": [version],
        },
        repository,
    )
    classification = _classification(tag)
    if classification["release_type"] == "nightly":
        if schema_version in {6, 9}:
            _legacy_nightly_resources(record, repository)
        else:
            for resource in record["resources"]:
                _resource(resource, repository, classification, [version])
    return record


def _legacy_nightly_resources(record: dict[str, Any], repository: str) -> None:
    """Preserve the resource restrictions of historical Nightly wire formats."""
    classification = _classification(record["tag"])
    suffix = "-ucm-" + str(classification["image_version"])
    for value in record["resources"]:
        resource = _resource(value, repository, classification)
        package, tag = runtime.parse_runtime_reference(resource["reference"])
        kind = resource["kind"]
        owner = repository.split("/", 1)[0].lower()
        if kind == "chart-oci":
            if package != f"ghcr.io/{owner}/charts/unified-cache-chart":
                raise CleanupRecordError("Chart reference is not owned by this Nightly")
        else:
            registry = "ghcr.io" if kind.startswith("ghcr-") else "docker.io"
            components = package.split("/")
            if (
                len(components) != 3
                or components[0] != registry
                or components[2] not in {"vllm-openai", "vllm-ascend"}
            ):
                raise CleanupRecordError(
                    "Image repository is not a release publication target"
                )
            endings = [suffix]
            if kind.endswith("-member"):
                endings.extend(
                    suffix + "-" + arch for arch in runtime.SUPPORTED_ARCHITECTURES
                )
            if not any(tag.endswith(ending) and tag != ending for ending in endings):
                raise CleanupRecordError(
                    "Image reference is not owned by this Nightly version"
                )


def decode_legacy_record(value: object, repository: str) -> dict[str, Any]:
    """Read the historical eight-field Nightly aggregate wire contract."""
    wire = _object(value, "cleanup record")
    _keys(wire, _RECORD_KEYS, "cleanup record")
    record = validate_record(wire, repository)
    if _classification(record["tag"])["release_type"] != "nightly":
        raise CleanupRecordError("legacy cleanup record must describe a Nightly")
    _legacy_nightly_resources(record, repository)
    return record


def decode_snapshot(
    value: object,
    repository: str,
    filename: str,
    release_id: int | None = None,
) -> dict[str, Any]:
    """Bind an immutable snapshot to its filename and containing Release ID.

    Attempts and stages belong to the established filename contract. The JSON
    remains schema 1, and producer input cannot add internal provenance fields.
    """
    document = _object(value, "cleanup snapshot")
    _keys(
        document,
        {
            "kind",
            "schema_version",
            "repository",
            "tag",
            "release_type",
            "run_id",
            "source_sha",
            "version",
            "resources",
        },
        "cleanup snapshot",
    )
    match = _SNAPSHOT_NAME.fullmatch(filename) if isinstance(filename, str) else None
    if match is None:
        raise CleanupRecordError("cleanup recovery asset name is malformed")
    coordinates = _classification(document.get("tag"))
    if (
        document.get("kind") != CLEANUP_RECORD_KIND
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or not isinstance(document.get("repository"), str)
        or document["repository"].casefold() != repository.casefold()
        or document.get("release_type") != coordinates["release_type"]
        or type(document.get("run_id")) is not int
        or document["run_id"] != int(match.group("run"))
        or not isinstance(document.get("source_sha"), str)
        or _SHA.fullmatch(document["source_sha"]) is None
    ):
        raise CleanupRecordError(
            "cleanup recovery record differs from its Tag provenance"
        )
    resources = _array(document["resources"], "cleanup snapshot resources")
    if match.group("stage") == "opened" and resources:
        raise CleanupRecordError("opened cleanup snapshot must not contain resources")
    version = document["version"]
    if not _version_matches_tag(version, coordinates):
        raise CleanupRecordError("cleanup recovery version differs from its Tag")
    resources = [
        _resource(resource, repository, coordinates, [version])
        for resource in resources
    ]
    attempt = int(match.group("attempt"))
    return validate_record(
        {
            "kind": CLEANUP_KIND,
            "schema_version": CLEANUP_SCHEMA_VERSION,
            "repository": document["repository"],
            "tag": document["tag"],
            "source_sha": document["source_sha"],
            "release_ids": [] if release_id is None else [release_id],
            "runs": [{"id": document["run_id"], "attempt": attempt}],
            "resources": resources,
            "attempt_sources": [
                {
                    "id": document["run_id"],
                    "attempt": attempt,
                    "source_sha": document["source_sha"],
                }
            ],
            "versions": [document["version"]],
        },
        repository,
    )


def decode_record(
    value: object,
    repository: str,
    *,
    asset_name: str | None = None,
    release_id: int | None = None,
) -> dict[str, Any]:
    """Decode either existing wire format, preserving its storage context."""
    document = _object(value, "cleanup evidence")
    if document.get("kind") == CLEANUP_RECORD_KIND:
        return decode_snapshot(document, repository, asset_name, release_id)
    return decode_legacy_record(document, repository)


def new_record(
    repository: str,
    tag: str,
    source_sha: str,
    run_id: int,
    run_attempt: int,
    release_id: int | None = None,
    plan: object | None = None,
) -> dict[str, Any]:
    """Build the established per-attempt publication snapshot for any release type."""
    record = record_for_tag(
        repository, tag, source_sha, run_id, run_attempt, release_id
    )
    if plan is not None:
        record = record_from_plan(record, plan)
    classification = _classification(tag)
    version = str(classification["version"]) if plan is None else plan["version"]
    return {
        "kind": CLEANUP_RECORD_KIND,
        "schema_version": 1,
        "repository": repository,
        "tag": tag,
        "release_type": classification["release_type"],
        "run_id": run_id,
        "source_sha": source_sha,
        "version": version,
        "resources": record["resources"],
    }


def record_basename(record: object, run_attempt: int, *, planned: bool = False) -> str:
    """Return the immutable filename; an empty planned inventory is still planned."""
    document = _object(record, "cleanup snapshot")
    run_id = _positive_int(document.get("run_id"), "cleanup run ID")
    attempt = _positive_int(run_attempt, "cleanup run attempt")
    if not isinstance(planned, bool):
        raise CleanupRecordError("planned must be boolean")
    stage = "planned" if planned else "opened"
    return f"{CLEANUP_RECORD_PREFIX}{run_id}-{attempt}-{stage}.json"


def plan_resources(
    plan: dict[str, Any], *, repository: str, tag: str
) -> list[Resource]:
    """Validate a saved publication plan and project its exact enabled targets."""
    record = {
        "kind": CLEANUP_KIND,
        "schema_version": CLEANUP_SCHEMA_VERSION,
        "repository": repository,
        "tag": tag,
        "source_sha": None,
        "release_ids": [],
        "runs": [],
        "resources": [],
    }
    normalized = record_from_plan(record, plan)
    return registry_resources_from_references(
        [(item["kind"], item["reference"]) for item in normalized["resources"]]
    )


@dataclass(frozen=True)
class RetentionSelection:
    kept: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]
    deferred: tuple[dict[str, Any], ...]
    skipped_reason: str | None = None


def select_retention(
    records: Sequence[object],
    max_count: int,
    active_tags: Iterable[str] = (),
    *,
    release_type: str | None = None,
    current_tag: str | None = None,
    pypi_enabled: bool = False,
) -> RetentionSelection:
    """Count actual unique versions and defer active excess versions.

    Nightly ordering uses its date and numeric sequence. Other types use the
    oldest Release ID for each Tag; a rerun of an old current Tag cannot evict a
    newer version. A current Tag missing its Release reserves its existing slot.
    """
    if type(max_count) is not int or max_count == 0 or max_count < -1:
        raise CleanupRecordError("max_count must be -1 or an integer >= 1")
    if not isinstance(pypi_enabled, bool):
        raise CleanupRecordError("pypi_enabled must be boolean")
    if release_type is not None and release_type not in RELEASE_TYPES:
        raise CleanupRecordError("retention release type is invalid")
    if current_tag is not None:
        current_type = str(_classification(current_tag)["release_type"])
        if release_type is not None and release_type != current_type:
            raise CleanupRecordError("current Tag differs from retention release type")
        release_type = current_type
    active = set(active_tags)
    for tag in active:
        _classification(tag)
    merged = ()
    if records:
        repository = _object(records[0], "cleanup record").get("repository")
        merged = merge_records(records, repository)
        types = {
            str(_classification(record["tag"])["release_type"]) for record in merged
        }
        if release_type is None:
            if len(types) != 1:
                raise CleanupRecordError("retention requires one release type")
            release_type = next(iter(types))
        merged = tuple(
            record
            for record in merged
            if _classification(record["tag"])["release_type"] == release_type
        )
    if max_count == -1:
        return RetentionSelection(
            merged, (), (), "retention skipped: max_count is unlimited"
        )
    if pypi_enabled:
        return RetentionSelection(
            merged, (), (), "retention skipped: PyPI is enabled for this release type"
        )
    if release_type == "nightly":
        excess = merged[max_count:]
    else:
        count = len(merged) + int(
            current_tag is not None
            and all(record["tag"] != current_tag for record in merged)
        )
        excess_count = max(0, count - max_count)
        excess = tuple(sorted(merged, key=_sort_key)[:excess_count])
    candidate_tags = {
        record["tag"] for record in excess if record["tag"] != current_tag
    }
    return RetentionSelection(
        tuple(record for record in merged if record["tag"] not in candidate_tags),
        tuple(
            record
            for record in excess
            if record["tag"] in candidate_tags and record["tag"] not in active
        ),
        tuple(
            record
            for record in excess
            if record["tag"] in candidate_tags and record["tag"] in active
        ),
    )
