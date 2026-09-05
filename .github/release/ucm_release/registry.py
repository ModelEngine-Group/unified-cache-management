# fmt: off
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from . import builders, core
from .core import _LINKED_WHEEL_FIELDS, _RUNTIME_KEYS

DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
OCI_TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
REPOSITORY_RE = re.compile('[a-z0-9]+(?:[._:-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+')  # fmt: skip  # noqa: E501
RESOLVED_TASK_FIELDS = {'wheel': frozenset('task_id spec_id profile_id accelerator accelerator_runtime npu_arch_or_na os cpu_arch python_version python_abi wheel_version wheel_platform binary_profile_id dist_name validation_targets required_native forbidden_native allowed_dt_needed external_required_dependencies declaration_sha256 runner platform builder builder_sha256 build dependency_lock_sha256 dependency_lock runtime_requirements write_authority build_eligible artifact_name task_sha256'.split()), 'image': frozenset('task_id family_task_id wheel_task_id spec_id profile_id compatibility_rule_id runner cpu_arch platform builder builder_sha256 build runtime runtime_sha256 target_repository target_tag python_abi python_version wheel_version wheel_platform required_native forbidden_native allowed_dt_needed external_required_dependencies dependency_lock_sha256 dependency_lock runtime_requirements write_authority build_eligible artifact_name wheel_artifact_name task_sha256'.split()), 'family': frozenset('task_id product_id control_task_id control_arch control_runner runner cpu_arch platform builder builder_sha256 runtime runtime_sha256 snapshot_sha256 target_repository target_tag image_task_ids wheel_task_ids member_set_sha256 write_authority artifact_name task_sha256'.split())}  # fmt: skip  # noqa: E501
_RESOLVED_PLAN_FIELDS = frozenset('kind schema_version fixture_only lane source chart publish config_sha256 source_sha256 scan_sha256 resolved_upstreams wheel_tasks image_tasks family_tasks github_wheel_matrix github_image_matrix github_family_matrix expected_artifacts exclusions operations counts resolved_plan_sha256'.split())  # fmt: skip  # noqa: E501
_PLAN_SOURCE_KEYS = frozenset('repository staging_repository default_branch release_tag ucm_version commit'.split())  # fmt: skip  # noqa: E501
_PLAN_CHART_KEYS = frozenset('source name version app_version validation_cases'.split())  # fmt: skip  # noqa: E501
_PLAN_OPERATION_TYPES = frozenset('crane-tag-list crane-digest crane-manifest crane-config fixture-tag-page-read fixture-snapshot-read'.split())  # fmt: skip  # noqa: E501
# Legacy Task 3 regression authority. Production resolution starts at
# ``resolve_catalog`` and must never consume these concrete fixture coordinates.
SNAPSHOT_KEYS = frozenset('schema_version kind repository upstream_tag index_digest platforms'.split())  # fmt: skip  # noqa: E501
PLATFORM_KEYS = frozenset("os architecture manifest_digest config_digest".split())
OCI_INDEX_MEDIA_TYPES = {'application/vnd.oci.image.index.v1+json', 'application/vnd.docker.distribution.manifest.list.v2+json'}  # fmt: skip  # noqa: E501
OCI_MANIFEST_MEDIA_TYPES = {'application/vnd.oci.image.manifest.v1+json', 'application/vnd.docker.distribution.manifest.v2+json'}  # fmt: skip  # noqa: E501
CRANE_VERSION = "0.20.3"
SECONDARY_RATE_LIMIT_BACKOFF_SECONDS = (60.0, 120.0, 240.0)
IDEMPOTENT_REGISTRY_READ_OPERATIONS = frozenset({'blob', 'digest', 'ls', 'manifest', 'validate'})  # fmt: skip  # noqa: E501
SECONDARY_RATE_LIMIT_MARKERS = ('you have exceeded a secondary rate limit', 'you have triggered an abuse detection mechanism')  # fmt: skip  # noqa: E501
CRANE_BINARY_SHA256 = {('linux', 'x86_64'): 'sha256:675f3b2f1696c1f6bc55b1ef535163364119776999f3d1471e4558ed35bab548', ('linux', 'aarch64'): 'sha256:34bdb2ae7a56139c69cf745ab5cad3d7368e69896d8980e7bcf1ca194854a2ef', ('darwin', 'arm64'): 'sha256:d34f51061a226d1b183480cc7fdc1f7ec410676445cbb2432d89900ac2eb1cb3'}  # fmt: skip  # noqa: E501
CONTENT_IDENTITY_LAYER_KEYS = {"mediaType", "digest", "size"}
BUILDKIT_REWRITTEN_TIMESTAMP_ANNOTATION = "buildkit/rewritten-timestamp"


class RegistryBlocker(ValueError):

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _created_epoch(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value
        )
        is None
    ):
        raise ValueError(f"{label} created timestamp is invalid")
    try:
        parsed = dt.datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)  # fmt: skip  # noqa: E501
    except ValueError as error:
        raise ValueError(f"{label} created timestamp is invalid") from error
    return str(int(parsed.timestamp()))


def _validate_layer_descriptor_annotations(
    descriptor: dict[str, Any], *, created: object, label: str
) -> dict[str, str] | None:
    allowed_keys = CONTENT_IDENTITY_LAYER_KEYS | {"annotations"}
    if set(descriptor) not in (CONTENT_IDENTITY_LAYER_KEYS, allowed_keys):
        raise ValueError(f"{label} descriptor fields are invalid")
    if "annotations" not in descriptor:
        return None
    annotations = descriptor["annotations"]
    if not isinstance(annotations, dict) or set(annotations) != {
        BUILDKIT_REWRITTEN_TIMESTAMP_ANNOTATION
    }:
        raise ValueError(f"{label} descriptor annotations are invalid")
    timestamp = annotations[BUILDKIT_REWRITTEN_TIMESTAMP_ANNOTATION]
    if (
        not isinstance(timestamp, str)
        or re.fullmatch(r"(?:0|[1-9][0-9]*)", timestamp) is None
    ):
        raise ValueError(f"{label} rewritten timestamp annotation is invalid")
    if timestamp != _created_epoch(created, label): raise ValueError(f'{label} rewritten timestamp differs from created')  # noqa: E701,E501
    return copy.deepcopy(annotations)


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f'{label} fields mismatch: missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}')  # fmt: skip  # noqa: E501


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f'{label} must be an immutable sha256:<64 lowercase hex> digest')  # fmt: skip  # noqa: E501
    return value


def _repository(value: object) -> str:
    if not isinstance(value, str) or REPOSITORY_RE.fullmatch(value) is None:
        raise ValueError('repository must be a canonical lowercase OCI repository without tag or digest')  # fmt: skip  # noqa: E501
    return value


def _unique_json(text: str, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result: raise ValueError(f'duplicate JSON key {key!r} in {label}')  # noqa: E701,E501
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=pairs)
    if not isinstance(value, dict): raise ValueError(f'{label} must be a JSON object')  # noqa: E701,E501
    return value


def _crane_binary(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        if not path.is_file(): raise ValueError(f'crane executable does not exist: {value}')  # noqa: E701,E501
        return value
    if "/" in value or re.fullmatch(r"[A-Za-z0-9_.+-]+", value) is None:
        raise ValueError('crane executable must be an absolute path or an explicit PATH name')  # fmt: skip  # noqa: E501
    return value


def _host_platform_key() -> tuple[str, str]:
    system = platform.system().lower()
    cpu_authority = core.host_cpu_toolchain_authority(platform.machine())
    machine = 'arm64' if system == 'darwin' and cpu_authority.cpu_arch == 'arm64' else cpu_authority.wheel_arch  # fmt: skip  # noqa: E501
    return system, machine


def _minimal_registry_environment(
    *, docker_config: str | None = None
) -> dict[str, str]:
    environment: dict[str, str] = {}
    selected_config = docker_config or os.environ.get("DOCKER_CONFIG")
    if selected_config: environment['DOCKER_CONFIG'] = selected_config  # noqa: E701
    for key in (
        "HOME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        if key in os.environ: environment[key] = os.environ[key]  # noqa: E701
    return environment


def resolve_pinned_crane() -> str:
    located = shutil.which("crane")
    if located is None: raise ValueError('pinned crane v0.20.3 is not installed on PATH')  # noqa: E701,E501
    executable = str(Path(located).resolve())
    result = subprocess.run([executable, 'version'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_minimal_registry_environment(), check=False)  # fmt: skip  # noqa: E501
    version = result.stdout.strip()
    if result.returncode != 0 or version not in {CRANE_VERSION, "v" + CRANE_VERSION}:
        raise ValueError(f"registry transport requires crane v0.20.3, got {version!r}")
    expected = CRANE_BINARY_SHA256.get(_host_platform_key())
    if expected is None: raise ValueError(f'unsupported crane host platform: {_host_platform_key()}')  # noqa: E701,E501
    observed = "sha256:" + hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    if observed != expected: raise ValueError(f'crane binary digest mismatch: expected {expected}, observed {observed}')  # noqa: E701,E501
    return executable


def _crane(crane_binary: str, operation: str, reference: str) -> str:
    if operation not in {"digest", "manifest", "config"}:
        raise ValueError('only read-only crane digest, manifest, and config operations are allowed')  # fmt: skip  # noqa: E501
    try:
        result = subprocess.run([_crane_binary(crane_binary), operation, reference], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_minimal_registry_environment(), check=False)  # fmt: skip  # noqa: E501
    except OSError as error:
        raise ValueError(f"failed to execute pinned crane binary: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise ValueError(f"crane {operation} failed for {reference}: {detail}")
    return result.stdout.strip()


def enumerate_repository_tags(
    repository: str,
    *,
    fixture: dict[str, Any] | None = None,
    max_tags: int,
) -> dict[str, Any]:
    repository = _repository(repository)
    if not isinstance(max_tags, int) or isinstance(max_tags, bool) or max_tags < 1:
        raise ValueError("max_tags must be a positive integer")
    operations: list[dict[str, Any]] = []
    tags: list[str] = []
    if fixture is not None:
        if not isinstance(fixture, dict) or set(fixture) not in (
            {"pages", "snapshots"},
            {"pages", "snapshots", "list_error"},
        ):
            raise ValueError("registry enumeration fixture fields are noncanonical")
        if "list_error" in fixture:
            detail = fixture["list_error"]
            if not isinstance(detail, str) or not detail: raise ValueError('registry fixture list_error must be non-empty')  # noqa: E701,E501
            raise ValueError(f"fixture tag listing failed: {detail}")
        pages = fixture["pages"]
        if not isinstance(pages, list) or not pages: raise ValueError('registry enumeration fixture requires complete pages')  # noqa: E701,E501
        for page_index, page in enumerate(pages, start=1):
            if not isinstance(page, dict) or set(page) != {"tags", "next_page"}:
                raise ValueError(f"registry fixture page {page_index} is malformed")
            page_tags = page["tags"]
            if not isinstance(page_tags, list) or any(
                not isinstance(tag, str) or OCI_TAG_RE.fullmatch(tag) is None
                for tag in page_tags
            ):
                raise ValueError(f'registry fixture page {page_index} tags are malformed')  # fmt: skip  # noqa: E501
            expected_next = None if page_index == len(pages) else f'page-{page_index + 1}'  # fmt: skip  # noqa: E501
            if page['next_page'] != expected_next: raise ValueError('registry enumeration fixture pagination is incomplete')  # noqa: E701,E501
            tags.extend(page_tags)
            operations.append({'type': 'fixture-tag-page-read', 'capability': 'read', 'reference': repository, 'page': page_index})  # fmt: skip  # noqa: E501
    else:
        crane_binary = resolve_pinned_crane()
        result = _run_registry_tool(crane_binary, ["ls", repository])
        tags = result.stdout.splitlines()
        if any((OCI_TAG_RE.fullmatch(tag) is None for tag in tags)): raise ValueError('crane ls returned a malformed OCI tag')  # noqa: E701,E501
        operations.append({'type': 'crane-tag-list', 'capability': 'read', 'reference': repository})  # fmt: skip  # noqa: E501
    normalized = sorted(set(tags))
    if len(normalized) > max_tags:
        raise ValueError(f'registry tag limit max_tags={max_tags} exceeded by exact set of {len(normalized)}')  # fmt: skip  # noqa: E501
    return {'schema_version': 1, 'kind': 'registry-tag-list', 'repository': repository, 'tags': normalized, 'operations': operations}  # fmt: skip  # noqa: E501


def resolve_repository_tag(
    repository: str,
    upstream_tag: str,
    *,
    required_architectures: list[str],
    fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repository = _repository(repository)
    if not isinstance(upstream_tag, str) or OCI_TAG_RE.fullmatch(upstream_tag) is None:
        raise ValueError("upstream tag must use canonical OCI tag syntax")
    if (
        not isinstance(required_architectures, list)
        or not 1 <= len(required_architectures) <= 64
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", item) is None
            for item in required_architectures
        )
        or len(required_architectures) != len(set(required_architectures))
    ):
        raise ValueError("required architectures must be a bounded unique array")
    operations: list[dict[str, Any]] = []
    tagged_reference = f"{repository}:{upstream_tag}"
    if fixture is not None:
        raw_snapshot = copy.deepcopy(fixture)
        operations.append({'type': 'fixture-snapshot-read', 'capability': 'read', 'reference': tagged_reference})  # fmt: skip  # noqa: E501
    else:
        crane_binary = resolve_pinned_crane()
        operations.append({'type': 'crane-digest', 'capability': 'read', 'reference': tagged_reference})  # fmt: skip  # noqa: E501
        digest_result = _run_registry_tool(crane_binary, ["digest", tagged_reference])
        index_digest = _digest(digest_result.stdout.strip(), "crane index")
        index_reference = f"{repository}@{index_digest}"
        operations.append({'type': 'crane-manifest', 'capability': 'read', 'reference': index_reference})  # fmt: skip  # noqa: E501
        index_result = _run_registry_tool(crane_binary, ["manifest", index_reference])
        index = _unique_json(index_result.stdout, "crane index")
        if index.get("mediaType") not in OCI_INDEX_MEDIA_TYPES:
            raise ValueError("resolved index digest did not return an OCI/Docker index")
        descriptors = index.get("manifests")
        if not isinstance(descriptors, list): raise ValueError('crane index must contain a manifests array')  # noqa: E701,E501
        platforms: list[dict[str, Any]] = []
        for descriptor in descriptors:
            if not isinstance(descriptor, dict) or not isinstance(
                descriptor.get("platform"), dict
            ):
                raise ValueError("crane index descriptors require a platform object")
            if descriptor.get("mediaType") not in OCI_MANIFEST_MEDIA_TYPES:
                raise ValueError('index platform descriptor is not an OCI/Docker manifest')  # fmt: skip  # noqa: E501
            platform_value = descriptor["platform"]
            manifest_digest = _digest(descriptor.get("digest"), "platform manifest")
            child_reference = f"{repository}@{manifest_digest}"
            operations.append({'type': 'crane-manifest', 'capability': 'read', 'reference': child_reference})  # fmt: skip  # noqa: E501
            child_result = _run_registry_tool(crane_binary, ['manifest', child_reference])  # fmt: skip  # noqa: E501
            child = _unique_json(child_result.stdout, f'platform manifest {manifest_digest}')  # fmt: skip  # noqa: E501
            if child.get("mediaType") != descriptor.get("mediaType"):
                raise ValueError('platform manifest media type does not match index descriptor')  # fmt: skip  # noqa: E501
            config = child.get("config")
            if not isinstance(config, dict): raise ValueError('platform manifest requires a config descriptor')  # noqa: E701,E501
            platforms.append({'os': platform_value.get('os'), 'architecture': platform_value.get('architecture'), 'manifest_digest': manifest_digest, 'config_digest': _digest(config.get('digest'), 'platform config')})  # fmt: skip  # noqa: E501
        raw_snapshot = {'schema_version': 1, 'kind': 'upstream-registry-snapshot', 'repository': repository, 'upstream_tag': upstream_tag, 'index_digest': index_digest, 'platforms': platforms}  # fmt: skip  # noqa: E501
    if not isinstance(raw_snapshot, dict): raise ValueError('registry snapshot must be an object')  # noqa: E701,E501
    _exact_keys(raw_snapshot, SNAPSHOT_KEYS, "registry snapshot")
    if (
        raw_snapshot["schema_version"] != 1
        or raw_snapshot["kind"] != "upstream-registry-snapshot"
        or raw_snapshot["repository"] != repository
        or raw_snapshot["upstream_tag"] != upstream_tag
    ):
        raise ValueError("registry snapshot identity differs from exact request")
    index_digest = _digest(raw_snapshot["index_digest"], "snapshot index")
    platform_values = raw_snapshot["platforms"]
    if not isinstance(platform_values, list): raise ValueError('snapshot platforms must be an array')  # noqa: E701,E501
    members: dict[str, dict[str, str]] = {}
    digest_chain = [index_digest]
    for index, member in enumerate(platform_values):
        if not isinstance(member, dict): raise ValueError(f'snapshot platform {index} must be an object')  # noqa: E701,E501
        _exact_keys(member, PLATFORM_KEYS, f"snapshot platform {index}")
        architecture = member["architecture"]
        if member["os"] != "linux" or architecture not in required_architectures:
            raise ValueError("snapshot contains an unselected platform")
        if architecture in members: raise ValueError(f'duplicate snapshot platform: linux/{architecture}')  # noqa: E701,E501
        manifest_digest = _digest(member['manifest_digest'], f'linux/{architecture} manifest')  # fmt: skip  # noqa: E501
        config_digest = _digest(member["config_digest"], f"linux/{architecture} config")
        digest_chain.extend((manifest_digest, config_digest))
        members[architecture] = {'manifest_digest': manifest_digest, 'config_digest': config_digest}  # fmt: skip  # noqa: E501
    missing = sorted(set(required_architectures) - set(members))
    if missing:
        raise RegistryBlocker(f'missing-linux-{missing[0]}', f'snapshot is missing required linux architectures: {missing}')  # fmt: skip  # noqa: E501
    if len(digest_chain) != len(set(digest_chain)):
        raise ValueError("snapshot digest chain contains duplicate mutable identities")
    return {'schema_version': 1, 'kind': 'registry-scan-result', 'fixture_only': fixture is not None, 'snapshot': {'repository': repository, 'tag': upstream_tag, 'index_digest': index_digest, 'members': {key: members[key] for key in sorted(members)}}, 'operations': operations}  # fmt: skip  # noqa: E501


def resolve_builder_root(
    repository: str,
    upstream_tag: str,
    *,
    architecture: str,
) -> dict[str, Any]:
    """Resolve a per-arch builder tag to its index/manifest/config digest chain.

    Unlike resolve_repository_tag (which scans a multi-arch upstream index for
    every required architecture at once), this binds a single architecture
    against a selected project builder tag that may be either a single-arch
    image manifest or a manifest list. resolve_catalog calls this after builder
    capability selection and before ReleasePlan.build consumes the catalog.
    """
    repository = _repository(repository)
    if not isinstance(upstream_tag, str) or OCI_TAG_RE.fullmatch(upstream_tag) is None:
        raise ValueError("builder tag must use canonical OCI tag syntax")
    if not isinstance(architecture, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", architecture) is None:
        raise ValueError("builder architecture must be canonical")
    operations: list[dict[str, Any]] = []
    tagged_reference = f"{repository}:{upstream_tag}"
    crane_binary = resolve_pinned_crane()
    operations.append({'type': 'crane-digest', 'capability': 'read', 'reference': tagged_reference})  # fmt: skip  # noqa: E501
    digest_result = _run_registry_tool(crane_binary, ["digest", tagged_reference])
    top_digest = _digest(digest_result.stdout.strip(), "builder digest")
    top_reference = f"{repository}@{top_digest}"
    operations.append({'type': 'crane-manifest', 'capability': 'read', 'reference': top_reference})  # fmt: skip  # noqa: E501
    manifest_result = _run_registry_tool(crane_binary, ["manifest", top_reference])
    doc = _unique_json(manifest_result.stdout, "builder manifest")
    media_type = doc.get("mediaType")
    if media_type in OCI_INDEX_MEDIA_TYPES:
        descriptors = doc.get("manifests")
        if not isinstance(descriptors, list): raise ValueError('builder index must contain a manifests array')  # noqa: E701,E501
        matches = [descriptor for descriptor in descriptors if isinstance(descriptor, dict) and isinstance(descriptor.get('platform'), dict) and descriptor['platform'].get('os') == 'linux' and descriptor['platform'].get('architecture') == architecture]  # fmt: skip  # noqa: E501
        if not matches: raise ValueError(f"builder {tagged_reference} has no linux/{architecture} member")  # noqa: E701,E501
        if len(matches) != 1: raise ValueError(f"builder {tagged_reference} has multiple linux/{architecture} members")  # noqa: E701,E501
        manifest_digest = _digest(matches[0].get("digest"), "builder member manifest")
        child_reference = f"{repository}@{manifest_digest}"
        operations.append({'type': 'crane-manifest', 'capability': 'read', 'reference': child_reference})  # fmt: skip  # noqa: E501
        child_result = _run_registry_tool(crane_binary, ['manifest', child_reference])  # fmt: skip  # noqa: E501
        child = _unique_json(child_result.stdout, f'builder member manifest {manifest_digest}')  # fmt: skip  # noqa: E501
        config_descriptor = child.get("config")
        if not isinstance(config_descriptor, dict): raise ValueError('builder member manifest requires a config descriptor')  # noqa: E701,E501
        config_digest = _digest(config_descriptor.get("digest"), "builder member config")
        return {'index_digest': top_digest, 'manifest_digest': manifest_digest, 'config_digest': config_digest, 'operations': operations}  # fmt: skip  # noqa: E501
    if media_type in OCI_MANIFEST_MEDIA_TYPES:
        config_descriptor = doc.get("config")
        if not isinstance(config_descriptor, dict): raise ValueError('builder manifest requires a config descriptor')  # noqa: E701,E501
        config_digest = _digest(config_descriptor.get("digest"), "builder config")
        operations.append({'type': 'crane-config', 'capability': 'read', 'reference': top_reference})  # fmt: skip  # noqa: E501
        config_result = _run_registry_tool(crane_binary, ["config", top_reference])
        config = _unique_json(config_result.stdout, "builder config")
        if config.get("os") != "linux" or config.get("architecture") != architecture:
            raise ValueError(f"builder {tagged_reference} config does not match requested linux/{architecture}")  # fmt: skip  # noqa: E501
        return {'index_digest': top_digest, 'manifest_digest': top_digest, 'config_digest': config_digest, 'operations': operations}  # fmt: skip  # noqa: E501
    raise ValueError(f"builder {tagged_reference} returned unrecognized mediaType {media_type!r}")


_CANONICAL_UPSTREAM_TAG = re.compile('^v(?P<version>(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)\\.(?:0|[1-9][0-9]*)(?:rc[1-9][0-9]*)?)(?P<suffix>-[a-z0-9][a-z0-9.-]*)?$')  # fmt: skip  # noqa: E501


def _excluded_by_pattern(tag: str, patterns: list[str]) -> bool:
    normalized = tag.casefold()
    for pattern in patterns:
        if pattern.casefold() in normalized:
            return True
    return False


def _canonical_variant_suffixes(product: dict[str, Any]) -> dict[str, str]:
    suffixes: dict[str, str] = {}
    for variant_value in product["variants"]:
        variant = variant_value["id"]
        suffix = variant_value.get("tag_suffix")
        if (
            not isinstance(suffix, str)
            or re.fullmatch(r"(?:|-[a-z0-9][a-z0-9.-]*)", suffix) is None
        ):
            raise ValueError(f'upstream variant {variant!r} has an invalid declared tag suffix')  # fmt: skip  # noqa: E501
        previous = suffixes.get(suffix)
        if previous is not None:
            raise ValueError(f"duplicate canonical variant suffix {suffix!r} for product {product['id']!r}: {previous!r} and {variant!r}")  # fmt: skip  # noqa: E501
        suffixes[suffix] = variant
    return suffixes


def _variant_by_soc(product: dict[str, Any], soc_version: str) -> str | None:
    """Match an upstream image's SOC_VERSION env value to a declared variant.

    Ascend variants carry a `soc_versions` list (e.g. a2 -> [ascend910b1],
    a3 -> [ascend910_9391]); cuda variants have none and are singletons.
    """
    for variant_value in product["variants"]:
        socs = variant_value.get("soc_versions")
        if isinstance(socs, list) and soc_version in socs:
            return variant_value["id"]
    return None


def _inspect_upstream_variant(
    crane_binary: str, repository: str, index_digest: str, product: dict[str, Any]
) -> tuple[str | None, str | None]:
    """Inspect an upstream image's config to determine its variant.

    `crane config repo@<index_digest>` returns the default-platform config
    blob; SOC_VERSION / ASCEND_* / CUDA env are the same across arches for
    vllm / vllm-ascend images. Returns (variant, error): variant is None when
    the image does not encode a detectable variant (caller falls back).
    """
    reference = f"{repository}@{index_digest}"
    config_blob = _crane(crane_binary, "config", reference)
    try:
        config = json.loads(config_blob)
    except json.JSONDecodeError as error:
        return None, f"crane config returned non-JSON for {reference}: {error}"
    env = ((config.get("config") or {}).get("Env")) or []
    is_ascend = product["runtime_product"] == "vllm-ascend" or any(
        "ASCEND" in entry or "/usr/local/Ascend" in entry for entry in env
    )
    if is_ascend:
        soc_version = next(
            (entry.split("=", 1)[1] for entry in env if entry.startswith("SOC_VERSION=")),
            None,
        )
        if soc_version is None:
            return None, "ascend image has no SOC_VERSION env"
        variant = _variant_by_soc(product, soc_version)
        if variant is None:
            return None, f"SOC_VERSION {soc_version!r} matches no declared variant of {product['id']!r}"
        return variant, None
    # cuda / single-variant product (e.g. vllm -> default)
    if len(product["variants"]) == 1:
        return product["variants"][0]["id"], None
    return None, f"non-ascend image for multi-variant product {product['id']!r}"


def validate_catalog_tag_grammar(catalog: dict[str, Any]) -> None:
    for product in catalog["upstream_products"]:
        _canonical_variant_suffixes(product)


def _parse_product_tag(product: dict[str, Any], tag: str) -> dict[str, str]:
    match = _CANONICAL_UPSTREAM_TAG.fullmatch(tag)
    if match is None: raise ValueError('malformed-tag')  # noqa: E701
    version_text = match.group("version")
    try:
        version = Version(version_text)
    except InvalidVersion as error:  # pragma: no cover - guarded by the regex.
        raise ValueError("malformed-tag") from error
    if (
        version.epoch != 0
        or version.local is not None
        or version.dev is not None
        or version.post is not None
        or str(version) != version_text
    ):
        raise ValueError("malformed-tag")
    channel = "rc" if version.pre is not None and version.pre[0] == "rc" else "stable"
    suffix = match.group("suffix") or ""
    suffixes = _canonical_variant_suffixes(product)
    variant = suffixes.get(suffix)
    if variant is None: raise ValueError('unsupported-variant')  # noqa: E701
    return {'tag': tag, 'version': version_text, 'channel': channel, 'variant': variant}  # fmt: skip


def select_catalog_tags(
    catalog: dict[str, Any], tag_lists: dict[str, list[str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    validate_catalog_tag_grammar(catalog)
    patterns = catalog["compatibility"]["excluded_upstream_patterns"]
    for product in sorted(catalog["upstream_products"], key=lambda item: item["id"]):
        repository = product["repository"]
        tags = tag_lists.get(repository)
        if not isinstance(tags, list): raise ValueError(f'tag list is missing for configured repository {repository}')  # noqa: E701,E501
        for tag in sorted(set(tags)):
            reason: str | None = None
            parsed: dict[str, str] | None = None
            if _excluded_by_pattern(tag, patterns):
                reason = "excluded-pattern"
            else:
                try:
                    parsed = _parse_product_tag(product, tag)
                except ValueError as error:
                    reason = str(error)
            if parsed is not None and parsed['channel'] not in product['channels']: reason = 'unsupported-channel'  # noqa: E701,E501
            if parsed is not None and reason is None:
                if Version(parsed['version']) not in SpecifierSet(product['version_specifier']): reason = 'version-outside-specifier'  # noqa: E701,E501
            if reason is not None:
                exclusions.append({'product_id': product['id'], 'repository': repository, 'tag': tag, 'reason': reason})  # fmt: skip  # noqa: E501
                continue
            assert parsed is not None
            selected.append({'product_id': product['id'], 'repository': repository, **copy.deepcopy(parsed)})  # fmt: skip  # noqa: E501
    selected.sort(key=lambda item: (item['product_id'], Version(item['version']), item['variant'], item['tag']))  # fmt: skip  # noqa: E501
    exclusions.sort(key=lambda item: (item['product_id'], item['repository'], item['tag'], item['reason']))  # fmt: skip  # noqa: E501
    return selected, exclusions


def _artifact_set(tasks: list[dict[str, Any]], prefix: str) -> list[dict[str, str]]:
    return [{'task_id': task['task_id'], 'name': task['artifact_name']} for task in tasks]  # fmt: skip  # noqa: E501


def _wheel_matrix(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {'include': [{'task_id': task['task_id'], 'runner': task['runner']} for task in tasks]}  # fmt: skip  # noqa: E501


def _image_matrix(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {'include': [{'task_id': task['task_id'], 'runner': task['runner'], 'wheel_task_id': task['wheel_task_id']} for task in tasks]}  # fmt: skip  # noqa: E501


def _family_matrix(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {'include': [{'task_id': task['task_id'], 'family_task_id': task['task_id'], 'runner': task['control_runner'], 'control_task_id': task['control_task_id'], 'control_arch': task['control_arch']} for task in tasks]}  # fmt: skip  # noqa: E501


def _loose_tag_version_channel(tag: str) -> tuple[str, str] | None:
    """Extract (version, channel) from a tag matching the upstream grammar.

    Returns None for non-grammar tags. Unlike _parse_product_tag, this does
    NOT derive or reject on the variant suffix (the PR pin path determines
    the variant by inspecting the image, not the tag).
    """
    match = _CANONICAL_UPSTREAM_TAG.fullmatch(tag)
    if match is None: return None  # noqa: E701
    version_text = match.group("version")
    try:
        version = Version(version_text)
    except InvalidVersion:
        return None
    if (version.epoch != 0 or version.local is not None or version.dev is not None
            or version.post is not None or str(version) != version_text):
        return None
    channel = "rc" if version.pre is not None and version.pre[0] == "rc" else "stable"
    return version_text, channel


def _resolve_pinned_upstreams(
    catalog: dict[str, Any],
    pin_upstreams: list[str],
    products_by_repo: dict[str, dict[str, Any]],
    crane_binary: str,
    operations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build resolved_upstreams from user-pinned repo:tag refs (PR path).

    For each ref: resolve tag->digest, inspect the image config to determine
    the variant (SOC_VERSION/accelerator), derive version+channel from the
    tag (grammar if it matches, else a loose version). No scan/select.
    """
    resolved_upstreams: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for ref in pin_upstreams:
        repo, sep, tag = ref.rpartition(":")
        if not sep or not repo or not tag:
            raise ValueError(f"pin_upstream {ref!r} must be 'repository:tag'")
        product = products_by_repo.get(repo)
        if product is None:
            raise ValueError(f"pin_upstream {ref!r}: {repo!r} is not a configured catalog upstream repository")
        try:
            scan = resolve_repository_tag(repo, tag, required_architectures=product['required_cpu_architectures'])  # fmt: skip  # noqa: E501
        except RegistryBlocker as error:
            if not error.code.startswith("missing-linux-"):
                raise
            exclusions.append({'product_id': product['id'], 'repository': repo, 'tag': tag, 'reason': 'required-architecture-missing'})  # fmt: skip  # noqa: E501
            continue
        operations.extend(scan["operations"])
        index_digest = scan['snapshot']['index_digest']
        loose = _loose_tag_version_channel(tag)
        if loose is not None:
            version, channel = loose
        else:
            version = tag[1:] if tag.startswith("v") else tag
            channel = product['channels'][0] if product.get('channels') else "stable"
        inspect_ref = f"{repo}@{index_digest}"
        operations.append({'type': 'crane-config', 'capability': 'read', 'reference': inspect_ref})  # fmt: skip  # noqa: E501
        variant, inspect_err = _inspect_upstream_variant(crane_binary, repo, index_digest, product)  # fmt: skip  # noqa: E501
        if variant is None:
            raise ValueError(f"pin_upstream {ref!r}: inspect could not determine variant: {inspect_err}")
        candidate = {'product_id': product['id'], 'repository': repo, 'tag': tag, 'version': version, 'channel': channel, 'variant': variant}  # fmt: skip  # noqa: E501
        reason = core.candidate_exclusion_reason(
            catalog, product, candidate, relaxed=True
        )
        if reason is not None:
            exclusions.append({'product_id': product['id'], 'repository': repo, 'tag': tag, 'reason': reason})  # fmt: skip  # noqa: E501
            continue
        resolved_upstreams.append({**candidate, 'index_digest': index_digest, 'members': scan['snapshot']['members'], 'target_repository': product['target_repository'], 'target_tag': tag + product['target_tag_suffix']})  # fmt: skip  # noqa: E501
    return resolved_upstreams, exclusions


def resolve_catalog(
    catalog: dict[str, Any],
    *,
    builder_catalog: dict[str, Any],
    source_sha: str,
    lane: str,
    fixture: dict[str, Any] | None = None,
    pin_upstreams: list[str] | None = None,
) -> dict[str, Any]:
    core._exact_keys(catalog, core.RELEASE_KEYS, 'release catalog', optional=core.OPTIONAL_CATALOG_KEYS)  # fmt: skip  # noqa: E501
    core.validate_catalog(catalog)
    selection = builders.select_builders(builder_catalog, catalog)
    catalog = builders.bind_selection(catalog, selection)
    if fixture is not None and lane == "protected-tag":
        raise ValueError("fixture resolution cannot acquire protected-tag authority")
    if pin_upstreams is not None and fixture is not None:
        raise ValueError("pin_upstreams (PR path) cannot be combined with a fixture")
    if pin_upstreams is not None and lane == "protected-tag":
        raise ValueError("pin_upstreams (PR path) is only valid for the feature-candidate lane")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source_sha must be an exact lowercase 40-hex commit")
    limits = catalog.get("scan_limits")
    if not isinstance(limits, dict) or set(limits) != {
        "max_tags_per_repository",
        "max_selected_upstreams",
    }:
        raise ValueError("catalog requires exact scan_limits")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in limits.values()
    ):
        raise ValueError("catalog scan limits must be positive integers")
    configured_repositories = {product['repository'] for product in catalog['upstream_products']}  # fmt: skip  # noqa: E501
    repositories_fixture: dict[str, Any] | None = None
    if fixture is not None:
        if not isinstance(fixture, dict) or set(fixture) != {
            "kind",
            "schema_version",
            "repositories",
        }:
            raise ValueError("registry discovery fixture fields are noncanonical")
        if (
            fixture["kind"] != "registry-discovery-fixture"
            or fixture["schema_version"] != 1
            or not isinstance(fixture["repositories"], dict)
            or set(fixture["repositories"]) != configured_repositories
        ):
            raise ValueError("registry discovery fixture is incomplete")
        repositories_fixture = fixture["repositories"]

    tag_lists: dict[str, list[str]] = {}
    operations: list[dict[str, Any]] = []
    for profile in catalog["wheel_profiles"]:
        for architecture, builder in profile["builders"].items():
            unresolved_root = builder["root"]
            repository = unresolved_root["repository"]
            tag = unresolved_root["tag"]
            resolved = resolve_builder_root(repository, tag, architecture=architecture)
            builder["root"] = {
                "repository": repository,
                "tag": tag,
                "index_digest": resolved["index_digest"],
                "manifest_digest": resolved["manifest_digest"],
                "config_digest": resolved["config_digest"],
            }
            operations.extend(resolved["operations"])
    products = {item["id"]: item for item in catalog["upstream_products"]}
    products_by_repo = {product["repository"]: product for product in catalog["upstream_products"]}
    crane_binary = resolve_pinned_crane() if fixture is None else None
    resolved_upstreams: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    scanned_tags = 0
    if pin_upstreams is not None:
        # PR pin path: build resolved_upstreams from user repo:tag refs; skip scan+select.
        resolved_upstreams, exclusions = _resolve_pinned_upstreams(
            catalog, pin_upstreams, products_by_repo, crane_binary, operations
        )
        scanned_tags = len(pin_upstreams)
    else:
        for repository in sorted(configured_repositories):
            repository_fixture = repositories_fixture[repository] if repositories_fixture is not None else None  # fmt: skip  # noqa: E501
            tag_result = enumerate_repository_tags(repository, fixture=repository_fixture, max_tags=limits['max_tags_per_repository'])  # fmt: skip  # noqa: E501
            tag_lists[repository] = tag_result["tags"]
            operations.extend(tag_result["operations"])
        candidates, exclusions = select_catalog_tags(catalog, tag_lists)
        admissions = [
            (
                candidate,
                core.candidate_exclusion_reason(
                    catalog,
                    products[candidate["product_id"]],
                    candidate,
                ),
            )
            for candidate in candidates
        ]
        grouped: dict[
            tuple[str, str], list[tuple[dict[str, str], str | None]]
        ] = {}
        for candidate, reason in admissions:
            grouped.setdefault(
                (candidate["product_id"], candidate["variant"]), []
            ).append((candidate, reason))
        selected_fixture_tags: dict[str, set[str]] = {repository: set() for repository in configured_repositories}  # fmt: skip  # noqa: E501
        for group_key in sorted(grouped):
            group = sorted(
                grouped[group_key],
                key=lambda entry: (Version(entry[0]["version"]), entry[0]["tag"]),
                reverse=True,
            )
            for index, (item, reason) in enumerate(group):
                product = products[item["product_id"]]
                if reason is not None:
                    exclusions.append({'product_id': item['product_id'], 'repository': item['repository'], 'tag': item['tag'], 'reason': reason})  # fmt: skip  # noqa: E501
                    continue
                snapshot_fixture = None
                if repositories_fixture is not None:
                    snapshots = repositories_fixture[item["repository"]]["snapshots"]
                    if not isinstance(snapshots, dict) or item["tag"] not in snapshots:
                        raise ValueError(f"registry fixture is missing selected snapshot {item['tag']}")  # fmt: skip  # noqa: E501
                    snapshot_fixture = snapshots[item["tag"]]
                    selected_fixture_tags[item["repository"]].add(item["tag"])
                try:
                    scan = resolve_repository_tag(item['repository'], item['tag'], required_architectures=product['required_cpu_architectures'], fixture=snapshot_fixture)  # fmt: skip  # noqa: E501
                except RegistryBlocker as error:
                    if not error.code.startswith("missing-linux-"):
                        raise
                    exclusions.append({'product_id': item['product_id'], 'repository': item['repository'], 'tag': item['tag'], 'reason': 'required-architecture-missing'})  # fmt: skip  # noqa: E501
                    continue
                operations.extend(scan["operations"])
                variant = item["variant"]
                # Live image metadata is authoritative over the tag suffix. A
                # changed variant must pass the same static admission checks.
                if fixture is None:
                    inspect_ref = f"{item['repository']}@{scan['snapshot']['index_digest']}"
                    operations.append({'type': 'crane-config', 'capability': 'read', 'reference': inspect_ref})  # fmt: skip  # noqa: E501
                    inspected, _inspect_err = _inspect_upstream_variant(crane_binary, item['repository'], scan['snapshot']['index_digest'], product)  # fmt: skip  # noqa: E501
                    if inspected is not None:
                        if inspected != item["variant"]:
                            exclusions.append({'product_id': item['product_id'], 'repository': item['repository'], 'tag': item['tag'], 'reason': 'inspected-variant-mismatch'})  # fmt: skip  # noqa: E501
                            continue
                        variant = inspected
                    inspected_candidate = {**item, "variant": variant}
                    inspected_reason = core.candidate_exclusion_reason(
                        catalog, product, inspected_candidate
                    )
                    if inspected_reason is not None:
                        exclusions.append({'product_id': item['product_id'], 'repository': item['repository'], 'tag': item['tag'], 'reason': inspected_reason})  # fmt: skip  # noqa: E501
                        continue
                resolved_upstreams.append({**copy.deepcopy(item), 'variant': variant, 'index_digest': scan['snapshot']['index_digest'], 'members': scan['snapshot']['members'], 'target_repository': product['target_repository'], 'target_tag': item['tag'] + product['target_tag_suffix']})  # fmt: skip  # noqa: E501
                for superseded, superseded_reason in group[index + 1 :]:
                    exclusions.append({'product_id': superseded['product_id'], 'repository': superseded['repository'], 'tag': superseded['tag'], 'reason': superseded_reason or 'superseded-compatible-version'})  # fmt: skip  # noqa: E501
                break
        if repositories_fixture is not None:
            for repository in sorted(configured_repositories):
                snapshots = repositories_fixture[repository]["snapshots"]
                if set(snapshots) != selected_fixture_tags[repository]:
                    raise ValueError(f'registry fixture snapshots are not the exact selected set for {repository}')  # fmt: skip  # noqa: E501
        scanned_tags = sum((len(tags) for tags in tag_lists.values()))

    if len(resolved_upstreams) > limits["max_selected_upstreams"]:
        raise ValueError(f"registry selection limit max_selected_upstreams={limits['max_selected_upstreams']} exceeded by exact set of {len(resolved_upstreams)}")  # fmt: skip  # noqa: E501
    exclusions.sort(
        key=lambda item: (
            item["product_id"],
            item["repository"],
            item["tag"],
            item["reason"],
        )
    )

    plan = core.ReleasePlan.build(catalog, resolved_upstreams, lane=lane, relaxed=(pin_upstreams is not None))
    wheel_tasks = plan.wheel_tasks
    image_tasks = plan.image_tasks
    family_tasks = plan.family_tasks
    _src = catalog["source"]
    source = {'repository': _src['repository'], 'staging_repository': _src['staging_repository'], 'default_branch': _src['default_branch'], 'release_tag': _src['release_tag'], 'ucm_version': catalog['ucm_version'], 'commit': source_sha}  # fmt: skip  # noqa: E501
    scan_evidence = {'resolved_upstreams': resolved_upstreams, 'exclusions': exclusions, 'operations': operations}  # fmt: skip  # noqa: E501
    result: dict[str, Any] = {'kind': 'ucm-resolved-build-plan', 'schema_version': 1, 'fixture_only': fixture is not None, 'lane': lane, 'source': source, 'chart': copy.deepcopy(catalog['chart']), 'publish': core.compute_publish_plan(catalog), 'config_sha256': core.sha256_value(catalog), 'source_sha256': core.sha256_value(source), 'scan_sha256': core.sha256_value(scan_evidence), 'resolved_upstreams': resolved_upstreams, 'wheel_tasks': wheel_tasks, 'image_tasks': image_tasks, 'family_tasks': family_tasks, 'github_wheel_matrix': _wheel_matrix(wheel_tasks), 'github_image_matrix': _image_matrix(image_tasks), 'github_family_matrix': _family_matrix(family_tasks), 'expected_artifacts': {'resolved_plan': f'ucm-resolved-plan-{source_sha}', 'wheels': _artifact_set(wheel_tasks, 'wheel'), 'images': _artifact_set(image_tasks, 'image'), 'families': _artifact_set(family_tasks, 'family')}, 'exclusions': exclusions, 'operations': operations, 'counts': {'scanned_tags': scanned_tags, 'selected_upstreams': len(resolved_upstreams), 'excluded_tags': len(exclusions), 'wheel_tasks': len(wheel_tasks), 'image_tasks': len(image_tasks), 'family_tasks': len(family_tasks)}}  # fmt: skip  # noqa: E501
    result["resolved_plan_sha256"] = core.sha256_value(result)
    return result


_SOURCE_STR_FIELDS = ("default_branch", "ucm_version")
_CHART_STR_FIELDS = ("source", "name", "version", "app_version")
_FAMILY_PROJECT_KEYS = ("runner", "cpu_arch", "platform", "builder", "builder_sha256")


def validate_resolved_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict): raise ValueError('resolved plan must be an object')  # noqa: E701,E501
    if set(plan) != _RESOLVED_PLAN_FIELDS:
        raise ValueError(f'resolved plan top-level fields mismatch: missing={sorted(_RESOLVED_PLAN_FIELDS - set(plan))}, extra={sorted(set(plan) - _RESOLVED_PLAN_FIELDS)}')  # fmt: skip  # noqa: E501
    if plan.get("kind") != "ucm-resolved-build-plan" or plan.get("schema_version") != 1:
        raise ValueError("resolved plan identity must be schema version 1")
    hash_payload = {key: value for key, value in plan.items() if key != "resolved_plan_sha256"}
    if plan.get("resolved_plan_sha256") != core.sha256_value(hash_payload):
        raise ValueError("resolved plan hash mismatch")
    if not isinstance(plan['fixture_only'], bool): raise ValueError('resolved plan fixture_only must be boolean')  # noqa: E701,E501
    if plan['lane'] not in {'feature-candidate', 'protected-tag'}: raise ValueError('resolved plan lane is invalid')  # noqa: E701,E501
    if plan["fixture_only"] and plan["lane"] == "protected-tag":
        raise ValueError("fixture plan cannot carry protected-tag authority")

    source = plan["source"]
    if not isinstance(source, dict) or set(source) != _PLAN_SOURCE_KEYS:
        raise ValueError("resolved plan source is malformed")
    for field in _SOURCE_STR_FIELDS:
        if not isinstance(source[field], str) or not source[field]: raise ValueError('resolved plan source is malformed')  # noqa: E701,E501
    if (
        not isinstance(source["repository"], str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repository"])
        is None
        or not isinstance(source["staging_repository"], str)
        or REPOSITORY_RE.fullmatch(source["staging_repository"]) is None
        or not isinstance(source["release_tag"], str)
        or re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:rc[0-9]+)?(?:\.dev[0-9]+)?(?:\+[a-z0-9.]+)?", source["release_tag"])
        is None
        or not isinstance(source["commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", source["commit"]) is None
        or source["release_tag"] != f"v{source['ucm_version']}"
    ):
        raise ValueError("resolved plan source is malformed")

    chart = plan["chart"]
    if (
        not isinstance(chart, dict)
        or set(chart) - {"provenance"} != _PLAN_CHART_KEYS
        or any(
            not isinstance(chart.get(f), str) or not chart[f] for f in _CHART_STR_FIELDS
        )
        or not isinstance(chart.get("validation_cases"), list)
        or not chart["validation_cases"]
    ):
        raise ValueError("resolved plan Chart authority is malformed")

    if plan["publish"] != core.compute_publish_plan({"publish": plan["publish"]}):
        raise ValueError("resolved plan publish authority is malformed")

    core.validate_resolved_upstreams(plan["resolved_upstreams"])
    snapshots = plan["resolved_upstreams"]
    snapshot_hashes = [core.sha256_value(snapshot) for snapshot in snapshots]

    exclusions = plan["exclusions"]
    exclusion_fields = {"product_id", "repository", "tag", "reason"}
    if not isinstance(exclusions, list) or any(
        not isinstance(item, dict)
        or set(item) != exclusion_fields
        or any(not isinstance(item[key], str) or not item[key] for key in item)
        for item in exclusions
    ):
        raise ValueError("resolved plan exclusions are malformed")
    if exclusions != sorted(
        exclusions,
        key=lambda item: (
            item["product_id"],
            item["repository"],
            item["tag"],
            item["reason"],
        ),
    ):
        raise ValueError("resolved plan exclusions are not canonical")

    operations = plan["operations"]
    if not isinstance(operations, list) or not operations:
        raise ValueError("resolved plan operations must be a non-empty array")
    for operation in operations:
        if (
            not isinstance(operation, dict)
            or operation.get("type") not in _PLAN_OPERATION_TYPES
            or operation.get("capability") != "read"
            or not isinstance(operation.get("reference"), str)
            or not operation["reference"]
            or set(operation)
            != (
                {"type", "capability", "reference", "page"}
                if operation.get("type") == "fixture-tag-page-read"
                else {"type", "capability", "reference"}
            )
            or "page" in operation
            and (
                not isinstance(operation["page"], int)
                or isinstance(operation["page"], bool)
                or operation["page"] < 1
            )
        ):
            raise ValueError("resolved plan operation is malformed")
    fixture_operations = [operation['type'].startswith('fixture-') for operation in operations]  # fmt: skip  # noqa: E501
    if (plan["fixture_only"] and not all(fixture_operations)) or (
        not plan["fixture_only"] and any(fixture_operations)
    ):
        raise ValueError("resolved plan fixture authority differs from operations")

    task_ids: set[str] = set()
    tasks_by_kind: dict[str, list[dict[str, Any]]] = {}
    expected_write_authority = [] if plan['lane'] == 'feature-candidate' else ['github-prerelease', 'ghcr-final-index', 'ghcr-private-staging']  # fmt: skip  # noqa: E501
    for task_kind in ("wheel", "image", "family"):
        tasks = plan.get(f"{task_kind}_tasks")
        if not isinstance(tasks, list): raise ValueError(f'resolved plan {task_kind} tasks must be an array')  # noqa: E701,E501
        tasks_by_kind[task_kind] = tasks
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("task_id"), str):
                raise ValueError(f"resolved plan {task_kind} task is malformed")
            if set(task) != RESOLVED_TASK_FIELDS[task_kind]:
                raise ValueError(f'resolved plan {task_kind} task fields mismatch: missing={sorted(RESOLVED_TASK_FIELDS[task_kind] - set(task))}, extra={sorted(set(task) - RESOLVED_TASK_FIELDS[task_kind])}')  # fmt: skip  # noqa: E501
            if re.fullmatch(f"{task_kind}-[0-9a-f]{{64}}", task["task_id"]) is None:
                raise ValueError(f"resolved plan {task_kind} task ID is malformed")
            if task['task_id'] in task_ids: raise ValueError('resolved plan task IDs must be globally unique')  # noqa: E701,E501
            task_ids.add(task["task_id"])
            task_payload = {key: value for key, value in task.items() if key != 'task_sha256'}  # fmt: skip  # noqa: E501
            if task.get("task_sha256") != core.sha256_value(task_payload):
                raise ValueError(f"resolved plan {task_kind} task hash mismatch")
            if task.get("write_authority") != expected_write_authority:
                raise ValueError(f'resolved plan {task_kind} task lane authority mismatch')  # fmt: skip  # noqa: E501
            if task_kind in {"wheel", "image"}:
                cpu_authority = core.cpu_toolchain_authority(task.get('cpu_arch'), location=f'resolved plan {task_kind} task cpu_arch')  # fmt: skip  # noqa: E501
                if task.get("platform") != cpu_authority.oci_platform:
                    raise ValueError(f'resolved plan {task_kind} platform differs from its CPU/tool architecture')  # fmt: skip  # noqa: E501
            else:
                architectures = task.get("cpu_arch")
                platforms = task.get("platform")
                if (
                    not isinstance(architectures, list)
                    or not isinstance(platforms, list)
                    or len(architectures) != len(platforms)
                ):
                    raise ValueError('resolved plan family CPU/tool architecture projection is malformed')  # fmt: skip  # noqa: E501
                for index, architecture in enumerate(architectures):
                    cpu_authority = core.cpu_toolchain_authority(architecture, location=f'resolved plan family task cpu_arch[{index}]')  # fmt: skip  # noqa: E501
                    if platforms[index] != cpu_authority.oci_platform:
                        raise ValueError('resolved plan family platform differs from its CPU/tool architecture')  # fmt: skip  # noqa: E501
                core.cpu_toolchain_authority(task.get('control_arch'), location='resolved plan family control_arch')  # fmt: skip  # noqa: E501

    wheel_tasks = tasks_by_kind["wheel"]
    image_tasks = tasks_by_kind["image"]
    family_tasks = tasks_by_kind["family"]
    wheels_by_id = {task["task_id"]: task for task in wheel_tasks}
    snapshots_by_hash = {core.sha256_value(snapshot): snapshot for snapshot in snapshots}  # fmt: skip  # noqa: E501
    for image in image_tasks:
        wheel = wheels_by_id.get(image.get("wheel_task_id"))
        if wheel is None or image.get("family_task_id") not in {
            task["task_id"] for task in family_tasks
        }:
            raise ValueError("resolved plan image references an unknown task")
        if any(
            image[field] != wheel[field] for field in _LINKED_WHEEL_FIELDS
        ) or image.get("wheel_artifact_name") != wheel.get("artifact_name"):
            raise ValueError("resolved plan image/wheel linkage is inconsistent")

    for family in family_tasks:
        linked = [img for img in image_tasks if img.get('family_task_id') == family['task_id']]  # fmt: skip  # noqa: E501
        snapshot = snapshots_by_hash.get(family.get("snapshot_sha256"))
        if not linked or snapshot is None: raise ValueError('resolved plan family snapshot linkage is inconsistent')  # noqa: E701,E501
        expected_runtime = {k: snapshot[k] for k in _RUNTIME_KEYS}
        expected_family = {'control_task_id': linked[0]['task_id'], 'control_arch': linked[0]['cpu_arch'], 'control_runner': linked[0]['runner'], **{k: [img[k] for img in linked] for k in _FAMILY_PROJECT_KEYS}, 'member_set_sha256': core.sha256_value([img['task_sha256'] for img in linked]), 'image_task_ids': [img['task_id'] for img in linked], 'wheel_task_ids': {img['cpu_arch']: img['wheel_task_id'] for img in linked}, 'product_id': snapshot['product_id'], 'runtime': expected_runtime, 'runtime_sha256': core.sha256_value(expected_runtime), 'target_repository': snapshot['target_repository'], 'target_tag': snapshot['target_tag']}  # fmt: skip  # noqa: E501
        if any(family.get(field) != value for field, value in expected_family.items()):
            raise ValueError("resolved plan family/image projection is inconsistent")
        for image in linked:
            member = snapshot["members"].get(image["cpu_arch"])
            expected_img_runtime = {'product_id': snapshot['product_id'], **expected_runtime, **(member or {})}  # fmt: skip  # noqa: E501
            if (
                image.get("runtime") != expected_img_runtime
                or image.get("runtime_sha256")
                != core.sha256_value(expected_img_runtime)
                or image.get("target_repository") != family["target_repository"]
                or image.get("target_tag") != family["target_tag"]
            ):
                raise ValueError("resolved plan image/snapshot linkage is inconsistent")
    if {family["snapshot_sha256"] for family in family_tasks} != set(snapshot_hashes):
        raise ValueError("resolved plan snapshot set differs from family tasks")

    expected_counts = {'scanned_tags': len(snapshots) + len(exclusions), 'selected_upstreams': len(snapshots), 'excluded_tags': len(exclusions), 'wheel_tasks': len(wheel_tasks), 'image_tasks': len(image_tasks), 'family_tasks': len(family_tasks)}  # fmt: skip  # noqa: E501
    if plan['counts'] != expected_counts: raise ValueError('resolved plan counts mismatch')  # noqa: E701,E501
    for key, fn, tasks in (
        ("github_wheel_matrix", _wheel_matrix, wheel_tasks),
        ("github_image_matrix", _image_matrix, image_tasks),
        ("github_family_matrix", _family_matrix, family_tasks),
    ):
        if plan[key] != fn(tasks): raise ValueError(f'resolved plan {key} mismatch')  # noqa: E701,E501
    expected_artifacts = {'resolved_plan': f"ucm-resolved-plan-{source['commit']}", 'wheels': _artifact_set(wheel_tasks, 'wheel'), 'images': _artifact_set(image_tasks, 'image'), 'families': _artifact_set(family_tasks, 'family')}  # fmt: skip  # noqa: E501
    if plan['expected_artifacts'] != expected_artifacts: raise ValueError('resolved plan artifact set mismatch')  # noqa: E701,E501


def validate_main_full_loop_plan(
    plan: dict[str, Any], catalog: dict[str, Any]
) -> dict[str, int]:
    wheel_tasks = plan.get("wheel_tasks")
    image_tasks = plan.get("image_tasks")
    family_tasks = plan.get("family_tasks")
    if (
        not isinstance(wheel_tasks, list)
        or not isinstance(image_tasks, list)
        or not isinstance(family_tasks, list)
        or not all(
            isinstance(task, dict)
            for tasks in (wheel_tasks, image_tasks, family_tasks)
            for task in tasks
        )
    ):
        raise ValueError("main full loop task lists are malformed")
    topology = core.release_topology(catalog)
    wheel_coordinates = [
        {"profile_id": task.get("profile_id"), "cpu_arch": task.get("cpu_arch")}
        for task in wheel_tasks
    ]
    family_coordinates = [
        {
            "product_id": task.get("product_id"),
            "variant": task.get("runtime", {}).get("variant")
            if isinstance(task.get("runtime"), dict)
            else None,
        }
        for task in family_tasks
    ]
    image_coordinates = [
        {
            "product_id": task.get("runtime", {}).get("product_id")
            if isinstance(task.get("runtime"), dict)
            else None,
            "variant": task.get("runtime", {}).get("variant")
            if isinstance(task.get("runtime"), dict)
            else None,
            "cpu_arch": task.get("cpu_arch"),
        }
        for task in image_tasks
    ]
    if any(
        sorted(core.canonical_bytes(item) for item in actual)
        != sorted(core.canonical_bytes(item) for item in topology[kind])
        for kind, actual in (
            ("wheels", wheel_coordinates),
            ("families", family_coordinates),
            ("images", image_coordinates),
        )
    ):
        raise ValueError("main full loop task coordinates differ from catalog topology")
    expected_architectures = {
        (product["id"], variant["id"]): sorted(
            product["required_cpu_architectures"]
        )
        for product in catalog["upstream_products"]
        for variant in product["variants"]
    }
    for family in family_tasks:
        members = [
            task
            for task in image_tasks
            if task.get("family_task_id") == family.get("task_id")
        ]
        runtime = family.get("runtime")
        coordinate = (
            family.get("product_id"),
            runtime.get("variant") if isinstance(runtime, dict) else None,
        )
        declared = family.get("cpu_arch")
        if (
            not isinstance(declared, list)
            or sorted(declared) != expected_architectures[coordinate]
            or sorted(task.get("cpu_arch") for task in members) != sorted(declared)
        ):
            raise ValueError(
                "main full loop family/image linkage differs from catalog-declared architectures"
            )
    validate_resolved_plan(plan)
    return {
        "wheel_tasks": len(wheel_tasks),
        "image_tasks": len(image_tasks),
        "family_tasks": len(family_tasks),
        "profile_architectures": len(topology["wheels"]),
    }


def select_task(
    plan: dict[str, Any],
    *,
    task_kind: str,
    task_id: str,
    expected_plan_sha256: str | None = None,
) -> dict[str, Any]:
    validate_resolved_plan(plan)
    if plan["fixture_only"] is True and plan["lane"] == "protected-tag":
        raise ValueError("fixture plan cannot authorize protected task selection")
    if task_kind not in {'wheel', 'image', 'family'}: raise ValueError('task_kind must be wheel, image, or family')  # noqa: E701,E501
    if not isinstance(task_id, str) or not task_id: raise ValueError('task_id must be a non-empty opaque identifier')  # noqa: E701,E501
    matches = [task for task in plan[f'{task_kind}_tasks'] if task['task_id'] == task_id]  # fmt: skip  # noqa: E501
    if len(matches) != 1: raise ValueError(f'resolved plan {task_kind} task {task_id!r} does not resolve exactly once')  # noqa: E701,E501
    return copy.deepcopy(matches[0])


def _is_retryable_secondary_limit(arguments: list[str], detail: str) -> bool:
    if not arguments or arguments[0] not in IDEMPOTENT_REGISTRY_READ_OPERATIONS:
        return False
    normalized = " ".join(detail.casefold().split())
    return any(marker in normalized for marker in SECONDARY_RATE_LIMIT_MARKERS)


def _run_registry_tool(
    binary: str,
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
    missing_ok: bool = False,
) -> subprocess.CompletedProcess[str]:
    executable = _crane_binary(binary)
    for retry_index in range(len(SECONDARY_RATE_LIMIT_BACKOFF_SECONDS) + 1):
        try:
            result = subprocess.run([executable, *arguments], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment if environment is not None else _minimal_registry_environment(), check=False)  # fmt: skip  # noqa: E501
        except OSError as error:
            raise ValueError(f'failed to execute pinned registry tool: {error}') from error  # fmt: skip  # noqa: E501
        if result.returncode == 0:
            return result
        retryable = _is_retryable_secondary_limit(arguments, result.stderr + '\n' + result.stdout)  # fmt: skip  # noqa: E501
        if retryable and retry_index < len(SECONDARY_RATE_LIMIT_BACKOFF_SECONDS):
            time.sleep(SECONDARY_RATE_LIMIT_BACKOFF_SECONDS[retry_index])
            continue
        detail = result.stderr.strip() or f"exit {result.returncode}"
        if retryable or not missing_ok: raise ValueError(f"registry tool {' '.join(arguments[:1])} failed: {detail}")  # noqa: E701,E501
        return result
    raise AssertionError("registry read retry loop exhausted without a result")
# fmt: on
