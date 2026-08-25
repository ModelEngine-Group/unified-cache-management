"""Pure authority for ``/ucm-build image`` runtime inspection and projection.

The workflow owns transport (``crane``, Docker fallback, artifacts, and PR
comments). This module owns config-fact extraction and the data contracts
between those operations so the arbitrary-runtime-tag path can be tested
without a registry or container runtime.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import core

__all__ = [
    "aggregate_runtime_probes",
    "build_receipt",
    "inspect_runtime_references",
    "match_runtime_builders",
    "parse_runtime_reference",
    "project_pr_publication",
    "project_pr_tag",
    "render_receipt_markdown",
    "sanitize_oci_tag_component",
]

SUPPORTED_ARCHITECTURES = ("amd64", "arm64")
RUNTIME_PROBE_SCHEMA_VERSION = 2
OCI_INDEX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)
OCI_MANIFEST_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    }
)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PYTHON_VERSION = re.compile(r"(?<![0-9])(\d+)\.(\d+)(?:\.\d+)?")
_NUMERIC_VERSION = re.compile(r"(?<![0-9])(\d+)\.(\d+)(?:\.(\d+))?")
_MANYLINUX = re.compile(r"manylinux_?(\d+)_(\d+)")
_ASCEND_BACKEND_BY_SOC = {
    "ascend910b1": "cann-a2",
    "ascend910_9391": "cann-a3",
    "ascend950dt_9582": "cann-a5",
}
_UNREPORTED = "unreported"

JsonLoader = Callable[[str], object]


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a mapping")
    return dict(value)


def _string(mapping: Mapping[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value.strip()


def _json_mapping(value: object, context: str) -> dict[str, Any]:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{context}: JSON is not UTF-8") from error
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: malformed JSON") from error
    return _mapping(value, context)


def parse_runtime_reference(reference: str) -> tuple[str, str]:
    """Parse one opaque OCI ``repository:tag`` without interpreting the tag."""

    if not isinstance(reference, str):
        raise ValueError("runtime reference must be a string")
    repository, separator, tag = reference.strip().rpartition(":")
    if not separator or not repository or not tag or "@" in repository:
        raise ValueError(f"runtime reference {reference!r} must be repository:tag")
    if core.OCI_REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise ValueError(f"runtime reference {reference!r} has an invalid repository")
    if core.OCI_TAG_PATTERN.fullmatch(tag) is None:
        raise ValueError(f"runtime reference {reference!r} has an invalid OCI tag")
    return repository, tag


def _policy_by_repository(
    products: Sequence[Mapping[str, object]], runners: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(products, Sequence) or isinstance(products, (str, bytes)):
        raise ValueError("runtime products policy must be a sequence")
    runner_values = _mapping(runners, "runtime runners policy")
    for architecture, runner in runner_values.items():
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"runtime runners policy has unsupported {architecture!r}")
        if not isinstance(runner, str) or not runner:
            raise ValueError(f"runtime runner for {architecture} must be non-empty")

    result: dict[str, dict[str, Any]] = {}
    product_ids: set[str] = set()
    for index, raw_product in enumerate(products):
        context = f"runtime products policy[{index}]"
        product = _mapping(raw_product, context)
        product_id = _string(product, "id", context)
        repository = _string(product, "runtime_repository", context)
        target_repository = _string(product, "target_repository", context)
        raw_accelerator = product.get("accelerator")
        accelerator = (
            str(raw_accelerator).lower()
            if isinstance(raw_accelerator, str) and raw_accelerator
            else "cuda" if product_id == "vllm" else "ascend"
        )
        if product_id in product_ids:
            raise ValueError(f"duplicate runtime product id {product_id!r}")
        product_ids.add(product_id)
        if repository in result:
            raise ValueError(f"duplicate runtime repository {repository!r}")
        if core.OCI_REPOSITORY_PATTERN.fullmatch(repository) is None:
            raise ValueError(f"{context}: invalid runtime_repository")
        if core.OCI_REPOSITORY_PATTERN.fullmatch(target_repository) is None:
            raise ValueError(f"{context}: invalid target_repository")
        if accelerator not in {"cuda", "ascend"}:
            raise ValueError(f"{context}: accelerator must be cuda or ascend")

        backend = product.get("backend", "cuda" if product_id == "vllm" else "")
        backend_by_soc = product.get(
            "backend_by_soc",
            _ASCEND_BACKEND_BY_SOC if product_id == "vllm-ascend" else {},
        )
        if accelerator == "cuda":
            if not isinstance(backend, str) or not backend:
                raise ValueError(f"{context}: CUDA product requires backend")
            if backend_by_soc not in ({}, None):
                raise ValueError(
                    f"{context}: CUDA product cannot define backend_by_soc"
                )
            normalized_backend_by_soc: dict[str, str] = {}
        else:
            mapping = _mapping(backend_by_soc, f"{context}.backend_by_soc")
            if not mapping:
                raise ValueError(f"{context}: Ascend product requires backend_by_soc")
            normalized_backend_by_soc = {}
            for soc, soc_backend in mapping.items():
                if not isinstance(soc, str) or not soc.strip():
                    raise ValueError(f"{context}: backend_by_soc key is invalid")
                if not isinstance(soc_backend, str) or not soc_backend.strip():
                    raise ValueError(f"{context}: backend_by_soc value is invalid")
                normalized_backend_by_soc[soc.strip().casefold()] = soc_backend.strip()
            backend = ""

        result[repository] = {
            "product_id": product_id,
            "runtime_repository": repository,
            "target_repository": target_repository,
            "accelerator": accelerator,
            "backend": str(backend),
            "backend_by_soc": normalized_backend_by_soc,
        }
    if not result:
        raise ValueError("runtime products policy must not be empty")
    return result


def _config_platform(config: Mapping[str, object], context: str) -> tuple[str, str]:
    operating_system = _string(config, "os", context).lower()
    architecture = _string(config, "architecture", context).lower()
    return operating_system, architecture


def _load_platform_config(
    reference: str,
    *,
    config_loader: JsonLoader,
    expected_platform: tuple[str, str] | None,
) -> tuple[str, str, dict[str, Any]]:
    context = f"runtime config {reference}"
    config = _json_mapping(config_loader(reference), context)
    operating_system, architecture = _config_platform(config, context)
    if (
        expected_platform is not None
        and (
            operating_system,
            architecture,
        )
        != expected_platform
    ):
        raise ValueError(
            f"{context}: config platform {operating_system}/{architecture} differs "
            f"from manifest {expected_platform[0]}/{expected_platform[1]}"
        )
    return operating_system, architecture, config


def _config_env(config: Mapping[str, object], context: str) -> dict[str, str]:
    raw_config = config.get("config")
    if raw_config is None:
        return {}
    values = _mapping(raw_config, f"{context}.config").get("Env", [])
    if values is None:
        return {}
    if not isinstance(values, list) or not all(
        isinstance(item, str) for item in values
    ):
        raise ValueError(f"{context}.config.Env must be a string list")
    result: dict[str, str] = {}
    for entry in values:
        key, separator, value = entry.partition("=")
        if separator and key:
            result[key] = value
    return result


def _history_assignments(config: Mapping[str, object], key: str) -> list[str]:
    history = config.get("history", [])
    if not isinstance(history, list):
        return []
    pattern = re.compile(rf"(?:^|[\s|]){re.escape(key)}=([^\s]+)")
    values: list[str] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        command = item.get("created_by")
        if not isinstance(command, str):
            continue
        values.extend(
            match.group(1).strip("'\"") for match in pattern.finditer(command)
        )
    return values


def _consistent_version(
    values: Sequence[str], normalizer: Callable[[object], str], context: str
) -> tuple[str | None, str | None]:
    normalized: set[str] = set()
    for value in values:
        try:
            normalized.add(normalizer(value))
        except ValueError:
            continue
    if not normalized:
        return None, None
    if len(normalized) != 1:
        return None, f"{context} metadata conflicts: {sorted(normalized)}"
    return normalized.pop(), None


def _python_from_config(
    config: Mapping[str, object], env: Mapping[str, str], context: str
) -> tuple[str | None, str | None, str | None]:
    sources: list[tuple[str, str]] = []
    if env.get("PYTHON_VERSION"):
        sources.append(("env:PYTHON_VERSION", env["PYTHON_VERSION"]))
    for key in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH"):
        value = env.get(key, "")
        for match in re.finditer(r"/(?:python|Python)(\d+\.\d+(?:\.\d+)?)", value):
            sources.append((f"env:{key}", match.group(1)))
        for match in re.finditer(r"/cp(\d)(\d{2})-cp\d{3}(?:/|$)", value):
            sources.append((f"env:{key}", f"{match.group(1)}.{int(match.group(2))}"))
    sources.extend(
        ("history:PYTHON_VERSION", value)
        for value in _history_assignments(config, "PYTHON_VERSION")
    )
    version, conflict = _consistent_version(
        [value for _, value in sources],
        lambda value: _python_coordinates(value, context)[0],
        context,
    )
    source = ",".join(sorted({name for name, _ in sources})) if version else None
    return version, source, conflict


def _runtime_from_config(
    config: Mapping[str, object],
    env: Mapping[str, str],
    accelerator: str,
    context: str,
) -> tuple[str | None, str | None, str | None]:
    if accelerator == "cuda":
        sources = []
        if env.get("CUDA_VERSION"):
            sources.append(("env:CUDA_VERSION", env["CUDA_VERSION"]))
        sources.extend(
            ("history:CUDA_VERSION", value)
            for value in _history_assignments(config, "CUDA_VERSION")
        )
    else:
        sources = []
        for name in ("CANN_VERSION", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
            if env.get(name):
                sources.append((f"env:{name}", env[name]))
        for name in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH"):
            value = env.get(name, "")
            for match in re.finditer(r"/cann[-_/]?(\d+\.\d+(?:\.\d+)?)", value):
                sources.append((f"env:{name}", match.group(1)))
        sources.extend(
            ("history:CANN_VERSION", value)
            for value in _history_assignments(config, "CANN_VERSION")
        )
    version, conflict = _consistent_version(
        [value for _, value in sources],
        lambda value: _runtime_version(value, accelerator, context),
        context,
    )
    source = ",".join(sorted({name for name, _ in sources})) if version else None
    return version, source, conflict


def _config_facts(
    config: Mapping[str, object], *, tag: str, accelerator: str, context: str
) -> tuple[dict[str, str], dict[str, str], list[str], list[str]]:
    env = _config_env(config, context)
    python_version, python_source, python_conflict = _python_from_config(
        config, env, f"{context}.python_version"
    )
    runtime_version, runtime_source, runtime_conflict = _runtime_from_config(
        config, env, accelerator, f"{context}.{accelerator}_version"
    )
    os_id = "openeuler" if "openeuler" in tag.casefold() else "linux"
    facts = {
        "python_version": python_version or "",
        "os_id": os_id,
        "os_version": _UNREPORTED,
        "cuda_version": (
            runtime_version if accelerator == "cuda" and runtime_version else ""
        ),
        "cann_version": (
            runtime_version if accelerator == "ascend" and runtime_version else ""
        ),
        "soc_version": (
            "na" if accelerator == "cuda" else env.get("SOC_VERSION", "").casefold()
        ),
    }
    sources = {"os": "tag:openeuler" if os_id == "openeuler" else "oci"}
    if python_source:
        sources["python_version"] = python_source
    runtime_field = "cuda_version" if accelerator == "cuda" else "cann_version"
    if runtime_source:
        sources[runtime_field] = runtime_source
    if accelerator == "ascend" and facts["soc_version"]:
        sources["soc_version"] = "env:SOC_VERSION"
    required = ["python_version", runtime_field]
    if accelerator == "ascend":
        required.append("soc_version")
    missing = [field for field in required if not facts[field]]
    conflicts = [
        value for value in (python_conflict, runtime_conflict) if value is not None
    ]
    return facts, sources, missing, conflicts


def _inspect_members(
    reference: str,
    repository: str,
    *,
    manifest_loader: JsonLoader,
    config_loader: JsonLoader,
    digest_loader: Callable[[str], str] | None,
) -> list[dict[str, Any]]:
    manifest = _json_mapping(
        manifest_loader(reference), f"runtime manifest {reference}"
    )
    media_type = manifest.get("mediaType")
    members: list[dict[str, Any]] = []
    if media_type in OCI_INDEX_MEDIA_TYPES:
        descriptors = manifest.get("manifests")
        if not isinstance(descriptors, list):
            raise ValueError(
                f"runtime manifest {reference}: index has no manifests array"
            )
        observed: set[str] = set()
        for index, raw_descriptor in enumerate(descriptors):
            descriptor = _mapping(
                raw_descriptor, f"runtime manifest {reference}.manifests[{index}]"
            )
            platform = descriptor.get("platform")
            if not isinstance(platform, Mapping):
                continue
            operating_system = str(platform.get("os", "")).lower()
            architecture = str(platform.get("architecture", "")).lower()
            if (
                operating_system != "linux"
                or architecture not in SUPPORTED_ARCHITECTURES
            ):
                continue
            if architecture in observed:
                raise ValueError(
                    f"runtime manifest {reference}: duplicate linux/{architecture} member"
                )
            if descriptor.get("mediaType") not in OCI_MANIFEST_MEDIA_TYPES:
                raise ValueError(
                    f"runtime manifest {reference}: linux/{architecture} is not an image manifest"
                )
            digest = descriptor.get("digest")
            if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
                raise ValueError(
                    f"runtime manifest {reference}: linux/{architecture} has invalid digest"
                )
            member_reference = f"{repository}@{digest}"
            _, _, config = _load_platform_config(
                member_reference,
                config_loader=config_loader,
                expected_platform=(operating_system, architecture),
            )
            members.append(
                {
                    "cpu_arch": architecture,
                    "platform": f"linux/{architecture}",
                    "image_reference": member_reference,
                    "config": config,
                }
            )
            observed.add(architecture)
    elif media_type in OCI_MANIFEST_MEDIA_TYPES:
        operating_system, architecture, config = _load_platform_config(
            reference, config_loader=config_loader, expected_platform=None
        )
        if operating_system == "linux" and architecture in SUPPORTED_ARCHITECTURES:
            image_reference = reference
            if digest_loader is not None:
                digest = digest_loader(reference)
                if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
                    raise ValueError(
                        f"runtime manifest {reference}: invalid resolved digest"
                    )
                image_reference = f"{repository}@{digest}"
            members.append(
                {
                    "cpu_arch": architecture,
                    "platform": f"linux/{architecture}",
                    "image_reference": image_reference,
                    "config": config,
                }
            )
    else:
        raise ValueError(
            f"runtime manifest {reference}: unsupported mediaType {media_type!r}"
        )
    if not members:
        raise ValueError(
            f"runtime manifest {reference}: no supported Linux architecture"
        )
    order = {value: index for index, value in enumerate(SUPPORTED_ARCHITECTURES)}
    return sorted(members, key=lambda item: order[item["cpu_arch"]])


def inspect_runtime_references(
    references: Sequence[str],
    *,
    products: Sequence[Mapping[str, object]],
    runners: Mapping[str, str],
    manifest_loader: JsonLoader,
    config_loader: JsonLoader,
    digest_loader: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Inspect immutable member configs and schedule native fallback only when needed."""

    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise ValueError("runtime references must be a sequence")
    if not references:
        raise ValueError("runtime references must not be empty")
    policy_by_repository = _policy_by_repository(products, runners)
    runner_values = dict(runners)
    seen_references: set[str] = set()
    runtimes: list[dict[str, Any]] = []
    all_members: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    for index, raw_reference in enumerate(references, start=1):
        repository, tag = parse_runtime_reference(raw_reference)
        reference = f"{repository}:{tag}"
        if reference in seen_references:
            raise ValueError(f"duplicate runtime reference {reference!r}")
        seen_references.add(reference)
        policy = policy_by_repository.get(repository)
        if policy is None:
            raise ValueError(
                f"runtime repository {repository!r} is not configured in products policy"
            )
        if digest_loader is None:
            raise ValueError(
                f"runtime {reference}: digest_loader is required for immutable selection"
            )
        runtime_digest = digest_loader(reference)
        if (
            not isinstance(runtime_digest, str)
            or _DIGEST.fullmatch(runtime_digest) is None
        ):
            raise ValueError(f"runtime {reference}: resolved digest is invalid")
        pinned_reference = f"{repository}@{runtime_digest}"
        image_members = _inspect_members(
            pinned_reference,
            repository,
            manifest_loader=manifest_loader,
            config_loader=config_loader,
            digest_loader=None,
        )
        request_id = f"runtime-{index:03d}"
        probe_ids: list[str] = []
        for member in image_members:
            architecture = member["cpu_arch"]
            runner = runner_values.get(architecture)
            if not isinstance(runner, str) or not runner:
                raise ValueError(f"no runner configured for linux/{architecture}")
            probe_id = f"{request_id}-{architecture}"
            probe_ids.append(probe_id)
            context = f"runtime config {member['image_reference']}"
            facts, fact_sources, missing, conflicts = _config_facts(
                _mapping(member.get("config"), context),
                tag=tag,
                accelerator=str(policy["accelerator"]),
                context=context,
            )
            request = {
                "probe_id": probe_id,
                "request_id": request_id,
                "product_id": policy["product_id"],
                "runtime_ref": reference,
                "repository": repository,
                "tag": tag,
                "target_repository": policy["target_repository"],
                "accelerator": policy["accelerator"],
                "backend": policy["backend"],
                "backend_by_soc": copy.deepcopy(policy["backend_by_soc"]),
                "cpu_arch": architecture,
                "platform": member["platform"],
                "runner": runner,
                "image_reference": member["image_reference"],
                "runtime_digest": runtime_digest,
                "config_facts": facts,
                "fact_sources": fact_sources,
                "missing_required_fields": missing,
                "metadata_conflicts": conflicts,
                "fallback_required": bool(missing or conflicts),
            }
            all_members.append(request)
            if request["fallback_required"]:
                fallbacks.append(copy.deepcopy(request))
        runtimes.append(
            {
                "request_id": request_id,
                "product_id": policy["product_id"],
                "runtime_ref": reference,
                "repository": repository,
                "tag": tag,
                "target_repository": policy["target_repository"],
                "runtime_digest": runtime_digest,
                "architectures": [member["cpu_arch"] for member in image_members],
                "probe_ids": probe_ids,
            }
        )
    return {
        "kind": "ucm-runtime-inspection",
        "schema_version": 2,
        "runtimes": runtimes,
        "members": all_members,
        "probe_matrix": {"include": fallbacks},
    }


def _version_match(value: object, context: str) -> re.Match[str]:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    match = _NUMERIC_VERSION.search(value.strip())
    if match is None:
        raise ValueError(f"{context} has no numeric version")
    return match


def _python_coordinates(value: object, context: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    match = _PYTHON_VERSION.search(value.strip())
    if match is None:
        raise ValueError(f"{context} has no Python major/minor")
    version = f"{int(match.group(1))}.{int(match.group(2))}"
    return version, f"cp{int(match.group(1))}{int(match.group(2))}"


def _runtime_version(value: object, accelerator: str, context: str) -> str:
    match = _version_match(value, context)
    major, minor, patch = match.groups()
    if accelerator == "cuda":
        return f"cuda-{int(major)}.{int(minor)}"
    return f"cann-{int(major)}.{int(minor)}.{int(patch or 0)}"


def _unquote(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip().strip("'\"")


def aggregate_runtime_probes(
    inspection: Mapping[str, object], raw_probes: Sequence[Mapping[str, object]]
) -> dict[str, Any]:
    """Merge Crane config facts with native fallback results."""

    inspected = _mapping(inspection, "runtime inspection")
    if (
        inspected.get("kind") != "ucm-runtime-inspection"
        or inspected.get("schema_version") != 2
    ):
        raise ValueError("runtime inspection has an unsupported contract")
    requests = inspected.get("members")
    if not isinstance(requests, list) or not requests:
        raise ValueError("runtime inspection members must not be empty")
    expected: dict[str, dict[str, Any]] = {}
    fallback_ids: set[str] = set()
    for raw_request in requests:
        request = _mapping(raw_request, "runtime probe request")
        probe_id = _string(request, "probe_id", "runtime probe request")
        if probe_id in expected:
            raise ValueError(f"duplicate runtime probe request {probe_id!r}")
        expected[probe_id] = request
        if request.get("fallback_required") is True:
            fallback_ids.add(probe_id)

    actual: dict[str, dict[str, Any]] = {}
    for raw_probe in raw_probes:
        probe = _mapping(raw_probe, "raw runtime probe")
        probe_id = _string(probe, "probe_id", "raw runtime probe")
        if probe_id in actual:
            raise ValueError(f"duplicate raw runtime probe {probe_id!r}")
        actual[probe_id] = probe
    missing = sorted(fallback_ids - set(actual))
    extra = sorted(set(actual) - fallback_ids)
    if missing or extra:
        raise ValueError(
            f"runtime probe set differs from inspection: missing={missing}, extra={extra}"
        )

    normalized: list[dict[str, Any]] = []
    for raw_request in requests:
        request = _mapping(raw_request, "runtime probe request")
        probe_id = str(request["probe_id"])
        fallback = request.get("fallback_required") is True
        raw = (
            actual[probe_id]
            if fallback
            else _mapping(
                request.get("config_facts"), f"runtime config facts {probe_id}"
            )
        )
        context = f"runtime probe {probe_id}"
        python_version, python_abi = _python_coordinates(
            raw.get("python_version"), f"{context}.python_version"
        )
        os_id = _unquote(raw.get("os_id"), f"{context}.os_id").casefold()
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", os_id) is None:
            raise ValueError(f"{context}.os_id is invalid")
        os_version = _unquote(raw.get("os_version"), f"{context}.os_version")
        accelerator = str(request["accelerator"])
        if accelerator == "cuda":
            runtime = _runtime_version(
                raw.get("cuda_version"), accelerator, f"{context}.cuda_version"
            )
            if raw.get("cann_version") not in (None, ""):
                raise ValueError(f"{context}: CUDA probe cannot report CANN")
            soc_version = "na"
            backend = str(request["backend"])
        else:
            runtime = _runtime_version(
                raw.get("cann_version"), accelerator, f"{context}.cann_version"
            )
            if raw.get("cuda_version") not in (None, ""):
                raise ValueError(f"{context}: Ascend probe cannot report CUDA")
            soc_version = _unquote(
                raw.get("soc_version"), f"{context}.soc_version"
            ).casefold()
            backend_by_soc = _mapping(
                request.get("backend_by_soc"), f"{context}.backend_by_soc"
            )
            backend = backend_by_soc.get(soc_version)
            if not isinstance(backend, str) or not backend:
                raise ValueError(
                    f"{context}: SOC_VERSION {soc_version!r} has no configured backend"
                )
        normalized.append(
            {
                "probe_id": probe_id,
                "request_id": request["request_id"],
                "product_id": request["product_id"],
                "runtime_ref": request["runtime_ref"],
                "repository": request["repository"],
                "tag": request["tag"],
                "target_repository": request["target_repository"],
                "runtime_digest": request["runtime_digest"],
                "cpu_arch": request["cpu_arch"],
                "platform": request["platform"],
                "runner": request["runner"],
                "image_reference": request["image_reference"],
                "backend": backend,
                "accelerator_runtime": runtime,
                "soc_version": soc_version,
                "python_version": python_version,
                "python_abi": python_abi,
                "os_id": os_id,
                "os_version": os_version,
                "glibc_version": None,
                "capability_source": "runtime-pull" if fallback else "crane-config",
            }
        )
    return {
        "kind": "ucm-runtime-probe",
        "schema_version": RUNTIME_PROBE_SCHEMA_VERSION,
        "probes": normalized,
    }


def _manylinux_floor(value: object, context: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    match = _MANYLINUX.fullmatch(value)
    if match is None:
        raise ValueError(f"{context} must be manylinux_<major>_<minor>")
    return int(match.group(1)), int(match.group(2))


def _created(value: object, context: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be an RFC3339 timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError as error:
        raise ValueError(f"{context} must be an RFC3339 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{context} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def sanitize_oci_tag_component(value: str, *, max_length: int = 128) -> str:
    """Normalize one human/source component for embedding in an OCI tag."""

    if not isinstance(value, str) or not value:
        raise ValueError("OCI tag component must be a non-empty string")
    if (
        not isinstance(max_length, int)
        or isinstance(max_length, bool)
        or max_length < 1
    ):
        raise ValueError("OCI tag component max_length must be positive")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    normalized = normalized[:max_length].rstrip(".-")
    if not normalized or core.OCI_TAG_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"value {value!r} cannot be represented in an OCI tag")
    return normalized


def _capability(probe: Mapping[str, object]) -> dict[str, str]:
    return {
        key: str(probe[key])
        for key in (
            "backend",
            "accelerator_runtime",
            "soc_version",
            "python_version",
            "python_abi",
            "cpu_arch",
        )
    }


def _builder_record(raw_builder: Mapping[str, object], index: int) -> dict[str, Any]:
    context = f"Builder records[{index}]"
    builder = _mapping(raw_builder, context)
    required = (
        "backend",
        "accelerator_runtime",
        "soc_version",
        "python_abi",
        "cpu_arch",
        "manylinux",
        "target_repository",
        "target_tag",
        "created",
        "checked",
    )
    for field in required[:-1]:
        _string(builder, field, context)
    if not isinstance(builder.get("checked"), bool):
        raise ValueError(f"{context}: checked must be boolean")
    if builder["cpu_arch"] not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"{context}: unsupported cpu_arch")
    if core.OCI_REPOSITORY_PATTERN.fullmatch(str(builder["target_repository"])) is None:
        raise ValueError(f"{context}: invalid target_repository")
    if core.OCI_TAG_PATTERN.fullmatch(str(builder["target_tag"])) is None:
        raise ValueError(f"{context}: invalid target_tag")
    floor = _manylinux_floor(builder["manylinux"], f"{context}.manylinux")
    created_at = _created(builder["created"], f"{context}.created")
    python_version = builder.get("python_version")
    if python_version is not None:
        normalized, _ = _python_coordinates(python_version, f"{context}.python_version")
        builder["python_version"] = normalized
    builder["soc_version"] = str(builder["soc_version"]).casefold()
    builder["_floor"] = floor
    builder["_created"] = created_at
    builder["_id"] = str(
        builder.get("id") or f"{builder['target_repository']}:{builder['target_tag']}"
    )
    return builder


def match_runtime_builders(
    runtime_probe: Mapping[str, object],
    builders: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Choose compatible checked Builders for probes without source revisions."""

    document = _mapping(runtime_probe, "runtime probe")
    if (
        document.get("kind") != "ucm-runtime-probe"
        or document.get("schema_version") != RUNTIME_PROBE_SCHEMA_VERSION
    ):
        raise ValueError("runtime probe has an unsupported contract")
    probes = document.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("runtime probe must contain probes")
    if not isinstance(builders, Sequence) or isinstance(builders, (str, bytes)):
        raise ValueError("Builder records must be a sequence")
    available = [
        _builder_record(raw_builder, index)
        for index, raw_builder in enumerate(builders)
    ]

    matches: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    for raw_probe in probes:
        probe = _mapping(raw_probe, "runtime probe item")
        probe_id = _string(probe, "probe_id", "runtime probe item")
        capability = _capability(probe)
        exact = [
            builder
            for builder in available
            if all(
                str(builder[field]) == capability[field]
                for field in (
                    "backend",
                    "accelerator_runtime",
                    "soc_version",
                    "python_abi",
                    "cpu_arch",
                )
            )
            and (
                "python_version" not in builder
                or builder["python_version"] == capability["python_version"]
            )
        ]
        checked = [builder for builder in exact if builder["checked"]]
        if not checked:
            requested = ", ".join(
                f"{key}={capability[key]}"
                for key in (
                    "backend",
                    "accelerator_runtime",
                    "soc_version",
                    "python_abi",
                    "cpu_arch",
                )
            )
            floors = sorted({str(builder["manylinux"]) for builder in checked})
            detail = (
                f"no checked Builder for {requested}; exact_candidates={len(exact)}, "
                f"checked_candidates={len(checked)}, checked_manylinux={floors}"
            )
            problems.append(
                {
                    "stage": "builder-match",
                    "reason": "missing-compatible-builder",
                    "probe_id": probe_id,
                    "request_id": probe["request_id"],
                    "runtime_ref": probe["runtime_ref"],
                    "capability": capability,
                    "exact_candidates": len(exact),
                    "checked_candidates": len(checked),
                    "checked_manylinux": floors,
                    "detail": detail,
                }
            )
            continue
        lowest_floor = min(builder["_floor"] for builder in checked)
        floor_candidates = [
            builder for builder in checked if builder["_floor"] == lowest_floor
        ]
        newest_created = max(builder["_created"] for builder in floor_candidates)
        newest = [
            builder
            for builder in floor_candidates
            if builder["_created"] == newest_created
        ]
        selected = max(
            newest,
            key=lambda builder: (
                str(builder["target_repository"]),
                str(builder["target_tag"]),
            ),
        )
        matches.append(
            {
                "probe_id": probe_id,
                "request_id": probe["request_id"],
                "runtime_ref": probe["runtime_ref"],
                "cpu_arch": probe["cpu_arch"],
                "wheel_id": str(selected["_id"]),
                "capability": capability,
                "builder": {
                    "id": selected["_id"],
                    "repository": selected["target_repository"],
                    "tag": selected["target_tag"],
                    "manylinux": selected["manylinux"],
                    "created": selected["created"],
                },
                "builder_record": {
                    key: copy.deepcopy(value)
                    for key, value in selected.items()
                    if not key.startswith("_")
                },
            }
        )
    return {
        "kind": "ucm-runtime-builder-matches",
        "schema_version": 1,
        "ok": not problems,
        "matches": matches,
        "problems": problems,
    }


def _positive_integer_text(value: object, context: str) -> str:
    text = str(value)
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise ValueError(f"{context} must be a positive integer")
    return text


def project_pr_tag(
    runtime_tag: str,
    *,
    pr_number: int | str,
    author: str,
    run_id: int | str,
) -> str:
    """Build the collision-resistant PR base tag, reserving member suffix room."""

    pr = _positive_integer_text(pr_number, "PR number")
    run = _positive_integer_text(run_id, "run id")
    author_component = sanitize_oci_tag_component(author.lower(), max_length=39)
    prefix = f"pr-{pr}-{author_component}-run-{run}-"
    member_suffix_room = max(len(f"-{arch}") for arch in SUPPORTED_ARCHITECTURES)
    maximum_runtime_length = 128 - len(prefix) - member_suffix_room
    if maximum_runtime_length < 1:
        raise ValueError("PR tag prefix leaves no room for the runtime tag")
    runtime_component = sanitize_oci_tag_component(
        runtime_tag, max_length=maximum_runtime_length
    )
    result = prefix + runtime_component
    if core.OCI_TAG_PATTERN.fullmatch(result) is None:
        raise ValueError("projected PR tag is invalid")
    return result


def project_pr_publication(
    runtime_probe: Mapping[str, object],
    builder_matches: Mapping[str, object],
    *,
    pr_number: int | str,
    author: str,
    run_id: int | str,
) -> dict[str, Any]:
    """Project per-member tags and indexes from actual probed architectures."""

    probe_document = _mapping(runtime_probe, "runtime probe")
    match_document = _mapping(builder_matches, "runtime Builder matches")
    if (
        probe_document.get("kind") != "ucm-runtime-probe"
        or probe_document.get("schema_version") != RUNTIME_PROBE_SCHEMA_VERSION
    ):
        raise ValueError("runtime probe has an unsupported contract")
    if match_document.get("kind") != "ucm-runtime-builder-matches":
        raise ValueError("runtime Builder matches have an unsupported contract")
    problems = match_document.get("problems")
    if (
        not isinstance(problems, list)
        or problems
        or match_document.get("ok") is not True
    ):
        raise ValueError("cannot project publication with unresolved Builder problems")
    raw_matches = match_document.get("matches")
    if not isinstance(raw_matches, list):
        raise ValueError("runtime Builder matches must contain matches")
    matches = {
        _string(item, "probe_id", "runtime Builder match"): _mapping(
            item, "runtime Builder match"
        )
        for item in raw_matches
        if isinstance(item, Mapping)
    }
    raw_probes = probe_document.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError("runtime probe must contain probes")

    grouped: dict[str, list[dict[str, Any]]] = {}
    request_order: list[str] = []
    for raw_probe in raw_probes:
        probe = _mapping(raw_probe, "runtime probe item")
        request_id = _string(probe, "request_id", "runtime probe item")
        if request_id not in grouped:
            grouped[request_id] = []
            request_order.append(request_id)
        grouped[request_id].append(probe)

    families: list[dict[str, Any]] = []
    member_matrix: list[dict[str, Any]] = []
    index_matrix: list[dict[str, Any]] = []
    coordinates: set[tuple[str, str]] = set()
    for request_id in request_order:
        probes = grouped[request_id]
        runtime_refs = {str(probe["runtime_ref"]) for probe in probes}
        repositories = {str(probe["target_repository"]) for probe in probes}
        tags = {str(probe["tag"]) for probe in probes}
        if len(runtime_refs) != 1 or len(repositories) != 1 or len(tags) != 1:
            raise ValueError(
                f"runtime request {request_id} has inconsistent probe identity"
            )
        runtime_ref = next(iter(runtime_refs))
        target_repository = next(iter(repositories))
        base_tag = project_pr_tag(
            next(iter(tags)), pr_number=pr_number, author=author, run_id=run_id
        )
        if (target_repository, base_tag) in coordinates:
            raise ValueError(
                f"PR tag sanitization collides at {target_repository}:{base_tag}"
            )
        coordinates.add((target_repository, base_tag))
        member_ids: list[str] = []
        member_refs: list[str] = []
        wheel_ids: set[str] = set()
        has_index = len(probes) > 1
        for probe in probes:
            probe_id = str(probe["probe_id"])
            match = matches.get(probe_id)
            if match is None:
                raise ValueError(f"runtime probe {probe_id!r} has no Builder match")
            architecture = str(probe["cpu_arch"])
            member_tag = f"{base_tag}-{architecture}" if has_index else base_tag
            member_ref = f"{target_repository}:{member_tag}"
            member_ids.append(probe_id)
            member_refs.append(member_ref)
            wheel_ids.add(str(match["wheel_id"]))
            member_matrix.append(
                {
                    "id": probe_id,
                    "request_id": request_id,
                    "runtime_ref": runtime_ref,
                    "cpu_arch": architecture,
                    "runner": probe["runner"],
                    "wheel_id": match["wheel_id"],
                    "target_repository": target_repository,
                    "target_tag": member_tag,
                    "target_ref": member_ref,
                }
            )
        index_ref = f"{target_repository}:{base_tag}" if has_index else ""
        family = {
            "id": request_id,
            "runtime_ref": runtime_ref,
            "target_repository": target_repository,
            "target_tag": base_tag,
            "member_ids": member_ids,
            "member_refs": member_refs,
            "wheel_ids": sorted(wheel_ids),
            "has_index": has_index,
            "index_ref": index_ref,
            "final_refs": [index_ref] if has_index else member_refs,
        }
        families.append(family)
        if has_index:
            index_matrix.append(
                {
                    "id": request_id,
                    "runtime_ref": runtime_ref,
                    "target_repository": target_repository,
                    "target_tag": base_tag,
                    "target_ref": index_ref,
                    "members": member_refs,
                }
            )
    return {
        "kind": "ucm-pr-image-publication",
        "schema_version": 1,
        "families": families,
        "member_matrix": {"include": member_matrix},
        "index_matrix": {"include": index_matrix},
    }


def _receipt_problem(raw: Mapping[str, object], index: int) -> dict[str, Any]:
    problem = _mapping(raw, f"receipt problems[{index}]")
    stage = _string(problem, "stage", f"receipt problems[{index}]")
    reason = _string(problem, "reason", f"receipt problems[{index}]")
    detail = _string(problem, "detail", f"receipt problems[{index}]")
    return {
        **copy.deepcopy(problem),
        "stage": stage,
        "reason": reason,
        "detail": detail,
    }


def build_receipt(
    *,
    requested_refs: Sequence[str],
    stage_results: Mapping[str, str],
    inspection: Mapping[str, object] | None = None,
    runtime_probe: Mapping[str, object] | None = None,
    builder_matches: Mapping[str, object] | None = None,
    publication: Mapping[str, object] | None = None,
    failures: Sequence[Mapping[str, object]] = (),
    run_url: str = "",
) -> dict[str, Any]:
    """Generate structured PR receipt data for both success and early failure."""

    runtime_rows: list[dict[str, Any]] = []
    rows_by_ref: dict[str, dict[str, Any]] = {}
    input_problems: list[dict[str, Any]] = []
    for raw_reference in requested_refs:
        try:
            repository, tag = parse_runtime_reference(raw_reference)
            reference = f"{repository}:{tag}"
        except ValueError as error:
            reference = str(raw_reference)
            input_problems.append(
                {
                    "stage": "inspect",
                    "reason": "invalid-runtime-reference",
                    "detail": str(error),
                    "runtime_ref": reference,
                }
            )
        if reference in rows_by_ref:
            raise ValueError(f"duplicate receipt runtime reference {reference!r}")
        row = {
            "runtime_ref": reference,
            "architectures": [],
            "capabilities": [],
            "wheel_ids": [],
            "member_refs": [],
            "final_refs": [],
        }
        runtime_rows.append(row)
        rows_by_ref[reference] = row

    stages: list[dict[str, str]] = []
    allowed_results = {"success", "failure", "cancelled", "skipped"}
    for stage, result in stage_results.items():
        if not isinstance(stage, str) or not stage:
            raise ValueError("receipt stage name must be non-empty")
        if result not in allowed_results:
            raise ValueError(
                f"receipt stage {stage!r} has unsupported result {result!r}"
            )
        stages.append({"stage": stage, "result": result})

    if inspection is not None:
        inspected = _mapping(inspection, "receipt runtime inspection")
        for raw_runtime in inspected.get("runtimes", []):
            runtime = _mapping(raw_runtime, "receipt inspected runtime")
            reference = str(runtime["runtime_ref"])
            row = rows_by_ref.get(reference)
            if row is not None:
                row["architectures"] = list(runtime.get("architectures", []))

    matches_by_probe: dict[str, dict[str, Any]] = {}
    collected_problems: list[dict[str, Any]] = input_problems
    if builder_matches is not None:
        matched = _mapping(builder_matches, "receipt Builder matches")
        for raw_match in matched.get("matches", []):
            match = _mapping(raw_match, "receipt Builder match")
            matches_by_probe[str(match["probe_id"])] = match
        for raw_problem in matched.get("problems", []):
            collected_problems.append(
                _receipt_problem(
                    _mapping(raw_problem, "receipt Builder problem"),
                    len(collected_problems),
                )
            )

    if runtime_probe is not None:
        probed = _mapping(runtime_probe, "receipt runtime probe")
        for raw_probe in probed.get("probes", []):
            probe = _mapping(raw_probe, "receipt runtime probe item")
            row = rows_by_ref.get(str(probe["runtime_ref"]))
            if row is None:
                continue
            capability = {
                "probe_id": probe["probe_id"],
                "cpu_arch": probe["cpu_arch"],
                "backend": probe["backend"],
                "accelerator_runtime": probe["accelerator_runtime"],
                "soc_version": probe["soc_version"],
                "python_abi": probe["python_abi"],
                "os": f"{probe['os_id']}:{probe['os_version']}",
                "glibc": probe["glibc_version"],
            }
            match = matches_by_probe.get(str(probe["probe_id"]))
            if match is not None:
                capability["wheel_id"] = match["wheel_id"]
                row["wheel_ids"].append(match["wheel_id"])
            row["capabilities"].append(capability)
        for row in runtime_rows:
            row["wheel_ids"] = sorted(set(row["wheel_ids"]))

    if publication is not None:
        published = _mapping(publication, "receipt publication")
        member_succeeded = stage_results.get("member") == "success"
        index_succeeded = stage_results.get("index") == "success"
        for raw_family in published.get("families", []):
            family = _mapping(raw_family, "receipt publication family")
            row = rows_by_ref.get(str(family["runtime_ref"]))
            if row is not None:
                if member_succeeded:
                    row["member_refs"] = list(family.get("member_refs", []))
                if member_succeeded and (
                    not family.get("has_index") or index_succeeded
                ):
                    row["final_refs"] = list(family.get("final_refs", []))

    for raw_failure in failures:
        collected_problems.append(
            _receipt_problem(raw_failure, len(collected_problems))
        )
    required_build_stages = {"wheel", "image", "member"}
    if required_build_stages <= set(stage_results):
        for stage in sorted(required_build_stages):
            if stage_results[stage] != "success":
                collected_problems.append(
                    {
                        "stage": stage,
                        "reason": "required-stage-incomplete",
                        "detail": (
                            f"required image build stage finished as "
                            f"{stage_results[stage]}"
                        ),
                    }
                )
        has_index = bool(
            publication
            and any(
                bool(item.get("has_index"))
                for item in publication.get("families", [])  # type: ignore[union-attr]
                if isinstance(item, dict)
            )
        )
        if has_index and stage_results.get("index") != "success":
            collected_problems.append(
                {
                    "stage": "index",
                    "reason": "required-stage-incomplete",
                    "detail": (
                        "multi-architecture publication requires a successful index "
                        f"stage, got {stage_results.get('index', 'missing')}"
                    ),
                }
            )
    failed = bool(collected_problems) or any(
        stage["result"] in {"failure", "cancelled"} for stage in stages
    )
    return {
        "kind": "ucm-pr-build-receipt",
        "schema_version": 1,
        "command": "/ucm-build image",
        "status": "failure" if failed else "success",
        "run_url": run_url,
        "stages": stages,
        "runtimes": runtime_rows,
        "problems": collected_problems,
    }


def render_receipt_markdown(value: Mapping[str, object]) -> str:
    """Render one structured receipt into the PR comment body."""
    receipt = _mapping(value, "PR build receipt")
    if receipt.get("kind") != "ucm-pr-build-receipt":
        raise ValueError("PR build receipt has an unsupported contract")
    status = _string(receipt, "status", "PR build receipt")
    lines = [f"## `/ucm-build image` build receipt · `{status}`", ""]
    stages = receipt.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("PR build receipt stages must be a list")
    lines.extend(("| Stage | Result |", "| --- | --- |"))
    for raw_stage in stages:
        stage = _mapping(raw_stage, "PR build receipt stage")
        lines.append(f"| `{stage['stage']}` | `{stage['result']}` |")
    runtimes = receipt.get("runtimes", [])
    if not isinstance(runtimes, list):
        raise ValueError("PR build receipt runtimes must be a list")
    for raw_runtime in runtimes:
        item = _mapping(raw_runtime, "PR build receipt runtime")
        lines.extend(("", f"### `{item['runtime_ref']}`", ""))
        architectures = item.get("architectures", [])
        lines.append(
            "Architectures: "
            + (", ".join(f"`{arch}`" for arch in architectures) or "not resolved")
        )
        capabilities = item.get("capabilities", [])
        if capabilities:
            lines.extend(
                (
                    "",
                    "| Arch | Backend | Runtime | SOC | Python | OS | glibc | Wheel |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                )
            )
            for raw_capability in capabilities:
                capability = _mapping(raw_capability, "receipt capability")
                lines.append(
                    "| "
                    + " | ".join(
                        f"`{capability.get(key, '')}`"
                        for key in (
                            "cpu_arch",
                            "backend",
                            "accelerator_runtime",
                            "soc_version",
                            "python_abi",
                            "os",
                            "glibc",
                            "wheel_id",
                        )
                    )
                    + " |"
                )
        final_refs = item.get("final_refs", [])
        if final_refs:
            lines.extend(("", "Published images:", ""))
            lines.extend(f"- `docker pull {reference}`" for reference in final_refs)
    raw_problems = receipt.get("problems", [])
    if raw_problems:
        lines.extend(("", "### Problems", ""))
        for raw_problem in raw_problems:
            problem = _mapping(raw_problem, "receipt problem")
            lines.append(
                f"- `{problem.get('stage', 'unknown')}` / "
                f"`{problem.get('reason', 'failure')}`: {problem.get('detail', '')}"
            )
    run_url = str(receipt.get("run_url", ""))
    if run_url:
        lines.extend(("", f"Artifacts and logs: {run_url}"))
    return "\n".join(lines) + "\n"
