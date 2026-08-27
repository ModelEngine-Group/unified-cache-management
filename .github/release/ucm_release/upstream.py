"""Registry-owned runtime selection and raw Builder capability resolution.

Published OCI objects are the formal source of truth: runtime tags decide which
versions exist, member configs (or targeted native fallback) decide the Wheel
capability, and raw Builder member digests decide Builder identity. Upstream
Git is not read.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from . import core
from . import runtime as runtime_contract

SELECTION_KIND = "ucm-runtime-selection"
SELECTION_SCHEMA_VERSION = 3
CANDIDATES_KIND = "ucm-runtime-candidates"
CANDIDATES_SCHEMA_VERSION = 1
RELEASE_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ARCHITECTURES = ("amd64", "arm64")

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MANYLINUX = re.compile(r"manylinux_(\d+)_(\d+)")
_FORMAL_TAG = re.compile(
    r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+(?:rc[0-9]+)?)"
    r"(?P<suffix>(?:-[A-Za-z0-9_.]+)*)$"
)
_NIGHTLY_TAG = re.compile(
    r"^nightly-releases-v(?P<version>[0-9]+\.[0-9]+\.[0-9]+rc(?:[0-9]+)?)"
    r"(?P<suffix>(?:-[A-Za-z0-9_.]+)*)$"
)
_CUDA_BUILDER_TAG = re.compile(r"^cuda(?P<runtime>[0-9]+\.[0-9]+)$")
_ASCEND_BUILDER_TAG = re.compile(
    r"^(?P<runtime>[0-9]+\.[0-9]+\.[0-9]+)-"
    r"(?P<variant>310p|910b|a3|950)-"
    r"(?P<manylinux>manylinux_[0-9]+_[0-9]+)-"
    r"py(?P<python>[0-9]+\.[0-9]+)$"
)

_WHEEL_BUILD_FIELDS = {
    "id",
    "product_id",
    "build_group",
    "backend",
    "accelerator",
    "accelerator_runtime",
    "variant",
    "soc_version",
    "runtime_variant",
    "python_version",
    "python_abi",
    "manylinux",
    "cpu_arch",
    "source_image",
    "source_image_digest",
    "build_mode",
    "recipe_revision",
    "sync_mode",
}
_RUNTIME_FIELDS = {
    "id",
    "product_id",
    "runtime_repository",
    "runtime_tag",
    "runtime_digest",
    "runtime_variant",
    "backend",
    "accelerator_runtime",
    "variant",
    "soc_version",
    "python_version",
    "python_abi",
    "os_id",
    "os_version",
    "glibc_version",
    "architectures",
    "member_references",
    "wheel_build_ids",
    "version",
    "channel",
    "target_repository",
    "target_tag",
}
_PROBLEM_FIELDS = {"backend", "capability", "reason", "runtime"}


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a mapping")
    return dict(value)


def _string(mapping: Mapping[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value.strip()


def _crane(operation: str, reference: str) -> str:
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


def _fixture_tags(
    fixture: Mapping[str, object] | None, repository: str
) -> list[str] | None:
    if fixture is None:
        return None
    repositories = fixture.get("repositories")
    if not isinstance(repositories, Mapping):
        return None
    raw_repository = repositories.get(repository)
    if not isinstance(raw_repository, Mapping):
        return None
    pages = raw_repository.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"Registry fixture {repository}: pages must be a list")
    tags: list[str] = []
    for index, raw_page in enumerate(pages):
        page = _mapping(raw_page, f"Registry fixture {repository}.pages[{index}]")
        values = page.get("tags")
        if not isinstance(values, list) or not all(
            isinstance(tag, str) for tag in values
        ):
            raise ValueError(
                f"Registry fixture {repository}.pages[{index}].tags is invalid"
            )
        tags.extend(values)
    return sorted(set(tags))


def _repository_tags(
    repository: str,
    *,
    tag_fixture: Mapping[str, object] | None,
    tag_loader: Callable[[str], Sequence[str]] | None,
) -> list[str]:
    fixture = _fixture_tags(tag_fixture, repository)
    if fixture is not None:
        return fixture
    values = (
        tag_loader(repository)
        if tag_loader is not None
        else _crane("ls", repository).splitlines()
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(
            f"Registry tag loader for {repository} returned an invalid value"
        )
    return sorted(set(str(tag) for tag in values if str(tag)))


def _version_window(
    product: Mapping[str, object],
) -> tuple[Version, Version | None]:
    context = f"release product {_string(product, 'id', 'release product')}"
    try:
        minimum = Version(_string(product, "minimum_version", context))
        raw_maximum = product.get("maximum_version")
        maximum = Version(str(raw_maximum)) if raw_maximum is not None else None
    except InvalidVersion as error:
        raise ValueError(f"{context}: version window is invalid") from error
    for name, value in (("minimum_version", minimum), ("maximum_version", maximum)):
        if value is not None and (value.local is not None or value.dev is not None):
            raise ValueError(f"{context}: {name} must be a formal version")
    if maximum is not None and maximum < minimum:
        raise ValueError(f"{context}: maximum_version must be >= minimum_version")
    return minimum, maximum


def _minor_limit(value: object, context: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or (value != -1 and value < 1)
    ):
        raise ValueError(f"{context}: must be -1 or an integer >= 1")
    return value


def _parsed_runtime_tag(product_id: str, tag: str) -> dict[str, object] | None:
    nightly = _NIGHTLY_TAG.fullmatch(tag) if product_id == "vllm-ascend" else None
    formal = _FORMAL_TAG.fullmatch(tag)
    match = nightly or formal
    if match is None:
        return None
    suffix = match.group("suffix")
    tokens = [token for token in suffix.removeprefix("-").split("-") if token]
    if product_id == "vllm":
        if any(token in {"x86_64", "aarch64"} for token in tokens):
            return None
        if any(
            re.fullmatch(r"cu[0-9]+|ubuntu[0-9]+", token) is None for token in tokens
        ):
            return None
        if (
            sum(token.startswith("cu") for token in tokens) > 1
            or sum(token.startswith("ubuntu") for token in tokens) > 1
        ):
            return None
    elif product_id == "vllm-ascend":
        if any(token not in {"310p", "a3", "a5", "openeuler"} for token in tokens):
            return None
        accelerator_tokens = set(tokens) & {"310p", "a3", "a5"}
        if len(accelerator_tokens) > 1 or tokens.count("openeuler") > 1:
            return None
    else:
        raise ValueError(f"unsupported runtime product {product_id!r}")
    try:
        version = Version(match.group("version"))
    except InvalidVersion:
        return None
    channel = (
        "nightly"
        if nightly is not None
        else "rc" if version.is_prerelease else "stable"
    )
    return {
        "tag": tag,
        "version": version,
        "channel": channel,
        "suffix": suffix,
        "tokens": tokens,
    }


def _select_runtime_tags(
    product: Mapping[str, object],
    tags: Sequence[str],
    *,
    max_minor_versions: int = -1,
) -> list[dict[str, str]]:
    """Select one published version per major/minor and keep its real variants."""

    product_id = _string(product, "id", "release product")
    if product.get("channel_policy") != "latest-stable-or-rc-or-nightly-per-minor":
        raise ValueError(f"{product_id}: unsupported channel policy")
    minor_limit = _minor_limit(max_minor_versions, f"{product_id} max_minor_versions")
    minimum, maximum = _version_window(product)
    parsed = [
        item
        for tag in sorted(set(tags))
        if (item := _parsed_runtime_tag(product_id, str(tag))) is not None
        and item["version"] >= minimum
        and (maximum is None or item["version"] <= maximum)
    ]
    by_minor: dict[tuple[int, int], list[dict[str, object]]] = {}
    for item in parsed:
        version = item["version"]
        assert isinstance(version, Version)
        by_minor.setdefault((version.major, version.minor), []).append(item)

    selected: list[dict[str, str]] = []
    selected_minors = sorted(by_minor)
    if minor_limit != -1:
        selected_minors = selected_minors[:minor_limit]
    for minor in selected_minors:
        values = by_minor[minor]
        chosen_channel = next(
            (
                channel
                for channel in ("stable", "rc", "nightly")
                if any(item["channel"] == channel for item in values)
            ),
            None,
        )
        if chosen_channel is None:
            continue
        chosen_version = max(
            item["version"] for item in values if item["channel"] == chosen_channel
        )
        for item in sorted(values, key=lambda value: str(value["tag"])):
            if item["channel"] == chosen_channel and item["version"] == chosen_version:
                selected.append(
                    {
                        "runtime_tag": str(item["tag"]),
                        "version": str(chosen_version),
                        "channel": chosen_channel,
                    }
                )
    return selected


def _blocked_problem(
    *, backend: str, capability: str, reason: str, repository: str, tag: str
) -> dict[str, object]:
    return {
        "backend": backend,
        "capability": capability,
        "reason": reason,
        "runtime": {"repository": repository, "tag": tag},
    }


def resolve_runtime_candidates(
    release: Mapping[str, object],
    *,
    tag_fixture: Mapping[str, object] | None = None,
    tag_loader: Callable[[str], Sequence[str]] | None = None,
    pr_default: bool = False,
) -> dict[str, object]:
    """Select formal Runtime references strictly from published Registry tags."""

    products = release.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("formal runtime selection requires release products")
    backends = _mapping(release.get("backends"), "platform backends")
    release_profile = _mapping(
        release.get("release_profile"), "selected release profile"
    )
    max_minor_versions = _minor_limit(
        release_profile.get("max_minor_versions"),
        "selected release profile max_minor_versions",
    )
    excluded_by_product = _mapping(
        release.get("excluded_upstream_variants"), "excluded upstream variants"
    )
    runtimes: list[dict[str, str]] = []
    problems: list[dict[str, object]] = []
    for index, raw_product in enumerate(products):
        product = _mapping(raw_product, f"release products[{index}]")
        product_id = _string(product, "id", f"release products[{index}]")
        repository = _string(product, "runtime_repository", product_id)
        selected = _select_runtime_tags(
            product,
            _repository_tags(
                repository, tag_fixture=tag_fixture, tag_loader=tag_loader
            ),
            max_minor_versions=max_minor_versions,
        )
        if not selected:
            raise ValueError(
                f"{product_id}: no Runtime Registry tags satisfy the version window"
            )
        for item in selected:
            tag = item["runtime_tag"]
            parsed = _parsed_runtime_tag(product_id, tag)
            assert parsed is not None
            tokens = set(parsed["tokens"])
            variant = (
                next(token for token in ("310p", "a3", "a5") if token in tokens)
                if tokens & {"310p", "a3", "a5"}
                else "a2" if product_id == "vllm-ascend" else "default"
            )
            excluded = excluded_by_product.get(product_id, [])
            if not isinstance(excluded, list) or not all(
                isinstance(item, str) for item in excluded
            ):
                raise ValueError(f"{product_id}: excluded variants must be a list")
            if variant in excluded:
                continue
            if product_id == "vllm-ascend" and "a5" in tokens:
                backend = "cann-a5"
                backend_policy = _mapping(backends.get(backend), f"backend {backend}")
                problems.append(
                    _blocked_problem(
                        backend=backend,
                        capability="Ascend A5 runtime",
                        reason=str(backend_policy.get("reason", "backend is blocked")),
                        repository=repository,
                        tag=tag,
                    )
                )
                continue
            runtimes.append(
                {
                    "product_id": product_id,
                    "runtime_repository": repository,
                    "runtime_tag": tag,
                    "runtime_ref": f"{repository}:{tag}",
                    "version": item["version"],
                    "channel": item["channel"],
                }
            )
    if pr_default:
        eligible = []
        for item in runtimes:
            if item["product_id"] != "vllm-ascend":
                continue
            parsed = _parsed_runtime_tag("vllm-ascend", item["runtime_tag"])
            if parsed is not None and not parsed["tokens"]:
                eligible.append(item)
        channel = next(
            (
                value
                for value in ("stable", "rc", "nightly")
                if any(item["channel"] == value for item in eligible)
            ),
            None,
        )
        if channel is None:
            raise ValueError("PR default selector found no Ascend A2 Ubuntu Runtime")
        pool = [item for item in eligible if item["channel"] == channel]
        selected_default = max(
            pool,
            key=lambda item: (
                Version(str(item["version"])),
                str(item["runtime_tag"]),
            ),
        )
        runtimes = [selected_default]
        problems = []

    document = {
        "kind": CANDIDATES_KIND,
        "schema_version": CANDIDATES_SCHEMA_VERSION,
        "runtimes": sorted(
            runtimes, key=lambda item: (item["product_id"], item["runtime_tag"])
        ),
        "references": sorted(item["runtime_ref"] for item in runtimes),
        "problems": sorted(
            problems,
            key=lambda item: (
                str(item["backend"]),
                str(item["runtime"]["repository"]),
                str(item["runtime"]["tag"]),
            ),
        ),
    }
    return validate_runtime_candidates(document)


def validate_runtime_candidates(value: object) -> dict[str, object]:
    document = _mapping(value, "runtime candidates")
    expected = {"kind", "schema_version", "runtimes", "references", "problems"}
    if set(document) != expected:
        raise ValueError("runtime candidates fields must be exact")
    if (
        document.get("kind") != CANDIDATES_KIND
        or document.get("schema_version") != CANDIDATES_SCHEMA_VERSION
    ):
        raise ValueError("runtime candidates contract is unsupported")
    runtimes = document.get("runtimes")
    references = document.get("references")
    problems = document.get("problems")
    if not isinstance(runtimes, list) or not runtimes:
        raise ValueError("runtime candidates must contain runtimes")
    if not isinstance(references, list) or not all(
        isinstance(item, str) for item in references
    ):
        raise ValueError("runtime candidate references must be a string list")
    expected_refs: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(runtimes):
        item = _mapping(raw, f"runtime candidates[{index}]")
        fields = {
            "product_id",
            "runtime_repository",
            "runtime_tag",
            "runtime_ref",
            "version",
            "channel",
        }
        if set(item) != fields:
            raise ValueError(f"runtime candidates[{index}] fields must be exact")
        reference = _string(item, "runtime_ref", f"runtime candidates[{index}]")
        expected_ref = (
            f"{_string(item, 'runtime_repository', reference)}:"
            f"{_string(item, 'runtime_tag', reference)}"
        )
        if reference != expected_ref:
            raise ValueError(
                f"{reference}: runtime candidate reference is inconsistent"
            )
        if reference in seen:
            raise ValueError(f"duplicate runtime candidate {reference}")
        seen.add(reference)
        if item.get("channel") not in {"stable", "rc", "nightly"}:
            raise ValueError(f"{reference}: invalid channel")
        Version(_string(item, "version", reference))
        expected_refs.append(reference)
    if sorted(references) != sorted(expected_refs):
        raise ValueError("runtime candidate references do not match runtimes")
    _validate_problems(problems)
    return document


def _manylinux_floor(value: str) -> tuple[int, int]:
    match = _MANYLINUX.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid manylinux policy {value!r}")
    return int(match.group(1)), int(match.group(2))


def _source_repositories(family: Mapping[str, object], context: str) -> dict[str, str]:
    values = _mapping(
        family.get("source_repositories"), f"{context}.source_repositories"
    )
    if set(values) != set(SUPPORTED_ARCHITECTURES):
        raise ValueError(f"{context}: source_repositories must define amd64 and arm64")
    return {
        architecture: _string(values, architecture, context)
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
            members = _mapping(values[reference], f"source image fixture {reference}")
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
        else _crane("digest", reference).strip()
    )
    if not isinstance(root_digest, str) or _DIGEST.fullmatch(root_digest) is None:
        raise ValueError(f"raw Builder {reference} has invalid digest")
    pinned_reference = f"{repository}@{root_digest}"
    raw_manifest = (
        manifest_loader(pinned_reference)
        if manifest_loader is not None
        else _crane("manifest", pinned_reference)
    )
    if isinstance(raw_manifest, str):
        try:
            raw_manifest = json.loads(raw_manifest)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"raw Builder manifest {reference} is malformed"
            ) from error
    manifest = _mapping(raw_manifest, f"raw Builder manifest {reference}")
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
            else _crane("config", pinned_reference)
        )
        if isinstance(raw_config, str):
            try:
                raw_config = json.loads(raw_config)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"raw Builder config {reference} is malformed"
                ) from error
        config = _mapping(raw_config, f"raw Builder config {reference}")
        if (
            config.get("os", "linux") != "linux"
            or config.get("architecture") != architecture
        ):
            raise ValueError(f"raw Builder {reference} is not linux/{architecture}")
        return root_digest
    matches: list[str] = []
    for raw_descriptor in descriptors:
        descriptor = _mapping(
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
    families = _mapping(release.get("builder_families"), "Builder families")
    values: list[dict[str, str]] = []
    tag_cache: dict[str, list[str]] = {}
    digest_cache: dict[tuple[str, str], str] = {}

    def tags(repository: str) -> list[str]:
        if repository not in tag_cache:
            tag_cache[repository] = _repository_tags(
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

    cuda = _mapping(families.get("cuda"), "CUDA Builder family")
    cuda_repositories = _source_repositories(cuda, "CUDA Builder family")
    cuda_manylinux = _string(cuda, "manylinux", "CUDA Builder family")
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

    ascend = _mapping(families.get("ascend"), "Ascend Builder family")
    ascend_repositories = _source_repositories(ascend, "Ascend Builder family")
    seen_ascend: set[tuple[str, str]] = set()
    variant_by_token = {"910b": "a2", "a3": "a3", "950": "a5"}
    for architecture, repository in ascend_repositories.items():
        for tag in tags(repository):
            match = _ASCEND_BUILDER_TAG.fullmatch(tag)
            if match is None or match.group("variant") == "310p":
                continue
            runtime = f"cann-{match.group('runtime')}"
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


def _variant(probe: Mapping[str, object]) -> tuple[str, str, str]:
    backend = _string(probe, "backend", "runtime probe")
    accelerator_runtime = _string(probe, "accelerator_runtime", "runtime probe")
    if backend == "cuda":
        version = accelerator_runtime.removeprefix("cuda-")
        return "cuda", "default", f"cu{version.replace('.', '')}"
    if backend.startswith("cann-"):
        variant = backend.removeprefix("cann-")
        version = accelerator_runtime.removeprefix("cann-")
        return "ascend", variant, f"cann{version.replace('.', '')}-{variant}"
    raise ValueError(f"unsupported runtime backend {backend!r}")


def _required_files(family: Mapping[str, object], variant: str) -> list[str]:
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
    return hashlib.sha256(core.canonical_bytes(payload)).hexdigest()[:12]


def _build_for_probe(
    probe: Mapping[str, object],
    candidates: Sequence[Mapping[str, str]],
    release: Mapping[str, object],
) -> dict[str, object]:
    accelerator, variant, build_group = _variant(probe)
    architecture = _string(probe, "cpu_arch", "runtime probe")
    runtime_value = _string(probe, "accelerator_runtime", "runtime probe")
    python_abi = _string(probe, "python_abi", "runtime probe")
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
    families = _mapping(release.get("builder_families"), "Builder families")
    family = _mapping(families.get(accelerator), f"Builder family {accelerator}")
    commands = family.get("required_commands")
    if not isinstance(commands, list) or not all(
        isinstance(item, str) and item for item in commands
    ):
        raise ValueError(f"Builder family {accelerator}: required_commands is invalid")
    required_files = _required_files(family, variant)
    build: dict[str, object] = {
        "id": f"{build_group}-{python_abi}-{architecture}",
        "product_id": _string(probe, "product_id", "runtime probe"),
        "build_group": build_group,
        "backend": _string(probe, "backend", "runtime probe"),
        "accelerator": accelerator,
        "accelerator_runtime": runtime_value,
        "variant": variant,
        "soc_version": _string(probe, "soc_version", "runtime probe"),
        "runtime_variant": build_group,
        "python_version": _string(probe, "python_version", "runtime probe"),
        "python_abi": python_abi,
        "manylinux": raw["manylinux"],
        "cpu_arch": architecture,
        "source_image": raw["source_image"],
        "source_image_digest": raw["source_image_digest"],
        "build_mode": "mirror",
        "recipe_revision": "",
        "sync_mode": "mirror",
    }
    build["recipe_revision"] = _mirror_revision(build, commands, required_files)
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
            _variant(probe)[0],
            _string(probe, "accelerator_runtime", "runtime probe"),
            _variant(probe)[1],
            _string(probe, "python_abi", "runtime probe"),
            _string(probe, "cpu_arch", "runtime probe"),
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
    backends = _mapping(release.get("backends"), "platform backends")
    for raw_probe in probes:
        probe = _mapping(raw_probe, "runtime probe")
        backend = _string(probe, "backend", "runtime probe")
        backend_policy = _mapping(backends.get(backend), f"backend {backend}")
        if backend_policy.get("status") == "blocked":
            continue
        build = _build_for_probe(probe, raw_candidates, release)
        existing = builds.get(str(build["id"]))
        if existing is not None and existing != build:
            raise ValueError(f"{build['id']}: raw Builder capability identity drift")
        builds[str(build["id"])] = build
    return sorted(builds.values(), key=lambda item: str(item["id"]))


def _runtime_probe_document(value: object) -> list[dict[str, Any]]:
    document = _mapping(value, "runtime probe")
    if (
        document.get("kind") != "ucm-runtime-probe"
        or document.get("schema_version")
        != runtime_contract.RUNTIME_PROBE_SCHEMA_VERSION
    ):
        raise ValueError("runtime probe has an unsupported contract")
    values = document.get("probes")
    if not isinstance(values, list) or not values:
        raise ValueError("runtime probe must contain probes")
    return [
        _mapping(item, f"runtime probes[{index}]") for index, item in enumerate(values)
    ]


def resolve_upstreams(
    release: Mapping[str, object],
    _legacy_builder_config: Mapping[str, object] | None = None,
    *,
    candidates: Mapping[str, object] | None = None,
    runtime_probe: Mapping[str, object] | None = None,
    tag_fixture: Mapping[str, object] | None = None,
    pinned_upstreams: list[str] | None = None,
    tag_loader: Callable[[str], Sequence[str]] | None = None,
    manifest_loader: Callable[[str], object] | None = None,
    config_loader: Callable[[str], object] | None = None,
    digest_loader: Callable[[str], str] | None = None,
    **legacy: object,
) -> dict[str, object]:
    """Resolve probed Registry runtimes into the formal Wheel union."""

    if pinned_upstreams:
        raise ValueError("opaque pinned runtime tags require runtime inspection")
    # Legacy callers may still pass former source-resolution options. They are
    # intentionally ignored: no value from them can influence Registry output.
    selected = validate_runtime_candidates(
        candidates
        if candidates is not None
        else resolve_runtime_candidates(
            release, tag_fixture=tag_fixture, tag_loader=tag_loader
        )
    )
    if runtime_probe is None and tag_fixture is not None:
        fixture_probe = tag_fixture.get("runtime_probe")
        if isinstance(fixture_probe, Mapping):
            runtime_probe = fixture_probe
    if runtime_probe is None:
        raise ValueError(
            "formal Registry selection requires per-member runtime probe results"
        )
    probes = _runtime_probe_document(runtime_probe)
    candidate_by_ref = {
        str(item["runtime_ref"]): item
        for item in selected["runtimes"]  # type: ignore[index]
    }
    observed_refs = {str(probe.get("runtime_ref")) for probe in probes}
    missing = sorted(set(candidate_by_ref) - observed_refs)
    extra = sorted(observed_refs - set(candidate_by_ref))
    if missing or extra:
        raise ValueError(
            f"runtime probes differ from candidates: missing={missing}, extra={extra}"
        )

    wheel_builds = resolve_probe_builds(
        release,
        probes,
        tag_fixture=tag_fixture,
        tag_loader=tag_loader,
        manifest_loader=manifest_loader,
        config_loader=config_loader,
        digest_loader=digest_loader,
    )
    builds_by_capability = {
        (
            str(build["backend"]),
            str(build["accelerator_runtime"]),
            str(build["soc_version"]),
            str(build["python_abi"]),
            str(build["cpu_arch"]),
        ): build
        for build in wheel_builds
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for probe in probes:
        grouped.setdefault(_string(probe, "runtime_ref", "runtime probe"), []).append(
            probe
        )

    products = {
        str(item["id"]): _mapping(item, "release product")
        for item in release["products"]  # type: ignore[index]
    }
    image_suffix = f"-ucm-{core._oci_tag_version(str(release['ucm_version']))}"
    runtimes: list[dict[str, object]] = []
    for reference, group in sorted(grouped.items()):
        candidate = candidate_by_ref[reference]
        first = group[0]
        invariant_fields = (
            "product_id",
            "repository",
            "tag",
            "runtime_digest",
            "backend",
            "accelerator_runtime",
            "soc_version",
            "python_version",
            "python_abi",
            "os_id",
            "os_version",
            "glibc_version",
        )
        for field in invariant_fields:
            if len({str(item.get(field, "")) for item in group}) != 1:
                raise ValueError(f"{reference}: Runtime members disagree on {field}")
        product_id = _string(first, "product_id", reference)
        product = products[product_id]
        _, variant, runtime_variant = _variant(first)
        architectures = sorted(_string(item, "cpu_arch", reference) for item in group)
        if len(set(architectures)) != len(architectures):
            raise ValueError(f"{reference}: duplicate probed architecture")
        member_references = {
            _string(item, "cpu_arch", reference): _string(
                item, "image_reference", reference
            )
            for item in group
        }
        wheel_ids: dict[str, str] = {}
        for item in group:
            key = (
                str(item["backend"]),
                str(item["accelerator_runtime"]),
                str(item["soc_version"]),
                str(item["python_abi"]),
                str(item["cpu_arch"]),
            )
            build = builds_by_capability.get(key)
            if build is None:
                raise ValueError(f"{reference}: no Wheel build for {key}")
            wheel_ids[str(item["cpu_arch"])] = str(build["id"])
        tag = _string(first, "tag", reference)
        runtime_id = re.sub(r"[^a-z0-9._-]+", "-", f"{product_id}-{tag}".lower()).strip(
            ".-"
        )
        target_tag = runtime_contract.project_runtime_image_tag(
            tag + image_suffix,
            tag_prefix=str(release.get("runtime_image_tag_prefix", "")),
            architectures=architectures,
        )
        runtimes.append(
            {
                "id": runtime_id,
                "product_id": product_id,
                "runtime_repository": _string(first, "repository", reference),
                "runtime_tag": tag,
                "runtime_digest": _string(first, "runtime_digest", reference),
                "runtime_variant": runtime_variant,
                "backend": _string(first, "backend", reference),
                "accelerator_runtime": _string(first, "accelerator_runtime", reference),
                "variant": variant,
                "soc_version": _string(first, "soc_version", reference),
                "python_version": _string(first, "python_version", reference),
                "python_abi": _string(first, "python_abi", reference),
                "os_id": _string(first, "os_id", reference),
                "os_version": _string(first, "os_version", reference),
                "glibc_version": first.get("glibc_version"),
                "architectures": architectures,
                "member_references": member_references,
                "wheel_build_ids": wheel_ids,
                "version": candidate["version"],
                "channel": candidate["channel"],
                "target_repository": product["target_repository"],
                "target_tag": target_tag,
            }
        )
    return validate_selection(
        {
            "kind": SELECTION_KIND,
            "schema_version": SELECTION_SCHEMA_VERSION,
            "wheel_builds": wheel_builds,
            "runtimes": runtimes,
            "problems": copy.deepcopy(selected["problems"]),
        }
    )


def _validate_problems(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("problems must be a list")
    seen: set[bytes] = set()
    for index, raw in enumerate(value):
        problem = _mapping(raw, f"problems[{index}]")
        if set(problem) != _PROBLEM_FIELDS:
            raise ValueError(f"problems[{index}] fields must be exact")
        for field in ("backend", "capability", "reason"):
            _string(problem, field, f"problems[{index}]")
        runtime = _mapping(problem.get("runtime"), f"problems[{index}].runtime")
        if set(runtime) != {"repository", "tag"}:
            raise ValueError(f"problems[{index}].runtime fields must be exact")
        _string(runtime, "repository", f"problems[{index}].runtime")
        _string(runtime, "tag", f"problems[{index}].runtime")
        identity = core.canonical_bytes(problem)
        if identity in seen:
            raise ValueError(f"duplicate problem at index {index}")
        seen.add(identity)


def validate_selection(value: object) -> dict[str, object]:
    selection = _mapping(value, "runtime selection")
    expected = {
        "kind",
        "schema_version",
        "wheel_builds",
        "runtimes",
        "problems",
    }
    if set(selection) != expected:
        raise ValueError("runtime selection fields must be exact")
    if (
        selection.get("kind") != SELECTION_KIND
        or selection.get("schema_version") != SELECTION_SCHEMA_VERSION
    ):
        raise ValueError("runtime selection contract is unsupported")
    builds = selection.get("wheel_builds")
    runtimes = selection.get("runtimes")
    if not isinstance(builds, list) or not builds:
        raise ValueError("runtime selection wheel_builds must be non-empty")
    if not isinstance(runtimes, list) or not runtimes:
        raise ValueError("runtime selection runtimes must be non-empty")
    build_ids: set[str] = set()
    for index, raw in enumerate(builds):
        build = _mapping(raw, f"wheel_builds[{index}]")
        if set(build) != _WHEEL_BUILD_FIELDS:
            raise ValueError(f"wheel_builds[{index}] fields must be exact")
        build_id = _string(build, "id", f"wheel_builds[{index}]")
        if build_id in build_ids:
            raise ValueError(f"duplicate Wheel build {build_id}")
        build_ids.add(build_id)
        if build.get("cpu_arch") not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"{build_id}: unsupported cpu_arch")
        if build.get("build_mode") != "mirror":
            raise ValueError(f"{build_id}: build_mode must be mirror")
        if build.get("sync_mode") not in {"mirror", "registry-only"}:
            raise ValueError(f"{build_id}: sync_mode is invalid")
        if re.fullmatch(r"cp[0-9]+", str(build.get("python_abi"))) is None:
            raise ValueError(f"{build_id}: malformed python_abi")
        if _DIGEST.fullmatch(str(build.get("source_image_digest"))) is None:
            raise ValueError(f"{build_id}: malformed source_image_digest")
        if re.fullmatch(r"[0-9a-f]{12}", str(build.get("recipe_revision"))) is None:
            raise ValueError(f"{build_id}: malformed recipe_revision")
    runtime_ids: set[str] = set()
    coordinates: set[tuple[str, str]] = set()
    targets: set[tuple[str, str]] = set()
    for index, raw in enumerate(runtimes):
        runtime = _mapping(raw, f"runtimes[{index}]")
        if set(runtime) != _RUNTIME_FIELDS:
            raise ValueError(f"runtimes[{index}] fields must be exact")
        runtime_id = _string(runtime, "id", f"runtimes[{index}]")
        if (
            runtime_id in runtime_ids
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", runtime_id) is None
        ):
            raise ValueError(f"duplicate or malformed runtime {runtime_id!r}")
        runtime_ids.add(runtime_id)
        coordinate = (
            _string(runtime, "runtime_repository", runtime_id),
            _string(runtime, "runtime_tag", runtime_id),
        )
        target = (
            _string(runtime, "target_repository", runtime_id),
            _string(runtime, "target_tag", runtime_id),
        )
        if coordinate in coordinates or target in targets:
            raise ValueError(f"{runtime_id}: duplicate runtime or target coordinate")
        coordinates.add(coordinate)
        targets.add(target)
        if _DIGEST.fullmatch(_string(runtime, "runtime_digest", runtime_id)) is None:
            raise ValueError(f"{runtime_id}: malformed runtime_digest")
        if runtime.get("channel") not in {
            "stable",
            "rc",
            "nightly",
            "pinned",
        }:
            raise ValueError(f"{runtime_id}: unsupported runtime channel")
        architectures = runtime.get("architectures")
        wheel_ids = runtime.get("wheel_build_ids")
        members = runtime.get("member_references")
        if (
            not isinstance(architectures, list)
            or not architectures
            or len(set(architectures)) != len(architectures)
            or not set(architectures) <= set(SUPPORTED_ARCHITECTURES)
        ):
            raise ValueError(f"{runtime_id}: invalid architectures")
        if not isinstance(wheel_ids, Mapping) or set(wheel_ids) != set(architectures):
            raise ValueError(f"{runtime_id}: wheel_build_ids must match architectures")
        if not isinstance(members, Mapping) or set(members) != set(architectures):
            raise ValueError(
                f"{runtime_id}: member_references must match architectures"
            )
        if any(str(value) not in build_ids for value in wheel_ids.values()):
            raise ValueError(f"{runtime_id}: unknown Wheel build")
        if not all(
            isinstance(value, str) and "@sha256:" in value for value in members.values()
        ):
            raise ValueError(f"{runtime_id}: member references must be digest-pinned")
    _validate_problems(selection.get("problems"))
    return selection
