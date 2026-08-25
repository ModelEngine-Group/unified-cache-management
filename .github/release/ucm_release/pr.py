"""Resolve one probed PR Runtime through checked or raw mirror Builders."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import builders, runtime, upstream


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a mapping")
    return dict(value)


def _build_from_registry_record(record: Mapping[str, object]) -> dict[str, object]:
    excluded = {"target_repository", "target_tag", "created", "checked", "checks"}
    result = {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in excluded
    }
    return result


def _runtime_variant(probe: Mapping[str, object]) -> tuple[str, str]:
    backend = str(probe["backend"])
    if backend == "cuda":
        version = str(probe["accelerator_runtime"]).removeprefix("cuda-")
        return f"cu{version.replace('.', '')}", "default"
    if backend.startswith("cann-"):
        variant = backend.removeprefix("cann-")
        version = str(probe["accelerator_runtime"]).removeprefix("cann-")
        return f"cann{version.replace('.', '')}-{variant}", variant
    raise ValueError(f"unsupported probed backend {backend!r}")


def _failure(
    *, stage: str, reason: str, detail: str, probe: Mapping[str, object]
) -> dict[str, object]:
    return {
        "stage": stage,
        "reason": reason,
        "detail": detail,
        "probe_id": probe["probe_id"],
        "request_id": probe["request_id"],
        "runtime_ref": probe["runtime_ref"],
    }


def _match_for_build(
    probe: Mapping[str, object], build: Mapping[str, object]
) -> dict[str, object]:
    return {
        "probe_id": probe["probe_id"],
        "request_id": probe["request_id"],
        "runtime_ref": probe["runtime_ref"],
        "cpu_arch": probe["cpu_arch"],
        "wheel_id": build["id"],
        "capability": {
            key: probe[key]
            for key in (
                "backend",
                "accelerator_runtime",
                "soc_version",
                "python_version",
                "python_abi",
                "cpu_arch",
            )
        },
        "builder": {"id": build["id"]},
    }


def resolve_pr_request(
    formal_policy: Mapping[str, object],
    runtime_probe: Mapping[str, object],
    registry_inventory: Mapping[str, object],
    *,
    pr_number: int | str,
    author: str,
    run_id: int | str,
    raw_build_resolver: (
        Callable[[Sequence[Mapping[str, object]]], list[dict[str, object]]] | None
    ) = None,
) -> dict[str, object]:
    """Resolve a single opaque Runtime tag without consulting upstream source."""

    probe_document = _mapping(runtime_probe, "runtime probe")
    if (
        probe_document.get("kind") != "ucm-runtime-probe"
        or probe_document.get("schema_version") != runtime.RUNTIME_PROBE_SCHEMA_VERSION
    ):
        raise ValueError("runtime probe has an unsupported contract")
    raw_probes = probe_document.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError("runtime probe must contain probes")
    probes = [
        _mapping(item, f"runtime probes[{index}]")
        for index, item in enumerate(raw_probes)
    ]
    request_ids = {str(probe.get("request_id")) for probe in probes}
    if len(request_ids) != 1:
        raise ValueError("/ucm-build image accepts exactly one Runtime image")

    inventory = _mapping(registry_inventory, "Builder Registry inventory")
    if inventory.get("kind") != "ucm-builder-registry":
        raise ValueError("Builder Registry inventory has an unsupported contract")
    records = inventory.get("builders")
    if not isinstance(records, list):
        raise ValueError("Builder Registry inventory builders must be a list")

    backend_policies = _mapping(
        formal_policy.get("backends"), "formal backend policies"
    )
    supported: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for probe in probes:
        backend = str(probe["backend"])
        backend_policy = _mapping(
            backend_policies.get(backend), f"formal backend policy {backend}"
        )
        if backend_policy.get("status") == "blocked":
            failures.append(
                _failure(
                    stage="platform-policy",
                    reason="blocked-backend",
                    detail=str(backend_policy.get("reason", "backend is blocked")),
                    probe=probe,
                )
            )
        else:
            supported.append(probe)
    if failures:
        return {
            "kind": "ucm-pr-resolution",
            "schema_version": 2,
            "ok": False,
            "builder_matches": {
                "kind": "ucm-runtime-builder-matches",
                "schema_version": 1,
                "ok": False,
                "matches": [],
                "problems": failures,
            },
            "problems": failures,
        }

    checked = runtime.match_runtime_builders(
        {
            "kind": "ucm-runtime-probe",
            "schema_version": runtime.RUNTIME_PROBE_SCHEMA_VERSION,
            "probes": supported,
        },
        records,
    )
    existing_matches = list(checked["matches"])
    existing_probe_ids = {str(item["probe_id"]) for item in existing_matches}
    unresolved = [
        probe for probe in supported if str(probe["probe_id"]) not in existing_probe_ids
    ]

    raw_builds: list[dict[str, object]] = []
    raw_matches: list[dict[str, object]] = []
    for probe in unresolved:
        try:
            resolved = (
                raw_build_resolver([probe])
                if raw_build_resolver is not None
                else upstream.resolve_probe_builds(formal_policy, [probe])
            )
            if len(resolved) != 1:
                raise ValueError(f"expected one raw Builder, resolved {len(resolved)}")
            build = resolved[0]
            raw_builds.append(build)
            raw_matches.append(_match_for_build(probe, build))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(
                _failure(
                    stage="builder-match",
                    reason="missing-compatible-builder",
                    detail=str(error),
                    probe=probe,
                )
            )
    all_matches = existing_matches + raw_matches
    match_document = {
        "kind": "ucm-runtime-builder-matches",
        "schema_version": 1,
        "ok": not failures,
        "matches": all_matches,
        "problems": failures,
    }
    if failures:
        return {
            "kind": "ucm-pr-resolution",
            "schema_version": 2,
            "ok": False,
            "builder_matches": match_document,
            "problems": failures,
        }

    publication = runtime.project_pr_publication(
        probe_document,
        match_document,
        pr_number=pr_number,
        author=author,
        run_id=run_id,
    )
    existing_builds = {
        str(match["wheel_id"]): _build_from_registry_record(match["builder_record"])
        for match in existing_matches
    }
    build_map = {
        **existing_builds,
        **{str(build["id"]): build for build in raw_builds},
    }
    match_by_probe = {str(item["probe_id"]): item for item in all_matches}
    family = publication["families"][0]
    first = supported[0]
    runtime_variant, variant = _runtime_variant(first)
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
        if len({str(probe.get(field, "")) for probe in supported}) != 1:
            raise ValueError(f"PR Runtime members disagree on {field}")
    architectures = sorted(str(probe["cpu_arch"]) for probe in supported)
    selection = upstream.validate_selection(
        {
            "kind": upstream.SELECTION_KIND,
            "schema_version": upstream.SELECTION_SCHEMA_VERSION,
            "wheel_builds": sorted(
                build_map.values(), key=lambda item: str(item["id"])
            ),
            "runtimes": [
                {
                    "id": str(first["request_id"]),
                    "product_id": first["product_id"],
                    "runtime_repository": first["repository"],
                    "runtime_tag": first["tag"],
                    "runtime_digest": first["runtime_digest"],
                    "runtime_variant": runtime_variant,
                    "backend": first["backend"],
                    "accelerator_runtime": first["accelerator_runtime"],
                    "variant": variant,
                    "soc_version": first["soc_version"],
                    "python_version": first["python_version"],
                    "python_abi": first["python_abi"],
                    "os_id": first["os_id"],
                    "os_version": first["os_version"],
                    "glibc_version": first["glibc_version"],
                    "architectures": architectures,
                    "member_references": {
                        str(probe["cpu_arch"]): str(probe["image_reference"])
                        for probe in supported
                    },
                    "wheel_build_ids": {
                        str(probe["cpu_arch"]): str(
                            match_by_probe[str(probe["probe_id"])]["wheel_id"]
                        )
                        for probe in supported
                    },
                    "version": str(first["tag"]),
                    "channel": "pinned",
                    "target_repository": family["target_repository"],
                    "target_tag": family["target_tag"],
                }
            ],
            "problems": [],
        }
    )
    builder_catalog = builders.catalog_from_selection(
        selection,
        owner=str(formal_policy.get("repository", "")).split("/", 1)[0] or None,
        formal_policy=formal_policy,
    )
    catalog_by_id = {str(item["id"]): item for item in builder_catalog["builders"]}
    for match in existing_matches:
        selected = catalog_by_id[str(match["wheel_id"])]
        record = match["builder_record"]
        if (
            selected["target_repository"] != record["target_repository"]
            or selected["target_tag"] != record["target_tag"]
        ):
            raise ValueError("matched Registry Builder coordinate is not reproducible")
    return {
        "kind": "ucm-pr-resolution",
        "schema_version": 2,
        "ok": True,
        "selection": selection,
        "builder_catalog": builder_catalog,
        "builder_matches": match_document,
        "publication": publication,
        "problems": [],
    }
