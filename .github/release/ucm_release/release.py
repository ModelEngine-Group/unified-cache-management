"""Build staged Release state and the compact public cleanup manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
STATE_KIND = "ucm-release-state"
STATE_SCHEMA_VERSION = 2
PUBLIC_MANIFEST_KIND = "ucm-release-manifest"
PUBLIC_MANIFEST_SCHEMA_VERSION = 6
PUBLIC_MANIFEST_FILENAME = "release-manifest.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _one(paths: list[Path], context: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"{context} must resolve exactly once, found {len(paths)}")
    return paths[0]


def _validate_wheel_result_contract(result: dict[str, Any], result_path: Path) -> None:
    context = f"Wheel result {result_path}"
    if result.get("kind") != "ucm-wheel-result" or result.get("schema_version") != 2:
        raise ValueError(f"{context} must use ucm-wheel-result schema 2")

    audit_fields = {
        "platform_tags",
        "auditwheel_platform_tag",
        "glibc_versions",
        "glibc_floor",
        "external_libraries",
        "auditwheel_report",
    }
    missing_fields = sorted(audit_fields - result.keys())
    if missing_fields:
        raise ValueError(f"{context} is missing audit fields: {missing_fields}")

    platform_tags = result.get("platform_tags")
    if (
        not isinstance(platform_tags, list)
        or not platform_tags
        or any(not isinstance(value, str) or not value for value in platform_tags)
    ):
        raise ValueError(f"{context} has invalid platform_tags")
    auditwheel_platform = result.get("auditwheel_platform_tag")
    if not isinstance(auditwheel_platform, str) or not auditwheel_platform:
        raise ValueError(f"{context} has no auditwheel_platform_tag")

    glibc_versions = result.get("glibc_versions")
    if not isinstance(glibc_versions, list) or any(
        not isinstance(value, str) or not value for value in glibc_versions
    ):
        raise ValueError(f"{context} has invalid glibc_versions")
    glibc_floor = result.get("glibc_floor")
    if glibc_floor is not None and (
        not isinstance(glibc_floor, str) or glibc_floor not in glibc_versions
    ):
        raise ValueError(f"{context} has an invalid glibc_floor")
    external_libraries = result.get("external_libraries")
    if not isinstance(external_libraries, list) or any(
        not isinstance(value, str) or not value for value in external_libraries
    ):
        raise ValueError(f"{context} has invalid external_libraries")

    report = _mapping(result.get("auditwheel_report"), f"{context} auditwheel report")
    report_name = report.get("filename")
    if (
        not isinstance(report_name, str)
        or not report_name
        or report_name in {".", ".."}
        or Path(report_name).name != report_name
    ):
        raise ValueError(f"{context} has an invalid auditwheel report filename")
    report_path = result_path.parent / report_name
    if not report_path.is_file():
        raise ValueError(f"{context} has no matching auditwheel report file")
    report_digest = report.get("sha256")
    report_text = report.get("text")
    if not isinstance(report_digest, str) or _sha256(report_path) != report_digest:
        raise ValueError(f"{context} auditwheel report digest does not match")
    if (
        not isinstance(report_text, str)
        or report_path.read_text(encoding="utf-8") != report_text
    ):
        raise ValueError(f"{context} auditwheel report text does not match")


def _family_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    families = {
        str(item["id"]): _mapping(item, "release plan family")
        for item in _list(plan.get("families"), "release plan families")
    }
    if len(families) != len(plan["families"]):
        raise ValueError("release plan family IDs must be unique")
    return families


def _expected_targets(plan: dict[str, Any], reference: str) -> dict[str, str]:
    publish = _mapping(plan.get("publish"), "release plan publish")
    expected: dict[str, str] = {}
    if _mapping(publish.get("ghcr"), "release plan GHCR").get("enabled") is True:
        expected["ghcr"] = reference
    dockerhub = _mapping(publish.get("dockerhub"), "release plan Docker Hub")
    if dockerhub.get("enabled") is True:
        repository, separator, tag = reference.rpartition(":")
        if not separator or not repository or not tag:
            raise ValueError(f"planned image reference is invalid: {reference!r}")
        namespace = dockerhub.get("namespace")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("enabled Docker Hub publication has no namespace")
        expected["dockerhub"] = f"{namespace}/{repository.rsplit('/', 1)[-1]}:{tag}"
    return expected


def build_artifacts_manifest(
    plan: dict[str, Any],
    wheels_root: Path,
    chart_root: Path,
    *,
    actions_run_id: int,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Validate Wheel/Chart outputs and return the artifacts-ready manifest."""
    tasks = {
        str(item["id"]): _mapping(item, "release plan Wheel")
        for item in _list(plan.get("wheels"), "release plan Wheels")
    }
    if len(tasks) != len(plan["wheels"]):
        raise ValueError("release plan Wheel IDs must be unique")

    results: dict[str, dict[str, Any]] = {}
    wheel_files: dict[str, Path] = {}
    for result_path in sorted(wheels_root.rglob("wheel-result.json")):
        result = _mapping(_load_json(result_path), f"Wheel result {result_path}")
        _validate_wheel_result_contract(result, result_path)
        task_id = str(result.get("task_id", ""))
        if task_id not in tasks or task_id in results:
            raise ValueError(f"Wheel result {task_id!r} is unknown or duplicated")
        filename = str(result.get("filename", ""))
        wheel_path = result_path.parent / filename
        if not filename or not wheel_path.is_file():
            raise ValueError(f"Wheel result {task_id!r} has no matching file")
        task = tasks[task_id]
        expected = {
            "distribution": task["dist_name"],
            "version": task["wheel_version"],
            "python_abi": task["python_abi"],
            "cpu_arch": task["cpu_arch"],
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Wheel result {task_id!r} does not match its plan")
        results[task_id] = result
        wheel_files[task_id] = wheel_path

    if set(results) != set(tasks):
        missing = sorted(set(tasks) - set(results))
        raise ValueError(f"Wheel results do not cover the plan: {missing}")
    discovered_wheels = {path.resolve() for path in wheels_root.rglob("*.whl")}
    declared_wheels = {path.resolve() for path in wheel_files.values()}
    if discovered_wheels != declared_wheels:
        raise ValueError("Wheel files and result manifests must match exactly")
    filenames = [str(item["filename"]) for item in results.values()]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Wheel result filenames must be unique")

    chart_path = _one(sorted(chart_root.rglob("*.tgz")), "Chart package")
    checksums: list[tuple[str, str]] = []
    wheels: list[dict[str, Any]] = []
    for task_id in sorted(results):
        result = copy.deepcopy(results[task_id])
        task = tasks[task_id]
        digest = _sha256(wheel_files[task_id])
        result["id"] = result.pop("task_id")
        result["sha256"] = digest
        result["backend"] = task["backend"]
        result["runtime_variant"] = task["runtime_variant"]
        result["manylinux_builder"] = task["manylinux"]
        builder = _mapping(task.get("builder"), f"Wheel task {task_id} Builder")
        if (
            not isinstance(builder.get("digest"), str)
            or _DIGEST.fullmatch(builder["digest"]) is None
        ):
            raise ValueError(f"Wheel task {task_id} has no immutable Builder digest")
        result["builder"] = copy.deepcopy(builder)
        wheels.append(result)
        checksums.append((digest, str(result["filename"])))

    chart_digest = _sha256(chart_path)
    checksums.append((chart_digest, chart_path.name))
    families = _family_map(plan)
    images = []
    for raw in _list(plan.get("images"), "release plan Images"):
        item = _mapping(raw, "release plan Image")
        family = families.get(str(item["family_id"]))
        if family is None:
            raise ValueError(f"Image {item.get('id')!r} has no release family")
        member = _one(
            [
                value
                for value in family["members"]
                if value.get("image_id") == item.get("id")
            ],
            f"Image {item.get('id')!r} family member",
        )
        expected_targets = _expected_targets(plan, member["reference"])
        images.append(
            {
                "id": item["id"],
                "family_id": item["family_id"],
                "wheel_id": item["wheel_id"],
                "cpu_arch": item["cpu_arch"],
                "runtime": copy.deepcopy(item["runtime"]),
                "planned_reference": member["reference"],
                "expected_targets": expected_targets,
                "status": "building" if expected_targets else "not-requested",
                "targets": [],
            }
        )

    family_records = []
    for family in sorted(families.values(), key=lambda item: str(item["id"])):
        expected_targets = _expected_targets(plan, family["published_reference"])
        family_records.append(
            {
                "id": family["id"],
                "runtime": copy.deepcopy(family["runtime"]),
                "planned_reference": family["published_reference"],
                "expected_targets": expected_targets,
                "create_index": family["create_index"],
                "status": "building" if expected_targets else "not-requested",
                "targets": [],
            }
        )
    publish = copy.deepcopy(_mapping(plan.get("publish"), "release plan publish"))
    publication_requested = any(
        item["expected_targets"] for item in [*images, *family_records]
    ) or any(
        _mapping(publish.get(channel), f"release plan {channel}").get("enabled") is True
        for channel in ("pypi", "chart_oci")
    )
    if isinstance(actions_run_id, bool) or actions_run_id < 1:
        raise ValueError("Actions run ID must be a positive integer")
    chart_oci = _mapping(publish.get("chart_oci"), "release plan Chart OCI")
    chart_oci_reference = (
        f"{chart_oci['namespace']}/{plan['chart']['name']}:{plan['chart']['version']}"
        if chart_oci.get("enabled") is True
        else None
    )
    manifest = {
        "kind": STATE_KIND,
        "schema_version": STATE_SCHEMA_VERSION,
        "publish": publish,
        "release": {
            "git_tag": plan["git_tag"],
            "release_type": plan["release_type"],
            "release_kind": plan.get("release_kind", "publish"),
            "is_prerelease": plan.get("is_prerelease", True),
            "version": plan["version"],
            "actions_run_id": actions_run_id,
            "status": "artifacts-ready" if publication_requested else "complete",
        },
        "chart": {
            "name": plan["chart"]["name"],
            "version": plan["chart"]["version"],
            "app_version": plan["chart"]["app_version"],
            "filename": chart_path.name,
            "sha256": chart_digest,
            "oci_reference": chart_oci_reference,
        },
        "wheels": wheels,
        "images": sorted(images, key=lambda item: str(item["id"])),
        "families": family_records,
    }
    return manifest, sorted(checksums, key=lambda item: item[1])


def _receipt_map(receipts_root: Path, kind: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not receipts_root.exists():
        return result
    for path in sorted(receipts_root.rglob("*.json")):
        receipt = _mapping(_load_json(path), f"publication receipt {path}")
        if receipt.get("kind") != kind:
            continue
        if receipt.get("schema_version") != 1 or set(receipt) != {
            "kind",
            "schema_version",
            "id",
            "status",
            "targets",
        }:
            raise ValueError(f"publication receipt {path} has an invalid contract")
        identity = str(receipt.get("id", ""))
        if not identity or identity in result:
            raise ValueError(
                f"publication receipt {identity!r} is missing or duplicated"
            )
        result[identity] = receipt
    return result


def _validated_receipt_targets(
    receipt: dict[str, Any], expected: object, context: str
) -> list[dict[str, str]]:
    expected_targets = _mapping(expected, f"{context} expected targets")
    if receipt.get("status") not in {"published", "failed"}:
        raise ValueError(f"{context} has an invalid status")
    raw_targets = _list(receipt.get("targets"), f"{context} targets")
    targets: list[dict[str, str]] = []
    observed: dict[str, str] = {}
    for index, raw_target in enumerate(raw_targets):
        target = _mapping(raw_target, f"{context} targets[{index}]")
        if set(target) != {"channel", "reference", "digest"}:
            raise ValueError(f"{context} target fields must be exact")
        channel = target.get("channel")
        reference = target.get("reference")
        digest = target.get("digest")
        if (
            not isinstance(channel, str)
            or channel in observed
            or channel not in expected_targets
        ):
            raise ValueError(f"{context} has an unexpected or duplicate channel")
        if not isinstance(reference, str) or reference != expected_targets[channel]:
            raise ValueError(f"{context} target does not match its planned reference")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError(f"{context} target digest is invalid")
        observed[channel] = reference
        targets.append({"channel": channel, "reference": reference, "digest": digest})
    if receipt["status"] == "published" and set(observed) != set(expected_targets):
        raise ValueError(f"{context} does not cover its expected publication targets")
    return targets


def validate_member_receipts(
    plan: dict[str, Any], receipts_root: Path
) -> dict[str, dict[str, Any]]:
    """Require one complete member receipt for every planned image."""
    families = _family_map(plan)
    expected: dict[str, dict[str, str]] = {}
    for raw_image in _list(plan.get("images"), "release plan Images"):
        image = _mapping(raw_image, "release plan Image")
        image_id = str(image.get("id", ""))
        family = families.get(str(image.get("family_id", "")))
        if not image_id or family is None:
            raise ValueError("release plan Image has no family")
        member = _one(
            [
                item
                for item in _list(family.get("members"), "release family members")
                if _mapping(item, "release family member").get("image_id") == image_id
            ],
            f"Image {image_id!r} family member",
        )
        expected[image_id] = _expected_targets(plan, str(member["reference"]))
    receipts = _receipt_map(receipts_root, "ucm-image-member-receipt")
    if set(receipts) != set(expected):
        raise ValueError("member receipts do not exactly cover planned Images")
    for image_id, receipt in receipts.items():
        _validated_receipt_targets(
            receipt, expected[image_id], f"Image {image_id} receipt"
        )
        if receipt.get("status") != "published":
            raise ValueError(f"Image {image_id} receipt is not published")
    return receipts


def finalize_manifest(
    manifest: dict[str, Any],
    receipts_root: Path,
    *,
    build_outcome: str,
    member_outcome: str,
    index_outcome: str,
    pypi_outcome: str = "success",
    chart_oci_outcome: str = "success",
) -> dict[str, Any]:
    """Merge immutable publication receipts and close the image stage."""
    result = copy.deepcopy(manifest)
    members = _receipt_map(receipts_root, "ucm-image-member-receipt")
    indexes = _receipt_map(receipts_root, "ucm-image-index-receipt")
    expected_image_ids = {str(item["id"]) for item in result["images"]}
    expected_index_ids = {
        str(item["id"]) for item in result["families"] if item["create_index"]
    }
    if set(members) - expected_image_ids:
        raise ValueError("member receipts contain unknown Image IDs")
    if set(indexes) - expected_index_ids:
        raise ValueError("index receipts contain unknown family IDs")

    publication_items = [*result["images"], *result["families"]]
    publication_not_requested = all(
        "expected_targets" in item
        and not _mapping(
            item["expected_targets"],
            f"publication target contract {item.get('id')!r}",
        )
        for item in publication_items
    )
    if publication_not_requested:
        if (build_outcome, member_outcome, index_outcome) != (
            "skipped",
            "skipped",
            "skipped",
        ):
            raise ValueError("disabled image publication jobs must all be skipped")
        if members or indexes:
            raise ValueError("disabled image publication must not produce receipts")
        for item in publication_items:
            item["status"] = "not-requested"
            item["targets"] = []
        result["release"]["status"] = (
            "complete"
            if pypi_outcome == "success" and chart_oci_outcome == "success"
            else "publication-failed"
        )
        return result

    failed = build_outcome != "success" or member_outcome != "success"
    for image in result["images"]:
        receipt = members.get(str(image["id"]))
        targets = (
            _validated_receipt_targets(
                receipt, image.get("expected_targets"), f"Image {image['id']} receipt"
            )
            if receipt is not None
            else []
        )
        if receipt is not None and receipt.get("status") == "published":
            image["status"] = "published"
            image["targets"] = targets
        else:
            image["status"] = "failed"
            image["targets"] = targets
            failed = True

    if expected_index_ids and index_outcome != "success":
        failed = True
    for family in result["families"]:
        family_id = str(family["id"])
        if family["create_index"]:
            receipt = indexes.get(family_id)
            targets = (
                _validated_receipt_targets(
                    receipt,
                    family.get("expected_targets"),
                    f"family {family_id} receipt",
                )
                if receipt is not None
                else []
            )
            if receipt is not None and receipt.get("status") == "published":
                family["status"] = "published"
                family["targets"] = targets
            else:
                family["status"] = "failed"
                family["targets"] = targets
                failed = True
        else:
            members_for_family = [
                image
                for image in result["images"]
                if image["family_id"] == family["id"]
            ]
            if len(members_for_family) != 1:
                raise ValueError(f"single-arch family {family_id!r} is ambiguous")
            member = members_for_family[0]
            family["status"] = member["status"]
            family["targets"] = copy.deepcopy(member["targets"])
            failed = failed or member["status"] != "published"

    if failed:
        result["release"]["status"] = "images-failed"
    elif pypi_outcome != "success" or chart_oci_outcome != "success":
        result["release"]["status"] = "publication-failed"
    else:
        result["release"]["status"] = "complete"
    return result


def _published_references(records: list[dict[str, Any]], *, channel: str) -> list[str]:
    references = {
        str(target["reference"])
        for record in records
        if record.get("status") == "published"
        for target in _list(record.get("targets"), "publication targets")
        if _mapping(target, "publication target").get("channel") == channel
    }
    return sorted(references)


def build_public_manifest(
    state: dict[str, Any], release_document: dict[str, Any]
) -> dict[str, Any]:
    """Project completed internal state into the exact public schema-v6 surface."""
    release = _mapping(state.get("release"), "release state release")
    tag = release.get("git_tag")
    if release.get("status") != "complete":
        raise ValueError("public release manifest requires complete publication")
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

    asset_names = {
        str(_mapping(asset, "GitHub Release asset").get("name", ""))
        for asset in _list(release_document.get("assets"), "GitHub Release assets")
    }
    if "" in asset_names:
        raise ValueError("GitHub Release contains an unnamed asset")
    asset_names.add(PUBLIC_MANIFEST_FILENAME)

    images = [
        _mapping(item, "release state image")
        for item in _list(state.get("images"), "release state images")
    ]
    families = [
        _mapping(item, "release state family")
        for item in _list(state.get("families"), "release state families")
    ]
    indexes = [item for item in families if item.get("create_index") is True]
    chart = _mapping(state.get("chart"), "release state Chart")
    chart_oci = chart.get("oci_reference")
    if chart_oci is not None and (not isinstance(chart_oci, str) or not chart_oci):
        raise ValueError("release state Chart OCI reference is invalid")
    return {
        "kind": PUBLIC_MANIFEST_KIND,
        "schema_version": PUBLIC_MANIFEST_SCHEMA_VERSION,
        "tag": tag,
        "release_type": release_type,
        "actions_run_id": actions_run_id,
        "chart_oci": chart_oci,
        "runtime_images": {
            channel: {
                "members": _published_references(images, channel=channel),
                "indexes": _published_references(indexes, channel=channel),
            }
            for channel in ("ghcr", "dockerhub")
        },
        "github_release_assets": sorted(asset_names),
    }


def _github_asset_urls(
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
    chart = _mapping(manifest.get("chart"), "release manifest Chart")
    required.add(str(chart.get("filename", "")))
    missing = sorted(required - urls.keys())
    if missing:
        raise ValueError(f"GitHub Release is missing required assets: {missing}")
    return urls


def _target_repository(reference: str, context: str) -> str:
    repository, separator, tag = reference.rpartition(":")
    if not separator or not repository or not tag:
        raise ValueError(f"{context} is not a tagged OCI reference")
    return repository


def _ghcr_package_link(repository: str, target_repository: str) -> str:
    source_parts = repository.split("/")
    target_parts = target_repository.removeprefix("ghcr.io/").split("/")
    if (
        len(source_parts) != 2
        or any(re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None for part in source_parts)
        or len(target_parts) < 2
        or not target_repository.startswith("ghcr.io/")
        or target_parts[0].lower() != source_parts[0].lower()
    ):
        raise ValueError("GHCR target does not belong to the release repository owner")
    package = "/".join(target_parts[1:])
    url = f"https://github.com/{repository}/pkgs/container/{quote(package, safe='')}"
    return f"[`{target_repository}`]({url})"


def _architecture_label(cpu_arch: str) -> str:
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(cpu_arch, cpu_arch)


def _product_title(runtime_repository: str) -> tuple[int, str]:
    name = runtime_repository.rsplit("/", 1)[-1]
    known = {
        "vllm-openai": (0, "vLLM OpenAI"),
        "vllm-ascend": (1, "vLLM-Ascend"),
    }
    return known.get(name, (2, name))


def _capability_label(row: dict[str, Any]) -> str:
    accelerators = sorted(row["accelerators"])
    if len(accelerators) != 1:
        raise ValueError("Wheel capability maps to conflicting accelerator runtimes")
    accelerator = accelerators[0]
    name, separator, version = accelerator.partition("-")
    label = f"{name.upper()} {version}" if separator else accelerator
    backend = str(row["backend"])
    if backend.startswith("cann-"):
        label += f" / {backend.removeprefix('cann-').upper()}"
    return label


def _render_product_tables(
    manifest: dict[str, Any], *, repository: str, asset_urls: dict[str, str]
) -> list[str]:
    wheels = {str(item["id"]): item for item in manifest["wheels"]}
    images_by_family: dict[str, list[dict[str, Any]]] = {}
    for image in manifest["images"]:
        images_by_family.setdefault(str(image["family_id"]), []).append(image)
    families_by_product: dict[str, list[dict[str, Any]]] = {}
    for family in manifest["families"]:
        repository_name = str(family["runtime"]["repository"])
        families_by_product.setdefault(repository_name, []).append(family)

    rendered: list[str] = []
    product_order = sorted(
        families_by_product,
        key=lambda item: (*_product_title(item), item),
    )
    for runtime_repository in product_order:
        product_families = families_by_product[runtime_repository]
        rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        ghcr_repositories: set[str] = set()
        member_count = 0
        published_count = 0
        for family in product_families:
            family_id = str(family["id"])
            members = images_by_family[family_id]
            member_count += len(members)
            family_wheels = [wheels[str(member["wheel_id"])] for member in members]
            keys = {
                (
                    str(wheel["backend"]),
                    str(wheel["runtime_variant"]),
                    str(wheel["python_abi"]),
                )
                for wheel in family_wheels
            }
            if len(keys) != 1:
                raise ValueError(
                    f"release family {family_id!r} spans multiple Wheel capabilities"
                )
            key = next(iter(keys))
            row = rows.setdefault(
                key,
                {
                    "backend": key[0],
                    "python_abi": key[2],
                    "accelerators": set(),
                    "families": [],
                    "wheels": {},
                },
            )
            runtime = family["runtime"]
            row["accelerators"].add(str(runtime["accelerator_runtime"]))
            row["families"].append(
                {
                    "tag": str(runtime["tag"]),
                    "architectures": {str(member["cpu_arch"]) for member in members},
                    "target_tag": None,
                }
            )
            for wheel in family_wheels:
                row["wheels"][str(wheel["cpu_arch"])] = wheel

            actual_ghcr_references = [
                str(target["reference"])
                for target in family.get("targets", [])
                if target.get("channel") == "ghcr"
                and isinstance(target.get("reference"), str)
            ]
            if len(actual_ghcr_references) > 1:
                raise ValueError(
                    f"release family {family_id!r} has multiple GHCR targets"
                )
            ghcr_reference = (
                actual_ghcr_references[0]
                if actual_ghcr_references
                else family["expected_targets"].get("ghcr")
            )
            if isinstance(ghcr_reference, str):
                target_repository = _target_repository(
                    ghcr_reference, "release family GHCR target"
                )
                ghcr_repositories.add(target_repository)
                row["families"][-1]["target_tag"] = ghcr_reference.rpartition(":")[2]
            if family.get("status") == "published" and any(
                target.get("channel") == "ghcr" for target in family["targets"]
            ):
                published_count += 1

        _, title = _product_title(runtime_repository)
        rendered.extend([f"## {title}", "", f"Runtime: `{runtime_repository}`", ""])
        if len(ghcr_repositories) > 1:
            raise ValueError(f"{title} maps to multiple GHCR packages")
        if ghcr_repositories:
            ghcr_repository = next(iter(ghcr_repositories))
            package = _ghcr_package_link(repository, ghcr_repository)
            family_count = len(product_families)
            release_status = manifest["release"]["status"]
            if release_status == "complete":
                image_status = (
                    f"{family_count} image families / "
                    f"{member_count} architecture members"
                )
            elif release_status == "images-failed":
                image_status = (
                    f"{published_count}/{family_count} image families published"
                )
            else:
                image_status = (
                    f"building {family_count} image families / "
                    f"{member_count} architecture members"
                )
            rendered.extend([f"Images: {package} — {image_status}", ""])

        rendered.extend(
            [
                "| Runtime capability | "
                f"Upstream Runtime tags<br>{runtime_repository}： | "
                "Runtime tags"
                + (f"<br>{ghcr_repository}：" if ghcr_repositories else "")
                + " | Python ABI | Wheel |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for key in sorted(rows, key=lambda item: (item[1], item[2], item[0])):
            row = rows[key]
            available_arches = set(row["wheels"])
            upstream_runtime_tags = []
            packaged_runtime_tags = []
            for family in sorted(row["families"], key=lambda item: item["tag"]):
                tag = family["tag"]
                arch_suffix = ""
                if family["architectures"] != available_arches:
                    architectures = ", ".join(
                        _architecture_label(item)
                        for item in sorted(family["architectures"])
                    )
                    arch_suffix = f" ({architectures} only)"
                upstream_runtime_tags.append(f"`{tag}`{arch_suffix}")
                if family["target_tag"] is not None:
                    packaged_runtime_tags.append(
                        f"`{family['target_tag']}`{arch_suffix}"
                    )
            if not packaged_runtime_tags:
                packaged_runtime_tags.append("—")
            wheel_links = []
            wheel_architectures = [
                architecture
                for architecture in ("arm64", "amd64")
                if architecture in row["wheels"]
            ]
            wheel_architectures.extend(
                sorted(set(row["wheels"]) - set(wheel_architectures))
            )
            for architecture in wheel_architectures:
                wheel = row["wheels"][architecture]
                filename = str(wheel["filename"])
                wheel_links.append(
                    f"[{_architecture_label(architecture)}]({asset_urls[filename]})"
                )
            rendered.append(
                "| "
                + " | ".join(
                    (
                        _capability_label(row),
                        "<br>".join(upstream_runtime_tags),
                        "<br>".join(packaged_runtime_tags),
                        f"`{row['python_abi']}`",
                        "<br>".join(wheel_links),
                    )
                )
                + " |"
            )
        rendered.append("")
    return rendered


def render_notes(
    manifest: dict[str, Any], *, repository: str, asset_urls: dict[str, str]
) -> str:
    """Render compact, linked Release notes from the current manifest stage."""
    release = _mapping(manifest.get("release"), "release manifest release")
    lines = [
        f"Status: `{release['status']}`",
        "",
    ]
    if release["status"] == "release-open":
        lines.append("Release artifacts are building.")
        return "\n".join(lines) + "\n"

    chart = _mapping(manifest.get("chart"), "release manifest Chart")
    chart_filename = str(chart.get("filename", ""))
    chart_url = asset_urls.get(chart_filename)
    if not chart_filename or not chart_url:
        raise ValueError("Release notes require a Chart URL")
    lines.extend(
        [
            f"Wheels: {len(manifest.get('wheels', []))}",
            f"Chart: [{chart_filename}]({chart_url})",
            "",
        ]
    )
    if release["status"] == "artifacts-ready":
        lines.extend(
            ["> Wheels and Chart are available. Images are still building.", ""]
        )
    elif release["status"] == "images-failed":
        lines.extend(
            [
                "> Wheels and Chart remain available, but one or more images failed to publish.",
                "",
            ]
        )
    elif release["status"] == "publication-failed":
        lines.extend(
            [
                "> Wheels and Chart remain available, but one or more publication channels failed.",
                "",
            ]
        )
    lines.extend(
        _render_product_tables(manifest, repository=repository, asset_urls=asset_urls)
    )
    return "\n".join(lines) + "\n"


def _artifacts(arguments: argparse.Namespace) -> None:
    plan = _mapping(_load_json(arguments.plan), "release plan")
    manifest, _ = build_artifacts_manifest(
        plan,
        arguments.wheels,
        arguments.chart,
        actions_run_id=arguments.run_id,
    )
    _write_json(arguments.output / "release-state.json", manifest)


def _finalize(arguments: argparse.Namespace) -> None:
    manifest = _mapping(_load_json(arguments.manifest), "release manifest")
    result = finalize_manifest(
        manifest,
        arguments.receipts,
        build_outcome=arguments.build_outcome,
        member_outcome=arguments.member_outcome,
        index_outcome=arguments.index_outcome,
        pypi_outcome=arguments.pypi_outcome,
        chart_oci_outcome=arguments.chart_oci_outcome,
    )
    _write_json(arguments.output / "release-state.json", result)


def _notes(arguments: argparse.Namespace) -> None:
    manifest = _mapping(_load_json(arguments.manifest), "release manifest")
    release_document = _mapping(
        _load_json(arguments.release), "GitHub Release document"
    )
    asset_urls = _github_asset_urls(manifest, release_document)
    (arguments.output / "release-notes.md").write_text(
        render_notes(
            manifest,
            repository=arguments.repository,
            asset_urls=asset_urls,
        ),
        encoding="utf-8",
    )


def _manifest(arguments: argparse.Namespace) -> None:
    state = _mapping(_load_json(arguments.state), "release state")
    release_document = _mapping(
        _load_json(arguments.release), "GitHub Release document"
    )
    _write_json(
        arguments.output / PUBLIC_MANIFEST_FILENAME,
        build_public_manifest(state, release_document),
    )


def _members(arguments: argparse.Namespace) -> None:
    plan = _mapping(_load_json(arguments.plan), "release plan")
    validate_member_receipts(plan, arguments.receipts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("--plan", type=Path, required=True)
    artifacts.add_argument("--wheels", type=Path, required=True)
    artifacts.add_argument("--chart", type=Path, required=True)
    artifacts.add_argument("--run-id", type=int, required=True)
    artifacts.add_argument("--output", type=Path, required=True)
    artifacts.set_defaults(func=_artifacts)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--receipts", type=Path, required=True)
    finalize.add_argument("--build-outcome", required=True)
    finalize.add_argument("--member-outcome", required=True)
    finalize.add_argument("--index-outcome", required=True)
    finalize.add_argument("--pypi-outcome", required=True)
    finalize.add_argument("--chart-oci-outcome", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(func=_finalize)
    notes = commands.add_parser("notes")
    notes.add_argument("--manifest", type=Path, required=True)
    notes.add_argument("--release", type=Path, required=True)
    notes.add_argument("--repository", required=True)
    notes.add_argument("--output", type=Path, required=True)
    notes.set_defaults(func=_notes)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--state", type=Path, required=True)
    manifest.add_argument("--release", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(func=_manifest)
    members = commands.add_parser("members")
    members.add_argument("--plan", type=Path, required=True)
    members.add_argument("--receipts", type=Path, required=True)
    members.add_argument("--output", type=Path, required=True)
    members.set_defaults(func=_members)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.func(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
