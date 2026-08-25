"""Build staged GitHub Release manifests from validated build receipts."""

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
    if (
        dockerhub.get("enabled") is True
        and plan.get("release_kind", "publish") != "draft"
    ):
        repository, separator, tag = reference.rpartition(":")
        if not separator or not repository or not tag:
            raise ValueError(f"planned image reference is invalid: {reference!r}")
        namespace = dockerhub.get("namespace")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("enabled Docker Hub publication has no namespace")
        expected["dockerhub"] = f"{namespace}/{repository.rsplit('/', 1)[-1]}:{tag}"
    return expected


def build_artifacts_manifest(
    plan: dict[str, Any], wheels_root: Path, chart_root: Path
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
        images.append(
            {
                "id": item["id"],
                "family_id": item["family_id"],
                "wheel_id": item["wheel_id"],
                "cpu_arch": item["cpu_arch"],
                "runtime": copy.deepcopy(item["runtime"]),
                "planned_reference": member["reference"],
                "expected_targets": _expected_targets(plan, member["reference"]),
                "status": "building",
                "targets": [],
            }
        )

    manifest = {
        "kind": "ucm-release-manifest",
        "schema_version": 1,
        "release": {
            "git_tag": plan["git_tag"],
            "release_kind": plan.get("release_kind", "publish"),
            "is_prerelease": plan.get("is_prerelease", True),
            "version": plan["version"],
            "status": "artifacts-ready",
        },
        "chart": {
            "name": plan["chart"]["name"],
            "version": plan["chart"]["version"],
            "app_version": plan["chart"]["app_version"],
            "filename": chart_path.name,
            "sha256": chart_digest,
        },
        "wheels": wheels,
        "images": sorted(images, key=lambda item: str(item["id"])),
        "families": [
            {
                "id": family["id"],
                "runtime": copy.deepcopy(family["runtime"]),
                "planned_reference": family["published_reference"],
                "expected_targets": _expected_targets(
                    plan, family["published_reference"]
                ),
                "create_index": family["create_index"],
                "status": "building",
                "targets": [],
            }
            for family in sorted(families.values(), key=lambda item: str(item["id"]))
        ],
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


def finalize_manifest(
    manifest: dict[str, Any],
    receipts_root: Path,
    *,
    build_outcome: str,
    member_outcome: str,
    index_outcome: str,
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

    result["release"]["status"] = "images-failed" if failed else "complete"
    return result


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
                }
            )
            for wheel in family_wheels:
                row["wheels"][str(wheel["cpu_arch"])] = wheel

            ghcr_reference = family["expected_targets"].get("ghcr")
            if isinstance(ghcr_reference, str):
                ghcr_repositories.add(
                    _target_repository(ghcr_reference, "release family GHCR target")
                )
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
                "| Runtime capability | Upstream Runtime tags | Python ABI | Wheel downloads |",
                "| --- | --- | --- | --- |",
            ]
        )
        for key in sorted(rows, key=lambda item: (item[1], item[2], item[0])):
            row = rows[key]
            available_arches = set(row["wheels"])
            runtime_tags = []
            for family in sorted(row["families"], key=lambda item: item["tag"]):
                tag = family["tag"]
                arch_suffix = ""
                if family["architectures"] != available_arches:
                    architectures = ", ".join(
                        _architecture_label(item)
                        for item in sorted(family["architectures"])
                    )
                    arch_suffix = f" ({architectures} only)"
                runtime_tags.append(f"`{tag}`{arch_suffix}")
            wheel_links = []
            for _, wheel in sorted(row["wheels"].items()):
                filename = str(wheel["filename"])
                wheel_links.append(f"[{filename}]({asset_urls[filename]})")
            rendered.append(
                "| "
                + " | ".join(
                    (
                        _capability_label(row),
                        "<br>".join(runtime_tags),
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
        lines.append("Wheels, Chart, and images are building.")
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
    lines.extend(
        _render_product_tables(manifest, repository=repository, asset_urls=asset_urls)
    )
    return "\n".join(lines) + "\n"


def _artifacts(arguments: argparse.Namespace) -> None:
    plan = _mapping(_load_json(arguments.plan), "release plan")
    manifest, _ = build_artifacts_manifest(plan, arguments.wheels, arguments.chart)
    _write_json(arguments.output / "release-state.json", manifest)


def _finalize(arguments: argparse.Namespace) -> None:
    manifest = _mapping(_load_json(arguments.manifest), "release manifest")
    result = finalize_manifest(
        manifest,
        arguments.receipts,
        build_outcome=arguments.build_outcome,
        member_outcome=arguments.member_outcome,
        index_outcome=arguments.index_outcome,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("--plan", type=Path, required=True)
    artifacts.add_argument("--wheels", type=Path, required=True)
    artifacts.add_argument("--chart", type=Path, required=True)
    artifacts.add_argument("--output", type=Path, required=True)
    artifacts.set_defaults(func=_artifacts)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--receipts", type=Path, required=True)
    finalize.add_argument("--build-outcome", required=True)
    finalize.add_argument("--member-outcome", required=True)
    finalize.add_argument("--index-outcome", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(func=_finalize)
    notes = commands.add_parser("notes")
    notes.add_argument("--manifest", type=Path, required=True)
    notes.add_argument("--release", type=Path, required=True)
    notes.add_argument("--repository", required=True)
    notes.add_argument("--output", type=Path, required=True)
    notes.set_defaults(func=_notes)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.func(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
