"""Functional contract for the shared compact Release/PR plan."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
TAG_FIXTURE = RELEASE_ROOT / "tests" / "fixtures" / "catalog-registry.json"
sys.path.insert(0, str(RELEASE_ROOT))

builders = importlib.import_module("ucm_release.builders")
compact = importlib.import_module("ucm_release.compact")
core = importlib.import_module("ucm_release.core")
policy = importlib.import_module("ucm_release.policy")
upstream = importlib.import_module("ucm_release.upstream")


def _fixture_policy():
    formal = policy.resolve(
        repository="release-org/unified-cache-management",
        version_override="0.7.60rc1",
    )
    for product in formal["products"]:
        product["minimum_version"] = "0"
        product.pop("maximum_version", None)
    return formal


def _inputs():
    formal = _fixture_policy()
    selection = upstream.resolve_upstreams(
        formal,
        tag_fixture=core.load_json(TAG_FIXTURE),
    )
    catalog = builders.catalog_from_selection(
        selection, owner="release-org", formal_policy=formal
    )
    observations = {
        item["id"]: {
            "target_digest": f"sha256:{index + 1:064x}",
            "config": {
                "created": "2026-08-24T00:00:00Z",
                "config": {"Labels": builders.builder_labels(item)},
            },
        }
        for index, item in enumerate(catalog["builders"])
    }
    catalog = builders.finalize_catalog(catalog, observations)
    return formal, selection, catalog


def _plan():
    formal, selection, catalog = _inputs()
    return compact.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="release",
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _write_wheel_fixture(path: Path, metadata_tag: str) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            "fixture-1.0.dist-info/WHEEL",
            "\n".join(
                (
                    "Wheel-Version: 1.0",
                    "Generator: ucm-release-test",
                    "Root-Is-Purelib: false",
                    f"Tag: {metadata_tag}",
                    "",
                )
            ),
        )


def _wheel_result_task(cpu_arch: str) -> dict[str, str]:
    return {
        "id": f"cuda130-cp312-{cpu_arch}",
        "dist_name": "uc-manager-cuda-cu130",
        "wheel_version": "0.7.60rc1",
        "python_abi": "cp312",
        "cpu_arch": cpu_arch,
    }


def _write_auditwheel_report(
    wheel: Path,
    *,
    compatible_platform: str,
    constrained_platform: str | None = None,
    reported_filename: str | None = None,
    glibc_versions: tuple[str, ...] = ("GLIBC_2.17", "GLIBC_2.34"),
    external_libraries: tuple[str, ...] = ("libmetrics.so", "libascendcl.so"),
) -> Path:
    report = wheel.with_name(compact.AUDITWHEEL_REPORT)
    libraries = json.dumps({name: None for name in external_libraries}, indent=4)
    constraint = (
        f'This constrains the platform tag to "{constrained_platform}". In order '
        "to achieve a more compatible tag, you would need to recompile a new Wheel."
        if constrained_platform is not None
        else ""
    )
    report.write_text(
        "\n".join(
            (
                f"{reported_filename or wheel.name} is consistent with the following "
                f'platform tag: "{compatible_platform}".',
                "",
                "The wheel references external versioned symbols in these system-provided shared libraries: "
                f"libc.so.6 with versions {set(glibc_versions)!r}",
                "",
                constraint,
                "",
                "The following external shared libraries are required by the wheel:",
                libraries,
                "",
            )
        ),
        encoding="utf-8",
    )
    return report


def test_blocked_a5_projects_only_runtime_referenced_wheel_union() -> None:
    plan = _plan()

    assert (len(plan["wheels"]), len(plan["images"]), len(plan["families"])) == (
        4,
        4,
        3,
    )
    assert sum(bool(item["create_index"]) for item in plan["families"]) == 1
    assert sorted(len(item["members"]) for item in plan["families"]) == [1, 1, 2]
    assert len({item["id"] for item in plan["families"]}) == len(plan["families"])
    assert len(
        {(item["target_repository"], item["target_tag"]) for item in plan["families"]}
    ) == len(plan["families"])
    assert not any("a5" in item["id"] for item in plan["wheels"])
    assert not any("a5" in item["id"] for item in plan["families"])
    assert all(
        item["builder"]["digest"].startswith("sha256:") for item in plan["wheels"]
    )


def test_enabling_absent_a5_does_not_invent_registry_tasks() -> None:
    formal, selection, catalog = _inputs()
    formal = copy.deepcopy(formal)
    formal["backends"]["cann-a5"] = {
        "status": "supported",
        "platform": "ascend-a5",
        "distribution_template": "uc-manager-{runtime_variant}",
    }

    plan = compact.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="release",
    )

    assert (len(plan["wheels"]), len(plan["images"]), len(plan["families"])) == (
        4,
        4,
        3,
    )


def test_runtime_members_use_explicit_cp312_wheel_ids() -> None:
    plan = _plan()
    images = {item["id"]: item for item in plan["images"]}

    assert images["vllm-v0.22.1-cu129-amd64"]["wheel_id"] == "cu129-cp312-amd64"
    assert images["vllm-v0.22.1-cu129-arm64"]["wheel_id"] == "cu129-cp312-arm64"
    assert images["vllm-ascend-v0.22.1rc1-amd64"]["wheel_id"] == (
        "cann900-a2-cp312-amd64"
    )
    assert images["vllm-ascend-v0.22.1rc1-a3-arm64"]["wheel_id"] == (
        "cann900-a3-cp312-arm64"
    )
    assert all(item["runtime"]["python_abi"] == "cp312" for item in plan["images"])
    assert {item["runtime"]["os_id"] for item in plan["images"]} == {"ubuntu"}


def test_plan_defers_runtime_glibc_until_final_image_validation() -> None:
    formal, selection, catalog = _inputs()
    selection = copy.deepcopy(selection)
    for runtime in selection["runtimes"]:
        runtime["glibc_version"] = None

    plan = compact.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="release",
    )

    assert plan["images"]
    assert all(item["runtime"]["glibc_version"] is None for item in plan["images"])


def test_plan_projects_runtime_referenced_distributions_from_platform_policy() -> None:
    plan = _plan()

    assert plan["publish"]["pypi"]["distributions"] == [
        "uc-manager-cann900-a2",
        "uc-manager-cann900-a3",
        "uc-manager-cuda-cu129",
    ]
    assert {item["dist_name"] for item in plan["wheels"]} == set(
        plan["publish"]["pypi"]["distributions"]
    )


def test_cann_distribution_identity_keeps_runtime_versions_distinct() -> None:
    formal, _, _ = _inputs()
    backend = formal["backends"]["cann-a2"]

    assert compact._distribution(backend, "cann900-a2") == "uc-manager-cann900-a2"
    assert compact._distribution(backend, "cann910-a2") == "uc-manager-cann910-a2"


def test_plan_keeps_top_level_contract_without_problem_or_index_matrix() -> None:
    plan = _plan()
    assert set(plan) == {
        "kind",
        "route",
        "version",
        "image_version",
        "git_tag",
        "release_kind",
        "is_prerelease",
        "publish",
        "chart",
        "wheels",
        "images",
        "families",
        "wheel_matrix",
        "image_matrix",
    }
    assert "image_index_matrix" not in plan
    assert "problems" not in plan
    keys = _all_keys(plan)
    assert not {key for key in keys if "authority" in key or "mooncake" in key}
    assert "@sha256:" in json.dumps(plan)


def test_draft_coordinates_are_owned_by_the_plan_contract() -> None:
    formal, selection, catalog = _inputs()

    plan = compact.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="release",
        git_tag="draft/v0.7.62-4",
        release_kind="draft",
        is_prerelease=True,
        chart_version="0.7.62-draft.4",
    )

    assert plan["git_tag"] == "draft/v0.7.62-4"
    assert plan["release_kind"] == "draft"
    assert plan["is_prerelease"] is True
    assert plan["chart"]["version"] == "0.7.62-draft.4"


def test_family_members_are_the_authority_for_index_inputs() -> None:
    plan = _plan()
    family = next(
        item for item in plan["families"] if item["id"] == "vllm-v0.22.1-cu129"
    )

    assert family["members"] == [
        {
            "image_id": "vllm-v0.22.1-cu129-amd64",
            "cpu_arch": "amd64",
            "reference": (
                "ghcr.io/release-org/vllm-openai:"
                "v0.22.1-cu129-ucm-0.7.60rc1-r1-amd64"
            ),
        },
        {
            "image_id": "vllm-v0.22.1-cu129-arm64",
            "cpu_arch": "arm64",
            "reference": (
                "ghcr.io/release-org/vllm-openai:"
                "v0.22.1-cu129-ucm-0.7.60rc1-r1-arm64"
            ),
        },
    ]


def test_formal_all_plan_can_be_retagged_for_pr_publication() -> None:
    formal, selection, catalog = _inputs()
    plan = compact.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="pr",
    )
    retagged = compact.retag_pr_plan(
        plan, pr_number=12, author="Release-Author", run_id=345
    )

    assert all(
        family["target_tag"].startswith("pr-12-release-author-run-345-")
        for family in retagged["families"]
    )
    assert all(
        image["target_tag"]
        == next(
            family["target_tag"]
            for family in retagged["families"]
            if family["id"] == image["family_id"]
        )
        for image in retagged["images"]
    )


def test_single_arch_pr_build_has_one_member_and_no_index() -> None:
    formal, selection, catalog = _inputs()
    selection = copy.deepcopy(selection)
    runtime = next(
        item for item in selection["runtimes"] if item["id"] == "vllm-v0.22.1-cu129"
    )
    runtime["architectures"] = ["amd64"]
    runtime["member_references"] = {"amd64": runtime["member_references"]["amd64"]}
    runtime["wheel_build_ids"] = {"amd64": runtime["wheel_build_ids"]["amd64"]}
    runtime["channel"] = "pinned"
    selection["runtimes"] = [runtime]

    plan = compact.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="pr",
    )

    assert (len(plan["wheels"]), len(plan["images"]), len(plan["families"])) == (
        1,
        1,
        1,
    )
    family = plan["families"][0]
    assert family["create_index"] is False
    assert family["published_reference"] == (
        f"{family['target_repository']}:{family['target_tag']}"
    )
    assert family["members"] == [
        {
            "image_id": "vllm-v0.22.1-cu129-amd64",
            "cpu_arch": "amd64",
            "reference": family["published_reference"],
        }
    ]


def test_multiple_pr_runtime_tags_with_same_capability_reuse_wheels() -> None:
    formal, selection, catalog = _inputs()
    selection = copy.deepcopy(selection)
    first = next(item for item in selection["runtimes"] if item["product_id"] == "vllm")
    second = copy.deepcopy(first)
    second["id"] = "vllm-v0.22.1-cu129-ubuntu2404"
    second["runtime_tag"] = "v0.22.1-cu129-ubuntu2404"
    second["runtime_digest"] = "sha256:" + "d" * 64
    second["os_version"] = "24.04"
    second["target_tag"] = "v0.22.1-cu129-ubuntu2404-ucm-0.7.60rc1-r1"
    selected = [first, second]
    for runtime in selected:
        runtime["channel"] = "pinned"
    selection["runtimes"] = selected

    plan = compact.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="pr",
    )

    assert len(plan["wheels"]) == 2
    assert len(plan["images"]) == 4
    assert len(plan["families"]) == 2


def test_wheel_result_manifest_uses_actual_filename_tags(tmp_path: Path) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-linux_x86_64.whl"
    _write_wheel_fixture(wheel, "cp312-cp312-linux_x86_64")
    report = _write_auditwheel_report(
        wheel,
        compatible_platform="linux_x86_64",
        constrained_platform="manylinux_2_34_x86_64",
    )
    result = compact.record_wheel_result(task, wheel)
    assert result["task_id"] == "cuda130-cp312-amd64"
    assert result["filename"] == wheel.name
    assert result["platform_tags"] == ["linux_x86_64"]
    assert result["auditwheel_platform_tag"] == "linux_x86_64"
    assert result["glibc_versions"] == ["GLIBC_2.17", "GLIBC_2.34"]
    assert result["glibc_floor"] == "GLIBC_2.34"
    assert result["external_libraries"] == ["libascendcl.so", "libmetrics.so"]
    assert result["auditwheel_report"]["filename"] == report.name
    assert result["auditwheel_report"]["text"] == report.read_text(encoding="utf-8")
    assert len(result["auditwheel_report"]["sha256"]) == 64


def test_wheel_result_maps_arm64_to_aarch64_only(tmp_path: Path) -> None:
    task = _wheel_result_task("arm64")
    wheel = tmp_path / ("uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-linux_aarch64.whl")
    _write_wheel_fixture(wheel, "cp312-cp312-linux_aarch64")
    _write_auditwheel_report(
        wheel,
        compatible_platform="linux_aarch64",
        constrained_platform="manylinux_2_34_aarch64",
    )

    result = compact.record_wheel_result(task, wheel)

    assert result["cpu_arch"] == "arm64"
    assert result["platform_tags"] == ["linux_aarch64"]
    assert result["auditwheel_platform_tag"] == "linux_aarch64"


def test_wheel_result_rejects_additional_compressed_abis(tmp_path: Path) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / (
        "uc_manager_cuda_cu130-0.7.60rc1-cp311.cp312-cp311.cp312-linux_x86_64.whl"
    )
    metadata_tag = "cp311.cp312-cp311.cp312-linux_x86_64"
    _write_wheel_fixture(wheel, metadata_tag)
    _write_auditwheel_report(wheel, compatible_platform="linux_x86_64")

    with pytest.raises(ValueError, match="Wheel ABI"):
        compact.record_wheel_result(task, wheel)


def test_wheel_result_rejects_report_for_a_different_wheel(tmp_path: Path) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-linux_x86_64.whl"
    _write_wheel_fixture(wheel, "cp312-cp312-linux_x86_64")
    _write_auditwheel_report(
        wheel,
        compatible_platform="linux_x86_64",
        reported_filename="different-1.0-cp312-cp312-linux_x86_64.whl",
    )

    with pytest.raises(ValueError, match="different Wheel"):
        compact.record_wheel_result(task, wheel)


@pytest.mark.parametrize(
    "filename",
    [
        "wrong-0.7.60rc1-cp312-cp312-linux_x86_64.whl",
        "uc_manager_cuda_cu130-0.7.60rc1-cp311-cp311-linux_x86_64.whl",
        "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-linux_aarch64.whl",
    ],
)
def test_wheel_result_rejects_distribution_abi_and_arch_drift(
    tmp_path: Path, filename: str
) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / filename
    parsed_tag = "-".join(filename.removesuffix(".whl").rsplit("-", 3)[-3:])
    _write_wheel_fixture(wheel, parsed_tag)
    _write_auditwheel_report(wheel, compatible_platform="manylinux_2_34_x86_64")
    with pytest.raises(ValueError):
        compact.record_wheel_result(task, wheel)


@pytest.mark.parametrize(
    ("task_id", "wheel_platform", "compatible_platform"),
    [
        ("cuda130-cp312-amd64", "linux_amd64", "manylinux_2_34_x86_64"),
        ("cuda130-cp312-arm64", "linux_arm64", "manylinux_2_34_aarch64"),
        (
            "cuda130-cp312-amd64",
            "manylinux_2_28_x86_64",
            "manylinux_2_28_x86_64",
        ),
    ],
)
def test_wheel_result_rejects_oci_aliases_and_premature_manylinux_tags(
    tmp_path: Path,
    task_id: str,
    wheel_platform: str,
    compatible_platform: str,
) -> None:
    task = _wheel_result_task(task_id.rsplit("-", 1)[-1])
    wheel = tmp_path / (
        f"{task['dist_name'].replace('-', '_')}-{task['wheel_version']}-"
        f"cp312-cp312-{wheel_platform}.whl"
    )
    _write_wheel_fixture(wheel, f"cp312-cp312-{wheel_platform}")
    _write_auditwheel_report(wheel, compatible_platform=compatible_platform)

    with pytest.raises(ValueError, match="Wheel platform"):
        compact.record_wheel_result(task, wheel)


def test_wheel_result_rejects_filename_metadata_tag_drift(tmp_path: Path) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-linux_x86_64.whl"
    _write_wheel_fixture(wheel, "cp312-cp312-manylinux_2_28_x86_64")
    _write_auditwheel_report(wheel, compatible_platform="manylinux_2_34_x86_64")

    with pytest.raises(ValueError, match="filename and WHEEL metadata"):
        compact.record_wheel_result(task, wheel)
