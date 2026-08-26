"""Registry-only runtime selection and mirror Builder contracts."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
TAG_FIXTURE = RELEASE_ROOT / "tests" / "fixtures" / "catalog-registry.json"
sys.path.insert(0, str(RELEASE_ROOT))

builders = importlib.import_module("ucm_release.builders")
core = importlib.import_module("ucm_release.core")
policy = importlib.import_module("ucm_release.policy")
upstream = importlib.import_module("ucm_release.upstream")


def _policy(release_type: str = "stable") -> dict[str, object]:
    release = copy.deepcopy(
        policy.resolve(
            repository="release-org/unified-cache-management",
            version_override="0.7.60rc1",
            release_type=release_type,
        )
    )
    for product in release["products"]:
        product["minimum_version"] = "0"
        product.pop("maximum_version", None)
    return release


def _fixture() -> dict[str, object]:
    return core.load_json(TAG_FIXTURE)


def _selection(fixture: dict[str, object] | None = None) -> dict[str, object]:
    registry = fixture or _fixture()
    return upstream.resolve_upstreams(
        _policy(),
        candidates=upstream.resolve_runtime_candidates(_policy(), tag_fixture=registry),
        runtime_probe=registry["runtime_probe"],
        tag_fixture=registry,
    )


def _catalog(selection: dict[str, object] | None = None) -> dict[str, object]:
    return builders.catalog_from_selection(
        selection or _selection(), owner="release-org", formal_policy=_policy()
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_registry_tag_selection_is_per_minor_and_uses_inclusive_window() -> None:
    product = {
        "id": "vllm-ascend",
        "minimum_version": "0.22.0",
        "maximum_version": "0.26.0",
        "channel_policy": "latest-stable-or-rc-or-nightly-per-minor",
    }
    selected = upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "v0.21.0",
            "v0.22.1rc1",
            "v0.22.1rc1-a3",
            "v0.23.0rc1",
            "v0.23.0",
            "v0.23.0-a3",
            "nightly-releases-v0.24.0rc",
            "nightly-releases-v0.24.0rc-a3",
            "v0.25.0rc1",
            "nightly-releases-v0.25.1rc",
            "nightly-releases-v0.26.0rc",
            "v0.26.1rc1",
        ],
    )

    assert {(item["version"], item["channel"]) for item in selected} == {
        ("0.22.1rc1", "rc"),
        ("0.23.0", "stable"),
        ("0.24.0rc0", "nightly"),
        ("0.25.0rc1", "rc"),
        ("0.26.0rc0", "nightly"),
    }
    assert all(not item["runtime_tag"].startswith("releases/") for item in selected)


def test_registry_tag_selection_limits_first_actual_minors_and_keeps_variants() -> None:
    product = {
        "id": "vllm-ascend",
        "minimum_version": "0.23.0",
        "maximum_version": "0.27.0",
        "channel_policy": "latest-stable-or-rc-or-nightly-per-minor",
    }
    tags = [
        "v0.22.9",
        "v0.23.1",
        "v0.23.1-a3",
        "v0.23.2rc1",
        "v0.25.2rc1",
        "v0.25.2rc1-a3",
        "nightly-releases-v0.26.0rc",
        "nightly-releases-v0.26.0rc-a3",
    ]

    limited = upstream._select_runtime_tags(  # noqa: SLF001
        product, tags, max_minor_versions=2
    )
    unlimited = upstream._select_runtime_tags(  # noqa: SLF001
        product, tags, max_minor_versions=-1
    )

    assert limited == [
        {"runtime_tag": "v0.23.1", "version": "0.23.1", "channel": "stable"},
        {
            "runtime_tag": "v0.23.1-a3",
            "version": "0.23.1",
            "channel": "stable",
        },
        {
            "runtime_tag": "v0.25.2rc1",
            "version": "0.25.2rc1",
            "channel": "rc",
        },
        {
            "runtime_tag": "v0.25.2rc1-a3",
            "version": "0.25.2rc1",
            "channel": "rc",
        },
    ]
    assert {(item["version"], item["channel"]) for item in unlimited} == {
        ("0.23.1", "stable"),
        ("0.25.2rc1", "rc"),
        ("0.26.0rc0", "nightly"),
    }


@pytest.mark.parametrize("value", [-2, 0, True])
def test_registry_tag_selection_rejects_invalid_minor_limit(value: object) -> None:
    product = {
        "id": "vllm",
        "minimum_version": "0.23.0",
        "channel_policy": "latest-stable-or-rc-or-nightly-per-minor",
    }

    with pytest.raises(ValueError, match="must be -1 or an integer >= 1"):
        upstream._select_runtime_tags(  # noqa: SLF001
            product, ["v0.23.0"], max_minor_versions=value
        )


def test_candidate_selection_uses_selected_profile_minor_limit_per_product() -> None:
    release = _policy("nightly")
    tags = {
        "docker.io/vllm/vllm-openai": ["v0.22.1", "v0.24.0"],
        "quay.io/ascend/vllm-ascend": [
            "v0.21.0",
            "v0.21.0-a3",
            "v0.23.0",
        ],
    }

    candidates = upstream.resolve_runtime_candidates(
        release, tag_loader=lambda repository: tags[repository]
    )

    by_product = {
        product_id: {
            (item["version"], item["runtime_tag"])
            for item in candidates["runtimes"]
            if item["product_id"] == product_id
        }
        for product_id in ("vllm", "vllm-ascend")
    }
    assert by_product == {
        "vllm": {("0.22.1", "v0.22.1")},
        "vllm-ascend": {
            ("0.21.0", "v0.21.0"),
            ("0.21.0", "v0.21.0-a3"),
        },
    }


def test_candidates_are_real_registry_tags_and_filter_arch_310p_and_a5() -> None:
    candidates = upstream.resolve_runtime_candidates(_policy(), tag_fixture=_fixture())

    assert candidates["references"] == [
        "docker.io/vllm/vllm-openai:v0.22.1-cu129",
        "quay.io/ascend/vllm-ascend:v0.22.1rc1",
        "quay.io/ascend/vllm-ascend:v0.22.1rc1-a3",
    ]
    assert all("aarch64" not in reference for reference in candidates["references"])
    assert all("310p" not in reference for reference in candidates["references"])
    assert candidates["problems"] == [
        {
            "backend": "cann-a5",
            "capability": "Ascend A5 runtime",
            "reason": "A5 requires a dedicated UCM native implementation",
            "runtime": {
                "repository": "quay.io/ascend/vllm-ascend",
                "tag": "v0.22.1rc1-a5",
            },
        }
    ]


def test_excluded_variant_policy_is_the_runtime_filter_authority() -> None:
    release = _policy()
    release["excluded_upstream_variants"]["vllm-ascend"] = ["310p", "a3"]

    candidates = upstream.resolve_runtime_candidates(release, tag_fixture=_fixture())

    assert all("-a3" not in reference for reference in candidates["references"])
    assert all("-310p" not in reference for reference in candidates["references"])


def test_pr_default_selects_one_latest_ascend_a2_ubuntu_runtime() -> None:
    release = _policy()
    tags = {
        "docker.io/vllm/vllm-openai": ["v0.22.1", "v0.27.1"],
        "quay.io/ascend/vllm-ascend": [
            "v0.22.1rc1",
            "v0.23.0",
            "v0.23.0-a3",
            "v0.23.0-openeuler",
            "nightly-releases-v0.26.0rc",
        ],
    }

    candidates = upstream.resolve_runtime_candidates(
        release,
        tag_loader=lambda repository: tags[repository],
        pr_default=True,
    )

    assert candidates["references"] == ["quay.io/ascend/vllm-ascend:v0.23.0"]
    assert candidates["problems"] == []


def test_selection_is_source_free_and_wheels_are_the_runtime_union() -> None:
    selection = _selection()

    assert selection["schema_version"] == 3
    assert len(selection["runtimes"]) == 3
    assert {item["id"] for item in selection["wheel_builds"]} == {
        "cu129-cp312-amd64",
        "cu129-cp312-arm64",
        "cann900-a2-cp312-amd64",
        "cann900-a3-cp312-arm64",
    }
    assert not {
        "source_repository",
        "source_ref",
        "source_commit",
        "mooncake_version",
        "recipe",
    } & _all_keys(selection)
    assert all(item["build_mode"] == "mirror" for item in selection["wheel_builds"])
    assert all(
        "@sha256:" in reference
        for item in selection["runtimes"]
        for reference in item["member_references"].values()
    )


def test_each_runtime_member_has_one_exact_wheel_link() -> None:
    selection = _selection()
    builds = {item["id"]: item for item in selection["wheel_builds"]}

    for runtime in selection["runtimes"]:
        assert set(runtime["architectures"]) == set(runtime["wheel_build_ids"])
        for architecture, wheel_id in runtime["wheel_build_ids"].items():
            build = builds[wheel_id]
            assert build["backend"] == runtime["backend"]
            assert build["accelerator_runtime"] == runtime["accelerator_runtime"]
            assert build["python_abi"] == runtime["python_abi"]
            assert build["cpu_arch"] == architecture


def test_raw_member_digest_changes_only_its_mirror_identity() -> None:
    before = _selection()
    fixture = _fixture()
    reference = "docker.io/pytorch/manylinux2_28-builder:cuda12.9"
    fixture["source_image_members"][reference]["amd64"] = "sha256:" + "e" * 64
    after = _selection(fixture)
    baseline = {item["id"]: item for item in before["wheel_builds"]}
    changed = {item["id"]: item for item in after["wheel_builds"]}

    assert (
        baseline["cu129-cp312-amd64"]["recipe_revision"]
        != changed["cu129-cp312-amd64"]["recipe_revision"]
    )
    assert (
        baseline["cu129-cp312-arm64"]["recipe_revision"]
        == changed["cu129-cp312-arm64"]["recipe_revision"]
    )


def test_required_file_contract_changes_the_matching_mirror_identity() -> None:
    release = _policy()
    fixture = _fixture()
    before = upstream.resolve_upstreams(
        release,
        candidates=upstream.resolve_runtime_candidates(release, tag_fixture=fixture),
        runtime_probe=fixture["runtime_probe"],
        tag_fixture=fixture,
    )
    changed_policy = copy.deepcopy(release)
    changed_policy["builder_families"]["ascend"]["variant_required_files"]["a3"].append(
        "new-runtime-contract.so"
    )
    after = upstream.resolve_upstreams(
        changed_policy,
        candidates=upstream.resolve_runtime_candidates(
            changed_policy, tag_fixture=fixture
        ),
        runtime_probe=fixture["runtime_probe"],
        tag_fixture=fixture,
    )
    baseline = {item["id"]: item for item in before["wheel_builds"]}
    changed = {item["id"]: item for item in after["wheel_builds"]}

    assert (
        baseline["cann900-a3-cp312-arm64"]["recipe_revision"]
        != changed["cann900-a3-cp312-arm64"]["recipe_revision"]
    )
    assert (
        baseline["cann900-a2-cp312-amd64"]["recipe_revision"]
        == changed["cann900-a2-cp312-amd64"]["recipe_revision"]
    )


def test_single_platform_raw_builder_uses_its_verified_manifest_digest() -> None:
    digest = "sha256:" + "d" * 64
    pinned = "docker.io/pytorch/manylinuxaarch64-builder@" + digest
    seen: list[str] = []

    def manifest(reference: str) -> object:
        seen.append(reference)
        return {"mediaType": "application/vnd.docker.distribution.manifest.v2+json"}

    def config(reference: str) -> object:
        seen.append(reference)
        return {"os": "linux", "architecture": "arm64"}

    resolved = upstream._manifest_member_digest(  # noqa: SLF001
        "docker.io/pytorch/manylinuxaarch64-builder:cuda12.9",
        "arm64",
        tag_fixture=None,
        manifest_loader=manifest,
        config_loader=config,
        digest_loader=lambda _reference: digest,
    )

    assert resolved == digest
    assert seen == [pinned, pinned]


def test_single_platform_raw_builder_rejects_wrong_architecture() -> None:
    with pytest.raises(ValueError, match="is not linux/arm64"):
        upstream._manifest_member_digest(  # noqa: SLF001
            "docker.io/pytorch/manylinuxaarch64-builder:cuda12.9",
            "arm64",
            tag_fixture=None,
            manifest_loader=lambda _reference: {
                "mediaType": "application/vnd.oci.image.manifest.v1+json"
            },
            config_loader=lambda _reference: {
                "os": "linux",
                "architecture": "amd64",
            },
            digest_loader=lambda _reference: "sha256:" + "e" * 64,
        )


def test_raw_builder_selection_prefers_lowest_manylinux_floor() -> None:
    fixture = _fixture()
    repository = "quay.io/ascend/manylinux"
    tag = "9.0.0-910b-manylinux_2_34-py3.12"
    fixture["repositories"][repository]["pages"][0]["tags"].append(tag)
    fixture["source_image_members"][f"{repository}:{tag}"] = {
        "amd64": "sha256:" + "f" * 64
    }

    selection = _selection(fixture)
    build = next(
        item
        for item in selection["wheel_builds"]
        if item["backend"] == "cann-a2" and item["cpu_arch"] == "amd64"
    )
    assert build["manylinux"] == "manylinux_2_28"
    assert build["source_image"].endswith("9.0.0-910b-manylinux_2_28-py3.12")


def test_runtime_glibc_is_not_required_for_wheel_or_builder_planning() -> None:
    fixture = _fixture()
    for probe in fixture["runtime_probe"]["probes"]:
        probe["glibc_version"] = None

    selection = _selection(fixture)
    catalog = _catalog(selection)

    assert selection["wheel_builds"]
    assert all(runtime["glibc_version"] is None for runtime in selection["runtimes"])
    assert catalog["builders"]


def test_catalog_is_mirror_only_and_checks_ascend_variant_files() -> None:
    catalog = _catalog()
    by_backend = {item["backend"]: item for item in catalog["builders"]}

    assert catalog["schema_version"] == 3
    assert all(item["build_mode"] == "mirror" for item in catalog["builders"])
    assert by_backend["cann-a2"]["checks"]["required_files"] == ["acl.h"]
    assert by_backend["cann-a3"]["checks"]["required_files"] == [
        "acl.h",
        "libruntime.so",
    ]
    assert by_backend["cann-a2"]["checks"]["soc_version"] == "ascend910b1"
    assert by_backend["cann-a3"]["checks"]["variant"] == "a3"
    assert "mooncake" not in repr(catalog).lower()


def test_sync_is_append_only_and_registry_records_reopen_exactly() -> None:
    catalog = _catalog()
    first = catalog["builders"][0]
    existing = {first["target_repository"]: [first["target_tag"]]}
    sync = builders.compute_sync_plan(catalog, existing)

    assert len(sync["builders"]) == len(catalog["builders"]) - 1
    assert len(sync["matrix"]["include"]) == len(catalog["builders"])
    assert "deletions" not in sync
    assert all(item["build_mode"] == "mirror" for item in sync["builders"])

    labels = builders.builder_labels(first)
    record = builders.registry_builder_record(
        first["target_repository"],
        first["target_tag"],
        {
            "created": "2026-08-24T00:00:00Z",
            "config": {"Labels": labels},
        },
    )
    reopened = builders.catalog_from_registry_records([record])
    assert reopened["schema_version"] == 3
    assert (
        reopened["builders"][0]["source_image_digest"] == first["source_image_digest"]
    )


def test_final_catalog_binds_checked_labels_and_target_digests() -> None:
    catalog = _catalog()
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

    finalized = builders.finalize_catalog(catalog, observations)

    assert finalized["schema_version"] == 4
    assert all(
        item["target_digest"].startswith("sha256:") for item in finalized["builders"]
    )
    with pytest.raises(ValueError, match="unfinalized Catalog"):
        builders.compute_sync_plan(finalized, {})


def test_final_catalog_rejects_stale_builder_labels() -> None:
    catalog = _catalog()
    item = catalog["builders"][0]
    labels = builders.builder_labels(item)
    labels["io.ucm.builder.source_image_digest"] = "sha256:" + "f" * 64
    observations = {
        current["id"]: {
            "target_digest": f"sha256:{index + 1:064x}",
            "config": {
                "created": "2026-08-24T00:00:00Z",
                "config": {
                    "Labels": (
                        labels
                        if current["id"] == item["id"]
                        else builders.builder_labels(current)
                    )
                },
            },
        }
        for index, current in enumerate(catalog["builders"])
    }

    with pytest.raises(ValueError, match="label source_image_digest differs"):
        builders.finalize_catalog(catalog, observations)


def test_old_source_recipe_builder_labels_are_not_selected() -> None:
    builder = _catalog()["builders"][0]
    labels = builders.builder_labels(builder)
    labels["io.ucm.builder.schema"] = "1"

    assert (
        builders.registry_builder_record(
            builder["target_repository"],
            builder["target_tag"],
            {
                "created": "2026-08-24T00:00:00Z",
                "config": {"Labels": labels},
            },
        )
        is None
    )


def test_legacy_source_arguments_cannot_change_registry_output() -> None:
    fixture = _fixture()
    candidates = upstream.resolve_runtime_candidates(_policy(), tag_fixture=fixture)
    baseline = upstream.resolve_upstreams(
        _policy(),
        candidates=candidates,
        runtime_probe=fixture["runtime_probe"],
        tag_fixture=fixture,
    )
    ignored = upstream.resolve_upstreams(
        _policy(),
        candidates=candidates,
        runtime_probe=fixture["runtime_probe"],
        tag_fixture=fixture,
        snapshot_dir=Path("/does/not/exist"),
        source_commit_resolver=lambda *_args: "f" * 40,
    )

    assert ignored == baseline
