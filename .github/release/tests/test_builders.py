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


def _selector(version: str, tag: str | None = None) -> dict[str, str | None]:
    return {
        "raw": version if tag is None else f"{version}@{tag}",
        "version": version,
        "tag": tag,
    }


def _policy(release_type: str = "stable") -> dict[str, object]:
    resolved = copy.deepcopy(
        policy.resolve(
            repository="release-org/unified-cache-management",
            version_override="0.7.60rc1",
            release_type=release_type,
        )
    )
    selectors = {
        "vllm": [_selector("0.22.1")],
        "vllm-ascend": [_selector("0.22.1")],
    }
    resolved["runtime_selectors"] = copy.deepcopy(selectors)
    for product in resolved["products"]:
        product["runtime_selectors"] = copy.deepcopy(selectors[product["id"]])
    return resolved


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


def test_registry_tag_selection_uses_version_ranges_and_all_winner_variants() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [
            _selector("0.23"),
            _selector("0.24"),
        ],
    }

    selected = upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "v0.23.0rc1",
            "v0.23.0",
            "v0.23.0-a3",
            "nightly-releases-v0.24.0rc",
            "nightly-releases-v0.24.0rc-a3",
        ],
    )

    assert selected == [
        {"runtime_tag": "v0.23.0", "version": "0.23.0", "channel": "stable"},
        {"runtime_tag": "v0.23.0-a3", "version": "0.23.0", "channel": "stable"},
        {
            "runtime_tag": "nightly-releases-v0.24.0rc",
            "version": "0.24.0rc",
            "channel": "nightly",
        },
        {
            "runtime_tag": "nightly-releases-v0.24.0rc-a3",
            "version": "0.24.0rc",
            "channel": "nightly",
        },
    ]


def test_explicit_runtime_tag_is_range_checked_without_expansion() -> None:
    product = {
        "id": "vllm",
        "runtime_selectors": [_selector("0.29", "v0.29.0rc1-cu129")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product, ["v0.29.0rc1-cu129", "v0.29.0rc2"]
    ) == [
        {
            "runtime_tag": "v0.29.0rc1-cu129",
            "version": "0.29.0rc1",
            "channel": "rc",
        }
    ]


def test_explicit_runtime_tag_flows_through_candidate_contract() -> None:
    release = _policy()
    vllm = next(product for product in release["products"] if product["id"] == "vllm")
    vllm["runtime_selectors"] = [_selector("0.29", "v0.29.0rc1-cu129")]
    tags = {
        "docker.io/vllm/vllm-openai": [
            "v0.29.0rc1-cu129",
            "v0.29.0rc2",
        ],
        "quay.io/ascend/vllm-ascend": ["v0.22.1rc1"],
    }

    candidates = upstream.resolve_runtime_candidates(
        release, tag_loader=lambda repository: tags[repository]
    )

    pinned = next(
        item for item in candidates["runtimes"] if item["product_id"] == "vllm"
    )
    assert pinned["runtime_tag"] == "v0.29.0rc1-cu129"
    assert pinned["version"] == "0.29.0rc1"
    assert pinned["channel"] == "rc"


def test_patch_selector_does_not_merge_rc_or_other_patch_versions() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.25.1")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        ["v0.25.1", "v0.25.1-a3", "v0.25.1rc1", "v0.25.10"],
    ) == [
        {"runtime_tag": "v0.25.1", "version": "0.25.1", "channel": "stable"},
        {"runtime_tag": "v0.25.1-a3", "version": "0.25.1", "channel": "stable"},
    ]


def test_minor_selector_prefers_channel_before_newer_version() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.27")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "v0.27.0",
            "v0.27.0-a3",
            "v0.27.1rc2",
            "nightly-releases-v0.27.2rc-a3",
        ],
    ) == [
        {"runtime_tag": "v0.27.0", "version": "0.27.0", "channel": "stable"},
        {
            "runtime_tag": "v0.27.0-a3",
            "version": "0.27.0",
            "channel": "stable",
        },
    ]


def test_minor_selector_uses_highest_rc_and_only_its_variants() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.26")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "v0.26.0rc1",
            "v0.26.0rc1-openeuler",
            "v0.26.0rc2",
            "v0.26.0rc2-a3",
            "nightly-releases-v0.26.1rc-a3",
        ],
    ) == [
        {"runtime_tag": "v0.26.0rc2", "version": "0.26.0rc2", "channel": "rc"},
        {
            "runtime_tag": "v0.26.0rc2-a3",
            "version": "0.26.0rc2",
            "channel": "rc",
        },
    ]


def test_minor_selector_uses_highest_nightly_and_only_its_variants() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.25")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "nightly-releases-v0.25.0rc",
            "nightly-releases-v0.25.0rc-a3",
            "nightly-releases-v0.25.1rc",
            "nightly-releases-v0.25.1rc-a3",
            "nightly-releases-v0.25.1rc-openeuler",
        ],
    ) == [
        {
            "runtime_tag": "nightly-releases-v0.25.1rc",
            "version": "0.25.1rc",
            "channel": "nightly",
        },
        {
            "runtime_tag": "nightly-releases-v0.25.1rc-a3",
            "version": "0.25.1rc",
            "channel": "nightly",
        },
        {
            "runtime_tag": "nightly-releases-v0.25.1rc-openeuler",
            "version": "0.25.1rc",
            "channel": "nightly",
        },
    ]


@pytest.mark.parametrize(
    ("selector", "tags", "message"),
    [
        (
            _selector("0.27.1"),
            [],
            "matches the version range",
        ),
        (
            _selector("0.27.1", "v0.27.1"),
            [],
            "not published",
        ),
        (
            _selector("0.27.1", "custom"),
            ["custom"],
            "does not match product grammar",
        ),
        (
            _selector("0.27.1", "v0.28.0"),
            ["v0.28.0"],
            "outside its version range",
        ),
    ],
)
def test_runtime_selector_fails_closed(
    selector: dict[str, object], tags: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        upstream._select_runtime_tags(  # noqa: SLF001
            {"id": "vllm", "runtime_selectors": [selector]}, tags
        )


def test_all_release_profiles_use_the_same_runtime_selectors() -> None:
    assert {
        release_type: _policy(release_type)["runtime_selectors"]
        for release_type in policy.RELEASE_TYPES
    } == {
        release_type: _policy()["runtime_selectors"]
        for release_type in policy.RELEASE_TYPES
    }


def test_candidate_resolution_selects_each_minor_independently() -> None:
    release = _policy()
    selectors = {
        "vllm": [_selector("0.26"), _selector("0.27")],
        "vllm-ascend": [
            _selector("0.24"),
            _selector("0.25"),
            _selector("0.26"),
        ],
    }
    for product in release["products"]:
        product["runtime_selectors"] = selectors[product["id"]]
    tags = {
        "docker.io/vllm/vllm-openai": [
            "v0.26.0",
            "v0.26.1",
            "v0.26.1-cu129",
            "v0.27.0rc2",
            "v0.27.1rc1",
        ],
        "quay.io/ascend/vllm-ascend": [
            "nightly-releases-v0.24.0rc",
            "nightly-releases-v0.24.0rc-a3",
            "nightly-releases-v0.25.0rc",
            "nightly-releases-v0.25.1rc",
            "nightly-releases-v0.25.1rc-a3",
            "v0.26.0rc1",
            "v0.26.0rc1-a3",
            "nightly-releases-v0.26.1rc-a3",
        ],
    }

    candidates = upstream.resolve_runtime_candidates(
        release, tag_loader=lambda repository: tags[repository]
    )

    assert candidates["references"] == [
        "docker.io/vllm/vllm-openai:v0.26.1",
        "docker.io/vllm/vllm-openai:v0.26.1-cu129",
        "docker.io/vllm/vllm-openai:v0.27.1rc1",
        "quay.io/ascend/vllm-ascend:nightly-releases-v0.24.0rc",
        "quay.io/ascend/vllm-ascend:nightly-releases-v0.24.0rc-a3",
        "quay.io/ascend/vllm-ascend:nightly-releases-v0.25.1rc",
        "quay.io/ascend/vllm-ascend:nightly-releases-v0.25.1rc-a3",
        "quay.io/ascend/vllm-ascend:v0.26.0rc1",
        "quay.io/ascend/vllm-ascend:v0.26.0rc1-a3",
    ]
    assert {
        (item["product_id"], item["version"], item["channel"])
        for item in candidates["runtimes"]
    } == {
        ("vllm", "0.26.1", "stable"),
        ("vllm", "0.27.1rc1", "rc"),
        ("vllm-ascend", "0.24.0rc", "nightly"),
        ("vllm-ascend", "0.25.1rc", "nightly"),
        ("vllm-ascend", "0.26.0rc1", "rc"),
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


def test_excluded_winner_does_not_fall_back_to_an_older_runtime() -> None:
    release = _policy()
    for product in release["products"]:
        product["runtime_selectors"] = [_selector("0.27")]
    tags = {
        "docker.io/vllm/vllm-openai": ["v0.27.0"],
        "quay.io/ascend/vllm-ascend": ["v0.27.0", "v0.27.1-310p"],
    }

    with pytest.raises(ValueError, match="winning Runtime version"):
        upstream.resolve_runtime_candidates(
            release, tag_loader=lambda repository: tags[repository]
        )


def test_product_with_only_blocked_variants_fails_selection() -> None:
    release = _policy()
    for product in release["products"]:
        product["runtime_selectors"] = [_selector("0.27")]
    tags = {
        "docker.io/vllm/vllm-openai": ["v0.27.0"],
        "quay.io/ascend/vllm-ascend": ["v0.27.0", "v0.27.1-a5"],
    }

    with pytest.raises(ValueError, match="vllm-ascend: no publishable Runtime"):
        upstream.resolve_runtime_candidates(
            release, tag_loader=lambda repository: tags[repository]
        )


def test_pr_default_selects_one_latest_ascend_a2_ubuntu_runtime() -> None:
    release = _policy()
    tags = {
        "docker.io/vllm/vllm-openai": ["v0.22.1", "v0.27.1"],
        "quay.io/ascend/vllm-ascend": [
            "v0.22.1rc1",
            "v0.26.0",
            "v0.26.0-a3",
            "v0.26.0-openeuler",
            "nightly-releases-v0.27.0rc",
        ],
    }

    candidates = upstream.resolve_runtime_candidates(
        release,
        tag_loader=lambda repository: tags[repository],
        pr_default=True,
    )

    assert candidates["references"] == ["quay.io/ascend/vllm-ascend:v0.26.0"]
    assert candidates["problems"] == []


def test_pr_default_does_not_require_unrelated_vllm_candidates() -> None:
    release = _policy()
    tags = {
        "docker.io/vllm/vllm-openai": [],
        "quay.io/ascend/vllm-ascend": ["v0.27.0"],
    }

    candidates = upstream.resolve_runtime_candidates(
        release,
        tag_loader=lambda repository: tags[repository],
        pr_default=True,
    )

    assert candidates["references"] == ["quay.io/ascend/vllm-ascend:v0.27.0"]


def test_selection_is_source_free_and_wheels_are_the_runtime_union() -> None:
    selection = _selection()

    assert selection["schema_version"] == 3
    assert len(selection["runtimes"]) == 3
    assert {item["id"] for item in selection["wheel_builds"]} == {
        "cu129-cp312-amd64",
        "cu129-cp312-arm64",
        "cann901-a2-cp312-amd64",
        "cann901-a3-cp312-arm64",
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
        baseline["cann901-a3-cp312-arm64"]["recipe_revision"]
        != changed["cann901-a3-cp312-arm64"]["recipe_revision"]
    )
    assert (
        baseline["cann901-a2-cp312-amd64"]["recipe_revision"]
        == changed["cann901-a2-cp312-amd64"]["recipe_revision"]
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


def test_raw_builder_selection_honors_configured_manylinux_policy() -> None:
    fixture = _fixture()
    repository = "quay.io/ascend/manylinux"
    tag = "9.0.1-910b-manylinux_2_28-py3.12"
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
    assert build["manylinux"] == "manylinux_2_34"
    assert build["source_image"].endswith("9.0.1-910b-manylinux_2_34-py3.12")


def test_raw_builder_selection_does_not_downgrade_cann900_to_manylinux_2_28() -> None:
    fixture = _fixture()
    repository = "quay.io/ascend/manylinux"
    fixture["repositories"][repository]["pages"][0]["tags"].append(
        "9.0.0-910b-manylinux_2_28-py3.12"
    )
    probe = next(
        item
        for item in fixture["runtime_probe"]["probes"]
        if item["backend"] == "cann-a2"
    )
    probe["accelerator_runtime"] = "cann-9.0.0"

    with pytest.raises(
        ValueError, match="expected one compatible raw Builder.*found 0"
    ):
        _selection(fixture)


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
