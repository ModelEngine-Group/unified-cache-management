"""Single-image PR resolution through Registry mirror Builders."""

from __future__ import annotations

import importlib
import sys
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
pr = importlib.import_module("ucm_release.pr")
upstream = importlib.import_module("ucm_release.upstream")


def _fixture_policy():
    return policy.resolve(
        repository="release-org/unified-cache-management",
        version_override="0.7.60rc1",
    )


def _inputs():
    formal = _fixture_policy()
    fixture = core.load_json(TAG_FIXTURE)
    selection = upstream.resolve_upstreams(
        formal,
        candidates=upstream.resolve_runtime_candidates(formal, tag_fixture=fixture),
        runtime_probe=fixture["runtime_probe"],
        tag_fixture=fixture,
    )
    catalog = builders.catalog_from_selection(
        selection, owner="release-org", formal_policy=formal
    )
    return formal, fixture, selection, catalog


def _cuda_probe(*, dual_arch: bool = False) -> dict[str, object]:
    probes = [
        item
        for item in _inputs()[1]["runtime_probe"]["probes"]
        if item["product_id"] == "vllm"
    ]
    return {
        "kind": "ucm-runtime-probe",
        "schema_version": 2,
        "probes": probes if dual_arch else probes[:1],
    }


def _inventory(catalog: dict[str, object], ids: set[str]) -> dict[str, object]:
    return {
        "kind": "ucm-builder-registry",
        "schema_version": 1,
        "builders": [
            {
                **item,
                "created": "2026-08-24T00:00:00Z",
                "checked": True,
            }
            for item in catalog["builders"]
            if item["id"] in ids
        ],
    }


def _finalized_catalog(catalog: dict[str, object]) -> dict[str, object]:
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
    return builders.finalize_catalog(catalog, observations)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def test_checked_registry_builder_reuse_preserves_immutable_identity() -> None:
    formal, _fixture, _selection, catalog = _inputs()
    result = pr.resolve_pr_request(
        formal,
        _cuda_probe(),
        _inventory(catalog, {"cu129-cp312-amd64"}),
        pr_number=30,
        author="SuperMarioYL",
        run_id=100,
    )

    assert result["ok"] is True
    build = result["selection"]["wheel_builds"][0]
    assert build["sync_mode"] == "mirror"
    assert result["selection"]["runtimes"][0]["channel"] == "pinned"
    assert result["selection"]["runtimes"][0]["target_tag"].startswith(
        "release-org-pr-30-supermarioyl-run-100-"
    )
    assert not {
        "source_repository",
        "source_ref",
        "source_commit",
        "mooncake_version",
    } & _all_keys(result)


def test_missing_registry_builder_resolves_one_raw_mirror_for_sync() -> None:
    formal, _fixture, selection, _catalog = _inputs()
    raw = next(
        item for item in selection["wheel_builds"] if item["id"] == "cu129-cp312-amd64"
    )
    result = pr.resolve_pr_request(
        formal,
        _cuda_probe(),
        {"kind": "ucm-builder-registry", "schema_version": 1, "builders": []},
        pr_number=30,
        author="SuperMarioYL",
        run_id=101,
        raw_build_resolver=lambda _probes: [raw],
    )

    assert result["ok"] is True
    assert result["selection"]["wheel_builds"] == [raw]
    assert result["builder_catalog"]["builders"][0]["sync_mode"] == "mirror"
    sync = builders.compute_sync_plan(result["builder_catalog"], {})
    assert len(sync["builders"]) == 1


def test_dual_arch_tag_expands_both_members_and_one_index() -> None:
    formal, _fixture, _selection, catalog = _inputs()
    result = pr.resolve_pr_request(
        formal,
        _cuda_probe(dual_arch=True),
        _inventory(
            catalog,
            {"cu129-cp312-amd64", "cu129-cp312-arm64"},
        ),
        pr_number=30,
        author="SuperMarioYL",
        run_id=102,
    )

    assert result["ok"] is True
    assert len(result["publication"]["member_matrix"]["include"]) == 2
    assert len(result["publication"]["index_matrix"]["include"]) == 1
    assert set(result["selection"]["runtimes"][0]["wheel_build_ids"]) == {
        "amd64",
        "arm64",
    }


def test_single_arch_publication_and_compact_plan_share_the_bare_tag() -> None:
    formal, _fixture, _selection, catalog = _inputs()
    result = pr.resolve_pr_request(
        formal,
        _cuda_probe(),
        _inventory(catalog, {"cu129-cp312-amd64"}),
        pr_number=30,
        author="SuperMarioYL",
        run_id=105,
    )
    plan = compact.resolve_plan(
        formal,
        runtime_selection=result["selection"],
        builder_catalog=_finalized_catalog(result["builder_catalog"]),
        route="pr",
    )

    expected = plan["families"][0]["published_reference"]
    assert result["publication"]["families"][0]["final_refs"] == [expected]
    assert result["publication"]["member_matrix"]["include"][0]["target_ref"] == (
        expected
    )


def test_blocked_a5_stops_before_raw_builder_resolution() -> None:
    formal, _fixture, _selection, _catalog = _inputs()
    probe = _cuda_probe()
    item = probe["probes"][0]
    item.update(
        {
            "product_id": "vllm-ascend",
            "runtime_ref": "quay.io/ascend/vllm-ascend:v0.26.0rc-a5",
            "repository": "quay.io/ascend/vllm-ascend",
            "tag": "v0.26.0rc-a5",
            "target_repository": "ghcr.io/release-org/vllm-ascend",
            "backend": "cann-a5",
            "accelerator_runtime": "cann-9.1.0",
            "soc_version": "ascend950dt_9582",
        }
    )
    called = False

    def raw_resolver(_probes):
        nonlocal called
        called = True
        return []

    result = pr.resolve_pr_request(
        formal,
        probe,
        {"kind": "ucm-builder-registry", "schema_version": 1, "builders": []},
        pr_number=30,
        author="SuperMarioYL",
        run_id=103,
        raw_build_resolver=raw_resolver,
    )

    assert result["ok"] is False
    assert result["problems"][0]["reason"] == "blocked-backend"
    assert called is False


def test_image_command_rejects_multiple_runtime_requests() -> None:
    formal, fixture, _selection, _catalog = _inputs()
    probes = (
        fixture["runtime_probe"]["probes"][:1] + fixture["runtime_probe"]["probes"][2:3]
    )

    with pytest.raises(ValueError, match="exactly one Runtime image"):
        pr.resolve_pr_request(
            formal,
            {"kind": "ucm-runtime-probe", "schema_version": 2, "probes": probes},
            {
                "kind": "ucm-builder-registry",
                "schema_version": 1,
                "builders": [],
            },
            pr_number=30,
            author="SuperMarioYL",
            run_id=104,
        )
