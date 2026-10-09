"""Nightly count retention and destructive inventory ownership contracts."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
nightly = importlib.import_module("ucm_release.nightly_cleanup")
manifest_ops = importlib.import_module("ucm_release.manifest")
REPOSITORY = "release-org/unified-cache-management"
TAG = "nightly/v0.9.0-20261009-1"
VERSION = "0.9.0.dev20261009001"
IMAGE = f"ghcr.io/release-org/vllm-openai:v1-ucm-{VERSION}"
CHART = "ghcr.io/release-org/charts/unified-cache-chart:0.9.0-nightly.20261009.1"


def _record(tag=TAG, *, run_id=1, attempt=1, release_id=None):
    return nightly.record_for_tag(
        REPOSITORY, tag, "a" * 40, run_id, attempt, release_id
    )


def _plan(*, multi_arch=True, ghcr=True, dockerhub=False, chart=True):
    members = (
        [
            {"image_id": "x-amd64", "cpu_arch": "amd64", "reference": IMAGE + "-amd64"},
            {"image_id": "x-arm64", "cpu_arch": "arm64", "reference": IMAGE + "-arm64"},
        ]
        if multi_arch
        else [{"image_id": "x-amd64", "cpu_arch": "amd64", "reference": IMAGE}]
    )
    return {
        "kind": "ucm-release-plan",
        "repository": REPOSITORY,
        "route": "release",
        "release_type": "nightly",
        "git_tag": TAG,
        "version": VERSION,
        "image_version": VERSION,
        "publish": {
            "ghcr": {"enabled": ghcr},
            "dockerhub": {"enabled": dockerhub, "namespace": "docker.io/other-owner"},
            "chart_oci": {"enabled": chart, "namespace": "ghcr.io/release-org/charts"},
        },
        "chart": {"name": "unified-cache-chart", "version": "0.9.0-nightly.20261009.1"},
        "families": [
            {
                "create_index": multi_arch,
                "published_reference": IMAGE,
                "members": members,
            }
        ],
        "wheels": [
            {"builder": {"repository": "ghcr.io/release-org/builders", "tag": "shared"}}
        ],
        "images": [{"runtime": {"image_reference": "ghcr.io/release-org/base:latest"}}],
    }


def _legacy_manifest():
    return {
        "kind": "ucm-release-manifest",
        "schema_version": 6,
        "tag": TAG,
        "release_type": "nightly",
        "actions_run_id": 123,
        "chart_oci": CHART,
        "github_release_assets": ["release-manifest.json", "wheel.whl"],
        "runtime_images": {
            "ghcr": {
                "indexes": [IMAGE],
                "members": [IMAGE + "-amd64", IMAGE + "-arm64"],
            },
            "dockerhub": {"indexes": [], "members": []},
        },
    }


def _public_manifest():
    document = json.loads(
        (RELEASE_ROOT / "tests/fixtures/release-manifest.json").read_text()
    )
    text = (
        json.dumps(document).replace("0.9.3", VERSION).replace("example", "release-org")
    )
    document = json.loads(text)
    document["release"].update(
        tag=TAG,
        type="nightly",
        url=f"https://github.com/{REPOSITORY}/releases/tag/{TAG}",
    )
    publication = document["images"][0]["publications"]["ghcr"]
    publication["pull"] = IMAGE
    publication["members"][0]["reference"] = IMAGE + "-amd64"
    publication["members"][1]["reference"] = IMAGE + "-arm64"
    document["chart"]["oci"] = CHART
    return document


def test_plan_inventory_contains_enabled_targets_and_excludes_shared_inputs():
    record = nightly.record_from_plan(_record(), _plan(dockerhub=True))
    assert [(item["kind"], item["reference"]) for item in record["resources"]] == [
        ("chart-oci", CHART),
        ("ghcr-index", IMAGE),
        ("ghcr-member", IMAGE + "-amd64"),
        ("ghcr-member", IMAGE + "-arm64"),
        (
            "dockerhub-index",
            IMAGE.replace("ghcr.io/release-org", "docker.io/other-owner"),
        ),
        (
            "dockerhub-member",
            (IMAGE + "-amd64").replace("ghcr.io/release-org", "docker.io/other-owner"),
        ),
        (
            "dockerhub-member",
            (IMAGE + "-arm64").replace("ghcr.io/release-org", "docker.io/other-owner"),
        ),
    ]


def test_single_architecture_pull_is_only_one_member():
    record = nightly.record_from_plan(_record(), _plan(multi_arch=False, chart=False))
    assert record["resources"] == [{"kind": "ghcr-member", "reference": IMAGE}]


def test_disabled_publication_has_a_legitimate_empty_inventory():
    record = nightly.record_from_plan(_record(), _plan(ghcr=False, chart=False))
    assert record["resources"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", "other/repo"),
        ("git_tag", "nightly/v0.9.0-20261008-1"),
        ("version", "0.9.0"),
        ("image_version", "0.9.0"),
        ("route", "pr"),
    ],
)
def test_plan_identity_cannot_be_grafted_onto_a_different_binding(field, value):
    plan = _plan()
    plan[field] = value
    with pytest.raises(nightly.CleanupRecordError, match="identity differs"):
        nightly.record_from_plan(_record(), plan)


@pytest.mark.parametrize(
    "kind,reference",
    [
        ("ghcr-index", IMAGE.replace("release-org", "other-owner")),
        ("ghcr-index", "ghcr.io/release-org/vllm-openai:v1"),
        ("ghcr-index", IMAGE + "-amd64"),
        ("ghcr-member", IMAGE.replace("20261009001", "20261008001")),
        ("ghcr-member", IMAGE.replace("vllm-openai", "builders")),
        ("ghcr-member", IMAGE.replace("ghcr.io", "docker.io")),
        ("chart-oci", CHART.replace("20261009", "20261008")),
        ("chart-oci", CHART.replace("release-org", "other-owner")),
        ("pypi", "uc-manager==" + VERSION),
        ("ghcr-member", " " + IMAGE),
        ("ghcr-member", IMAGE + "@sha256:" + "b" * 64),
    ],
)
def test_inventory_rejects_cross_version_or_shared_resource_deletion(kind, reference):
    record = _record()
    record["resources"] = [{"kind": kind, "reference": reference}]
    with pytest.raises(nightly.CleanupRecordError):
        nightly.validate_record(record, REPOSITORY)


def test_legacy_projection_does_not_widen_public_manifest_contract():
    legacy = _legacy_manifest()
    record = nightly.record_from_manifest(legacy, REPOSITORY, release_id=456)
    assert record["source_sha"] is None
    assert record["release_ids"] == [456]
    assert record["runs"] == [{"id": 123, "attempt": 1}]
    assert len(record["resources"]) == 4
    with pytest.raises(manifest_ops.ManifestError, match="schema_version must be 9"):
        manifest_ops.validate_manifest(legacy)


def test_current_public_manifest_projects_the_same_internal_inventory():
    record = nightly.record_from_manifest(_public_manifest(), REPOSITORY, 456)
    assert (
        record["resources"]
        == nightly.record_from_manifest(_legacy_manifest(), REPOSITORY)["resources"]
    )


@pytest.mark.parametrize("mutation", ["schema", "fields", "kind", "run", "ownership"])
def test_legacy_projection_is_strict(mutation):
    manifest = _legacy_manifest()
    if mutation == "schema":
        manifest["schema_version"] = 7
    elif mutation == "fields":
        manifest["untrusted"] = True
    elif mutation == "kind":
        manifest["kind"] = "other"
    elif mutation == "run":
        manifest["actions_run_id"] = True
    else:
        manifest["runtime_images"]["ghcr"]["indexes"] = [
            IMAGE.replace("release-org", "someone")
        ]
    with pytest.raises(nightly.CleanupRecordError):
        nightly.record_from_manifest(manifest, REPOSITORY)


def test_duplicate_releases_and_attempts_union_one_version_without_mutating_inputs():
    first = _record(release_id=10)
    second = nightly.record_from_plan(_record(attempt=2, release_id=11), _plan())
    original = copy.deepcopy([first, second])
    merged = nightly.merge_records([first, second, second], REPOSITORY)
    assert len(merged) == 1
    assert merged[0]["release_ids"] == [10, 11]
    assert merged[0]["runs"] == [{"id": 1, "attempt": 1}, {"id": 1, "attempt": 2}]
    assert len(merged[0]["resources"]) == 4
    assert [first, second] == original


def test_recovered_identity_gains_known_source_but_rejects_conflicting_sources():
    legacy = _record()
    legacy.update(source_sha=None, runs=[])
    merged = nightly.merge_records([legacy, _record()], REPOSITORY)
    assert merged[0]["source_sha"] == "a" * 40
    conflicting = _record()
    conflicting["source_sha"] = "b" * 40
    with pytest.raises(nightly.CleanupRecordError, match="conflicting source SHAs"):
        nightly.merge_records([_record(), conflicting], REPOSITORY)


def test_retention_counts_all_versions_once_without_reserving_an_imaginary_current_slot():
    records = [
        _record(f"nightly/v0.9.0-2026100{day}-1", run_id=day) for day in range(1, 5)
    ]
    records[0].update(source_sha=None, runs=[])  # Recovered manifestless failed Draft.
    records += [_record("nightly/v0.9.0-20261004-1", run_id=4, attempt=2, release_id=9)]
    selected = nightly.select_retention(records, 3)
    assert [item["tag"] for item in selected.kept] == [
        "nightly/v0.9.0-20261004-1",
        "nightly/v0.9.0-20261003-1",
        "nightly/v0.9.0-20261002-1",
    ]
    assert [item["tag"] for item in selected.candidates] == [
        "nightly/v0.9.0-20261001-1"
    ]


def test_sort_uses_tag_date_then_numeric_sequence_then_numeric_base():
    tags = [
        "nightly/v0.9.0-20261009-2",
        "nightly/v0.8.0-20261009-10",
        "nightly/v0.10.0-20261009-10",
        "nightly/v0.1.0-20261010-1",
    ]
    selected = nightly.select_retention([_record(tag) for tag in tags], 2)
    assert [item["tag"] for item in selected.kept] == [tags[3], tags[2]]
    assert [item["tag"] for item in selected.candidates] == [tags[1], tags[0]]


def test_active_excess_versions_count_toward_quota_and_are_reported_as_deferred():
    tags = [f"nightly/v0.9.0-2026100{day}-1" for day in range(1, 5)]
    selected = nightly.select_retention([_record(tag) for tag in tags], 2, {tags[1]})
    assert [item["tag"] for item in selected.kept] == [tags[3], tags[2]]
    assert [item["tag"] for item in selected.deferred] == [tags[1]]
    assert [item["tag"] for item in selected.candidates] == [tags[0]]


def test_unlimited_retention_disables_deletion_even_for_old_active_versions():
    records = [_record(), _record("nightly/v0.9.0-20261008-1")]
    selected = nightly.select_retention(records, -1, {TAG})
    assert len(selected.kept) == 2
    assert selected.candidates == selected.deferred == ()


@pytest.mark.parametrize("count", [0, -2, True, "7"])
def test_invalid_quota_is_rejected(count):
    with pytest.raises(nightly.CleanupRecordError, match="max_count"):
        nightly.select_retention([], count)


@pytest.mark.parametrize(
    "tag",
    [
        "v0.9.0",
        "draft/v0.9.0",
        "nightly/v0.9.0-20260230-1",
        "nightly/v0.9.0-20261009-01",
    ],
)
def test_noncanonical_or_invalid_dates_never_enter_nightly_inventory(tag):
    with pytest.raises(nightly.CleanupRecordError):
        _record(tag)


def test_resource_kind_conflict_and_implicit_schema_extensions_are_rejected():
    record = _record()
    record["resources"] = [
        {"kind": "ghcr-index", "reference": IMAGE},
        {"kind": "ghcr-member", "reference": IMAGE},
    ]
    with pytest.raises(nightly.CleanupRecordError, match="conflicting resource kinds"):
        nightly.validate_record(record, REPOSITORY)
    record = _record()
    record["complete"] = True
    with pytest.raises(nightly.CleanupRecordError, match="fields differ"):
        nightly.validate_record(record, REPOSITORY)
