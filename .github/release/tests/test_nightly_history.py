"""Keep the one-time destructive inventory bound to its reviewed evidence."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
nightly = importlib.import_module("ucm_release.nightly_cleanup")
INVENTORY = RELEASE_ROOT / "history/nightly-20261009.json"
REPOSITORY = "ModelEngine-Group/unified-cache-management"


def _inventory():
    return json.loads(INVENTORY.read_text())


def test_historical_targets_have_valid_identity_and_provenance():
    document = _inventory()
    assert document["kind"] == "ucm-nightly-cleanup-inventory"
    assert document["schema_version"] == 1
    assert document["repository"] == REPOSITORY
    records = nightly.merge_records(document["targets"], REPOSITORY)
    assert len(records) == len(document["targets"]) == 30
    assert set(document["evidence"]) == {record["tag"] for record in records}
    assert all(document["evidence"][record["tag"]] for record in records)
    assert sum(len(record["resources"]) for record in records) == 935


def test_snapshot_applies_one_quota_to_successful_and_failed_releases():
    document = _inventory()
    live = [
        record
        for record in document["targets"]
        if any(
            proof["kind"] == "github-release"
            for proof in document["evidence"][record["tag"]]
        )
    ]
    assert len(live) == 22
    selection = nightly.select_retention(live, 7)
    assert [record["tag"] for record in selection.kept] == [
        f"nightly/v0.9.0-202610{day:02d}-1" for day in (9, 8, 7, 6, 5, 4, 3)
    ]
    assert len(selection.candidates) == 15
    draft_candidates = [
        record
        for record in selection.candidates
        if any(
            proof["kind"] == "github-release" and proof["observed_draft"]
            for proof in document["evidence"][record["tag"]]
        )
    ]
    assert len(draft_candidates) == 14
    assert {record["tag"] for record in selection.candidates} - {
        record["tag"] for record in draft_candidates
    } == {"nightly/v0.9.0-20261002-1"}


def test_deleted_github_versions_retain_complete_oci_target_sets():
    document = _inventory()
    old_records = [
        record
        for record in document["targets"]
        if any(
            proof["kind"] == "archived-release-manifest"
            for proof in document["evidence"][record["tag"]]
        )
    ]
    assert len(old_records) == 7
    assert sum(len(record["resources"]) for record in old_records) == 595
    for record in old_records:
        counts = {}
        for resource in record["resources"]:
            counts[resource["kind"]] = counts.get(resource["kind"], 0) + 1
        assert counts == {"chart-oci": 1, "ghcr-index": 28, "ghcr-member": 56}


def test_missing_deleted_run_plan_is_explicitly_blocked():
    document = _inventory()
    tag = "nightly/v0.7.0-20260910-1"
    assert set(document["blocked"]) == {tag}
    assert "complete" in document["blocked"][tag]
    record = next(record for record in document["targets"] if record["tag"] == tag)
    assert record["resources"] == []
    assert record["release_ids"] == [385811397]
    assert record["runs"] == [{"id": 34399106016, "attempt": 1}]
