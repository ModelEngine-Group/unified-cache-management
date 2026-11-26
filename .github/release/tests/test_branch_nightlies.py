"""Branch discovery, immutable wheel identity and retention across the cutover."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucm_release import cleanup_records, version_config
from ucm_release.nightly import Nightly, latest_release_branch


def test_branch_discovery_and_numbering():
    assert (
        latest_release_branch(
            [
                "0.9.9-release",
                "0.10.0-release",
                "0.11.0rc2-release",
                "feature",
                "1.0.0-release",
            ]
        )
        == "0.10.0-release"
    )
    assert latest_release_branch(["develop"]) is None
    tags = [
        "nightly/develop-20261126",
        "nightly/develop-20261126-2",
        "nightly/develop-20261126-10",
        "nightly/0.10.0-release-20261126",
        "nightly/v0.10.0-20261126-7",
    ]
    for branch, n, label in [
        ("develop", 11, "develop"),
        ("0.10.0-release", 2, "release.0.10.0"),
    ]:
        build = version_config.next_nightly_classification(
            tags, base_version="0.10.0", release_date="20261126", branch=branch
        )
        assert build["git_tag"] == f"nightly/{branch}-20261126-{n}"
        assert build["version"] == f"0.10.0.dev20261126{n:03d}+{label}"
        assert (
            version_config.derive_chart_version(build["version"])
            == build["chart_version"]
        )
        assert version_config.custom_draft_tag(build["version"]) is None
    with pytest.raises(ValueError):
        Nightly.parse("nightly/develop-20261126-1")


def test_retention_groups_failures_and_old_tags_without_deleting_active_runs():
    tags = [
        "nightly/v0.9.0-20261124-1",
        "nightly/v0.10.0-20261125-1",
        "nightly/develop-20261126",
        "nightly/develop-20261126-2",
        "nightly/develop-20261126-10",
        "nightly/0.10.0-release-20261126",
        "nightly/0.9.0-release-20261126",
    ]
    repo = "example/ucm"
    records = [
        cleanup_records.record_for_tag(repo, tag, "a" * 40, i, 1)
        for i, tag in enumerate(tags, 1)
    ]
    records.append(cleanup_records.record_for_tag(repo, tags[-4], "a" * 40, 4, 2))
    selected = cleanup_records.select_retention(
        records, 3, [tags[0]], latest_branch="0.10.0-release"
    )
    assert [r["tag"] for r in selected.deferred] == [tags[0]]
    assert {r["tag"] for r in selected.kept} == set(tags[2:6])
    assert {r["tag"] for r in selected.candidates} == {tags[1], tags[-1]}
    assert tags[-1] in {
        r["tag"] for r in cleanup_records.select_retention(records, 3).kept
    }


def test_new_tag_only_snapshots_use_source_version_and_detect_conflicting_evidence():
    repo, tag = "example/ucm", "nightly/develop-20261126"

    def snapshot(base):
        doc = cleanup_records.new_record(repo, tag, "a" * 40, 1, 1, base_version=base)
        return cleanup_records.decode_snapshot(
            doc, repo, cleanup_records.record_basename(doc, 1)
        )

    record = snapshot("0.10.0")
    assert record["versions"] == ["0.10.0.dev20261126001+develop"]
    with pytest.raises(cleanup_records.CleanupRecordError, match="version differs"):
        cleanup_records.merge_records([record, snapshot("0.11.0")], repo)
    with pytest.raises(ValueError, match="tagged source"):
        version_config.classify_tag(tag)
