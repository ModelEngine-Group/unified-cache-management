"""Select installation data through GitHub release/asset contracts."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import release_manifest as releases
from manifest_fixtures import published_release

REPOSITORY = "example/ucm"
API = f"https://api.github.com/repos/{REPOSITORY}/releases"


def install_responses(monkeypatch, pairs):
    responses = {f"{API}?per_page=100&page=1": [metadata for metadata, _ in pairs]}
    for metadata, manifest in pairs:
        responses[f"{API}/tags/{metadata['tag_name']}"] = metadata
        for asset in metadata["assets"]:
            if asset["name"] == releases.RELEASE_MANIFEST_FILENAME:
                responses[asset["browser_download_url"]] = manifest
    called = []

    def fetch(url):
        called.append(url)
        assert url in responses, f"unexpected URL: {url}"
        return copy.deepcopy(responses[url])

    monkeypatch.setattr(releases, "fetch_json", fetch)
    return called


@pytest.mark.parametrize("version,prerelease", [("0.9.3", False), ("0.9.4rc1", True)])
def test_exact_tag_uses_its_own_completed_stable_or_rc_manifest(
    monkeypatch, version, prerelease
):
    pair = published_release(version, prerelease=prerelease)
    called = install_responses(monkeypatch, [pair])
    result = releases.resolve_manifest(REPOSITORY, tag=f"v{version}")
    assert result == pair[1]
    assert called == [
        f"{API}/tags/v{version}",
        f"https://github.com/{REPOSITORY}/releases/download/v{version}/release-manifest.json",
    ]


def test_latest_selects_highest_completed_schema9_stable_not_api_order(monkeypatch):
    newest_legacy = published_release("0.9.5")
    newest_legacy[1]["schema_version"] = 6
    newer_legacy = published_release("0.9.4")
    newer_legacy[1]["schema_version"] = 7
    selected = published_release("0.9.3")
    rc = published_release("0.9.6rc1", prerelease=True)
    old = published_release("0.9.2")
    called = install_responses(
        monkeypatch, [old, rc, newer_legacy, selected, newest_legacy]
    )
    assert releases.resolve_manifest(REPOSITORY) == selected[1]
    assert not any("v0.9.6rc1/release-manifest" in url for url in called)
    assert not any("v0.9.2/release-manifest" in url for url in called)


def test_latest_uses_highest_completed_rc_when_no_stable_is_available(monkeypatch):
    old = published_release("0.7.0rc15", prerelease=True)
    selected = published_release("0.7.0rc16", prerelease=True)
    incomplete = published_release("0.7.0rc17", prerelease=True)
    incomplete[0]["assets"] = []
    called = install_responses(monkeypatch, [old, incomplete, selected])

    assert releases.resolve_manifest(REPOSITORY) == selected[1]
    assert not any("v0.7.0rc15/release-manifest" in url for url in called)


@pytest.mark.parametrize("pending", ["no-release", "draft", "no-manifest"])
def test_incomplete_tag_is_pending(monkeypatch, pending):
    pair = published_release()
    if pending == "no-release":

        def fetch(url):
            raise HTTPError(url, 404, "Not found", {}, None)

        monkeypatch.setattr(releases, "fetch_json", fetch)
    else:
        if pending == "draft":
            pair[0]["draft"] = True
        else:
            pair[0]["assets"] = []
        install_responses(monkeypatch, [pair])
    with pytest.raises(releases.ReleasePending):
        releases.resolve_manifest(REPOSITORY, tag="v0.9.3")


def test_no_completed_release_leaves_installation_unavailable(monkeypatch):
    pair = published_release()
    pair[0]["assets"] = []
    install_responses(monkeypatch, [pair])
    assert releases.resolve_manifest(REPOSITORY) is None


def test_damaged_latest_manifest_fails_instead_of_hiding_behind_older_release(
    monkeypatch,
):
    damaged = published_release("0.9.4")
    del damaged[1]["python"]
    called = install_responses(monkeypatch, [published_release(), damaged])
    with pytest.raises(releases.ManifestError, match="fields differ"):
        releases.resolve_manifest(REPOSITORY)
    assert not any("v0.9.3/release-manifest" in url for url in called)


@pytest.mark.parametrize(
    "mismatch,message",
    [
        ("tag", "another tag"),
        ("repository", "another repository"),
        ("wheel-url", "download URL differs"),
        ("chart-url", "download URL differs"),
        ("assets", "assets differ"),
    ],
)
def test_exact_release_rejects_mismatched_provenance(monkeypatch, mismatch, message):
    metadata, manifest = published_release()
    if mismatch == "tag":
        manifest["release"]["tag"] = "v0.9.2"
    elif mismatch == "repository":
        manifest["release"]["url"] = "https://github.com/other/ucm/releases/tag/v0.9.3"
    elif mismatch.endswith("-url"):
        item = (
            manifest["wheels"][0]
            if mismatch == "wheel-url"
            else manifest[mismatch.removesuffix("-url")]
        )
        item["url"] = item["url"].replace("example/ucm", "other/ucm")
    else:
        metadata["assets"].append(
            {
                "name": "unfinished-receipt.json",
                "browser_download_url": "https://github.com/example/ucm/releases/download/v0.9.3/unfinished-receipt.json",
            }
        )
    install_responses(monkeypatch, [(metadata, manifest)])
    with pytest.raises(releases.ManifestError, match=message):
        releases.resolve_manifest(REPOSITORY, tag="v0.9.3")


def test_exact_unsupported_manifest_is_an_error_not_pending(monkeypatch):
    pair = published_release()
    pair[1]["schema_version"] = 8
    install_responses(monkeypatch, [pair])
    with pytest.raises(
        releases.ManifestError, match="schema_version must be 9"
    ) as error:
        releases.resolve_manifest(REPOSITORY, tag="v0.9.3")
    assert not isinstance(error.value, releases.ReleasePending)


def test_release_without_chart_keeps_its_wheel_installation_data(monkeypatch):
    metadata, manifest = published_release()
    filename = manifest["chart"]["filename"]
    manifest["chart"] = None
    manifest["github_release_assets"].remove(filename)
    metadata["assets"] = [
        asset for asset in metadata["assets"] if asset["name"] != filename
    ]
    install_responses(monkeypatch, [(metadata, manifest)])

    assert releases.resolve_manifest(REPOSITORY, tag="v0.9.3") == manifest
