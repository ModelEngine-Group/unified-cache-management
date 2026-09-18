"""Current published release data shared by documentation consumer tests."""

import json
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / ".github/release/tests/fixtures/release-manifest.json"
)


def manifest_fixture(version="0.9.3"):
    return json.loads(FIXTURE.read_text().replace("0.9.3", version))


def published_release(version="0.9.3", *, prerelease=False):
    manifest = manifest_fixture(version)
    manifest["release"]["type"] = "prerelease" if prerelease else "stable"
    download = f"https://github.com/example/ucm/releases/download/v{version}/"
    for artifact in [*manifest["wheels"], manifest["chart"]]:
        artifact["url"] = download + artifact["filename"]
    return {
        "tag_name": f"v{version}",
        "draft": False,
        "prerelease": prerelease,
        "assets": [
            {"name": name, "browser_download_url": download + name}
            for name in manifest["github_release_assets"]
        ],
    }, manifest
