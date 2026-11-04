from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
runtime = importlib.import_module("ucm_release.runtime")
SCRIPT = RELEASE_ROOT / "check-builder-cann.py"


def test_full_toolkit_path_is_not_erased_by_short_package_version():
    facts = {
        "env:ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.2.0-beta.2",
        "file:runtime/version.info": "9.2.0",
    }
    assert runtime.verify_cann_installation(facts, "9.2.0-beta.2") == "9.2.0-beta.2"


def test_native_probe_resolves_bare_env_to_actual_beta_identity(tmp_path):
    facts = tmp_path / "cann-facts.json"
    facts.write_text(
        json.dumps(
            {
                "env:CANN_VERSION": "9.2.0",
                "env:ASCEND_TOOLKIT_HOME": "/usr/local/Ascend/cann-9.2.0-beta.1",
                "file:runtime/version.info": "9.2.0-beta.1",
            }
        )
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "version", "--facts", str(facts)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "9.2.0-beta.1"


@pytest.mark.parametrize(
    "facts, expected, message",
    [
        ({"env:CANN_VERSION": "9.2.0"}, "9.2.0-beta.2", "differs"),
        ({"file:runtime/version.info": "9.1.0"}, "9.2.0", "differs"),
        (
            {"env:CANN_VERSION": "9.2.0-beta.1", "file:version.info": "9.2.0-beta.2"},
            "9.2.0-beta.2",
            "disagree",
        ),
        ({}, "9.2.0", "no readable"),
    ],
)
def test_source_version_mismatch_blocks_build(facts, expected, message):
    with pytest.raises(ValueError, match=message):
        runtime.verify_cann_installation(facts, expected)


def test_source_only_build_and_mirror_sync_both_verify_actual_version():
    root = RELEASE_ROOT.parents[1]
    dockerfile = (RELEASE_ROOT / "docker/Dockerfile.wheel").read_text()
    build = (root / ".github/workflows/_build-wheel.yml").read_text()
    sync = (root / ".github/workflows/sync-builders.yml").read_text()
    assert "check-builder-cann.py verify" in dockerfile
    assert "UCM_SOURCE_CANN_VERSION=${source_cann_version}" in build
    assert "check-builder-cann.py inspect" in sync
    assert "check-builder-cann.py verify" in sync
