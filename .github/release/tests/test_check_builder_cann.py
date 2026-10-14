from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RELEASE_ROOT / "check-builder-cann.py"
spec = importlib.util.spec_from_file_location("check_builder_cann", SCRIPT)
assert spec is not None and spec.loader is not None
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_inspection_reads_active_runtime_package_not_unrelated_tools(tmp_path):
    root = tmp_path / "cann-9.2.0-beta.2"
    version = root / "share/info/runtime/version.info"
    version.parent.mkdir(parents=True)
    version.write_text("Version=9.2.0-beta.2\ntimestamp=20260930_161523958\n")
    other = root / "share/info/mindstudio-profiler/version.info"
    other.parent.mkdir(parents=True)
    other.write_text("Version=26.2.0\n")
    facts = check.inspect_cann(
        {"CANN_VERSION": "9.2.0", "ASCEND_TOOLKIT_HOME": str(root)}
    )
    assert facts[f"file:{version}"] == "9.2.0-beta.2"
    assert not any("mindstudio" in source for source in facts)
    assert check.verify_cann(facts, "cann-9.2.0-beta.2") == "9.2.0-beta.2"


def test_full_toolkit_path_is_not_erased_by_short_package_version():
    facts = {
        "env:ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.2.0-beta.2",
        "file:runtime/version.info": "9.2.0",
    }
    assert check.verify_cann(facts, "9.2.0-beta.2") == "9.2.0-beta.2"


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
    workflow = (
        RELEASE_ROOT.parents[1] / ".github/workflows/_probe-runtime.yml"
    ).read_text()
    assert "check-builder-cann.py inspect" in workflow
    assert "check-builder-cann.py version" in workflow
    assert 'head -n 1 "${version_file}"' not in workflow
    assert workflow.index("actions/checkout@") < workflow.index("check-builder-cann.py")
    assert workflow.index("actions/setup-python@") < workflow.index("- id: probe")
    assert "-r .github/release/requirements.txt" in workflow


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
        check.verify_cann(facts, expected)


def test_raw_builder_inspection_needs_no_python_dependencies():
    completed = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "inspect"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert isinstance(json.loads(completed.stdout), dict)


def test_source_only_build_and_mirror_sync_both_verify_actual_version():
    root = RELEASE_ROOT.parents[1]
    dockerfile = (RELEASE_ROOT / "docker/Dockerfile.wheel").read_text()
    build = (root / ".github/workflows/_build-wheel.yml").read_text()
    sync = (root / ".github/workflows/sync-builders.yml").read_text()
    assert "check-builder-cann.py verify" in dockerfile
    assert "UCM_SOURCE_CANN_VERSION=${source_cann_version}" in build
    assert 'source_cann_version(task["builder"]["source_image"])' in build
    assert "check-builder-cann.py inspect" in sync
    assert "check-builder-cann.py verify" in sync
    assert ">out/verification/cann-facts.json" not in sync
