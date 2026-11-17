"""Build the actual portable Toolkit Wheel and verify its release identity."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from ucm_release import toolkit, version_config

ROOT = Path(__file__).resolve().parents[3]


def test_toolkit_wheel_contains_external_resources_and_fork_identity(
    tmp_path,
):
    source = tmp_path / "source"
    shutil.copytree(
        ROOT / "toolkit",
        source / "toolkit",
        ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"),
    )
    shutil.copy2(ROOT / "version.ini", source / "version.ini")
    version = "0.9.0.dev0+draft.test"
    authority = source / "version.ini"
    authority.write_bytes(
        version_config.materialize_bytes(authority.read_text(), version)
    )
    worker = source / "ucm/store/test/e2e/posixstore_aio_test.py"
    worker.parent.mkdir(parents=True)
    shutil.copy2(ROOT / worker.relative_to(source), worker)
    repository = "SuperMarioYL/unified-cache-management"
    plan = {
        "repository": repository,
        "version": version,
        "toolkit_package": toolkit.package_identity(repository, version),
    }
    toolkit.prepare_source(plan, source)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(tmp_path / "dist"),
            str(source / "toolkit"),
        ],
        check=True,
        capture_output=True,
    )
    wheel = next((tmp_path / "dist").glob("*.whl"))
    result = toolkit.record_result(plan, wheel)
    assert result["distribution"] == "supermarioyl-ucm-toolkit"
    assert result["version"] == version
    toolkit.validate_result(plan["toolkit_package"], result)
    (wheel.parent / "toolkit-result.json").write_text(json.dumps(result))
    assert toolkit.load_artifact(plan, wheel.parent) == (result, wheel)
    result["sha256"] = "sha256:" + "0" * 64
    (wheel.parent / "toolkit-result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not match"):
        toolkit.load_artifact(plan, wheel.parent)


def test_planned_toolkit_cannot_be_silently_missing():
    repository = "SuperMarioYL/unified-cache-management"
    plan = {
        "repository": repository,
        "version": "0.7.0rc10",
        "toolkit_package": toolkit.package_identity(repository, "0.7.0rc10"),
    }
    with pytest.raises(ValueError, match="no artifact"):
        toolkit.load_artifact(plan, None)
