from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_VERSION = "0.7.58"
BUILD_SCRIPTS = {
    "build_cuda.sh": "UNIFIEDCACHE-CUDA",
    "build_ascend.sh": "ASCEND",
    "build_mindie.sh": "UNIFIEDCACHE-MINDIE-ASCEND",
    "build_sglang.sh": "UNIFIEDCACHE-SGLANG-CUDA",
}
BASH_SUPPORTS_UPPERCASE_EXPANSION = (
    subprocess.run(
        ["bash", "-c", 'value=ucm; test "${value^^}" = UCM'],
        check=False,
    ).returncode
    == 0
)


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _source_tree(tmp_path: Path, script_name: str) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    source = workspace / "unified-cache-management"
    package = workspace / "package"
    source.mkdir(parents=True)

    script_dir = source / "scripts"
    script_dir.mkdir()
    shutil.copy2(REPOSITORY_ROOT / "scripts" / script_name, script_dir / script_name)
    (source / "version.ini").write_text(
        f"VLLM_UC_VERSION={SOURCE_VERSION}\n", encoding="utf-8"
    )

    (source / "install.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (source / "docker").mkdir()
    (source / "test").mkdir()
    (source / "examples" / "deployments").mkdir(parents=True)
    (source / "examples" / "metrics").mkdir(parents=True)
    (source / "examples" / "ucm_config_example.yaml").write_text(
        "enabled: true\n", encoding="utf-8"
    )
    (source / "examples" / "metrics" / "metrics_configs.yaml").write_text(
        "metrics: []\n", encoding="utf-8"
    )
    e2e = source / "ucm" / "store" / "test" / "e2e"
    e2e.mkdir(parents=True)
    (e2e / "posixstore_aio_test.py").write_text("", encoding="utf-8")

    vllm_patch = source / "ucm" / "integration" / "vllm" / "patch" / "0.11.0"
    vllm_patch.mkdir(parents=True)
    for name in ("vllm-adapt-sparse.patch", "vllm-ascend-adapt.patch"):
        (vllm_patch / name).write_text("", encoding="utf-8")
    sglang_patch = source / "ucm" / "integration" / "sglang" / "patch" / "0.5.5"
    sglang_patch.mkdir(parents=True)
    (sglang_patch / "sglang-adapt.patch").write_text("", encoding="utf-8")
    ascend_ops = source / "ucm" / "sparse" / "gsa_on_device" / "csrc" / "ascend"
    ascend_ops.mkdir(parents=True)
    (ascend_ops / "install.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return workspace, source, package


def _fake_toolchain(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python3",
        """#!/bin/bash
set -eu
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "build" ]; then
    mkdir -p dist ucm/sparse/gsa_on_device/csrc/ascend/dist \
        ucm/sparse/gsa_on_device/csrc/ascend/output
    : > "dist/uc_manager-${SOURCE_VERSION}-py3-none-any.whl"
    : > "ucm/sparse/gsa_on_device/csrc/ascend/dist/ucm_ops-${SOURCE_VERSION}.whl"
    : > "ucm/sparse/gsa_on_device/csrc/ascend/output/ucm_ops-${SOURCE_VERSION}.run"
    exit 0
fi
printf 'unexpected python3 invocation: %s\n' "$*" >&2
exit 97
""",
    )
    _write_executable(bin_dir / "pip3", "#!/bin/sh\nexit 0\n")
    _write_executable(bin_dir / "uname", "#!/bin/sh\nprintf 'x86_64\\n'\n")
    _write_executable(
        bin_dir / "tar",
        """#!/bin/bash
set -eu
printf '%s\n' "$@" > "${TAR_RECORD}"
cat version.ini > "${VERSION_RECORD}"
: > "$2"
""",
    )
    return bin_dir


def _run_script(
    tmp_path: Path, script_name: str, *, skip_tar: bool
) -> tuple[Path, Path]:
    workspace, source, package = _source_tree(tmp_path, script_name)
    bin_dir = _fake_toolchain(tmp_path)
    tar_record = tmp_path / "tar-arguments.txt"
    version_record = tmp_path / "packaged-version.ini"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "SOURCE_VERSION": SOURCE_VERSION,
            "TAR_RECORD": str(tar_record),
            "VERSION_RECORD": str(version_record),
            "WORKSPACE": str(workspace),
        }
    )
    if skip_tar:
        env["SKIP_TAR"] = "1"
    else:
        env.pop("SKIP_TAR", None)

    completed = subprocess.run(
        ["bash", str(source / "scripts" / script_name)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return package, tar_record


@pytest.mark.parametrize("script_name", BUILD_SCRIPTS)
def test_docker_source_build_keeps_checked_in_version_ini(
    tmp_path: Path, script_name: str
) -> None:
    package, tar_record = _run_script(tmp_path, script_name, skip_tar=True)

    assert (package / "version.ini").read_text(encoding="utf-8") == (
        f"VLLM_UC_VERSION={SOURCE_VERSION}\n"
    )
    assert not tar_record.exists()


@pytest.mark.parametrize("script_name,package_platform", BUILD_SCRIPTS.items())
@pytest.mark.skipif(
    not BASH_SUPPORTS_UPPERCASE_EXPANSION,
    reason="the original packaging scripts require Bash 4 uppercase expansion",
)
def test_tar_package_keeps_checked_in_version_file(
    tmp_path: Path, script_name: str, package_platform: str
) -> None:
    _, tar_record = _run_script(tmp_path, script_name, skip_tar=False)

    assert (tmp_path / "packaged-version.ini").read_text(encoding="utf-8") == (
        f"VLLM_UC_VERSION={SOURCE_VERSION}\n"
    )
    arguments = tar_record.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == [
        "-czvf",
        (f"AI-Storage-Kit_{SOURCE_VERSION}_{package_platform}_" "x86_64_DEBUG.tar.gz"),
    ]
