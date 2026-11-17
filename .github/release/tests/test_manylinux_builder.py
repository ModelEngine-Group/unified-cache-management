from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest
import setuptools

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "platform", ["cuda", "ascend", "ascend-a3", "ascend-a5", "musa", "maca"]
)
def test_wheel_setup_passes_native_platform_to_cmake(
    platform, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PLATFORM", platform)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    configuration = runpy.run_path(str(ROOT / "setup.py"))
    build = configuration["CMakeBuild"](setuptools.Distribution())
    build.build_temp = str(tmp_path / "temp")
    build.build_lib = str(tmp_path / "lib")
    commands = []
    monkeypatch.setattr(
        subprocess, "check_call", lambda args, **kwargs: commands.append(args)
    )

    build.build_cmake(configuration["CMakeExtension"]("ucm", str(ROOT)))

    assert f"-DRUNTIME_ENVIRONMENT={platform}" in commands[0]
    assert "-DRUNTIME_ENVIRONMENT=simu" not in commands[0]


def test_ucm_owned_metrics_library_has_wheel_relative_install_rpaths() -> None:
    expected = {
        "ucm/shared/metrics/CMakeLists.txt": 'INSTALL_RPATH "$ORIGIN"',
        "ucm/store/cache/CMakeLists.txt": (
            'INSTALL_RPATH "$ORIGIN/../../shared/metrics"'
        ),
        "ucm/store/posix/CMakeLists.txt": (
            'INSTALL_RPATH "$ORIGIN/../../shared/metrics"'
        ),
        "ucm/store/pipeline/CMakeLists.txt": (
            'INSTALL_RPATH "$ORIGIN/../../shared/metrics"'
        ),
    }
    for relative_path, contract in expected.items():
        assert contract in (ROOT / relative_path).read_text(encoding="utf-8")
