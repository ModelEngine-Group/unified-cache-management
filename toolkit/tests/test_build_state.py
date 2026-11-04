"""Build state belongs to the user, never to installed adapter source."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucm_toolkit.errors import ToolkitError
from ucm_toolkit.tools.dev_sandbox import adapter


def test_successful_build_is_reused_without_modifying_installed_source(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    build = tmp_path / "custom-build"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(adapter, "resource_path", lambda name: source)
    monkeypatch.setattr(adapter, "command_exists", lambda name: True)
    calls = []

    def command(arguments):
        calls.append(arguments)
        if "-B" in arguments:
            build.mkdir()
            (build / "CMakeCache.txt").write_text(
                f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n"
            )

    monkeypatch.setattr(adapter, "check_command", command)
    before = Path(adapter.__file__).read_bytes()
    tool = adapter.DevSandboxTool()
    tool.build(
        argparse.Namespace(
            build_dir=str(build), build_type="Release", cmake_arg=[], jobs=2
        )
    )
    assert adapter.DevSandboxTool()._build_path() == build
    assert len(calls) == 2
    assert Path(adapter.__file__).read_bytes() == before
    tool.clean(argparse.Namespace(dry_run=True))
    assert build.exists()
    tool.clean()
    assert not build.exists()


def test_failed_build_does_not_save_path_and_clean_rejects_unowned_directory(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(adapter, "resource_path", lambda name: source)
    monkeypatch.setattr(adapter, "command_exists", lambda name: True)

    def fail(arguments):
        raise ToolkitError("compile failed")

    monkeypatch.setattr(adapter, "check_command", fail)
    tool = adapter.DevSandboxTool()
    with pytest.raises(ToolkitError, match="compile failed"):
        tool.build(
            argparse.Namespace(
                build_dir=str(tmp_path / "failed"),
                build_type="Release",
                cmake_arg=[],
                jobs=2,
            )
        )
    assert not tool._state_file().exists()
    directory = tmp_path / "user-data"
    directory.mkdir()
    (directory / "keep.txt").write_text("keep")
    monkeypatch.setattr(tool, "_build_path", lambda: directory)
    with pytest.raises(ToolkitError, match="not a build"):
        tool.clean()
    assert (directory / "keep.txt").read_text() == "keep"
