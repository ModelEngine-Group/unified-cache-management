"""Setuptools hooks for resources maintained outside the Python package."""

import shutil
from pathlib import Path

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent


def source_version():
    authority = ROOT.parent / "version.ini"
    if authority.is_file():
        for line in authority.read_text().splitlines():
            if line.startswith("UCM_VERSION="):
                return line.partition("=")[2]
        raise ValueError("version.ini has no UCM_VERSION")
    return (ROOT / "VERSION").read_text().strip()


def benchmark_source():
    source = ROOT.parent / "ucm/store/test/e2e/posixstore_aio_test.py"
    return source if source.is_file() else ROOT / "src/posix_aio/benchmark.py"


class BuildPy(build_py):
    def run(self):
        super().run()
        package = Path(self.build_lib) / "ucm_toolkit"
        (package / "_version.py").write_text(
            f"__version__ = {source_version()!r}\n", encoding="utf-8"
        )
        resources = package / "_resources"
        resources.mkdir(exist_ok=True)
        shutil.copytree(
            ROOT / "src/dev-sandbox",
            resources / "dev-sandbox",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "build",
                "build-*",
                "cmake-build-*",
                ".git",
                "__pycache__",
                "*.pyc",
                "*.o",
                "*.so",
                "*.a",
                "compile_commands.json",
            ),
        )
        shutil.copy2(ROOT / "src/nic_monitor/nic_monitor_pro.sh", resources)
        shutil.copy2(benchmark_source(), resources / "posixstore_aio_test.py")
        shutil.copy2(ROOT / "src/dev-sandbox/LICENSE", resources / "LICENSE")


class SourceDistribution(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        root = Path(base_dir)
        (root / "VERSION").write_text(source_version() + "\n", encoding="utf-8")
        benchmark = root / "src/posix_aio/benchmark.py"
        benchmark.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(benchmark_source(), benchmark)
