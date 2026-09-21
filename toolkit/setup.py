"""Build the CLI and its source resources without compiling native tools."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _build_support import BuildPy, SourceDistribution, source_version
from setuptools import setup

setup(
    version=source_version(),
    cmdclass={"build_py": BuildPy, "sdist": SourceDistribution},
)
