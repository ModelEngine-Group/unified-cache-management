import os

from distutils.core import setup
from setuptools import find_packages

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


print("FOUND PACKAGES:", find_packages())

setup(
    name="unified_cache",
    version="0.0.1",
    author="Unified Cache Team",
    description="Unified Cache Management",
    packages=find_packages(),
    ext_modules=[],
    package_data={},
    include_package_data=True,
    install_requires=[],
    extras_require={},
    python_requires=">=3.10",
)
