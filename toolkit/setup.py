"""Legacy setuptools entry point for editable installs."""

from setuptools import find_packages, setup

setup(
    name="ucm-toolkit",
    version="0.1.0",
    description="Unified CLI for UCM toolkit utilities.",
    python_requires=">=3.9",
    packages=find_packages(include=["ucm_toolkit", "ucm_toolkit.*"]),
    package_data={
        "ucm_toolkit.tools.metrics_view": ["configs/*.json"],
    },
    entry_points={
        "console_scripts": [
            "ucm-toolkit=ucm_toolkit.cli:main",
        ],
    },
)
