"""Legacy setuptools entry point for editable installs.

Mirrors the package layout declared in pyproject.toml: the ucm_toolkit tree is
discovered with find_packages, and the ucm_toolkit._native namespace is mapped
onto toolkit/src so the C++/shell sources ship inside the wheel without being
moved out of toolkit/src.
"""

from setuptools import find_packages, setup

setup(
    name="ucm-toolkit",
    version="0.1.0",
    description="Unified CLI for UCM toolkit utilities.",
    python_requires=">=3.9",
    packages=find_packages(include=["ucm_toolkit", "ucm_toolkit.*"])
    + [
        "ucm_toolkit._native",
        "ucm_toolkit._native.dev_sandbox",
        "ucm_toolkit._native.nic_monitor",
    ],
    # dev_sandbox is mapped explicitly because its on-disk dir uses a hyphen.
    package_dir={
        "ucm_toolkit._native": "src",
        "ucm_toolkit._native.dev_sandbox": "src/dev-sandbox",
    },
    include_package_data=True,
    package_data={
        "ucm_toolkit.tools.metrics_view": ["configs/*.json"],
        "ucm_toolkit.tools.precheck": ["*.json"],
    },
    entry_points={
        "console_scripts": [
            "ucm-toolkit=ucm_toolkit.cli:main",
        ],
    },
)
