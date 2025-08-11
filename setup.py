#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import subprocess
from distutils.core import setup
from pathlib import Path

from setuptools import Extension, find_packages
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.dirname(__file__)
PLATFORM = os.getenv("PLATFORM")


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def _is_cuda() -> bool:
    return PLATFORM == "cuda"


def _is_npu() -> bool:
    return PLATFORM == "ascend"


class BuildUCMExtension(build_ext):
    """Build UCM Extensions Using Cmake"""

    def run(self):
        package_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "unifiedcache")
        )
        ucm_nfs_path = os.path.join(package_path, "csrc", "ucmnfsstore")
        if not os.path.exists(ucm_nfs_path):
            raise RuntimeError(f"Expected directory {ucm_nfs_path} does not exist")

        build_path = os.path.join(ucm_nfs_path, "build")
        if not os.path.exists(build_path):
            os.makedirs(build_path)

        os.chdir(build_path)
        if _is_npu():
            cmake_command = [
                "cmake",
                "-DDOWNLOAD_DEPENDENCE=ON",
                "-DRUNTIME_ENVIRONMENT=ascend",
                "..",
                ucm_nfs_path,
            ]
        elif _is_cuda():
            cmake_command = [
                "cmake",
                "-DDOWNLOAD_DEPENDENCE=ON",
                "-DRUNTIME_ENVIRONMENT=cuda",
                "..",
                ucm_nfs_path,
            ]
        else:
            raise RuntimeError(
                "No supported accelerator found. "
                "Please ensure either CUDA or NPU is available."
            )
        subprocess.check_call(cmake_command)

        make_command = ["make", "-j", "8"]
        subprocess.check_call(make_command)

        output_lib_path = os.path.join(ucm_nfs_path, "output", "lib")
        so_files = [f for f in os.listdir(output_lib_path) if f.endswith(".so")]
        for so_file in so_files:
            src = os.path.join(output_lib_path, so_file)
            dest = os.path.join(package_path, "ucm_connector", so_file)
            os.rename(src, dest)

        os.chdir(os.path.dirname(__file__))
        super().run()


cmdclass = {
    "build_ext": BuildUCMExtension,
}

ext_modules = [
    Extension(
        name="unifiedcache.ucm_connector.ucmnfsstore",
        sources=[],
    )
]

print("FOUND PACKAGES:", find_packages())
setup(
    name="unifiedcache",
    version="0.0.1",
    author="Unified Cache Team",
    description="Unified Cache Management",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    package_data={
        "unifiedcache.ucm_connector": ["*.so"],
    },
    include_package_data=True,
    install_requires=[],
    extras_require={},
    python_requires=">=3.10",
)
