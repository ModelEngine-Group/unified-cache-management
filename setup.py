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
import sys
import sysconfig
from glob import glob

import pybind11
import torch
import torch.utils.cpp_extension
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PLATFORM = os.getenv("PLATFORM")

ENABLE_SPARSE = os.getenv("ENABLE_SPARSE")


def _enable_sparse() -> bool:
    return ENABLE_SPARSE is not None and ENABLE_SPARSE.lower() == "true"


def _is_cuda() -> bool:
    return PLATFORM == "cuda"


def _is_npu() -> bool:
    return PLATFORM == "ascend"


def _is_musa() -> bool:
    return PLATFORM == "musa"


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext: CMakeExtension):
        build_dir = self.build_temp
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        torch_cmake_prefix = torch.utils.cmake_prefix_path
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        cmake_prefix_paths = [torch_cmake_prefix, pybind11_cmake_dir]
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={';'.join(cmake_prefix_paths)}")

        torch_includes = torch.utils.cpp_extension.include_paths()
        python_include = sysconfig.get_path("include")
        pybind11_include = pybind11.get_include()

        all_includes = torch_includes + [python_include, pybind11_include]
        cmake_include_string = ";".join(all_includes)
        cmake_args.append(f"-DEXTERNAL_INCLUDE_DIRS={cmake_include_string}")

        if _is_cuda():
            cmake_args.append("-DRUNTIME_ENVIRONMENT=cuda")
        elif _is_npu():
            cmake_args.append("-DRUNTIME_ENVIRONMENT=ascend")
        elif _is_musa():
            cmake_args.append("-DRUNTIME_ENVIRONMENT=musa")
        else:
            raise RuntimeError(
                "No supported accelerator found. "
                "Please ensure either CUDA/MUSA or NPU is available."
            )

        if _enable_sparse():
            cmake_args.append("-DBUILD_UCM_SPARSE=ON")

        cmake_args.append(ext.sourcedir)

        print(f"[INFO] Building {ext.name} module with CMake")
        print(f"[INFO] Source directory: {ext.sourcedir}")
        print(f"[INFO] Build directory: {build_dir}")
        print(f"[INFO] CMake command: {' '.join(cmake_args)}")

        subprocess.check_call(cmake_args, cwd=build_dir)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "--", "-j8"],
            cwd=build_dir,
        )


def _get_packages():
    """Discover Python packages, optionally filtering out sparse-related ones."""
    packages = find_packages()
    if not _enable_sparse():
        packages = [pkg for pkg in packages if not pkg.startswith("ucm.sparse")]
    return packages


def _get_package_data_with_so(packages=None):
    """Automatically discover all packages and include .so files."""
    if packages is None:
        packages = _get_packages()
    package_data = {}

    for package in packages:
        # Convert package name to directory path
        package_dir = os.path.join(ROOT_DIR, package.replace(".", os.sep))

        # Check if this package directory contains .so files
        so_files = glob(os.path.join(package_dir, "*.so"))
        if so_files:
            package_data[package] = ["*.so"]
            print(f"[INFO] Including .so files for package: {package}")

    print(f"[INFO] Package data: {package_data}")
    return package_data


ext_modules = []
ext_modules.append(CMakeExtension(name="ucm", sourcedir=ROOT_DIR))

packages = _get_packages()

setup(
    name="uc-manager",
    version="0.1.1",
    description="Unified Cache Management",
    author="Unified Cache Team",
    packages=packages,
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    package_data=_get_package_data_with_so(packages),
    zip_safe=False,
)
