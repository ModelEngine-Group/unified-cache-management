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
import shutil
import subprocess
import sys
import sysconfig
import warnings
from glob import glob

import pybind11
import torch
import torch.utils.cpp_extension
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Suppress warnings about packages absent from packages configuration
# These are expected for C++ source directories, test directories, etc.
warnings.filterwarnings(
    "ignore", message=".*Package.*is absent from the `packages` configuration.*"
)

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PLATFORM = os.getenv("PLATFORM")

ENABLE_SPARSE = os.getenv("ENABLE_SPARSE")


def _enable_sparse() -> bool:
    return ENABLE_SPARSE is not None and ENABLE_SPARSE.lower() == "true"


def _is_cuda() -> bool:
    return PLATFORM == "cuda" or (hasattr(torch, "cuda") and torch.cuda.is_available())


def _is_maca() -> bool:
    return PLATFORM == "maca"


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

        self._copy_so_files_to_build_lib()

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
        else:
            cmake_args.append("-DRUNTIME_ENVIRONMENT=ascend")

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

    def _copy_so_files_to_build_lib(self):
        """Copy .so files from source directories to build_lib for installation."""
        if not hasattr(self, "build_lib") or not self.build_lib:
            return

        packages = _get_packages()
        copied_count = 0

        for package in packages:
            # Source directory where CMake outputs .so files
            source_package_dir = os.path.join(ROOT_DIR, package.replace(".", os.sep))

            # Destination in build_lib
            build_package_dir = os.path.join(
                self.build_lib, package.replace(".", os.sep)
            )

            # Find all .so files in the source package directory
            so_files = glob(os.path.join(source_package_dir, "*.so"))

            if so_files:
                # Ensure destination directory exists
                os.makedirs(build_package_dir, exist_ok=True)

                # Copy each .so file
                for so_file in so_files:
                    dest_file = os.path.join(
                        build_package_dir, os.path.basename(so_file)
                    )
                    shutil.copy2(so_file, dest_file)
                    copied_count += 1
                    print(
                        f"[INFO] Copied {os.path.basename(so_file)} to {build_package_dir}"
                    )

        if copied_count > 0:
            print(f"[INFO] Successfully copied {copied_count} .so file(s) to build_lib")
        else:
            print(
                "[WARNING] No .so files found to copy. Extensions may not have been built."
            )


def _get_packages():
    """Discover Python packages, optionally filtering out sparse-related ones."""
    sparse_enabled = _enable_sparse()
    exclude_patterns = []
    if not sparse_enabled:
        exclude_patterns.append("ucm.sparse*")

    packages = find_packages(exclude=exclude_patterns)
    return packages


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
    zip_safe=False,
)
