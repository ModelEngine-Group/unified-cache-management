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

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PLATFORM = os.getenv("PLATFORM")
ENABLE_SPARSE = os.getenv("ENABLE_SPARSE")


def enable_sparse() -> bool:
    return ENABLE_SPARSE is not None and ENABLE_SPARSE.lower() == "true"


def is_editable_mode() -> bool:
    commands = [arg.lower() for arg in sys.argv]
    return "develop" in commands or "--editable" in commands or "-e" in commands


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = ""):
        super().__init__(name, sources=[])
        self.cmake_file_path = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    def run(self):
        build_dir = os.path.abspath(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext: CMakeExtension):
        build_dir = os.path.abspath(self.build_temp)
        install_dir = os.path.abspath(self.build_lib)
        if is_editable_mode:
            install_dir = ext.cmake_file_path

        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]

        if enable_sparse():
            cmake_args += ["-DBUILD_UCM_SPARSE=ON"]

        match PLATFORM:
            case "cuda":
                cmake_args += ["-DRUNTIME_ENVIRONMENT=cuda"]
            case "ascend":
                cmake_args += ["-DRUNTIME_ENVIRONMENT=ascend"]
            case "musa":
                cmake_args += ["-DRUNTIME_ENVIRONMENT=musa"]
            case "maca":
                cmake_args += ["-DRUNTIME_ENVIRONMENT=maca"]
                cmake_args += ["-DBUILD_UCM_SPARSE=OFF"]
            case _:
                cmake_args += ["-DRUNTIME_ENVIRONMENT=simu"]
                cmake_args += ["-DBUILD_UCM_SPARSE=OFF"]

        subprocess.check_call(
            ["cmake", *cmake_args, ext.cmake_file_path], cwd=build_dir
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "--", "-j8"],
            cwd=build_dir,
        )

        subprocess.check_call(
            ["cmake", "--install", ".", "--config", "Release", "--component", "ucm"],
            cwd=build_dir,
        )


setup(
    name="uc-manager",
    version="0.1.2",
    description="Unified Cache Management",
    author="Unified Cache Team",
    packages=find_packages(),
    python_requires=">=3.10",
    ext_modules=[CMakeExtension(name="ucm", source_dir=ROOT_DIR)],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    include_package_data=False,
)
