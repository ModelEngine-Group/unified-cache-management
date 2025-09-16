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

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "ucm", "csrc", "ucmnfsstore")
INSTALL_DIR = os.path.join(ROOT_DIR, "ucm", "store")
PLATFORM = os.getenv("PLATFORM")


def _is_cuda() -> bool:
    return PLATFORM == "cuda"


def _is_npu() -> bool:
    return PLATFORM == "ascend"


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext: CMakeExtension):
        build_dir = os.path.abspath(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)
        if _is_cuda():
            subprocess.check_call(
                [
                    "cmake",
                    "-DDOWNLOAD_DEPENDENCE=ON",
                    "-DRUNTIME_ENVIRONMENT=cuda",
                    ext.sourcedir,
                ],
                cwd=build_dir,
            )
        elif _is_npu():
            subprocess.check_call(
                [
                    "cmake",
                    "-DDOWNLOAD_DEPENDENCE=ON",
                    "-DRUNTIME_ENVIRONMENT=ascend",
                    ext.sourcedir,
                ],
                cwd=build_dir,
            )
        else:
            raise RuntimeError(
                "No supported accelerator found. "
                "Please ensure either CUDA or NPU is available."
            )

        subprocess.check_call(["make", "-j", "8"], cwd=build_dir)

        so_file = None
        so_search_dir = os.path.join(ext.sourcedir, "output", "lib")
        if not os.path.exists(so_search_dir):
            raise FileNotFoundError(f"{so_search_dir} does not exist!")

        so_file = None
        for file in os.listdir(so_search_dir):
            if file.startswith("ucmnfsstore") and file.endswith(".so"):
                so_file = file
                break

        if not so_file:
            raise FileNotFoundError(
                "Compiled .so file not found in output/lib directory."
            )

        src_path = os.path.join(so_search_dir, so_file)
        dev_path = os.path.join(INSTALL_DIR, so_file)
        dst_path = os.path.join(self.build_lib, "ucm", "store", so_file)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
        print(f"[INFO] Copied {src_path} → {dst_path}")
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            shutil.copy(src_path, dev_path)
            print(f"[INFO] Copied in editable mode {src_path} → {dev_path}")


setup(
    name="ucm",
    version="0.0.2",
    description="Unified Cache Management",
    author="Unified Cache Team",
    packages=find_packages(),
    python_requires=">=3.10",
    ext_modules=[CMakeExtension(name="ucmnfsstore", sourcedir=SRC_DIR)],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
