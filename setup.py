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

import pybind11
import torch
import torch.utils.cpp_extension
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
STORE_SRC_DIR = ROOT_DIR
GSA_SRC_DIR = os.path.join(ROOT_DIR, "ucm", "csrc", "gsaoffloadops")
PREFETCH_SRC_DIR = os.path.join(ROOT_DIR, "ucm", "csrc", "ucmprefetch")
RETRIEVAL_SRC_DIR = os.path.join(ROOT_DIR, "ucm", "csrc", "esaretrieval")
KVSTAR_RETRIEVAL_SRC_DIR = os.path.join(ROOT_DIR, "ucm", "csrc", "kvstar_retrieve")

GSA_INSTALL_DIR = os.path.join(ROOT_DIR, "ucm", "ucm_sparse")
RETRIEVAL_INSTALL_DIR = os.path.join(ROOT_DIR, "ucm", "ucm_sparse", "retrieval")
KVSTAR_RETRIEVAL_INSTALL_DIR = os.path.join(ROOT_DIR, "ucm", "ucm_sparse")

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
        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

        cmake_args = [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYTHON3_EXECUTABLE={sys.executable}",
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
        else:
            raise RuntimeError(
                "No supported accelerator found. "
                "Please ensure either CUDA or NPU is available."
            )

        cmake_args.append(ext.sourcedir)

        print(f"[INFO] Building {ext.name} module with CMake")
        print(f"[INFO] Source directory: {ext.sourcedir}")
        print(f"[INFO] Build directory: {build_dir}")
        print(f"[INFO] CMake command: {' '.join(cmake_args)}")

        subprocess.check_call(cmake_args, cwd=build_dir)

        if ext.name in ["store", "gsa_offload_ops", "esaretrieval", "kvstar_retrieve"]:
            subprocess.check_call(["make", "-j", "8"], cwd=build_dir)
        else:
            # 对于gsa_prefetch使用cmake --build
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release", "--", "-j8"],
                cwd=build_dir,
            )
        if not ext.name == "store":
            self._copy_so_files(ext)

    def _copy_so_files(self, ext: CMakeExtension):
        """复制编译好的.so文件"""
        so_search_dir = os.path.join(ext.sourcedir, "output", "lib")
        if not os.path.exists(so_search_dir):
            raise FileNotFoundError(f"{so_search_dir} does not exist!")

        so_files = []
        search_patterns = [ext.name]

        if ext.name == "gsa_offload_ops":
            search_patterns.extend(["gsa_offload_ops"])
        elif ext.name == "gsa_prefetch":
            search_patterns.extend(["prefetch"])
        elif ext.name == "esaretrieval":
            search_patterns.extend(["retrieval_backend"])
        elif ext.name == "kvstar_retrieve":
            search_patterns.extend(["kvstar_retrieve"])

        for file in os.listdir(so_search_dir):
            if file.endswith(".so") or ".so." in file:
                for pattern in search_patterns:
                    if pattern in file:
                        so_files.append(file)
                        break

        if ext.name == "esaretrieval":
            install_dir = RETRIEVAL_INSTALL_DIR
            build_install_dir = "ucm/ucm_sparse/retrieval"
        elif ext.name == "kvstar_retrieve":
            install_dir = KVSTAR_RETRIEVAL_INSTALL_DIR
            build_install_dir = "ucm/ucm_sparse/kvstar_retrieve"
        else:
            install_dir = GSA_INSTALL_DIR
            build_install_dir = "ucm/ucm_sparse"

        for so_file in so_files:
            src_path = os.path.join(so_search_dir, so_file)
            dev_path = os.path.join(install_dir, so_file)
            dst_path = os.path.join(self.build_lib, build_install_dir, so_file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
            print(f"[INFO] Copied {so_file} → {dst_path}")

            if isinstance(self.distribution.get_command_obj("develop"), develop):
                os.makedirs(os.path.dirname(dev_path), exist_ok=True)
                shutil.copy(src_path, dev_path)
                print(f"[INFO] Copied in editable mode {so_file} → {dev_path}")


ext_modules = []
ext_modules.append(CMakeExtension(name="store", sourcedir=STORE_SRC_DIR))
ext_modules.append(CMakeExtension(name="gsa_offload_ops", sourcedir=GSA_SRC_DIR))
ext_modules.append(CMakeExtension(name="gsa_prefetch", sourcedir=PREFETCH_SRC_DIR))
ext_modules.append(CMakeExtension(name="esaretrieval", sourcedir=RETRIEVAL_SRC_DIR))
ext_modules.append(
    CMakeExtension(name="kvstar_retrieve", sourcedir=KVSTAR_RETRIEVAL_SRC_DIR)
)

setup(
    name="ucm",
    version="0.0.2",
    description="Unified Cache Management",
    author="Unified Cache Team",
    packages=find_packages(),
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
