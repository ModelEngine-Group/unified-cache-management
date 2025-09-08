import os
import pathlib
import subprocess
import sys

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

# --- 动态查找路径 ---
import sysconfig
import torch.utils
import torch.utils.cpp_extension

# 辅助类，定义C++扩展
class Ext(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(pathlib.Path(sourcedir).resolve())

# 自定义构建命令
class Build(build_ext):
    def build_extension(self, ext: Ext) -> None:
        # 准备基本的 CMake 参数
        executable = sys.executable
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        output_dir = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_type = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DPYTHON_EXECUTABLE={executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]

        # =====================================================================
        # 动态获取所有路径并准备注入

        # 1. 获取 Torch 的 CMAKE_PREFIX_PATH。
        # 令CMake 的 `find_package(Torch)` 命令能够成功执行，从而找到库文件。
        torch_cmake_prefix = torch.utils.cmake_prefix_path
        cmake_args += [f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix}"]

        # 2. 动态获取所有必需的头文件（include）路径。
        torch_includes = torch.utils.cpp_extension.include_paths()
        python_include = sysconfig.get_path('include')
        all_includes = torch_includes + [python_include]

        # 3. 将所有路径合并成一个 CMake 能理解的、用分号分隔的字符串。
        cmake_include_string = ";".join(all_includes)

        # 4. 通过 -D 命令定义一个“动态的、可移植的硬编码”变量，将这个包含所有路径的字符串“注入”到 CMake 中。
        cmake_args += [f"-DEXTERNAL_INCLUDE_DIRS={cmake_include_string}"]
        # =====================================================================

        # 执行 CMake 配置和编译
        print("--- Running CMake with args:", cmake_args)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--config", build_type], cwd=build_temp)

# --- Setuptools 的主配置 ---
setuptools.setup(
    name="kvstar_retrieve",
    version="0.0.1",
    author="HUAWEI 2012 ACS_Lab",
    author_email="acslab@huawei.com",
    description="KVStar Sparse Attention Async Retrieve C++ Lib Framework",
    long_description="",
    ext_modules=[Ext("kvstar_retrieve", sourcedir="csrc")],
    cmdclass={"build_ext": Build},
    zip_safe=False,
)