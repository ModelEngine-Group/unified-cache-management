# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "1.7.1"

# libndskv.so（C++ 封装层，vllm/Makefile 的 make 产物）所在目录
LIB_DIR_WRAPPER = os.path.abspath("./vllm/NDS_bin")
# vendor libndsclient.so 所在目录：仅链接期供 ld 解析 libndskv.so 的传递依赖，
# 不进 runtime rpath —— 运行时由 libndskv.so 自带的 $ORIGIN/../NDS_lib 定位
LIB_DIR_NDS_SDK = os.path.abspath("./vllm/NDS_lib")
INC_DIR = os.path.abspath("./vllm/NDS_include")

ext_modules = [
    Pybind11Extension(
        "pynds_kv",
        ["src/main.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=[INC_DIR],
        library_dirs=[LIB_DIR_WRAPPER, LIB_DIR_NDS_SDK],
        # 只直接链接 libndskv：main.cpp 仅调用 NDS:: 封装接口（nds_interface.h），
        # 不直接调用 ndsclient 符号。libndsclient 由 libndskv.so 经自身
        # $ORIGIN rpath 加载，全进程只此一份，避免 pynds_kv.so 再 DT_NEEDED
        # 一份导致绝对路径双重依赖乃至双份加载、全局状态分裂。
        libraries=["ndskv"],
        # pip install . 后 pynds_kv.so 落在 site-packages，只能靠绝对路径 rpath
        # 找回仓库内的 NDS_bin/libndskv.so。注意：仓库目录移动后需重新 pip install。
        runtime_library_dirs=[LIB_DIR_WRAPPER] if sys.platform != "win32" else [],
    ),
]

setup(
    name="pynds_kv",
    version=__version__,
    description="NDS interface with KV semantics",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
