import glob
import os

import torch
import torch_npu
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

# get torch_npu install path
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))

# get current directory
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# get use ninja
USE_NINJA = os.getenv("USE_NINJA") == "1"


# get source files
source_files = [
    os.path.join(BASE_DIR, "torch_binding.cpp"),
    os.path.join(BASE_DIR, "torch_binding_meta.cpp"),
    os.path.join(BASE_DIR, "aclnn_torch_adapter/NPUBridge.cpp"),
    os.path.join(BASE_DIR, "aclnn_torch_adapter/NPUStorageImpl.cpp"),
]

# build extension
exts = [
    NpuExtension(
        name="ucm_custom_ops",
        sources=source_files,
        extra_compile_args=[
            "-I"
            + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        ],
    )
]


setup(
    name="ucm_custom_ops",
    version="1.1",
    keywords="ucm_custom_ops",
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
