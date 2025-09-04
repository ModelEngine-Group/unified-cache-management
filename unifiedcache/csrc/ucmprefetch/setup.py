from setuptools import setup, Extension
import pybind11
import torch
import os
import sys

def parse_args():
    device = "cpu"
    if "--cuda" in sys.argv:
        device = "cuda"
        sys.argv.remove("--cuda")  # 移除参数，避免setuptools报错
    elif "--npu" in sys.argv:
        device = "npu"
        sys.argv.remove("--npu")
    print(f"[INFO] Compiling for device: {device}")
    return device

target_device = parse_args()

PYTORCH_PATH = os.path.dirname(os.path.abspath(torch.__file__))

INCLUDE_DIRS = []
LIBRARIES = []
INCLUDE_DIRS = []
CXX11_API = "1"

if target_device == "npu":
    import torch_npu
    PYTORCH_NPU_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
    INCLUDE_DIRS = [
        os.path.join(PYTORCH_NPU_PATH, 'include'),
    ]
    LIBRARY_DIRS = [
        os.path.join(PYTORCH_NPU_PATH, 'lib'),
        '/usr/local/lib'
    ]
    LIBRARIES = ['torch_npu']
    CXX11_API = "0"

INCLUDE_DIRS += [
    os.path.join(PYTORCH_PATH, 'include/torch/csrc/api/include'),
    os.path.join(PYTORCH_PATH, 'include'),
    pybind11.get_include(),
    './include'
]

LIBRARY_DIRS += [
    os.path.join(PYTORCH_PATH, 'lib'),
    '/usr/local/lib'
]

LIBRARIES += [
    'torch',
    'c10',
    'torch_cpu',
    'torch_python',
    'gomp',
    'pthread'
]

# 定义C++扩展模块
ext_module = Extension(
    'gsa_prefetch',
    sources=[
        'src/pybinds.cpp',
        'src/kvcache_pre.cpp'],

    include_dirs=INCLUDE_DIRS,
    library_dirs=LIBRARY_DIRS,
    libraries=LIBRARIES,
    language='c++',
    extra_compile_args=[
        '-std=c++17',
        '-D_GLIBCXX_USE_CXX11_ABI=' + CXX11_API,
        '-O3',
        '-fopenmp',
        '-march=native']
)

setup(
    name='gsa_prefetch',
    version='0.1',
    ext_modules=[ext_module],
)