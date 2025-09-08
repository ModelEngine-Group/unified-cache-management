from setuptools import setup, Extension
import pybind11
import torch
import os

PYTORCH_PATH = os.path.dirname(os.path.abspath(torch.__file__))

INCLUDE_DIRS = [
    os.path.join(PYTORCH_PATH, 'include/torch/csrc/api/include'),
    os.path.join(PYTORCH_PATH, 'include'),
    pybind11.get_include(),
    './include'
]

LIBRARY_DIRS = [
    os.path.join(PYTORCH_PATH, 'lib'),
    '/usr/local/lib'
]

LIBRARIES = [
    'torch',
    'c10',
    'torch_cpu',
    'torch_python',
    'gomp',
    'pthread'
]

# 定义C++扩展模块
ext_module = Extension(
    'gsa_offload_ops',
    sources=[
        'src/thread_safe_queue.cpp',
        'src/vec_product.cpp',
        'src/k_repre.cpp',
        'src/select_topk_block.cpp',
        'src/cal_kpre_and_topk.cpp',
        'src/pybinds.cpp'],
    include_dirs=INCLUDE_DIRS,
    library_dirs=LIBRARY_DIRS,
    libraries=LIBRARIES,
    language='c++',
    extra_compile_args=[
        '-std=c++17',
        '-D_GLIBCXX_USE_CXX11_ABI=1',
        '-O3',
        '-fopenmp',
        '-march=native']
)

setup(
    name='gsa_offload_ops',
    version='0.1',
    ext_modules=[ext_module],
)
