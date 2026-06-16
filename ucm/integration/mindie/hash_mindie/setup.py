from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "uc_hash_ext",
        ["uc_hash_ext.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
        extra_link_args=["-Wl,-z,relro,-z,now", "-Wl,-s"],
    )
]

setup(
    name="uc_hash_ext",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
