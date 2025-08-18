import os
import shutil
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "unifiedcache", "csrc", "ucmnfsstore")
INSTALL_DIR = os.path.join(ROOT_DIR, "unifiedcache", "ucm_connector")


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

        subprocess.check_call(
            [
                "cmake",
                "-DDOWNLOAD_DEPENDENCE=ON",
                "-DRUNTIME_ENVIRONMENT=cuda",
                ext.sourcedir,
            ],
            cwd=build_dir,
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
        dst_path = os.path.join(
            self.build_lib, "unifiedcache", "ucm_connector", so_file
        )
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)
        print(f"[INFO] Copied {src_path} → {dst_path}")
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            shutil.copy(src_path, dev_path)
            print(f"[INFO] Copied in editable mode {src_path} → {dev_path}")


setup(
    name="unifiedcache",
    version="0.1.0",
    description="Unified Cache Management with C++ extension",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    ext_modules=[CMakeExtension(name="ucmnfsstore", sourcedir=SRC_DIR)],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
