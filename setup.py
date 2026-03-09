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

import atexit
import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PLATFORM = os.getenv("PLATFORM")
ENABLE_SPARSE = os.getenv("ENABLE_SPARSE")


_warning_printed = False


def print_platform_warning():
    global _warning_printed
    if not PLATFORM and not _warning_printed:
        _warning_printed = True
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        warning_msg = f"""
{RED}{'=' * 80}
{BOLD}⚠️  WARNING: PLATFORM environment variable is not set! ⚠️{RESET}
{RED}{'=' * 80}{RESET}
{YELLOW}Please set PLATFORM to one of: cuda, ascend, musa, maca{RESET}
Example:
  {BOLD}export PLATFORM=cuda{RESET}    # For CUDA platform
{YELLOW}In CI scenarios only, you don't need to specify PLATFORM. If it's not a CI scenario, please uninstall and then reinstall with PLATFORM specified.{RESET}
{RED}{'=' * 80}{RESET}
"""
        # Use write and flush to ensure output even without -v flag
        sys.stderr.write(warning_msg)
        sys.stderr.flush()


if not PLATFORM:
    atexit.register(print_platform_warning)


def enable_sparse() -> bool:
    return ENABLE_SPARSE is not None and ENABLE_SPARSE.lower() == "true"


def is_editable_mode() -> bool:
    commands = [arg.lower() for arg in sys.argv]
    return (
        "develop" in commands
        or "--editable" in commands
        or "-e" in commands
        or "editable_wheel" in commands
    )


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

        if enable_sparse() and PLATFORM == "ascend":
            try:
                print(
                    "Running bash ucm/sparse/gsa_on_device/csrc/ascend/build_aclnn.sh to compiling NPU custom ops for UCM..."
                )
                subprocess.check_call(
                    ["bash", "ucm/sparse/gsa_on_device/csrc/ascend/build_aclnn.sh"]
                )
                print(
                    "ucm/sparse/gsa_on_device/csrc/ascend/buid_aclnn.sh executed successfully!"
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"Error running ucm/sparse/gsa_on_device/csrc/ascend/build_aclnn.sh: {e}"
                )
                raise SystemExit(e.returncode)

    def build_cmake(self, ext: CMakeExtension):
        build_dir = os.path.abspath(self.build_temp)
        install_dir = os.path.abspath(self.build_lib)
        if is_editable_mode():
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


def inject_pth():
    if not ("-e" in sys.argv or "develop" in sys.argv or "editable_wheel" in sys.argv):
        return

    import site

    pth_name = "ucm_patch.pth"
    source = os.path.abspath(pth_name)

    if not os.path.exists(source):
        print(f"Error: {pth_name} not found in root directory.")
        return

    try:
        try:
            site_packages = site.getsitepackages()[0]
        except AttributeError:
            from distutils.sysconfig import get_python_lib

            site_packages = get_python_lib()

        target = os.path.join(site_packages, pth_name)

        if not os.path.exists(target):
            if sys.platform == "win32":
                import shutil

                shutil.copy(source, target)
            else:
                os.symlink(source, target)
            print("Injection successful.")

    except Exception as e:
        print(f"\033[93mWarning: Failed to inject .pth for editable mode: {e}\033[0m")


setup(
    name="uc-manager",
    version="0.3.0",
    description="Unified Cache Management",
    author="Unified Cache Team",
    packages=find_packages() + [""],
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires="wrapt>=2.0.1",
    ext_modules=[CMakeExtension(name="ucm", source_dir=ROOT_DIR)],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    include_package_data=False,
    package_data={
        "ucm": ["sparse/gsa_on_device/configs/**/*.json"],
        "": ["ucm_patch.pth"],
    },
)
if any(arg in sys.argv for arg in ["-e", "develop", "editable_wheel"]):
    inject_pth()
