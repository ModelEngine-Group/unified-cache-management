# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import shutil
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from setuptools.command.build_py import build_py

__version__ = "1.7.3"

ROOT = Path(__file__).resolve().parent
NATIVE_PACKAGE = "_pynds_kv_native"
NATIVE_DIR = ROOT / "vllm" / "NDS_bin"
REQUIRED_NATIVE_LIBS = (
    "libndskv.so",
    "libndsclient.so",
)


def native_libraries():
    missing = [
        name for name in REQUIRED_NATIVE_LIBS if not (NATIVE_DIR / name).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Missing native libraries in {NATIVE_DIR}: {', '.join(missing)}. "
            "Run `bash install.sh` instead of invoking pip directly. "
            "For cluster mode, add `--agent-lib-dir /path/to/agent/libs`."
        )
    return sorted(path for path in NATIVE_DIR.glob("*.so*") if path.is_file())


class BuildPy(build_py):
    """Ship every native dependency beside the extension, without absolute rpaths."""

    def run(self):
        libraries = native_libraries()
        super().run()
        target_dir = Path(self.build_lib) / NATIVE_PACKAGE
        target_dir.mkdir(parents=True, exist_ok=True)
        for library in libraries:
            shutil.copy2(library, target_dir / library.name)


ext_modules = [
    Pybind11Extension(
        "pynds_kv",
        ["src/main.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        cxx_std=17,
        include_dirs=[str(ROOT / "vllm" / "NDS_include")],
        library_dirs=[str(NATIVE_DIR)],
        # main.cpp only consumes NDS:: wrapper APIs. The SDK remains a transitive
        # dependency, so process-wide NDS state stays in one libndsclient.so.
        libraries=["ndskv"],
        runtime_library_dirs=(
            [f"$ORIGIN/{NATIVE_PACKAGE}"] if sys.platform != "win32" else []
        ),
        extra_link_args=[f"-Wl,-rpath-link,{NATIVE_DIR}", "-Wl,-z,relro,-z,now"],
    ),
]

setup(
    name="pynds_kv",
    version=__version__,
    author="Li Wei",
    author_email="liwei519@huawei.com",
    url="https://github.com/ModelEngine-Group/unified-cache-management/tree/develop/kv_semantics/patch/pynds_kv",
    description="NDS interface with KV semantics",
    ext_modules=ext_modules,
    packages=[NATIVE_PACKAGE],
    extras_require={"test": ["pytest", "numpy", "xxhash"]},
    cmdclass={"build_ext": build_ext, "build_py": BuildPy},
    zip_safe=False,
    python_requires=">=3.9",
)
