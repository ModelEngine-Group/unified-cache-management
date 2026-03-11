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
"""
Monkey patching module for vLLM to apply UCM patches automatically.
This replaces the need for manual `git apply` commands.
"""

from typing import Optional

from ucm.logger import init_logger

logger = init_logger(__name__)

import os

ENABLE_SPARSE = os.getenv("ENABLE_SPARSE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ENABLE_UCM_PATCH = os.environ.get("ENABLE_UCM_PATCH", "").lower() in ("1", "true")


def get_vllm_ascend_version() -> Optional[str]:
    """Detect vllm_ascend version if installed.

    Note: keep it simple and robust (no hard import required).
    """

    def _norm(v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = str(v).strip()
        # common suffixes: 0.11.0+xxx / 0.11.0.post1
        v = v.split("+", 1)[0]
        v = v.split(".post", 1)[0]
        return v

    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return _norm(version("vllm-ascend"))
        except PackageNotFoundError:
            return None
    except Exception:
        pass

    try:
        import importlib

        mod = importlib.import_module("vllm_ascend")
        return _norm(getattr(mod, "__version__", None))
    except Exception:
        return None


_vllm_version: Optional[str] = None


def get_vllm_version() -> Optional[str]:
    """Detect vLLM version."""
    global _vllm_version
    if _vllm_version is not None:
        return _vllm_version

    try:
        # Try to get version from vllm module
        import vllm as vllm_pkg

        vllm_version = vllm_pkg.__version__
        return vllm_version
    except ImportError:
        logger.warning("vLLM is not installed")
        return None
    except Exception as e:
        logger.warning(f"Failed to detect vLLM version: {e}")
        return None


def get_supported_versions() -> list[str]:
    """Get patch-required vLLM versions."""
    return ["0.11.0"]


def apply_all_patches() -> None:
    """Apply all vLLM patches based on detected version."""
    version: Optional[str] = None

    try:
        from ucm.integration.vllm.patch.logger_patch import patch_logger

        if not ENABLE_UCM_PATCH:
            return

        version = get_vllm_version()
        if version is None:
            raise ValueError("Could not detect vLLM version")

        supported_versions = get_supported_versions()
        if version not in supported_versions:
            logger.warning(
                f"vLLM version {version} detected. No patch application needed for this version. "
                f"Versions applicable for UCM patches: {', '.join(supported_versions)}. "
            )

        # Apply common patch here

        # Apply vllm/vllm-ascendversion-specific patches
        # vllm patches
        match version:
            case "0.11.0":
                import ucm.integration.vllm.patch.v0110.vllm.pc_patch

                if ENABLE_SPARSE:
                    import ucm.integration.vllm.patch.v0110.vllm.sparse_patch


        # vllm_ascend patches
        ascend_version = get_vllm_ascend_version()
        match ascend_version:
            case None:
                pass
            case "0.11.0":
                import ucm.integration.vllm.patch.v0110.vllm_ascend.pc_ascend_patch

                if ENABLE_SPARSE:
                    import ucm.integration.vllm.patch.v0110.vllm_ascend.sparse_ascend_patch


        logger.info(f"All vLLM patches applied successfully for version {version}")

    except Exception as e:
        logger.error(f"Failed to apply vLLM patches: {e}\n")
        raise
