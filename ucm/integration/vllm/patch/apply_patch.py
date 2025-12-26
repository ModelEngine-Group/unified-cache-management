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

import sys
from typing import Optional

from ucm.logger import init_logger

logger = init_logger(__name__)

import os

PLATFORM = os.getenv("PLATFORM")

vllm_use_rerope = os.getenv("VLLM_USE_REROPE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _patch_ascend() -> bool:
    return PLATFORM == "ascend"


# Track if patches have been applied
_patches_applied = False
_import_hook_installed = False
_vllm_version: Optional[str] = None
_vllm_import_hook = None


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
    """Get list of supported vLLM versions."""
    return ["0.9.2"]


def apply_all_patches() -> None:
    """Apply all vLLM patches based on detected version."""
    global _patches_applied
    if _patches_applied:
        return

    try:
        version = get_vllm_version()
        if version is None:
            raise ValueError("Could not detect vLLM version")

        supported_versions = get_supported_versions()
        if version not in supported_versions:
            logger.warning(
                f"vLLM version {version} is not explicitly supported to apply UCM patches. "
                f"Supported versions: {', '.join(supported_versions)}. "
            )

        # Apply version-specific patches
        match version:
            case "0.9.2" if vllm_use_rerope:
                _apply_patches_rerope()
            case "0.9.2":
                _apply_patches_v092()
            case _:
                logger.warning(
                    f"Unsupported vLLM version: {version} to apply UCM patches. "
                    f"Supported versions: {', '.join(supported_versions)}."
                )

        _patches_applied = True
        logger.info(f"All vLLM patches applied successfully for version {version}")
    except Exception as e:
        logger.error(f"Failed to apply vLLM patches: {e}", exc_info=True)
        raise


def _apply_patches_v092() -> None:
    """Apply patches for vLLM 0.9.2."""
    from .patch_funcs.v092.vllm_patch import _apply_sparse_adapt

    _apply_sparse_adapt()  # apply vllm-sparse-adapt.patch
    if _patch_ascend():
        from .patch_funcs.v092.vllm_ascend_patch import _apply_ascend_patch

        _apply_ascend_patch()  # apply vllm-ascend-adapt.patch


def _apply_patches_rerope() -> None:
    """Apply patches for vLLM 0.9.2 for triton rerope"""
    from .patch_funcs.v092.vllm_rerope_patch import _apply_rerope_adapt_patches

    _apply_rerope_adapt_patches()


def install_import_hook() -> None:
    """Install an import hook to automatically apply patches when vLLM is imported."""
    global _import_hook_installed, _vllm_import_hook

    if _import_hook_installed:
        return

    try:
        # Check if vLLM is already imported
        if "vllm" in sys.modules:
            # vLLM already imported, apply patches immediately
            apply_all_patches()
            _import_hook_installed = True
        else:
            # Install import hook by wrapping the builtin __import__ function
            # This intercepts all imports and applies patches when vLLM is imported
            import builtins

            original_import = builtins.__import__

            def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
                # Call original import
                module = original_import(name, globals, locals, fromlist, level)

                # If the main vLLM module is being imported, apply patches
                # We only check for 'vllm' (not submodules) to avoid multiple patch attempts
                if name == "vllm" and not _patches_applied:
                    try:
                        apply_all_patches()
                    except Exception as e:
                        logger.warning(f"Failed to apply patches during import: {e}")

                return module

            # Replace builtin __import__
            builtins.__import__ = import_hook
            _vllm_import_hook = import_hook
            _import_hook_installed = True
            logger.debug("Import hook installed to intercept vLLM imports")

    except Exception as e:
        logger.warning(f"Failed to install import hook: {e}")


def ensure_patches_applied() -> None:
    """Ensure patches are applied, installing import hook if needed."""
    if not _patches_applied:
        # Try to apply patches immediately
        try:
            apply_all_patches()
        except Exception:
            # If it fails (vLLM not imported yet), install hook
            install_import_hook()
