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


"""Monkey patches for vLLM Ascend 0.11.0 (UCM)."""

from __future__ import annotations

import types
from typing import Any

from ucm.logger import init_logger

logger = init_logger(__name__)


def _swap_function_impl_in_place(original: Any, replacement: Any) -> bool:
    """Swap a Python function's implementation without changing its identity.

    This is critical when callers used `from module import func` before patching:
    rebinding `module.func = new_func` won't update already-imported local names,
    but mutating the *original function object* will.

    Returns:
        bool: True if in-place swapping succeeded, False otherwise.
    """
    if not isinstance(original, types.FunctionType):
        return False
    if not isinstance(replacement, types.FunctionType):
        return False

    try:
        # NOTE: This keeps `original`'s identity and `__globals__` unchanged.
        # Only the executable payload and call defaults are swapped.
        original.__code__ = replacement.__code__
        original.__defaults__ = replacement.__defaults__
        original.__kwdefaults__ = replacement.__kwdefaults__

        # Keep these best-effort; they should not affect runtime behavior.
        try:
            original.__annotations__ = dict(getattr(replacement, "__annotations__", {}))
        except Exception:
            pass
        try:
            original.__doc__ = replacement.__doc__
        except Exception:
            pass

        return True
    except Exception:
        return False


def _apply_ascend_patches() -> None:
    """Apply all patches for vLLM 0.11.0 on Ascend."""
    _patch_ascend_attention_layer()
    logger.info("vLLM Ascend 0.11.0 patches applied successfully")


def _patch_ascend_attention_layer() -> None:
    """Patch ascend attention layer"""
    try:
        # vLLM-Ascend uses its own attention implementation; patch its module
        # rather than `vllm.attention.layer`.
        from typing import List

        import torch
        from vllm_ascend.attention import utils as va_utils_mod

        if getattr(va_utils_mod, "__ucm_patched__", False):
            return

        # TODO(ucm): Implement load-failure recovery hooks for vLLM-Ascend 0.11.0.
        # For now, keep as a no-op placeholder so that Ascend 0.11.0 can enable the
        # patch pipeline without breaking imports.
        def patched_wait_for_kv_layer_from_connector(layer_name: str):
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import ForwardContext, get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            # TODO: assert ascendMetadata
            connector.wait_for_layer_load(layer_name)

        def patched_maybe_save_kv_layer_to_connector(
            layer_name: str,
            kv_cache_layer: List[torch.Tensor],
        ):
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import ForwardContext, get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return
            # TODO: assert ascendMetadata
            connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)

        # Prefer in-place swapping so that earlier `from ... import ...` bindings
        # also observe the patched implementation.
        old_wait = getattr(va_utils_mod, "wait_for_kv_layer_from_connector", None)
        if old_wait is not None and _swap_function_impl_in_place(
            old_wait, patched_wait_for_kv_layer_from_connector
        ):
            logger.info(
                "Patched vllm_ascend.attention.utils.wait_for_kv_layer_from_connector "
                "via in-place code swap"
            )
        else:
            va_utils_mod.wait_for_kv_layer_from_connector = (
                patched_wait_for_kv_layer_from_connector
            )
            logger.info(
                "Patched wait_for_kv_layer_from_connector via rebinding. "
                "NOTE: callers that imported it via `from ... import ...` before "
                "patching may still hold the old reference."
            )

        old_save = getattr(va_utils_mod, "maybe_save_kv_layer_to_connector", None)
        if old_save is not None and _swap_function_impl_in_place(
            old_save, patched_maybe_save_kv_layer_to_connector
        ):
            logger.info(
                "Patched vllm_ascend.attention.utils.maybe_save_kv_layer_to_connector "
                "via in-place code swap"
            )
        else:
            va_utils_mod.maybe_save_kv_layer_to_connector = (
                patched_maybe_save_kv_layer_to_connector
            )
            logger.info(
                "Patched maybe_save_kv_layer_to_connector via rebinding. "
                "NOTE: callers that imported it via `from ... import ...` before "
                "patching may still hold the old reference."
            )

        va_utils_mod.__ucm_patched__ = True  # type: ignore[attr-defined]
    except ImportError as e:
        logger.warning(
            "Could not patch ascend attention layer (vllm_ascend missing): %s", e
        )
    except Exception as e:
        logger.warning("Could not patch ascend attention layer: %s", e)
