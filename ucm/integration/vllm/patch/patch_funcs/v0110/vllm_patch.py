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

"""Monkey patches for vLLM 0.11.0 (UCM)."""

from __future__ import annotations

from ucm.logger import init_logger

logger = init_logger(__name__)


def _apply_vllm_patches() -> None:
    """Apply all patches for vLLM 0.11.0."""
    _patch_kv_connector_base_v1()
    _patch_attention_layer()
    logger.info("vLLM 0.11.0 patches applied successfully")


def _patch_kv_connector_base_v1() -> None:
    """Add get_block_ids_with_load_errors to KVConnectorBase_V1."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import base as base_mod

        if getattr(base_mod, "__ucm_patched__", False):
            return

        KVConnectorBase_V1 = base_mod.KVConnectorBase_V1

        # add has_connector_metadata to KVConnectorBase_V1
        def has_connector_metadata(self) -> bool:
            """Check whether the connector metadata is currently set.

            Returns:
                bools: True if connector metadata exists, False otherwise.
            """
            return self._connector_metadata is not None

        KVConnectorBase_V1.has_connector_metadata = has_connector_metadata
        base_mod.__ucm_patched__ = True
    except ImportError as e:
        logger.warning("Could not patch KVConnectorBase_V1: %s", e)


def _patch_attention_layer() -> None:
    """Patch attention layer"""
    try:
        # IMPORTANT: Rebinding names imported with
        # `from vllm.attention.layer import foo` only updates local variables in
        # this patch module, NOT the original `vllm.attention.layer` module.
        # We must patch the module attributes directly.
        import vllm.attention.layer as layer_mod

        if getattr(layer_mod, "__ucm_patched__", False):
            return

        def patched_wait_for_kv_layer_from_connector(layer_name: str) -> None:
            # Keep imports local to reduce import-time coupling across vLLM versions.
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            # Guard for older connectors to avoid AttributeError.
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is None:
                return
            assert isinstance(attn_metadata, dict)
            connector.wait_for_layer_load(layer_name)

        def patched_maybe_save_kv_layer_to_connector(
            layer_name: str,
            kv_cache_layer: "list[object]",
        ) -> None:
            from vllm.distributed.kv_transfer import (
                get_kv_transfer_group,
                has_kv_transfer_group,
                is_v1_kv_transfer_group,
            )
            from vllm.forward_context import get_forward_context

            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return

            connector = get_kv_transfer_group()
            if (
                hasattr(connector, "has_connector_metadata")
                and not connector.has_connector_metadata()
            ):
                return

            attn_metadata = get_forward_context().attn_metadata
            if attn_metadata is None:
                return
            assert isinstance(attn_metadata, dict)
            connector.save_kv_layer(
                layer_name, kv_cache_layer, attn_metadata[layer_name]
            )

        layer_mod.wait_for_kv_layer_from_connector = (
            patched_wait_for_kv_layer_from_connector
        )
        layer_mod.maybe_save_kv_layer_to_connector = (
            patched_maybe_save_kv_layer_to_connector
        )
        layer_mod.__ucm_patched__ = True
    except Exception as e:
        logger.warning("Could not patch attention layer: %s", e)
