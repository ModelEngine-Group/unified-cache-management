# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer."""
from typing import Any, Dict, List, Optional

import torch

from vllm.forward_context import ForwardContext, get_forward_context

from unifiedcache.integration.vllm.ucm_sparse.state import get_ucm_sparse, has_ucm_sparse
from vllm.attention.layer import wait_for_kv_layer_from_connector, maybe_save_kv_layer_to_connector


def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    wait_for_kv_layer_from_connector(layer_name)

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    maybe_execute_sparse_attention_begin(query, key, value, layer_name, forward_context)
    output = self.impl.forward(self, query, key, value, kv_cache,
                               attn_metadata)

    maybe_execute_sparse_attention_finished(query, key, value, output, layer_name, forward_context)
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return output


def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: Optional[torch.Tensor] = None,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    maybe_execute_sparse_attention_begin(query, key, value, layer_name, forward_context)
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output=output,
                      output_scale=output_scale)

    maybe_execute_sparse_attention_finished(query, key, value, output, layer_name, forward_context)
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)


def maybe_execute_sparse_attention_begin(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
):
    if not has_ucm_sparse():
        return
    
    ucm_sparse = get_ucm_sparse()

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    
    ucm_sparse.attention_begin(query, key, value, layer_name, forward_context)

def maybe_execute_sparse_attention_finished(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
):
    if not has_ucm_sparse():
        return
    
    ucm_sparse = get_ucm_sparse()

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    
    ucm_sparse.attention_finished(query, key, value, attn_output, layer_name, forward_context)
