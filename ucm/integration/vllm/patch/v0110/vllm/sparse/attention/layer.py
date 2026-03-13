import os
from typing import List, Optional

import torch
from vllm.attention.layer import (
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm.forward_context import ForwardContext, get_forward_context

from ucm.sparse.state import (
    maybe_execute_sparse_attention_begin,
    maybe_execute_sparse_attention_finished,
)


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
    query, key, value, _ = maybe_execute_sparse_attention_begin(
        query, key, value, layer_name, forward_context
    )
    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)

    maybe_execute_sparse_attention_finished(
        query, key, value, output, layer_name, forward_context
    )
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return output


def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: Optional[torch.Tensor] = None,
    output_block_scale: Optional[torch.Tensor] = None,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    if not self.use_mla:
        if attn_metadata is not None:
            if os.getenv("VLLM_HASH_ATTENTION") == "1":
                kv_cache, k_hash = kv_cache
            else:
                k_hash = None
            query, key, value, output = maybe_execute_sparse_attention_begin(
                query, key, value, layer_name, forward_context, output, k_hash=k_hash
            )

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )

    if not self.use_mla:
        maybe_execute_sparse_attention_finished(
            query, key, value, output, layer_name, forward_context
        )

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
