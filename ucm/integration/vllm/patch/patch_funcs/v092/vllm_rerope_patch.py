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

from __future__ import annotations

from torch.library import Library

from ucm.logger import init_logger

logger = init_logger(__name__)

_UCM_UNIFIED_ATTENTION_WITH_OUTPUT_REGISTERED = False


def _apply_rerope_adapt_patches() -> None:
    try:
        _patch_attention_spec()
        _patch_request_succeed_dumped_blocks()
        _patch_qwen_model()
        _patch_attention_layer()
        _patch_triton_attn()

    except Exception as e:
        logger.error(f"Failed to apply aggre patch: {e}", exc_info=True)
        raise


# ==================== vllm/v1/kv_cache_interface.py  ====================
def _patch_attention_spec() -> None:
    """Patch modify the kv cache spec"""
    try:
        from vllm.utils import cdiv, get_dtype_size
        from vllm.v1.kv_cache_interface import AttentionSpec

        def _page_size_bytes_rerope(self: "AttentionSpec") -> int:
            """
            Patched version of page_size_bytes property.
            REROPE support with coefficient=3.
            """

            coef = 3

            return (
                coef
                * self.block_size
                * self.num_kv_heads
                * self.head_size
                * get_dtype_size(self.dtype)
            )

        AttentionSpec.page_size_bytes = property(_page_size_bytes_rerope)

    except ImportError:
        logger.warning(
            "Could not patch AttentionSpec with _page_size_bytes_rerope - module not found"
        )


# ==================== vllm/v1/request.py ====================
def _patch_request_succeed_dumped_blocks() -> None:
    """Patch Request to add succeed_dumped_blocks field."""
    try:
        from vllm.v1.request import Request

        original_init = Request.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.succeed_dumped_blocks = []

        Request.__init__ = __init__
    except ImportError:
        logger.warning("Could not patch Request.__init__ - module not found")


# ==================== vllm/model_executor/models/qwen2.py  ====================
def _patch_qwen_model() -> None:
    """Patch qwen to support rerope"""
    try:
        import math

        import torch
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.models.qwen2 import Qwen2Attention

        from ucm.sparse.rerope.rerope_utils import default_config

        REROPE_WINDOW = default_config.rerope_window
        TRAINING_LENGTH = default_config.training_length

        def Qwen2Attention_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            attn_metadata = get_forward_context().attn_metadata

            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            if attn_metadata and next(iter(attn_metadata.values())).use_rerope:
                q *= (
                    ((positions + 1)[:, None].log() / math.log(TRAINING_LENGTH))
                    .clip(1)
                    .to(q.dtype)
                )
                q2 = q.clone()
                k2 = k.clone()
                k0 = k.clone()

                q, k = self.rotary_emb(positions, q, k)
                q2, _ = self.rotary_emb(positions * 0 + REROPE_WINDOW, q2, k2)
                del k2
            else:
                q, k = self.rotary_emb(positions, q, k)
                q2, k0 = None, None

            attn_output = self.attn(q, k, q2, k0, v)
            output, _ = self.o_proj(attn_output)
            return output

        Qwen2Attention.forward = Qwen2Attention_forward

    except ImportError:
        logger.warning("Could not patch qwen2 modelr - module not found")


# ==================== vllm/attention/layer.py  ====================
def _patch_attention_layer() -> None:
    """Patch attention layer"""
    try:
        from typing import Optional

        import torch
        from vllm.attention.layer import (
            maybe_save_kv_layer_to_connector,
            wait_for_kv_layer_from_connector,
        )
        from vllm.forward_context import ForwardContext, get_forward_context

        def attn_forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            query2: Optional[torch.Tensor],
            key2: Optional[torch.Tensor],
            value: torch.Tensor,
            # For some alternate attention backends like MLA the attention output
            # shape does not match the query shape, so we optionally let the model
            # definition specify the output tensor shape.
            output_shape: Optional[torch.Size] = None,
        ) -> torch.Tensor:
            """
            The KV cache is stored inside this class and is accessed via
            `self.kv_cache`.
            Attention metadata (`attn_metadata`) is set using a context manager in
            the model runner's `execute_model` method. It is accessed via forward
            context using
            `vllm.forward_context.get_forward_context().attn_metadata`.
            """
            if self.calculate_kv_scales:
                attn_metadata = get_forward_context().attn_metadata
                if attn_metadata.enable_kv_scales_calculation:
                    self.calc_kv_scales(query, key, value)
            if self.use_output:
                output_shape = output_shape if output_shape is not None else query.shape
                output = torch.zeros(
                    output_shape, dtype=query.dtype, device=query.device
                )
                hidden_size = output_shape[-1]
                # We skip reshaping query, key and value tensors for the MLA
                # backend since these tensors have different semantics and are
                # processed differently.
                if not self.use_mla:
                    # Reshape the query, key, and value tensors.
                    # NOTE(woosuk): We do this outside the custom op to minimize the
                    # CPU overheads from the non-CUDA-graph regions.
                    query = query.view(-1, self.num_heads, self.head_size)
                    output = output.view(-1, self.num_heads, self.head_size)
                    if query2 is not None:
                        query2 = query2.view(-1, self.num_heads, self.head_size)
                    if key is not None:
                        key = key.view(-1, self.num_kv_heads, self.head_size)
                    if key2 is not None:
                        key2 = key2.view(-1, self.num_kv_heads, self.head_size)
                    if value is not None:
                        value = value.view(-1, self.num_kv_heads, self.head_size)

                if self.use_direct_call:
                    forward_context: ForwardContext = get_forward_context()
                    attn_metadata = forward_context.attn_metadata
                    if isinstance(attn_metadata, dict):
                        attn_metadata = attn_metadata[self.layer_name]
                    self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                    self.impl.forward(
                        self,
                        query,
                        key,
                        query2,
                        key2,
                        value,
                        self_kv_cache,
                        attn_metadata,
                        output=output,
                    )
                else:
                    torch.ops.vllm.unified_attention_with_output(
                        query, key, query2, key2, value, output, self.layer_name
                    )
                return output.view(-1, hidden_size)
            else:
                if self.use_direct_call:
                    forward_context = get_forward_context()
                    attn_metadata = forward_context.attn_metadata
                    if isinstance(attn_metadata, dict):
                        attn_metadata = attn_metadata[self.layer_name]
                    self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                    return self.impl.forward(
                        self,
                        query,
                        key,
                        query2,
                        key2,
                        value,
                        self_kv_cache,
                        attn_metadata,
                    )
                else:
                    return torch.ops.vllm.unified_attention(
                        query, key, query2, key2, value, self.layer_name
                    )

        vllm_ops = torch.ops.vllm
        orig_unified_attention_with_output = vllm_ops.unified_attention_with_output
        orig_unified_attention = vllm_ops.unified_attention

        def _wrap_op_overload(orig, impl):
            class _Wrapper:
                def __init__(self, orig):
                    self._orig = orig

                def __call__(self, *args, **kwargs):
                    return impl(*args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._orig, name)

            return _Wrapper(orig)

        def unified_attention_impl(
            query: torch.Tensor,
            key: torch.Tensor,
            query2: Optional[torch.Tensor],
            key2: Optional[torch.Tensor],
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

            output = self.impl.forward(
                self, query, key, query2, key2, value, kv_cache, attn_metadata
            )

            maybe_save_kv_layer_to_connector(layer_name, kv_cache)
            return output

        def unified_attention_with_output_impl(
            query: torch.Tensor,
            key: torch.Tensor,
            query2: Optional[torch.Tensor],
            key2: Optional[torch.Tensor],
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
            self.impl.forward(
                self,
                query,
                key,
                query2,
                key2,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
            )
            maybe_save_kv_layer_to_connector(layer_name, kv_cache)

        vllm_ops.unified_attention_with_output = _wrap_op_overload(
            orig_unified_attention_with_output, unified_attention_with_output_impl
        )
        vllm_ops.unified_attention = _wrap_op_overload(
            orig_unified_attention, unified_attention_impl
        )
        from vllm.attention import layer

        layer.Attention.forward = attn_forward
        layer.unified_attention = unified_attention_impl
        layer.unified_attention_with_output = unified_attention_with_output_impl

    except ImportError:
        logger.warning("Could not patch layer - module not found")


# ==================== vllm/v1/attention/backends/triton_attn.py  ====================
def _patch_triton_attn() -> None:
    """Patch triton_attn to support rerope"""
    try:
        from dataclasses import dataclass
        from typing import Optional

        import torch
        from vllm import _custom_ops as ops
        from vllm.attention.ops.triton_unified_attention import unified_attention
        from vllm.platforms import current_platform
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
        from vllm.v1.attention.backends.triton_attn import (
            TritonAttentionBackend,
            TritonAttentionImpl,
            TritonAttentionMetadata,
            TritonAttentionMetadataBuilder,
        )
        from vllm.v1.attention.backends.utils import (
            CommonAttentionMetadata,
            make_local_attention_virtual_batches,
        )

        from ucm.sparse.rerope.rerope_utils import default_config
        from ucm.sparse.rerope.triton_unified_attention_rerope import (
            unified_attention_rerope,
        )

        REROPE_WINDOW = default_config.rerope_window

        @dataclass
        class TritonAttentionMetadata_add:
            # NOTE(sang): Definition of context_len, query_len, and seq_len.
            # |---------- N-1 iteration --------|
            # |---------------- N iteration ---------------------|
            # |- tokenA -|......................|-- newTokens ---|
            # |---------- context_len ----------|
            # |-------------------- seq_len ---------------------|
            #                                   |-- query_len ---|
            num_actual_tokens: int  # Number of tokens excluding padding.
            max_query_len: int
            query_start_loc: torch.Tensor
            max_seq_len: int
            seq_lens: torch.Tensor
            block_table: torch.Tensor
            slot_mapping: torch.Tensor

            # For cascade attention.
            use_cascade: bool
            common_prefix_len: int
            cu_prefix_query_lens: Optional[torch.Tensor]
            prefix_kv_lens: Optional[torch.Tensor]
            suffix_kv_lens: Optional[torch.Tensor]

            use_rerope: bool = False

            # Optional aot scheduling
            scheduler_metadata: Optional[torch.Tensor] = None
            prefix_scheduler_metadata: Optional[torch.Tensor] = None

            # for local attention
            @dataclass
            class LocalAttentionMetadata:
                local_query_start_loc: torch.Tensor
                local_seqused_k: torch.Tensor
                local_block_table: torch.Tensor
                local_max_query_len: int
                local_max_seq_len: int
                local_scheduler_metadata: Optional[torch.Tensor]

            local_attn_metadata: Optional[LocalAttentionMetadata] = None

        TritonAttentionMetadata = TritonAttentionMetadata_add

        def TritonAttentionMetadataBuilder_build(
            self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata
        ) -> TritonAttentionMetadata:
            num_reqs = common_attn_metadata.num_reqs
            num_actual_tokens = common_attn_metadata.num_actual_tokens
            max_query_len = common_attn_metadata.max_query_len

            max_seq_len = int(self.runner.seq_lens_np[:num_reqs].max())
            query_start_loc = common_attn_metadata.query_start_loc
            seq_lens = common_attn_metadata.seq_lens
            block_table = self.block_table
            block_table_tensor = block_table.get_device_tensor()[:num_reqs]

            block_table.slot_mapping[:num_actual_tokens].copy_(
                block_table.slot_mapping_cpu[:num_actual_tokens], non_blocking=True
            )
            # Fill unused with -1. Needed for reshape_and_cache in full cuda graph
            # mode.
            block_table.slot_mapping[num_actual_tokens:].fill_(-1)

            slot_mapping = block_table.slot_mapping[:num_actual_tokens]

            # for local attention
            local_attn_metadata = None
            if self.runner.attention_chunk_size is not None:
                (
                    seqlens_q_local_np,
                    virt_q_cu_seqlens_np,
                    virt_k_seqlens_np,
                    virt_block_table_tensor,
                ) = make_local_attention_virtual_batches(
                    self.runner.attention_chunk_size,
                    self.runner.query_start_loc_np[: num_reqs + 1],
                    self.runner.seq_lens_np[:num_reqs],
                    block_table_tensor,
                    self.block_size,
                )
                local_query_start_loc = torch.from_numpy(virt_q_cu_seqlens_np).to(
                    self.runner.device, non_blocking=True
                )
                local_seqused_k = torch.from_numpy(virt_k_seqlens_np).to(
                    self.runner.device, non_blocking=True
                )
                local_max_query_len = seqlens_q_local_np.max()
                local_max_seq_len = virt_k_seqlens_np.max()

                local_attn_metadata = TritonAttentionMetadata.LocalAttentionMetadata(
                    local_query_start_loc=local_query_start_loc,
                    local_seqused_k=local_seqused_k,
                    local_block_table=virt_block_table_tensor,
                    local_max_query_len=local_max_query_len,
                    local_max_seq_len=local_max_seq_len,
                    local_scheduler_metadata=None,
                )

            # saving for the max input tokens length
            max_prompt_len = 0
            for req_id in self.runner.input_batch.req_id_to_index.keys():
                req_state = self.runner.requests.get(req_id)
                if req_state:
                    prompt_len = len(req_state.prompt_token_ids)
                    max_prompt_len = max(max_prompt_len, prompt_len)

            use_rerope = max_prompt_len > REROPE_WINDOW

            use_cascade = common_prefix_len > 0

            if use_cascade:
                cu_prefix_query_lens = torch.tensor(
                    [0, num_actual_tokens], dtype=torch.int32, device=self.runner.device
                )
                prefix_kv_lens = torch.tensor(
                    [common_prefix_len], dtype=torch.int32, device=self.runner.device
                )
                suffix_kv_lens = self.runner.seq_lens_np[:num_reqs] - common_prefix_len
                suffix_kv_lens = torch.from_numpy(suffix_kv_lens).to(self.runner.device)
            else:
                cu_prefix_query_lens = None
                prefix_kv_lens = None
                suffix_kv_lens = None
                prefix_scheduler_metadata = None

            attn_metadata = TritonAttentionMetadata(
                num_actual_tokens=num_actual_tokens,
                max_query_len=max_query_len,
                query_start_loc=query_start_loc,
                max_seq_len=max_seq_len,
                seq_lens=seq_lens,
                block_table=block_table_tensor,
                slot_mapping=slot_mapping,
                use_cascade=use_cascade,
                common_prefix_len=common_prefix_len,
                cu_prefix_query_lens=cu_prefix_query_lens,
                prefix_kv_lens=prefix_kv_lens,
                suffix_kv_lens=suffix_kv_lens,
                local_attn_metadata=local_attn_metadata,
                prefix_scheduler_metadata=prefix_scheduler_metadata,
                use_rerope=use_rerope,
            )
            return attn_metadata

        TritonAttentionMetadataBuilder.build = TritonAttentionMetadataBuilder_build

        def TritonAttentionBackend_get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
        ) -> tuple[int, ...]:
            if block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")

            return (3, num_blocks, block_size, num_kv_heads, head_size)

        TritonAttentionBackend.get_kv_cache_shape = staticmethod(
            TritonAttentionBackend_get_kv_cache_shape
        )

        def TritonAttentionImpl_forwad(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            query2: Optional[torch.Tensor],
            key2: Optional[torch.Tensor],
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: TritonAttentionMetadata,
            output: Optional[torch.Tensor] = None,
            output_scale: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass with FlashAttention.

            Args:
                query: shape = [num_tokens, num_heads, head_size]
                key: shape = [num_tokens, num_kv_heads, head_size]
                value: shape = [num_tokens, num_kv_heads, head_size]
                kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                attn_metadata: Metadata for attention.
            Returns:
                shape = [num_tokens, num_heads * head_size]
            """
            assert output is not None, "Output tensor must be provided."

            if output_scale is not None:
                raise NotImplementedError(
                    "fused output quantization is not yet supported"
                    " for TritonAttentionImpl"
                )

            if attn_metadata is None:
                # Profiling run.
                return output

            assert attn_metadata.use_cascade is False

            # IMPORTANT!
            # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
            # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
            # in this method. For example, `view` and `slice` (or `[:n]`) operations
            # are surprisingly slow even in the case they do not invoke any GPU ops.
            # Minimize the PyTorch ops in this method as much as possible.
            # Whenever making a change in this method, please benchmark the
            # performance to make sure it does not introduce any overhead.

            num_actual_tokens = attn_metadata.num_actual_tokens

            key_cache, value_cache, key_cache2 = kv_cache.unbind(0)

            if self.kv_sharing_target_layer_name is None:
                # Reshape the input keys and values and store them in the cache.
                # Skip this if sharing KV cache with an earlier attention layer.
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )
                if key2 is not None:
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        key2,
                        value,
                        key_cache2,
                        value_cache,
                        attn_metadata.slot_mapping,
                        self.kv_cache_dtype,
                        layer._k_scale,
                        layer._v_scale,
                    )

            if self.kv_cache_dtype.startswith("fp8"):
                key_cache = key_cache.view(self.fp8_dtype)
                if key_cache2 is not None:
                    key_cache2 = key_cache2.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
                num_tokens, num_heads, head_size = query.shape
                assert (
                    layer._q_scale == 1.0
                ), "A non 1.0 q_scale is not currently supported."
                if not current_platform.is_rocm():
                    # Skip Q quantization on ROCm, since dequantizing back to
                    # f32 in the attention kernel is not supported.
                    query, _ = ops.scaled_fp8_quant(
                        query.reshape((num_tokens, num_heads * head_size)).contiguous(),
                        layer._q_scale,
                    )
                    query = query.reshape((num_tokens, num_heads, head_size))
                    if query2 is not None:
                        query2, _ = ops.scaled_fp8_quant(
                            query2.reshape(
                                (num_tokens, num_heads * head_size)
                            ).contiguous(),
                            layer._q_scale,
                        )
                        query2 = query2.reshape((num_tokens, num_heads, head_size))

            use_local_attn = (
                self.use_irope and attn_metadata.local_attn_metadata is not None
            )

            if use_local_attn:
                assert attn_metadata.local_attn_metadata is not None
                local_metadata = attn_metadata.local_attn_metadata
                cu_seqlens_q = local_metadata.local_query_start_loc
                seqused_k = local_metadata.local_seqused_k
                max_seqlen_q = local_metadata.local_max_query_len
                max_seqlen_k = local_metadata.local_max_seq_len
                block_table = local_metadata.local_block_table
            else:
                cu_seqlens_q = attn_metadata.query_start_loc
                seqused_k = attn_metadata.seq_lens
                max_seqlen_q = attn_metadata.max_query_len
                max_seqlen_k = attn_metadata.max_seq_len
                block_table = attn_metadata.block_table

            descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

            if attn_metadata.use_rerope:
                unified_attention_rerope(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    q2=query2[:num_actual_tokens],
                    k2=key_cache2,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=True,
                    rerope_window=REROPE_WINDOW,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    q_descale=None,
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
            else:
                unified_attention(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    q_descale=None,
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )

            return output

        TritonAttentionImpl.forward = TritonAttentionImpl_forwad

    except ImportError:
        logger.warning("Could not patch triton attention - module not found")
        