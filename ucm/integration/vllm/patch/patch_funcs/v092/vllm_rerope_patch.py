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
        _patch_utils()
        _patch_gpu_model_runner()
        _patch_qwen2_model()
        _patch_qwen3_model()
        _patch_qwen3moe_model()
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

            ###################### rerope patch ###############
            coef = 3
            ###################### rerope patch ###############

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


# ==================== vllm/v1/attention/backends/utils.py ====================
def _patch_utils() -> None:
    """Patch common metadata"""
    try:
        from dataclasses import dataclass

        import torch
        from vllm.v1.attention.backends import utils

        @dataclass
        class CommonAttentionMetadata_add:
            """
            Per-batch attention metadata, shared across layers and backends.
            AttentionMetadataBuilder instances use it to construct per-layer metadata.
            """

            query_start_loc: torch.Tensor
            """(batch_size + 1,), the start location of each request in query Tensor"""
            seq_lens: torch.Tensor
            """(batch_size,), the length of each request including both computed tokens
            and newly scheduled tokens"""

            num_reqs: int
            """Number of requests"""
            num_actual_tokens: int
            """Total number of tokens in batch"""
            max_query_len: int
            """Longest query in batch"""

            ###################### rerope patch ###############
            use_rerope: bool
            ###################### rerope patch ###############

        utils.CommonAttentionMetadata = CommonAttentionMetadata_add
    except ImportError:
        logger.warning("Could not patch CommonAttentionMetadata - module not found")


# ==================== vllm/v1/worker/gpu_model_runner.py ====================
def _patch_gpu_model_runner() -> None:
    """Patch gpu model runner"""
    try:
        import torch
        from vllm.config import VllmConfig
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        from ucm.sparse.rerope.rerope_utils import default_config

        REROPE_WINDOW = default_config.rerope_window

        original_init = GPUModelRunner.__init__

        def add_init(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            original_init(self, vllm_config, device)

            ###################### rerope patch ###############
            self.vllm_use_rerope = os.getenv("VLLM_USE_REROPE", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

            # use_rerope: current batch rerope state
            # use_rerope_map: save every request rerope state
            self.use_rerope = False
            self.use_rerope_map: dict[str, bool] = {}  # type: ignore
            ###################### rerope patch ###############

        GPUModelRunner.__init__ = add_init

        import os
        from typing import TYPE_CHECKING, Any, Optional

        import numpy as np
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        from vllm.v1.worker.block_table import BlockTable

        if TYPE_CHECKING:
            from vllm.v1.core.sched.output import SchedulerOutput

        def _prepare_inputs_modify(
            self,
            scheduler_output: "SchedulerOutput",
        ) -> tuple[
            dict[str, Any], bool, torch.Tensor, Optional[SpecDecodeMetadata], np.ndarray
        ]:
            """
            :return: tuple[
                attn_metadata: layer-to-attention_metadata mapping,
                attention_cuda_graphs: whether attention can run in cudagraph
                logits_indices, spec_decode_metadata
            ]
            """
            total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            assert total_num_scheduled_tokens > 0
            num_reqs = self.input_batch.num_reqs
            assert num_reqs > 0

            # OPTIMIZATION: Start copying the block table first.
            # This way, we can overlap the copy with the following CPU operations.
            self.input_batch.block_table.commit(num_reqs)

            # Get the number of scheduled tokens for each request.
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = max(tokens)

            ###################### rerope patch ###############
            # Get use_rerope
            if self.vllm_use_rerope:
                use_rerope_this_batch = False
                for req in scheduler_output.scheduled_new_reqs:
                    self.use_rerope_map[req.req_id] = (
                        len(req.prompt_token_ids) > REROPE_WINDOW
                    )
                for req_id in req_ids:
                    use_rerope_this_batch |= self.use_rerope_map[req_id]
                self.use_rerope = use_rerope_this_batch
            ###################### rerope patch ###############

            # Get request indices.
            # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

            # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
            # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

            # Get positions.
            positions_np = self.positions_np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )

            # Calculate M-RoPE positions.
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._calc_mrope_positions(scheduler_output)

            # Get token indices.
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
            # where M is the max_model_len.
            token_indices = (
                positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
            )

            # NOTE(woosuk): We use torch.index_select instead of np.take here
            # because torch.index_select is much faster than np.take for large
            # tensors.
            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:total_num_scheduled_tokens],
            )

            # Calculate the slot mapping for each KV cache group.
            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):
                block_size = kv_cache_group_spec.kv_cache_spec.block_size
                block_table: BlockTable = self.input_batch.block_table[
                    kv_cache_group_id
                ]
                # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
                # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
                # where K is the max_num_blocks_per_req and the block size is 2.
                # NOTE(woosuk): We can't simply use `token_indices // block_size`
                # here because M (max_model_len) is not necessarily divisible by
                # block_size.
                block_table_indices = (
                    req_indices * block_table.max_num_blocks_per_req
                    + positions_np // block_size
                )
                block_table_cpu = block_table.get_cpu_tensor()
                block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
                block_offsets = positions_np % block_size
                np.add(
                    block_numbers * block_size,
                    block_offsets,
                    out=block_table.slot_mapping_np[:total_num_scheduled_tokens],
                )

            # Prepare the attention metadata.
            self.query_start_loc_np[0] = 0
            self.query_start_loc_np[1 : num_reqs + 1] = cu_num_tokens

            self.seq_lens_np[:num_reqs] = (
                self.input_batch.num_computed_tokens_cpu[:num_reqs]
                + num_scheduled_tokens
            )

            # Copy the tensors to the GPU.
            self.input_ids[:total_num_scheduled_tokens].copy_(
                self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True
            )
            if self.uses_mrope:
                # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
                self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                    self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                    non_blocking=True,
                )
            else:
                # Common case (1D positions)
                self.positions[:total_num_scheduled_tokens].copy_(
                    self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True
                )

            self.query_start_loc[: num_reqs + 1].copy_(
                self.query_start_loc_cpu[: num_reqs + 1], non_blocking=True
            )
            self.seq_lens[:num_reqs].copy_(
                self.seq_lens_cpu[:num_reqs], non_blocking=True
            )

            # Fill unused with -1. Needed for reshape_and_cache
            self.seq_lens[num_reqs:].fill_(0)
            # Note: pad query_start_loc to be non-decreasing, as kernels
            # like FlashAttention requires that
            self.query_start_loc[num_reqs + 1 :].fill_(
                self.query_start_loc_cpu[num_reqs].item()
            )

            query_start_loc = self.query_start_loc[: num_reqs + 1]
            seq_lens = self.seq_lens[:num_reqs]

            ###################### rerope patch ###############
            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                use_rerope=self.use_rerope,
            )
            ###################### rerope patch ###############

            attn_metadata: dict[str, Any] = {}
            # Prepare the attention metadata for each KV cache group and make layers
            # in the same group share the same metadata.
            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups
            ):

                # Prepare for cascade attention if enabled & beneficial.
                common_prefix_len = 0
                builder = self.attn_metadata_builders[kv_cache_group_id]
                if self.cascade_attn_enabled:
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        scheduler_output.num_common_prefix_blocks[kv_cache_group_id],
                        kv_cache_group_spec.kv_cache_spec,
                        builder,
                    )

                attn_metadata_i = builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                )

                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

            attention_cuda_graphs = all(
                b.can_run_in_cudagraph(common_attn_metadata)
                for b in self.attn_metadata_builders
            )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            if not use_spec_decode:
                # NOTE(woosuk): Due to chunked prefills, the batch may contain
                # partial requests. While we should not sample any token
                # from these partial requests, we do so for simplicity.
                # We will ignore the sampled tokens from the partial requests.
                # TODO: Support prompt logprobs.
                logits_indices = query_start_loc[1:] - 1
                spec_decode_metadata = None
            else:
                # Get the number of draft tokens for each request.
                # Iterate over the dictionary rather than all requests since not all
                # requests have draft tokens.
                num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
                for (
                    req_id,
                    draft_token_ids,
                ) in scheduler_output.scheduled_spec_decode_tokens.items():
                    req_idx = self.input_batch.req_id_to_index[req_id]
                    num_draft_tokens[req_idx] = len(draft_token_ids)

                spec_decode_metadata = self._calc_spec_decode_metadata(
                    num_draft_tokens, cu_num_tokens
                )
                logits_indices = spec_decode_metadata.logits_indices

            # Hot-Swap lora model
            if self.lora_config:
                self.set_active_loras(self.input_batch, num_scheduled_tokens)

            return (
                attn_metadata,
                attention_cuda_graphs,
                logits_indices,
                spec_decode_metadata,
                num_scheduled_tokens,
            )

        GPUModelRunner._prepare_inputs = _prepare_inputs_modify

    except ImportError:
        logger.warning("Could not patch gpu model runner - module not found")


# ==================== vllm/model_executor/models/qwen2.py  ====================
def _patch_qwen2_model() -> None:
    """Patch qwen to support rerope"""
    try:
        import math

        import torch
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.models.qwen2 import Qwen2Attention

        from ucm.sparse.rerope.rerope_utils import default_config

        def Qwen2Attention_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            attn_metadata = get_forward_context().attn_metadata

            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            ###################### rerope patch ###############
            REROPE_WINDOW = default_config.rerope_window
            TRAINING_LENGTH = default_config.training_length
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
                k0 = k.clone()
                q, k = self.rotary_emb(positions, q, k)
                q2 = q.clone()

            attn_output = self.attn(q, k, v, query2=q2, key2=k0)
            ###################### rerope patch ###############

            output, _ = self.o_proj(attn_output)
            return output

        Qwen2Attention.forward = Qwen2Attention_forward

    except ImportError:
        logger.warning("Could not patch qwen2 modelr - module not found")


# ==================== vllm/model_executor/models/qwen3.py  ====================
def _patch_qwen3_model() -> None:
    """Patch qwen to support rerope"""
    try:
        import math

        import torch
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.models.qwen3 import Qwen3Attention

        from ucm.sparse.rerope.rerope_utils import default_config

        def Qwen3Attention_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            attn_metadata = get_forward_context().attn_metadata

            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Add qk-norm
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)

            ###################### rerope patch ###############
            REROPE_WINDOW = default_config.rerope_window
            TRAINING_LENGTH = default_config.training_length
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
                k0 = k.clone()
                q, k = self.rotary_emb(positions, q, k)
                q2 = q.clone()

            attn_output = self.attn(q, k, v, query2=q2, key2=k0)
            ###################### rerope patch ###############

            output, _ = self.o_proj(attn_output)
            return output

        Qwen3Attention.forward = Qwen3Attention_forward

    except ImportError:
        logger.warning("Could not patch qwen3 modelr - module not found")


# ==================== vllm/model_executor/models/qwen3_moe.py  ====================
def _patch_qwen3moe_model() -> None:
    """Patch qwen to support rerope"""
    try:
        import math

        import torch
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention

        from ucm.sparse.rerope.rerope_utils import default_config

        def Qwen3MoeAttention_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            attn_metadata = get_forward_context().attn_metadata

            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Add qk-norm
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)

            ###################### rerope patch ###############
            REROPE_WINDOW = default_config.rerope_window
            TRAINING_LENGTH = default_config.training_length
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
                k0 = k.clone()
                q, k = self.rotary_emb(positions, q, k)
                q2 = q.clone()

            attn_output = self.attn(q, k, v, query2=q2, key2=k0)
            ###################### rerope patch ###############

            output, _ = self.o_proj(attn_output)
            return output

        Qwen3MoeAttention.forward = Qwen3MoeAttention_forward

    except ImportError:
        logger.warning("Could not patch qwen3 modelr - module not found")


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
            value: torch.Tensor,
            query2: Optional[torch.Tensor] = None,
            key2: Optional[torch.Tensor] = None,
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
                    if key is not None:
                        key = key.view(-1, self.num_kv_heads, self.head_size)
                    ###################### rerope patch ###############
                    if query2 is not None:
                        query2 = query2.view(-1, self.num_heads, self.head_size)
                    if key2 is not None:
                        key2 = key2.view(-1, self.num_kv_heads, self.head_size)
                    ###################### rerope patch ###############
                    if value is not None:
                        value = value.view(-1, self.num_kv_heads, self.head_size)

                if self.use_direct_call:
                    forward_context: ForwardContext = get_forward_context()
                    attn_metadata = forward_context.attn_metadata
                    if isinstance(attn_metadata, dict):
                        attn_metadata = attn_metadata[self.layer_name]
                    self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                    ###################### rerope patch ###############
                    self.impl.forward(
                        self,
                        query,
                        key,
                        value,
                        self_kv_cache,
                        attn_metadata,
                        query2=query2,
                        key2=key2,
                        output=output,
                    )
                else:
                    torch.ops.vllm.unified_attention_with_output(
                        query,
                        key,
                        value,
                        output,
                        self.layer_name,
                        query2=query2,
                        key2=key2,
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
                        value,
                        self_kv_cache,
                        attn_metadata,
                        query2=query2,
                        key2=key2,
                    )
                else:
                    return torch.ops.vllm.unified_attention(
                        query, key, value, self.layer_name, query2=query2, key2=key2
                    )
                    ###################### rerope patch ###############

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
            value: torch.Tensor,
            layer_name: str,
            query2: Optional[torch.Tensor] = None,
            key2: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            wait_for_kv_layer_from_connector(layer_name)

            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[layer_name]
            self = forward_context.no_compile_layers[layer_name]
            kv_cache = self.kv_cache[forward_context.virtual_engine]

            output = self.impl.forward(
                self,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                query2=query2,
                key2=key2,
            )

            maybe_save_kv_layer_to_connector(layer_name, kv_cache)
            return output

        def unified_attention_with_output_impl(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            output: torch.Tensor,
            layer_name: str,
            query2: Optional[torch.Tensor] = None,
            key2: Optional[torch.Tensor] = None,
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
                value,
                kv_cache,
                attn_metadata,
                query2=query2,
                key2=key2,
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

        ###################### rerope patch ###############
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

            ###################### rerope patch ###############
            use_rerope: bool
            ###################### rerope patch ###############

            # For cascade attention.
            use_cascade: bool
            common_prefix_len: int
            cu_prefix_query_lens: Optional[torch.Tensor]
            prefix_kv_lens: Optional[torch.Tensor]
            suffix_kv_lens: Optional[torch.Tensor]

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

            ###################### rerope patch ###############
            use_rerope = common_attn_metadata.use_rerope
            ###################### rerope patch ###############

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

            ###################### rerope patch ###############
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
            ###################### rerope patch ###############
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

            ###################### rerope patch ###############
            return (3, num_blocks, block_size, num_kv_heads, head_size)
            ###################### rerope patch ###############

        TritonAttentionBackend.get_kv_cache_shape = staticmethod(
            TritonAttentionBackend_get_kv_cache_shape
        )

        def TritonAttentionImpl_forwad(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: TritonAttentionMetadata,
            query2: Optional[torch.Tensor] = None,
            key2: Optional[torch.Tensor] = None,
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
            ###################### rerope patch ###############
            key_cache, value_cache, key_cache2 = kv_cache.unbind(0)
            ###################### rerope patch ###############

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
                ###################### rerope patch ###############
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
                ###################### rerope patch ###############

            if self.kv_cache_dtype.startswith("fp8"):
                key_cache = key_cache.view(self.fp8_dtype)
                ###################### rerope patch ###############
                if key_cache2 is not None:
                    key_cache2 = key_cache2.view(self.fp8_dtype)
                ###################### rerope patch ###############
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
                    ###################### rerope patch ###############
                    if query2 is not None:
                        query2, _ = ops.scaled_fp8_quant(
                            query2.reshape(
                                (num_tokens, num_heads * head_size)
                            ).contiguous(),
                            layer._q_scale,
                        )
                        query2 = query2.reshape((num_tokens, num_heads, head_size))
                    ###################### rerope patch ###############

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

            ###################### rerope patch ###############
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
            ###################### rerope patch ###############
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
