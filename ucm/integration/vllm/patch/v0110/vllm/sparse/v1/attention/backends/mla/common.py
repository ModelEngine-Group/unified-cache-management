import os
from typing import Generic, Optional

import torch
import vllm.v1.attention.backends.mla.common as common
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import AttentionLayer
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.distributed.parallel_state import get_dcp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.common import M, MLACommonBaseImpl

from ucm.sparse.state import (
    maybe_execute_sparse_attention_begin,
    maybe_execute_sparse_attention_finished,
)


class MLACommonImpl(MLACommonBaseImpl[M], Generic[M]):

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: M,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert output is not None, "Output tensor must be provided."

        forward_context: ForwardContext = get_forward_context()

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported" " for MLACommonImpl"
            )

        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        if self.dcp_world_size is None:
            self.dcp_world_size = get_dcp_group().world_size

        fp8_attention = self.kv_cache_dtype.startswith("fp8")

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]

        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        if os.getenv("VLLM_HASH_ATTENTION") == "1":
            kv_cache, k_hash = kv_cache
        else:
            k_hash = None

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if fp8_attention:
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        if has_prefill:
            prefill_q, prefill_k_c_normed, prefill_k_pe, output = (
                maybe_execute_sparse_attention_begin(
                    prefill_q,
                    prefill_k_c_normed,
                    prefill_k_pe,
                    layer.layer_name,
                    forward_context,
                    output=output,
                    phase="prefill",
                    k_hash=k_hash,
                )
            )

            output[num_decode_tokens:] = self._forward_prefill(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                kv_cache,
                attn_metadata,
                layer._k_scale,
            )

            maybe_execute_sparse_attention_finished(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                output[num_decode_tokens:],
                layer.layer_name,
                forward_context,
                "prefill",
            )

        if has_decode:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)

            # Pads the head_dim if necessary (for the underlying kernel)
            if self.q_pad_num_heads is not None:
                B, N, L = decode_q_pe.shape
                decode_pe_padded = decode_q_pe.new_empty((B, self.q_pad_num_heads, L))
                decode_pe_padded.resize_((B, N, L))
                decode_pe_padded.copy_(decode_q_pe)
                decode_q_pe = decode_pe_padded

            if common.is_rocm_aiter_fp8bmm_enabled():
                # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
                decode_ql_nope = common.aiter_triton_fp8_bmm(
                    decode_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    group_size=128,
                    transpose_bm=True,
                )
            else:
                # Pads the head_dim if necessary (for the underlying kernel)
                N, B, P = decode_q_nope.shape
                _, _, L = self.W_UK_T.shape
                if self.q_pad_num_heads is not None:
                    decode_ql_nope = decode_q_nope.new_empty(
                        (self.q_pad_num_heads, B, L)
                    )
                    decode_ql_nope.resize_((N, B, L))

                else:
                    decode_ql_nope = decode_q_nope.new_empty((N, B, L))

                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
                torch.bmm(decode_q_nope, self.W_UK_T, out=decode_ql_nope)
                # Convert from (N, B, L) to (B, N, L)
                decode_ql_nope = decode_ql_nope.transpose(0, 1)

            if fp8_attention:
                ql_nope_shape = decode_ql_nope.shape
                decode_ql_nope, _ = ops.scaled_fp8_quant(
                    decode_ql_nope.reshape(
                        [ql_nope_shape[0], ql_nope_shape[1] * ql_nope_shape[2]]
                    ),
                    layer._q_scale,
                )
                decode_ql_nope = decode_ql_nope.reshape(ql_nope_shape)
                q_pe_shape = decode_q_pe.shape
                decode_q_pe, _ = ops.scaled_fp8_quant(
                    decode_q_pe.reshape([q_pe_shape[0], q_pe_shape[1] * q_pe_shape[2]]),
                    layer._q_scale,
                )
                decode_q_pe = decode_q_pe.reshape(q_pe_shape)

            decode_q = (decode_ql_nope, decode_q_pe)
            if self.dcp_world_size > 1:
                assert not fp8_attention, "DCP not support fp8 kvcache now."
                # concatenate decode_ql_nope and decode_q_pe -> (B, N, L + P)
                decode_q = torch.cat(decode_q, dim=-1)
                # decode_q do allgather in head dim.
                decode_q = get_dcp_group().all_gather(decode_q, dim=1)

            # call decode attn
            _, k_c_normed, k_pe, output = maybe_execute_sparse_attention_begin(
                torch.cat([decode_ql_nope, decode_q_pe], dim=-1),
                k_c_normed,
                k_pe,
                layer.layer_name,
                forward_context,
                output=output,
                phase="decode",
                k_hash=k_hash,
                decode_ql_nope=decode_ql_nope,
                decode_q_pe=decode_q_pe,
            )

            attn_out, lse = self._forward_decode(
                decode_q, kv_cache, attn_metadata, layer
            )

            maybe_execute_sparse_attention_finished(
                torch.cat([decode_ql_nope, decode_q_pe], dim=-1),
                decode_ql_nope,
                decode_q_pe,
                output[:num_decode_tokens],
                layer.layer_name,
                forward_context,
                "decode",
            )

            # recorect dcp attn_out with lse.
            if self.dcp_world_size > 1:
                attn_out = cp_lse_ag_out_rs(attn_out, lse, get_dcp_group())

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])
        return output_padded
