import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch_npu
from vllm.attention.backends.abstract import (
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_310p, nd_to_nz_2d, nd_to_nz_spec

from ucm.sparse.state import (
    get_ucm_sparse,
    has_ucm_sparse,
    maybe_execute_sparse_attention_begin,
    maybe_execute_sparse_attention_finished,
)


@dataclass
class AscendMetadata:
    # **************************** Basic Properties ************************** #
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_device: torch.Tensor = None  # (ldeng) added for gsa on device
    seq_lens_list: List[int] = None  # type: ignore
    actual_seq_lengths_q: List[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    query_lens: torch.Tensor = None
    query_lens_device: torch.Tensor = None  # (ldeng) added for gsa on device
    # Maximum query length in the batch (None for decoding).
    max_query_len: Optional[int] = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None

    # *************************** Other Properties *************************** #
    enable_dbo_across_dp: bool = False


class AscendAttentionMetadataBuilder:
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: Optional[nn.Module] = None,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        block_table = common_attn_metadata.block_table_tensor
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        attn_mask = common_attn_metadata.attn_mask
        attn_state = common_attn_metadata.attn_state
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        if (
            attn_state == AscendAttentionState.DecodeOnly
            and common_attn_metadata.num_input_tokens > num_actual_tokens
        ):
            padded_num_tokens = (
                common_attn_metadata.num_input_tokens - num_actual_tokens
            )
            seq_lens = torch.cat(
                [
                    seq_lens,
                    torch.ones(
                        padded_num_tokens, dtype=seq_lens.dtype, device=seq_lens.device
                    ),
                ]
            )
            block_table_padding = torch.zeros(
                (padded_num_tokens,) + block_table.shape[1:],
                dtype=block_table.dtype,
                device=block_table.device,
            )
            block_table = torch.cat([block_table, block_table_padding], dim=0)
            query_start_loc_cpu = torch.cat(
                [
                    query_start_loc_cpu,
                    torch.arange(
                        query_start_loc_cpu[-1] + 1,
                        query_start_loc_cpu[-1] + padded_num_tokens,
                        dtype=query_start_loc_cpu.dtype,
                        device=query_start_loc_cpu.device,
                    ),
                ]
            )

        query_start_loc = query_start_loc_cpu.pin_memory().to(
            self.device, non_blocking=True
        )

        if is_310p():
            if attn_state == AscendAttentionState.PrefillNoCache:
                mask_nz = nd_to_nz_2d(attn_mask)
                attn_mask = torch_npu.npu_format_cast(
                    mask_nz.contiguous(), ACL_FORMAT_FRACTAL_NZ
                )
            elif attn_state == AscendAttentionState.ChunkedPrefill:
                mask_nz = nd_to_nz_spec(attn_mask)
                attn_mask = torch_npu.npu_format_cast(
                    mask_nz.contiguous(), ACL_FORMAT_FRACTAL_NZ
                )

        seq_lens_device = None
        query_lens_device = None
        if has_ucm_sparse():
            ucm_sparse = get_ucm_sparse()
            if os.getenv("VLLM_HASH_ATTENTION", "0") == "1":
                seq_lens_device = seq_lens.pin_memory().to(
                    self.device, non_blocking=True
                )
                query_lens_device = query_lens.pin_memory().to(
                    self.device, non_blocking=True
                )
                ucm_sparse.build_decode_attention_meta_npu(
                    query_lens, seq_lens, block_table
                )

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            query_lens_device=query_lens_device,
            seq_lens=seq_lens,
            seq_lens_device=seq_lens_device,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            enable_dbo_across_dp=common_attn_metadata.enable_dbo_across_dp,
        )
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [key_cache, value_cache]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]

        if isinstance(kv_cache, tuple) and isinstance(kv_cache[0], tuple):
            kv_cache = kv_cache[0]
        use_kv_cache_int8 = len(kv_cache) > 0 and kv_cache[0].dtype == torch.int8

        if output is None:
            output = torch.empty(
                num_tokens,
                self.num_heads,
                self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        ori_output = output
        if trace_flag:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name,
            )

        elif hasattr(layer, "quant_method") and use_kv_cache_int8:
            output = layer.quant_method.apply(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                self.attn_type,
                self.scale,
                output,
            )

        else:
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size).fill_(0)
            num_actual_tokens = attn_metadata.num_actual_tokens
            assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
            attn_type = self.attn_type
            if (
                attn_type != AttentionType.DECODER
                and attn_type != AttentionType.ENCODER_ONLY
            ):
                raise NotImplementedError(
                    "Encoder/decoder cross-attention "
                    "are not implemented for "
                    "PallasAttentionBackendImpl"
                )
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            # TODO: Remove this contiguous in the future.
            value = value.contiguous()

            if len(kv_cache) > 1:
                if self.key_cache is None:
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                slots = attn_metadata.slot_mapping
                torch_npu._npu_reshape_and_cache(
                    key=key[:num_actual_tokens],
                    value=value[:num_actual_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slots,
                )
            if attn_type == AttentionType.ENCODER_ONLY:
                cum_seq_len = attn_metadata.query_start_loc[1:].tolist()
                attn_out = torch_npu.npu_fusion_attention(
                    query,
                    key,
                    value,
                    head_num=self.num_heads,
                    input_layout="TND",
                    scale=self.scale,
                    sparse_mode=4,
                    atten_mask=attn_metadata.attn_mask,
                    pre_tockens=attn_metadata.max_query_len,
                    next_tockens=attn_metadata.max_query_len,
                    actual_seq_qlen=cum_seq_len,
                    actual_seq_kvlen=cum_seq_len,
                )
                output = attn_out[0]
            # V0-Style scheduler situation.
            elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                output = self._forward_prefill_no_cache(
                    query, key, value, attn_metadata, output, num_tokens
                )
            elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
                output = self._forward_prefill_cache_hit(query, attn_metadata, output)
            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                output = self._forward_decode_only(query, attn_metadata, output)
            # Normal V1 situation.
            else:
                # npu_fused_infer_attention_score does not support cases
                # where query.shape[0] != attn_metadata.query_start_loc[-1].
                # Thus we need unpad it here.
                num_tokens = attn_metadata.query_start_loc[-1]
                query = query[:num_tokens]
                output = self._forward_v1_style(query, attn_metadata, output)

        # to make in-place change to the output tensor
        if hasattr(layer, "quant_method") and use_kv_cache_int8:
            output = output.view(num_tokens, self.num_heads, self.head_size)
        ori_output[:num_tokens, :, :] = output[:num_tokens, :, :]
        return output.view(num_tokens, self.hidden_size)


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]

    # In NPU, during dummy_run, kv_cache could be a empty tensor, so we need to check the length of kv_cache
    if os.getenv("VLLM_HASH_ATTENTION", "0") == "1" and len(kv_cache) > 0:
        kv_cache, k_hash = kv_cache
    else:
        k_hash = None
    if attn_metadata is not None:
        maybe_execute_sparse_attention_begin(
            query, key, value, layer_name, forward_context, output, k_hash=k_hash
        )

    self.impl.forward(
        self, query, key, value, kv_cache, attn_metadata, output, trace_flag=False
    )
    if attn_metadata is not None:
        maybe_execute_sparse_attention_finished(
            query, key, value, output, layer_name, forward_context
        )

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return