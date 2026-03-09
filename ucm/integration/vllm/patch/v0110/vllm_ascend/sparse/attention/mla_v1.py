from dataclasses import dataclass
from typing import (NamedTuple, Optional, Tuple)

import os
import torch
from torch import nn
from vllm.attention.backends.abstract import MLAAttentionImpl
from vllm.forward_context import get_forward_context
from vllm.utils import cdiv, round_down

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         maybe_save_kv_layer_to_connector,
                                         split_decodes_and_prefills,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.multistream.context import get_multistream_comm_context
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.attention.attention_v1 import (AscendMLAPrefillMetadata,
                                                DecodeMLAPreprocessResult,
                                                M)

from ucm.sparse.state import (has_ucm_sparse,
                              get_ucm_sparse,
                              maybe_execute_sparse_attention_begin,
                              maybe_execute_sparse_attention_finished)



@dataclass
class AscendMLADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_lens: int
    seq_lens_list: list[int]
    actual_seq_lengths_q: Optional[list[int]] = None
    attn_mask: Optional[torch.Tensor] = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    seq_lens_device: torch.Tensor = None

@dataclass
class AscendMLAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    query_lens: Optional[list[int]] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: Optional[AscendMLADecodeMetadata] = None
    prefill: Optional[AscendMLAPrefillMetadata] = None
    enable_dbo_across_dp: bool = False

    slot_mapping_cpu: torch.Tensor = None
    num_prefill_tokens_device: torch.Tensor = None
    num_decode_tokens_device: torch.Tensor = None

class AscendMLAMetadataBuilder:
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ) -> AscendMLAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device

        block_table = (common_attn_metadata.block_table_tensor[:num_reqs])
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        input_positions = common_attn_metadata.positions[:
                                                         num_actual_tokens].long(
                                                         )

        if self.cos_cache is None:
            self.cos_cache = model.model.layers[
                model.model.start_layer].self_attn.rotary_emb.cos_cached
            self.sin_cache = model.model.layers[
                model.model.start_layer].self_attn.rotary_emb.sin_cached
        if self.cos_cache.dtype != self.model_config.dtype:  # type: ignore
            self.cos_cache = self.cos_cache.to(  # type: ignore
                self.model_config.dtype)  # type: ignore
            self.sin_cache = self.sin_cache.to(  # type: ignore
                self.model_config.dtype)  # type: ignore

        query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        query_lens = query_seq_lens_cpu[:num_reqs]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        num_computed_tokens_cpu = (seq_lens - query_lens)

        prefill_metadata = None
        chunked_context_metadata = None
        if num_prefills > 0:
            reqs_start = num_decodes  # prefill_start
            tokens_start = num_decode_tokens
            max_query_len = query_lens[reqs_start:].max().item()
            max_seq_lens = seq_lens[reqs_start:].max().item()
            prefill_query_start_loc = query_start_loc[
                reqs_start:] - query_start_loc[reqs_start]

            context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            if self.chunked_prefill_enabled and max_context_len_cpu > 0:
                max_context_chunk = (self.chunked_prefill_workspace_size //
                                     num_prefills_with_context_cpu)
                max_context_chunk = round_down(max_context_chunk,
                                               self.block_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)
                chunk_starts = torch.arange(num_chunks, dtype=torch.int32) \
                    .unsqueeze(1).expand(-1, num_prefills) * max_context_chunk
                chunk_ends = torch.min(context_lens_cpu.unsqueeze(0),
                                       chunk_starts + max_context_chunk)
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
                cu_seq_lens_cpu = torch.zeros(num_chunks,
                                              num_prefills + 1,
                                              dtype=torch.int32,
                                              pin_memory=True)
                torch.cumsum(chunk_seq_lens,
                             dim=1,
                             out=cu_seq_lens_cpu[:, 1:],
                             dtype=torch.int32)
                chunked_context_metadata = \
                    AscendMLAPrefillMetadata.ChunkedContextMetadata(
                    cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                    starts=chunk_starts.to(device, non_blocking=True),
                    seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                    max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                    chunk_seq_lens=chunk_seq_lens,
                    chunk_seq_lens_npu=chunk_seq_lens.npu(),
                    workspace=self.chunked_prefill_workspace,
                )
            prefill_input_positions = input_positions[tokens_start:]
            cos = self.cos_cache[
                prefill_input_positions].unsqueeze(  # type: ignore
                    1).unsqueeze(2)
            sin = self.sin_cache[
                prefill_input_positions].unsqueeze(  # type: ignore
                    1).unsqueeze(2)
            prefill_metadata = AscendMLAPrefillMetadata(
                attn_mask=common_attn_metadata.attn_mask,
                query_lens=query_lens[reqs_start:].to(torch.int32),
                seq_lens=seq_lens,
                context_lens=seq_lens[reqs_start:],
                input_positions=prefill_input_positions,
                block_table=block_table[reqs_start:, ...],
                max_query_len=max_query_len,
                max_seq_lens=max_seq_lens,
                query_start_loc=prefill_query_start_loc,
                chunked_context=chunked_context_metadata,
                sin=sin,
                cos=cos,
            )

        decode_metadata = None
        if num_decodes > 0:
            cos = common_attn_metadata.cos
            sin = common_attn_metadata.sin
            # Notice that num_decodes != num_decode_tokens in SpecDecoding Scenario
            actual_seq_lengths_q = query_start_loc[1:num_decodes + 1].tolist()
            max_seq_lens = seq_lens[:num_decodes].max().item()
            seq_lens = seq_lens[:num_decodes]
            input_positions = input_positions[:num_decode_tokens]
            block_table = block_table[:num_decodes, ...]
            seq_lens_list = seq_lens.tolist()

            # TODO: After the fullgraph supports MTP, the if branch needs to deleted
            assert self.cos_cache is not None
            assert self.sin_cache is not None
            if cos is None and sin is None:
                cos = self.cos_cache[
                    input_positions].unsqueeze(  # type: ignore
                        1).unsqueeze(2)
                sin = self.sin_cache[
                    input_positions].unsqueeze(  # type: ignore
                        1).unsqueeze(2)

                decode_metadata = AscendMLADecodeMetadata(
                    input_positions=input_positions,
                    block_table=block_table,
                    seq_lens=seq_lens,
                    seq_lens_list=seq_lens_list,
                    max_seq_lens=max_seq_lens,
                    attn_mask=common_attn_metadata.spec_attn_mask,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    sin=sin,
                    cos=cos)
            else:
                cos[:num_decode_tokens,
                    ...] = self.cos_cache[input_positions].unsqueeze(
                        1).unsqueeze(2)
                sin[:num_decode_tokens,
                    ...] = self.sin_cache[input_positions].unsqueeze(
                        1).unsqueeze(2)

                decode_metadata = AscendMLADecodeMetadata(
                    input_positions=input_positions,
                    block_table=block_table,
                    seq_lens=seq_lens,
                    seq_lens_list=seq_lens_list,
                    max_seq_lens=max_seq_lens,
                    attn_mask=common_attn_metadata.spec_attn_mask,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    sin=sin[:num_decode_tokens, ...],
                    cos=cos[:num_decode_tokens, ...])

        seq_lens_device = None
        slot_mapping_cpu = None
        num_prefill_tokens_device = None
        num_decode_tokens_device = None
        if has_ucm_sparse():
            ucm_sparse = get_ucm_sparse()
            if os.getenv("VLLM_HASH_ATTENTION", "0") == "1":
                slot_mapping_cpu = slot_mapping.to(device="cpu")
                num_decode_tokens_device = torch.tensor([num_decode_tokens], dtype=torch.int32).to(device=self.device, non_blocking=True)
                num_prefill_tokens_device = torch.tensor([num_prefill_tokens], dtype=torch.int32).to(device=self.device, non_blocking=True)
                if decode_metadata is not None:
                    seq_lens_device = decode_metadata.seq_lens.to(device=self.device, non_blocking=True)
                    decode_metadata.seq_lens_device = seq_lens_device
                    ucm_sparse.build_decode_attention_meta_npu(query_lens, seq_lens, block_table)

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            query_lens=query_lens.tolist(),
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_mask=common_attn_metadata.attn_mask,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=block_table,
            seq_lens=seq_lens,
            enable_dbo_across_dp=common_attn_metadata.enable_dbo_across_dp,
            slot_mapping_cpu=slot_mapping_cpu,
            num_prefill_tokens_device=num_prefill_tokens_device,
            num_decode_tokens_device=num_decode_tokens_device
        )

class PrefillMLAPreprocessResult(NamedTuple):
    q_nope: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    k_nope: Optional[torch.Tensor] = None
    k_pe: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    origin_k_nope: Optional[torch.Tensor] = None
    origin_k_pe: Optional[torch.Tensor] = None

class AscendMLAImpl(MLAAttentionImpl):
    def _mla_preprocess(self, layer_name, hidden_states, kv_cache,
                        attn_metadata, need_gather_q_kv):
        # MLA Preprocess:
        # 1. Perform q_a_proj and q_a_layernorm to obtain q_c
        # 2. Perform kv_a_proj_with_mqa to obtain kv_no_split
        # 3. If need_gather_q_kv, perform all_gather.
        # 4. Preprocess decode tokens, write kv cache and get:
        # decode_ql_nope, decode_q_pe, decode_k_pe, decode_k_nope
        # 5. Preprocess prefill tokens, write kv cache and get:
        # prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe, prefill_value
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        if self.fused_qkv_a_proj is not None:
            maybe_npu_prefetch(inputs=self.fused_qkv_a_proj.weight,
                               dependency=hidden_states,
                               enabled=self.enable_prefetch)
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            # allgather need contiguous data
            kv_no_split = kv_no_split.contiguous()
        else:
            q_c = hidden_states
            kv_no_split = self.kv_a_proj_with_mqa(hidden_states)[0]

        # Process for Flash Comm V1
        q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c, need_gather_q_kv)
        kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split, need_gather_q_kv)

        decode_preprocess_res = None
        prefill_preprocess_res = None
        if has_prefill:
            wait_for_kv_layer_from_connector(layer_name)
        # Preprocess for decode tokens
        if has_decode:
            decode_q_c = q_c[:num_decode_tokens]
            cos = attn_metadata.decode.cos
            sin = attn_metadata.decode.sin
            decode_ql_nope, decode_q_pe = \
                self._q_proj_and_k_up_proj(decode_q_c)
            decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
            decode_slots = attn_metadata.slot_mapping[:num_decode_tokens]
            decode_kv_no_split = kv_no_split[:num_decode_tokens]
            decode_k_pe, decode_k_nope = self.exec_kv_decode(
                decode_kv_no_split, cos, sin, kv_cache, decode_slots)
            decode_preprocess_res = DecodeMLAPreprocessResult(
                decode_ql_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        # Preprocess for prefill tokens
        if has_prefill:
            prefill_kv_no_split = kv_no_split[
                num_decode_tokens:num_actual_tokens]
            prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
            prefill_q = self.q_proj(prefill_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            prefill_q_nope = prefill_q[..., :self.qk_nope_head_dim]
            cos = attn_metadata.prefill.cos
            sin = attn_metadata.prefill.sin
            prefill_slots = attn_metadata.slot_mapping[
                num_decode_tokens:num_actual_tokens]
            prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
            prefill_k_pe, prefill_k_c_normed = self.exec_kv_prefill(
                prefill_kv_no_split, cos, sin, kv_cache, prefill_slots)
            origin_prefill_k_pe = prefill_k_pe
            origin_prefill_k_nope = prefill_k_c_normed
            prefill_k_pe = prefill_k_pe.view(prefill_q_c.shape[0],
                                             self.num_kv_heads, -1)
            prefill_k_nope, prefill_value = self.kv_b_proj(
                prefill_k_c_normed)[0].view(
                    -1, self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim).split(
                        [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            prefill_k_pe = prefill_k_pe.expand(
                (*prefill_k_nope.shape[:-1], -1))
            prefill_preprocess_res = PrefillMLAPreprocessResult(
                prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe,
                prefill_value, origin_prefill_k_nope, origin_prefill_k_pe)
        return decode_preprocess_res, prefill_preprocess_res

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        num_actual_tokens = attn_metadata.num_actual_tokens
        assert attn_metadata.num_decodes is not None and \
        attn_metadata.num_prefills is not None and \
        attn_metadata.num_decode_tokens is not None
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        o_proj_input_shape = (get_forward_context().num_tokens,
                              self.num_heads * self.v_head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

        if os.getenv("VLLM_HASH_ATTENTION") == "1":
            kv_cache, k_hash = kv_cache
        else:
            k_hash = None

        # MLA Preprocess
        forward_context = get_forward_context()
        if (self.enable_mlapo and
            (attn_metadata is None or not forward_context.with_prefill)):
            decode_preprocess_res, prefill_preprocess_res = self._mla_decode_preprocess(
                hidden_states, kv_cache, attn_metadata)
        else:
            decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess(
                layer_name, hidden_states, kv_cache, attn_metadata,
                need_gather_q_kv)

        if decode_preprocess_res is not None:
            # MLA Preprocess for decoding
            query, key, value, sp_out = maybe_execute_sparse_attention_begin(torch.cat([decode_preprocess_res.ql_nope, decode_preprocess_res.q_pe], dim=-1),
                                                                             decode_preprocess_res.k_nope,
                                                                             decode_preprocess_res.k_pe,
                                                                             layer_name,
                                                                             forward_context,
                                                                             output=output,
                                                                             phase="decode",
                                                                             k_hash=k_hash,
                                                                             decode_ql_nope=decode_preprocess_res.ql_nope,
                                                                             decode_q_pe=decode_preprocess_res.q_pe)

            output_decode = self._forward_decode(decode_preprocess_res.ql_nope,
                                                 decode_preprocess_res.q_pe,
                                                 decode_preprocess_res.k_nope,
                                                 decode_preprocess_res.k_pe,
                                                 kv_cache[0].shape[1],
                                                 attn_metadata)
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    o_proj_input[:num_decode_tokens] = output_decode
                    current_ms_metadata.after_comm_event.record()
            else:
                o_proj_input[:num_decode_tokens] = output_decode
            maybe_execute_sparse_attention_finished(torch.cat([decode_preprocess_res.ql_nope, decode_preprocess_res.q_pe], dim=-1),
                                                    decode_preprocess_res.k_nope,
                                                    decode_preprocess_res.k_pe,
                                                    output[:num_decode_tokens],
                                                    layer_name,
                                                    forward_context,
                                                    phase="decode")

        if prefill_preprocess_res is not None:
            # FIX: aicore move should be also placed on the comm stream in dbo,
            # otherwise it may affect the accuracy
            # TODO: use an elegant way to overlap
            query, key, value, sp_out = maybe_execute_sparse_attention_begin(torch.cat([prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe], dim=-1),
                                                                             prefill_preprocess_res.origin_k_nope,
                                                                             prefill_preprocess_res.origin_k_pe,
                                                                             layer_name,
                                                                             forward_context,
                                                                             phase="prefill",
                                                                             k_hash=k_hash)
            output_prefill = self._forward_prefill(
                prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe,
                prefill_preprocess_res.k_nope, prefill_preprocess_res.k_pe,
                prefill_preprocess_res.value, kv_cache, attn_metadata)
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is not None:
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    o_proj_input[num_decode_tokens:] = output_prefill
                    current_ms_metadata.after_comm_event.record()
            else:
                o_proj_input[
                    num_decode_tokens:num_actual_tokens] = output_prefill
            maybe_execute_sparse_attention_finished(torch.cat([prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe], dim=-1),
                                                              prefill_preprocess_res.k_nope,
                                                              prefill_preprocess_res.k_pe,
                                                              output[num_decode_tokens:],
                                                              layer_name,
                                                              forward_context,
                                                              phase="prefill")
        # O proj
        current_ms_metadata = get_multistream_comm_context()
        MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024
        if current_ms_metadata is None:
            maybe_npu_prefetch(inputs=self.o_proj.weight,
                               dependency=o_proj_input,
                               max_size=MAX_O_PROJ_PREFETCH_SIZE,
                               enabled=self.enable_prefetch)

            output[...] = self.o_proj(o_proj_input,
                                      is_prefill=prefill_preprocess_res
                                      is not None)[0]
        else:
            with torch.npu.stream(current_ms_metadata.comm_stream):
                maybe_npu_prefetch(inputs=self.o_proj.weight,
                                   dependency=o_proj_input,
                                   max_size=MAX_O_PROJ_PREFETCH_SIZE,
                                   enabled=self.enable_prefetch)
                output[...] = self.o_proj(o_proj_input,
                                          is_prefill=prefill_preprocess_res
                                          is not None)[0]
                current_ms_metadata.after_comm_event.record()
        del o_proj_input

        has_prefill = attn_metadata.num_prefills > 0
        if has_prefill:
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        return output_padded