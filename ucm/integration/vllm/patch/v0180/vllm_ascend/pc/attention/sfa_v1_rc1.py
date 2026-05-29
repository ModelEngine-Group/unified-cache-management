import torch
import torch_npu
from vllm.distributed import get_tp_group
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import (
    MAX_O_PROJ_PREFETCH_SIZE,
    MLAPO_MAX_SUPPORTED_TOKENS,
)
from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.layer_shard_linear import (
    is_hidden_layer,
    reach_layer_for_shard_weight_series,
)
from vllm_ascend.utils import get_weight_prefetch_method


class AscendSFAImpl:
    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            if self.enable_dsa_cp_with_layer_shard and not _EXTRA_CTX.in_profile_run:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)
            return output.fill_(0)

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        slot_mapping_cp = None
        if self.enable_dsa_cp:
            assert attn_metadata.dsa_cp_context is not None
            slot_mapping_cp = attn_metadata.dsa_cp_context.slot_mapping_cp
            actual_seq_lengths_query = (
                attn_metadata.dsa_cp_context.actual_seq_lengths_query
            )
            actual_seq_lengths_key = attn_metadata.dsa_cp_context.actual_seq_lengths_key
        else:
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            actual_seq_lengths_key = attn_metadata.seq_lens

        # Inputs and outputs may be padded for CUDA graphs
        num_input_tokens = attn_metadata.num_input_tokens
        output_padded = output

        # all-gather o_proj weight for prefill stage of PD mix node
        o_proj_full_handle = None
        # if is PD mix stage, using original TP o_proj weight, and also need to full gather for o_proj
        # weight for prefill stage.
        full_gather_o_proj_enabled = (
            self.enable_dsa_cp_with_o_proj_tp
            and attn_metadata.attn_state
            not in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            }
        )

        # run mlapo ops when dsa-cp is disabled, and ensure that num_tokens satisfies the count limitation
        if self.enable_mlapo and num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS:
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_with_mlapo(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping,
                num_input_tokens=num_input_tokens,
            )
            k_li, k_li_scale = self.indexer_select_pre_process(
                x=hidden_states, cos=cos, sin=sin
            )
            # [patch] Add 'wait_for_kv_layer_from_connector' call for mlapo path
            wait_for_kv_layer_from_connector(layer_name)
        # native
        else:
            assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
            weight_prefetch_method = get_weight_prefetch_method()
            weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
                inputs=self.fused_qkv_a_proj.weight, dependency=hidden_states
            )
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
            q_c = self.q_a_layernorm(q_c)

            k_li, k_li_scale = self.indexer_select_pre_process(
                x=hidden_states, cos=cos, sin=sin
            )

            wait_for_kv_layer_from_connector(layer_name)

            if self.enable_dsa_cp:
                assert slot_mapping_cp is not None
                k_pe, k_nope = self.exec_kv(
                    kv_no_split, cos, sin, kv_cache, slot_mapping_cp, attn_metadata
                )
            else:
                k_pe, k_nope = self.exec_kv(
                    kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata
                )

            if self.enable_dsa_cp:
                assert k_pe is not None
                assert k_nope is not None
                assert k_li is not None
                async_op = (
                    self.enable_dsa_cp_with_layer_shard or full_gather_o_proj_enabled
                )
                # support all_gather kv async for communication calculation overlap
                if not self.use_sparse_c8_indexer:
                    fused_kv_no_split, kv_ag_handle = all_gather_async(
                        torch.cat(
                            [
                                k_pe.view(-1, k_pe.shape[-1]),
                                k_nope.view(-1, k_nope.shape[-1]),
                                k_li.view(-1, k_li.shape[-1]),
                            ],
                            dim=1,
                        ),
                        get_tp_group(),
                        async_op=async_op,
                    )
                else:
                    # due to different dtypes, we have to split commu pass
                    assert k_li_scale is not None
                    fused_kv_no_split, _ = all_gather_async(
                        torch.cat(
                            [
                                k_pe.view(-1, k_pe.shape[-1]),
                                k_nope.view(-1, k_nope.shape[-1]),
                            ],
                            dim=1,
                        ),
                        get_tp_group(),
                        async_op=async_op,
                    )
                    k_li, _ = all_gather_async(
                        k_li,
                        get_tp_group(),
                        async_op=async_op,
                    )
                    k_li_scale, kv_ag_handle = all_gather_async(
                        k_li_scale,
                        get_tp_group(),
                        async_op=async_op,
                    )

            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, cos, sin)

            if self.enable_dsa_cp:
                if kv_ag_handle is not None:
                    kv_ag_handle.wait()

                if self.enable_dsa_cp_with_layer_shard:
                    for layer in self.layer_sharding_kwargs or []:
                        if is_hidden_layer(layer):
                            reach_layer_for_shard_weight_series(layer)
                elif full_gather_o_proj_enabled:
                    _, o_proj_full_handle = all_gather_async(
                        self.o_proj_tp_weight,
                        get_tp_group(),
                        output=type(self).o_proj_full_pool,
                    )

                if kv_cache is not None:
                    assert fused_kv_no_split is not None
                    if not self.use_sparse_c8_indexer:
                        k_pe, k_nope, k_li = fused_kv_no_split.split(
                            [self.qk_rope_head_dim, self.kv_lora_rank, self.head_dim],
                            dim=-1,
                        )
                    else:
                        k_pe, k_nope = fused_kv_no_split.split(
                            [self.qk_rope_head_dim, self.kv_lora_rank], dim=-1
                        )
                    k_nope = k_nope.view(k_nope.shape[0], 1, -1)
                    k_pe = k_pe.view(k_pe.shape[0], 1, -1)
                    DeviceOperator.reshape_and_cache(
                        key=k_nope[: attn_metadata.num_actual_tokens],
                        value=k_pe[: attn_metadata.num_actual_tokens],
                        key_cache=kv_cache[0],
                        value_cache=kv_cache[1],
                        slot_mapping=slot_mapping[: attn_metadata.num_actual_tokens],
                    )

            k_li = self._get_full_kv(k_li, attn_metadata)

        if kv_cache is not None:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2].view(-1, k_li.shape[-1]),
                slot_mapping.view(-1, 1),
                k_li.view(-1, k_li.shape[-1]),
            )  # b, s, n, d
            if self.use_sparse_c8_indexer:
                assert len(kv_cache) == 4
                assert k_li_scale is not None
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[3].view(-1, k_li_scale.shape[-1]),
                    slot_mapping.view(-1, 1),
                    k_li_scale.view(-1, k_li_scale.shape[-1]),
                )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()

        topk_indices = self.indexer_select_post_process(
            x=hidden_states,
            q_c=q_c,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
        )

        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

        attn_output = self._v_up_proj(attn_output)
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.o_proj.weight,
            dependency=attn_output,
            max_size=MAX_O_PROJ_PREFETCH_SIZE,
            linear_layer=self.o_proj,
        )

        if self.enable_dsa_cp_with_o_proj_tp:
            # When using SFA-CP with pd mixed, o_proj has two cases:
            # 1. prefill: o_proj is a TP weight, we need to all-gather o_proj weight to switch TP=1.
            # 2. decode: all-to-all the hidden_state before the o_proj forward.
            result, require_o_proj_forward = (
                self._handle_o_proj_weight_switch_and_forward(
                    attn_output=attn_output,
                    output=output,
                    o_proj_full_handle=o_proj_full_handle,
                    should_shard_weight=full_gather_o_proj_enabled,
                )
            )
            if not require_o_proj_forward:
                return result
            attn_output = result

        if self.enable_dsa_cp_strict_accuracy:
            send = (
                attn_output.view(-1, self.tp_size, self.num_heads * self.v_head_dim)
                .permute(1, 0, 2)
                .reshape(-1, self.num_heads * self.v_head_dim)
            )

            attn_output = torch.empty_like(send)
            torch.distributed.all_to_all_single(
                attn_output, send, group=get_tp_group().device_group
            )

        output[...] = self.o_proj(attn_output)[0]

        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output_padded
