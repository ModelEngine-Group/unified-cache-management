"""Replace the vLLM-Ascend 0.23.0 SFA forward with the GSA-aware KV-save fix.

This is the GSA counterpart of
``ucm/integration/vllm/patch/v0230/vllm_ascend/sfa_kv_transfer_patch.py``.
It is toggled by ``ENABLE_GSA_PATCH=1`` (in addition to ``ENABLE_UCM_PATCH=1``)
via ``apply_patch.apply_all_patches`` and patches ``AscendSFAImpl.forward`` with
the GSA-aware implementation that drives
``maybe_execute_sparse_attention_begin/finished`` and the
``gsa_prefetch/warmup/swap`` top-k bookkeeping introduced by
``vllm-ascend-adapt-1020-c8.patch``.

The remaining GSA SFA patches (``__init__``, ``_build``,
``indexer_select_post_process``, ``_execute_sparse_flash_attention_process``)
live in ``gsa/v0_23_0/vllm_ascend/sfa_v1_patch.py``.
"""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def forward(
    self,
    layer_name,
    hidden_states: torch.Tensor,  # query in unified attn
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: M,
    need_gather_q_kv: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    assert output is not None, "Output tensor must be provided."
    is_pdmix = self.vllm_config.kv_transfer_config is None or (
        self.vllm_config.kv_transfer_config.is_kv_producer
        and self.vllm_config.kv_transfer_config.is_kv_consumer
    )
    if attn_metadata is None:
        # Profiling run.
        if self.enable_dsa_cp_with_layer_shard and not _EXTRA_CTX.in_profile_run:
            for layer in self.layer_sharding_kwargs or []:
                if is_hidden_layer(layer):
                    reach_layer_for_shard_weight_series(layer)
        return output.fill_(0)
    if is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer:
        has_prefill = attn_metadata.has_prefill
    cos = attn_metadata.cos
    sin = attn_metadata.sin
    slot_mapping = attn_metadata.slot_mapping
    slot_mapping_cp = None
    if self.enable_dsa_cp:
        assert attn_metadata.dsa_cp_context is not None
        slot_mapping_cp = attn_metadata.dsa_cp_context.slot_mapping_cp
        actual_seq_lengths_query = attn_metadata.dsa_cp_context.actual_seq_lengths_query
        actual_seq_lengths_key = attn_metadata.dsa_cp_context.actual_seq_lengths_key
    else:
        actual_seq_lengths_query = attn_metadata.cum_query_lens
        actual_seq_lengths_key = attn_metadata.seq_lens
    # DCP replicated indexer stores LI cache with the full/no-CP metadata, while
    # SFA KV remains stored with the DCP-sharded slot mapping.
    slot_mapping_sfa = (
        attn_metadata.dcp_context.slot_mapping
        if attn_metadata.dcp_context is not None
        else attn_metadata.slot_mapping
    )

    # Inputs and outputs may be padded for CUDA graphs
    num_input_tokens = attn_metadata.num_input_tokens
    output_padded = output

    # all-gather o_proj weight for prefill stage of PD mix node
    o_proj_full_handle = None
    o_proj_full_param_handles = None
    # Prefill/mixed DSA-CP computes o_proj with a temporary full weight.
    # Decode keeps the original TP path and only exchanges activations.
    full_gather_o_proj_enabled = (
        self.enable_dsa_cp_with_o_proj_tp
        and attn_metadata.attn_state
        not in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        }
    )
    if is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer:
        forward_context: ForwardContext = get_forward_context()
    if self.enable_sfa_prolog_v3 and attn_metadata.attn_state in (
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.SpecDecoding,
    ):
        if self.enable_sp:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states.contiguous(), need_gather_q_kv
            )
        assert slot_mapping.numel() == hidden_states.shape[0], (
            "SFA Prolog V3 requires one cache index per input token, "
            f"got token_x={hidden_states.shape[0]} and cache_index={slot_mapping.numel()}."
        )
        if self.has_indexer:
            k_li, k_li_scale = self.indexer_select_pre_process(
                x=hidden_states, cos=cos, sin=sin
            )
        else:
            k_li, k_li_scale = None, None

        # Prolog updates the paged KV cache in place. Wait for the prompt
        # blocks before writing the first Decode token into their tail block.
        wait_for_kv_layer_from_connector(layer_name)
        hidden_states, ql_nope, q_pe, q_c, _, _ = self._sfa_preprocess_with_prolog_v3(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            cos=cos,
            sin=sin,
            slot_mapping=slot_mapping,
            cache_mode="PA_BSND",
        )
    # run mlapo ops when dsa-cp is disabled, and ensure that num_tokens satisfies the count limitation
    elif self.enable_mlapo and (
        get_ascend_device_type() == AscendDeviceType.A5
        or num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
    ):
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states.contiguous(), need_gather_q_kv
        )
        hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_with_mlapo(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            cos=cos,
            sin=sin,
            slot_mapping=slot_mapping,
            num_input_tokens=num_input_tokens,
        )
        if self.has_indexer:
            k_li, k_li_scale = self.indexer_select_pre_process(
                x=hidden_states,
                cos=cos,
                sin=sin,
            )
        else:
            k_li, k_li_scale = None, None
        wait_for_kv_layer_from_connector(layer_name)
    # native
    else:
        assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.fused_qkv_a_proj.weight, dependency=hidden_states
        )
        if self.enable_sp and not self.enable_dsa_cp:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states.contiguous(), need_gather_q_kv
            )
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
        q_c = self.q_a_layernorm(q_c)

        if self.has_indexer:
            k_li, k_li_scale = self.indexer_select_pre_process(
                x=hidden_states,
                cos=cos,
                sin=sin,
            )
        else:
            k_li, k_li_scale = None, None

        wait_for_kv_layer_from_connector(layer_name)

        if self.enable_dsa_cp:
            assert slot_mapping_cp is not None
            kv_slots = slot_mapping_cp
        else:
            kv_slots = slot_mapping_sfa
        kv_outputs = self.exec_kv(
            kv_no_split, cos, sin, kv_cache, kv_slots, attn_metadata
        )
        k_pe, k_nope = kv_outputs[:2]
        knope_scale = kv_outputs[2] if len(kv_outputs) == 3 else None

        if (
            self.enable_sparse_sfa_c8
            and not self.enable_dsa_cp
            and (
                get_ascend_device_type() != AscendDeviceType.A5 or not self.has_indexer
            )
        ):
            assert k_pe is not None
            assert k_nope is not None
            assert knope_scale is not None
            packed_kv = torch.cat([k_nope, k_pe, knope_scale], dim=-1)
            packed_head_dim = self.sfa_qsfa_packed_kv_head_dim
            assert packed_kv.shape[-1] == packed_head_dim
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(-1, packed_head_dim),
                slot_mapping_sfa.view(-1, 1),
                packed_kv.view(-1, packed_head_dim),
            )

        if self.enable_dsa_cp:
            assert k_pe is not None
            assert k_nope is not None
            async_op = self.enable_dsa_cp_with_layer_shard or full_gather_o_proj_enabled
            kv_ag_handles = []
            # support all_gather kv async for communication calculation overlap
            if self.enable_sparse_sfa_c8:
                assert knope_scale is not None
                fused_kv_parts = [
                    k_nope.view(-1, k_nope.shape[-1]),
                    k_pe.view(-1, k_pe.shape[-1]),
                    knope_scale.view(-1, knope_scale.shape[-1]),
                ]
            else:
                fused_kv_parts = [
                    k_pe.view(-1, k_pe.shape[-1]),
                    k_nope.view(-1, k_nope.shape[-1]),
                ]
                if self.has_indexer and not self.enable_sparse_li_c8:
                    assert k_li is not None
                    fused_kv_parts.append(k_li.view(-1, k_li.shape[-1]))

            fused_kv_input = torch.cat(fused_kv_parts, dim=1)
            fused_kv_no_split, kv_ag_handle = all_gather_async(
                fused_kv_input,
                get_tp_group(),
                async_op=async_op,
            )
            if kv_ag_handle is not None:
                kv_ag_handles.append(kv_ag_handle)

            if self.has_indexer and (
                self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8
            ):
                assert k_li is not None
                k_li, kv_ag_handle = all_gather_async(
                    k_li,
                    get_tp_group(),
                    async_op=async_op,
                )
                if kv_ag_handle is not None:
                    kv_ag_handles.append(kv_ag_handle)
            if self.has_indexer and self.enable_sparse_li_c8:
                assert k_li_scale is not None
                k_li_scale, kv_ag_handle = all_gather_async(
                    k_li_scale,
                    get_tp_group(),
                    async_op=async_op,
                )
                if kv_ag_handle is not None:
                    kv_ag_handles.append(kv_ag_handle)

        ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
        q_pe = self.rope_single(q_pe, cos, sin)
        self._record_dcp_query_gather_context(ql_nope, q_pe, attn_metadata)

        if self.enable_dsa_cp:
            for kv_ag_handle in kv_ag_handles:
                kv_ag_handle.wait()

            if self.enable_dsa_cp_with_layer_shard:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)
            elif full_gather_o_proj_enabled:
                _, o_proj_full_handle = all_gather_async(
                    self.o_proj_tp_weight_gather_input,
                    get_tp_group(),
                    output=self.o_proj_full_gather_pool,
                )
                o_proj_full_param_handles = []
                for (
                    param_name,
                    param,
                ) in self.o_proj_tp_input_sharded_quant_params.items():
                    _, param_handle = all_gather_async(
                        param,
                        get_tp_group(),
                        output=self.o_proj_full_input_sharded_quant_params[param_name],
                    )
                    o_proj_full_param_handles.append(param_handle)

            if kv_cache is not None:
                assert fused_kv_no_split is not None
                if self.enable_sparse_sfa_c8:
                    torch_npu.npu_scatter_nd_update_(
                        kv_cache[0].view(-1, fused_kv_no_split.shape[-1]),
                        slot_mapping_sfa[: attn_metadata.num_actual_tokens].view(-1, 1),
                        fused_kv_no_split[: attn_metadata.num_actual_tokens],
                    )
                    k_pe = None
                    k_nope = None
                elif not self.has_indexer:
                    k_pe, k_nope = fused_kv_no_split.split(
                        [self.qk_rope_head_dim, self.kv_lora_rank],
                        dim=-1,
                    )
                elif not self.enable_sparse_li_c8:
                    k_pe, k_nope, k_li = fused_kv_no_split.split(
                        [self.qk_rope_head_dim, self.kv_lora_rank, self.head_dim],
                        dim=-1,
                    )
                else:
                    k_pe, k_nope = fused_kv_no_split.split(
                        [self.qk_rope_head_dim, self.kv_lora_rank],
                        dim=-1,
                    )
                if not self.enable_sparse_sfa_c8:
                    assert k_pe is not None
                    assert k_nope is not None
                    k_nope = k_nope.view(k_nope.shape[0], 1, -1)
                    k_pe = k_pe.view(k_pe.shape[0], 1, -1)
                    DeviceOperator.reshape_and_cache(
                        key=k_nope[: attn_metadata.num_actual_tokens],
                        value=k_pe[: attn_metadata.num_actual_tokens],
                        key_cache=kv_cache[0],
                        value_cache=kv_cache[1],
                        slot_mapping=slot_mapping_sfa[
                            : attn_metadata.num_actual_tokens
                        ],
                    )

        # DCP's prefill path may all-gather only the blocks referenced by
        # this batch. It must start after the current layer's SFA KV write,
        # but before the indexer/top-k work so communication can overlap it.
        if kv_cache is not None:
            self._record_dcp_kv_gather_context(kv_cache, attn_metadata)

        if self.has_indexer:
            assert k_li is not None
            k_li = self._get_full_kv(k_li, attn_metadata)

    if kv_cache is not None and self.is_kv_producer:
        attn_metadata.reshape_cache_event = torch.npu.Event()

    if (
        (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
        and has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
        and self.layer_id < self.layer_num
    ):
        if self.enable_sparse_sfa_c8:
            dsa_k_cache_idx = 1
            dsa_k_scale_cache_idx = 2
        else:
            dsa_k_cache_idx = 2
            dsa_k_scale_cache_idx = 3
        k_cache = kv_cache[0]
        v_cache = kv_cache[0] if self.enable_sparse_sfa_c8 else kv_cache[1]
        vllm_indexer_cache = kv_cache[dsa_k_cache_idx] if self.has_indexer else None
        dsa_k_cache = self.dsa_k_scale if self.enable_sparse_li_c8 else None
        vllm_dsa_cache = (
            kv_cache[dsa_k_scale_cache_idx] if self.enable_sparse_li_c8 else None
        )
        _, _, _, _ = maybe_execute_sparse_attention_begin(
            None,
            k_cache,
            v_cache,
            layer_name,
            forward_context,
            indexer_cache=self.dsa_k_cache,
            indexer_scale=dsa_k_cache,
            vllm_indexer_cache=vllm_indexer_cache,
            vllm_indexer_scale=vllm_dsa_cache,
            has_indexer=self.has_indexer,
            use_li_c8=self.enable_sparse_li_c8,
        )

    if kv_cache is not None and self.has_indexer:
        assert k_li is not None
        if self.enable_sparse_sfa_c8:
            dsa_k_cache_idx = 1
            dsa_k_scale_cache_idx = 2
        else:
            dsa_k_cache_idx = 2
            dsa_k_scale_cache_idx = 3

        if (
            (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
            and has_ucm_sparse()
            and PTOPK_PREFETCH_ENABLE
            and self.layer_id < self.layer_num
        ):
            torch_npu.npu_scatter_nd_update_(
                self.dsa_k_cache.view(-1, k_li.shape[-1]),
                attn_metadata.dsa_slot_mapping[: attn_metadata.gsa_slot_len].view(
                    -1, 1
                ),
                k_li.view(-1, k_li.shape[-1]),
            )  # b, s, n, d
        elif get_ascend_config().c8_enable_reshape_optim:
            torch.ops._C_ascend.store_kv_block(
                k_li,
                kv_cache[dsa_k_cache_idx],
                attn_metadata.group_len,
                attn_metadata.group_key_idx,
                attn_metadata.group_key_cache_idx,
                attn_metadata.block_size,
            )
        else:
            torch_npu.npu_scatter_nd_update_(
                kv_cache[dsa_k_cache_idx].view(-1, k_li.shape[-1]),
                slot_mapping.view(-1, 1),
                k_li.view(-1, k_li.shape[-1]),
            )  # b, s, n, d
        if self.enable_sparse_li_c8:
            assert len(kv_cache) == (3 if self.enable_sparse_sfa_c8 else 4)
            if k_li_scale is not None:
                if get_ascend_config().c8_enable_reshape_optim:
                    torch.ops._C_ascend.store_kv_block(
                        k_li_scale,
                        kv_cache[dsa_k_scale_cache_idx],
                        attn_metadata.group_len,
                        attn_metadata.group_key_idx,
                        attn_metadata.group_key_cache_idx,
                        attn_metadata.block_size,
                    )
                else:
                    if (
                        (
                            is_pdmix
                            or not self.vllm_config.kv_transfer_config.is_kv_producer
                        )
                        and has_ucm_sparse()
                        and PTOPK_PREFETCH_ENABLE
                        and self.layer_id < self.layer_num
                    ):
                        torch_npu.npu_scatter_nd_update_(
                            self.dsa_k_scale.view(-1, k_li_scale.shape[-1]),
                            attn_metadata.dsa_slot_mapping[
                                : attn_metadata.gsa_slot_len
                            ].view(-1, 1),
                            k_li_scale.view(-1, k_li_scale.shape[-1]),
                        )
                    else:
                        torch_npu.npu_scatter_nd_update_(
                            kv_cache[dsa_k_scale_cache_idx].view(
                                -1, k_li_scale.shape[-1]
                            ),
                            slot_mapping.view(-1, 1),
                            k_li_scale.view(-1, k_li_scale.shape[-1]),
                        )

    if kv_cache is not None and self.is_kv_producer:
        attn_metadata.reshape_cache_event.record()
        notify_kv_cache_written(self.layer_name or "")

    if self.enable_dsa_cp and attn_metadata.dsa_cp_context is not None:
        topk_num_tokens = (
            attn_metadata.dsa_cp_context.local_end_with_pad
            - attn_metadata.dsa_cp_context.local_start
        )
    else:
        topk_num_tokens = num_input_tokens or hidden_states.shape[0]
    if self.skip_topk:
        topk_indices = self._get_indexcache_topk_indices(topk_num_tokens)
    else:
        if not self.has_indexer:
            raise RuntimeError(
                f"skip_topk is False but indexer is None. layer_name={self.layer_name}."
            )
        assert q_c is not None
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
        if self.use_index_cache:
            self._update_indexcache_topk_indices(topk_indices)

    if (
        (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
        and has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
        and self.layer_id < self.layer_num
    ):
        if has_prefill:
            if attn_metadata.gsa_prefetch_len > 0 and self.has_indexer:
                attn_metadata.gsa_prefetch_topk[self.cur_topk_index][
                    : attn_metadata.gsa_prefetch_len
                ].copy_(
                    topk_indices[
                        attn_metadata.gsa_prefetch_index[
                            : attn_metadata.gsa_prefetch_len
                        ]
                    ]
                )
            if attn_metadata.gsa_warmup_len > 0 and self.has_indexer:
                attn_metadata.gsa_warmup_topk[self.cur_topk_index][
                    : attn_metadata.gsa_warmup_len
                ].copy_(
                    topk_indices[
                        attn_metadata.gsa_warmup_index[: attn_metadata.gsa_warmup_len]
                    ]
                )
            if attn_metadata.gsa_swap_len != 0:
                topk_indices.index_copy_(
                    0,
                    attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len],
                    attn_metadata.gsa_swap_topk[self.cur_topk_index][
                        : attn_metadata.gsa_swap_len, :, :DEFAULT_TOPK_LEN
                    ],
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

            if attn_metadata.gsa_swap_len != 0:
                attn_output_gsa = self._execute_sparse_flash_attention_process(
                    ql_nope[attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len]],
                    q_pe[attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len]],
                    kv_cache,
                    attn_metadata.gsa_swap_topk[self.cur_topk_index][
                        : attn_metadata.gsa_swap_len
                    ],
                    attn_metadata,
                    attn_metadata.gsa_prefill_query[
                        : attn_metadata.gsa_swap_len // self.num_spec_tokens
                    ],
                    actual_seq_lengths_key[
                        attn_metadata.select_block_index[
                            : attn_metadata.gsa_swap_len // self.num_spec_tokens
                        ]
                    ],
                    attn_metadata.select_block_index[
                        : attn_metadata.gsa_swap_len // self.num_spec_tokens
                    ],
                )
                attn_output.index_copy_(
                    0,
                    attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len],
                    attn_output_gsa,
                )
        else:
            if attn_metadata.gsa_warmup_len > 0 and self.has_indexer:
                torch.ops._C_ascend.gsa_gather_copy(
                    attn_metadata.gsa_warmup_topk[self.cur_topk_index][
                        : attn_metadata.gsa_warmup_len
                    ].view(-1, DEFAULT_TOPK_LEN),
                    topk_indices.view(-1, DEFAULT_TOPK_LEN),
                    attn_metadata.gsa_warmup_index[: attn_metadata.gsa_warmup_len]
                    .view(-1, 1)
                    .expand((-1, DEFAULT_TOPK_LEN)),
                )
            if attn_metadata.gsa_prefetch_len != 0 and self.has_indexer:
                torch.ops._C_ascend.gsa_gather_copy(
                    attn_metadata.gsa_prefetch_topk[self.cur_topk_index][
                        : attn_metadata.gsa_prefetch_len
                    ].view(-1, DEFAULT_TOPK_LEN),
                    topk_indices.view(-1, DEFAULT_TOPK_LEN),
                    attn_metadata.gsa_prefetch_index[: attn_metadata.gsa_prefetch_len]
                    .view(-1, 1)
                    .expand((-1, DEFAULT_TOPK_LEN)),
                )
            topk_indices_out = torch.ops._C_ascend.gsa_topk_expand_tmp(
                topk_indices.view(-1, DEFAULT_TOPK_LEN),
                attn_metadata.gsa_swap_topk[self.cur_topk_index][
                    : attn_metadata.gsa_swap_len
                ].view(-1, SPA_TOKENS_LEN),
                attn_metadata.gsa_swap_index[: attn_metadata.gsa_swap_len]
                .view(-1, 1)
                .expand((-1, SPA_TOKENS_LEN)),
                attn_metadata.gsa_topk_win_tmp,
                [0, SPA_TOKENS_LEN - DEFAULT_TOPK_LEN, 0, 0],
                -1,
                SPA_TOKENS_LEN,
            )
            attn_output = self._execute_sparse_flash_attention_process(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices_out,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
            )
    else:
        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

    if (
        (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
        and has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
        and self.layer_id < self.layer_num
    ):
        k_cache = kv_cache[0]
        v_cache = kv_cache[0] if self.enable_sparse_sfa_c8 else kv_cache[1]
        maybe_execute_sparse_attention_finished(
            ql_nope,
            k_cache,
            v_cache,
            attn_output,
            layer_name,
            forward_context,
            has_indexer=self.has_indexer,
            use_li_c8=self.enable_sparse_li_c8,
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
        # SFA DSA-CP mixed mode keeps o_proj weight sharded in the TP domain:
        # 1. prefill/mixed: gather TP shards into a temporary full weight.
        # 2. decode-only: all-to-all hidden states, then run TP o_proj.
        result, require_o_proj_forward = self._handle_o_proj_weight_switch_and_forward(
            attn_output=attn_output,
            output=output,
            o_proj_full_handle=o_proj_full_handle,
            o_proj_full_param_handles=o_proj_full_param_handles,
            should_shard_weight=full_gather_o_proj_enabled,
        )
        if not require_o_proj_forward:
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
            return result
        attn_output = result

    output[...] = self.o_proj(attn_output)[0]

    maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

    return output_padded


@when_imported("vllm_ascend.attention.sfa_v1")
def patch_sfa_kv_transfer(mod):
    # Run the copied implementation with the original sfa_v1 module globals.
    patched_forward = FunctionType(
        forward.__code__,
        vars(mod),
        name=forward.__name__,
        argdefs=forward.__defaults__,
        closure=forward.__closure__,
    )
    patched_forward.__annotations__ = forward.__annotations__
    patched_forward.__kwdefaults__ = forward.__kwdefaults__

    patch_or_inject(mod.AscendSFAImpl, "forward", patched_forward)
    logger.info("UCM Ascend SFA GSA KV-transfer forward patch applied")
