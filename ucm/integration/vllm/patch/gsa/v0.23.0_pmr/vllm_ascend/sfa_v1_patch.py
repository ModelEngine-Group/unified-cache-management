"""GSA sfa_v1 scaffolding for vLLM-Ascend 0.23.0."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _build(
    self,
    common_attn_metadata: AscendCommonAttentionMetadata,
    draft_index: int | None = None,
) -> AscendSFAMetadata:
    num_reqs = common_attn_metadata.num_reqs
    num_actual_tokens = common_attn_metadata.num_actual_tokens
    num_input_tokens = common_attn_metadata.num_input_tokens

    block_table = common_attn_metadata.block_table_tensor[:num_reqs]
    slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
    input_positions = common_attn_metadata.positions[:num_input_tokens].long()

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    has_prefill = any(query_lens_cpu > self.decode_threshold)

    block_size = self.kernel_block_size

    cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
    seq_lens = common_attn_metadata.seq_lens[:num_reqs]

    # Prefer _seq_lens_cpu (always available, updated during draft
    # iterations) over seq_lens_cpu (None in async spec decode mode).
    if common_attn_metadata._seq_lens_cpu is not None:
        seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
    elif common_attn_metadata.seq_lens_cpu is not None:
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
    else:
        seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

    cos, sin = get_cos_and_sin_mla(input_positions, use_cache=(draft_index is None))

    dsa_cp_context = None
    if self.enable_dsa_cp:
        global_tp_size = get_tp_group().world_size
        num_tokens = num_input_tokens
        num_tokens_pad = _round_up(num_tokens, global_tp_size)
        num_tokens_per_device = num_tokens_pad // global_tp_size
        local_start = get_tp_group().rank_in_group * num_tokens_per_device
        local_end_with_pad = local_start + num_tokens_per_device
        local_end = min(local_end_with_pad, num_actual_tokens)

        pad_size = num_tokens_pad - cos.shape[0]
        assert (
            cos.shape == sin.shape
        ), f"cos.shape must be equal to sin.shape, got {cos.shape} and {sin.shape}"

        if pad_size > 0:
            cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
            sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))

        pad_size_slot = num_tokens_pad - slot_mapping.shape[0]
        if pad_size_slot > 0:
            slot_mapping = nn.functional.pad(slot_mapping, (0, pad_size_slot), value=-1)
        else:
            slot_mapping = slot_mapping[:num_tokens_pad]
        slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]

        cos = cos[local_start:local_end_with_pad]
        sin = sin[local_start:local_end_with_pad]

        assert (
            cos.shape[0] == num_tokens_per_device
        ), f"cos.shape[0] must be equal to num_tokens_per_device, \
                got {cos.shape[0]} and {num_tokens_per_device}"
        assert (
            slot_mapping_cp.shape[0] == num_tokens_per_device
        ), f"slot_mapping_cp.shape[0] must be equal to num_tokens_per_device, \
                got {slot_mapping_cp.shape[0]} and {num_tokens_per_device}"
        assert (
            slot_mapping.shape[0] == num_tokens_pad
        ), f"slot_mapping.shape[0] must be equal to num_tokens_pad, \
                got {slot_mapping.shape[0]} and {num_tokens_pad}"

        if draft_index is not None:
            assert self.spec_actual_seq_lengths_query is not None
            assert self.spec_actual_seq_lengths_key is not None
            # Per-draft-step buffers: independent, graph-stable storage so
            # later draft steps don't clobber earlier ones' metadata.
            actual_seq_lengths_query = self.spec_actual_seq_lengths_query[
                draft_index - 1
            ]
            actual_seq_lengths_key = self.spec_actual_seq_lengths_key[draft_index - 1]
        else:
            actual_seq_lengths_query = self.actual_seq_lengths_query
            actual_seq_lengths_key = self.actual_seq_lengths_key

        num_segs = cum_query_lens.shape[0]

        # Vectorized per-request local query/key lengths for this rank's
        # [local_start, local_end_with_pad) slice. Replaces a Python loop
        # that did 2 .item() NPU->CPU syncs per request (2 * num_reqs
        # syncs/step); now fully on-device with zero syncs.
        # global_start[i] = 0 for i==0, else cum_query_lens[i-1]
        global_start = common_attn_metadata.query_start_loc[:num_segs]
        global_end = cum_query_lens

        # Clip each request's [global_start, global_end) to the local range.
        # num_local_tokens may be < 0 when the request falls entirely
        # outside [local_start, local_end_with_pad); clamp before cumsum.
        req_local_start = global_start.clamp(min=local_start)
        req_local_end = global_end.clamp(max=local_end_with_pad)
        num_local_tokens = req_local_end - req_local_start

        local_query_lens = torch.cumsum(num_local_tokens.clamp(min=0), dim=0)
        offset = global_end - req_local_end  # request tokens on later ranks
        valid_local_req = (num_local_tokens > 0) & (seq_lens > 0)
        local_key_lens = torch.where(
            valid_local_req,
            torch.clamp_min(seq_lens - offset, 0),
            torch.zeros_like(seq_lens),
        )

        actual_seq_lengths_query[:num_segs] = local_query_lens
        actual_seq_lengths_key[:num_segs] = local_key_lens
        actual_seq_lengths_query = actual_seq_lengths_query[:num_reqs]
        actual_seq_lengths_key = actual_seq_lengths_key[:num_reqs]

        dsa_cp_context = DSACPContext(
            num_tokens=num_tokens,
            num_tokens_pad=num_tokens_pad,
            local_start=local_start,
            local_end=local_end,
            local_end_with_pad=local_end_with_pad,
            slot_mapping_cp=slot_mapping_cp,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
        )

    if get_ascend_config().c8_enable_reshape_optim:
        if draft_index is not None:
            assert self.spec_group_len is not None
            assert self.spec_group_key_idx is not None
            assert self.spec_group_key_cache_idx is not None
            group_len = self.spec_group_len[draft_index - 1]
            group_key_idx = self.spec_group_key_idx[draft_index - 1]
            group_key_cache_idx = self.spec_group_key_cache_idx[draft_index - 1]
        else:
            group_len = self.group_len
            group_key_idx = self.group_key_idx
            group_key_cache_idx = self.group_key_cache_idx
        actual_group_len = group_len[:num_input_tokens]
        actual_group_key_idx = group_key_idx[:num_input_tokens]
        actual_group_key_cache_idx = group_key_cache_idx[:num_input_tokens]
        torch.ops._C_ascend.store_kv_block_metadata(
            slot_mapping,
            actual_group_len,
            actual_group_key_idx,
            actual_group_key_cache_idx,
            block_size,
        )
    else:
        actual_group_len = None
        actual_group_key_idx = None
        actual_group_key_cache_idx = None

    return self.metadata_cls(  # type: ignore
        has_prefill=has_prefill,
        num_input_tokens=common_attn_metadata.num_input_tokens,
        num_actual_tokens=num_actual_tokens,
        cum_query_lens=cum_query_lens,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        slot_mapping=slot_mapping,
        head_dim=self.model_config.get_head_size(),
        attn_mask=self.attn_mask_builder.get_attention_mask(
            common_attn_metadata.causal, self.model_config
        ),
        attn_state=common_attn_metadata.attn_state,
        block_table=block_table,
        sin=sin[:num_input_tokens],
        cos=cos[:num_input_tokens],
        dsa_cp_context=dsa_cp_context,
        block_size=block_size,
        group_len=actual_group_len,
        group_key_idx=actual_group_key_idx,
        group_key_cache_idx=actual_group_key_cache_idx,
    )


def __init__(
    self,
    num_heads: int,
    head_size: int,
    scale: float,
    num_kv_heads: int,
    alibi_slopes: list[float] | None,
    sliding_window: int | None,
    kv_cache_dtype: str,
    logits_soft_cap: float | None,
    attn_type: str,
    kv_sharing_target_layer_name: str | None,
    **kwargs,
) -> None:
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import (
        MAX_INDEXER_BLOCK_NUM,
        PTOPK_PREFETCH_ENABLE,
        init_indexer_layer_map,
    )

    self.num_heads = num_heads
    self.head_size = head_size
    self.scale = float(scale)
    self.num_kv_heads = num_kv_heads
    self.kv_cache_dtype = kv_cache_dtype

    assert kv_sharing_target_layer_name is not None
    self.layer_id = int(kv_sharing_target_layer_name.split(".")[2])

    # MLA Args
    self.q_lora_rank = kwargs["q_lora_rank"]
    self.kv_lora_rank = kwargs["kv_lora_rank"]
    self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
    self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
    self.qk_head_dim = kwargs["qk_head_dim"]
    self.v_head_dim = kwargs["v_head_dim"]
    self.rotary_emb = kwargs["rotary_emb"]
    self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
    self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
    self.kv_b_proj = kwargs["kv_b_proj"]
    self.o_proj = kwargs["o_proj"]
    self.indexer = kwargs["indexer"]
    self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
    self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
    self.q_a_layernorm = kwargs.get("q_a_layernorm")
    self.num_queries_per_kv = self.num_heads // self.num_kv_heads
    self.tp_size = get_tensor_model_parallel_world_size()
    self.tp_rank = get_tp_group().rank_in_group
    self.q_b_proj = kwargs["q_b_proj"]
    self.skip_topk = kwargs.get("skip_topk", False)
    self.topk_indices_buffer = kwargs.get("topk_indices_buffer")

    ascend_config = get_ascend_config()
    self.vllm_config = get_current_vllm_config()
    kv_transfer_config = self.vllm_config.kv_transfer_config
    self.is_kv_producer = (
        kv_transfer_config is not None and kv_transfer_config.is_kv_producer
    )
    self.is_kv_consumer = (
        kv_transfer_config is not None and kv_transfer_config.is_kv_consumer
    )

    self.sfa_qsfa_tile_size = SFA_QSFA_TILE_SIZE
    self.sfa_qsfa_packed_kv_head_dim = 0
    self.sfa_qsfa_k_nope_clip_alpha: torch.Tensor | None = None
    self.sfa_qsfa_kr_cache_dummy: torch.Tensor | None = None

    self.local_num_heads = self.num_heads

    self.num_spec_tokens = 1
    speculative_config = self.vllm_config.speculative_config
    if speculative_config:
        self.num_spec_tokens += speculative_config.num_speculative_tokens
    self.layer_num = self.vllm_config.model_config.get_num_layers(
        self.vllm_config.parallel_config
    )
    self.is_pdmix = self.vllm_config.kv_transfer_config is None or (
        self.vllm_config.kv_transfer_config.is_kv_producer
        and self.vllm_config.kv_transfer_config.is_kv_consumer
    )
    self.layer_name = kwargs.get("layer_name")
    hf_config = self.vllm_config.model_config.hf_config
    hf_text_config = getattr(self.vllm_config.model_config, "hf_text_config", None)
    config_candidates = (hf_config, hf_text_config)
    self.index_cache_enabled = _get_config_bool(
        config_candidates,
        "use_index_cache",
    ) or _has_shared_indexer_layers(config_candidates)
    self.use_index_cache = self.skip_topk or self.index_cache_enabled
    self.has_indexer = self.indexer is not None
    if not self.has_indexer and not self.skip_topk:
        raise ValueError(
            "Indexer is required for DSA unless skip_topk is enabled. "
            f"Got indexer=None, skip_topk={self.skip_topk}, "
            f"layer_name={self.layer_name}."
        )
    if not self.has_indexer and self.topk_indices_buffer is None:
        raise ValueError(
            "topk_indices_buffer is required when indexer is None and "
            f"skip_topk is enabled. layer_name={self.layer_name}."
        )
    # indexer param
    if self.has_indexer:
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        self.wq_b = self.indexer.wq_b
        self.wk_weights_proj = self.indexer.wk_weights_proj
        self.k_norm = self.indexer.k_norm
    else:
        self.n_head = getattr(hf_config, "index_n_heads", 0)
        self.head_dim = getattr(hf_config, "index_head_dim", 0)
        self.wq_b = None
        self.wk_weights_proj = None
        self.k_norm = None
    self.cp_size = 1
    self.is_rope_neox_style = True
    self.use_torch_npu_lightning_indexer = False
    if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
        self.is_rope_neox_style = False
        self.use_torch_npu_lightning_indexer = True

    # Sparse C8 has two independent meanings in SFA:
    # - SFA packed KV cache for npu_kv_quant_sparse_flash_attention.
    # - C8 indexer cache for lightning indexer.
    # The user-facing switches control these layouts independently, and
    # layers without an indexer only apply the SFA setting.
    self.enable_sparse_sfa_c8 = ascend_config.enable_sparse_sfa_c8
    self.enable_sparse_li_c8 = self.has_indexer and ascend_config.is_sparse_li_c8_layer(
        self.layer_name
    )
    if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
        if get_ascend_device_type() == AscendDeviceType.A5:
            self.c8_k_cache_dtype = torch.float8_e4m3fn
            self.c8_k_scale_cache_dtype = torch.float32
        else:
            self.c8_k_cache_dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16

    if self.enable_sparse_sfa_c8:
        self.sfa_qsfa_packed_kv_head_dim = get_sfa_qsfa_packed_head_dim(
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.sfa_qsfa_tile_size,
        )
    # PD decode consumers with sparse C8 can use mla_prolog_v3 to write the packed KV cache.
    # TODO: Re-enable after the community CANN baseline upgrades from 9.0 to 9.1.
    # npu_mla_prolog_v3 depends on CANN 9.1 and is not available in the current community CANN 9.0.
    cann_version = getattr(torch.version, "cann", None)
    if cann_version is not None:
        from packaging.version import Version

        sfa_prolog_v3_supported = Version(cann_version) >= Version("9.1.0")
    else:
        sfa_prolog_v3_supported = False
    self.enable_sfa_prolog_v3 = (
        sfa_prolog_v3_supported
        and self.is_kv_consumer
        and self.enable_sparse_sfa_c8
        and get_ascend_device_type() != AscendDeviceType.A5
    )
    self.enable_mlapo = ascend_config.enable_mlapo and not (
        self.enable_sfa_prolog_v3
        or (
            self.enable_sparse_sfa_c8
            and get_ascend_device_type() != AscendDeviceType.A5
        )
    )

    # Effective in SFA when FlashComm is enabled.
    self.enable_dsa_cp = enable_dsa_cp()
    self.enable_sp = enable_sp()

    if has_ucm_sparse() and PTOPK_PREFETCH_ENABLE:
        if self.has_indexer and (
            self.is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer
        ):
            # gsa param
            num_slots = MAX_INDEXER_BLOCK_NUM
            if self.enable_sparse_li_c8:
                indexer_type = self.c8_k_cache_dtype
            else:
                indexer_type = self.vllm_config.model_config.dtype

            self.dsa_k_cache = torch.empty(
                (num_slots, self.vllm_config.cache_config.block_size, 1, 128),
                dtype=indexer_type,
                device=self.vllm_config.device_config.device_type,
                pin_memory=False,
            )
            if self.enable_sparse_li_c8:
                self.dsa_k_scale = torch.empty(
                    (num_slots, self.vllm_config.cache_config.block_size, 1, 1),
                    dtype=self.c8_k_scale_cache_dtype,
                    device=self.vllm_config.device_config.device_type,
                    pin_memory=False,
                )
        else:
            self.dsa_k_cache = None
    self.topk_layer_info, self.topk_layer_map, self.indexr_num = init_indexer_layer_map(
        self.vllm_config
    )
    if self.layer_id < self.layer_num:
        self.cur_topk_index = self.topk_layer_map[self.layer_id]
    else:
        self.cur_topk_index = self.layer_num - 1
    # Enable layer sharding via DSA-CP on the P node in the PD-disaggregated setup.
    self.enable_dsa_cp_with_layer_shard = enable_dsa_cp_with_layer_shard()

    # SFA DSA-CP mixed deployments keep o_proj in the existing TP layout.
    # Decode can use the TP-sharded o_proj directly after an activation
    # all-to-all, while prefill/mixed batches temporarily gather the TP
    # shards into a full-weight buffer because their SFA output is not
    # TP-sharded. This is part of the DSA-CP mixed-mode data path rather
    # than an independent user-facing feature switch.
    self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp()

    if self.enable_dsa_cp:
        self.local_num_heads = self.num_heads * self.tp_size
        if self.enable_dsa_cp_with_layer_shard:
            self.layer_sharding_kwargs = []
            for layer_name in get_ascend_config().layer_sharding or []:
                if layer_name in kwargs:
                    self.layer_sharding_kwargs.append(kwargs[layer_name])
                else:
                    logger.warning_once(
                        f"Layer '{layer_name}' not found in kwargs, skipping sharding. "
                        f"Check layer_sharding config and model layer names."
                    )
            register_all_layers_to_shard_weight_series(self.layer_sharding_kwargs)


def indexer_select_post_process(
    self,
    x: torch.Tensor,
    q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: M,
    cos: torch.Tensor,
    sin: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
):
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import IS_INDEXER_OFFLOAD, PTOPK_PREFETCH_ENABLE

    if not self.has_indexer:
        raise RuntimeError(
            f"indexer_select_post_process should not be called when indexer is None. layer_name={self.layer_name}."
        )

    assert self.wk_weights_proj is not None
    assert self.wq_b is not None

    kw, _ = self.wk_weights_proj(x)
    weights = kw[:, self.head_dim :]
    if isinstance(q_c, tuple):
        q_c_tensor, q_c_scale = q_c
        q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
        quant_matmul_kwargs = dict(
            bias=None,
            output_dtype=x.dtype,
        )
        if q_c_tensor.dtype == torch.float8_e4m3fn:
            if q_c_scale.dim() == 2:
                q_c_scale = q_c_scale.view(q_c_scale.shape[0], -1, 2)
            quant_matmul_kwargs.update(
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                group_sizes=[
                    1,
                    1,
                    getattr(self.wq_b.quant_method.quant_method, "group_size", 32),
                ],
            )
        elif q_c_scale.dim() > 1 and q_c_scale.shape[-1] == 1:
            q_c_scale = q_c_scale.squeeze(dim=-1)
        q_li = torch_npu.npu_quant_matmul(
            q_c_tensor,
            self.wq_b.weight,
            self.wq_b.weight_scale,
            pertoken_scale=q_c_scale,
            **quant_matmul_kwargs,
        )
    else:
        q_li, _ = self.wq_b(q_c)
    q_li = q_li.view(-1, self.n_head, self.head_dim)
    if HAS_TRITON:
        q_li = rope_forward_triton_siso(
            q_li,
            cos,
            sin,
            rope_dim=self.qk_rope_head_dim,
            is_neox_style=self.is_rope_neox_style,
        )
    else:
        q_li_pe, q_li_nope = torch.split(
            q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )

        q_li_pe = q_li_pe.unsqueeze(2)
        q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
        q_li_pe = q_li_pe.squeeze(2)
        q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

    q_li_scale = None
    q_li_shape_ori = None
    if self.enable_sparse_li_c8:
        q_li_shape_ori = q_li.shape
        q_li = q_li @ AscendSFAImpl.q_hadamard
        q_li, q_li_scale = torch_npu.npu_dynamic_quant(
            q_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype
        )
        q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]

    if (
        has_ucm_sparse()
        and PTOPK_PREFETCH_ENABLE
        and self.layer_id < self.layer_num
        and (self.is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
    ):
        actual_seq_lengths_key = attn_metadata.dsa_seq_len
        if IS_INDEXER_OFFLOAD and self.has_indexer:
            torch.ops._C_ascend.indexer_gather_qw(
                q_li.view(-1, self.n_head * self.head_dim),
                weights.view(-1, self.n_head),
                attn_metadata.indexer_last_query[self.layer_id][
                    : attn_metadata.indexer_last_query_len
                ].view(-1, self.n_head * self.head_dim),
                attn_metadata.indexer_last_weight[self.layer_id][
                    : attn_metadata.indexer_last_query_len
                ].view(-1, self.n_head),
                attn_metadata.indexer_last_q_index[
                    : attn_metadata.indexer_last_query_len
                ]
                .view(-1, 1)
                .expand((-1, self.n_head * self.head_dim)),
                attn_metadata.indexer_last_q_index[
                    : attn_metadata.indexer_last_query_len
                ]
                .view(-1, 1)
                .expand((-1, self.n_head)),
            )

    record_attention_compute_start()
    return DeviceOperator.indexer_select_post_process(
        self,
        q_li,
        q_li_scale,
        q_li_shape_ori,
        weights,
        kv_cache,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        self.enable_sparse_li_c8,
        self.use_torch_npu_lightning_indexer,
    )


def _execute_sparse_flash_attention_process(
    self,
    ql_nope,
    q_pe,
    kv_cache,
    topk_indices,
    attn_metadata,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    swap_len: torch.Tensor = None,
):
    return DeviceOperator.execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        swap_len,
    )


@when_imported("vllm_ascend.attention.sfa_v1")
def _patch(mod):
    # Inject GSA-related names into the sfa_v1 module namespace so that the
    # existing sfa_kv_transfer_patch forward (and the replacement methods above)
    # can resolve them through vars(mod).
    try:
        from vllm.forward_context import ForwardContext, get_forward_context
        from vllm.model_executor.layers.attention.attention import (
            maybe_execute_sparse_attention_begin,
            maybe_execute_sparse_attention_finished,
        )

        from ucm.sparse.state import has_ucm_sparse
        from ucm.sparse.utils import (
            DEFAULT_TOPK_LEN,
            IS_INDEXER_OFFLOAD,
            IS_OPEN_GSA,
            MAX_INDEXER_BLOCK_NUM,
            PTOPK_PREFETCH_ENABLE,
            SPA_TOKENS_LEN,
            init_indexer_layer_map,
        )

        for _name, _val in {
            "ForwardContext": ForwardContext,
            "get_forward_context": get_forward_context,
            "maybe_execute_sparse_attention_begin": maybe_execute_sparse_attention_begin,
            "maybe_execute_sparse_attention_finished": maybe_execute_sparse_attention_finished,
            "has_ucm_sparse": has_ucm_sparse,
            "PTOPK_PREFETCH_ENABLE": PTOPK_PREFETCH_ENABLE,
            "SPA_TOKENS_LEN": SPA_TOKENS_LEN,
            "DEFAULT_TOPK_LEN": DEFAULT_TOPK_LEN,
            "MAX_INDEXER_BLOCK_NUM": MAX_INDEXER_BLOCK_NUM,
            "IS_INDEXER_OFFLOAD": IS_INDEXER_OFFLOAD,
            "IS_OPEN_GSA": IS_OPEN_GSA,
            "init_indexer_layer_map": init_indexer_layer_map,
        }.items():
            if not hasattr(mod, _name):
                setattr(mod, _name, _val)
    except Exception as e:
        logger.warning(f"GSA sfa_v1 module-name injection failed: {e}")

    # Add has_prefill and gsa_* fields to AscendSFAMetadata.
    if hasattr(mod, "AscendSFAMetadata"):
        _cls = mod.AscendSFAMetadata
        _cls.__annotations__["has_prefill"] = bool
        for _fname, _fdefault in {
            "is_graph_dummy": False,
            "topk_indices_global": None,
            "dsa_block_table": None,
            "dsa_slot_mapping": None,
            "dsa_seq_len": None,
            "gsa_prefetch_index": None,
            "gsa_prefetch_topk": None,
            "gsa_swap_index": None,
            "gsa_swap_topk": None,
            "gsa_warmup_index": None,
            "gsa_warmup_topk": None,
            "gsa_prefetch_len": 0,
            "gsa_swap_len": 0,
            "gsa_warmup_len": 0,
            "gsa_slot_len": 0,
            "indexer_last_query": None,
            "indexer_last_q_index": None,
            "indexer_last_weight": None,
            "indexer_last_query_len": 0,
            "num_padding_reqs": 1,
            "gsa_prefill_query": None,
            "select_block_index": None,
            "gsa_topk_win_tmp": None,
        }.items():
            _cls.__annotations__.setdefault(_fname, object)
            if not hasattr(_cls, _fname):
                setattr(_cls, _fname, _fdefault)
        _orig_init = _cls.__init__

        def _patched_metadata_init(self, *args, **kwargs):
            _has_prefill = kwargs.pop("has_prefill", False)
            _orig_init(self, *args, **kwargs)
            self.has_prefill = _has_prefill

        patch_or_inject(_cls, "__init__", _patched_metadata_init)
        logger.info("UCM GSA AscendSFAMetadata has_prefill/gsa fields patch applied")

    if hasattr(mod.AscendSFAMetadataBuilder, "_build"):
        _patched = FunctionType(
            _build.__code__,
            vars(mod),
            name=_build.__name__,
            argdefs=_build.__defaults__,
            closure=_build.__closure__,
        )
        _patched.__annotations__ = _build.__annotations__
        _patched.__kwdefaults__ = _build.__kwdefaults__
        patch_or_inject(mod.AscendSFAMetadataBuilder, "_build", _patched)
        logger.info("UCM GSA AscendSFAMetadataBuilder._build patch applied")

    if hasattr(mod.AscendSFAImpl, "__init__"):
        _patched = FunctionType(
            __init__.__code__,
            vars(mod),
            name=__init__.__name__,
            argdefs=__init__.__defaults__,
            closure=__init__.__closure__,
        )
        _patched.__annotations__ = __init__.__annotations__
        _patched.__kwdefaults__ = __init__.__kwdefaults__
        patch_or_inject(mod.AscendSFAImpl, "__init__", _patched)
        logger.info("UCM GSA AscendSFAImpl.__init__ patch applied")

    if hasattr(mod.AscendSFAImpl, "indexer_select_post_process"):
        _patched = FunctionType(
            indexer_select_post_process.__code__,
            vars(mod),
            name=indexer_select_post_process.__name__,
            argdefs=indexer_select_post_process.__defaults__,
            closure=indexer_select_post_process.__closure__,
        )
        _patched.__annotations__ = indexer_select_post_process.__annotations__
        _patched.__kwdefaults__ = indexer_select_post_process.__kwdefaults__
        patch_or_inject(mod.AscendSFAImpl, "indexer_select_post_process", _patched)
        logger.info("UCM GSA AscendSFAImpl.indexer_select_post_process patch applied")

    if hasattr(mod.AscendSFAImpl, "_execute_sparse_flash_attention_process"):
        _patched = FunctionType(
            _execute_sparse_flash_attention_process.__code__,
            vars(mod),
            name=_execute_sparse_flash_attention_process.__name__,
            argdefs=_execute_sparse_flash_attention_process.__defaults__,
            closure=_execute_sparse_flash_attention_process.__closure__,
        )
        _patched.__annotations__ = (
            _execute_sparse_flash_attention_process.__annotations__
        )
        _patched.__kwdefaults__ = _execute_sparse_flash_attention_process.__kwdefaults__
        patch_or_inject(
            mod.AscendSFAImpl, "_execute_sparse_flash_attention_process", _patched
        )
        logger.info(
            "UCM GSA AscendSFAImpl._execute_sparse_flash_attention_process patch applied"
        )
