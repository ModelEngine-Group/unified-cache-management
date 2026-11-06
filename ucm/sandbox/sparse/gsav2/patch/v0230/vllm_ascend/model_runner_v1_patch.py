"""GSA NPUModelRunner patches for vLLM-Ascend 0.23.0."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def __init__(self, vllm_config: VllmConfig, device: torch.device):
    # TODO(qcs): These manual pad and unpad for GPUModelRunner are
    # used to expand some buffers, which need to be reverted after
    # the following PR is merged:
    # https://github.com/vllm-project/vllm/pull/28988
    max_pcp_pad_tokens = (
        vllm_config.parallel_config.prefill_context_parallel_size
        * 2
        * vllm_config.scheduler_config.max_num_seqs
    )
    vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens

    # Must be set before super().__init__() because parent init may call
    # _allocate_kv_cache_tensors which accesses self.use_compress.
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None) if model_config else None
    self.use_compress = hf_config is not None and hasattr(hf_config, "compress_ratios")

    with _torch_cuda_wrapper():
        super(NPUModelRunner, self).__init__(vllm_config, device)

    if not vllm_version_is("0.23.0"):
        self.pin_memory = PIN_MEMORY

    # Replace the CUDA PrefetchOffloader set by parent __init__ with NPU version.
    offload_cfg = vllm_config.offload_config
    if (
        offload_cfg is not None
        and getattr(offload_cfg, "prefetch", None) is not None
        and getattr(offload_cfg.prefetch, "offload_group_size", 0) > 0
    ):
        from vllm.model_executor.offloader.base import set_offloader
        from vllm_ascend.model_executor.offloader.prefetch import NPUPrefetchOffloader

        set_offloader(
            NPUPrefetchOffloader(
                group_size=offload_cfg.prefetch.offload_group_size,
                num_in_group=offload_cfg.prefetch.offload_num_in_group,
                prefetch_step=offload_cfg.prefetch.offload_prefetch_step,
                offload_params=offload_cfg.prefetch.offload_params,
            )
        )

    # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
    # See _pad_query_start_loc_for_fia.
    self.query_start_loc = self._make_buffer(
        self.max_num_reqs + 2,  # type: ignore[has-type]
        dtype=torch.int32,
    )

    # Now, query_start_loc is padded.
    # But gdn needs an unpadded one.
    # gdn_query_start_loc is an unpadded version of query_start_loc.
    # TODO delete it if fia's check is removed.
    self._has_gdn = check_gdn_layer(vllm_config)
    self._has_sinks = False
    if self._has_gdn:
        self.gdn_query_start_loc = self._make_buffer(
            self.max_num_reqs + 1,  # type: ignore[has-type]
            dtype=torch.int32,
        )

    vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens
    self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
    self.max_num_reqs = self.scheduler_config.max_num_seqs
    self.dp_size = vllm_config.parallel_config.data_parallel_size
    self.dp_rank = vllm_config.parallel_config.data_parallel_rank

    self.sampler = AscendSampler()
    self.attn_state: AscendAttentionState | None = None

    # Ascend-specific configurations
    self.ascend_config = get_ascend_config()
    set_weight_prefetch_method(self.ascend_config.weight_prefetch_config)

    # Dump / PrecisionDebugger configuration now comes from AscendConfig
    dump_cfg = self.ascend_config.dump_config_path
    self.debugger = None
    if dump_cfg is not None:
        self._debugger_started = False
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            from msprobe.pytorch import PrecisionDebugger

            self.debugger = PrecisionDebugger(dump_cfg)
        else:
            try:
                from msprobe.pytorch import AclGraphDumper
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import AclGraphDumper from msprobe. "
                    "Please install/rebuild msprobe with aclgraph_dump enabled."
                ) from exc

            self.debugger = AclGraphDumper(dump_cfg)
    # use_hybrid_blocks: if hybrid blocks is used.
    self.use_hybrid_blocks: bool = False
    self.need_accepted_tokens: bool = False

    self.is_multimodal_model = self.model_config.is_multimodal_model
    self.block_size = vllm_config.cache_config.block_size
    # Set up Attention
    self.use_sparse = (
        hasattr(vllm_config.model_config, "hf_text_config")
        and hasattr(vllm_config.model_config.hf_text_config, "index_topk")
        and not hasattr(vllm_config.model_config.hf_text_config, "compress_ratios")
    )
    if self.use_sparse:
        if (
            get_ascend_device_type() == AscendDeviceType.A5
            and self.ascend_config.enable_sparse_sfa_c8
        ):
            # A5 SFA C8 uses the merged/packed KV layout.
            packed_kv_head_dim = get_sfa_qsfa_packed_head_dim(
                self.model_config.hf_text_config.kv_lora_rank,
                self.model_config.hf_text_config.qk_rope_head_dim,
            )
            self.sparse_head_dim = (
                packed_kv_head_dim,
                0,
                self.model_config.hf_text_config.index_head_dim,
            )
        else:
            self.sparse_head_dim = (
                self.model_config.hf_text_config.kv_lora_rank,
                self.model_config.hf_text_config.qk_rope_head_dim,
                self.model_config.hf_text_config.index_head_dim,
            )
    # dsa c8
    self.enable_sparse_sfa_c8 = self.ascend_config.enable_sparse_sfa_c8
    self.enable_sparse_li_c8 = self.ascend_config.enable_sparse_li_c8
    if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
        if get_ascend_device_type() == AscendDeviceType.A5:
            self.c8_k_cache_dtype = torch.float8_e4m3fn
            self.c8_k_scale_cache_dtype = torch.float32
        else:
            self.c8_k_cache_dtype = torch.int8
            self.c8_k_scale_cache_dtype = torch.float16

    self.attn_backend = get_attn_backend(
        0,
        self.dtype,
        None,
        use_mla=self.model_config.use_mla,
        use_sparse=self.use_sparse,
        use_mm_prefix=self.model_config is not None
        and self.model_config.is_mm_prefix_lm,
    )

    # reinit valid_sampled_token_count_cpu with torch.int64 dtype
    if self.use_async_scheduling and self.num_spec_tokens:
        self.valid_sampled_token_count_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

    try:
        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
    except Exception:
        self.dcp_size = 1
        self.dcp_rank = 0
        self.pcp_size = 1
        self.pcp_rank = 0

    if self.pcp_size > 1:
        self.model_config.max_model_len += 2 * self.pcp_size * self.max_num_reqs
        if not self.vllm_config.cache_config.enable_prefix_caching:
            self.vllm_config.cache_config.mamba_block_size = (
                self.model_config.max_model_len
            )
    max_buffer_num_tokens = self.max_num_tokens
    if self.pcp_size * self.dcp_size > 1:
        max_buffer_num_tokens = (
            self.max_num_tokens + self.max_num_reqs * 2 * self.pcp_size
        )
        self.pcp_manager = PCPManager(
            self.pcp_size,
            self.pcp_rank,
            self.dcp_size,
            self.dcp_rank,
            max_buffer_num_tokens,
            self.max_num_reqs,
            self.device,
            self.vllm_config,
            self.use_async_scheduling,
            self.pin_memory,
            self.use_sparse,
        )
        # TODO(zhenwenqi) after https://github.com/vllm-project/vllm/pull/28988 is merged, we can delete this
        self.input_ids = self._make_buffer(max_buffer_num_tokens, dtype=torch.int32)
        self.positions = torch.zeros(
            max_buffer_num_tokens, dtype=torch.int64, device=self.device
        )

    self.sfa_dcp_replicated_indexer_size = 1
    if enable_sfa_dcp_replicated_indexer():
        self.sfa_dcp_replicated_indexer_size = self.dcp_size

    # Create a CPU numpy buffer for positions computation when
    # self.positions is a plain tensor (non-CpuGpuBuffer case).
    self._positions_cpu_buf = torch.zeros(
        max_buffer_num_tokens,
        dtype=torch.int64,
        pin_memory=self.pin_memory,
    )
    self._positions_np_buf = self._positions_cpu_buf.numpy()
    # For deepseek-v4 use only
    self._dsa_positions_cpu_buf = torch.zeros(
        max_buffer_num_tokens,
        dtype=torch.int64,
        pin_memory=self.pin_memory,
    )
    self._dsa_positions_np_buf = self._dsa_positions_cpu_buf.numpy()

    self.use_eagle = (
        vllm_config.speculative_config.use_eagle()
        if vllm_config.speculative_config
        else None
    )
    # When True, run update_full_graph_params before self.model (ENPU / graph capture order).
    # Internal / non-public toggle: read C getenv ``ENPU_ENABLE`` from enpu code (not in envs.py).
    _enpu = get_c_env("ENPU_ENABLE")
    self.enable_enpu = _enpu is not None and _enpu.lower() == "true"

    self._set_up_drafter()

    # Backends that consume CPU seq_lens (AscendAttentionBackend,
    # AscendMLABackend, and DSV4 compressed attention metadata) need
    # ``optimistic_seq_lens_cpu`` to match the corrected GPU seq_lens
    # in async spec decode mode; others (SFA, GDN, etc.) do not.
    self._needs_seq_lens_cpu_sync = self.use_compress or issubclass(
        self.attn_backend, (AscendAttentionBackend, AscendMLABackend)
    )

    # kv role
    self.is_kv_producer = False
    self.is_kv_consumer = False
    if vllm_config.kv_transfer_config is not None:
        self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.is_kv_consumer = vllm_config.kv_transfer_config.is_kv_consumer

    set_cos_and_sin(
        vllm_config,
        self.max_num_reqs,
        self.uniform_decode_query_len,
        self.dtype,
        self.device,
    )
    set_mc2_tokens_capacity(
        vllm_config, self.max_num_reqs, self.uniform_decode_query_len
    )
    set_mc2_mask(vllm_config, self.device)
    # Compute potential_max_tokens once here; it is reused by the skip-allreduce
    # decision and the o_proj static-exchange buffer sizing (see get_potential_max_tokens).
    set_potential_max_tokens(vllm_config)
    self.decode_threshold = 1 + (
        self.speculative_config.num_speculative_tokens if self.speculative_config else 0
    )

    self.use_aclgraph = self._use_aclgraph()

    eplb_config = self.ascend_config.eplb_config
    self.dynamic_eplb = eplb_config.dynamic_eplb
    self.eplb_enable = self.dynamic_eplb or (eplb_config.expert_map_path is not None)
    if self.dynamic_eplb:
        self.is_eplb_warmuped = False
        self.policy_type = eplb_config.eplb_policy_type
        self.eplb_loader = D2DExpertWeightLoader()
        self.manager = Manager()
        self.shared_dict = self.manager.dict(
            {"expert_map": None, "moe_load": None, "expert_maps": None}
        )
        self.eplb_process = EplbProcess(
            shared_dict=self.shared_dict,
            policy_type=self.policy_type,
            enable_d2d=True,
            tp_size=self.parallel_config.tensor_parallel_size,
        )
        self.process = self.eplb_process._launch_process()
        self.eplb_updator = EplbUpdator(
            eplb_config, self.eplb_loader, self.eplb_process, self.process
        )
        # In pd colocation scenarios, we find that prefill/decode requests result in different
        # expert workloads. To reduce expert imbalance more effectively, we can coolect eplb
        # heat exclusively on a single stage rather than both prefill/decode.
        self.eplb_heat_collection_stage = eplb_config.eplb_heat_collection_stage
        # Currently, we set the maximum of tokens in decode stage as the threshold to distinguish
        # prefill with decode.
        self.eplb_pd_thresholds = self.max_num_reqs * self.uniform_decode_query_len
        self.eplb_heat_collection_status = True

    # Input Batch
    # NOTE(Chen): Ideally, we should initialize the input batch inside
    # `initialize_kv_cache` based on the kv cache config. However, as in
    # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
    # reasons, we have to initialize the input batch before `load_model`,
    # quantization + weight offloading will fail otherwise. As a temporary
    # solution, we initialize the input batch here, and re-initialize it
    # in `initialize_kv_cache` if the block_sizes here is different from
    # the block_sizes in the kv cache config.
    self.input_batch = NPUInputBatch(
        max_num_reqs=self.max_num_reqs,
        max_model_len=max(self.model_config.max_model_len, self.max_encoder_len),
        max_num_batched_tokens=self.max_num_tokens,
        device=self.device,
        pin_memory=self.pin_memory,
        vocab_size=self.model_config.get_vocab_size(),
        block_sizes=[self.block_size],
        kernel_block_sizes=[[self.cache_config.block_size]],
        is_spec_decode=bool(self.vllm_config.speculative_config),
        logitsprocs=build_logitsprocs(
            self.vllm_config,
            self.device,
            self.pin_memory,
            self.is_pooling_model,
            self.vllm_config.model_config.logits_processors,
        ),
        logitsprocs_need_output_token_ids=bool(
            self.vllm_config.model_config.logits_processors
        ),
        is_pooling_model=self.is_pooling_model,
        num_speculative_tokens=(
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        ),
        cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
    )
    self.num_draft_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
    # here we use int32
    self.sampled_token_ids_pinned_cpu = torch.empty(
        (self.max_num_reqs, 1),
        dtype=torch.int32,
        device="cpu",
        pin_memory=self.pin_memory,
    )
    # for cleancode , actually the three attrs is defined in gpu_model_runner
    self.execute_model_state: ExecuteModelState | None = None
    # None in the first PP rank. The rest are set after load_model.
    self.intermediate_tensors: IntermediateTensors | None = None
    self.reorder_batch_threshold: int | None = None
    self.long_seq_metadata = None
    self.query_lens: torch.Tensor | None = None
    self.sampling_done_event: torch.npu.Event | None = None
    self.valid_sampled_token_count_gpu: torch.Tensor | None = None

    # self.cudagraph_batch_sizes sorts in ascending order.
    if (
        self.compilation_config.cudagraph_capture_sizes
        and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
    ):
        self.cudagraph_batch_sizes = sorted(
            self.compilation_config.cudagraph_capture_sizes
        )
    else:
        self.cudagraph_batch_sizes = []
    self.mamba_state_idx: dict[str, int] = {}
    self._mamba_bufs: Any | None = None
    self._mamba_copy_bufs: Any | None = None
    self.enable_hamming_sparse = self.ascend_config.enable_hamming_sparse is True
    self.enable_hamming_sparse = (
        self.enable_hamming_sparse and not vllm_config.speculative_config
    )
    if self.enable_hamming_sparse is True:
        from vllm_ascend.worker.kvcomp_utils import initialize_kvcomp_metadata

        self.kvcomp_meta_data = initialize_kvcomp_metadata(
            max_num_reqs=self.max_num_reqs,
            block_size=self.block_size,
            device=self.device,
            vllm_config=self.vllm_config,
            parallel_config=self.parallel_config,
            dtype=self.dtype,
        )


def _set_up_drafter(self):
    # Set up speculative decoding.
    self.drafter: (
        AscendNgramProposer
        | AscendNgramProposerNPU
        | AscendEagleProposer
        | AscendStep3p5MTPProposer
        | AscendDraftModelProposer
        | AscendDflashProposer
        | AscendSuffixDecodingProposer
        | AscendMedusaProposer
        | AscendExtractHiddenStatesProposer
        | None
    ) = None
    self.actual_seq_lengths_q: list[int] = []
    self.decode_token_per_req = 1
    if self.speculative_config:
        spec_token_num = self.speculative_config.num_speculative_tokens
        assert spec_token_num > 0
        self.decode_token_per_req = 1 + spec_token_num
        if get_pp_group().is_last_rank:
            self.drafter = self._get_drafter()
            if self.speculative_config.method == "eagle3":
                assert isinstance(self.drafter, AscendEagleProposer)
                self.use_aux_hidden_state_outputs = (
                    self.drafter.eagle3_use_aux_hidden_state
                )
            elif self.speculative_config.method == "extract_hidden_states":
                assert isinstance(self.drafter, AscendExtractHiddenStatesProposer)
                self.use_aux_hidden_state_outputs = True
            self.rejection_sampler = AscendRejectionSampler(self.sampler)
    self.discard_request_indices = self._make_buffer(
        self.max_num_reqs, dtype=torch.int64
    )
    self.num_discarded_requests = 0


def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
) -> tuple[
    torch.Tensor,
    SpecDecodeMetadata | None,
    int,
]:
    """
    :return: tuple[
        logits_indices,
        spec_decode_metadata,
        total_num_scheduled_tokens,
    ]
    """
    from ucm.sparse.base import INVALID_SLOT
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import PTOPK_PREFETCH_ENABLE

    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit_block_table(num_reqs)

    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    # Get the attention state.
    if not scheduler_output.scheduled_spec_decode_tokens:
        num_valid_tokens = num_scheduled_tokens
    else:
        num_valid_tokens = np.array(
            [
                scheduler_output.num_scheduled_tokens[i]
                - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                for i in self.input_batch.req_ids
            ],
            dtype=np.int32,
        )
    attn_state = self._build_attn_state(
        num_reqs, num_scheduled_tokens, num_valid_tokens
    )

    # Determine if it's a splitfuse batch
    with_prefill = attn_state not in [
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.SpecDecoding,
    ]
    self.with_prefill = with_prefill

    # Get positions.
    cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
    positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
    np.add(
        self.input_batch.num_computed_tokens_cpu[req_indices],
        self.query_pos.np[: cu_num_tokens[-1]],
        out=positions_np,
    )

    sparsed_positions = positions_np.copy()
    req_sparsed_slots = scheduler_output.req_sparsed_slots
    for req_id in self.input_batch.req_id_to_index:
        is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT
        req_index = self.input_batch.req_id_to_index[req_id]
        offset = (
            0 if req_index == 0 else cu_num_tokens[req_index - 1]
        )  # TODO: support MTP
        end = cu_num_tokens[req_index]
        if is_sparsed_request:
            # sparsed_positions[offset] = req_sparsed_slots[req_id] - 1
            sparsed_positions[offset:end] = list(
                range(
                    req_sparsed_slots[req_id] - end + offset, req_sparsed_slots[req_id]
                )
            )

    # For PCP, compute slot_mapping on GPU using pre-PCP-split positions.
    # Use blocking .to(device) to ensure data lands on GPU before PCP
    # modifies CPU position buffers. PCP and async spec decode are
    # mutually exclusive, so the sync is acceptable.
    if self.pcp_size > 1:
        pre_pcp_positions = torch.from_numpy(
            sparsed_positions[:total_num_scheduled_tokens]
        ).to(self.device)
        pre_pcp_qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=self.device)
        pre_pcp_qsl[1 : num_reqs + 1] = torch.from_numpy(cu_num_tokens).to(
            dtype=torch.int32, device=self.device
        )
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            pre_pcp_qsl,
            pre_pcp_positions,
        )

    if self.use_cp:
        self.pcp_manager.init_batch_info(
            num_scheduled_tokens,
            self.input_batch.num_reqs,
            self.input_batch.num_computed_tokens_cpu,
            self.input_batch.num_prompt_tokens,
        )

    # Build prev_positions before PCP prepares full-layout spec inputs so
    # PCP can repair async sampled/draft ids with device-side index math.
    prev_req_id_to_index = self.input_batch.prev_req_id_to_index
    self._compute_prev_positions(num_reqs)
    prev_positions_gpu = None
    if (
        self.use_async_scheduling
        and self.input_batch.prev_sampled_token_ids is not None
        and prev_req_id_to_index
    ):
        self.prev_positions.copy_to_gpu(num_reqs)
        prev_positions_gpu = self.prev_positions.gpu[:num_reqs]

    # for pcp, prefill mtp should use origin scheduleroutput ,
    if self.speculative_config and self.use_cp:
        self.pcp_manager.generate_pcp_mtp_input(
            total_num_scheduled_tokens,
            scheduler_output.num_scheduled_tokens,
            with_prefill,
            self.input_batch,
            self.arange_np,
            req_indices,
            sparsed_positions,
            cu_num_tokens,
            self._draft_token_ids,  # type: ignore[has-type]
            scheduler_output,
            self.num_spec_tokens,
            prev_positions=prev_positions_gpu,
        )

    if self.pcp_size > 1:
        num_scheduled_tokens[:num_reqs], position_pcp = (
            self.pcp_manager.update_tokens_for_pcp(
                num_scheduled_tokens[:num_reqs], self.arange_np
            )
        )
        # Re-update after PCP split sequences.
        total_num_scheduled_tokens = sum(num_scheduled_tokens[:num_reqs])
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        positions_np = self._positions_np_buf[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            position_pcp[:total_num_scheduled_tokens],
            out=positions_np,
        )
    if self.pcp_size > 1 and self.pcp_manager.pcp_use_hybrid_attn:
        assert self.pcp_manager.num_scheduled_tokens_padded is not None
        self.query_lens = torch.from_numpy(self.pcp_manager.num_scheduled_tokens_padded)
    else:
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    token_indices_tensor = torch.from_numpy(token_indices)
    # Prepare input_ids.
    # NOTE(woosuk): We use torch.index_select instead of np.take here
    # because torch.index_select is much faster than np.take for large
    # tensors.
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )
    if self.enable_prompt_embeds:
        is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
        torch.index_select(
            is_token_ids,
            0,
            token_indices_tensor,
            out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
        )

    # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
    # the InputBatch, we need to fill in the prompt embeds into the expected
    # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
    if self.input_batch.req_prompt_embeds and (
        self.is_multimodal_model or self.enable_prompt_embeds
    ):
        output_idx = 0
        for req_idx in range(num_reqs):
            num_sched = num_scheduled_tokens[req_idx]

            # Skip if this request doesn't have embeddings
            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += num_sched
                continue

            # Skip if no tokens scheduled
            if num_sched <= 0:
                output_idx += num_sched
                continue

            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            if self.pcp_size > 1:
                # PCP can split one request into non-contiguous token positions.
                # We must gather prompt embeds by actual scheduled positions.
                req_positions_np = positions_np[output_idx : output_idx + num_sched]
                dst_slice = self.inputs_embeds.cpu[output_idx : output_idx + num_sched]
                self.pcp_manager.fill_prompt_embeds_for_pcp(
                    req_embeds=req_embeds,
                    req_positions_np=req_positions_np,
                    dst_slice=dst_slice,
                )
            else:
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

            output_idx += num_sched

    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    copy_snapshot_to_gpu(self.query_start_loc)

    # Now, query_start_loc is padded.
    # But gdn needs an unpadded one.
    # gdn_query_start_loc is an unpadded version of query_start_loc.
    # TODO delete it if fia's check is removed.
    if self._has_gdn:
        self.gdn_query_start_loc.np[0] = 0
        self.gdn_query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.gdn_query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        copy_snapshot_to_gpu(self.gdn_query_start_loc)

    # Compute optimistic seq_lens (assumes all draft tokens from previous
    # iteration accepted). Store in optimistic_seq_lens_cpu for use by
    # _build_attention_metadata (max_seq_len) and discard_request_mask.
    # seq_lens (GPU) will be computed later using the same optimistic values.
    torch.add(
        self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
        torch.from_numpy(num_scheduled_tokens),
        out=self.optimistic_seq_lens_cpu[:num_reqs],
    )
    self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

    is_sparsed_np = np.zeros((num_reqs,), dtype=np.bool_)
    for req_id in self.input_batch.req_id_to_index:
        is_sparsed_request = scheduler_output.req_sparsed_slots[req_id] != INVALID_SLOT
        req_index = self.input_batch.req_id_to_index[req_id]
        if is_sparsed_request:
            self.optimistic_seq_lens_cpu[req_index] = req_sparsed_slots[req_id]
            is_sparsed_np[req_index] = True

    # Fill unused with -1. Needed for reshape_and_cache in attention_cp
    self.query_start_loc.gpu[num_reqs + 1 :].fill_(-1)

    # Copy the tensors to the NPU.
    self._prepare_input_ids(
        scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens
    )
    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self._calc_mrope_positions(scheduler_output)
        if self.pcp_size > 1:
            self.pcp_manager.remap_mrope_positions_for_pcp(
                positions_np,
                num_scheduled_tokens,
                num_reqs,
                self.input_batch,
                self.requests,
                self.mrope_positions,
            )
        self.mrope_positions.gpu.copy_(
            self.mrope_positions.cpu,
            non_blocking=True,
        )
    elif self.uses_xdrope_dim > 0:
        self._calc_xdrope_positions(scheduler_output)
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
            non_blocking=True,
        )

    # Record the index of requests that should not be sampled,
    # so that we could clear the sampled tokens before returning
    num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
    num_tokens_np = np.array(num_tokens, dtype=np.int32)
    base_num_reqs = self.input_batch.num_reqs
    num_reqs = base_num_reqs
    tokens_original = None
    if self.pcp_size > 1:
        # while pcp > 1, we need the original num_scheduled_tokens before split
        # to calculate discard_requests_mask
        tokens_original = [
            scheduler_output.num_scheduled_tokens[i] for i in self.input_batch.req_ids
        ]
        original_seq_lens_np = self.input_batch.num_computed_tokens_cpu[
            :num_reqs
        ] + np.array(tokens_original, dtype=np.int32)
        discard_requests_mask = original_seq_lens_np < num_tokens_np
    else:
        discard_requests_mask = (
            self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
        ) & (~is_sparsed_np)

    discard_request_indices = np.nonzero(discard_requests_mask)[0]
    self.num_discarded_requests = len(discard_request_indices)
    self.discard_request_indices.np[: self.num_discarded_requests] = (
        discard_request_indices
    )
    self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

    self.discard_request_mask.np[:num_reqs] = discard_requests_mask
    self.discard_request_mask.copy_to_gpu(num_reqs)

    # Sync num_accepted_tokens from CPU (set by
    # _update_states_after_model_execute for hybrid models).
    if self.num_accepted_tokens_event is not None:
        self.num_accepted_tokens_event.synchronize()
        self._sync_num_accepted_tokens(
            num_reqs, has_prev_mapping=bool(prev_req_id_to_index)
        )
        self.num_accepted_tokens.np[num_reqs:].fill(1)
        self.num_accepted_tokens.copy_to_gpu()
    else:
        self.num_accepted_tokens.np.fill(1)
        self.num_accepted_tokens.gpu.fill_(1)

    # Update num_computed_tokens on GPU. In async spec decode,
    # CPU values are optimistic (all drafts accepted). The kernel
    # corrects on GPU using the previous step's
    # valid_sampled_token_count_gpu. Otherwise, just copy from CPU.
    valid_sampled_token_count_gpu = self.valid_sampled_token_count_gpu
    if self.use_async_spec_decode:
        computed_token_tensor_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs
        ].to(device=self.device, non_blocking=True)
    if (
        self.use_async_spec_decode
        and valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    ):
        if prev_positions_gpu is None:
            self.prev_positions.copy_to_gpu(num_reqs)
        self.prev_num_draft_tokens.copy_to_gpu()
        update_num_computed_tokens_for_batch_change(
            self.num_computed_tokens,
            self.num_accepted_tokens.gpu[:num_reqs],
            self.prev_positions.gpu[:num_reqs],
            valid_sampled_token_count_gpu,
            self.prev_num_draft_tokens.gpu,
            computed_token_tensor_cpu,
        )
    else:
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )

    self.req_indices.np[:total_num_scheduled_tokens] = req_indices
    self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
    req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

    self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
    self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
    self.num_scheduled_tokens.copy_to_gpu(num_reqs)
    num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]

    pcp_manager = getattr(self, "pcp_manager", None)
    if pcp_manager is not None:
        cp_async_rebuild = pcp_manager.rebuild_async_spec_decode_inputs(
            use_async_spec_decode=self.use_async_spec_decode,
            valid_sampled_token_count_gpu=valid_sampled_token_count_gpu,
            prev_req_id_to_index=prev_req_id_to_index,
            prev_positions_gpu=prev_positions_gpu,
            with_prefill=with_prefill,
            enable_prompt_embeds=self.enable_prompt_embeds,
            has_req_prompt_embeds=bool(self.input_batch.req_prompt_embeds),
            supports_mm_inputs=self.supports_mm_inputs,
            num_reqs=num_reqs,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            req_indices=req_indices,
            req_indices_gpu=req_indices_gpu,
            position_pcp=position_pcp if self.pcp_size > 1 else None,
            query_pos_gpu=self.query_pos.gpu,
            query_pos_np=self.query_pos.np,
            positions=self.positions,
            positions_np=positions_np,
            num_computed_tokens=self.num_computed_tokens,
            num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu,
            prev_positions_np=self.prev_positions.np,
            prev_num_draft_tokens_np=self.prev_num_draft_tokens.np,
            valid_sampled_token_count_event=self.valid_sampled_token_count_event,
            valid_sampled_token_count_cpu=self.valid_sampled_token_count_cpu,
            input_batch=self.input_batch,
            input_ids=self.input_ids,
            scheduler_output=scheduler_output,
            arange_np=self.arange_np,
            cu_num_tokens=cu_num_tokens,
            draft_token_ids=self._draft_token_ids,  # type: ignore[has-type]
            num_spec_tokens=self.num_spec_tokens,
            prepare_input_ids=self._prepare_input_ids,
        )
    else:
        cp_async_rebuild = PCPAsyncSpecDecodeRebuildResult(
            rebuilt=False,
            positions_ready_on_device=False,
        )

    if cp_async_rebuild.positions_ready_on_device:
        pass
    elif self.pcp_size > 1 or cp_async_rebuild.rebuilt:
        # PCP and async rebuild both compute the correct positions on CPU.
        # Copy positions_np to GPU so input_ids and positions stay aligned.

        self.positions[:total_num_scheduled_tokens].copy_(
            torch.from_numpy(positions_np[:total_num_scheduled_tokens]).to(self.device),
            non_blocking=True,
        )
    else:
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )

    if has_ucm_sparse() and PTOPK_PREFETCH_ENABLE:
        self.sparse_positions[:total_num_scheduled_tokens].copy_(
            torch.from_numpy(sparsed_positions[:total_num_scheduled_tokens]),
            non_blocking=True,
        )
        self.seq_lens[:num_reqs].copy_(
            self.optimistic_seq_lens_cpu[:num_reqs], non_blocking=True
        )
    else:
        self.sparse_positions = self.positions
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
    self.seq_lens[num_reqs:].fill_(0)

    # In async spec decode mode, optimistic_seq_lens_cpu assumes all
    # tokens from the previous speculative step were accepted. Correct it
    # on CPU using the valid-sampled-token counts that are already copied
    # asynchronously for scheduler bookkeeping. This avoids an extra
    # NPU->CPU seq_lens copy and the synchronize() in attention metadata.
    # Mirrors update_num_computed_tokens_for_batch_change on the GPU side.
    async_spec_decode_active = (
        self.use_async_spec_decode
        and valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    )
    if self._needs_seq_lens_cpu_sync and async_spec_decode_active:
        self._correct_optimistic_seq_lens_cpu(num_reqs)

    # For non-PCP, compute slot_mapping on GPU. PCP slot_mapping was
    # already computed on GPU before PCP split the positions.
    if self.pcp_size <= 1:
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.sparse_positions[:total_num_scheduled_tokens],
        )

    if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
        drift = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            - computed_token_tensor_cpu[req_indices_gpu]
        )
        target = self.mrope_positions if self.uses_mrope else self.xdrope_positions
        target.gpu[:, :total_num_scheduled_tokens] += drift

    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        spec_decode_metadata = None
        num_draft_tokens = None
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        if self.use_cp:
            logits_indices = self.pcp_manager.get_logits_indices(
                cu_num_tokens, num_reqs, tokens_original
            )
            logits_indices = logits_indices.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        # For chunked prefills, use -1 as mask rather than 0, as guided
        # decoding may rollback speculative tokens.
        new_schedule_reqs = [x.req_id for x in scheduler_output.scheduled_new_reqs]
        num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
        for (
            req_id,
            draft_token_ids,
        ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            draft_len = len(draft_token_ids)
            num_draft_tokens[req_idx] = draft_len
            if (self.is_kv_consumer and req_id in new_schedule_reqs) or (
                self.input_batch.num_computed_tokens_cpu[req_idx]
                >= self.input_batch.num_prompt_tokens[req_idx]
            ):
                num_decode_draft_tokens[req_idx] = draft_len
            else:
                num_decode_draft_tokens[req_idx] = -1

        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens,
            cu_num_tokens,
            num_pcp_pads=(
                self.pcp_manager.num_pcp_pads_cpu[:num_reqs]
                if self.pcp_size > 1
                else None
            ),
        )
        logits_indices = spec_decode_metadata.logits_indices
        num_sampled_tokens = num_draft_tokens + 1

        # For DECODE only cuda graph of some attention backends (e.g., GDN).
        self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_decode_draft_tokens.copy_to_gpu()
    # save logits_indices for pcp spec decode usage
    self.logits_indices = logits_indices

    # Hot-Swap lora model
    if self.lora_config:
        assert (
            np.sum(num_sampled_tokens)
            <= self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.set_active_loras(
            self.input_batch, num_scheduled_tokens, num_sampled_tokens
        )
    if lmhead_tp_enable():
        max_num_reqs_across_dp = self.max_num_reqs * self.uniform_decode_query_len
        logits_indices = nn.functional.pad(
            logits_indices, (0, max_num_reqs_across_dp - logits_indices.shape[0])
        )

    # Cache local scheduled token layout for PCP-aware multimodal preprocess.
    if (
        self.pcp_size > 1
        and self.supports_mm_inputs
        and get_pp_group().is_first_rank
        and not self.model_config.is_encoder_decoder
    ):
        self.pcp_manager.cache_local_schedule_layout(
            num_scheduled_tokens=num_scheduled_tokens,
            num_reqs=base_num_reqs,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
        )

    return (
        logits_indices,
        spec_decode_metadata,
        total_num_scheduled_tokens,
    )


def _build_attn_state(self, num_reqs, num_scheduled_tokens, num_valid_tokens):
    if np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] == 0):
        attn_state = AscendAttentionState.PrefillNoCache
    # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
    elif np.all(num_scheduled_tokens == 1):
        attn_state = AscendAttentionState.DecodeOnly
        if self.speculative_config and self.speculative_config.method == "mtp":
            # SpecDecoding now supports seq_len=1 and seq_len=2
            # In Prefilling Decoding Disaggregation scenario, SpecDecoding need to supports seq_len=1
            attn_state = AscendAttentionState.SpecDecoding
    # Speculative decoding.
    elif np.all(num_valid_tokens == 1):
        if self.speculative_config:
            attn_state = AscendAttentionState.SpecDecoding
        else:
            attn_state = AscendAttentionState.ChunkedPrefill
    # splitfuse
    elif self.scheduler_config.enable_chunked_prefill:
        attn_state = AscendAttentionState.ChunkedPrefill
    else:
        attn_state = AscendAttentionState.PrefillCacheHit

    # For the overlay of the PCP feature and the eagle3, attn_state needs to be recovered
    # TODO: Resolved the conflict between the sunset of attn_state and the PCP that requires this interface.
    if (
        attn_state == AscendAttentionState.SpecDecoding
        and self.speculative_config.method != "mtp"
    ):
        self.attn_state = AscendAttentionState.ChunkedPrefill  # type: ignore
    else:
        self.attn_state = attn_state  # type: ignore

    return attn_state


def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | IntermediateTensors | None:

    if self.vllm_config.model_config.enable_return_routed_experts:
        if self.routed_experts_initialized:
            self.routed_experts_capturer.clear_buffer()

    if self.ascend_config.profiling_chunk_config.need_timing:
        # Check if the scheduler signaled that calibration is complete.
        # This flag is set cross-process via scheduler_output because
        # modifying the config singleton in the scheduler process does
        # not affect this worker process.
        if getattr(scheduler_output, "disable_profiling_timing", False):
            self.ascend_config.profiling_chunk_config.need_timing = False
        else:
            self._sync_device()
            self._execution_start_time = time.perf_counter()
    if self.execute_model_state is not None:
        raise RuntimeError(
            "State error: sample_tokens() must be called after execute_model() returns None."
        )

    # If ngram_gpu is used, we need to copy the scheduler_output to avoid
    # the modification has influence on the scheduler_output in engine core process.
    # The replace is much faster than deepcopy.
    if self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
        num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
        spec_decode_tokens_copy = scheduler_output.scheduled_spec_decode_tokens.copy()
        scheduler_output = replace(
            scheduler_output,
            num_scheduled_tokens=num_scheduled_tokens_copy,
            scheduled_spec_decode_tokens=spec_decode_tokens_copy,
        )

    self._start_dump_data()
    # self._draft_token_ids is None when `input_fits_in_drafter=False`
    # and there is no draft tokens scheduled. so it need to update the
    # spec_decoding info in scheduler_output with async_scheduling.
    # use deepcopy to avoid the modification has influence on the
    # scheduler_output in engine core process.
    # TODO(Ronald1995): deepcopy is expensive when there is a large
    # number of requests, optimize it later.
    if (
        self.use_async_scheduling
        and self.num_spec_tokens
        and self._draft_token_ids is None  # type: ignore[has-type]
    ) or (
        # NOTE: This branch specifically triggers a deepcopy during the prefill phase
        # only for PCP (Parallel Context Processing) + Multi-Modal (MM) scenarios.
        # It does not affect other use cases. This is a temporary workaround and
        # will be removed once upstream vLLM provides native support for PCP + MM.
        self.pcp_size > 1
        and self.supports_mm_inputs
        and get_pp_group().is_first_rank
        and not self.model_config.is_encoder_decoder
    ):
        scheduler_output = deepcopy(scheduler_output)
    pp_group = get_pp_group()
    if pp_group.world_size > 1 and not pp_group.is_last_rank:
        new_token_ids = scheduler_output.scheduled_cached_reqs.new_token_ids
        if new_token_ids and all(not token_ids for token_ids in new_token_ids):
            scheduler_output = deepcopy(scheduler_output)
            scheduler_output.scheduled_cached_reqs.new_token_ids = []

    if has_kv_transfer_group():
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        # Preemption stores must run before _update_states() zeroes newly
        # allocated blocks that may reuse the same physical KV cache IDs.
        get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

    with record_function_or_nullcontext("prepare input"):
        with self.synchronize_input_prep():
            # Fix up prev_req_id_to_index for requests that were discarded
            # in the previous sample_tokens step. If a request has
            # prev_num_draft_len > 0 but is missing from
            # prev_req_id_to_index, the parent _update_states would
            # hit a KeyError. Reset prev_num_draft_len to 0 for such
            # requests so they fall through safely.
            if (
                self.use_async_scheduling
                and self.num_spec_tokens
                and self.input_batch.prev_req_id_to_index is not None
            ):
                for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                    if (
                        req_id not in self.input_batch.prev_req_id_to_index
                        and (req_state := self.requests.get(req_id)) is not None
                        and req_state.prev_num_draft_len
                    ):
                        req_state.prev_num_draft_len = 0

            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(scheduler_output)

            if has_ec_transfer() and get_ec_transfer().is_producer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)
                    self._finalize_dump_data()
                    return make_empty_encoder_model_runner_output(scheduler_output)

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend
                    == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    # this is a corner case when both external launcher
                    # and DP are enabled, num_scheduled_tokens could be
                    # 0, and has_unfinished_requests in the outer loop
                    # returns True. before returning early here we call
                    # dummy run to ensure coordinate_batch_across_dp
                    # is called into to avoid out of sync issues.
                    self._dummy_run(1)
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            if (
                scheduler_output.total_num_scheduled_tokens <= 0
                or not tokens
                or sum(tokens) == 0
            ):
                if not has_kv_transfer_group():
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

            (
                logits_indices,
                spec_decode_metadata,
                total_num_scheduled_tokens,
            ) = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
            if self.pcp_size > 1:
                num_tokens_unpadded = self.pcp_manager.total_num_sampled_tokens_pcp
            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                force_eager=self.model_config.enforce_eager,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = (
                batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            )
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            if self.dynamic_eplb:
                self.update_eplb_heat_collection_status(num_tokens_padded)

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            # postprocess_mamba runs later in
            # _update_states_after_model_execute.
            if self.cache_config.mamba_cache_mode == "align":
                # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                # to decide copy operations, so we must apply deferred
                # corrections before it runs.
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                mamba_bufs = self._get_mamba_bufs()
                preprocess_bufs = mamba_bufs.preprocess
                mamba_utils.preprocess_mamba(
                    scheduler_output,
                    self.kv_cache_config,
                    self.cache_config,
                    self.mamba_state_idx,
                    self.input_batch,
                    self.requests,
                    self.compilation_config.static_forward_context,
                    self.model.get_mamba_state_copy_func(),
                    preprocess_bufs,
                )
                # preprocess_mamba resets num_accepted_tokens_cpu to 1
                # for requests whose state was copied to a new block.
                # Re-sync to GPU so the mamba kernel reads from the
                # correct initial state slot (init_token_idx = 0).
                self.num_accepted_tokens.np[:num_reqs] = (
                    self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                )
                self.num_accepted_tokens.copy_to_gpu(num_reqs)

                if mamba_bufs.postprocess_align is not None:
                    mamba_utils.stage_postprocess_inputs_to_gpu(
                        mamba_bufs.postprocess_align,
                        scheduler_output,
                        self.input_batch.req_ids,
                        num_reqs,
                        self.requests,
                        self.mamba_state_idx,
                    )
            if self.use_compress:
                if deferred_state_corrections_fn:
                    deferred_state_corrections_fn()
                    deferred_state_corrections_fn = None
                num_reqs = self.input_batch.num_reqs
                req_indices = np.repeat(
                    self.arange_np[:num_reqs], num_scheduled_tokens_np
                )
                dsa_positions_np = self._dsa_positions_np_buf[
                    :total_num_scheduled_tokens
                ]
                np.add(
                    self.input_batch.num_computed_tokens_cpu[req_indices],
                    self.query_pos.np[:total_num_scheduled_tokens],
                    out=dsa_positions_np,
                )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            if (
                cudagraph_mode == CUDAGraphMode.FULL
                or (enable_sp() and not self.model_config.use_mla)
                and self.pcp_size * self.dcp_size == 1
            ):
                # Currently, Graph Mode and SP will both pad num_tokens,
                # Another possible condition is num_tokens_padded != num_tokens_unpadded
                # but this scope is way too big and the consequences are unpredictable
                num_reqs_padded = self._pad_query_start_loc_for_fia(
                    self.query_start_loc,
                    num_tokens_padded,
                    num_reqs_padded,
                    num_reqs,
                    cudagraph_mode,
                    batch_desc.num_reqs,
                )

            (attn_metadata, spec_decode_common_attn_metadata) = (
                self._build_attention_metadata(
                    num_tokens=(
                        num_tokens_unpadded
                        if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                        else total_num_scheduled_tokens
                    ),
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                )
            )

            self._sanitize_placeholder_input_ids_for_forward(
                scheduler_output,
                (
                    num_tokens_padded
                    if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                    else total_num_scheduled_tokens
                ),
            )

        (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        ) = self._preprocess(
            scheduler_output,
            num_tokens_padded,
            intermediate_tensors,
        )

        # update global cos, sin
        update_cos_sin(positions)

    if self.dynamic_eplb:
        self.eplb_updator.forward_before()

    # Set cudagraph mode to none if calc_kv_scales is true.
    # KV scales calculation involves dynamic operations that are incompatible
    # with CUDA graph capture.
    if self.calculate_kv_scales:  # type: ignore[has-type]
        cudagraph_mode = CUDAGraphMode.NONE
        # Mark KV scales as calculated after the first forward pass
        self.calculate_kv_scales = False  # type: ignore[has-type]
    if self.ascend_config.enable_async_exponential:
        self.sampler.do_async_exponential(
            b_s=logits_indices.shape[0],
            head_dim=self.model_config.get_vocab_size(),
            generators=self.input_batch.sampling_metadata.generators,
        )

    # Encoder-decoder models can only compile the pure decode steps where no
    # encoder inputs are present. Use eager for the first pass.
    num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
    has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

    # Run forward pass
    defer_kv_connector_finalize = self.speculative_config is not None and (
        get_pp_group().is_last_rank or self.broadcast_pp_output
    )
    with (
        record_function_or_nullcontext("forward"),
        set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            aclgraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_desc,
            num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
            model_instance=self.model,
            max_tokens_across_pcp=(
                0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp
            ),
            skip_compiled=has_encoder_input,
            has_sinks=self._has_sinks,
            input_ids=input_ids,
            eplb_heat_collection_status=(
                self.eplb_heat_collection_status if self.dynamic_eplb else False
            ),
        ),
        self.maybe_get_kv_connector_output(
            scheduler_output,
            **({"defer_finalize": defer_kv_connector_finalize}),
        ) as kv_connector_output,
    ):
        if self.cache_config.mamba_cache_mode == "align":
            mamba_utils.do_mamba_copy_block(preprocess_bufs)

        self.maybe_execute_ucm_sparse_begin(scheduler_output, attn_metadata)

        hidden_states = self._model_forward(
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

        self.maybe_execute_ucm_sparse_warmup()
        if self.vllm_config.speculative_config is None:
            self.maybe_execute_ucm_sparse_finished()

    with record_function_or_nullcontext("post process"):
        aux_hidden_states = None
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = hidden_states
        if self.pcp_size > 1:
            # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
            # ignores the padding from CUDA Graph.
            hidden_states = self.pcp_manager.get_restore_hidden_states(hidden_states)
            if aux_hidden_states is not None:
                aux_hidden_states = [
                    self.pcp_manager.get_restore_hidden_states(aux_hidden_states_pcp)
                    for aux_hidden_states_pcp in aux_hidden_states
                ]

        if not self.broadcast_pp_output:
            # Common case.
            if not get_pp_group().is_last_rank:
                # Return the intermediate tensors.
                assert isinstance(hidden_states, IntermediateTensors)
                hidden_states.kv_connector_output = kv_connector_output
                self.kv_connector_output = kv_connector_output
                self._finalize_dump_data()
                if self.dynamic_eplb:
                    self.eplb_updator.forward_end(self.eplb_heat_collection_status)
                return hidden_states
            if self.is_pooling_model:
                # Return the pooling output.
                output = self._pool(
                    hidden_states,
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                    kv_connector_output,
                )
                output.kv_connector_output = kv_connector_output
                self._finalize_dump_data()
                return output

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        else:
            # Rare case.
            assert not self.is_pooling_model

            if not get_pp_group().is_last_rank:
                sample_hidden_states = hidden_states[logits_indices]
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors, all_gather_group=get_tp_group()
                )
                logits = None
            else:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)

            model_output_broadcast_data: dict[str, Any] = {}
            if logits is not None:
                model_output_broadcast_data["logits"] = logits.contiguous()
            broadcasted = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
            )
            assert broadcasted is not None
            logits = broadcasted["logits"]

        # Apply structured output bitmasks if present
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
        )
        self.kv_connector_output = kv_connector_output

    # Now the batch has been launched we can wait for corrections from the
    # previous model forward without breaking async scheduling.
    if deferred_state_corrections_fn:
        deferred_state_corrections_fn()

    return None


def sample_tokens(
    self, grammar_output: "GrammarOutput | None"
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
    from ucm.sparse.state import has_ucm_sparse

    kv_connector_output = self.kv_connector_output
    self.kv_connector_output = None
    pp = get_pp_group()
    skip_pp_pd_broadcast = self.is_kv_producer and pp.world_size > 1

    if self.execute_model_state is None:
        # Nothing to do (PP non-final rank case), output isn't used.
        # receive sampled token ids from the last PP rank when using
        # async scheduling + pipeline parallelism so downstream code
        # (e.g., PCP input preparation) can access them.
        if self.use_async_scheduling and pp.world_size > 1 and not skip_pp_pd_broadcast:
            self._pp_receive_prev_sampled_token_ids_to_input_batch()
        if not kv_connector_output:
            return None  # noqa
        # In case of PP with kv transfer, we need to pass through the
        # kv_connector_output
        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    # Unpack ephemeral state.
    (
        scheduler_output,
        logits,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        attn_metadata,
        positions,
        ec_connector_output,
        cudagraph_stats,
        batch_desc,
    ) = self.execute_model_state
    # Clear ephemeral state.
    self.execute_model_state = None

    # Apply structured output bitmasks if present.
    if grammar_output is not None:
        # here we are different from gpu_model_runner,
        # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
        logits_dtype = logits.dtype
        logits = logits.to("cpu").float()
        apply_grammar_bitmask(
            scheduler_output, grammar_output, self.input_batch, logits
        )
        logits = logits.to(self.device).to(logits_dtype)

    with record_function_or_nullcontext("sample_token"):
        sampler_output = self._sample(logits, spec_decode_metadata)

    if self.need_accepted_tokens:
        if self.sampling_done_event is None:
            self.sampling_done_event = torch.npu.Event()

        assert self.sampling_done_event is not None
        self.sampling_done_event.record()

    self.valid_sampled_token_count_gpu = None

    def propose_draft_token_ids(sampled_token_ids):
        assert spec_decode_common_attn_metadata is not None
        self._draft_token_ids = self.propose_draft_token_ids(
            sampled_token_ids,
            self.input_batch.sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            positions,
            scheduler_output.total_num_scheduled_tokens,
            hidden_states,
            aux_hidden_states,
            sample_hidden_states,
            batch_desc,
        )
        self._copy_draft_token_ids_to_cpu(scheduler_output)

    (
        logprobs_lists,
        valid_sampled_token_ids,
        prompt_logprobs_dict,
        req_ids_output_copy,
        req_id_to_index_output_copy,
        invalid_req_indices,
    ) = self._bookkeeping_sync(
        scheduler_output,
        sampler_output,
        logits,
        hidden_states,
        scheduler_output.total_num_scheduled_tokens,
        spec_decode_metadata,
    )

    with record_function_or_nullcontext("draft_token"):
        if self.speculative_config:
            use_padded_batch = (
                self.speculative_config
                and (
                    self.speculative_config.use_eagle()
                    or self.speculative_config.uses_draft_model()
                    or self.speculative_config.uses_extract_hidden_states()
                    or self.speculative_config.use_ngram_gpu()
                )
                and not self.speculative_config.disable_padded_drafter_batch
            )
            if use_padded_batch:
                # EAGLE speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                propose_draft_token_ids(sampler_output.sampled_token_ids)
            if self.speculative_config and not use_padded_batch:
                # ngram and other speculative decoding methods use the sampled
                # tokens on the CPU, so they are run after bookkeeping.
                propose_draft_token_ids(valid_sampled_token_ids)

        # vLLM v0.18 defers KV connector finalization during target-model
        # forward when speculative decoding is enabled. Finalize here after
        # draft model runs so KV pool save/put can complete.
        if self.speculative_config is not None:
            self.finalize_kv_connector()

    if has_ucm_sparse() and self.vllm_config.speculative_config:
        if spec_decode_metadata is None:
            self.maybe_execute_ucm_sparse_finished()
        else:
            valid_counts = (sampler_output.sampled_token_ids != -1).sum(dim=1)
            if has_ucm_sparse() and self.vllm_config.speculative_config:
                self.maybe_execute_ucm_sparse_finished(
                    torch.where(
                        valid_counts > 0, valid_counts - 1, torch.tensor(-1)
                    ).tolist()
                )

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        kv_connector_output=kv_connector_output,
        pooler_output=[],
        ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
        cudagraph_stats=cudagraph_stats,
        routed_experts=None,
    )
    if self.ascend_config.profiling_chunk_config.need_timing and hasattr(
        self, "_execution_start_time"
    ):
        self._sync_device()
        model_runner_output.execution_time_ms = (
            time.perf_counter() - self._execution_start_time
        ) * 1000.0

    if self.dynamic_eplb:
        self.eplb_updator.forward_end(self.eplb_heat_collection_status)

    self._finalize_dump_data()

    if self.need_accepted_tokens:
        assert self.sampling_done_event is not None
        with (
            record_function_or_nullcontext("async_state_update"),
            torch.npu.stream(global_stream()),
        ):
            global_stream().wait_event(self.sampling_done_event)
            self._update_states_after_model_execute(
                sampler_output.sampled_token_ids, scheduler_output
            )

    # In async scheduling + PP, broadcast sampled token ids from the
    # last PP rank so other PP ranks can receive them without going
    # through the scheduler/engine IPC path.
    if self.use_async_scheduling:
        if pp.world_size > 1 and pp.is_last_rank and not skip_pp_pd_broadcast:
            self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

    if not self.use_async_scheduling:
        if self.routed_experts_initialized:
            # Sync path: D2H was issued in ``_bookkeeping_sync`` and
            # synchronized by ``_to_list``'s event.synchronize(), so
            # the pinned buffers are ready to be wrapped as numpy.
            total = scheduler_output.total_num_scheduled_tokens
            model_runner_output.routed_experts = RoutedExpertsLists(
                routing_data=self.routed_experts_cpu[:total].numpy(),
                slot_mapping=self.routed_experts_slot_mapping_cpu[:total].numpy(),
            )
        return model_runner_output

    # Async path: produce a device-side snapshot that the async
    # copy stream can D2H later. Both tensors must be private
    # clones because:
    #   - ``routing_data`` source is the shared capturer buffer,
    #     which is ``clear_buffer()``-ed at the start of the
    #     next step on the default stream.
    #   - ``slot_mapping`` source is our own
    #     ``routed_experts_slot_mapping_device``, which the
    #     next ``_prepare_inputs`` overwrites on the default
    #     stream while the D2H is still pending on the copy
    #     stream.
    # Without clones, the copy stream would read torn data.
    routed_experts_snapshot = None
    if self.routed_experts_initialized:
        buf = self.routed_experts_capturer.get_device_buffer()
        total = scheduler_output.total_num_scheduled_tokens
        routed_experts_snapshot = RoutedExpertsTensors(
            routing_data=buf[:total].clone(),
            slot_mapping=self.routed_experts_slot_mapping_device[:total].clone(),
        )
    async_output = AsyncGPUModelRunnerOutput(
        model_runner_output=model_runner_output,
        sampled_token_ids=sampler_output.sampled_token_ids,
        logprobs_tensors=sampler_output.logprobs_tensors,
        invalid_req_indices=invalid_req_indices,
        async_output_copy_stream=self.async_output_copy_stream,
        vocab_size=self.input_batch.vocab_size,
        routed_experts=routed_experts_snapshot,
    )
    self.input_batch.set_async_sampled_token_ids(
        async_output.sampled_token_ids_cpu,
        async_output.async_copy_ready_event,
    )
    return async_output


def _build_attention_metadata(
    self,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    num_tokens_padded: int | None = None,
    num_reqs_padded: int | None = None,
    ubatch_slices: UBatchSlices | None = None,
    logits_indices: torch.Tensor | None = None,
    use_spec_decode: bool = False,
    for_cudagraph_capture: bool = False,
    num_scheduled_tokens: dict[str, int] | None = None,
    num_scheduled_tokens_np: np.ndarray | None = None,
    cascade_attn_prefix_lens: list[list[int]] | None = None,
) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
    """
    :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
    """
    from ucm.sparse.state import has_ucm_sparse
    from ucm.sparse.utils import IS_INDEXER_OFFLOAD, PTOPK_PREFETCH_ENABLE

    # Attention metadata is not needed for attention free models
    if len(self.kv_cache_config.kv_cache_groups) == 0:
        return {}, None
    num_tokens_padded = num_tokens_padded or num_tokens
    num_reqs_padded = num_reqs_padded or num_reqs
    attn_metadata: PerLayerAttnMetadata = {}
    if ubatch_slices is not None:
        attn_metadata = [dict() for _ in range(len(ubatch_slices))]

    if for_cudagraph_capture:
        # For some attention backends (e.g. FA) with sliding window models we need
        # to make sure the backend see a max_seq_len that is larger to the sliding
        # window size when capturing to make sure the correct kernel is selected.
        max_seq_len = self.max_model_len
    else:
        max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

    kv_cache_groups = self.kv_cache_config.kv_cache_groups

    def _get_pcp_metadata(block_table_tensor):
        if not self.use_cp:
            return None, block_table_tensor

        fixed_decode_seq_lens_cpu = None
        if self.use_async_spec_decode:
            fixed_decode_seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs].numpy()

        assert num_reqs_padded is not None
        return self.pcp_manager.generate_pcp_metadata(
            num_tokens,
            self.query_lens,
            self.input_batch,
            num_scheduled_tokens_np,
            block_table_tensor,
            num_reqs_padded,
            num_reqs,
            fixed_decode_seq_lens_cpu,
        )

    def _get_block_table_and_slot_mapping(
        kv_cache_gid: int,
    ):
        assert num_reqs_padded is not None and num_tokens_padded is not None
        kv_cache_spec = kv_cache_groups[kv_cache_gid].kv_cache_spec
        if self.pcp_size > 1:
            total_num_pcp_pads = sum(self.pcp_manager.num_pcp_pads_cpu[:num_reqs])
            if self.pcp_manager.pcp_use_hybrid_attn:
                num_scheduled_tokens_padded = (
                    self.pcp_manager.num_scheduled_tokens_padded
                )
                assert num_scheduled_tokens_padded is not None
                maybe_pcp_full_tokens = (
                    sum(num_scheduled_tokens_padded) * self.pcp_size
                    - total_num_pcp_pads
                )
            else:
                maybe_pcp_full_tokens = num_tokens * self.pcp_size - total_num_pcp_pads
        else:
            maybe_pcp_full_tokens = num_tokens_padded
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            blk_table_tensor = torch.zeros(
                (num_reqs_padded, 1),
                dtype=torch.int32,
                device=self.device,
            )
            slot_mapping = torch.zeros(
                (num_tokens_padded,),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            blk_table = self.input_batch.block_table[kv_cache_gid]
            slot_mapping = blk_table.slot_mapping.gpu[:maybe_pcp_full_tokens]
            blk_table_tensor = blk_table.get_device_tensor()[:num_reqs_padded]
            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
            if self.pcp_size == 1:
                slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
                blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
        if self.pcp_size > 1:
            slot_mapping = self.pcp_manager.get_padded_slot_mapping(
                num_tokens,
                num_tokens_padded,
                slot_mapping,
                kv_cache_gid,
            )
        if self.model_config.enable_return_routed_experts and kv_cache_gid == 0:
            if self.routed_experts_initialized:
                # snapshot slot_mapping into a private device
                # buffer so the next ``_prepare_inputs`` does not
                # overwrite it while D2H is still pending.
                n = slot_mapping.shape[0]
                self.routed_experts_slot_mapping_device[:n].copy_(slot_mapping)
        return blk_table_tensor, slot_mapping

    block_table_gid_0, slot_mapping_gid_0 = _get_block_table_and_slot_mapping(0)
    self.long_seq_metadata, block_table_gid_0 = _get_pcp_metadata(block_table_gid_0)
    num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[
        :num_reqs_padded
    ]
    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
    is_prefilling[num_reqs:] = False
    seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]
    if self.use_async_spec_decode:
        # GPU tensors are authoritative in async mode.
        seq_lens_cpu = None
        num_computed_tokens_cpu = None

    cm_base = AscendCommonAttentionMetadata(
        query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
        query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
        seq_lens=self.seq_lens[:num_reqs_padded],
        # Always pass optimistic_seq_lens_cpu via _seq_lens_cpu so NPU
        # attention backends can get CPU seq_lens without GPU->CPU sync.
        # This is separate from seq_lens_cpu (None in async) which eagle
        # proposer checks to distinguish async/non-async behavior.
        _seq_lens_cpu=self.optimistic_seq_lens_cpu[:num_reqs_padded],
        seq_lens_cpu_upper_bound=self.optimistic_seq_lens_cpu[:num_reqs_padded],
        # TODO
        seq_lens_cpu=seq_lens_cpu,
        # TODO
        # num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs_padded],
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=num_reqs_padded,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_gid_0,
        slot_mapping=slot_mapping_gid_0,
        causal=True,
        is_prefilling=is_prefilling,
        num_input_tokens=num_tokens_padded,
        actual_seq_lengths_q=self.actual_seq_lengths_q,
        positions=self.positions,
        positions_cpu=self._dsa_positions_cpu_buf if self.use_compress else None,
        attn_state=self.attn_state,
        decode_token_per_req=self.decode_token_per_req,
        prefill_context_parallel_metadata=self.long_seq_metadata,
    )

    if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
        cm_base.num_logits_indices = logits_indices.size(0)
        cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
            logits_indices
        )

    def _build_attn_group_metadata(
        kv_cache_gid: int,
        attn_gid: int,
        common_attn_metadata: CommonAttentionMetadata,
        prefill_ratio_to_sas_metadata: dict,
        decode_ratio_to_sas_metadata: dict,
        common_ratio_to_sas_metadata: dict,
        ubid: int | None = None,
    ) -> None:
        attn_group = self.attn_groups[kv_cache_gid][attn_gid]
        builder = attn_group.get_metadata_builder(ubid or 0)
        cascade_attn_prefix_len = (
            cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
            if cascade_attn_prefix_lens
            else 0
        )

        extra_attn_metadata_args = {}
        if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
            assert ubid is None, "UBatching not supported with GDN yet"
            extra_attn_metadata_args = dict(
                num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs_padded],
                num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                    :num_reqs_padded
                ],
            )

        if isinstance(builder, (AscendDSAMetadataBuilder, AscendDSACPMetadataBuilder)):
            if for_cudagraph_capture:
                prefill_ratio_to_sas_metadata = {}
                decode_ratio_to_sas_metadata = {}
                common_ratio_to_sas_metadata = {}
            extra_attn_metadata_args = dict(
                num_reqs_actual=num_reqs,
                prefill_ratio_to_sas_metadata=prefill_ratio_to_sas_metadata,
                decode_ratio_to_sas_metadata=decode_ratio_to_sas_metadata,
                common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                block_size=attn_group.kv_cache_spec.block_size,
            )

        # add kvcomp_metadata into common_attn_metadata
        is_pdmix = self.vllm_config.kv_transfer_config is None or (
            self.vllm_config.kv_transfer_config.is_kv_producer
            and self.vllm_config.kv_transfer_config.is_kv_consumer
        )
        if for_cudagraph_capture and not isinstance(
            builder,
            (
                AscendDSAMetadataBuilder,
                AscendDSACPMetadataBuilder,
                AscendSFADCPMetadataBuilder,
            ),
        ):
            attn_metadata_i = builder.build_for_cudagraph_capture(common_attn_metadata)
            if (
                hasattr(self.vllm_config.model_config, "hf_text_config")
                and hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
                and has_ucm_sparse()
                and PTOPK_PREFETCH_ENABLE
                and (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
            ):
                attn_metadata_i.gsa_slot_len = common_attn_metadata.num_input_tokens
                attn_metadata_i.gsa_prefetch_len = common_attn_metadata.num_input_tokens
                attn_metadata_i.gsa_warmup_len = 0
                attn_metadata_i.gsa_swap_len = common_attn_metadata.num_input_tokens + 1
                if IS_INDEXER_OFFLOAD:
                    attn_metadata_i.indexer_last_query_len = (
                        common_attn_metadata.num_input_tokens
                    )
        else:
            attn_metadata_i = builder.build(
                common_prefix_len=cascade_attn_prefix_len,
                common_attn_metadata=common_attn_metadata,
                **extra_attn_metadata_args,
            )
            # NOTE(zxr): Due to the Triton operator does not deal with -1 padding in FullGraph mode,
            # the padding needs to be changed from -1 to 0 to avoid writing invalid mamba block.
            if (
                self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
                and isinstance(builder, GDNAttentionMetadataBuilder)
                and attn_metadata_i.num_prefills == 0
            ):
                if (
                    attn_metadata_i.num_decodes == 0
                    and attn_metadata_i.num_spec_decodes > 0
                ):
                    attn_metadata_i.spec_state_indices_tensor[
                        attn_metadata_i.num_spec_decodes :
                    ].fill_(0)

        if (
            hasattr(self.vllm_config.model_config, "hf_text_config")
            and hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
            and has_ucm_sparse()
            and PTOPK_PREFETCH_ENABLE
            and (is_pdmix or not self.vllm_config.kv_transfer_config.is_kv_producer)
        ):
            attn_metadata_i.num_padding_reqs = common_attn_metadata.num_input_tokens
            attn_metadata_i.gsa_prefill_query = self.gsa_prefill_query
            attn_metadata_i.gsa_topk_win_tmp = self.gsa_topk_win_tmp
            # TODO
            attn_metadata_i.dsa_seq_len = self.dsa_seq_len.npu[
                : common_attn_metadata.num_reqs
            ]
            attn_metadata_i.dsa_block_table = self.dsa_block_table.npu[
                :, : common_attn_metadata.num_reqs
            ]
            attn_metadata_i.dsa_slot_mapping = self.dsa_slot_mapping.npu[
                : common_attn_metadata.num_input_tokens
            ]
            attn_metadata_i.gsa_prefetch_index = self.gsa_prefetch_index.npu[
                : common_attn_metadata.num_input_tokens
            ]
            attn_metadata_i.gsa_swap_index = self.gsa_swap_index.npu[
                : common_attn_metadata.num_input_tokens
            ]
            attn_metadata_i.gsa_warmup_index = self.gsa_warmup_index.npu
            attn_metadata_i.gsa_prefetch_topk = self.gsa_prefetch_topk.npu[
                :, : common_attn_metadata.num_input_tokens
            ]
            attn_metadata_i.gsa_swap_topk = self.gsa_swap_topk.npu[
                :, : common_attn_metadata.num_input_tokens + 1
            ]
            attn_metadata_i.gsa_warmup_topk = self.gsa_warmup_topk.npu
            #### indexer offload
            if IS_INDEXER_OFFLOAD:
                attn_metadata_i.indexer_last_query = self.indexer_last_query.npu
                attn_metadata_i.indexer_last_q_index = self.indexer_last_q_index.npu
                attn_metadata_i.indexer_last_weight = self.indexer_last_weight.npu

        if isinstance(builder, AscendDSAMetadataBuilder):
            prefill_ratio_to_sas_metadata = builder.prefill_ratio_to_sas_metadata  # type: ignore[assignment]
            decode_ratio_to_sas_metadata = builder.decode_ratio_to_sas_metadata  # type: ignore[assignment]
            common_ratio_to_sas_metadata = builder.common_ratio_to_sas_metadata  # type: ignore[assignment]

        if ubid is None:
            assert isinstance(attn_metadata, dict)
            attn_metadata_dict = attn_metadata
        else:
            assert isinstance(attn_metadata, list)
            attn_metadata_dict = attn_metadata[ubid]

        for layer_name in attn_group.layer_names:
            attn_metadata_dict[layer_name] = attn_metadata_i

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    prefill_ratio_to_sas_metadata: dict[Any, Any] = {}
    decode_ratio_to_sas_metadata: dict[Any, Any] = {}
    common_ratio_to_sas_metadata: dict[Any, Any] = {}
    spec_decode_common_attn_metadata = None
    for kv_cache_gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
        cm = copy(cm_base)  # shallow copy
        # Basically only the encoder seq_lens, block_table and slot_mapping change
        # for each kv_cache_group.
        cm.encoder_seq_lens, cm.encoder_seq_lens_cpu = self._get_encoder_seq_lens(
            num_scheduled_tokens or {},
            kv_cache_group.kv_cache_spec,
            num_reqs_padded,
        )

        # Now, query_start_loc is padded.
        # But gdn needs an unpadded one.
        # gdn_query_start_loc is an unpadded version of query_start_loc.
        # TODO delete it if fia's check is removed.
        if self._has_gdn:
            attn_group = self.attn_groups[kv_cache_gid][0]
            builder = attn_group.get_metadata_builder(0)
            if isinstance(builder, GDNAttentionMetadataBuilder):
                cm.query_start_loc_cpu = self.gdn_query_start_loc.cpu[
                    : num_reqs_padded + 1
                ]
                cm.query_start_loc = self.gdn_query_start_loc.gpu[: num_reqs_padded + 1]

        if kv_cache_gid > 0:
            cm.block_table_tensor, cm.slot_mapping = _get_block_table_and_slot_mapping(
                kv_cache_gid
            )
        if self.speculative_config and isinstance(
            self.drafter, AscendStep3p5MTPProposer
        ):
            # step3p5 MTP draft layers span multiple KV cache groups; capture
            # each group's block table / slot mapping so the proposer can
            # build per-step attention metadata for the active MTP layer.
            self.drafter.set_per_group_attn_metadata(
                kv_cache_gid, cm.block_table_tensor, cm.slot_mapping
            )
        if self.speculative_config and spec_decode_common_attn_metadata is None:
            if isinstance(
                self.drafter,
                AscendEagleProposer | AscendDraftModelProposer | AscendDflashProposer,
            ):
                if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                    spec_decode_common_attn_metadata = cm
            else:
                spec_decode_common_attn_metadata = cm
        if self.enable_hamming_sparse is True:
            from vllm_ascend.attention.kvcomp_attn.attention_utils import (
                build_kvcomp_metadata,
            )

            build_kvcomp_metadata(self.kvcomp_meta_data, cm)
        for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
            _build_attn_group_metadata(
                kv_cache_gid,
                attn_gid,
                cm,
                prefill_ratio_to_sas_metadata,
                decode_ratio_to_sas_metadata,
                common_ratio_to_sas_metadata,
            )
    if self.is_mm_prefix_lm:
        req_doc_ranges = {}
        for req_id in self.input_batch.req_ids:
            image_doc_ranges = []
            req_state = self.requests[req_id]
            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                img_doc_range = pos_info.extract_embeds_range()
                image_doc_ranges.extend(img_doc_range)
            req_idx = self.input_batch.req_id_to_index[req_id]
            req_doc_ranges[req_idx] = image_doc_ranges

        if isinstance(attn_metadata, list):
            for ub_metadata in attn_metadata:
                for _metadata in ub_metadata.values():
                    _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]
        else:
            for _metadata in attn_metadata.values():
                _metadata.mm_prefix_range = req_doc_ranges  # type: ignore[attr-defined]

    if spec_decode_common_attn_metadata is not None and (
        num_reqs != num_reqs_padded or num_tokens != num_tokens_padded
    ):
        # Currently the drafter still only uses piecewise cudagraphs (and modifies
        # the attention metadata in directly), and therefore does not want to use
        # padded attention metadata.
        spec_decode_common_attn_metadata = spec_decode_common_attn_metadata.unpadded(
            num_tokens, num_reqs
        )
    return attn_metadata, spec_decode_common_attn_metadata


def maybe_execute_ucm_sparse_begin(
    self,
    scheduler_output: "SchedulerOutput",
    attn_metadata: AscendCommonAttentionMetadata,
):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    if has_kv_transfer_group():
        uc_connector = get_kv_transfer_group()
        uc_setup_model = getattr(uc_connector, "setup_model", None)
        if callable(uc_setup_model):
            uc_setup_model(self.model)
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.build_sparse_meta(
        scheduler_output, self.requests, self.input_batch, attn_metadata
    )
    ucm_sparse.execute_begin(scheduler_output, attn_metadata)


def maybe_execute_ucm_sparse_warmup(self):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.warmup()


def maybe_execute_ucm_sparse_finished(self, rec_result=None):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    # return ucm_sparse.execute_finished(logits_indices)
    ucm_sparse.execute_finished(rec_result)


def ucm_sparse_request_finished_in_worker(self, request_id: str | int):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.request_finished_in_worker(request_id)


@when_imported("vllm_ascend.worker.model_runner_v1")
def _patch(mod):
    if hasattr(mod.NPUModelRunner, "__init__"):
        _patched = FunctionType(
            __init__.__code__,
            vars(mod),
            name=__init__.__name__,
            argdefs=__init__.__defaults__,
            closure=__init__.__closure__,
        )
        _patched.__annotations__ = __init__.__annotations__
        _patched.__kwdefaults__ = __init__.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "__init__", _patched)
        logger.info("UCM GSA NPUModelRunner.__init__ patch applied")

    if hasattr(mod.NPUModelRunner, "_set_up_drafter"):
        _patched = FunctionType(
            _set_up_drafter.__code__,
            vars(mod),
            name=_set_up_drafter.__name__,
            argdefs=_set_up_drafter.__defaults__,
            closure=_set_up_drafter.__closure__,
        )
        _patched.__annotations__ = _set_up_drafter.__annotations__
        _patched.__kwdefaults__ = _set_up_drafter.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "_set_up_drafter", _patched)
        logger.info("UCM GSA NPUModelRunner._set_up_drafter patch applied")

    if hasattr(mod.NPUModelRunner, "_prepare_inputs"):
        _patched = FunctionType(
            _prepare_inputs.__code__,
            vars(mod),
            name=_prepare_inputs.__name__,
            argdefs=_prepare_inputs.__defaults__,
            closure=_prepare_inputs.__closure__,
        )
        _patched.__annotations__ = _prepare_inputs.__annotations__
        _patched.__kwdefaults__ = _prepare_inputs.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "_prepare_inputs", _patched)
        logger.info("UCM GSA NPUModelRunner._prepare_inputs patch applied")

    if hasattr(mod.NPUModelRunner, "_build_attn_state"):
        _patched = FunctionType(
            _build_attn_state.__code__,
            vars(mod),
            name=_build_attn_state.__name__,
            argdefs=_build_attn_state.__defaults__,
            closure=_build_attn_state.__closure__,
        )
        _patched.__annotations__ = _build_attn_state.__annotations__
        _patched.__kwdefaults__ = _build_attn_state.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "_build_attn_state", _patched)
        logger.info("UCM GSA NPUModelRunner._build_attn_state patch applied")

    if hasattr(mod.NPUModelRunner, "execute_model"):
        _patched = FunctionType(
            execute_model.__code__,
            vars(mod),
            name=execute_model.__name__,
            argdefs=execute_model.__defaults__,
            closure=execute_model.__closure__,
        )
        _patched.__annotations__ = execute_model.__annotations__
        _patched.__kwdefaults__ = execute_model.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "execute_model", _patched)
        logger.info("UCM GSA NPUModelRunner.execute_model patch applied")

    if hasattr(mod.NPUModelRunner, "sample_tokens"):
        _patched = FunctionType(
            sample_tokens.__code__,
            vars(mod),
            name=sample_tokens.__name__,
            argdefs=sample_tokens.__defaults__,
            closure=sample_tokens.__closure__,
        )
        _patched.__annotations__ = sample_tokens.__annotations__
        _patched.__kwdefaults__ = sample_tokens.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "sample_tokens", _patched)
        logger.info("UCM GSA NPUModelRunner.sample_tokens patch applied")

    if hasattr(mod.NPUModelRunner, "_build_attention_metadata"):
        _patched = FunctionType(
            _build_attention_metadata.__code__,
            vars(mod),
            name=_build_attention_metadata.__name__,
            argdefs=_build_attention_metadata.__defaults__,
            closure=_build_attention_metadata.__closure__,
        )
        _patched.__annotations__ = _build_attention_metadata.__annotations__
        _patched.__kwdefaults__ = _build_attention_metadata.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "_build_attention_metadata", _patched)
        logger.info("UCM GSA NPUModelRunner._build_attention_metadata patch applied")

    if not hasattr(mod.NPUModelRunner, "maybe_execute_ucm_sparse_begin"):
        _patched = FunctionType(
            maybe_execute_ucm_sparse_begin.__code__,
            vars(mod),
            name=maybe_execute_ucm_sparse_begin.__name__,
            argdefs=maybe_execute_ucm_sparse_begin.__defaults__,
            closure=maybe_execute_ucm_sparse_begin.__closure__,
        )
        _patched.__annotations__ = maybe_execute_ucm_sparse_begin.__annotations__
        _patched.__kwdefaults__ = maybe_execute_ucm_sparse_begin.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "maybe_execute_ucm_sparse_begin", _patched)
        logger.info("UCM GSA NPUModelRunner.maybe_execute_ucm_sparse_begin injected")

    if not hasattr(mod.NPUModelRunner, "maybe_execute_ucm_sparse_warmup"):
        _patched = FunctionType(
            maybe_execute_ucm_sparse_warmup.__code__,
            vars(mod),
            name=maybe_execute_ucm_sparse_warmup.__name__,
            argdefs=maybe_execute_ucm_sparse_warmup.__defaults__,
            closure=maybe_execute_ucm_sparse_warmup.__closure__,
        )
        _patched.__annotations__ = maybe_execute_ucm_sparse_warmup.__annotations__
        _patched.__kwdefaults__ = maybe_execute_ucm_sparse_warmup.__kwdefaults__
        patch_or_inject(mod.NPUModelRunner, "maybe_execute_ucm_sparse_warmup", _patched)
        logger.info("UCM GSA NPUModelRunner.maybe_execute_ucm_sparse_warmup injected")

    if not hasattr(mod.NPUModelRunner, "maybe_execute_ucm_sparse_finished"):
        _patched = FunctionType(
            maybe_execute_ucm_sparse_finished.__code__,
            vars(mod),
            name=maybe_execute_ucm_sparse_finished.__name__,
            argdefs=maybe_execute_ucm_sparse_finished.__defaults__,
            closure=maybe_execute_ucm_sparse_finished.__closure__,
        )
        _patched.__annotations__ = maybe_execute_ucm_sparse_finished.__annotations__
        _patched.__kwdefaults__ = maybe_execute_ucm_sparse_finished.__kwdefaults__
        patch_or_inject(
            mod.NPUModelRunner, "maybe_execute_ucm_sparse_finished", _patched
        )
        logger.info("UCM GSA NPUModelRunner.maybe_execute_ucm_sparse_finished injected")

    if not hasattr(mod.NPUModelRunner, "ucm_sparse_request_finished_in_worker"):
        _patched = FunctionType(
            ucm_sparse_request_finished_in_worker.__code__,
            vars(mod),
            name=ucm_sparse_request_finished_in_worker.__name__,
            argdefs=ucm_sparse_request_finished_in_worker.__defaults__,
            closure=ucm_sparse_request_finished_in_worker.__closure__,
        )
        _patched.__annotations__ = ucm_sparse_request_finished_in_worker.__annotations__
        _patched.__kwdefaults__ = ucm_sparse_request_finished_in_worker.__kwdefaults__
        patch_or_inject(
            mod.NPUModelRunner, "ucm_sparse_request_finished_in_worker", _patched
        )
        logger.info(
            "UCM GSA NPUModelRunner.ucm_sparse_request_finished_in_worker injected"
        )
