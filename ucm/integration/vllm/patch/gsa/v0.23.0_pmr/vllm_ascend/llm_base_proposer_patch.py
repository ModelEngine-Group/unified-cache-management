"""GSA llm_base_proposer patches for vLLM-Ascend 0.23.0."""

from __future__ import annotations

from types import FunctionType

from vllm.config import CUDAGraphMode

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def __init__(
    self,
    vllm_config: VllmConfig,
    device: torch.device,
    pass_hidden_states_to_model: bool,
    runner=None,
):
    from ucm.utils import Config

    super(AscendSpecDecodeBaseProposer, self).__init__(
        vllm_config, device, pass_hidden_states_to_model, runner=runner
    )

    ucm_config = Config(vllm_config.kv_transfer_config)
    self.launch_config = ucm_config.get_config()
    self.pmr_tree_config = self.launch_config.get("pmr_config", None)
    self.use_pmr = False
    if self.pmr_tree_config:
        self.use_pmr = self.pmr_tree_config["use_pmr"]
    self.num_speculative_tokens_mtp = self.num_speculative_tokens
    if self.use_pmr:
        self.num_speculative_tokens_pmr = self.pmr_tree_config["num_speculative_tokens"]
        self.num_speculative_tokens_mtp -= self.num_speculative_tokens_pmr

    # Assign runner before it's used in the methods below
    self.runner = runner

    logger.debug(
        "[spec_decode/base] Initializing spec decode proposer: method=%s,"
        " num_speculative_tokens=%s, hidden_size=%s, pass_hidden_states=%s,"
        " parallel_drafting=%s, use_cuda_graph=%s, device=%s",
        self.method,
        self.num_speculative_tokens,
        self.hidden_size,
        pass_hidden_states_to_model,
        self.speculative_config.parallel_drafting if self.speculative_config else False,
        runner._use_aclgraph() if runner else False,
        device,
    )

    self.use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling
    self.use_compress = hasattr(
        self.vllm_config.model_config.hf_config, "compress_ratios"
    )
    self.has_gdn = check_gdn_layer(self.vllm_config)
    self.pass_hidden_states_to_model = pass_hidden_states_to_model
    self.decode_threshold = 1 + self.num_speculative_tokens_mtp
    self.query_start_loc = self.runner._make_buffer(
        self.runner.max_num_reqs + 2, dtype=torch.int32
    )
    self.arange_cpu = torch.arange(
        self.arange.shape[0], device="cpu", dtype=torch.int32
    )
    self.attn_mask_builder = AttentionMaskBuilder(self.device)

    self.enable_shared_expert_dp = shared_expert_dp_enabled()

    self.pcp_size = self.runner.pcp_size
    self.dcp_size = self.runner.dcp_size

    self.use_sparse = hasattr(vllm_config.model_config.hf_text_config, "index_topk")

    self._share_mtp_indices = False
    spec_config = self.vllm_config.speculative_config
    draft_model_config = getattr(spec_config, "draft_model_config", None)
    draft_hf_config = (
        draft_model_config.hf_config if draft_model_config is not None else None
    )
    self._share_mtp_indices = getattr(
        draft_hf_config, "index_share_for_mtp_iteration", False
    )

    # NOTE:
    # `draft_tensor_parallel_size` does not take effect for Eagle:
    # the draft model uses the same TP size as the target model in practice.
    # so we applied this patch to set tp=1 of draft model separately.
    # Due to verification of `_verify_and_get_draft_tp` in vllm,
    # the value of `draft_tensor_parallel_size` here will either be 1 separately
    # or the same as target model.
    # TODO(zhaomingyu13): If we want to adapt to the case where draft model tp
    # is not 1 and differs from target model, this part should be rewritten.
    if (
        vllm_config.parallel_config.tensor_parallel_size
        != self.speculative_config.draft_tensor_parallel_size
    ):
        tp_group = init_model_parallel_group(
            [[get_world_group().rank]],
            get_world_group().rank,
            torch.distributed.get_backend(get_world_group().device_group),
            use_message_queue_broadcaster=True,
            group_name="tp",
        )
        self.tp_group_context: AbstractContextManager[Any] = (
            patch_tensor_parallel_group(tp_group)
        )
    else:
        self.tp_group_context = nullcontext()

    self.use_cuda_graph = (
        self.runner._use_aclgraph() and not self.speculative_config.enforce_eager
    )
    self._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    # GLM series models: speculative decoding does not yet support running
    # the draft model in graph mode. Force the draft model to always use
    # eager mode. This is equivalent to the user adding
    # `"enforce_eager": true` to the `--speculative-config`, and keeps
    # the target model's graph-mode setting untouched.
    # TODO(lilinsiman): Remove this code segment after future versions of the GLM
    # series models support graph input for speculative inference.
    if _is_glm_model(self.vllm_config.model_config):
        if self.use_cuda_graph:
            logger.warning(
                "GLM series models with speculative decoding currently do "
                "not support graph mode. The draft model has been "
                "automatically switched to eager mode "
                "(enforce_eager=true). Graph mode support for GLM "
                "speculative decoding will be added in a future release. "
            )
        self.use_cuda_graph = False

    # TODO: Remove it when the bug of fx-graph is solved
    self.maybe_eager_context: AbstractContextManager[Any] = nullcontext()
    if not self.use_cuda_graph and enable_sp(vllm_config):
        self.maybe_eager_context = _maybe_eager_context(vllm_config)

    self.token_indices_to_sample = torch.zeros(
        self.vllm_config.scheduler_config.max_num_batched_tokens,
        dtype=torch.int32,
        device=device,
    )
    metadata_lens = (
        self.runner.max_num_tokens + 2 * self.pcp_size * self.runner.max_num_reqs
    )
    num_mtp_draft_slots = (
        max(self.num_speculative_tokens - 1, 0) * self.runner.max_num_reqs
    )
    slot_mapping_lens = metadata_lens + num_mtp_draft_slots
    self.slot_mapping_group = [
        torch.zeros(
            slot_mapping_lens,
            dtype=torch.int32,
            device=device,
            pin_memory=self.runner.pin_memory,
        )
        for _ in range(self.num_speculative_tokens_mtp)
    ]

    # dsv32 needs seq_lens and query_start_loc persistent tensors for full graph mode
    self.seq_lens_group = [
        torch.zeros(
            metadata_lens,
            dtype=torch.int32,
            device=device,
            pin_memory=self.runner.pin_memory,
        )
        for _ in range(self.num_speculative_tokens_mtp)
    ]
    self.query_start_loc_group = [
        torch.zeros(
            metadata_lens,
            dtype=torch.int32,
            device=device,
            pin_memory=self.runner.pin_memory,
        )
        for _ in range(self.num_speculative_tokens_mtp)
    ]

    # pcp needs independent block table tensor in step=0 and step>0, and the following is for step>0
    # since final block table tensor is not ready in __init__, it is delayed until dummy_run
    self.block_table_tensor_clone: torch.Tensor | None = None

    self._runnable = self._run_merged_draft
    self.is_multimodal_model = self.vllm_config.model_config.is_multimodal_model
    if self.uses_mrope:
        self.mrope_positions = torch.zeros(
            (3, self.max_num_tokens + 1), dtype=torch.int32, device=device
        )
    elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
        self.xdrope_positions = torch.zeros(
            (self.uses_xdrope_dim, self.max_num_tokens + 1),
            dtype=torch.int32,
            device=device,
        )
    else:
        # RoPE need (max_num_tokens,)
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )

    self.token_arange_np = np.arange(self.max_num_tokens + 1, dtype=np.int32)
    self.enable_enpu = self.runner.enable_enpu
    self.use_eagle = self.runner.use_eagle


def dummy_run(
    self,
    num_tokens: int,
    with_prefill: bool = False,
    in_graph_capturing: bool = False,
    num_reqs: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor=None,
    dummy_compute_logits=lambda hidden_states: None,
    is_profile=False,
):
    (
        num_tokens,
        num_tokens_across_dp,
        _,
    ) = self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)
    pcp_manager = getattr(self.runner, "pcp_manager", None)

    multi_steps_attn_metadata = []
    if not self.use_cuda_graph:
        aclgraph_runtime_mode = CUDAGraphMode.NONE

    # init block table tensor clone is only available after profile run and is only used for graph mode
    if (
        self.pcp_size * self.dcp_size > 1
        and self.use_cuda_graph
        and not is_profile
        and self.block_table_tensor_clone is None
    ):
        self.block_table_tensor_clone = torch.zeros(
            (
                self.runner.max_num_tokens
                + 2 * self.pcp_size * self.runner.max_num_reqs,
                self.runner.input_batch.block_table[0].get_device_tensor().shape[1],
            ),
            dtype=torch.int32,
            device=self.device,
            pin_memory=self.runner.pin_memory,
        )

    if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
        num_computed_tokens_cpu = (
            self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
        )

        # num_reqs is already the padded version
        self.query_start_loc.cpu[: num_reqs + 1].copy_(
            self.runner.query_start_loc.cpu[: num_reqs + 1]
        )
        copy_snapshot_to_gpu(self.query_start_loc)

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
            seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
            _seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
            seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
            seq_lens=self.runner.seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            num_input_tokens=num_tokens,
            max_query_len=self.num_speculative_tokens_mtp + 1,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=self.runner.input_batch.block_table[
                self.kv_cache_gid
            ].get_device_tensor()[:num_reqs],
            # This is used to hold a position.
            slot_mapping=self.runner.input_batch.block_table[
                self.kv_cache_gid
            ].slot_mapping.gpu,
            positions=self.runner.positions,
            positions_cpu=(
                self.runner._dsa_positions_cpu_buf if self.use_compress else None
            ),
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
            max_seq_len=0,
        )
        if pcp_manager is not None:
            # update long_seq related params and flatten block_table
            common_attn_metadata.prefill_context_parallel_metadata = (
                pcp_manager.long_seq_metadata
            )

        assert len(self.draft_attn_groups) > 0
        builder = self.draft_attn_groups[0].get_metadata_builder()
        kv_cache_spec = self.draft_attn_groups[0].kv_cache_spec
        # update the tensor's address for each step.
        for draft_index in range(self.num_speculative_tokens_mtp):
            common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
            extra_attn_metadata_args: dict = {}
            if self.use_compress:
                extra_attn_metadata_args.update(
                    prefill_ratio_to_sas_metadata=dict(),
                    decode_ratio_to_sas_metadata=dict(),
                    common_ratio_to_sas_metadata=dict(),
                    block_size=kv_cache_spec.block_size,
                )
            # Set the real slot_mapping.
            slot_mapping_lens = common_attn_metadata.slot_mapping.shape[0]
            self.slot_mapping_group[draft_index][:slot_mapping_lens].copy_(
                common_attn_metadata.slot_mapping
            )
            self.slot_mapping_group[draft_index][slot_mapping_lens:].fill_(
                PADDING_SLOT_ID
            )
            common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_index]
            self.seq_lens_group[draft_index][:num_reqs].copy_(
                common_attn_metadata.seq_lens
            )
            self.seq_lens_group[draft_index][num_reqs:].fill_(0)
            common_attn_metadata.seq_lens = self.seq_lens_group[draft_index][:num_reqs]
            self.query_start_loc_group[draft_index][: num_reqs + 1].copy_(
                common_attn_metadata.query_start_loc
            )
            self.query_start_loc_group[draft_index][num_reqs + 1 :].fill_(0)
            common_attn_metadata.query_start_loc = self.query_start_loc_group[
                draft_index
            ][: num_reqs + 1]
            if self.pcp_size * self.dcp_size > 1 and draft_index > 0:
                assert (
                    self.block_table_tensor_clone is not None
                ), "block_table_tensor_clone is not init"
                common_attn_metadata.block_table_tensor = self.block_table_tensor_clone[
                    :num_reqs
                ]
            if not self.use_compress or draft_index == 0:
                attn_metadata_eagle = builder.build_for_graph_capture(
                    common_attn_metadata,
                    (
                        AscendAttentionState.SpecDecoding
                        if self.method == "mtp"
                        else AscendAttentionState.ChunkedPrefill
                    ),
                    **extra_attn_metadata_args,
                )
            else:
                attn_metadata_eagle = builder.build_for_drafting(
                    common_attn_metadata,
                    draft_index,
                    **extra_attn_metadata_args,
                )
            per_layer_attn_metadata = dict()
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata_eagle
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

    model_positions = self._get_positions(num_tokens)

    batch_size = max(num_tokens // (self.num_speculative_tokens_mtp + 1), 1)
    # TODO: temporarily hack here, we should find out batch_size for profile_run
    if is_profile:
        batch_size = min(batch_size, self.runner.max_num_reqs)

    if self.supports_mm_inputs:
        mm_embeds, is_mm_embed = (None, None)
        inputs_embeds = self.model.embed_input_ids(
            self.input_ids[:num_tokens],
            multimodal_embeddings=mm_embeds,
            is_multimodal=is_mm_embed,
        )
        self.inputs_embeds[:num_tokens] = inputs_embeds
        inputs_embeds = self.inputs_embeds[:num_tokens]
    else:
        inputs_embeds = None

    self.token_indices_to_sample.fill_(0)

    with set_ascend_forward_context(
        multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
        self.vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        num_actual_tokens=0,
        in_profile_run=is_profile,
        batch_descriptor=batch_descriptor,
        aclgraph_runtime_mode=aclgraph_runtime_mode,
        is_draft_model=True,
        draft_attn_metadatas=multi_steps_attn_metadata,
    ):
        # Reset MOE layer index before first model call
        forward_context = get_forward_context()
        if forward_context is not None:
            forward_context.moe_layer_index = 0

        self._runnable(
            num_input_tokens=num_tokens,
            batch_size=batch_size,
            token_indices_to_sample=self.token_indices_to_sample[
                : batch_size * self.extra_slots_per_request
            ],
            # The target_position's address is same as the model_positions's
            target_positions=model_positions,
            inputs_embeds=inputs_embeds,
            multi_steps_attn_metadata=multi_steps_attn_metadata,
            num_tokens=num_tokens,
        )
        forward_context = get_forward_context()
        if (
            forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and not _EXTRA_CTX.capturing
        ):
            self._update_full_graph_params(
                forward_context, num_tokens, multi_steps_attn_metadata
            )


def _propose(
    self,
    # [num_tokens]
    target_token_ids: torch.Tensor,
    # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
    target_positions: torch.Tensor,
    # [num_tokens, hidden_size]
    target_hidden_states: torch.Tensor,
    # [batch_size]
    next_token_ids: torch.Tensor,
    token_indices_to_sample: torch.Tensor | None,
    common_attn_metadata: CommonAttentionMetadata,
    target_model_batch_desc: BatchDescriptor,
    sampling_metadata: SamplingMetadata,
    mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    req_scheduled_tokens=None,
    long_seq_metadata=None,
    num_prefill_reqs=0,
    num_decode_reqs=0,
    scheduler_output: SchedulerOutput = None,
    num_scheduled_tokens: int = 0,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = common_attn_metadata.batch_size()

    if token_indices_to_sample is None:
        token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

    if self.method in ("eagle3", "dflash"):
        assert isinstance(
            self.get_model(),
            (
                Eagle3LlamaForCausalLM,
                DFlashQwen3ForCausalLM,
                Eagle3VwnLlamaForCausalLM,
                Eagle3DeepseekV2ForCausalLM,
            ),
        )
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        assert target_hidden_states.shape[-1] == self.hidden_size

    num_tokens, token_indices_to_sample, common_attn_metadata, long_seq_args = (
        self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
    )
    assert self.runner is not None
    pcp_manager = getattr(self.runner, "pcp_manager", None)
    if pcp_manager is not None:
        assert long_seq_args is not None
        _, ori_token_indices_to_sample = long_seq_args

    has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0
    uniform_decode = target_model_batch_desc.uniform

    if self.use_cuda_graph:
        _, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora
        )
        num_input_tokens = batch_descriptor.num_tokens
    else:
        num_input_tokens = num_tokens

    (
        num_input_tokens,
        num_tokens_across_dp,
        _,
    ) = self.runner._sync_metadata_across_dp(num_input_tokens, is_draft_model=True)

    if self.use_cuda_graph:
        aclgraph_runtime_mode, batch_descriptor = (
            self.runner.cudagraph_dispatcher.dispatch(
                num_tokens=num_input_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
            )
        )
        num_input_tokens = batch_descriptor.num_tokens
    else:
        aclgraph_runtime_mode = CUDAGraphMode.NONE
        batch_descriptor = None

    if aclgraph_runtime_mode == CUDAGraphMode.FULL:
        # TODO: Due to the inconsistency between the proposer `dispatcher` and model runner, this padding
        # should have been done in model runner but not. For example, at prefill stage, target model
        # is run in eager mode currently, which means `_pad_query_start_loc_for_fia` is not called,
        # while draft model is run in graph model, which means we should pad the `query_start_loc`.
        # Need to be fixed in the future.
        num_reqs = common_attn_metadata.query_start_loc.shape[0]
        self.query_start_loc.gpu[:num_reqs].copy_(common_attn_metadata.query_start_loc)
        self.query_start_loc.cpu[:num_reqs].copy_(
            common_attn_metadata.query_start_loc_cpu
        )
        num_reqs_padded = self.runner._pad_query_start_loc_for_fia(
            self.query_start_loc,
            num_input_tokens,
            (
                batch_descriptor.num_reqs
                if batch_descriptor.num_reqs is not None
                else common_attn_metadata.num_reqs
            ),
            common_attn_metadata.num_reqs,
            aclgraph_runtime_mode,
            batch_descriptor.num_reqs,
        )
        common_attn_metadata.num_reqs = num_reqs_padded
        common_attn_metadata.query_start_loc = self.query_start_loc.gpu[
            : num_reqs_padded + 1
        ]
        common_attn_metadata.query_start_loc_cpu = self.query_start_loc.cpu[
            : num_reqs_padded + 1
        ]
        slicing_length = (
            num_reqs_padded * self.decode_threshold
            if self.pcp_size * self.dcp_size > 1
            else num_reqs_padded
        )
        common_attn_metadata.block_table_tensor = self._adjust_tensor(
            common_attn_metadata.block_table_tensor, slicing_length
        )
        if self.method == "dflash":
            common_attn_metadata.seq_lens = self._adjust_tensor(
                common_attn_metadata.seq_lens, num_reqs_padded
            )
        else:
            common_attn_metadata.seq_lens = self._adjust_tensor(
                self.runner.seq_lens, num_reqs_padded
            )
            common_attn_metadata.seq_lens_cpu = self._adjust_tensor(
                self.runner.optimistic_seq_lens_cpu, num_reqs_padded
            )
            # Keep the upstream-canonical mirror length-aligned with the
            # padded subclass field, but only if the caller already
            # populated it (production cm_base does; some unit-test mocks
            # leave it None and assert it stays None). ``.clone()`` keeps
            # the two fields independent so per-step in-place updates in
            # ``attn_update_stack_num_spec_norm`` don't double-count.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu = (
                    common_attn_metadata.seq_lens_cpu.clone()
                )
        if common_attn_metadata.num_computed_tokens_cpu is not None:
            common_attn_metadata.num_computed_tokens_cpu = self._adjust_tensor(
                common_attn_metadata.num_computed_tokens_cpu, num_reqs_padded
            )

        if pcp_manager is not None and self.pcp_size > 1:
            pcp_manager.mask_spec_decode_restore_idx_for_graph(
                common_attn_metadata.prefill_context_parallel_metadata.pcp_allgather_restore_idx
            )
    else:
        num_reqs_padded = common_attn_metadata.num_reqs
        # In the below scenario, padding has been applied by _pad_query_start_loc_for_fia in the model runner.
        # We need to unpad here for eager mode to maintain compatibility.
        if (
            not self.vllm_config.model_config.use_mla
            and self.pcp_size * self.dcp_size == 1
        ):
            common_attn_metadata.block_table_tensor = self._adjust_tensor(
                common_attn_metadata.block_table_tensor, num_reqs_padded
            )

    if self.supports_mm_inputs:
        mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)
        inputs_embeds = self.model.embed_input_ids(
            self.input_ids[:num_tokens],
            multimodal_embeddings=mm_embeds,
            is_multimodal=is_mm_embed,
        )
        self.inputs_embeds[:num_tokens] = inputs_embeds
        inputs_embeds = self.inputs_embeds[:num_input_tokens]
    else:
        inputs_embeds = None

    # Update slot_mapping for different speculative.
    # NOTE: Currently, we only remake the slot_mapping, because it's the
    # only tensor which will be used in current FIA.
    # Strictly speaking, `query_start_loc`, `seq_lens` should also have
    # their memory allocated separately for each step just like `slot_mapping`.
    slot_mapping_lens = common_attn_metadata.slot_mapping.shape[0]
    self.slot_mapping_group[0][:slot_mapping_lens].copy_(
        common_attn_metadata.slot_mapping
    )
    self.slot_mapping_group[0][slot_mapping_lens:].fill_(-1)
    common_attn_metadata.slot_mapping = self.slot_mapping_group[0]

    self.seq_lens_group[0][:num_reqs_padded].copy_(common_attn_metadata.seq_lens)
    self.seq_lens_group[0][num_reqs_padded:].fill_(0)
    common_attn_metadata.seq_lens = self.seq_lens_group[0][:num_reqs_padded]

    self.query_start_loc_group[0][: num_reqs_padded + 1].copy_(
        common_attn_metadata.query_start_loc
    )
    self.query_start_loc_group[0][num_reqs_padded + 1 :].fill_(0)
    common_attn_metadata.query_start_loc = self.query_start_loc_group[0][
        : num_reqs_padded + 1
    ]

    common_attn_metadata.num_input_tokens = num_input_tokens
    # FIXME(woosuk): The below two ops cause synchronization. Optimize.
    assert len(self.draft_attn_groups) > 0
    builder = self.draft_attn_groups[0].get_metadata_builder()
    extra_attn_metadata_args: dict = {}
    if self.use_compress:
        extra_attn_metadata_args = dict(
            prefill_ratio_to_sas_metadata=dict(),
            decode_ratio_to_sas_metadata=dict(),
            common_ratio_to_sas_metadata=dict(),
            block_size=self.draft_attn_groups[0].kv_cache_spec.block_size,
        )
    attn_metadata = builder.build(
        0, common_attn_metadata, self.runner.get_model(), **extra_attn_metadata_args
    )

    if hasattr(attn_metadata, "causal") and not attn_metadata.causal:
        attn_metadata.attn_mask = None

    if self.uses_mrope:
        used_update_positions = self.mrope_positions[:, token_indices_to_sample]
    else:
        used_update_positions = self.positions[token_indices_to_sample]
    per_layer_attn_metadata = dict()
    # The first step of speculative.
    for layer_name in self.attn_layer_names:
        per_layer_attn_metadata[layer_name] = attn_metadata
    multi_steps_attn_metadata = [per_layer_attn_metadata]

    # Copy the old attn_metadata and update
    attn_metadata_i = per_layer_attn_metadata[self.attn_layer_names[0]]

    # Clone the data so that when calculating the data at position 2 and position 3
    # in the merged graph, it does not affect position 1
    # FIXME(lilinsiman)
    if self.pcp_size * self.dcp_size > 1 and self.use_cuda_graph:
        assert (
            self.block_table_tensor_clone is not None
        ), "block_table_tensor_clone is not init"
        self.block_table_tensor_clone[
            : common_attn_metadata.block_table_tensor.shape[0]
        ] = common_attn_metadata.block_table_tensor
        common_attn_metadata.block_table_tensor = self.block_table_tensor_clone[
            : common_attn_metadata.block_table_tensor.shape[0]
        ]
    else:
        common_attn_metadata.block_table_tensor = (
            common_attn_metadata.block_table_tensor.clone()
        )

    metadata_has_prefill = bool(getattr(attn_metadata_i, "num_prefills", 0))
    is_prefill_batch = num_prefill_reqs > 0 or metadata_has_prefill
    pcp_mtp_inputs = None
    draft_cp_kwargs = {
        "ori_seq_len": None,
        "ori_seq_len_cpu": None,
        "slot_indices": None,
        "mtp_slot_mapping": None,
    }
    if pcp_manager is not None:
        pcp_mtp_inputs = pcp_manager.prepare_spec_decode_mtp_drafting_inputs(
            common_attn_metadata=common_attn_metadata,
            attn_metadata=attn_metadata_i,
            ori_token_indices_to_sample=ori_token_indices_to_sample,
            batch_size=batch_size,
            num_decode_reqs=num_decode_reqs,
            is_prefill_batch=is_prefill_batch,
            num_speculative_tokens=self.num_speculative_tokens_mtp,
        )
        if pcp_mtp_inputs is not None:
            draft_cp_kwargs.update(
                ori_seq_len=pcp_mtp_inputs.seq_lens,
                ori_seq_len_cpu=pcp_mtp_inputs.seq_lens_cpu,
                slot_indices=pcp_mtp_inputs.slot_indices,
                mtp_slot_mapping=pcp_mtp_inputs.slot_mapping,
            )

    should_update_next_steps = not self.parallel_drafting and (
        self.pcp_size * self.dcp_size == 1 or pcp_mtp_inputs is not None
    )
    if should_update_next_steps:
        # Copy the old attn_metadata and update
        for draft_index in range(1, self.num_speculative_tokens_mtp):
            per_layer_attn_metadata = dict()
            for attn_group in self.draft_attn_groups:
                common_attn_metadata, attn_metadata = (
                    self.attn_update_stack_num_spec_norm(
                        draft_index,
                        attn_metadata,
                        common_attn_metadata,
                        batch_size,
                        num_input_tokens,
                        used_update_positions,
                        aclgraph_runtime_mode,
                        **draft_cp_kwargs,
                        attn_group=attn_group,
                    )
                )
                for layer_name in self.attn_layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

    token_indices_to_sample_len = token_indices_to_sample.shape[0]
    self.token_indices_to_sample[:token_indices_to_sample_len].copy_(
        token_indices_to_sample
    )
    self.token_indices_to_sample[token_indices_to_sample_len:].fill_(0)

    with set_ascend_forward_context(
        multi_steps_attn_metadata[0],
        self.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        num_actual_tokens=num_tokens,
        batch_descriptor=batch_descriptor,
        aclgraph_runtime_mode=aclgraph_runtime_mode,
        is_draft_model=True,
        draft_attn_metadatas=multi_steps_attn_metadata,
        eplb_heat_collection_status=(
            self.runner.eplb_heat_collection_status
            if self.runner.dynamic_eplb
            else False
        ),
    ):
        # Reset MOE layer index for forward pass
        forward_context = get_forward_context()
        if forward_context is not None:
            forward_context.moe_layer_index = 0

        model_inputs: dict[str, Any] = {
            "num_input_tokens": num_input_tokens,
            "batch_size": batch_size,
            "token_indices_to_sample": self.token_indices_to_sample[
                :token_indices_to_sample_len
            ],
            "target_positions": target_positions,
            "inputs_embeds": inputs_embeds,
            "multi_steps_attn_metadata": multi_steps_attn_metadata,
            "num_tokens": num_tokens,
            "is_prefill": is_prefill_batch,
        }
        runnable = cast(Callable[..., Any], self._runnable)
        run_draft: Callable[[], Any] = partial(runnable, **model_inputs)

        if self.enable_enpu:
            self._update_full_graph_params_if_needed(
                forward_context, num_input_tokens, multi_steps_attn_metadata
            )
            draft_token_ids = run_draft()
        else:
            draft_token_ids = run_draft()
            self._update_full_graph_params_if_needed(
                forward_context, num_input_tokens, multi_steps_attn_metadata
            )
    return draft_token_ids


def _run_merged_draft(
    self,
    num_input_tokens,
    batch_size,
    token_indices_to_sample,
    target_positions,
    inputs_embeds,
    multi_steps_attn_metadata,
    num_tokens,
    is_prefill=None,
) -> torch.Tensor:
    # The lifecycle of `input_ids`, `positions`, `hidden_states` runs through all
    # speculative tokens' proposings. `model_input_ids`, `model_positions` and
    # `model_hidden_states` represent the speculative model inputs.
    model_input_ids = self.input_ids[:num_input_tokens]
    model_positions = self._get_positions(num_input_tokens)

    if self.method == "dflash":
        model_kwargs = self.build_model_inputs_first_pass(num_input_tokens)
    else:
        model_kwargs = {
            "input_ids": model_input_ids,
            "positions": model_positions,
            "inputs_embeds": inputs_embeds,
        }

        if self.pass_hidden_states_to_model:
            model_hidden_states = self.hidden_states[:num_input_tokens]
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(
                model_hidden_states, model_positions
            )
            model_kwargs["hidden_states"] = model_hidden_states
            if self.method == "mtp":
                model_kwargs["positions"] = model_positions

    # step 0
    draft_model = getattr(self.model, "model", None)
    if (
        self._share_mtp_indices
        and draft_model is not None
        and hasattr(draft_model, "set_skip_topk")
    ):
        draft_model.set_skip_topk(False)

    ret_hidden_states = self.model(**model_kwargs)
    if not self.model_returns_tuple():
        last_hidden_states = ret_hidden_states
        hidden_states = last_hidden_states
    else:
        last_hidden_states, hidden_states = ret_hidden_states

    # step 1+ skip indexer
    draft_model = getattr(self.model, "model", None)
    if (
        self._share_mtp_indices
        and draft_model is not None
        and hasattr(draft_model, "set_skip_topk")
    ):
        draft_model.set_skip_topk(True)

    if self.method != "dflash":
        last_hidden_states, model_positions, hidden_states = (
            self.maybe_all_gather_and_unpad(
                last_hidden_states, model_positions, hidden_states
            )
        )

    num_indices = token_indices_to_sample.shape[0]
    if lmhead_tp_enable():
        max_num_reqs_across_dp = (
            self.vllm_config.scheduler_config.max_num_seqs
            * self.runner.uniform_decode_query_len
        )
        # It is necessary to evaluate the case where num_indices becomes large
        # in the context of the dummy‑run accompaniment of p‑eagle.
        if num_indices > max_num_reqs_across_dp:
            ori_token_indices_to_sample = token_indices_to_sample
        else:
            ori_token_indices_to_sample = None

    pcp_manager = getattr(self.runner, "pcp_manager", None)
    if pcp_manager is not None and self.pcp_size > 1:
        # remove graph padding before all_gather
        hidden_states = pcp_manager.get_restore_hidden_states(
            hidden_states,
            num_input_tokens=num_input_tokens,
        )
        if self.method == "mtp":
            last_hidden_states = hidden_states
        else:
            # eagle and eagle3 need allgather last_hidden_states.
            last_hidden_states = pcp_manager.get_restore_hidden_states(
                last_hidden_states,
                num_input_tokens=num_input_tokens,
            )

    if lmhead_tp_enable():
        token_indices_to_sample = nn.functional.pad(
            token_indices_to_sample, (0, max_num_reqs_across_dp - num_indices)
        )

    sample_hidden_states = last_hidden_states[token_indices_to_sample]

    if get_ascend_config().enable_reduce_sample:
        if self.method in ("eagle3", "dflash", "mtp"):
            draft_token_ids = self.compute_draft_token_ids(sample_hidden_states)
            if lmhead_tp_enable():
                draft_token_ids, token_indices_to_sample = (
                    self._align_tensor_and_indices(
                        draft_token_ids,
                        num_indices,
                        token_indices_to_sample,
                        ori_token_indices_to_sample,
                        is_logits=False,
                    )
                )
        else:
            logits = self.model.compute_logits(sample_hidden_states)
            if lmhead_tp_enable():
                logits = get_lmhead_tp_group().all_to_all(logits)
            else:
                logits = self.model.model.logits_processor._gather_logits(logits)
            if lmhead_tp_enable():
                logits, token_indices_to_sample = self._align_tensor_and_indices(
                    logits,
                    num_indices,
                    token_indices_to_sample,
                    ori_token_indices_to_sample,
                    is_logits=True,
                )
            draft_token_ids = logits.argmax(dim=-1)
    else:
        logits = self.model.compute_logits(sample_hidden_states)
        if lmhead_tp_enable():
            logits, token_indices_to_sample = self._align_tensor_and_indices(
                logits,
                num_indices,
                token_indices_to_sample,
                ori_token_indices_to_sample,
                is_logits=True,
            )
        draft_token_ids = logits.argmax(dim=-1)

    # Early exit if there is only one draft token to be generated.
    if self.num_speculative_tokens_mtp == 1 or self.parallel_drafting:
        # [batch_size, 1]
        return draft_token_ids.view(-1, self.num_speculative_tokens_mtp)

    if self.pcp_size > 1 and is_prefill:
        draft_token_ids_list = []
        for _ in range(self.num_speculative_tokens_mtp):
            draft_token_ids_list.append(draft_token_ids)
        return torch.stack(draft_token_ids_list, dim=1)

    # The logits are split and then merged only when lmhead_tp_enable() is enabled.
    # As a result, the batch size length becomes the actual length 32.
    # However, when lmhead_tp_enable() is disabled, the batch size uses the length after padding.
    # To decouple the scenarios, a judgment is required.
    # That is, the batch size needs to be modified only when lmhead_tp_enable() is enabled.
    if lmhead_tp_enable() and self.method == "mtp":
        batch_size = draft_token_ids.shape[0]

    # Generate the remaining draft tokens.
    draft_token_ids_tensor = torch.zeros(
        (self.num_speculative_tokens_mtp, *draft_token_ids.shape),
        dtype=draft_token_ids.dtype,
        device=self.device,
    )
    draft_token_ids_tensor[0] = draft_token_ids
    if self.uses_mrope:
        positions = self.mrope_positions[:, token_indices_to_sample]
    else:
        positions = self.positions[token_indices_to_sample]
    hidden_states = hidden_states[token_indices_to_sample]
    token_indices_to_sample = self.arange[:batch_size]

    input_batch_size = (
        num_input_tokens
        if (self.method == "mtp" or self.use_cuda_graph)
        else batch_size
    )

    forward_context = get_forward_context()
    _EXTRA_CTX.num_tokens = input_batch_size
    _EXTRA_CTX.num_accept_tokens = batch_size

    for draft_index in range(self.num_speculative_tokens_mtp - 1):
        # Reset MOE layer index for each draft step iteration
        forward_context = get_forward_context()
        if forward_context is not None:
            forward_context.moe_layer_index = 0

        # Update the inputs.
        # cast to int32 is crucial when eagle model is compiled.
        # tensor.argmax() returns int64 by default.
        input_ids = draft_token_ids_tensor[draft_index]
        positions += 1

        # NOTE(woosuk): We should handle the case where the draft model
        # generates tokens beyond the max model length. Since it is complex
        # to remove such requests from the batch, we keep them in the batch
        # but adjust the position ids and slot mappings to avoid the
        # out-of-range access during the model execution. The draft tokens
        # generated with this adjustment should be ignored.
        if self.uses_mrope:
            exceeds_max_model_len = (
                positions[0] >= self.vllm_config.model_config.max_model_len
            )
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(
                exceeds_max_model_len.unsqueeze(0),
                torch.zeros_like(positions),
                positions,
            )
        else:
            exceeds_max_model_len = (
                positions >= self.vllm_config.model_config.max_model_len
            )
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

        # copy inputs to buffer for cudagraph
        self.input_ids[:batch_size] = input_ids
        self._set_positions(batch_size, clamped_positions)
        self.hidden_states[:batch_size] = hidden_states.view(batch_size, -1)
        if self.supports_mm_inputs:
            self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

            input_ids = self.input_ids[:input_batch_size]
            inputs_embeds = self.inputs_embeds[:input_batch_size]
        else:
            input_ids = self.input_ids[:input_batch_size]
            inputs_embeds = None

        # Run the model.

        # The lifecycle of `input_ids`, `positions`, `hidden_states` runs through all
        # speculative tokens' proposings. `model_input_ids`, `model_positions` and
        # `model_hidden_states` represent the speculative model inputs.
        model_input_ids = self.input_ids[:input_batch_size]
        model_positions = self._get_positions(input_batch_size)
        model_hidden_states = self.hidden_states[:input_batch_size]

        model_hidden_states, model_positions = self.maybe_pad_and_reduce(
            model_hidden_states, model_positions
        )

        forward_context.attn_metadata = (
            multi_steps_attn_metadata[draft_index + 1]
            if multi_steps_attn_metadata
            else None
        )

        model_kwargs = {
            "input_ids": model_input_ids,
            "positions": model_positions,
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = model_hidden_states

        ret_hidden_states = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

        last_hidden_states, model_positions, hidden_states = (
            self.maybe_all_gather_and_unpad(
                last_hidden_states, model_positions, hidden_states
            )
        )

        num_indices = token_indices_to_sample.shape[0]
        if lmhead_tp_enable():
            max_num_reqs_across_dp = (
                self.vllm_config.scheduler_config.max_num_seqs
                * self.runner.uniform_decode_query_len
            )
            token_indices_to_sample = nn.functional.pad(
                token_indices_to_sample,
                (0, max_num_reqs_across_dp - num_indices),
            )

        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        if get_ascend_config().enable_reduce_sample:
            if self.method in ("eagle3", "dflash", "mtp"):
                draft_token_ids = self.compute_draft_token_ids(sample_hidden_states)
                if lmhead_tp_enable() and num_indices < draft_token_ids.shape[0]:
                    draft_token_ids = draft_token_ids[:num_indices]
                    token_indices_to_sample = token_indices_to_sample[:num_indices]
            else:
                logits = self.model.compute_logits(sample_hidden_states)
                if lmhead_tp_enable():
                    logits = get_lmhead_tp_group().all_to_all(logits)
                else:
                    logits = self.model.model.logits_processor._gather_logits(logits)
                if lmhead_tp_enable() and num_indices < logits.shape[0]:
                    logits = logits[:num_indices]
                    token_indices_to_sample = token_indices_to_sample[:num_indices]
                draft_token_ids = logits.argmax(dim=-1)
        else:
            logits = self.model.compute_logits(sample_hidden_states)
            if lmhead_tp_enable() and num_indices < logits.shape[0]:
                logits = logits[:num_indices]
                token_indices_to_sample = token_indices_to_sample[:num_indices]
            draft_token_ids = logits.argmax(dim=-1)

        # TODO(wenlong): get more than one token for tree attention
        hidden_states = hidden_states[:batch_size]
        draft_token_ids_tensor[draft_index + 1] = draft_token_ids

    # [batch_size, num_speculative_tokens]
    draft_token_ids = draft_token_ids_tensor.swapaxes(0, 1)
    return draft_token_ids


@when_imported("vllm_ascend.spec_decode.llm_base_proposer")
def _patch(mod):
    if hasattr(mod.AscendSpecDecodeBaseProposer, "__init__"):
        _patched = FunctionType(
            __init__.__code__,
            vars(mod),
            name=__init__.__name__,
            argdefs=__init__.__defaults__,
            closure=__init__.__closure__,
        )
        _patched.__annotations__ = __init__.__annotations__
        _patched.__kwdefaults__ = __init__.__kwdefaults__
        patch_or_inject(mod.AscendSpecDecodeBaseProposer, "__init__", _patched)
        logger.info("UCM GSA AscendSpecDecodeBaseProposer.__init__ patch applied")

    if hasattr(mod.AscendSpecDecodeBaseProposer, "dummy_run"):
        _patched = FunctionType(
            dummy_run.__code__,
            vars(mod),
            name=dummy_run.__name__,
            argdefs=dummy_run.__defaults__,
            closure=dummy_run.__closure__,
        )
        _patched.__annotations__ = dummy_run.__annotations__
        _patched.__kwdefaults__ = dummy_run.__kwdefaults__
        patch_or_inject(mod.AscendSpecDecodeBaseProposer, "dummy_run", _patched)
        logger.info("UCM GSA AscendSpecDecodeBaseProposer.dummy_run patch applied")

    if hasattr(mod.AscendSpecDecodeBaseProposer, "_propose"):
        _patched = FunctionType(
            _propose.__code__,
            vars(mod),
            name=_propose.__name__,
            argdefs=_propose.__defaults__,
            closure=_propose.__closure__,
        )
        _patched.__annotations__ = _propose.__annotations__
        _patched.__kwdefaults__ = _propose.__kwdefaults__
        patch_or_inject(mod.AscendSpecDecodeBaseProposer, "_propose", _patched)
        logger.info("UCM GSA AscendSpecDecodeBaseProposer._propose patch applied")

    if hasattr(mod.AscendSpecDecodeBaseProposer, "_run_merged_draft"):
        _patched = FunctionType(
            _run_merged_draft.__code__,
            vars(mod),
            name=_run_merged_draft.__name__,
            argdefs=_run_merged_draft.__defaults__,
            closure=_run_merged_draft.__closure__,
        )
        _patched.__annotations__ = _run_merged_draft.__annotations__
        _patched.__kwdefaults__ = _run_merged_draft.__kwdefaults__
        patch_or_inject(mod.AscendSpecDecodeBaseProposer, "_run_merged_draft", _patched)
        logger.info(
            "UCM GSA AscendSpecDecodeBaseProposer._run_merged_draft patch applied"
        )
