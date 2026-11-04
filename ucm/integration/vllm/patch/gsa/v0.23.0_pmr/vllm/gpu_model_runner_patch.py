"""GSA GPUModelRunner patches for vLLM 0.23.0."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def __init__(
    self,
    vllm_config: VllmConfig,
    device: torch.device,
):
    from ucm.sparse.utils import IS_INDEXER_OFFLOAD, IS_OPEN_GSA, PTOPK_PREFETCH_ENABLE

    self.vllm_config = vllm_config
    self.model_config = vllm_config.model_config
    self.cache_config = vllm_config.cache_config
    self.offload_config = vllm_config.offload_config
    self.compilation_config = vllm_config.compilation_config
    self.lora_config = vllm_config.lora_config
    self.load_config = vllm_config.load_config
    self.parallel_config = vllm_config.parallel_config
    self.scheduler_config = vllm_config.scheduler_config
    self.speculative_config = vllm_config.speculative_config
    self.observability_config = vllm_config.observability_config

    model_config = self.model_config
    cache_config = self.cache_config
    scheduler_config = self.scheduler_config
    parallel_config = self.parallel_config
    self.device = device
    self.pin_memory = is_pin_memory_available()
    self.dtype = self.model_config.dtype

    self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
        cache_config.cache_dtype, self.model_config
    )

    self.is_pooling_model = model_config.runner_type == "pooling"
    self.enable_prompt_embeds = model_config.enable_prompt_embeds
    self.is_multimodal_raw_input_only_model = (
        model_config.is_multimodal_raw_input_only_model
    )
    # These will be overridden in load_model()
    self.is_multimodal_pruning_enabled = False
    self.requires_sequential_video_encoding = False
    # Set to True after init_routed_experts_capturer() completes.
    # Prevents routed experts code from running during profiling/dummy run.
    self.routed_experts_initialized = False
    self.max_model_len = model_config.max_model_len

    # Always set to false after the first forward pass
    self.calculate_kv_scales = self.cache_config.calculate_kv_scales
    self.dcp_world_size = self.parallel_config.decode_context_parallel_size
    self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
    self.max_num_tokens = scheduler_config.max_num_batched_tokens
    self.max_num_reqs = scheduler_config.max_num_seqs

    # Broadcast PP output for external_launcher (torchrun)
    # to make sure we are synced across pp ranks
    # TODO: Support overlapping micro-batches
    # https://github.com/vllm-project/vllm/issues/18019
    self.broadcast_pp_output = (
        self.parallel_config.distributed_executor_backend == "external_launcher"
        and len(get_pp_group().ranks) > 1
    )

    # Model-related.
    self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
    self.inputs_embeds_size = model_config.get_inputs_embeds_size()
    # Only relevant for models using ALiBi (e.g, MPT)
    self.use_alibi = model_config.uses_alibi

    self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
    self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm

    # Multi-modal data support
    self.mm_registry = MULTIMODAL_REGISTRY
    self.uses_mrope = model_config.uses_mrope
    self.uses_xdrope_dim = model_config.uses_xdrope_dim
    self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(model_config)

    if self.model_config.is_encoder_decoder:
        # Maximum length of the encoder input, only for encoder-decoder
        # models.
        self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
    else:
        self.max_encoder_len = 0

    # Async scheduling
    self.use_async_scheduling = self.scheduler_config.async_scheduling

    # Sampler
    self.sampler = Sampler(
        logprobs_mode=self.model_config.logprobs_mode,
        use_fp64_gumbel=self.model_config.use_fp64_gumbel,
    )

    self.eplb_state: EplbState | None = None
    self._moe_model: MixtureOfExperts | None = None
    # NOTE(yongji): flag to temporarily disable EPLB during scaling up/down
    self.eep_eplb_suppressed = False
    """
    State of the expert parallelism load balancer.

    Will be lazily initialized when the model is loaded.
    """

    # Lazy initializations
    # self.model: nn.Module  # Set after load_model
    # Initialize in initialize_kv_cache
    self.kv_caches: list[torch.Tensor] = []
    # Initialize in initialize_kv_cache_tensors
    self.cross_layers_kv_cache: torch.Tensor | None = None
    self.cross_layers_attn_backend: type[AttentionBackend] | None = None
    # indexes: [kv_cache_group_id][attn_group]
    self.attn_groups: list[list[AttentionGroup]] = []
    # self.kv_cache_config: KVCacheConfig

    # mm_hash ->  encoder_output
    self.encoder_cache: dict[str, torch.Tensor] = {}
    self.late_interaction_runner = LateInteractionRunner()

    # Encoder CUDA graph manager (initialized after model load if enabled)
    self.encoder_cudagraph_manager: EncoderCudaGraphManager | None = None

    self.use_aux_hidden_state_outputs = False
    # Set up speculative decoding.
    # NOTE(Jiayi): currently we put the entire draft model on
    # the last PP rank. This is not ideal if there are many
    # layers in the draft model.
    if self.speculative_config and get_pp_group().is_last_rank:
        self.drafter: (
            NgramProposer  # noqa: F823
            | NgramProposerGPU
            | SuffixDecodingProposer
            | EagleProposer
            | DFlashProposer
            | DraftModelProposer
            | MedusaProposer
            | ExtractHiddenStatesProposer
            | Gemma4Proposer
            | Step3p5MTPProposer
        )
        if self.speculative_config.method == "custom_class":
            self.drafter = create_custom_proposer(  # type: ignore[assignment]
                self.vllm_config
            )
        elif self.speculative_config.method == "ngram":
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            self.drafter = NgramProposer(self.vllm_config)
        elif self.speculative_config.uses_draft_model():
            self.drafter = DraftModelProposer(
                vllm_config=self.vllm_config,
                device=self.device,
                runner=self,
            )
        elif self.speculative_config.use_ngram_gpu():
            self.drafter = NgramProposerGPU(self.vllm_config, self.device, self)
            self.num_tokens_no_spec_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            self.token_ids_gpu_tensor = torch.zeros(
                self.max_num_reqs,
                self.max_model_len,
                dtype=torch.int32,
                device=device,
            )
            self._ngram_pinned_idx_buf = torch.zeros(
                self.max_num_reqs, dtype=torch.long, pin_memory=True
            )
            self._ngram_pinned_val_buf = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, pin_memory=True
            )
        elif self.speculative_config.use_gemma4_mtp():
            self.drafter = Gemma4Proposer(self.vllm_config, self.device, self)
        elif self.speculative_config.use_step3p5_mtp():
            self.drafter = Step3p5MTPProposer(self.vllm_config, self.device, self)
        elif self.speculative_config.use_dflash():
            self.drafter = DFlashProposer(self.vllm_config, self.device, self)
            self.use_aux_hidden_state_outputs = True
        elif self.speculative_config.method == "suffix":
            self.drafter = SuffixDecodingProposer(self.vllm_config)
        elif self.speculative_config.use_eagle():
            self.drafter = EagleProposer(self.vllm_config, self.device, self)
            if self.speculative_config.method == "eagle3":
                self.use_aux_hidden_state_outputs = (
                    self.drafter.eagle3_use_aux_hidden_state
                )
        elif self.speculative_config.method == "medusa":
            self.drafter = MedusaProposer(
                vllm_config=self.vllm_config, device=self.device
            )
        elif self.speculative_config.method == "extract_hidden_states":
            self.drafter = ExtractHiddenStatesProposer(
                vllm_config=self.vllm_config, device=self.device
            )
            self.use_aux_hidden_state_outputs = True
        else:
            raise ValueError(
                "Unknown speculative decoding method: "
                f"{self.speculative_config.method}"
            )
        self.rejection_sampler = RejectionSampler(
            self.sampler, self.speculative_config, self.device
        )

    self.num_spec_tokens = 0
    self.valid_sampled_token_count_gpu: torch.Tensor | None = None
    if self.speculative_config:
        self.num_spec_tokens = self.speculative_config.num_speculative_tokens
        draft_config = self.speculative_config.draft_model_config
        if draft_config is not None and draft_config.max_model_len is not None:
            self.effective_drafter_max_model_len = draft_config.max_model_len
        else:
            self.effective_drafter_max_model_len = self.max_model_len
    self.use_async_spec_decode = self.use_async_scheduling and self.num_spec_tokens > 0

    # Request states.
    self.requests: dict[str, CachedRequestState] = {}
    # NOTE(rob): num_prompt_logprobs only includes reqs
    # that are currently in the prefill phase.
    self.num_prompt_logprobs: dict[str, int] = {}

    # Input Batch
    # NOTE(Chen): Ideally, we should initialize the input batch inside
    # `initialize_kv_cache` based on the kv cache config. However, as in
    # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
    # reasons, we have to initialize the input batch before `load_model`,
    # quantization + weight offloading will fail otherwise. As a temporary
    # solution, we initialize the input batch here, and re-initialize it
    # in `initialize_kv_cache` if the block_sizes here is different from
    # the block_sizes in the kv cache config.
    logits_processors = model_config.logits_processors
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
        tuple(logits_processors) if logits_processors is not None else ()
    )
    placeholder_block_size = (
        self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE
    )
    self._init_block_sizes = [placeholder_block_size]
    self._init_kernel_block_sizes = [placeholder_block_size]
    self.input_batch = InputBatch(
        max_num_reqs=self.max_num_reqs,
        # We need to use the encoder length for encoder-decoder
        # because of KV cache for cross-attention.
        max_model_len=max(self.max_model_len, self.max_encoder_len),
        max_num_batched_tokens=self.max_num_tokens,
        device=self.device,
        pin_memory=self.pin_memory,
        vocab_size=self.model_config.get_vocab_size(),
        block_sizes=[placeholder_block_size],
        kernel_block_sizes=[placeholder_block_size],
        num_spec_tokens=self.num_spec_tokens,
        logitsprocs=build_logitsprocs(
            self.vllm_config,
            self.device,
            self.pin_memory,
            self.is_pooling_model,
            custom_logitsprocs,
        ),
        # We currently don't know whether a particular custom logits processor
        # uses output token ids so we set this conservatively.
        # ThinkingTokenBudgetLogitsProcessor also needs output token ids to
        # correctly track think start/end token sequences in async scheduling.
        logitsprocs_need_output_token_ids=bool(custom_logitsprocs)
        or self.vllm_config.reasoning_config is not None,
        is_pooling_model=self.is_pooling_model,
        cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        reasoning_config=self.vllm_config.reasoning_config,
    )

    # Separate cuda stream for overlapping transfer of sampled token ids from
    # GPU to CPU when async scheduling is enabled.
    self.async_output_copy_stream: torch.cuda.Stream | None = None
    # cuda event to synchronize use of reused CPU tensors between steps
    # when async scheduling is enabled.
    self.prepare_inputs_event: torch.Event | None = None
    if self.use_async_scheduling:
        self.async_output_copy_stream = torch.cuda.Stream()
        self.prepare_inputs_event = torch.Event()

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

    # Cache the device properties.
    self._init_device_properties()

    # Encoder timing registry for observability
    self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
    self._encoder_timing_lock = threading.Lock()

    # Persistent buffers for CUDA graphs.
    self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)

    self.sparse_positions = torch.zeros(
        self.max_num_tokens, dtype=torch.int64, device=self.device
    )

    self.positions = torch.zeros(
        self.max_num_tokens, dtype=torch.int64, device=self.device
    )
    self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, dtype=torch.int32)
    self.seq_lens = torch.zeros(
        self.max_num_reqs, dtype=torch.int32, device=self.device
    )
    self.optimistic_seq_lens_cpu = torch.zeros(
        self.max_num_reqs, dtype=torch.int32, pin_memory=self.pin_memory
    )
    self.num_computed_tokens = torch.zeros(
        self.max_num_reqs, dtype=torch.int32, device=self.device
    )
    self.prev_num_draft_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
    self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
    # Maps current batch position -> previous batch position (-1 for new reqs)
    self.prev_positions = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
    self.num_scheduled_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)

    self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
    if self.dcp_world_size > 1:
        self.dcp_local_seq_lens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
    # Because inputs_embeds may be bfloat16 and we don't need a numpy
    # version of this tensor, avoid a RuntimeError by not creating a
    # numpy buffer.
    self.inputs_embeds = self._make_buffer(
        self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
    )
    self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
    self.discard_request_mask = self._make_buffer(self.max_num_reqs, dtype=torch.bool)
    self.num_decode_draft_tokens = self._make_buffer(
        self.max_num_reqs, dtype=torch.int32
    )
    self.num_accepted_tokens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)

    ######[GSA] add attention metadata info
    if PTOPK_PREFETCH_ENABLE and IS_OPEN_GSA:
        self.dsa_block_table = None
        self.dsa_seq_len = None
        self.dsa_slot_mapping = None

        self.gsa_prefetch_index = None
        self.gsa_prefetch_topk = None
        self.gsa_swap_index = None
        self.gsa_swap_topk = None
        self.gsa_warmup_index = None
        self.gsa_warmup_topk = None
        self.gsa_prefill_query = None
        self.gsa_topk_win_tmp = None

        #### indexer
        if IS_INDEXER_OFFLOAD:
            self.indexer_last_query = None
            self.indexer_last_weight = None
            self.indexer_last_q_index = None

    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        # NOTE: `mrope_positions` is implemented with one additional dummy
        # position on purpose to make it non-contiguous so that it can work
        # with torch compile.
        # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

        # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
        # the modality of inputs. For text-only inputs, each dimension has
        # identical position IDs, making M-RoPE functionally equivalent to
        # 1D-RoPE.
        # See page 5 of https://arxiv.org/abs/2409.12191
        self.mrope_positions = self._make_buffer(
            (3, self.max_num_tokens + 1), dtype=torch.int64
        )

    # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
    if self.uses_xdrope_dim > 0:
        # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
        self.xdrope_positions = self._make_buffer(
            (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
        )

    # None in the first PP rank. The rest are set after load_model.
    self.intermediate_tensors: IntermediateTensors | None = None

    # OPTIMIZATION: Cache the arange tensors rather than creating them
    # every step. Keep in int64 to avoid overflow with long context.
    # - arange_np: immutable [0, 1, 2, ...] used as source for batched computation
    # - query_pos: CpuGpuBuffer for the computed batched arange result
    arange_size = max(self.max_num_reqs + 1, self.max_num_tokens)
    self.arange_np = np.arange(arange_size, dtype=np.int64)

    self.sparse_query_pos = self._make_buffer(arange_size, dtype=torch.int64)

    self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
    self._arange_scratch = np.empty(arange_size, dtype=np.int64)

    # Layer pairings for cross-layer KV sharing.
    # If an Attention layer `layer_name` is in the keys of this dict, it
    # means this layer will perform attention using the keys and values
    # from the KV cache of `shared_kv_cache_layers[layer_name]`.
    self.shared_kv_cache_layers: dict[str, str] = {}
    self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

    self.kv_sharing_fast_prefill_logits_indices = None
    if self.cache_config.kv_sharing_fast_prefill:
        self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=self.device
        )

    self.uniform_decode_query_len = 1 + self.num_spec_tokens

    # Cudagraph dispatcher for runtime cudagraph dispatching.
    self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

    self.mm_budget = (
        MultiModalBudget(self.vllm_config, self.mm_registry)
        if self.supports_mm_inputs
        else None
    )

    self.reorder_batch_threshold: int | None = None

    # Attention layers that are only in the KVCacheConfig of the runner
    # (e.g., KV sharing, encoder-only attention), but not in the
    # KVCacheConfig of the scheduler.
    self.runner_only_attn_layers: set[str] = set()

    # Cached outputs.
    self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
    self._draft_probs: torch.Tensor | None = None
    self._draft_prob_req_ids: list[str] | None = None
    # N-gram GPU path: async D2H buffer/event for per-request valid draft counts.
    self._num_valid_draft_tokens: torch.Tensor | None = None
    self._num_valid_draft_tokens_cpu: torch.Tensor | None = None
    self._num_valid_draft_tokens_event: torch.cuda.Event | None = None
    self._num_valid_draft_tokens_copy_stream: torch.cuda.Stream | None = None
    if self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
        self._num_valid_draft_tokens_cpu = torch.empty(
            self.max_num_reqs, dtype=torch.int32, pin_memory=self.pin_memory
        )
        self._num_valid_draft_tokens_event = torch.cuda.Event()
        self._num_valid_draft_tokens_copy_stream = torch.cuda.Stream()

    self._draft_token_req_ids: list[str] | None = None
    self.transfer_event = torch.Event()
    self.sampled_token_ids_pinned_cpu = torch.empty(
        (self.max_num_reqs, 1),
        dtype=torch.int64,
        device="cpu",
        pin_memory=self.pin_memory,
    )

    # Pre-allocated tensor for copying valid sampled token counts to CPU,
    # with dedicated stream for overlapping and event for coordination.
    self.valid_sampled_token_count_event: torch.Event | None = None
    self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
    # We also copy the drafted tokens to the CPU asynchronously,
    # in case we need them for structured outputs.
    self.draft_token_ids_event: torch.Event | None = None
    self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
    self.valid_sampled_token_count_cpu: torch.Tensor | None = None
    self.draft_token_ids_cpu: torch.Tensor | None = None
    self.num_accepted_tokens_event: torch.Event | None = None

    from ucm.utils import Config

    ucm_config = Config(self.vllm_config.kv_transfer_config)
    self.launch_config = ucm_config.get_config()
    pmr_tree_config = self.launch_config.get("pmr_config", None)
    use_pmr = False
    if pmr_tree_config:
        use_pmr = pmr_tree_config["use_pmr"]

    if self.num_spec_tokens:
        self.draft_token_ids_event = torch.Event()
        self.num_accepted_tokens_event = torch.Event()
        self.draft_token_ids_copy_stream = torch.cuda.Stream()

        if use_pmr and self.speculative_config.method != "suffix":
            self.num_spec_tokens_mtp = (
                self.num_spec_tokens - pmr_tree_config["num_speculative_tokens"]
            )
        else:
            self.num_spec_tokens_mtp = self.num_spec_tokens

        self.draft_token_ids_cpu = torch.empty(
            (self.max_num_reqs, self.num_spec_tokens_mtp),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        if self.use_async_scheduling:
            self.valid_sampled_token_count_event = torch.Event()
            self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
            self.valid_sampled_token_count_cpu = torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                device="cpu",
                pin_memory=self.pin_memory,
            )

    # Model weight offloader
    # Make sure this is called before any get_offloader call
    set_offloader(create_offloader(self.offload_config))

    # Ephemeral state transferred between execute_model() and sample_tokens().
    self.execute_model_state: ExecuteModelState | None = None
    self.kv_connector_output: KVConnectorOutput | None = None
    self.mamba_state_idx: dict[str, int] = {}
    self._mamba_bufs: mamba_utils.MambaBuffers | None = None
    self.mamba_prev_last_scheduled_idx: CpuGpuBuffer | None = None
    if self.cache_config.mamba_cache_mode == "all" and self.num_spec_tokens > 0:
        self.mamba_prev_last_scheduled_idx = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
    self.layerwise_nvtx_hooks_registered = False


def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
    """Update the cached states and the persistent batch with the scheduler
    output.

    The updated states are used by the `_prepare_inputs` function to create
    the input GPU tensors for the model.

    The SamplingMetadata is updated and copied to the GPU if there is a
    new/resumed/paused/finished request in the batch.
    """
    from ucm.sparse.base import INVALID_SLOT

    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        self.ucm_sparse_request_finished_in_worker(req_id)
        self.requests.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)
    self.late_interaction_runner.on_requests_finished(scheduler_output.finished_req_ids)
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)

    # Zero GPU memory for freshly allocated cache blocks to prevent
    # stale NaN/data from corrupting attention or SSM computation.
    if scheduler_output.new_block_ids_to_zero:
        self._zero_block_ids(scheduler_output.new_block_ids_to_zero)

    # Free the cached encoder outputs.
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        self.encoder_cache.pop(mm_hash, None)

    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
    # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
    # apart from the forced-preemption case in reset_prefix_cache. And in
    # that case we include the resumed_req_ids in the unscheduled set so
    # that they get cleared from the persistent batch before being re-scheduled
    # in the normal resumed request path.
    unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
    # NOTE(woosuk): The persistent batch optimization assumes that
    # consecutive batches contain mostly the same requests. If batches
    # have low request overlap (e.g., alternating between two distinct
    # sets of requests), this optimization becomes very inefficient.
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)

    is_ngram_gpu = (
        self.speculative_config is not None and self.speculative_config.use_ngram_gpu()
    )
    if is_ngram_gpu:
        ngram_gpu_new_reqs: list[CachedRequestState] = []

    reqs_to_add: list[CachedRequestState] = []
    deferred_spec_decode_corrections = []

    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        if req_id in self.requests:
            # For streaming case only.
            req_state = self._update_streaming_request(req_id, new_req_data)
            reqs_to_add.append(req_state)
            continue

        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params

        if (
            sampling_params
            and sampling_params.sampling_type == SamplingType.RANDOM_SEED
        ):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        if self.is_pooling_model:
            assert pooling_params is not None
            task = pooling_params.task
            assert task is not None, "You did not set `task` in the API"

            model = cast(VllmModelForPooling, self.get_model())
            to_update = model.pooler.get_pooling_updates(task)
            to_update.apply(pooling_params)

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt_embeds=new_req_data.prompt_embeds,
            prompt_is_token_ids=new_req_data.prompt_is_token_ids,
            mm_features=new_req_data.mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )
        self.requests[req_id] = req_state
        self.late_interaction_runner.register_request(req_id, pooling_params)

        if sampling_params and sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = (
                self.input_batch.vocab_size
                if sampling_params.prompt_logprobs == -1
                else sampling_params.prompt_logprobs
            )

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._init_xdrope_positions(req_state)

        reqs_to_add.append(req_state)
        # Track new requests for ngram_gpu full tensor copy
        if is_ngram_gpu:
            ngram_gpu_new_reqs.append(req_state)

    # Update the states of the running/resumed requests.
    is_last_rank = get_pp_group().is_last_rank
    req_data = scheduler_output.scheduled_cached_reqs
    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

    # Save scheduler-allocated spec lengths before trimming so
    # prev_num_draft_len keeps the optimistic count for rejection correction.
    original_num_spec_per_req: dict[str, int] = {}
    if self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
        for req_id, toks in scheduled_spec_tokens.items():
            original_num_spec_per_req[req_id] = len(toks)
        update_scheduler_for_invalid_drafts(
            self._num_valid_draft_tokens_event,
            self._num_valid_draft_tokens_cpu,
            scheduler_output,
            self.input_batch.req_id_to_index,
        )
    if self.use_async_spec_decode:
        self.prev_num_draft_tokens.np.fill(0)

    req_sparsed_slots = scheduler_output.req_sparsed_slots
    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_id in req_data.resumed_req_ids
        num_output_tokens = req_data.num_output_tokens[i]
        req_index = self.input_batch.req_id_to_index.get(req_id)
        is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT

        if req_state.prev_num_draft_len and self.use_async_scheduling:
            # prev_num_draft_len is used in async scheduling mode with
            # spec decode. it indicates if need to update num_computed_tokens
            # of the request. for example:
            # first step: num_computed_tokens = 0, spec_tokens = [],
            # prev_num_draft_len = 0.
            # second step: num_computed_tokens = 100(prompt length),
            # spec_tokens = [a,b], prev_num_draft_len = 0.
            # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
            # prev_num_draft_len = 2.
            # num_computed_tokens in first step and second step doesn't contain
            # the spec tokens length, but in third step it contains the
            # spec tokens length. we only need to update num_computed_tokens
            # when prev_num_draft_len > 0.
            if req_index is None:
                req_state.prev_num_draft_len = 0
            else:
                # Optimistically assume all accepted; queue up a correction
                # to be called after the model forward to preserve async
                # scheduling. Corrected on GPU in _prepare_inputs.
                optimistic_num_accepted = req_state.prev_num_draft_len
                req_state.output_token_ids.extend([-1] * optimistic_num_accepted)

                deferred_spec_decode_corrections.append(
                    (req_id, optimistic_num_accepted, req_state)
                )

                prev_req_index = (
                    self.input_batch.prev_req_id_to_index.get(req_id)
                    if self.input_batch.prev_req_id_to_index
                    else None
                )
                if prev_req_index is not None:
                    self.prev_num_draft_tokens.np[prev_req_index] = (
                        optimistic_num_accepted
                    )

                if is_ngram_gpu and optimistic_num_accepted > 0:
                    self.input_batch.num_tokens_no_spec[
                        req_index
                    ] += optimistic_num_accepted

        # Update the cached states.
        req_state.num_computed_tokens = num_computed_tokens

        if not is_last_rank:
            if not req_data.new_token_ids:
                # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                new_token_ids: list[int] = []
            else:
                # Non-async scheduling with PP: The scheduler sends
                # sampled token ids back because there's no direct communication
                # between the first-stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
        elif num_output_tokens < len(req_state.output_token_ids):
            # Some output tokens were discarded due to a sync-KV-load
            # failure, or output_token_ids was inflated by the optimistic
            # extend above (async spec decode). Align the cached state.
            del req_state.output_token_ids[num_output_tokens:]
            if req_index is not None:
                end_idx = (
                    self.input_batch.num_prompt_tokens[req_index] + num_output_tokens
                )
                self.input_batch.num_tokens_no_spec[req_index] = end_idx

        # Update the block IDs.
        if resumed_from_preemption or is_sparsed_request:
            if resumed_from_preemption:
                assert req_index is None
                assert new_block_ids is not None
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = new_block_ids
        else:
            if new_block_ids is not None:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)

        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.

            if self.use_async_scheduling and num_output_tokens > 0:
                # We must recover the output token ids for resumed requests in the
                # async scheduling case, so that correct input_ids are obtained.
                resumed_token_ids = req_data.all_token_ids[req_id]
                req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

            reqs_to_add.append(req_state)
            # Track resumed requests for ngram_gpu full tensor copy
            if is_ngram_gpu:
                ngram_gpu_new_reqs.append(req_state)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
        if is_sparsed_request:
            self.input_batch.block_table.reset_row(req_index)
        if new_block_ids is not None:
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # For the last rank, we don't need to update the token_ids_cpu
        # because the sampled tokens are already cached.
        if not is_last_rank:
            start_token_index = self.input_batch.num_tokens_no_spec[req_index]
            # For chunked prefill, num_computed_tokens may less
            # than num_tokens_no_spec.
            # Async scheduled PP: no new_token_ids, advance num_tokens_no_spec
            # according to num_computed_tokens.
            end_token_index = max(
                start_token_index,
                num_computed_tokens + len(new_token_ids),
            )
            if end_token_index > start_token_index:
                if new_token_ids:
                    # Add new_token_ids to token_ids_cpu.
                    num_new_tokens = end_token_index - start_token_index
                    tokens_to_append = new_token_ids[-num_new_tokens:]
                    self.input_batch.token_ids_cpu[
                        req_index, start_token_index:end_token_index
                    ] = tokens_to_append
                self.input_batch.is_token_ids[
                    req_index, start_token_index:end_token_index
                ] = True
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

        # Add spec_token_ids to token_ids_cpu.
        self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
        # Restore scheduler-side draft count after ngram trimming.
        if original_num_spec_per_req:
            orig = original_num_spec_per_req.get(req_id, 0)
            if orig != req_state.prev_num_draft_len:
                req_state.prev_num_draft_len = orig

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    for request in reqs_to_add:
        self.input_batch.add_request(request)
        self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

    # Condense the batched states if there are gaps left by removed requests
    self.input_batch.condense()
    # Allow attention backend to reorder the batch, potentially
    self._may_reorder_batch(scheduler_output)
    # Refresh batch metadata with any pending updates.
    self.input_batch.refresh_metadata()

    # Incrementally update ngram_gpu tensors after batch is stable
    if is_ngram_gpu:
        update_ngram_gpu_tensors_incremental(
            self.input_batch,
            self.token_ids_gpu_tensor,
            self.num_tokens_no_spec_gpu,
            ngram_gpu_new_reqs,
            self.device,
            _pinned_idx_buf=self._ngram_pinned_idx_buf,
            _pinned_val_buf=self._ngram_pinned_val_buf,
        )

    if deferred_spec_decode_corrections:

        def correct_spec_decode_token_counts():
            valid_sampled_token_count = self._get_valid_sampled_token_count()
            if not valid_sampled_token_count:
                return
            prev_req_id_to_index = self.input_batch.prev_req_id_to_index
            if not prev_req_id_to_index:
                return
            for (
                req_id,
                optimistic_num_accepted,
                req_state,
            ) in deferred_spec_decode_corrections:
                prev_req_index = prev_req_id_to_index.get(req_id)
                if prev_req_index is None:
                    continue
                num_accepted = valid_sampled_token_count[prev_req_index] - 1
                correction = optimistic_num_accepted - num_accepted
                req_state.num_computed_tokens -= correction
                cur_req_index = self.input_batch.req_id_to_index.get(req_id)
                if cur_req_index is None:
                    continue
                self.input_batch.num_computed_tokens_cpu[cur_req_index] -= correction
                if is_ngram_gpu and correction > 0:
                    self.input_batch.num_tokens_no_spec[cur_req_index] -= correction
                    self.num_tokens_no_spec_gpu[cur_req_index] -= correction

        return correct_spec_decode_token_counts
    else:
        return None


def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
) -> tuple[
    torch.Tensor,
    SpecDecodeMetadata | None,
]:
    """
    Returns:
        tuple[logits_indices, spec_decode_metadata]
    """
    from ucm.sparse.base import INVALID_SLOT

    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit_block_table(num_reqs)

    # Get request indices.
    # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
    # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    cu_num_tokens = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)

    # Get positions.
    positions_np = (
        self.input_batch.num_computed_tokens_cpu[req_indices]
        + self.query_pos.np[: cu_num_tokens[-1]]
    )

    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        self._calc_mrope_positions(scheduler_output)

    # Calculate XD-RoPE positions.
    # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
    if self.uses_xdrope_dim > 0:
        self._calc_xdrope_positions(scheduler_output)

    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
    token_indices_tensor = torch.from_numpy(token_indices)

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
    if self.input_batch.req_prompt_embeds:
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

    # Prepare the attention metadata.
    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
    # Note: pad query_start_loc to be non-decreasing, as kernels
    # like FlashAttention requires that
    self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
    self.query_start_loc.copy_to_gpu()
    query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

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

    # Build prev_positions mapping: current pos -> prev pos (-1 if new).
    # Used for gathering from previous iteration's GPU tensors.
    prev_req_id_to_index = self.input_batch.prev_req_id_to_index
    self._compute_prev_positions(num_reqs)

    num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
    num_tokens_np = np.array(num_tokens, dtype=np.int32)

    # Record which requests should not be sampled,
    # so that we could clear the sampled tokens before returning
    self.discard_request_mask.np[:num_reqs] = (
        self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
    )
    self.discard_request_mask.copy_to_gpu(num_reqs)

    # Sync num_accepted_tokens from CPU (set by
    # _update_states_after_model_execute for hybrid models).
    if self.num_accepted_tokens_event is not None:
        self.num_accepted_tokens_event.synchronize()
        # Async mode: condense() reordered indices, use prev_positions mapping
        if self.use_async_scheduling and prev_req_id_to_index:
            prev_idx = self.prev_positions.np[:num_reqs]
            new_mask = prev_idx < 0
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[
                    np.where(new_mask, 0, prev_idx)
                ]
            )
            self.num_accepted_tokens.np[:num_reqs][new_mask] = 1
            self.input_batch.num_accepted_tokens_cpu[:num_reqs] = (
                self.num_accepted_tokens.np[:num_reqs]
            )
        else:
            # Non-async mode: use values directly
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
        self.num_accepted_tokens.np[num_reqs:].fill(1)
        self.num_accepted_tokens.copy_to_gpu()
    else:
        self.num_accepted_tokens.np.fill(1)
        self.num_accepted_tokens.gpu.fill_(1)

    if self.mamba_prev_last_scheduled_idx is not None:
        mamba_utils.preprocess_mamba_all_specdec(
            scheduler_output,
            self.input_batch,
            self.mamba_state_idx,
            num_reqs,
            self.mamba_prev_last_scheduled_idx,
        )

    # Update num_computed_tokens on GPU. In async spec decode,
    # CPU values are optimistic (all drafts accepted). The kernel
    # corrects on GPU using the previous step's
    # valid_sampled_token_count_gpu. Otherwise, just copy from CPU.
    if (
        self.use_async_spec_decode
        and self.valid_sampled_token_count_gpu is not None
        and prev_req_id_to_index
    ):
        self.prev_positions.copy_to_gpu(num_reqs)
        self.prev_num_draft_tokens.copy_to_gpu()
        cpu_values = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].to(
            device=self.device, non_blocking=True
        )
        update_num_computed_tokens_for_batch_change(
            self.num_computed_tokens,
            self.num_accepted_tokens.gpu[:num_reqs],
            self.prev_positions.gpu[:num_reqs],
            self.valid_sampled_token_count_gpu,
            self.prev_num_draft_tokens.gpu,
            cpu_values,
        )
    else:
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )

    self.req_indices.np[:total_num_scheduled_tokens] = req_indices
    self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
    req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

    self.sparse_query_pos.np[:] = self.query_pos.np.copy()
    req_sparsed_slots = scheduler_output.req_sparsed_slots
    for req_id in self.input_batch.req_id_to_index:
        is_sparsed_request = req_sparsed_slots[req_id] != INVALID_SLOT
        req_index = self.input_batch.req_id_to_index[req_id]
        offset = (
            0 if req_index == 0 else cu_num_tokens[req_index - 1]
        )  # TODO: support MTP
        end = cu_num_tokens[req_index]
        if is_sparsed_request:
            self.sparse_query_pos.np[offset:end] = list(
                range(
                    req_sparsed_slots[req_id] - end + offset, req_sparsed_slots[req_id]
                )
            )

    self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
    self.sparse_query_pos.copy_to_gpu(total_num_scheduled_tokens)
    self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
    self.num_scheduled_tokens.copy_to_gpu(num_reqs)
    num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
    self.positions[:total_num_scheduled_tokens] = (
        self.num_computed_tokens[req_indices_gpu].to(torch.int64)
        + self.query_pos.gpu[:total_num_scheduled_tokens]
    )
    self.seq_lens[:num_reqs] = (
        self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
    )

    is_sparsed_np = np.zeros((num_reqs,), dtype=np.bool_)
    for req_id in self.input_batch.req_id_to_index:
        is_sparsed_request = scheduler_output.req_sparsed_slots[req_id] != INVALID_SLOT
        req_index = self.input_batch.req_id_to_index[req_id]
        if is_sparsed_request:
            self.seq_lens.np[req_index] = req_sparsed_slots[req_id]
            is_sparsed_np[req_index] = True

    self.seq_lens[num_reqs:].fill_(0)

    self.input_batch.block_table.compute_slot_mapping(
        num_reqs,
        self.query_start_loc.gpu[: num_reqs + 1],
        self.sparse_query_pos[:total_num_scheduled_tokens],
    )

    # Copy the tensors to the GPU.
    self._prepare_input_ids(
        scheduler_output,
        num_reqs,
        total_num_scheduled_tokens,
        cu_num_tokens,
    )

    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
            non_blocking=True,
        )
    elif self.uses_xdrope_dim > 0:
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
            non_blocking=True,
        )
    if self.use_async_spec_decode and (self.uses_mrope or self.uses_xdrope_dim > 0):
        drift = self.num_computed_tokens[req_indices_gpu].to(
            torch.int64
        ) - self.input_batch.num_computed_tokens_cpu_tensor[req_indices].to(
            device=self.device, dtype=torch.int64, non_blocking=True
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
        logits_indices = query_start_loc[1:] - 1
        spec_decode_metadata = None
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        # For chunked prefills, use -1 as mask rather than 0, as guided
        # decoding may rollback speculative tokens.
        num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
        for (
            req_id,
            draft_token_ids,
        ) in scheduler_output.scheduled_spec_decode_tokens.items():
            req_idx = self.input_batch.req_id_to_index[req_id]
            draft_len = len(draft_token_ids)
            num_draft_tokens[req_idx] = draft_len
            if (
                self.input_batch.num_computed_tokens_cpu[req_idx]
                >= self.input_batch.num_prompt_tokens[req_idx]
            ):
                num_decode_draft_tokens[req_idx] = draft_len
        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens, cu_num_tokens
        )
        logits_indices = spec_decode_metadata.logits_indices
        num_sampled_tokens = num_draft_tokens + 1
        # For DECODE only cuda graph of some attention backends (e.g., GDN).
        self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_decode_draft_tokens.copy_to_gpu()

    # Hot-Swap lora model
    if self.lora_config:
        assert (
            np.sum(num_sampled_tokens)
            <= self.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.set_active_loras(
            self.input_batch, num_scheduled_tokens, num_sampled_tokens
        )

    return (
        logits_indices,
        spec_decode_metadata,
    )


def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: IntermediateTensors | None = None,
) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
    if self.execute_model_state is not None:
        raise RuntimeError(
            "State error: sample_tokens() must be called "
            "after execute_model() returns None."
        )

    if self.routed_experts_initialized:
        self.routed_experts_capturer.clear_buffer()

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

    if has_kv_transfer_group():
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    with (
        record_function_or_nullcontext("gpu_model_runner: preprocess"),
        self.synchronize_input_prep(),
    ):
        # Update persistent batch states.
        deferred_state_corrections_fn = self._update_states(scheduler_output)

        if has_ec_transfer() and not get_ec_transfer().is_consumer:
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                return make_empty_encoder_model_runner_output(scheduler_output)

        if not num_scheduled_tokens:
            if (
                self.parallel_config.distributed_executor_backend == "external_launcher"
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
        num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

        logits_indices, spec_decode_metadata = self._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens_np,
        )

        cascade_attn_prefix_lens = None
        # Disable cascade attention when using microbatching (DBO)
        if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
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
            num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
        )

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

        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        # True if any attention backend handles KV cache update separately
        # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
        # slot_mappings must use padded dimensions to match the key/value tensors.
        has_separate_kv_update = not all(
            all(
                g.backend.forward_includes_kv_cache_update for g in self.attn_groups[id]
            )
            for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
            if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
        )
        pad_attn = cudagraph_mode == CUDAGraphMode.FULL

        if self.cache_config.mamba_cache_mode == "align":
            # preprocess_mamba reads req_state.num_computed_tokens (CPU)
            # to decide copy operations, so we must apply deferred
            # corrections before it runs.
            if deferred_state_corrections_fn:
                deferred_state_corrections_fn()
                deferred_state_corrections_fn = None
            mamba_bufs = self._get_mamba_bufs()
            mamba_utils.preprocess_mamba(
                scheduler_output,
                self.kv_cache_config,
                self.cache_config,
                self.mamba_state_idx,
                self.input_batch,
                self.requests,
                self.compilation_config.static_forward_context,
                self.model.get_mamba_state_copy_func(),
                mamba_bufs.preprocess,
            )
            # preprocess_mamba resets num_accepted_tokens_cpu to 1
            # for requests whose state was copied to a new block.
            # Re-sync to GPU so the mamba kernel reads from the
            # correct initial state slot (init_token_idx = 0).
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.copy_to_gpu(num_reqs)

            # Stage per-request inputs for the fused postprocess kernel
            # only when that kernel will actually run. The kernel is
            # gated on spec-decode + hybrid (see MambaBuffers.create);
            # without it, ``mamba_bufs.postprocess_align`` is None and
            # the staging buffers don't exist.
            if mamba_bufs.postprocess_align is not None:
                mamba_utils.stage_postprocess_inputs_to_gpu(
                    mamba_bufs.postprocess_align,
                    scheduler_output,
                    self.input_batch.req_ids,
                    num_reqs,
                    self.requests,
                    self.mamba_state_idx,
                )

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=(
                num_tokens_padded
                if pad_attn or has_separate_kv_update
                else num_tokens_unpadded
            ),
            num_reqs_padded=(
                num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs
            ),
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )

        attn_metadata, spec_decode_common_attn_metadata = (
            self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                slot_mappings=slot_mappings_by_group,
            )
        )

        (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        ) = self._preprocess(scheduler_output, num_tokens_padded, intermediate_tensors)

    # Set cudagraph mode to none if calc_kv_scales is true.
    # KV scales calculation involves dynamic operations that are incompatible
    # with CUDA graph capture.
    if self.calculate_kv_scales:
        cudagraph_mode = CUDAGraphMode.NONE
        # Mark KV scales as calculated after the first forward pass
        self.calculate_kv_scales = False

    # Encoder-decoder models can only compile the pure decode steps where no
    # encoder inputs are present. Use eager for the first pass.
    num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
    has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

    # Run the model.
    # Use persistent buffers for CUDA graphs.
    # When spec decode is enabled, defer connector finalization
    # (wait_for_save + clear metadata) until after draft model runs.
    defer_kv_connector_finalize = self.speculative_config is not None
    with (
        set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_desc,
            ubatch_slices=ubatch_slices_padded,
            slot_mapping=slot_mappings,
            skip_compiled=has_encoder_input,
        ),
        record_function_or_nullcontext("gpu_model_runner: forward"),
        self.maybe_get_kv_connector_output(
            scheduler_output,
            defer_finalize=defer_kv_connector_finalize,
        ) as kv_connector_output,
    ):

        self.maybe_execute_ucm_sparse_begin(scheduler_output, attn_metadata)

        model_output = self._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        self.maybe_execute_ucm_sparse_finished()

    with record_function_or_nullcontext("gpu_model_runner: postprocess"):
        if self.use_aux_hidden_state_outputs:
            # True when EAGLE 3 is used.
            hidden_states, aux_hidden_states = model_output
        else:
            # Common case.
            hidden_states = model_output
            aux_hidden_states = None

        if not self.broadcast_pp_output:
            # Common case.
            if not get_pp_group().is_last_rank:
                # Return the intermediate tensors.
                assert isinstance(hidden_states, IntermediateTensors)
                self.kv_connector_output = kv_connector_output
                return hidden_states

            if self.is_pooling_model:
                # Return the pooling output.
                return self._pool(
                    hidden_states,
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                    kv_connector_output,
                )

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
        else:
            # Rare case.
            assert not self.is_pooling_model

            sample_hidden_states = hidden_states[logits_indices]
            if not get_pp_group().is_last_rank:
                all_gather_tensors = {
                    "residual": not is_residual_scattered_for_sp(
                        self.vllm_config, num_tokens_padded
                    )
                }
                get_pp_group().send_tensor_dict(
                    hidden_states.tensors,
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
                logits = None
            else:
                logits = self.model.compute_logits(sample_hidden_states)

            model_output_broadcast_data: dict[str, Any] = {}
            if logits is not None:
                model_output_broadcast_data["logits"] = logits.contiguous()

            broadcasted = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
            )
            assert broadcasted is not None
            logits = broadcasted["logits"]

    self.execute_model_state = ExecuteModelState(
        scheduler_output,
        logits,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        hidden_states,
        sample_hidden_states,
        aux_hidden_states,
        ec_connector_output,
        cudagraph_stats,
        slot_mappings,
    )
    self.kv_connector_output = kv_connector_output

    # Now the batch has been launched we can wait for corrections from the
    # previous model forward without breaking async scheduling.
    if deferred_state_corrections_fn:
        deferred_state_corrections_fn()

    return None


def maybe_execute_ucm_sparse_begin(
    self, scheduler_output: "SchedulerOutput", attn_metadata: CommonAttentionMetadata
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


def maybe_execute_ucm_sparse_finished(self):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.execute_finished()


def ucm_sparse_request_finished_in_worker(self, request_id: str | int):
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse

    if not has_ucm_sparse():
        return
    ucm_sparse = get_ucm_sparse()
    ucm_sparse.request_finished_in_worker(request_id)


@when_imported("vllm.v1.worker.gpu_model_runner")
def _patch(mod):
    if hasattr(mod.GPUModelRunner, "__init__"):
        _patched = FunctionType(
            __init__.__code__,
            vars(mod),
            name=__init__.__name__,
            argdefs=__init__.__defaults__,
            closure=__init__.__closure__,
        )
        _patched.__annotations__ = __init__.__annotations__
        _patched.__kwdefaults__ = __init__.__kwdefaults__
        patch_or_inject(mod.GPUModelRunner, "__init__", _patched)
        logger.info("UCM GSA GPUModelRunner.__init__ patch applied")

    if hasattr(mod.GPUModelRunner, "_update_states"):
        _patched = FunctionType(
            _update_states.__code__,
            vars(mod),
            name=_update_states.__name__,
            argdefs=_update_states.__defaults__,
            closure=_update_states.__closure__,
        )
        _patched.__annotations__ = _update_states.__annotations__
        _patched.__kwdefaults__ = _update_states.__kwdefaults__
        patch_or_inject(mod.GPUModelRunner, "_update_states", _patched)
        logger.info("UCM GSA GPUModelRunner._update_states patch applied")

    if hasattr(mod.GPUModelRunner, "_prepare_inputs"):
        _patched = FunctionType(
            _prepare_inputs.__code__,
            vars(mod),
            name=_prepare_inputs.__name__,
            argdefs=_prepare_inputs.__defaults__,
            closure=_prepare_inputs.__closure__,
        )
        _patched.__annotations__ = _prepare_inputs.__annotations__
        _patched.__kwdefaults__ = _prepare_inputs.__kwdefaults__
        patch_or_inject(mod.GPUModelRunner, "_prepare_inputs", _patched)
        logger.info("UCM GSA GPUModelRunner._prepare_inputs patch applied")

    if hasattr(mod.GPUModelRunner, "execute_model"):
        _patched = FunctionType(
            execute_model.__code__,
            vars(mod),
            name=execute_model.__name__,
            argdefs=execute_model.__defaults__,
            closure=execute_model.__closure__,
        )
        _patched.__annotations__ = execute_model.__annotations__
        _patched.__kwdefaults__ = execute_model.__kwdefaults__
        patch_or_inject(mod.GPUModelRunner, "execute_model", _patched)
        logger.info("UCM GSA GPUModelRunner.execute_model patch applied")

    if not hasattr(mod.GPUModelRunner, "maybe_execute_ucm_sparse_begin"):
        _patched = FunctionType(
            maybe_execute_ucm_sparse_begin.__code__,
            vars(mod),
            name=maybe_execute_ucm_sparse_begin.__name__,
            argdefs=maybe_execute_ucm_sparse_begin.__defaults__,
            closure=maybe_execute_ucm_sparse_begin.__closure__,
        )
        _patched.__annotations__ = maybe_execute_ucm_sparse_begin.__annotations__
        _patched.__kwdefaults__ = maybe_execute_ucm_sparse_begin.__kwdefaults__
        patch_or_inject(mod.GPUModelRunner, "maybe_execute_ucm_sparse_begin", _patched)
        logger.info("UCM GSA GPUModelRunner.maybe_execute_ucm_sparse_begin injected")

    if not hasattr(mod.GPUModelRunner, "maybe_execute_ucm_sparse_finished"):
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
            mod.GPUModelRunner, "maybe_execute_ucm_sparse_finished", _patched
        )
        logger.info("UCM GSA GPUModelRunner.maybe_execute_ucm_sparse_finished injected")

    if not hasattr(mod.GPUModelRunner, "ucm_sparse_request_finished_in_worker"):
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
            mod.GPUModelRunner, "ucm_sparse_request_finished_in_worker", _patched
        )
        logger.info(
            "UCM GSA GPUModelRunner.ucm_sparse_request_finished_in_worker injected"
        )
