"""Inject the GSA PMR speculative-config block into vLLM 0.23.0 EngineArgs."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def create_engine_config(
    self,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig:
    """
    Create the VllmConfig.

    NOTE: If VllmConfig is incompatible, we raise an error.
    """

    from ucm.utils import Config

    ucm_config = Config(self.kv_transfer_config)
    self.launch_config = ucm_config.get_config()
    pmr_tree_config = self.launch_config.get("pmr_config", None)
    use_pmr = False
    if pmr_tree_config:
        use_pmr = pmr_tree_config["use_pmr"]
    use_mtp = (
        self.speculative_config is not None
        and "mtp" in self.speculative_config["method"]
    )
    if use_pmr:
        if not use_mtp:
            self.speculative_config = {"method": "suffix"}
        import torch_npu

        npu_count = torch_npu.npu.device_count()
        multi_node = (
            self.data_parallel_size
            * self.pipeline_parallel_size
            * self.tensor_parallel_size
        ) > npu_count or self.data_parallel_size > 1
        has_added = int(os.environ.get("HAS_ADDED", 0))
        if "num_speculative_tokens" in self.speculative_config and not has_added:
            self.speculative_config["num_speculative_tokens"] += pmr_tree_config[
                "num_speculative_tokens"
            ]
            if multi_node:
                os.environ["HAS_ADDED"] = "1"
        else:
            if not multi_node:
                self.speculative_config["num_speculative_tokens"] = pmr_tree_config[
                    "num_speculative_tokens"
                ]
            os.environ["HAS_ADDED"] = "0"
        print(
            f"[GSA] PMR self.speculative_config {self.speculative_config['num_speculative_tokens']}"
        )

    current_platform.pre_register_and_update()

    device_config = DeviceConfig(device=cast(Device, current_platform.device_type))

    envs.validate_environ(self.fail_on_environ_validation)

    # Check if the model is a speculator and override model/tokenizer/config
    # BEFORE creating ModelConfig, so the config is created with the target model
    # Skip speculator detection for cloud storage models (eg: S3, GCS) since
    # HuggingFace cannot load configs directly from S3 URLs. S3 models can still
    # use speculators with explicit --speculative-config.
    if not is_cloud_storage(self.model):
        (self.model, self.tokenizer, self.speculative_config) = (
            maybe_override_with_speculators(
                model=self.model,
                tokenizer=self.tokenizer,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                vllm_speculative_config=self.speculative_config,
                hf_token=self.hf_token,
            )
        )

    model_config = self.create_model_config()
    self.model = model_config.model
    self.model_weights = model_config.model_weights
    self.tokenizer = model_config.tokenizer

    self._check_feature_supported()
    self._set_default_chunked_prefill_and_prefix_caching_args(model_config)
    self._set_default_reasoning_config_args()
    sliding_window: int | None = None
    if not is_interleaved(model_config.hf_text_config):
        # Only set CacheConfig.sliding_window if the model is all sliding
        # window. Otherwise CacheConfig.sliding_window will override the
        # global layers in interleaved sliding window models.
        sliding_window = model_config.get_sliding_window()

    # Resolve "auto" kv_cache_dtype to actual value from model config
    resolved_cache_dtype = resolve_kv_cache_dtype_string(
        self.kv_cache_dtype, model_config
    )

    assert (
        self.enable_prefix_caching is not None
    ), "enable_prefix_caching must be set by this point"

    cache_config = CacheConfig(
        block_size=self.block_size,  # type: ignore[arg-type]
        gpu_memory_utilization=self.gpu_memory_utilization,
        kv_cache_memory_bytes=self.kv_cache_memory_bytes,
        cache_dtype=resolved_cache_dtype,  # type: ignore[arg-type]
        is_attention_free=model_config.is_attention_free,
        num_gpu_blocks_override=self.num_gpu_blocks_override,
        sliding_window=sliding_window,
        enable_prefix_caching=self.enable_prefix_caching,
        prefix_caching_hash_algo=self.prefix_caching_hash_algo,
        calculate_kv_scales=self.calculate_kv_scales,
        kv_cache_dtype_skip_layers=self.kv_cache_dtype_skip_layers,
        kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,
        mamba_cache_dtype=self.mamba_cache_dtype,
        mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,
        mamba_block_size=self.mamba_block_size,
        mamba_cache_mode=self.mamba_cache_mode,
        kv_offloading_size=self.kv_offloading_size,
        kv_offloading_backend=self.kv_offloading_backend,
    )

    if resolved_cache_dtype.startswith("turboquant_"):
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        boundary = TurboQuantConfig.get_boundary_skip_layers(model_config)
        existing = set(cache_config.kv_cache_dtype_skip_layers)
        cache_config.kv_cache_dtype_skip_layers = sorted(
            existing | set(boundary), key=int
        )

    ray_runtime_env = None
    if is_ray_initialized():
        # Ray Serve LLM calls `create_engine_config` in the context
        # of a Ray task, therefore we check is_ray_initialized()
        # as opposed to is_in_ray_actor().
        import ray

        ray_runtime_env = ray.get_runtime_context().runtime_env
        # Avoid logging sensitive environment variables
        sanitized_env = ray_runtime_env.to_dict() if ray_runtime_env else {}
        if "env_vars" in sanitized_env:
            sanitized_env["env_vars"] = {k: "***" for k in sanitized_env["env_vars"]}
        logger.info("Using ray runtime env (env vars redacted): %s", sanitized_env)

    # Get the current placement group if Ray is initialized and
    # we are in a Ray actor. If so, then the placement group will be
    # passed to spawned processes.
    placement_group = None
    if is_in_ray_actor():
        import ray

        # This call initializes Ray automatically if it is not initialized,
        # but we should not do this here.
        placement_group = ray.util.get_current_placement_group()

    assert (
        not headless or not self.data_parallel_hybrid_lb
    ), "data_parallel_hybrid_lb is not applicable in headless mode"
    assert not (
        self.data_parallel_hybrid_lb and self.data_parallel_external_lb
    ), "data_parallel_hybrid_lb and data_parallel_external_lb cannot both be True."
    assert (
        self.data_parallel_backend == "mp" or self.nnodes == 1
    ), "nnodes > 1 is only supported with data_parallel_backend=mp"
    inferred_data_parallel_rank = 0
    if self.nnodes > 1:
        world_size = (
            self.data_parallel_size
            * self.pipeline_parallel_size
            * self.tensor_parallel_size
        )
        world_size_within_dp = self.pipeline_parallel_size * self.tensor_parallel_size
        local_world_size = world_size // self.nnodes
        assert (
            world_size % self.nnodes == 0
        ), f"world_size={world_size} must be divisible by nnodes={self.nnodes}."
        assert (
            self.node_rank < self.nnodes
        ), f"node_rank={self.node_rank} must be less than nnodes={self.nnodes}."
        inferred_data_parallel_rank = (
            self.node_rank * local_world_size
        ) // world_size_within_dp
        if self.data_parallel_size > 1 and self.data_parallel_external_lb:
            self.data_parallel_rank = inferred_data_parallel_rank
            logger.info(
                "Inferred data_parallel_rank %d from node_rank %d for external lb",
                self.data_parallel_rank,
                self.node_rank,
            )
        elif self.data_parallel_size_local is None:
            # Infer data parallel size local for internal dplb:
            self.data_parallel_size_local = max(
                local_world_size // world_size_within_dp, 1
            )
    data_parallel_external_lb = (
        self.data_parallel_external_lb or self.data_parallel_rank is not None
    )
    if (
        self.data_parallel_size > 1
        and data_parallel_external_lb
        and not model_config.is_moe
    ):
        raise ValueError(
            "Non-MoE models do not support external data parallel mode. "
            "For external load balancing, launch independent vLLM "
            "instances without --data-parallel-* arguments."
        )
    # Local DP rank = 1, use pure-external LB.
    if data_parallel_external_lb:
        assert self.data_parallel_rank is not None, (
            "data_parallel_rank or node_rank must be specified if "
            "data_parallel_external_lb is enable."
        )
        assert self.data_parallel_size_local in (1, None), (
            "data_parallel_size_local must be 1 or None when data_parallel_rank "
            "is set"
        )
        data_parallel_size_local = 1
        # Use full external lb if we have local_size of 1.
        self.data_parallel_hybrid_lb = False
    elif self.data_parallel_size_local is not None:
        data_parallel_size_local = self.data_parallel_size_local

        if self.data_parallel_start_rank and not headless:
            # Infer hybrid LB mode.
            self.data_parallel_hybrid_lb = True

        if self.data_parallel_hybrid_lb and data_parallel_size_local == 1:
            # Use full external lb if we have local_size of 1.
            logger.warning(
                "data_parallel_hybrid_lb is not eligible when "
                "data_parallel_size_local = 1, autoswitch to "
                "data_parallel_external_lb."
            )
            data_parallel_external_lb = True
            self.data_parallel_hybrid_lb = False

        if data_parallel_size_local == self.data_parallel_size:
            # Disable hybrid LB mode if set for a single node
            self.data_parallel_hybrid_lb = False

        self.data_parallel_rank = (
            self.data_parallel_start_rank or inferred_data_parallel_rank
        )
        if self.nnodes > 1:
            logger.info(
                "Inferred data_parallel_rank %d from node_rank %d",
                self.data_parallel_rank,
                self.node_rank,
            )
    else:
        assert (
            not self.data_parallel_hybrid_lb
        ), "data_parallel_size_local must be set to use data_parallel_hybrid_lb."

        if self.data_parallel_backend == "ray" and (
            envs.VLLM_RAY_DP_PACK_STRATEGY == "span"
        ):
            # Data parallel size defaults to 1 if DP ranks are spanning
            # multiple nodes
            data_parallel_size_local = 1
        else:
            # Otherwise local DP size defaults to global DP size if not set
            data_parallel_size_local = self.data_parallel_size

    # DP address, used in multi-node case for torch distributed group
    # and ZMQ sockets.
    if self.data_parallel_address is None:
        if self.data_parallel_backend == "ray":
            host_ip = get_ip()
            logger.info("Using host IP %s as ray-based data parallel address", host_ip)
            data_parallel_address = host_ip
        else:
            assert self.data_parallel_backend == "mp", (
                "data_parallel_backend can only be ray or mp, got %s",
                self.data_parallel_backend,
            )
            data_parallel_address = (
                self.master_addr or ParallelConfig.data_parallel_master_ip
            )
    else:
        data_parallel_address = self.data_parallel_address

    # This port is only used when there are remote data parallel engines,
    # otherwise the local IPC transport is used.
    data_parallel_rpc_port = (
        self.data_parallel_rpc_port
        if (self.data_parallel_rpc_port is not None)
        else ParallelConfig.data_parallel_rpc_port
    )

    if self.tokens_only and not model_config.skip_tokenizer_init:
        model_config.skip_tokenizer_init = True
        logger.info("Skipping tokenizer initialization for tokens-only mode.")

    parallel_config = ParallelConfig(
        pipeline_parallel_size=self.pipeline_parallel_size,
        tensor_parallel_size=self.tensor_parallel_size,
        prefill_context_parallel_size=self.prefill_context_parallel_size,
        data_parallel_size=self.data_parallel_size,
        data_parallel_rank=self.data_parallel_rank or 0,
        data_parallel_external_lb=data_parallel_external_lb,
        data_parallel_size_local=data_parallel_size_local,
        master_addr=self.master_addr,
        master_port=self.master_port,
        nnodes=self.nnodes,
        node_rank=self.node_rank,
        distributed_timeout_seconds=self.distributed_timeout_seconds,
        cpu_distributed_timeout_seconds=self.cpu_distributed_timeout_seconds,
        data_parallel_master_ip=data_parallel_address,
        data_parallel_rpc_port=data_parallel_rpc_port,
        data_parallel_backend=self.data_parallel_backend,
        data_parallel_hybrid_lb=self.data_parallel_hybrid_lb,
        is_moe_model=model_config.is_moe,
        enable_expert_parallel=self.enable_expert_parallel,
        enable_ep_weight_filter=self.enable_ep_weight_filter,
        all2all_backend=self.all2all_backend,
        enable_elastic_ep=self.enable_elastic_ep,
        enable_dbo=self.enable_dbo,
        ubatch_size=self.ubatch_size,
        dbo_decode_token_threshold=self.dbo_decode_token_threshold,
        dbo_prefill_token_threshold=self.dbo_prefill_token_threshold,
        disable_nccl_for_dp_synchronization=self.disable_nccl_for_dp_synchronization,
        enable_eplb=self.enable_eplb,
        eplb_config=self.eplb_config,
        expert_placement_strategy=self.expert_placement_strategy,
        max_parallel_loading_workers=self.max_parallel_loading_workers,
        disable_custom_all_reduce=self.disable_custom_all_reduce,
        ray_workers_use_nsight=self.ray_workers_use_nsight,
        ray_runtime_env=ray_runtime_env,
        placement_group=placement_group,
        distributed_executor_backend=self.distributed_executor_backend,
        worker_cls=self.worker_cls,
        worker_extension_cls=self.worker_extension_cls,
        decode_context_parallel_size=self.decode_context_parallel_size,
        dcp_comm_backend=self.dcp_comm_backend,
        dcp_kv_cache_interleave_size=self.dcp_kv_cache_interleave_size,
        cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
        _api_process_count=self._api_process_count,
        _api_process_rank=self._api_process_rank,
        numa_bind=self.numa_bind,
        numa_bind_nodes=self.numa_bind_nodes,
        numa_bind_cpus=self.numa_bind_cpus,
    )

    speculative_config = self.create_speculative_config(
        target_model_config=model_config,
        target_parallel_config=parallel_config,
    )

    self._set_default_max_num_seqs_and_batched_tokens_args(
        usage_context,
        model_config,
        parallel_config,
    )

    assert (
        self.max_num_batched_tokens is not None
    ), "max_num_batched_tokens must be set by this point"
    assert self.max_num_seqs is not None, "max_num_seqs must be set by this point"
    assert (
        self.enable_chunked_prefill is not None
    ), "enable_chunked_prefill must be set by this point"
    assert (
        model_config.max_model_len is not None
    ), "max_model_len must be set by this point"
    scheduler_config = SchedulerConfig(
        runner_type=model_config.runner_type,
        max_num_batched_tokens=self.max_num_batched_tokens,
        max_num_seqs=self.max_num_seqs,
        max_model_len=model_config.max_model_len,
        enable_chunked_prefill=self.enable_chunked_prefill,
        disable_chunked_mm_input=self.disable_chunked_mm_input,
        is_multimodal_model=model_config.is_multimodal_model,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy=self.scheduling_policy,
        scheduler_cls=self.scheduler_cls,
        max_num_partial_prefills=self.max_num_partial_prefills,
        max_long_partial_prefills=self.max_long_partial_prefills,
        long_prefill_token_threshold=self.long_prefill_token_threshold,
        scheduler_reserve_full_isl=self.scheduler_reserve_full_isl,
        disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
        async_scheduling=self.async_scheduling,
        stream_interval=self.stream_interval,
    )

    if not model_config.is_multimodal_model and self.default_mm_loras:
        raise ValueError(
            "Default modality-specific LoRA(s) were provided for a "
            "non multimodal model"
        )

    lora_config = (
        LoRAConfig(
            max_lora_rank=self.max_lora_rank,
            max_loras=self.max_loras,
            default_mm_loras=self.default_mm_loras,
            fully_sharded_loras=self.fully_sharded_loras,
            lora_dtype=self.lora_dtype,
            target_modules=self.lora_target_modules,
            enable_tower_connector_lora=self.enable_tower_connector_lora,
            specialize_active_lora=self.specialize_active_lora,
            enable_mixed_moe_lora_format=self.enable_mixed_moe_lora_format,
            max_cpu_loras=(
                self.max_cpu_loras
                if self.max_cpu_loras and self.max_cpu_loras > 0
                else None
            ),
        )
        if self.enable_lora
        else None
    )

    if (
        lora_config is not None
        and speculative_config is not None
        and scheduler_config.max_num_batched_tokens
        < (
            scheduler_config.max_num_seqs
            * (speculative_config.num_speculative_tokens + 1)
        )
    ):
        raise ValueError(
            "Consider increasing max_num_batched_tokens or "
            "decreasing num_speculative_tokens"
        )

    # bitsandbytes pre-quantized model need a specific model loader
    if model_config.quantization == "bitsandbytes":
        self.quantization = self.load_format = "bitsandbytes"

    # Attention config overrides
    attention_config = copy.deepcopy(self.attention_config)
    if self.attention_backend is not None:
        if attention_config.backend is not None:
            raise ValueError(
                "attention_backend and attention_config.backend "
                "are mutually exclusive"
            )
        # Reuse the validator to handle "auto" and string-to-enum conversion
        attention_config.backend = AttentionConfig.validate_backend_before(
            self.attention_backend
        )

    # TurboQuant requires FlashAttention 2 — FA3 boundary layers assert
    # FlashAttentionImpl which fails with TurboQuantAttentionImpl.
    if resolved_cache_dtype.startswith("turboquant_") and (
        attention_config.flash_attn_version is None
        or attention_config.flash_attn_version >= 3
    ):
        logger.warning(
            "TurboQuant is not yet compatible with FlashAttention >= 3. "
            "Overriding flash_attn_version to 2. To silence this "
            "warning, pass --attention-config.flash_attn_version=2"
        )
        attention_config.flash_attn_version = 2

    # Mamba config overrides
    mamba_config = copy.deepcopy(self.mamba_config)
    # Convert string to enum if needed (CLI parsing returns a string)
    if isinstance(self.mamba_backend, str):
        mamba_config.backend = MambaBackendEnum[self.mamba_backend.upper()]
    else:
        mamba_config.backend = self.mamba_backend
    if self.enable_mamba_cache_stochastic_rounding:
        mamba_config.enable_stochastic_rounding = (
            self.enable_mamba_cache_stochastic_rounding
        )
    if self.mamba_cache_philox_rounds:
        mamba_config.stochastic_rounding_philox_rounds = self.mamba_cache_philox_rounds

    # Kernel config overrides
    kernel_config = copy.deepcopy(self.kernel_config)
    if self.enable_flashinfer_autotune is not None:
        if kernel_config.enable_flashinfer_autotune is not None:
            raise ValueError(
                "enable_flashinfer_autotune and "
                "kernel_config.enable_flashinfer_autotune "
                "are mutually exclusive"
            )
        kernel_config.enable_flashinfer_autotune = self.enable_flashinfer_autotune
    if self.moe_backend != "auto":
        kernel_config.moe_backend = self.moe_backend
    if self.linear_backend != "auto":
        kernel_config.linear_backend = self.linear_backend

    # Transfer top-level ir_op_priority into KernelConfig.ir_op_priority
    for op_name, op_priority in asdict(self.ir_op_priority).items():
        # Empty means unset
        if not op_priority:
            continue

        # Priority cannot be set 2x for the same op
        if getattr(kernel_config.ir_op_priority, op_name):
            raise ValueError(
                f"Op priority for {op_name} specified via both ir_op_priority "
                f"and KernelConfig.ir_op_priority, only one allowed at a time."
            )

        # Set the attribute
        setattr(kernel_config.ir_op_priority, op_name, op_priority)

    load_config = self.create_load_config()

    # Pass reasoning_parser into StructuredOutputsConfig
    if self.reasoning_parser:
        self.structured_outputs_config.reasoning_parser = self.reasoning_parser

    if self.reasoning_parser_plugin:
        self.structured_outputs_config.reasoning_parser_plugin = (
            self.reasoning_parser_plugin
        )

    observability_config = ObservabilityConfig(
        show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,
        otlp_traces_endpoint=self.otlp_traces_endpoint,
        collect_detailed_traces=self.collect_detailed_traces,
        kv_cache_metrics=self.kv_cache_metrics,
        kv_cache_metrics_sample=self.kv_cache_metrics_sample,
        cudagraph_metrics=self.cudagraph_metrics,
        enable_layerwise_nvtx_tracing=self.enable_layerwise_nvtx_tracing,
        enable_mfu_metrics=self.enable_mfu_metrics,
        enable_mm_processor_stats=self.enable_mm_processor_stats,
        enable_logging_iteration_details=self.enable_logging_iteration_details,
    )

    # Compilation config overrides
    compilation_config = copy.deepcopy(self.compilation_config)
    if self.cudagraph_capture_sizes is not None:
        if compilation_config.cudagraph_capture_sizes is not None:
            raise ValueError(
                "cudagraph_capture_sizes and compilation_config."
                "cudagraph_capture_sizes are mutually exclusive"
            )
        compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes
    if self.max_cudagraph_capture_size is not None:
        if compilation_config.max_cudagraph_capture_size is not None:
            raise ValueError(
                "max_cudagraph_capture_size and compilation_config."
                "max_cudagraph_capture_size are mutually exclusive"
            )
        compilation_config.max_cudagraph_capture_size = self.max_cudagraph_capture_size

    offload_config = OffloadConfig(
        offload_backend=self.offload_backend,
        uva=UVAOffloadConfig(
            cpu_offload_gb=self.cpu_offload_gb,
            cpu_offload_params=self.cpu_offload_params,
        ),
        prefetch=PrefetchOffloadConfig(
            offload_group_size=self.offload_group_size,
            offload_num_in_group=self.offload_num_in_group,
            offload_prefetch_step=self.offload_prefetch_step,
            offload_params=self.offload_params,
        ),
    )

    if self.gdn_prefill_backend is not None:
        self.additional_config["gdn_prefill_backend"] = self.gdn_prefill_backend

    config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        offload_config=offload_config,
        attention_config=attention_config,
        mamba_config=mamba_config,
        kernel_config=kernel_config,
        lora_config=lora_config,
        speculative_config=speculative_config,
        structured_outputs_config=self.structured_outputs_config,
        observability_config=observability_config,
        compilation_config=compilation_config,
        kv_transfer_config=self.kv_transfer_config,
        kv_events_config=self.kv_events_config,
        ec_transfer_config=self.ec_transfer_config,
        reasoning_config=self.reasoning_config,
        profiler_config=self.profiler_config,
        additional_config=self.additional_config,
        optimization_level=self.optimization_level,
        performance_mode=self.performance_mode,
        weight_transfer_config=self.weight_transfer_config,
        shutdown_timeout=self.shutdown_timeout,
    )

    return config


@when_imported("vllm.engine.arg_utils")
def _patch(mod):
    try:
        import os as _os

        if not hasattr(mod, "os"):
            setattr(mod, "os", _os)
    except Exception as e:
        logger.warning(f"GSA arg_utils os injection failed {e}")

    if hasattr(mod.EngineArgs, "create_engine_config"):
        _patched = FunctionType(
            create_engine_config.__code__,
            vars(mod),
            name=create_engine_config.__name__,
            argdefs=create_engine_config.__defaults__,
            closure=create_engine_config.__closure__,
        )
        _patched.__annotations__ = create_engine_config.__annotations__
        _patched.__kwdefaults__ = create_engine_config.__kwdefaults__
        patch_or_inject(mod.EngineArgs, "create_engine_config", _patched)
        logger.info("UCM GSA EngineArgs.create_engine_config patch applied")
