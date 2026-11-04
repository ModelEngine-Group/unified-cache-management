"""Add a GSA cudagraph debug print in vLLM-Ascend 0.23.0 NPUPlatform."""

from __future__ import annotations

from types import FunctionType

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
    from vllm_ascend.quantization.utils import maybe_auto_detect_quantization

    device_config = getattr(vllm_config, "device_config", None)
    if (
        device_config is not None
        and getattr(device_config, "device_type", cls.device_type) != cls.device_type
    ):
        logger.debug(
            "Skipping Ascend-specific config updates for device type %s.",
            device_config.device_type,
        )
        return

    if vllm_config.model_config is None:
        logger.warning(
            "Model config is missing. Skipping Ascend-specific config updates."
        )
        return

    maybe_auto_detect_quantization(vllm_config)

    cls._validate_layer_sharding_config(vllm_config)
    cls._validate_draft_decode_context_parallel_config(vllm_config)
    cls._validate_parallel_config(vllm_config)
    cls._validate_pd_pp_mtp_config(vllm_config)

    # initialize ascend config from vllm additional_config
    cls._fix_incompatible_config(vllm_config)

    ascend_config = init_ascend_config(vllm_config)

    from vllm_ascend.logger import (
        configure_ascend_file_logging,
        configure_ascend_logging,
    )

    configure_ascend_file_logging()
    configure_ascend_logging()

    if vllm_config.kv_transfer_config is not None:
        check_kv_extra_config(vllm_config)
        if not getattr(vllm_config.kv_transfer_config, "_engine_id_patched", False):
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}-{uuid4().hex}"
            )
            vllm_config.kv_transfer_config._engine_id_patched = True
    from vllm.config import CompilationMode  # noqa: E402

    compilation_config = vllm_config.compilation_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config
    ascend_compilation_config = ascend_config.ascend_compilation_config
    if ascend_compilation_config:
        vllm_config.additional_config.setdefault(
            "ascend_compilation_config", {}
        ).update(
            vars(ascend_compilation_config)
            if not isinstance(ascend_compilation_config, dict)
            else ascend_compilation_config
        )

    ascend_config.update_compile_ranges_split_points()

    if model_config and hasattr(model_config.hf_text_config, "index_topk"):
        vllm_config.cache_config.cache_dtype = str(model_config.dtype).replace(
            "torch.", ""
        )

    ascend_fusion_config = ascend_config.ascend_fusion_config
    if ascend_fusion_config:
        vllm_config.additional_config.setdefault("ascend_fusion_config", {}).update(
            vars(ascend_fusion_config)
            if not isinstance(ascend_fusion_config, dict)
            else ascend_fusion_config
        )

    enforce_eager = getattr(model_config, "enforce_eager", False)

    from vllm.config.compilation import CUDAGraphMode

    if ascend_config.xlite_graph_config.enabled:
        if (
            ascend_config.xlite_graph_config.full_mode
            and vllm_config.speculative_config is None
        ):
            logger.info(
                "ACLGraph has been disabled when speculation is disabled in xlite full mode"
            )
            enforce_eager = True
            model_config.enforce_eager = True
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        else:
            logger.info("Falling back to FULL_DECODE_ONLY under xlite decode-only mode")
            compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

    if enforce_eager:
        logger.info("Compilation disabled, using eager mode by default")
        compilation_config.mode = CompilationMode.NONE
        if compilation_config.splitting_ops is None:
            compilation_config.splitting_ops = []

    compilation_config.cudagraph_num_of_warmups = 1

    if compilation_config.mode not in [
        CompilationMode.NONE,
        CompilationMode.VLLM_COMPILE,
    ]:
        logger.warning(
            "NPU does not support compilation mode. mode=%s, action: setting CUDAGraphMode to NONE.",
            compilation_config.mode,
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    # Recompute cudagraph sizes after Ascend-specific compatibility updates.
    # The platform default max is injected earlier via
    # `apply_config_platform_defaults`, so this late pass should only honor
    # the current max / size inputs after the mode adjustments above.
    vllm_config._set_cudagraph_sizes()
    # TODO delete graph size update here when compilation_config.pass_config.enable_sp
    # is supported by vllm-ascend.
    if (
        vllm_config.parallel_config.tensor_parallel_size > 1
        and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and not vllm_config.model_config.enforce_eager
        and enable_sp(vllm_config)
    ):
        original_sizes = compilation_config.cudagraph_capture_sizes
        sp_aclgraph_sizes = vllm_config.update_sizes_for_sequence_parallelism(
            original_sizes
        )
        assert sp_aclgraph_sizes, (
            f"cudagraph_capture_sizes {original_sizes} does not contain"
            f"values that are multiples of tp_size "
            f"{vllm_config.parallel_config.tensor_parallel_size}"
        )
        if len(sp_aclgraph_sizes) != len(original_sizes):
            # If user set the max_num_seqs miss fit the multiple of tp_size,
            # we need to match the max_cudagraph_capture_size with the valid max size,
            # so we can avoid initialization error of vllm server.
            compilation_config.max_cudagraph_capture_size = sp_aclgraph_sizes[-1]
            compilation_config.cudagraph_capture_sizes = sp_aclgraph_sizes
            update_cudagraph_capture_sizes(vllm_config, sp_aclgraph_sizes)

    # Encoder-decoder models currently only support PIECEWISE mode
    # TODO(Jian Li): Confirm this behavior and explain why
    if (
        model_config
        and model_config.is_encoder_decoder
        and compilation_config.cudagraph_mode
        not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
    ):
        cudagraph_mode = (
            CUDAGraphMode.PIECEWISE
            if compilation_config.mode == CompilationMode.VLLM_COMPILE
            else CUDAGraphMode.NONE
        )
        logger.info_once(
            "Encoder-decoder models don't support %s, fallback to %s.",
            compilation_config.cudagraph_mode,
            cudagraph_mode,
        )
        compilation_config.cudagraph_mode = cudagraph_mode

    # get custom compile backend for graph fusion
    compilation_config.oot_compiler = cls.get_compile_backend()

    compilation_config.use_inductor = False
    if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        compilation_config.mode = CompilationMode.NONE
        ascend_config.ascend_compilation_config.enable_npugraph_ex = False
        ascend_config.ascend_compilation_config.enable_static_kernel = False
        vllm_config.additional_config["ascend_compilation_config"][
            "enable_npugraph_ex"
        ] = False
        vllm_config.additional_config["ascend_compilation_config"][
            "enable_static_kernel"
        ] = False
    elif compilation_config.cudagraph_mode.requires_piecewise_compilation():
        # Our is_cuda_alike is False so we cannot reuse the assertion of upstream
        assert compilation_config.mode == CompilationMode.VLLM_COMPILE, (
            "Compilation mode should be CompilationMode.VLLM_COMPILE "
            "when cudagraph_mode piecewise cudagraphs is used, "
            "cudagraph_mode=%s",
            compilation_config.cudagraph_mode,
        )
        compilation_config.set_splitting_ops_for_v1(
            all2all_backend=vllm_config.parallel_config.all2all_backend,
            data_parallel_size=vllm_config.parallel_config.data_parallel_size,
        )
        # NOTE: Theoretically, we should also add this in the attention ops.
        # Since the process is created in the spawn mode, the value of the class attribute
        # attention ops transmitted is still the one before modification, so it has not been modified.
        # This will cause in scenarios where both piecewise and splitting ops are configured simultaneously,
        # If splitting ops does not contain the this value, this configuration issue will
        # not be detected in advance assert.
        compilation_config.splitting_ops.extend(
            [
                "vllm::mla_forward",
                "vllm::dsa_forward",
            ]
        )
        # TODO(2026/7/15): Delete the reduced gear after the new driver is released.
        if get_ascend_device_type() == AscendDeviceType.A5:
            prune_capture_sizes_for_950(vllm_config)
        ascend_config.ascend_compilation_config.enable_npugraph_ex = False
        ascend_config.ascend_compilation_config.enable_static_kernel = False
        vllm_config.additional_config["ascend_compilation_config"][
            "enable_npugraph_ex"
        ] = False
        vllm_config.additional_config["ascend_compilation_config"][
            "enable_static_kernel"
        ] = False
    elif compilation_config.cudagraph_mode.has_full_cudagraphs():
        # We don't want to have our FX graph split for the sake of static kernel feature,
        # because it will compile multiple times, so we set splitting_ops to empty manually.
        compilation_config.splitting_ops = []
    else:
        logger.info(
            "%s cudagraph_mode is not support on NPU. falling back to NONE",
            compilation_config.cudagraph_mode,
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.mode = CompilationMode.NONE
        ascend_config.ascend_compilation_config.enable_npugraph_ex = False
        ascend_config.ascend_compilation_config.enable_static_kernel = False
        vllm_config.additional_config["ascend_compilation_config"][
            "enable_npugraph_ex"
        ] = False
        vllm_config.additional_config["ascend_compilation_config"][
            "enable_static_kernel"
        ] = False

    print(
        f"[GSA] graph cudagraph_capture_sizes {compilation_config.cudagraph_capture_sizes} {compilation_config.cudagraph_mode}"
    )

    # TODO: Remove this check when ACL Graph supports ASCEND_LAUNCH_BLOCKING=1
    # Then, we will have to discuss the error handling strategy and user experience
    if (
        compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1"
    ):
        raise ValueError(
            "ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. "
            "Please unset ASCEND_LAUNCH_BLOCKING or set it to 0. If you "
            "need ASCEND_LAUNCH_BLOCKING for debugging, consider other methods — "
            "for example, check the plog files (default: $HOME/ascend/log/debug) "
            "for more information about runtime errors."
        )

    if parallel_config and parallel_config.worker_cls == "auto":
        # TODO: this is a tricky way to disable `use_sequence_parallel_moe` in vllm.
        if not vllm_config.compilation_config.pass_config.enable_sp:
            parallel_config.all2all_backend = "flashinfer_all2allv"
        if is_310p():
            parallel_config.worker_cls = "vllm_ascend._310p.worker_310p.NPUWorker310"
        elif ascend_config.xlite_graph_config.enabled:
            logger.info(
                "openEuler Xlite enabled. See: https://atomgit.com/openeuler/GVirt/tree/master/xlite"
            )
            parallel_config.worker_cls = "vllm_ascend.xlite.xlite_worker.XliteWorker"
        else:
            parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"

    refresh_block_size(vllm_config)

    # Activate custom ops for v1, except on 310P
    if get_ascend_device_type() != AscendDeviceType._310P:
        compilation_config.custom_ops = ["all"]

    if ascend_config.enable_balance_scheduling:
        kv_transfer_config = vllm_config.kv_transfer_config
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        if kv_transfer_config is not None and kv_role != "kv_both":
            raise ValueError(
                "enable_balance_scheduling only supports PD-mixed mode "
                "(kv_role='kv_both' or no kv_transfer_config), and is not supported in "
                "PD-disaggregated mode (kv_role='kv_producer'/'kv_consumer')."
            )

    cls._validate_kv_load_failure_policy(vllm_config)

    if ascend_config.recompute_scheduler_enable:
        kv_transfer_config = vllm_config.kv_transfer_config
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        if kv_role == "kv_producer":
            logger.warning(
                "recompute_scheduler_enable is ignored on PD-disaggregated P nodes "
                "(kv_role='kv_producer') and will be deprecated on P nodes in a future release. "
                "Please remove it from P-node configs and keep it only on PD-disaggregated D nodes "
                "(kv_role='kv_consumer')."
            )
            ascend_config.recompute_scheduler_enable = False
            vllm_config.additional_config["recompute_scheduler_enable"] = False
        elif kv_transfer_config is None or kv_role != "kv_consumer":
            raise ValueError(
                "recompute_scheduler_enable can only be enabled on PD-disaggregated D nodes "
                f"(kv_role='kv_consumer', but got kv_role={kv_role!r}), and is not supported in PD-mixed mode."
            )
        else:
            from vllm_ascend.core.recompute_scheduler import RecomputeSchedulerConfig

            recompute_scheduler_config = (
                RecomputeSchedulerConfig.initialize_from_config(vllm_config)
            )
            vllm_config.scheduler_config = recompute_scheduler_config

    # Extend original scheduler_config to use SchedulerDynamicBatch.
    if ascend_config.SLO_limits_for_dynamic_batch != -1:
        vllm_config.scheduler_config.scheduler_cls = (
            "vllm_ascend.core.scheduler_dynamic_batch.SchedulerDynamicBatch"
        )
        vllm_config.scheduler_config.enable_chunked_prefill = True
        vllm_config.scheduler_config.SLO_limits_for_dynamic_batch = (
            ascend_config.SLO_limits_for_dynamic_batch
        )

    # Use ProfilingChunkScheduler when profiling-based chunk sizing is on.
    if ascend_config.profiling_chunk_config.enabled:
        vllm_config.scheduler_config.scheduler_cls = (
            "vllm_ascend.core.scheduler_profiling_chunk.ProfilingChunkScheduler"
        )
        import vllm_ascend.patch.platform.patch_profiling_chunk  # noqa

    cp_size = (
        parallel_config.decode_context_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    use_sparse = model_uses_sfa_sparse(model_config)
    sfa_dcp_replicated_indexer = enable_sfa_dcp_replicated_indexer(vllm_config)
    if sfa_dcp_replicated_indexer:
        if (
            parallel_config.decode_context_parallel_size
            != parallel_config.tensor_parallel_size
        ):
            raise AssertionError(
                f"DCP for SFA is only supported when dcp_size({parallel_config.decode_context_parallel_size}) "
                f"== tp_size({parallel_config.tensor_parallel_size})."
            )
        enable_sparse_c8 = (
            ascend_config.enable_sparse_sfa_c8 or ascend_config.enable_sparse_li_c8
        ) and use_sparse
        if enable_sparse_c8 and get_ascend_device_type() == AscendDeviceType.A5:
            raise NotImplementedError(
                "SFA DCP with sparse C8 cache is not supported on A5 yet. "
                "A5 uses the fused CKV quant sparse attention path, which needs a separate DCP LSE merge."
            )

    if (
        vllm_config.kv_transfer_config is not None
        and cache_config.block_size != parallel_config.cp_kv_cache_interleave_size
        and cp_size > 1
    ):
        raise AssertionError(
            f"cp_kv_cache_interleave_size({parallel_config.cp_kv_cache_interleave_size}) "
            f"and block_size({cache_config.block_size}) "
            "needs to be equal if use pcp or dcp > 1 in P/D disaggregate and kv pool scenario."
        )

    if (
        use_sparse
        and cp_size > 1
        and parallel_config.cp_kv_cache_interleave_size != cache_config.block_size
    ):
        logger.warning_once(
            "The current SFA CP implementation requires "
            f"cp_kv_cache_interleave_size({parallel_config.cp_kv_cache_interleave_size})"
            f" == block_size({cache_config.block_size}). "
            f"Override cp_kv_cache_interleave_size to {cache_config.block_size}."
        )
        vllm_config.parallel_config.cp_kv_cache_interleave_size = (
            cache_config.block_size
        )

    if enable_sp(vllm_config):
        assert (
            vllm_config.parallel_config.tensor_parallel_size > 1
        ), "Flash Comm v1 is only supported when tp_size > 1."

        assert (
            not is_moe_model(vllm_config)
            or vllm_config.parallel_config.enable_expert_parallel
        ), "Flash Comm v1 requires enable_expert_parallel=True for MoE models."

    # Set "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" by default to optimize NPU memory management.
    # Find more details at https://docs.vllm.ai/projects/ascend/en/latest/faqs.html#how-to-handle-the-out-of-memory-issue
    # NOTE: We should not set this environment variable in RL (sleep mode) scenarios.
    # Find more details about how to configure this environment variable at https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_012.html
    if model_config and not model_config.enable_sleep_mode:
        npu_alloc_configs = os.getenv(
            "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"
        )
        # This environment variable may have more than one key-value pairs.
        # We should append ",expandable_segments:True" to the current configs.
        # For example: "page_size:1g" + ",expandable_segments:True".
        # NOTE: `max_split_size_mb` or `garbage_collection_threshold` cannot
        # be enabled together with `expandable_segments=True`.
        if (
            "expandable_segments" not in npu_alloc_configs
            and "max_split_size_mb" not in npu_alloc_configs
            and "garbage_collection_threshold" not in npu_alloc_configs
        ):
            npu_alloc_configs += ",expandable_segments:True"
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = npu_alloc_configs
        logger.info("Set PYTORCH_NPU_ALLOC_CONF=%s", npu_alloc_configs)

    if ascend_config.enable_mc2_hierarchy_comm and ascend_config.enable_fused_mc2:
        raise ValueError(
            "fused mc2 op cannot be used with hierarchy communication."
            "Please disable VLLM_ASCEND_ENABLE_FUSED_MC2 by setting it to 0."
        )


@when_imported("vllm_ascend.platform")
def _patch(mod):
    if hasattr(mod.NPUPlatform, "check_and_update_config"):
        _patched = FunctionType(
            check_and_update_config.__code__,
            vars(mod),
            name=check_and_update_config.__name__,
            argdefs=check_and_update_config.__defaults__,
            closure=check_and_update_config.__closure__,
        )
        _patched.__annotations__ = check_and_update_config.__annotations__
        _patched.__kwdefaults__ = check_and_update_config.__kwdefaults__
        patch_or_inject(mod.NPUPlatform, "check_and_update_config", _patched)
        logger.info("UCM GSA NPUPlatform.check_and_update_config patch applied")
