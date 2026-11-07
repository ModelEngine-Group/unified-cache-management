"""Build the real model on meta and allocate its MRV2 KV views on the host."""

import json
import os
from contextlib import ExitStack
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch


def make_engine(args):
    if args.platform == "ascend-sim":
        from vllm_ascend.utils import adapt_patch

        adapt_patch(is_global_patch=True)

    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    extra = {"use_layerwise": True}
    if args.ucm_config:
        extra = {"UCM_CONFIG_FILE": str(args.ucm_config.resolve())}
    elif args.chunk_size:
        extra["chunk_size"] = args.chunk_size
    model_config = json.loads((args.model / "config.json").read_text())
    is_v41 = any(
        "DeepseekV41" in name for name in model_config.get("architectures", ())
    )
    engine = EngineArgs(
        model=str(args.model.resolve()),
        # The V4.1 multimodal scheduler constructs a processor even for text.
        skip_tokenizer_init=not is_v41,
        trust_remote_code=args.trust_remote_code,
        load_format="dummy",
        max_model_len=args.tokens + 1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        max_num_seqs=1,
        tensor_parallel_size=args.tp,
        prefill_context_parallel_size=args.pcp,
        decode_context_parallel_size=args.dcp,
        distributed_executor_backend="mp" if args.tp * args.pcp > 1 else None,
        dtype=args.dtype,
        enforce_eager=True,
        enable_prefix_caching=True,
        async_scheduling=False,
        disable_hybrid_kv_cache_manager=False,
        block_size=args.block_size,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UCMConnector",
            kv_connector_module_path="ucm.integration.vllm.v2.ucm_connector",
            kv_role="kv_both",
            kv_load_failure_policy="fail",
            kv_connector_extra_config=extra,
        ),
    ).create_engine_config()
    if not engine.use_v2_model_runner:
        raise ValueError("This model-check requires the platform's MRV2 path")
    return engine


def install_ascend_worker(engine):
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ascend_forward_context import sync_v2_extra_kwargs
    from vllm_ascend.mrv2_utils import apply_v2_model_runner_config_patch
    from vllm_ascend.utils import (
        adapt_patch,
        register_ascend_customop,
        set_potential_max_tokens,
    )

    set_potential_max_tokens(engine)
    apply_v2_model_runner_config_patch()
    sync_v2_extra_kwargs(engine)
    adapt_patch()
    from vllm_ascend import ops
    from vllm_ascend.patch.worker.patch_kda_recoverssm import patch_kda_recoverssm

    patch_kda_recoverssm(engine)
    ops.register_dummy_fusion_op()
    register_ascend_customop(engine)
    init_ascend_config(engine)


def make_model(engine):
    import torch
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    class MetaPlacement(torch.overrides.TorchFunctionMode):
        # A few model constructors set the accelerator explicitly, overriding
        # the default device. Keep their weights/scales on meta as well.
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            device = kwargs.get("device")
            if isinstance(device, (str, torch.device)) and torch.device(
                device
            ).type in ("cuda", "npu"):
                kwargs["device"] = torch.device("meta")
            if (
                func is torch.Tensor.to
                and len(args) > 1
                and isinstance(args[1], (str, torch.device))
                and torch.device(args[1]).type in ("cuda", "npu")
            ):
                args = (args[0], torch.device("meta"), *args[2:])
            return func(*args, **kwargs)

    with (
        set_default_torch_dtype(engine.model_config.dtype),
        torch.device("meta"),
        MetaPlacement(),
        ExitStack() as model_patches,
    ):
        architectures = engine.model_config.architectures or ()
        if any("DeepseekV41" in name for name in architectures):
            if engine.device_config.device_type == "npu":
                from vllm_ascend.models.deepseek_v41.engram.embedding import (
                    AscendParallelEngramEmbedding as EngramEmbedding,
                )

            else:
                from vllm.models.deepseek_v41.common import engram

                EngramEmbedding = engram.ParallelEngramEmbedding
                model_patches.enter_context(
                    patch.object(engram, "is_uva_available", lambda: True)
                )

            def meta_engram_weights(embedding):
                dtype = getattr(embedding, "storage_dtype", torch.float8_e4m3fn)
                codes = torch.empty(
                    (embedding.part_num_embeddings, embedding.dim),
                    dtype=dtype,
                    device="meta",
                )
                scales = None
                if dtype != torch.bfloat16:
                    scales = torch.empty(
                        (
                            embedding.part_num_embeddings,
                            embedding.dim // embedding.block_size,
                        ),
                        dtype=getattr(embedding, "scale_dtype", torch.uint8),
                        device="meta",
                    )
                return codes, scales

            model_patches.enter_context(
                patch.object(EngramEmbedding, "_allocate_weights", meta_engram_weights)
            )
        model = initialize_model(engine)
    if any(not param.is_meta for param in model.parameters()):
        raise RuntimeError("Model construction materialized weights outside meta")
    return model


def allocate(engine, kv_mib, scratch_dir, cleanup):
    import torch
    from parallel import gather
    from vllm.config import set_current_vllm_config
    from vllm.distributed.kv_transfer.kv_connector.utils import (
        get_current_attn_backends,
    )
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts,
        resolve_kv_cache_layout,
    )

    is_ascend = engine.device_config.device_type == "npu"
    cleanup.enter_context(set_current_vllm_config(engine))
    if is_ascend:
        install_ascend_worker(engine)
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    from vllm.v1.worker.gpu import attn_utils, model_runner
    from vllm.v1.worker.utils import AttentionGroup

    # Inspect the functions the installed runner and planner actually call.
    if is_ascend:
        live = (
            model_runner.get_kv_cache_spec,
            model_runner.init_kv_cache.__globals__["allocate_kv_cache"],
            model_runner.init_kv_cache.__globals__["bind_kv_cache_to_layers"],
            attn_utils.prepare_kernel_block_sizes,
            kv_cache_utils.get_kv_cache_configs.__globals__[
                "get_kv_cache_config_from_groups"
            ],
            kv_cache_utils.resolve_kv_cache_block_sizes,
        )
        for fn in live:
            print(f"Ascend MRV2: {fn.__module__}.{fn.__name__}")
            if not fn.__module__.startswith("vllm_ascend."):
                raise RuntimeError(f"Ascend patch missing at {fn.__name__}")

    from vllm.distributed import parallel_state

    parallel = engine.parallel_config
    rank = int(os.environ.get("RANK", "0"))
    world = parallel.world_size
    rendezvous = "env://" if world > 1 else (Path(scratch_dir) / "rendezvous").as_uri()
    # Keep the platform's real group constructors and memberships. Disable
    # accelerator collectives and the executor's unused shared-memory queue.
    coordinator = parallel_state.GroupCoordinator

    def host_group(**kwargs):
        kwargs.update(
            use_device_communicator=False, use_message_queue_broadcaster=False
        )
        return coordinator(**kwargs)

    with ExitStack() as group_options:
        group_options.enter_context(
            patch.object(parallel_state, "GroupCoordinator", host_group)
        )
        if is_ascend:
            from vllm_ascend.patch.worker import patch_distributed

            # The patched coordinator normally creates HCCL options even for
            # the device-side group. Both groups use Gloo on this host.
            group_options.enter_context(
                patch.object(
                    patch_distributed, "create_hccl_pg_options", lambda name: None
                )
            )
        init_distributed_environment(
            world, rank, rendezvous, rank, "gloo", timeout=timedelta(minutes=5)
        )
        cleanup.callback(destroy_distributed_environment)
        initialize_model_parallel(
            tensor_model_parallel_size=parallel.tensor_parallel_size,
            prefill_context_model_parallel_size=parallel.prefill_context_parallel_size,
            decode_context_model_parallel_size=parallel.decode_context_parallel_size,
            backend="gloo",
        )
        cleanup.callback(destroy_model_parallel)
        if is_ascend:
            from vllm_ascend.distributed.parallel_state import (
                destroy_ascend_model_parallel,
                init_ascend_model_parallel,
            )

            init_ascend_model_parallel(parallel, engine.engram_config)
            cleanup.callback(destroy_ascend_model_parallel)

    model = make_model(engine)
    register_all_kvcache_specs(engine)
    specs = model_runner.get_kv_cache_spec(engine)
    if not specs:
        raise ValueError(f"{type(model).__name__} registered no KV cache specs")
    if any(getattr(spec, "non_causal", False) for spec in specs.values()):
        raise ValueError("Non-causal prefill disables native prefix caching")
    supported = get_supported_kv_cache_layouts(get_current_attn_backends(engine))
    packing = resolve_kv_cache_layout(
        engine, gather([p.name for p in supported]), specs.values()
    )
    if kv_mib is None:
        # With PP=1 all stages have the same layers; this is a per-rank budget.
        # Use the same two costs as MRV2 admission, including the reserved null block.
        groups = kv_cache_utils.get_kv_cache_groups(engine, dict(specs))
        kv_bytes = kv_cache_utils._max_memory_usage_bytes_from_groups(engine, groups)
        kv_bytes += kv_cache_utils._pool_bytes_per_block(engine, groups)
    else:
        kv_bytes = kv_mib << 20
    print(f"Host KV pool budget: {kv_bytes / (1 << 20):.1f} MiB", flush=True)
    inputs = gather((specs, kv_bytes))
    all_specs, budgets = zip(*inputs)
    configs = kv_cache_utils.get_kv_cache_configs(
        engine, list(all_specs), list(budgets)
    )
    for config in configs:
        config.kv_cache_layout = packing.name
    config = configs[rank]
    scheduler_config = kv_cache_utils.generate_scheduler_kv_cache_config(configs)
    engine.cache_config.num_gpu_blocks = scheduler_config.num_blocks
    groups = (
        scheduler_config.prefix_cacheable_groups or scheduler_config.kv_cache_groups
    )
    engine.cache_config.block_size = min(g.kv_cache_spec.block_size for g in groups)
    kv_cache_utils.update_kv_cache_capacity(engine, scheduler_config)
    engine.adjust_dcp_kv_cache_interleave_size(config)
    engine.validate_block_size()
    block_size, hash_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        scheduler_config, engine
    )
    alignment = kv_cache_utils.resolve_cache_hit_alignment_tokens(
        scheduler_config, engine, block_size, hash_size
    )
    engine.cache_config.hash_block_size = hash_size
    engine.cache_config.cache_hit_alignment_tokens = alignment
    for cfg in (config, scheduler_config):
        cfg.hash_block_size = hash_size
        cfg.cache_hit_alignment_tokens = alignment

    # The per-layer backends of the constructed model constrain kernel pages.
    # No attention metadata builder or forward workspace is needed.
    layers = engine.compilation_config.static_forward_context
    physical = deepcopy(config)
    if is_ascend:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
            apply_layerwise_kv_cache_plan,
        )

        apply_layerwise_kv_cache_plan(physical, engine)
    attn_utils.add_kv_sharing_layers_to_kv_cache_groups(
        attn_utils.get_shared_kv_cache_layers(engine), physical.kv_cache_groups
    )
    attn_groups = []
    for group_id, group in enumerate(physical.kv_cache_groups):
        spec = group.kv_cache_spec
        attn_groups.append(
            [
                AttentionGroup(
                    backend=layers[name].get_attn_backend(),
                    layer_names=[name],
                    kv_cache_spec=(
                        spec.kv_cache_specs[name]
                        if isinstance(spec, UniformTypeKVCacheSpecs)
                        else spec
                    ),
                    kv_cache_group_id=group_id,
                )
                for name in group.layer_names
                if spec.has_layer_views
            ]
        )
    sizes = attn_utils.prepare_kernel_block_sizes(physical, attn_groups)
    views = model_runner.init_kv_cache(
        layers, physical, torch.device("cpu"), sizes, engine
    )
    for name, view in views.items():
        tensors = view if isinstance(view, (tuple, list)) else (view,)
        if any(t.device.type != "cpu" for t in tensors):
            raise RuntimeError(f"Native allocator placed {name} outside CPU memory")
    print(
        f"{type(model).__name__}: {packing.name}, {config.num_blocks} blocks, {len(views)} layer views"
    )
    return config, scheduler_config, block_size, hash_size, views
