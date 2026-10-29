"""Add dynamic construction to the compatible public unified-cache factory."""

from __future__ import annotations


def apply() -> None:
    from sglang.srt.mem_cache import registry

    original = registry.create_unified_radix_cache
    if getattr(original, "_ucm_patched", False):
        return

    def create_unified_radix_cache(ctx, *, cache_class=None):
        from sglang.srt.runtime_context import get_disagg, get_memory

        memory = get_memory()
        if (
            not memory.enable_unified_cache_external_linker
            or memory.unified_cache_external_linker_backend != "dynamic"
            or ctx.enable_hierarchical_cache
            or get_disagg().disaggregation_decode_retraction_backup == "host_pool"
        ):
            return original(ctx, cache_class=cache_class)

        # Follow the release factory's pool and counter setup; only linker
        # construction differs for the dynamic backend.
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx
        from sglang.srt.mem_cache.unified_cache.components import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        from ucm.integration.sglang.patch.v0520.linker_factory import (
            create_dynamic_linker,
        )

        server_args, params = ctx.server_args, ctx.params
        tree_components = [ComponentType.FULL]
        if ctx.is_hybrid_swa:
            tree_components.append(ComponentType.SWA)
        if ctx.is_hybrid_ssm:
            tree_components.append(ComponentType.MAMBA)
        if hasattr(params.req_to_token_pool, "req_to_c128_sidecar"):
            from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
                C128SidecarComponent,
            )

            tree_components.append(ComponentType.C128)
            params.component_registry_override = {
                **(params.component_registry_override or {}),
                ComponentType.C128: C128SidecarComponent,
            }
        params.tree_components = tuple(tree_components)
        if use_mlx() and ctx.is_hybrid_ssm:
            from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
                MlxAuxiliaryStateComponent,
            )

            params.component_registry_override = {
                ComponentType.MAMBA: MlxAuxiliaryStateComponent,
            }
        cache = (cache_class or UnifiedRadixCache)(params)
        model_runner = getattr(ctx.tp_worker, "model_runner", None)
        forward_stream = getattr(model_runner, "forward_stream", None)
        cache.init_cache_linker(
            create_dynamic_linker(
                server_args,
                params,
                components=set(cache.components),
                forward_stream=forward_stream,
            )
        )
        counter = cache.linker.layer_done_counter
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        kvcache.register_layer_transfer_counter(counter)
        ctx.tp_worker.register_hicache_layer_transfer_counter(counter)
        return cache

    create_unified_radix_cache._ucm_patched = True
    registry.create_unified_radix_cache = create_unified_radix_cache
