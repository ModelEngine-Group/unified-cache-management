"""Fix hybrid prefix-cache accounting in vLLM-Ascend 0.23.0rc1."""

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.core.single_type_kv_cache_manager")
def patch_ascend_single_type_kv_cache_manager(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0230.vllm_ascend.core import (
        single_type_kv_cache_manager,
    )

    if not hasattr(mod, "CompressAttentionManager"):
        logger.warning(
            "Skip Ascend compressed-attention KV allocation patch: "
            "CompressAttentionManager is missing"
        )
        return

    patched_manager_cls = single_type_kv_cache_manager.CompressAttentionManager
    patch_or_inject(
        mod.CompressAttentionManager,
        "allocate_new_computed_blocks",
        patched_manager_cls.allocate_new_computed_blocks,
    )
    logger.info(
        "UCM Ascend compressed-attention KV allocation patch applied: "
        "CompressAttentionManager.allocate_new_computed_blocks"
    )


@when_imported("vllm_ascend.patch.platform.patch_kv_cache_coordinator")
def patch_ascend_kv_cache_coordinator(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0230.vllm_ascend.patch.platform import (
        patch_kv_cache_coordinator,
    )

    if not hasattr(mod, "AscendHybridKVCacheCoordinator"):
        logger.warning(
            "Skip Ascend hybrid KV cache coordinator patch: "
            "AscendHybridKVCacheCoordinator is missing"
        )
        return

    coordinator_cls = mod.AscendHybridKVCacheCoordinator
    patched_methods = []
    for method_name in (
        "find_longest_cache_hit",
        "find_longest_cache_hit_per_group",
    ):
        if not hasattr(coordinator_cls, method_name):
            logger.warning(
                "Skip Ascend hybrid KV cache coordinator patch: %s is missing",
                method_name,
            )
            continue

        original_method = getattr(coordinator_cls, method_name)
        replacement_method = (
            patch_kv_cache_coordinator.wrap_full_attention_cache_hit_lookup(
                original_method
            )
        )
        patch_or_inject(coordinator_cls, method_name, replacement_method)
        patched_methods.append(method_name)

    if patched_methods:
        logger.info(
            "UCM Ascend hybrid KV cache coordinator patch applied: %s",
            ", ".join(patched_methods),
        )
