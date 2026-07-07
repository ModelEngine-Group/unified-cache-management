import ucm.integration.vllm.patch.scheduler_metrics_patch  # noqa: F401
from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.core.single_type_kv_cache_manager")
def patch_ascend_single_type_kv_cache_manager(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0202.vllm_ascend.core import (
        single_type_kv_cache_manager,
    )

    if not hasattr(mod, "CompressAttentionManager"):
        logger.warning(
            "Skip Ascend compressed-attention KV allocation patch: "
            "CompressAttentionManager is missing"
        )
        return

    patch_or_inject(
        mod.CompressAttentionManager,
        "allocate_new_computed_blocks",
        single_type_kv_cache_manager.CompressAttentionManager.allocate_new_computed_blocks,
    )
    logger.info(
        "UCM Ascend compressed-attention KV allocation patch applied: "
        "CompressAttentionManager.allocate_new_computed_blocks"
    )


# UCM patch for vllm-ascend 0.20.2:
# keep the fix in UCM instead of modifying the vllm-ascend repository directly.
@when_imported("vllm_ascend.patch.platform.patch_kv_cache_coordinator")
def patch_ascend_kv_cache_coordinator(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0202.vllm_ascend.patch.platform import (
        patch_kv_cache_coordinator,
    )

    if not hasattr(mod, "AscendHybridKVCacheCoordinator"):
        logger.warning(
            "Skip Ascend hybrid KV cache coordinator patch: "
            "AscendHybridKVCacheCoordinator is missing"
        )
        return

    patch_or_inject(
        mod.AscendHybridKVCacheCoordinator,
        "find_longest_cache_hit",
        patch_kv_cache_coordinator.AscendHybridKVCacheCoordinator.find_longest_cache_hit,
    )
    logger.info(
        "UCM Ascend hybrid KV cache coordinator patch applied: "
        "AscendHybridKVCacheCoordinator.find_longest_cache_hit"
    )
