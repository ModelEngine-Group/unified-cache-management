# UCM patch for vllm-ascend 0.22.1:
# keep the fix in UCM instead of modifying the vllm-ascend repository directly.

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.core.single_type_kv_cache_manager")
def patch_ascend_single_type_kv_cache_manager(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0221.vllm_ascend.core import (
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

    from ucm.integration.vllm.patch.v0221.vllm_ascend.patch.platform import (
        patch_kv_cache_coordinator,
    )

    if not hasattr(mod, "AscendHybridKVCacheCoordinator"):
        logger.warning(
            "Skip Ascend hybrid KV cache coordinator patch: "
            "AscendHybridKVCacheCoordinator is missing"
        )
        return

    coordinator_cls = mod.AscendHybridKVCacheCoordinator
    patched_coordinator_cls = patch_kv_cache_coordinator.AscendHybridKVCacheCoordinator
    patch_or_inject(
        coordinator_cls,
        "find_longest_cache_hit",
        patched_coordinator_cls.find_longest_cache_hit,
    )

    if hasattr(coordinator_cls, "find_longest_cache_hit_per_group"):
        patch_or_inject(
            coordinator_cls,
            "find_longest_cache_hit_per_group",
            patched_coordinator_cls.find_longest_cache_hit_per_group,
        )
    else:
        logger.warning(
            "Skip Ascend per-group hybrid KV cache coordinator patch: "
            "find_longest_cache_hit_per_group is missing"
        )

    logger.info(
        "UCM Ascend hybrid KV cache coordinator patch applied: "
        "AscendHybridKVCacheCoordinator.find_longest_cache_hit and "
        "find_longest_cache_hit_per_group"
    )
