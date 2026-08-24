"""Fix hybrid prefix-cache accounting in vLLM-Ascend 0.25.1.

Unlike 0.23.0/0.24.0, the 0.25.1 ``find_longest_cache_hit_per_group``
returns per-group hit-length tuples rather than a single shared hit_length.
The truncation wrapper is therefore applied only to ``find_longest_cache_hit``
which still returns a single shared hit length.
"""

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.core.single_type_kv_cache_manager")
def patch_ascend_single_type_kv_cache_manager(mod):
    logger.debug(f"Patched {mod} called")

    from ucm.integration.vllm.patch.v0251.vllm_ascend.core import (
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

    from ucm.integration.vllm.patch.v0251.vllm_ascend.patch.platform import (
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
    # 0.25.1 find_longest_cache_hit_per_group returns per-group hit-length
    # tuples; the shared-hit-length truncation wrapper is incompatible.
    # Only wrap find_longest_cache_hit.
    for method_name in ("find_longest_cache_hit",):
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
