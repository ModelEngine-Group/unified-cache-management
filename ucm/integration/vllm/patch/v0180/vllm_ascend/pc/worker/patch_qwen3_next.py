from functools import wraps

from ucm.integration.vllm.patch.utils import patch_or_inject, when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


def _patch_empty_gdn_save(mod) -> None:
    original_save = getattr(mod, "maybe_save_kv_layer_to_connector", None)
    if original_save is None or getattr(
        original_save, "_ucm_skip_empty_gdn_save", False
    ):
        return

    @wraps(original_save)
    def skip_empty_gdn_save(layer_name, kv_cache_layer):
        if layer_name == "" and kv_cache_layer == []:
            return
        return original_save(layer_name, kv_cache_layer)

    skip_empty_gdn_save._ucm_skip_empty_gdn_save = True
    mod.maybe_save_kv_layer_to_connector = skip_empty_gdn_save


def _wrap_gdn_forward_core(mod) -> None:
    target_cls = getattr(mod, "Qwen3NextGatedDeltaNet", None)
    if target_cls is None:
        logger.warning("Skip Qwen3Next GDN UCM patch: target class not found.")
        return

    original_forward_core = getattr(target_cls, "_forward_core", None)
    if original_forward_core is None:
        logger.warning("Skip Qwen3Next GDN UCM patch: _forward_core not found.")
        return
    if getattr(original_forward_core, "_ucm_gdn_layerwise_patched", False):
        return

    @wraps(original_forward_core)
    def ucm_forward_core(self, mixed_qkv, b, a, core_attn_out):
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm_ascend.attention.utils import (
            maybe_save_kv_layer_to_connector,
            wait_for_kv_layer_from_connector,
        )
        from vllm_ascend.utils import vllm_version_is

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        should_save = False
        if isinstance(attn_metadata, dict):
            layer_attn_metadata = attn_metadata.get(self.prefix)
            if isinstance(layer_attn_metadata, GDNAttentionMetadata):
                wait_for_kv_layer_from_connector(self.prefix)
                should_save = True

        result = original_forward_core(self, mixed_qkv, b, a, core_attn_out)

        if should_save:
            kv_cache_index = (
                forward_context.virtual_engine if vllm_version_is("0.18.0") else 0
            )
            self_kv_cache = self.kv_cache[kv_cache_index]
            maybe_save_kv_layer_to_connector(self.prefix, list(self_kv_cache))
        return result

    ucm_forward_core._ucm_gdn_layerwise_patched = True
    patch_or_inject(target_cls, "_forward_core", ucm_forward_core)

    ascend_cls = getattr(mod, "AscendQwen3Next_GatedDeltaNet", None)
    if ascend_cls is not None:
        patch_or_inject(ascend_cls, "_forward_core", ucm_forward_core)


@when_imported("vllm_ascend.patch.worker.patch_qwen3_next")
def patch_qwen3_next_gdn_layerwise(mod):
    logger.debug(f"Patched {mod} called")
    _patch_empty_gdn_save(mod)
    _wrap_gdn_forward_core(mod)
