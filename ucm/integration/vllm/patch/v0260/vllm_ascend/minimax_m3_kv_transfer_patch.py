"""Connect M3 sparse attention to UCM's layerwise KV hooks."""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm_ascend.models.minimax_m3.minimax_m3")
def patch_minimax_m3_kv_hooks(mod):
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )
    from vllm.forward_context import get_forward_context

    layer_cls = getattr(mod, "MiniMaxM3SparseAttention", None)
    original = getattr(layer_cls, "_run_sparse_attention", None)
    if not callable(original):
        raise RuntimeError(
            "UCM MiniMax M3 Ascend KV hooks require "
            "MiniMaxM3SparseAttention._run_sparse_attention; "
            "check compatibility with the installed vllm-ascend version."
        )
    if getattr(original, "_ucm_kv_hooks_patched", False):
        return

    @wraps(original)
    def run_sparse_attention(self, *args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return original(self, *args, **kwargs)

        connector = get_kv_transfer_group()
        metadata = get_forward_context().attn_metadata
        if not connector.has_connector_metadata() or not isinstance(metadata, dict):
            return original(self, *args, **kwargs)

        # One UCM row owns K, V and Indexer. Wait before either cache is updated
        # and save only after both updates; an Indexer-only hook is too early.
        connector.wait_for_layer_load(self.layer_name)
        result = original(self, *args, **kwargs)
        connector.save_kv_layer(
            self.layer_name, self.kv_cache, metadata[self.layer_name]
        )
        return result

    run_sparse_attention._ucm_kv_hooks_patched = True
    layer_cls._run_sparse_attention = run_sparse_attention
    logger.info("UCM MiniMax M3 sparse-attention KV hooks applied")
