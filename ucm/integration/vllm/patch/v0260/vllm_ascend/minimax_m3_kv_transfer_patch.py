"""Connect M3 sparse attention to UCM's layerwise KV hooks."""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

# MiniMaxM3SparseAttention.forward() chooses between two implementations:
#   * _run_sparse_attention       - eager q/k/v and indexer inserts
#   * _run_fused_sparse_attention - Triton qknorm/rope/cache-insert fusion
# The fused one is taken whenever _can_fuse_sparse_prepare() holds, which is the
# normal case on NPU with Triton installed, bf16 weights and <= 256K tokens.
# Both write the same K/V/Indexer pages, so a hook on the eager method alone
# never fires in that configuration: the sparse layers then save nothing, their
# shards never reach the store and every block lookup misses.
_EAGER_METHOD = "_run_sparse_attention"
_FUSED_METHOD = "_run_fused_sparse_attention"


@when_imported("vllm_ascend.models.minimax_m3.minimax_m3")
def patch_minimax_m3_kv_hooks(mod):
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )
    from vllm.forward_context import get_forward_context

    layer_cls = getattr(mod, "MiniMaxM3SparseAttention", None)
    original = getattr(layer_cls, _EAGER_METHOD, None)
    if not callable(original):
        raise RuntimeError(
            "UCM MiniMax M3 Ascend KV hooks require "
            "MiniMaxM3SparseAttention._run_sparse_attention; "
            "check compatibility with the installed vllm-ascend version."
        )

    def bracket_kv_update(method):
        @wraps(method)
        def hooked(self, *args, **kwargs):
            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return method(self, *args, **kwargs)

            connector = get_kv_transfer_group()
            metadata = get_forward_context().attn_metadata
            if not connector.has_connector_metadata() or not isinstance(metadata, dict):
                return method(self, *args, **kwargs)

            # One UCM row owns K, V and Indexer. Wait before either cache is
            # updated and save only after both updates; an Indexer-only hook is
            # too early. The fused kernel performs the same inserts, so it needs
            # the identical bracket rather than a second, partial hook.
            connector.wait_for_layer_load(self.layer_name)
            result = method(self, *args, **kwargs)
            connector.save_kv_layer(
                self.layer_name, self.kv_cache, metadata[self.layer_name]
            )
            return result

        hooked._ucm_kv_hooks_patched = True
        return hooked

    if getattr(original, "_ucm_kv_hooks_patched", False):
        return

    patched = []
    # The fused method only exists from the release that introduced the Triton
    # fusion; keep working on older versions that have the eager path alone.
    for method_name in (_EAGER_METHOD, _FUSED_METHOD):
        method = getattr(layer_cls, method_name, None)
        if not callable(method):
            continue
        if not getattr(method, "_ucm_kv_hooks_patched", False):
            setattr(layer_cls, method_name, bracket_kv_update(method))
        patched.append(method_name)

    logger.info(
        "UCM MiniMax M3 sparse-attention KV hooks applied: %s", ", ".join(patched)
    )
