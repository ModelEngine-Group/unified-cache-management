"""Connect M3 sparse attention to UCM's layerwise KV hooks.

Hotfix (2026-11-21): vllm-ascend >= 0.32.0rc1 routes MiniMax M3/M3.1 sparse
attention through a Triton fused fast path (_run_fused_sparse_attention via
torch.ops.vllm.minimax_m3_fused_sparse_forward) that bypasses
_run_sparse_attention entirely. Without hooking the fused path, only the
dense (non-indexer) layers ever call save_kv_layer, so each dumped block
receives 5 of 65 shards, the Posix commit gate never fires, and lookups
miss 100%. This hotfix adds a symmetric wrapper around
_run_fused_sparse_attention.
"""

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

    def _wrap(original, label):
        @wraps(original)
        def wrapper(self, *args, **kwargs):
            if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
                return original(self, *args, **kwargs)

            connector = get_kv_transfer_group()
            metadata = get_forward_context().attn_metadata
            if not connector.has_connector_metadata() or not isinstance(metadata, dict):
                return original(self, *args, **kwargs)

            # One UCM row owns K, V and Indexer. Wait before either cache is
            # updated and save only after all updates; an Indexer-only hook
            # is too early.
            connector.wait_for_layer_load(self.layer_name)
            result = original(self, *args, **kwargs)
            connector.save_kv_layer(
                self.layer_name, self.kv_cache, metadata[self.layer_name]
            )
            return result

        wrapper._ucm_kv_hooks_patched = True
        logger.info("UCM MiniMax M3 sparse-attention KV hooks applied (%s)", label)
        return wrapper

    layer_cls = getattr(mod, "MiniMaxM3SparseAttention", None)

    original = getattr(layer_cls, "_run_sparse_attention", None)
    if not callable(original):
        raise RuntimeError(
            "UCM MiniMax M3 Ascend KV hooks require "
            "MiniMaxM3SparseAttention._run_sparse_attention; "
            "check compatibility with the installed vllm-ascend version."
        )
    if not getattr(original, "_ucm_kv_hooks_patched", False):
        layer_cls._run_sparse_attention = _wrap(original, "sparse")

    # vllm-ascend >= 0.32.0rc1 fused fast path: forward() takes this path
    # whenever _can_fuse_sparse_prepare() is True, so the hook above never
    # fires. Wrap it the same way.
    fused = getattr(layer_cls, "_run_fused_sparse_attention", None)
    if callable(fused) and not getattr(fused, "_ucm_kv_hooks_patched", False):
        layer_cls._run_fused_sparse_attention = _wrap(fused, "fused-sparse")
