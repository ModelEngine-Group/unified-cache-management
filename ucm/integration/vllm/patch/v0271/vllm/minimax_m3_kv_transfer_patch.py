"""Connect CUDA M3 sparse attention to UCM's layerwise KV hooks."""

from functools import wraps

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm.models.minimax_m3.nvidia.model")
def patch_minimax_m3_kv_hooks(mod):
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
        is_v1_kv_transfer_group,
    )
    from vllm.forward_context import get_forward_context

    layer_cls = getattr(mod, "MiniMaxM3SparseAttention", None)
    original = getattr(layer_cls, "forward", None)
    if not callable(original):
        raise RuntimeError(
            "UCM MiniMax M3 CUDA KV hooks require MiniMaxM3SparseAttention.forward; "
            "check compatibility with the installed vLLM version."
        )
    if getattr(original, "_ucm_kv_hooks_patched", False):
        return

    @wraps(original)
    def forward(self, *args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return original(self, *args, **kwargs)

        connector = get_kv_transfer_group()
        context = get_forward_context()
        metadata = context.attn_metadata
        if not connector.has_connector_metadata() or not isinstance(metadata, dict):
            return original(self, *args, **kwargs)
        if (
            not isinstance(context.slot_mapping, dict)
            or self.layer_name not in context.slot_mapping
        ):
            return original(self, *args, **kwargs)

        # CUDA writes both caches before _run_attention. Bracket forward so the
        # load precedes those writes; saving after o_proj only delays overlap.
        connector.wait_for_layer_load(self.layer_name)
        result = original(self, *args, **kwargs)
        connector.save_kv_layer(
            self.layer_name, self.kv_cache, metadata[self.layer_name]
        )
        return result

    forward._ucm_kv_hooks_patched = True
    layer_cls.forward = forward
    logger.info("UCM MiniMax M3 CUDA sparse-attention KV hooks applied")
