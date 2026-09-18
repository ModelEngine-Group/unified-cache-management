"""Wire wait_for_layer_load / save_kv_layer into Kimi-K3's custom MLA layer.

vLLM >= 0.27.0's Kimi-K3 uses a custom ``MultiHeadLatentAttention`` whose
``forward`` bypasses ``unified_mla_attention_with_output`` — the custom op
that carries the ``@maybe_transfer_kv_layer`` decorator.  As a result,
``wait_for_layer_load`` and ``save_kv_layer`` are never called.

This patch wraps ``MultiHeadLatentAttention.forward`` to call the hooks
manually, matching the decorator's contract:

  - On entry: ``connector.wait_for_layer_load(layer_name)``
  - On exit:  ``connector.save_kv_layer(layer_name, kv_cache, attn_metadata)``

``save_kv_layer`` fires after the full ``forward`` (including ``o_proj``),
slightly later than the standard decorator which fires after attention only.
This is a minor overlap trade-off, not a correctness issue — the KV cache is
already updated inside ``_forward_attn`` before ``o_proj`` runs.

The wrapper is a no-op when no v1 KV transfer group is active.
"""

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)


@when_imported("vllm.models.kimi_k3.nvidia.mla")
def patch_kimi_k3_mla_kv_hooks(mod):
    """Attach wait_for_layer_load / save_kv_layer to Kimi-K3 MLA."""
    try:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
            is_v1_kv_transfer_group,
        )
    except ImportError:
        logger.warning("Skip Kimi-K3 MLA KV-hook patch: kv_transfer not found")
        return

    MLAcls = getattr(mod, "MultiHeadLatentAttention", None)
    if MLAcls is None:
        logger.warning(
            "Skip Kimi-K3 MLA KV-hook patch: MultiHeadLatentAttention not found"
        )
        return

    original_forward = MLAcls.forward
    if getattr(original_forward, "_ucm_kv_hook_patched", False):
        return

    def patched_forward(self, *args, **kwargs):
        active = has_kv_transfer_group() and is_v1_kv_transfer_group()
        connector = None
        if active:
            connector = get_kv_transfer_group()
            if not connector.has_connector_metadata():
                connector = None

        if connector is not None:
            connector.wait_for_layer_load(self.layer_name)

        result = original_forward(self, *args, **kwargs)

        if connector is not None:
            try:
                from vllm.forward_context import get_forward_context

                ctx = get_forward_context()
                attn_meta_dict = ctx.attn_metadata
                if isinstance(attn_meta_dict, dict):
                    attn_metadata = attn_meta_dict.get(self.layer_name)
                elif isinstance(attn_meta_dict, list):
                    attn_metadata = attn_meta_dict[0].get(self.layer_name)
                else:
                    attn_metadata = attn_meta_dict
                connector.save_kv_layer(self.layer_name, self.kv_cache, attn_metadata)
            except Exception as e:
                logger.warning(
                    f"save_kv_layer failed for {self.layer_name}: "
                    f"{type(e).__name__}: {e}"
                )

        return result

    patched_forward._ucm_kv_hook_patched = True
    MLAcls.forward = patched_forward
    logger.info(
        "UCM Kimi-K3 MLA KV-hook patch applied: "
        "wait_for_layer_load / save_kv_layer wired into "
        "MultiHeadLatentAttention.forward"
    )
