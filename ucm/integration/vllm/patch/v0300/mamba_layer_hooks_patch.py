"""Wire wait_for_layer_load / save_kv_layer into hybrid linear-attention forwards.

vLLM's ``@maybe_transfer_kv_layer`` decorator only covers full-attention
layers (``unified_attention_with_output`` / ``unified_mla_attention_with_output``).
GDN/KDA (linear-attention) layers never receive the per-layer KV hooks, so the
UCM per-layer layerwise connector (``hla_connector``) cannot dump a linear
layer right after its forward nor wait for its load before it runs.

Two hook tiers are installed, matched to what each model family exposes:

1. **Op-body hooks (compile-safe, preferred).** The Qwen GDN core
   ``torch.ops.vllm.qwen_gdn_attention_core`` (plus its ``fused_norm_packed``
   and ROCm variants) dispatches into the layer module's ``_forward_core``
   via ``forward_context.no_compile_layers``. Custom ops are opaque to
   Dynamo and are piecewise split points, so a method wrapped at this
   delegation point executes eagerly at runtime even when the surrounding
   module forward is traced and compiled — the same location vLLM itself
   uses for ``@maybe_transfer_kv_layer`` on attention ops, and the site
   vllm-ascend already uses for its built-in
   ``wait_for_kv_layer_from_connector`` inside the GDN op. This tier keeps
   the depth-2 load pipeline alive under ``mode=VLLM_COMPILE`` (the default
   on both platforms).

2. **Module-forward hooks (fallback, eager-only).** Models whose linear
   layers have no custom-op boundary (Kimi KDA on both platforms, GLM's
   indexer/KDA on NPU, GLM's indexer-first MLA wrapper) are wrapped at the
   module ``forward`` instead. That wrapper is pruned by Dynamo tracing, so
   this tier requires ``mode=NONE`` (e.g. ``--enforce-eager``); under
   compiled execution the hooks silently vanish.

Classes covered per tier:

- Tier 1 (op body): ``QwenGatedDeltaNetAttention`` (vllm core, CUDA),
  ``vllm_ascend.ops.gdn.AscendGatedDeltaNetAttention`` (Qwen on NPU).
- Tier 2 (module forward):
  ``vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention``
  ``vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn.KimiGatedDeltaNetAttention``
  ``vllm_ascend.ops.gdn.AscendGatedDeltaNetAttention`` (Qwen/Kimi on NPU)
  ``vllm_ascend.ops.kimi_kda.AscendKimiK3DeltaAttention``
  ``vllm_ascend.models.glm5next.kda.Glm5NextLinearAttention`` (GLM KDA on NPU)

matching the decorator's contract: on entry
``connector.wait_for_layer_load(prefix)``, on exit
``connector.save_kv_layer(prefix, kv_cache, attn_metadata)``.  Both connector
sides are idempotent per step (v2 dedups by layer position; v1's row logic
guards submitted/waited rows), so double firing — e.g. alongside
vllm-ascend's built-in ``wait_for_kv_layer_from_connector`` inside the GDN
custom op, or tier-1 alongside tier-2 in eager mode — is harmless.

Decode batches captured in full graphs replay without Python and lose both
tiers; this matches UCM's prefill-only dump/load design (``wait_for_save``
still catches up undumped layers after the forward).
"""

import torch

from ucm.integration.vllm.patch.utils import when_imported
from ucm.logger import init_logger

logger = init_logger(__name__)

_HOOK_FLAG = "_ucm_mamba_kv_hook_patched"


def _active_connector():
    """Return the KV transfer group when per-layer hooks should fire."""
    try:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
            is_v1_kv_transfer_group,
        )
    except ImportError:
        return None
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return None
    connector = get_kv_transfer_group()
    if not connector.has_connector_metadata():
        return None
    return connector


def _current_attn_metadata(layer_name: str):
    """Read the forward context's attention metadata (dict or list shaped)."""
    try:
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
    except Exception:
        return None
    if isinstance(attn_metadata, dict):
        return attn_metadata.get(layer_name)
    if isinstance(attn_metadata, list) and attn_metadata:
        first = attn_metadata[0]
        if isinstance(first, dict):
            return first.get(layer_name)
        return first
    return attn_metadata


def _wrap_gdn_forward(cls) -> None:
    """Wrap one GDN-family class's own ``forward`` with the KV hooks."""
    original = cls.__dict__.get("forward")
    if original is None:
        return
    if getattr(original, _HOOK_FLAG, False):
        return

    def patched_forward(self, *args, **kwargs):
        layer_name = getattr(self, "prefix", None)
        # Skip side-effecting connector calls while Dynamo traces the region:
        # fullgraph compilation prunes this branch, and eager runs (prefill,
        # piecewise-graph runtime, cudagraph capture warmups) fire the hooks.
        connector = None
        if layer_name and not torch.compiler.is_compiling():
            connector = _active_connector()

        if connector is not None:
            connector.wait_for_layer_load(layer_name)

        result = original(self, *args, **kwargs)

        if connector is not None:
            try:
                connector.save_kv_layer(
                    layer_name,
                    getattr(self, "kv_cache", None),
                    _current_attn_metadata(layer_name),
                )
            except Exception as e:
                logger.warning(
                    f"save_kv_layer failed for {layer_name}: "
                    f"{type(e).__name__}: {e}"
                )
        return result

    setattr(patched_forward, _HOOK_FLAG, True)
    cls.forward = patched_forward
    logger.info(
        "UCM GDN KV-hook patch applied: "
        f"wait_for_layer_load / save_kv_layer wired into {cls.__name__}.forward"
    )


def _wrap_op_delegated_core(cls) -> None:
    """Wrap the core method the GDN custom op dispatches into (tier 1).

    ``torch.ops.vllm.qwen_gdn_attention_core`` (and the
    ``fused_norm_packed`` / ROCm variants, which all funnel into
    ``_forward_core``) resolves the layer module from
    ``forward_context.no_compile_layers[layer_name]`` and calls its
    ``_forward_core``.  A method wrapped at this delegation point runs inside
    the op body: Dynamo never traces it (custom ops are opaque) and the
    piecewise runner invokes it eagerly at runtime, so the hooks survive
    ``mode=VLLM_COMPILE`` — unlike module-forward wraps, which tracing prunes.
    """
    original = cls.__dict__.get("_forward_core") if cls is not None else None
    if original is None:
        return
    if getattr(original, _HOOK_FLAG, False):
        return

    def patched_core(self, *args, **kwargs):
        layer_name = getattr(self, "prefix", None)
        # This body normally executes only inside op dispatch (never traced),
        # but keep the guard so any direct traced call degrades to the
        # pruned-branch behaviour instead of graph-breaking Dynamo.
        connector = None
        if layer_name and not torch.compiler.is_compiling():
            connector = _active_connector()

        if connector is not None:
            connector.wait_for_layer_load(layer_name)

        result = original(self, *args, **kwargs)

        if connector is not None:
            try:
                connector.save_kv_layer(
                    layer_name,
                    getattr(self, "kv_cache", None),
                    _current_attn_metadata(layer_name),
                )
            except Exception as e:
                logger.warning(
                    f"save_kv_layer failed for {layer_name}: "
                    f"{type(e).__name__}: {e}"
                )
        return result

    setattr(patched_core, _HOOK_FLAG, True)
    cls._forward_core = patched_core
    logger.info(
        "UCM GDN op-body KV-hook patch applied: wait_for_layer_load / "
        f"save_kv_layer wired into {cls.__name__}._forward_core "
        "(custom-op delegated, compile-safe)"
    )


def _patch_module_classes(mod, class_names) -> None:
    for name in class_names:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            _wrap_gdn_forward(cls)
        except Exception as e:
            logger.warning(
                f"UCM GDN KV-hook patch failed for {name}: " f"{type(e).__name__}: {e}"
            )


@when_imported("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
def _patch_qwen_gdn(mod):
    _patch_module_classes(
        mod,
        ["QwenGatedDeltaNetAttention"],
    )
    # Tier-1 op-body hooks: the qwen_gdn_attention_core op (all variants)
    # funnels into QwenGatedDeltaNetAttention._forward_core.
    _wrap_op_delegated_core(
        getattr(mod, "QwenGatedDeltaNetAttention", None)
    )


@when_imported("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
def _patch_kimi_gdn(mod):
    _patch_module_classes(
        mod,
        ["KimiGatedDeltaNetAttention"],
    )


# Kimi-K3's full-rank-gate KDA path: KimiK3DeltaAttention defines its own
# forward (projection-overlap pipeline), shadowing the base-class forward
# patched above — so the subclass needs its own wrap. The standalone
# Kimi-Linear path (use_full_rank_gate=False) aliases KimiGatedDeltaNetAttention
# and is already covered.
@when_imported("vllm.models.kimi_k3.nvidia.kda")
def _patch_kimi_k3_delta(mod):
    _patch_module_classes(
        mod,
        ["KimiK3DeltaAttention"],
    )


@when_imported("vllm_ascend.ops.kimi_kda")
def _patch_ascend_kimi_k3_delta(mod):
    _patch_module_classes(
        mod,
        ["AscendKimiK3DeltaAttention"],
    )


@when_imported("vllm_ascend.ops.gdn")
def _patch_ascend_gdn(mod):
    _patch_module_classes(
        mod,
        ["AscendGatedDeltaNetAttention"],
    )
    # Tier-1 op-body hooks on NPU: AscendGatedDeltaNetAttention._forward_core
    # is the op-delegated body (it already opens with vllm-ascend's built-in
    # wait_for_kv_layer_from_connector; our wrapper adds the save side and
    # keeps the wait — the connector dedups double firing per step).
    _wrap_op_delegated_core(
        getattr(mod, "AscendGatedDeltaNetAttention", None)
    )


@when_imported("vllm_ascend.models.glm5next.kda")
def _patch_glm_kda(mod):
    _patch_module_classes(
        mod,
        ["Glm5NextLinearAttention"],
    )


# GLM-5.3's Glm5NextMLAAttention wraps the sparse indexer AND the MLA op in
# one forward; the indexer runs FIRST and reads the small-page (indexer
# cache) before the MLA op's internal @maybe_transfer_kv_layer hook fires.
# The entry wait below orders the (group, pair) load ahead of the indexer;
# the exit save duplicates the op-internal save (deduped by the connector).
# Without this, GLM's indexer reads unloaded cache under the depth-2
# pipeline (v1 avoided it only by preloading every row).
@when_imported("vllm.models.glm5next.nvidia.attention")
def _patch_glm_mla(mod):
    _patch_module_classes(
        mod,
        ["Glm5NextMLAAttention"],
    )


@when_imported("vllm_ascend.models.glm5next.attention")
def _patch_glm_mla_ascend(mod):
    _patch_module_classes(
        mod,
        ["Glm5NextMLAAttention"],
    )
