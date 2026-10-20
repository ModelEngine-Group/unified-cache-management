"""Wire wait_for_layer_load / save_kv_layer into hybrid linear-attention forwards.

vLLM's ``@maybe_transfer_kv_layer`` decorator only covers full-attention
layers (``unified_attention_with_output`` / ``unified_mla_attention_with_output``).
GDN/KDA (linear-attention) layers never receive the per-layer KV hooks, so the
UCM per-layer layerwise connector (``hla_connector``) cannot dump a linear
layer right after its forward nor wait for its load before it runs.

This patch wraps ``forward`` of the GDN attention family:

- ``vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention``
- ``vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn.KimiGatedDeltaNetAttention``
- ``vllm_ascend.ops.gdn.AscendGatedDeltaNetAttention`` (Qwen/Kimi on NPU)
- ``vllm_ascend.models.glm5next.kda.Glm5NextLinearAttention`` (GLM KDA on NPU)

matching the decorator's contract: on entry
``connector.wait_for_layer_load(prefix)``, on exit
``connector.save_kv_layer(prefix, kv_cache, attn_metadata)``.  Both connector
sides are idempotent per step (v2 dedups by layer position; v1's row logic
guards submitted/waited rows), so double firing — e.g. alongside
vllm-ascend's built-in ``wait_for_kv_layer_from_connector`` inside the GDN
custom op — is harmless.

Compile safety: the GDN ``forward`` lives in Dynamo-traced regions (its core
is an eager custom-op island).  The wrapper checks
``torch.compiler.is_compiling()`` so fullgraph tracing prunes the hook branch
instead of graph-breaking.  Hooks are prefill-relevant; decode captured in a
full graph loses them, which matches UCM's prefill-only dump/load design
(``wait_for_save`` still catches up undumped layers after the forward).
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
