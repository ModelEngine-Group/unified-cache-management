"""Build the per-layer KV-cache layout model for connector v2.

One entry point, :func:`build_layout_model`, produces a
:class:`~ucm.integration.vllm.v2.layout.model.LayoutModel` in one of two
modes, chosen by capability rather than version:

* ``declared`` -- vLLM exposes packed-tensor declarations
  (``kv_cache_tensors`` entries with per-layer offsets and strides); the
  model mirrors them and cross-checks them against the runtime views.
* ``inferred`` -- no declarations exist; runtime views and resolved specs
  are translated into equivalent synthetic declarations.

Both modes fill the same IR, so nothing downstream can tell them apart.
``UCM_V2_DESCRIPTOR_SOURCE`` overrides the automatic choice (``declared``,
``inferred``, or ``assert`` to build both and require them to agree).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from . import declared, inferred, validate
from .model import (
    BackingAllocation,
    BlockRegion,
    ComponentSlot,
    LayerSlot,
    LayoutModel,
    TensorDescriptor,
)

if TYPE_CHECKING:
    from ..ucm_kv_cache import UCMKVCacheSpec
    from ..ucm_proxy import KVCacheValue


def _has_declarations(kv_cache_tensors: Sequence[object]) -> bool:
    return bool(kv_cache_tensors) and all(
        getattr(tensor, "layers", None) is not None for tensor in kv_cache_tensors
    )


def build_layout_model(
    spec: "UCMKVCacheSpec",
    kv_caches: Mapping[str, "KVCacheValue"],
    *,
    num_blocks: int,
    kv_cache_tensors: Sequence[object] = (),
) -> LayoutModel:
    """Build the layout model, preferring vLLM's own declarations."""

    source = os.environ.get("UCM_V2_DESCRIPTOR_SOURCE", "auto").strip().lower()
    source = source or "auto"
    has_declarations = _has_declarations(kv_cache_tensors)
    if source == "assert":
        if not has_declarations:
            raise ValueError(
                "UCM_V2_DESCRIPTOR_SOURCE=assert requires declared tensors"
            )
        declared_model = declared.build(
            spec, kv_cache_tensors, kv_caches, num_blocks=num_blocks
        )
        inferred_model = inferred.build(spec, kv_caches, num_blocks=num_blocks)
        validate.assert_models_agree(declared_model, inferred_model)
        return declared_model
    if source == "declared":
        if not has_declarations:
            raise ValueError(
                "UCM_V2_DESCRIPTOR_SOURCE=declared requires declared tensors"
            )
        return declared.build(
            spec, kv_cache_tensors, kv_caches, num_blocks=num_blocks
        )
    if source not in ("auto", "inferred"):
        raise ValueError(f"Unknown UCM_V2_DESCRIPTOR_SOURCE={source!r}")
    if source == "auto" and has_declarations:
        return declared.build(
            spec, kv_cache_tensors, kv_caches, num_blocks=num_blocks
        )
    return inferred.build(spec, kv_caches, num_blocks=num_blocks)


__all__ = [
    "BackingAllocation",
    "BlockRegion",
    "ComponentSlot",
    "LayerSlot",
    "LayoutModel",
    "TensorDescriptor",
    "build_layout_model",
]
