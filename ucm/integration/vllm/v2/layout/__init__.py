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

The model then compiles one :class:`GroupLayout` per KV group: the
placement facts resolved above become precomputed runs and row tables, and
all addressing queries (``segments`` and the record-batch helpers) go
through them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from . import declared, inferred, validate
from .group import GroupLayout
from .model import (
    BackingAllocation,
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


def _build(
    spec: "UCMKVCacheSpec",
    kv_caches: Mapping[str, "KVCacheValue"],
    *,
    num_blocks: int,
    kv_cache_tensors: Sequence[object] = (),
) -> LayoutModel:
    has_declarations = _has_declarations(kv_cache_tensors)
    if has_declarations:
        model = declared.build(
            spec, kv_cache_tensors, kv_caches, num_blocks=num_blocks
        )
    else:
        model = inferred.build(spec, kv_caches, num_blocks=num_blocks)
    # Addressing plans: one per KV group, holding the whole-block runs and
    # the sub-block row tables the queries translate.
    model.group_layouts = {
        group.group_id: GroupLayout(
            group.group_id,
            model.groups[group.group_id],
            token_block_size=group.token_block_size,
            num_blocks=num_blocks,
        )
        for group in spec.groups
        if group.group_id in model.groups
    }
    return model


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
    if source == "assert":
        if not _has_declarations(kv_cache_tensors):
            raise ValueError(
                "UCM_V2_DESCRIPTOR_SOURCE=assert requires declared tensors"
            )
        declared_model = _build(
            spec,
            kv_caches,
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
        )
        # Rebuild only the placement facts for the inferred side; the group
        # plans are compiled per model, and comparing the placements is the
        # point of the check.
        inferred_model = inferred.build(spec, kv_caches, num_blocks=num_blocks)
        validate.assert_models_agree(declared_model, inferred_model)
        return declared_model
    if source == "declared":
        if not _has_declarations(kv_cache_tensors):
            raise ValueError(
                "UCM_V2_DESCRIPTOR_SOURCE=declared requires declared tensors"
            )
        return _build(
            spec,
            kv_caches,
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
        )
    if source not in ("auto", "inferred"):
        raise ValueError(f"Unknown UCM_V2_DESCRIPTOR_SOURCE={source!r}")
    if source == "auto" and _has_declarations(kv_cache_tensors):
        return _build(
            spec,
            kv_caches,
            num_blocks=num_blocks,
            kv_cache_tensors=kv_cache_tensors,
        )
    return _build(spec, kv_caches, num_blocks=num_blocks)


__all__ = [
    "BackingAllocation",
    "ComponentSlot",
    "GroupLayout",
    "LayerSlot",
    "LayoutModel",
    "TensorDescriptor",
    "build_layout_model",
]
