"""Build the per-layer KV-cache layout model for connector v2.

One entry point, :func:`build_layout_model`, produces a
:class:`~ucm.integration.vllm.v2.layout.model.LayoutModel`.  The runtime
views are the addressing source of truth for both generations (0.29
packed declarations / 0.26 per-tensor overlay); what differs is only
whether the model can carry vLLM's own placement declarations for
description and cross-checking.  ``UCM_V2_DESCRIPTOR_SOURCE`` overrides
the automatic choice (``declared`` requires declarations).

The model then compiles one :class:`GroupLayout` per KV group: the
placement facts resolved above become precomputed entries (plus, on
interleaved declared layouts, descriptor-sized whole-batch spans), and
all addressing queries go through them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from . import builder
from .group import DescriptorSpan, GroupLayout
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
    if source not in ("auto", "declared"):
        raise ValueError(
            f"Unknown UCM_V2_DESCRIPTOR_SOURCE={source!r} (assert mode was "
            "retired with the inferred builder; the declared builder now "
            "serves both generations)"
        )
    has_declarations = _has_declarations(kv_cache_tensors)
    if source == "declared" and not has_declarations:
        raise ValueError(
            "UCM_V2_DESCRIPTOR_SOURCE=declared requires declared tensors"
        )
    model = builder.build(
        spec,
        kv_caches,
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors if has_declarations else (),
    )
    model.group_layouts = {
        group.group_id: GroupLayout(
            group.group_id,
            model.groups[group.group_id],
            token_block_size=group.token_block_size,
            num_blocks=num_blocks,
            descriptors=model.descriptors,
        )
        for group in spec.groups
        if group.group_id in model.groups
    }
    return model


__all__ = [
    "BackingAllocation",
    "ComponentSlot",
    "DescriptorSpan",
    "GroupLayout",
    "LayerSlot",
    "LayoutModel",
    "TensorDescriptor",
    "build_layout_model",
]
