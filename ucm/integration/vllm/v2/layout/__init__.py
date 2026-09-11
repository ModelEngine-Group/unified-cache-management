"""Build the per-group KV-cache addressing plans for connector v2.

One entry point, :func:`build_group_layouts`: parse vLLM's packed-tensor
declarations when they exist (0.29), then walk every KV group's layers
and hand the runtime views to :class:`GroupLayout`, which resolves each
component straight off its view.  Without declarations (vLLM 0.26 /
Ascend per-tensor overlay) the groups build the same way with no
descriptors.  ``UCM_V2_DESCRIPTOR_SOURCE=declared`` requires them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .group import GroupLayout, TensorDescriptor

if TYPE_CHECKING:
    from ..ucm_kv_cache import UCMKVCacheSpec
    from ..ucm_proxy import KVCacheValue


def _has_declarations(kv_cache_tensors: Sequence[object]) -> bool:
    return bool(kv_cache_tensors) and all(
        getattr(tensor, "layers", None) is not None for tensor in kv_cache_tensors
    )


def _parse_descriptors(
    kv_cache_tensors: Sequence[object],
) -> tuple[TensorDescriptor, ...]:
    descriptors = []
    for entry in kv_cache_tensors:
        layers = tuple(str(name) for name in getattr(entry, "layers", ()) or ())
        if not layers:
            raise ValueError("A declared KV cache tensor covers no layers")
        descriptor = TensorDescriptor(
            layers=layers,
            offset=int(getattr(entry, "offset")),
            layer_stride=int(getattr(entry, "layer_stride")),
            block_stride=int(getattr(entry, "block_stride")),
        )
        if (
            descriptor.offset < 0
            or descriptor.layer_stride < 0
            or descriptor.block_stride <= 0
        ):
            raise ValueError(f"Invalid declared placement: {descriptor}")
        descriptors.append(descriptor)
    return tuple(descriptors)


def build_group_layouts(
    spec: "UCMKVCacheSpec",
    kv_caches: Mapping[str, "KVCacheValue"],
    *,
    num_blocks: int,
    kv_cache_tensors: Sequence[object] = (),
) -> dict[int, GroupLayout]:
    """One :class:`GroupLayout` per KV group of the parsed spec."""

    source = os.environ.get("UCM_V2_DESCRIPTOR_SOURCE", "auto").strip().lower()
    source = source or "auto"
    if source not in ("auto", "declared"):
        raise ValueError(f"Unknown UCM_V2_DESCRIPTOR_SOURCE={source!r}")
    has_declarations = _has_declarations(kv_cache_tensors)
    if source == "declared" and not has_declarations:
        raise ValueError(
            "UCM_V2_DESCRIPTOR_SOURCE=declared requires declared tensors"
        )
    descriptors = _parse_descriptors(kv_cache_tensors) if has_declarations else ()
    return {
        group.group_id: GroupLayout(
            group.group_id,
            group.layers,
            kv_caches,
            token_block_size=group.token_block_size,
            num_blocks=num_blocks,
            state_snapshot=group.is_state_snapshot,
            descriptors=descriptors,
        )
        for group in spec.groups
    }


__all__ = ["GroupLayout", "TensorDescriptor", "build_group_layouts"]
