"""Synthesize layout declarations when vLLM does not provide any.

Every structure that the runtime views expose at one base address with one
block stride becomes a synthetic descriptor at ``offset = layer_stride = 0``;
layers that share the address (the legacy ``shared_by`` overlay) share the
descriptor.  The result is a :class:`~ucm.integration.vllm.v2.layout.model.LayoutModel`
indistinguishable in kind from the declared one, so nothing downstream needs
to know which mode produced it.

Ascend 0.26 spec spellings (``compress_ratio``, the C4 storage span reported
as ``block_size``) are already translated by the time they get here: the
semantic layer guarantees ``layer.storage_block_size`` is the number of stored
states one manager block spans.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from . import geometry
from .model import (
    BackingAllocation,
    LayerSlot,
    LayoutModel,
    TensorDescriptor,
    layout_debug,
)

if TYPE_CHECKING:
    from ..ucm_kv_cache import UCMKVCacheSpec
    from ..ucm_proxy import KVCacheValue


def build(
    spec: "UCMKVCacheSpec",
    kv_caches: Mapping[str, "KVCacheValue"],
    *,
    num_blocks: int,
) -> LayoutModel:
    backing_max: dict[int, int] = {}
    descriptor_layers: dict[tuple[int, int], list[str]] = {}
    groups: dict[int, tuple[LayerSlot, ...]] = {}
    group_states: dict[int, bool] = {}

    for group in spec.groups:
        group_states[group.group_id] = group.is_state_snapshot
        group_slots: list[LayerSlot] = []
        for layer in sorted(
            group.layers,
            key=lambda item: (item.layer_index, item.layer_name),
        ):
            if layer.layer_name not in kv_caches:
                raise ValueError(f"Missing KV cache tensor for {layer.layer_name}")
            components = geometry.layer_structures(
                kv_caches[layer.layer_name],
                layer,
                num_blocks=num_blocks,
                state_snapshot=group.is_state_snapshot,
            )
            # Structures sharing (base address, block stride) describe the same
            # set of physical pages; group them under one descriptor.
            for component in components:
                backing_max[component.base_ptr] = max(
                    backing_max.get(component.base_ptr, 0),
                    component.buffer_size_bytes,
                )
                layers = descriptor_layers.setdefault(
                    (component.base_ptr, component.block_stride), []
                )
                if layer.layer_name not in layers:
                    layers.append(layer.layer_name)
            group_slots.append(
                LayerSlot(
                    layer.layer_name,
                    layer.layer_index,
                    group.group_id,
                    components,
                )
            )
        groups[group.group_id] = tuple(group_slots)

    # Materialize one degenerate backing per structure base: offset and
    # layer_stride are zero because every synthesized descriptor sits exactly
    # at its backing base.
    backings = tuple(
        BackingAllocation(backing_id, base_ptr, backing_max[base_ptr])
        for backing_id, base_ptr in enumerate(backing_max)
    )
    backing_index = {
        backing.base_ptr: backing.backing_id for backing in backings
    }
    descriptors = tuple(
        TensorDescriptor(
            descriptor_id,
            backing_index[base_ptr],
            tuple(layers),
            offset=0,
            layer_stride=0,
            block_stride=block_stride,
        )
        for descriptor_id, ((base_ptr, block_stride), layers) in enumerate(
            descriptor_layers.items()
        )
    )
    model = LayoutModel(
        backings=backings,
        descriptors=descriptors,
        groups=groups,
        num_blocks=num_blocks,
        mode="inferred",
    )
    for group_id, group_slots in groups.items():
        for slot in group_slots:
            model.log_registration(
                group_id, slot, state=group_states[group_id]
            )
    layout_debug(
        f"layout mode=inferred backings={len(backings)} "
        f"descriptors={len(descriptors)} num_blocks={num_blocks}"
    )
    return model
