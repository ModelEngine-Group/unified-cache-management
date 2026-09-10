"""Build the layout model from vLLM's packed-tensor declarations.

This is the primary mode on vLLM 0.29+: ``kv_cache_tensors`` states where
every layer's blocks sit (``offset + L * layer_stride + b * block_stride``
from one shared backing), and the runtime views carry the page geometry.
The build is a near-identity mapping of the declarations plus a cross-check
against the views, so a vLLM layout change that the declarations and the
views disagree on fails loudly here instead of corrupting transfers.

The backing base pointer is not part of the declarations (they are computed
before allocation); it is anchored from the first structure of declaration
zero's first layer, assumed to sit at its page offset 0, and every other
structure is validated relative to that anchor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import geometry
from .model import (
    BackingAllocation,
    BlockRegion,
    ComponentSlot,
    LayerSlot,
    LayoutModel,
    TensorDescriptor,
    layout_debug,
)

if TYPE_CHECKING:
    from ..ucm_kv_cache import UCMKVCacheSpec
    from ..ucm_proxy import KVCacheValue


@dataclass(frozen=True)
class _Declaration:
    descriptor_id: int
    layers: tuple[str, ...]
    offset: int
    layer_stride: int
    block_stride: int


def _first_base(
    components: tuple[ComponentSlot, ...], regions: tuple[BlockRegion, ...]
) -> int:
    if components:
        return components[0].base_ptr
    if regions:
        return regions[0].base_ptr
    raise ValueError("Layer registered no addressable structure")


def build(
    spec: "UCMKVCacheSpec",
    kv_cache_tensors: Sequence[object],
    kv_caches: Mapping[str, "KVCacheValue"],
    *,
    num_blocks: int,
) -> LayoutModel:
    if not kv_cache_tensors:
        raise ValueError("declared layout mode requires kv_cache_tensors")

    declarations: list[_Declaration] = []
    declared_at: dict[str, tuple[int, int]] = {}
    sizes: set[int] = set()
    for entry in kv_cache_tensors:
        layers = tuple(str(name) for name in getattr(entry, "layers", ()) or ())
        if not layers:
            raise ValueError("A declared KV cache tensor covers no layers")
        declaration = _Declaration(
            descriptor_id=len(declarations),
            layers=layers,
            offset=int(getattr(entry, "offset")),
            layer_stride=int(getattr(entry, "layer_stride")),
            block_stride=int(getattr(entry, "block_stride")),
        )
        if (
            declaration.offset < 0
            or declaration.layer_stride < 0
            or declaration.block_stride <= 0
        ):
            raise ValueError(f"Invalid declared placement: {declaration}")
        for position, name in enumerate(layers):
            if name in declared_at:
                raise ValueError(
                    f"Layer {name} is covered by more than one declared tensor"
                )
            declared_at[name] = (declaration.descriptor_id, position)
        declarations.append(declaration)
        sizes.add(int(getattr(entry, "size")))

    # vLLM 0.29 allocates exactly one backing per worker; every declaration
    # reports its total size.
    if len(sizes) != 1:
        raise ValueError(
            f"Declared KV cache tensors disagree on the backing size: {sorted(sizes)}"
        )
    backing_size = sizes.pop()

    groups: dict[int, tuple[LayerSlot, ...]] = {}
    group_states: dict[int, bool] = {}
    first_bases: dict[str, int] = {}
    for group in spec.groups:
        group_states[group.group_id] = group.is_state_snapshot
        group_slots: list[LayerSlot] = []
        for layer in sorted(
            group.layers,
            key=lambda item: (item.layer_index, item.layer_name),
        ):
            if layer.layer_name not in kv_caches:
                raise ValueError(f"Missing KV cache tensor for {layer.layer_name}")
            if layer.layer_name not in declared_at:
                raise ValueError(
                    f"No declared tensor covers layer {layer.layer_name}"
                )
            components, regions = geometry.layer_structures(
                kv_caches[layer.layer_name],
                layer,
                num_blocks=num_blocks,
                state_snapshot=group.is_state_snapshot,
            )
            if group.is_state_snapshot and any(
                component.rows_per_block != 1 for component in components
            ):
                raise ValueError(
                    "State components require exactly one "
                    f"physical row per vLLM block for {layer.layer_name}"
                )
            first_bases[layer.layer_name] = _first_base(components, regions)
            group_slots.append(
                LayerSlot(
                    layer.layer_name,
                    layer.layer_index,
                    group.group_id,
                    components,
                    regions,
                )
            )
        groups[group.group_id] = tuple(group_slots)

    seen = set(first_bases)
    if seen != set(declared_at):
        raise ValueError(
            "Declared tensors and the cache spec disagree on layer coverage: "
            f"spec_only={sorted(seen - set(declared_at))}, "
            f"declared_only={sorted(set(declared_at) - seen)}"
        )

    slots_by_name = {
        slot.layer_name: slot
        for group_slots in groups.values()
        for slot in group_slots
    }

    # Anchor the backing from declaration zero's first layer (assumed at page
    # offset 0); every structure is then validated against the six-line
    # contract, and consecutive layers must sit exactly one layer_stride apart.
    anchor_base = (
        first_bases[declarations[0].layers[0]] - declarations[0].offset
    )
    for name, (descriptor_id, position) in declared_at.items():
        declaration = declarations[descriptor_id]
        page_base = (
            anchor_base + declaration.offset + position * declaration.layer_stride
        )
        slot = slots_by_name[name]
        for structure in (*slot.components, *slot.regions):
            if structure.block_stride != declaration.block_stride:
                raise ValueError(
                    f"Layer {name}: runtime view block stride "
                    f"{structure.block_stride} disagrees with the declared "
                    f"{declaration.block_stride}"
                )
            if (
                structure.base_ptr < page_base
                or structure.base_ptr + structure.payload_bytes
                > page_base + declaration.block_stride
            ):
                raise ValueError(
                    f"Layer {name}: structure at {structure.base_ptr:#x} does "
                    f"not sit inside its declared block page "
                    f"[{page_base:#x}, {page_base + declaration.block_stride:#x})"
                )

    for declaration in declarations:
        previous_base: int | None = None
        for name in declaration.layers:
            base = first_bases[name]
            if (
                previous_base is not None
                and base - previous_base != declaration.layer_stride
            ):
                raise ValueError(
                    f"Declared tensor {declaration.descriptor_id}: layers are "
                    f"not {declaration.layer_stride} bytes apart as declared "
                    f"({previous_base:#x} -> {base:#x})"
                )
            previous_base = base

    # Every registered structure must end inside the backing.  The buffer
    # bound already reaches the last row of the last block, so this holds the
    # descriptor span without assuming padding after the final block
    # (interleaved DSV4 pages are far smaller than their block stride).
    backing_end = anchor_base + backing_size
    for slot in slots_by_name.values():
        for structure in (*slot.components, *slot.regions):
            if structure.base_ptr + structure.buffer_size_bytes > backing_end:
                raise ValueError(
                    f"Layer {slot.layer_name}: structure at "
                    f"{structure.base_ptr:#x} extends beyond the backing end "
                    f"{backing_end:#x}"
                )

    model = LayoutModel(
        backings=(BackingAllocation(0, anchor_base, backing_size),),
        descriptors=tuple(
            TensorDescriptor(
                declaration.descriptor_id,
                0,
                declaration.layers,
                declaration.offset,
                declaration.layer_stride,
                declaration.block_stride,
            )
            for declaration in declarations
        ),
        groups=groups,
        num_blocks=num_blocks,
    )
    for group_id, group_slots in groups.items():
        for slot in group_slots:
            model.log_registration(
                group_id, slot, state=group_states[group_id]
            )
    layout_debug(
        f"layout mode=declared backings=1 "
        f"descriptors={len(declarations)} num_blocks={num_blocks} "
        f"backing_base={anchor_base:#x}"
    )
    return model
