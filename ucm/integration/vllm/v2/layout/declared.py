"""Build the layout model from vLLM's packed-tensor declarations.

This is the primary mode on vLLM 0.29+: ``kv_cache_tensors`` states where
every layer's blocks sit (``offset + L * layer_stride + b * block_stride``
from one shared backing), and the runtime views carry the page geometry.
The build is a near-identity mapping of the declarations with three
cheap, exact checks: every structure's block stride must equal the
declared one, every layer's first structure must sit at its declared page
base (one anchor proves the whole placement arithmetic, subsuming the old
per-structure containment and layer-step checks -- a structure's payload
never exceeds its stride by construction), and every structure must end
inside the backing.  Anything deeper is ``UCM_V2_DESCRIPTOR_SOURCE=assert``
territory: build the inferred model too and require them to agree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import geometry
from .model import (
    BackingAllocation,
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


def _first_base(components: tuple[ComponentSlot, ...]) -> int:
    if not components:
        raise ValueError("Layer registered no addressable structure")
    return components[0].base_ptr


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
            components = geometry.layer_structures(
                kv_caches[layer.layer_name],
                layer,
                num_blocks=num_blocks,
                state_snapshot=group.is_state_snapshot,
            )
            declaration = declarations[declared_at[layer.layer_name][0]]
            for component in components:
                if component.block_stride != declaration.block_stride:
                    raise ValueError(
                        f"Layer {layer.layer_name}: runtime view block stride "
                        f"{component.block_stride} disagrees with the declared "
                        f"{declaration.block_stride}"
                    )
            first_bases[layer.layer_name] = _first_base(components)
            group_slots.append(
                LayerSlot(
                    layer.layer_name,
                    layer.layer_index,
                    group.group_id,
                    components,
                )
            )
        groups[group.group_id] = tuple(group_slots)

    if set(first_bases) != set(declared_at):
        raise ValueError(
            "Declared tensors and the cache spec disagree on layer coverage: "
            f"spec_only={sorted(set(first_bases) - set(declared_at))}, "
            f"declared_only={sorted(set(declared_at) - set(first_bases))}"
        )

    # Anchor the backing from declaration zero's first layer (assumed at page
    # offset 0), then prove the placement arithmetic with one check per layer:
    # the layer's first structure must sit exactly at its declared page base.
    anchor_base = (
        first_bases[declarations[0].layers[0]] - declarations[0].offset
    )
    backing_end = anchor_base + backing_size
    for name, (descriptor_id, position) in declared_at.items():
        declaration = declarations[descriptor_id]
        page_base = (
            anchor_base + declaration.offset + position * declaration.layer_stride
        )
        if first_bases[name] != page_base:
            raise ValueError(
                f"Layer {name}: runtime view at {first_bases[name]:#x} "
                f"disagrees with its declared page base {page_base:#x} "
                f"(backing {anchor_base:#x} + offset {declaration.offset} + "
                f"{position} * layer_stride {declaration.layer_stride})"
            )
    # Every layer's structures must end inside the backing; the buffer bound
    # already reaches the last row of the last block, so this holds the full
    # span without assuming padding after the final block.
    for group_slots in groups.values():
        for slot in group_slots:
            for component in slot.components:
                if (
                    component.base_ptr + component.buffer_size_bytes
                    > backing_end
                ):
                    raise ValueError(
                        f"Layer {slot.layer_name}: structure at "
                        f"{component.base_ptr:#x} extends beyond the backing "
                        f"end {backing_end:#x}"
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
        mode="declared",
    )
    for group_id, group_slots in groups.items():
        state = group_states[group_id]
        for slot in group_slots:
            model.log_registration(group_id, slot, state=state)
    layout_debug(
        f"layout mode=declared backings=1 "
        f"descriptors={len(declarations)} num_blocks={num_blocks} "
        f"backing_base={anchor_base:#x}"
    )
    return model
