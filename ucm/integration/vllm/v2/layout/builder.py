"""Build the layout model from vLLM's KVCacheConfig and runtime views.

One builder serves both generations, because the runtime views are
always the addressing source of truth: vLLM's allocation logic carves
every layer's view (data_ptr/shape/stride) at its final address, whether
the version packs all layers into one declared backing (0.29
``kv_cache_tensors`` with per-layer strides) or allocates one tensor per
view (0.26 -- per-tensor overlay, the Ascend layout).  Declarations
therefore only feed the model's descriptive layer and three exact
consistency checks against the views:

* every view's block stride equals the declared one,
* every layer's first view sits at its declared page base
  (``backing + offset + position * layer_stride``),
* every view ends inside the declared backing.

Without declarations (vLLM 0.26) the model carries no descriptors and
describes the per-tensor overlay directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _Declaration:
    descriptor_id: int
    layers: tuple[str, ...]
    offset: int
    layer_stride: int
    block_stride: int


def _parse_declarations(
    kv_cache_tensors: Sequence[object],
) -> tuple[tuple[_Declaration, ...], dict[str, tuple[int, int]]]:
    """Parse ``kv_cache_tensors``; undeclared configs yield nothing."""

    declarations: list[_Declaration] = []
    declared_at: dict[str, tuple[int, int]] = {}
    sizes: set[int] = set()
    for entry in kv_cache_tensors:
        layers = tuple(
            str(name) for name in getattr(entry, "layers", ()) or ()
        )
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
    if declarations:
        # vLLM 0.29 allocates exactly one backing per worker; every
        # declaration reports its total size.
        if len(sizes) != 1:
            raise ValueError(
                "Declared KV cache tensors disagree on the backing size: "
                f"{sorted(sizes)}"
            )
    return tuple(declarations), declared_at


def build(
    spec: "UCMKVCacheSpec",
    kv_caches: Mapping[str, "KVCacheValue"],
    *,
    num_blocks: int,
    kv_cache_tensors: Sequence[object] = (),
) -> LayoutModel:
    declarations, declared_at = _parse_declarations(kv_cache_tensors)

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
            if declarations and layer.layer_name not in declared_at:
                raise ValueError(
                    f"No declared tensor covers layer {layer.layer_name}"
                )
            components = geometry.layer_structures(
                kv_caches[layer.layer_name],
                layer,
                num_blocks=num_blocks,
                state_snapshot=group.is_state_snapshot,
            )
            if not components:
                raise ValueError(
                    f"Layer {layer.layer_name} registered no addressable structure"
                )
            placement = declared_at.get(layer.layer_name)
            if placement is not None:
                declaration = declarations[placement[0]]
                for component in components:
                    if component.block_stride != declaration.block_stride:
                        raise ValueError(
                            f"Layer {layer.layer_name}: runtime view block "
                            f"stride {component.block_stride} disagrees with "
                            f"the declared {declaration.block_stride}"
                        )
            first_bases[layer.layer_name] = components[0].base_ptr
            group_slots.append(
                LayerSlot(
                    layer.layer_name,
                    layer.layer_index,
                    group.group_id,
                    components,
                )
            )
        groups[group.group_id] = tuple(group_slots)

    if not declarations:
        model = LayoutModel(
            backings=(),
            descriptors=(),
            groups=groups,
            num_blocks=num_blocks,
            mode="per-tensor",
        )
        _log(model, groups, group_states)
        layout_debug(
            f"layout mode=per-tensor backings=0 descriptors=0 "
            f"num_blocks={num_blocks}"
        )
        return model

    if set(first_bases) != set(declared_at):
        raise ValueError(
            "Declared tensors and the cache spec disagree on layer coverage: "
            f"spec_only={sorted(set(first_bases) - set(declared_at))}, "
            f"declared_only={sorted(set(declared_at) - set(first_bases))}"
        )

    # Anchor the backing from declaration zero's first layer (assumed at
    # page offset 0), then prove the placement arithmetic with one check
    # per layer: the layer's first view must sit exactly at its declared
    # page base.  Every view must end inside the backing.
    anchor_base = first_bases[declarations[0].layers[0]] - declarations[0].offset
    backing_size = int(getattr(kv_cache_tensors[0], "size"))
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
    for group_slots in groups.values():
        for slot in group_slots:
            for component in slot.components:
                if component.base_ptr + component.buffer_size_bytes > backing_end:
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
    _log(model, groups, group_states)
    layout_debug(
        f"layout mode=declared backings=1 "
        f"descriptors={len(declarations)} num_blocks={num_blocks} "
        f"backing_base={anchor_base:#x}"
    )
    return model


def _log(
    model: LayoutModel,
    groups: Mapping[int, tuple[LayerSlot, ...]],
    group_states: Mapping[int, bool],
) -> None:
    for group_id, group_slots in groups.items():
        state = group_states[group_id]
        for slot in group_slots:
            model.log_registration(group_id, slot, state=state)
