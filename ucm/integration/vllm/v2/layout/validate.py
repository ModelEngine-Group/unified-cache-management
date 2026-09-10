"""Cross-check that the declared and inferred models agree on addressing.

``UCM_V2_DESCRIPTOR_SOURCE=assert`` builds the layout model both ways and
runs this check, so the synthesized declarations can be proven equivalent to
the official ones before anything depends on either.  The comparison covers
what addressing actually consumes -- per-layer components, regions, and
block strides -- which both modes derive independently (the declared mode
resolves bases through the declarations, the inferred mode through the raw
view pointers).  Backing and descriptor granularity is deliberately mode
specific and not compared.
"""

from __future__ import annotations

from .model import LayerSlot, LayoutModel


def assert_models_agree(declared_model: LayoutModel, inferred_model: LayoutModel) -> None:
    if declared_model.num_blocks != inferred_model.num_blocks:
        raise ValueError(
            "Layout models disagree on num_blocks: "
            f"{declared_model.num_blocks} != {inferred_model.num_blocks}"
        )
    declared_names = set(declared_model.slots)
    inferred_names = set(inferred_model.slots)
    if declared_names != inferred_names:
        raise ValueError(
            "Layout models disagree on layer coverage: "
            f"declared_only={sorted(declared_names - inferred_names)}, "
            f"inferred_only={sorted(inferred_names - declared_names)}"
        )
    for name in sorted(declared_names):
        declared_slot: LayerSlot = declared_model.slots[name]
        inferred_slot: LayerSlot = inferred_model.slots[name]
        if declared_slot == inferred_slot:
            continue
        mismatched = [
            index
            for index, (declared, inferred) in enumerate(
                zip(declared_slot.components, inferred_slot.components)
            )
            if declared != inferred
        ]
        raise ValueError(
            f"Layout models disagree on layer {name}: "
            f"components {declared_slot.components!r} != "
            f"{inferred_slot.components!r}, "
            f"regions {declared_slot.regions!r} != "
            f"{inferred_slot.regions!r} "
            f"(first mismatching component index: {mismatched})"
        )
