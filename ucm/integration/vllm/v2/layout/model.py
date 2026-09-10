"""Version-neutral layout IR for connector v2.

The IR speaks the vLLM 0.29 layout vocabulary: one backing allocation per
worker, ``KVCacheTensor``-style descriptors that place every layer's blocks
with ``offset + L * layer_stride + b * block_stride``, and per-component
page geometry read from the runtime views.  vLLM versions without the
declaration API are translated into the same IR by ``inferred.py``; nothing
below the ``build_layout_model`` entry point knows which mode produced a
model.

Addressing lives one level up, in :mod:`.group`: each group's slots are
compiled once into precomputed runs and row tables that queries only
translate.  This module holds the resolved placement facts (backings,
descriptors, per-layer components) and the human-readable
:meth:`LayoutModel.describe` summary.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .group import GroupLayout

# Debug trace for the v2 layout: set UCM_V2_LAYOUT_DEBUG=1 to log, on stderr,
# what the layout model registered per layer and which pointer each dispatched
# vLLM block resolves to.  Zero cost when disabled.
LAYOUT_DEBUG = os.environ.get("UCM_V2_LAYOUT_DEBUG", "0") not in ("", "0")


def layout_debug(message: str) -> None:
    if LAYOUT_DEBUG:
        print(f"[ucm-v2-layout] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class BackingAllocation:
    """One physical allocation holding KV cache bytes.

    vLLM 0.29 allocates a single backing per worker and carves the per-layer
    views out of it; older versions allocate one tensor per (set of) layer(s),
    which the inferred mode reports as separate backings.
    """

    backing_id: int
    base_ptr: int
    size_bytes: int


@dataclass(frozen=True)
class TensorDescriptor:
    """Placement of a layer set inside one backing (vLLM 0.29 KVCacheTensor).

    Layer ``L`` (its position in ``layers``) addresses block ``b`` at
    ``offset + L * layer_stride + b * block_stride`` bytes from the backing
    base.  On interleaved models (DSV4) ``block_stride`` is the shared
    manager-slot stride and differs from any single page size; on
    layer-contiguous models (GLM) it equals the page stride.  The inferred
    mode synthesizes one descriptor per runtime view base with
    ``offset = layer_stride = 0``; layers sharing a tensor (the legacy
    ``shared_by`` overlay) then share the descriptor naturally.
    """

    descriptor_id: int
    backing_id: int
    layers: tuple[str, ...]
    offset: int
    layer_stride: int
    block_stride: int


@dataclass(frozen=True)
class ComponentSlot:
    """One addressable component (K, V, latent, conv/SSM state, ...) of a layer.

    ``base_ptr`` is pre-resolved to the component's block-0 / row-0 address,
    so the hot path is ``base_ptr + block_id * block_stride`` plus in-page row
    arithmetic.  A kernel row stores ``states_per_row`` states of
    ``bytes_per_state`` bytes; a stored state may cover several logical tokens
    (DSV4's C4A cache stores one state per four tokens).
    """

    base_ptr: int
    block_stride: int
    row_stride_bytes: int
    rows_per_block: int
    states_per_row: int
    bytes_per_state: int
    buffer_size_bytes: int

    @property
    def states_per_block(self) -> int:
        return self.rows_per_block * self.states_per_row

    @property
    def row_payload_bytes(self) -> int:
        return self.states_per_row * self.bytes_per_state

    @property
    def payload_bytes(self) -> int:
        """Content bytes of one block; padding keeps this <= block_stride."""

        return self.rows_per_block * self.row_payload_bytes


@dataclass(frozen=True)
class LayerSlot:
    """Per-layer resolved placement."""

    layer_name: str
    layer_index: int
    group_id: int
    components: tuple[ComponentSlot, ...]


class LayoutModel:
    """Resolved per-layer placement plus the per-group addressing plans."""

    def __init__(
        self,
        *,
        backings: Sequence[BackingAllocation],
        descriptors: Sequence[TensorDescriptor],
        groups: Mapping[int, tuple[LayerSlot, ...]],
        num_blocks: int,
        mode: str = "",
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.backings = tuple(backings)
        self.descriptors = tuple(descriptors)
        self.groups = dict(groups)
        self.slots = {
            slot.layer_name: slot
            for group_slots in self.groups.values()
            for slot in group_slots
        }
        self.num_blocks = num_blocks
        self.mode = mode
        # Filled by ``build_layout_model`` once the spec's group info is at
        # hand; addressing queries go through these, not through the slots.
        self.group_layouts: dict[int, GroupLayout] = {}

    def log_registration(
        self, group_id: int, slot: LayerSlot, *, state: bool
    ) -> None:
        if not LAYOUT_DEBUG:
            return
        for view_index, component in enumerate(slot.components):
            layout_debug(
                f"register num_blocks={self.num_blocks} "
                f"group={group_id} layer={slot.layer_name} "
                f"view={view_index} state={int(state)} "
                f"base_ptr={component.base_ptr:#x} "
                f"rows_per_block={component.rows_per_block} "
                f"states_per_row={component.states_per_row} "
                f"row_stride={component.row_stride_bytes} "
                f"payload={component.row_payload_bytes} "
                f"bytes_per_state={component.bytes_per_state} "
                f"buffer={component.buffer_size_bytes}"
            )

    def describe(self) -> str:
        """One human-readable page describing the resolved KV cache layout.

        Groups layers by their placement signature so 60-layer models print
        as a handful of lines: same geometry -> one line with the layer
        range, instead of one line per layer.  Every line answers the same
        question -- where do this layer's blocks live and how are they
        addressed -- using the six-line placement arithmetic.
        """

        def _fmt_bytes(value: int) -> str:
            if value == 0:
                return "0"
            if value % (1024**3) == 0:
                return f"{value // 1024**3}G"
            if value % (1024**2) == 0:
                return f"{value // 1024**2}M"
            if value % 1024 == 0:
                return f"{value // 1024}K"
            return str(value)

        def _geometry_signature(slot: LayerSlot) -> tuple:
            # Geometry only: base pointers differ per layer by design, so
            # they must not keep same-geometry layers from collapsing.
            return tuple(
                (
                    component.block_stride,
                    component.row_stride_bytes,
                    component.rows_per_block,
                    component.states_per_row,
                    component.bytes_per_state,
                    component.buffer_size_bytes,
                )
                for component in slot.components
            )

        def _layer_range(names: Sequence[str]) -> str:
            if len(names) == 1:
                return names[0]
            return f"{names[0]} .. {names[-1]} ({len(names)} layers)"

        lines: list[str] = []
        backing_total = sum(b.size_bytes for b in self.backings)
        lines.append(
            f"KV cache layout [{self.mode or 'layout'}]: "
            f"{self.num_blocks} blocks x {len(self.backings)} backing(s), "
            f"total {_fmt_bytes(backing_total)}"
        )
        for backing in self.backings:
            lines.append(
                f"  backing #{backing.backing_id}: "
                f"base={backing.base_ptr:#x} size={_fmt_bytes(backing.size_bytes)}"
            )

        lines.append(
            "  placement: layer L block b -> "
            "backing + offset + L*layer_stride + b*block_stride"
        )
        for descriptor in self.descriptors:
            layers = descriptor.layers
            stride_kind = (
                "layers-interleaved-in-block"
                if descriptor.block_stride > descriptor.layer_stride > 0
                else "layer-contiguous"
                if descriptor.layer_stride > 0
                else "per-tensor-overlay"
            )
            backing = next(
                (b for b in self.backings if b.backing_id == descriptor.backing_id),
                None,
            )
            lines.append(
                f"  tensor #{descriptor.descriptor_id}"
                f" (backing #{descriptor.backing_id}"
                f"{f' base={backing.base_ptr:#x}' if backing else ''}): "
                f"{_layer_range(layers)} | offset={_fmt_bytes(descriptor.offset)} "
                f"layer_stride={_fmt_bytes(descriptor.layer_stride)} "
                f"block_stride={_fmt_bytes(descriptor.block_stride)} "
                f"[{stride_kind}]"
            )

        lines.append("  per-layer geometry:")
        for group_id in sorted(self.groups):
            slots = self.groups[group_id]
            # Collapse slots that share identical geometry into one line.
            collapsed: list[tuple[tuple, list[str]]] = []
            for slot in slots:
                signature = _geometry_signature(slot)
                for existing_signature, names in collapsed:
                    if existing_signature == signature:
                        names.append(slot.layer_name)
                        break
                else:
                    collapsed.append((signature, [slot.layer_name]))
            for _, names in collapsed:
                representative = self.slots[names[0]]
                component = (
                    representative.components[0]
                    if representative.components
                    else None
                )
                parts = [f"group {group_id}: {_layer_range(names)}"]
                if component is not None:
                    parts.append(
                        f"block={_fmt_bytes(component.block_stride)} "
                        f"rows/block={component.rows_per_block} "
                        f"states/row={component.states_per_row} "
                        f"state={component.bytes_per_state}B"
                    )
                    if component.payload_bytes != component.block_stride:
                        parts.append(
                            f"content={_fmt_bytes(component.payload_bytes)} "
                            f"(page-padded)"
                        )
                parts.append(f"components={len(representative.components)}")
                lines.append("    " + " | ".join(parts))

        if self.group_layouts:
            lines.append("  group addressing:")
            for group_id in sorted(self.group_layouts):
                group_layout = self.group_layouts[group_id]
                shared_runs = {
                    entry.run_index
                    for slices in group_layout.layer_slices.values()
                    for entry in slices
                }
                merged = len(shared_runs) < len(group_layout.runs)
                parts = [
                    f"group {group_id}: {len(group_layout.slots)} layers",
                    f"runs={len(group_layout.runs)}"
                    f"{' (merged)' if merged else ''}",
                    f"record/block={_fmt_bytes(group_layout.block_record_size)}",
                ]
                lines.append("    " + " | ".join(parts))
        return "\n".join(lines)
