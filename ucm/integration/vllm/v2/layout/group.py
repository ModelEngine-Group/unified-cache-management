"""Registered HBM geometry and block/layer addressing, without chunk rules.

Each column describes a physical segment. StoreLayout chooses token ranges,
byte offsets and IO sizes; this module resolves block bases and converts token
ranges using the segments' physical storage granularity.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING

import numpy as np

from . import view
from .view import LayerView

if TYPE_CHECKING:
    import torch

    from .kv_cache import UCMLayerSpec


@dataclass(frozen=True)
class GroupLayoutMetadata:
    """Registration-only facts about one complete local group block; no pointers."""

    sizes: tuple[int, ...]
    states_per_block: tuple[int, ...]
    supports_partial_tokens: bool
    tp_shards: int


@dataclass(frozen=True)
class TensorDescriptor:
    """One native ``KVCacheTensor`` declaration, as handed over.

    Layer ``layers[l]``'s block ``b`` starts at ``offset + l *
    layer_stride + b * block_stride`` bytes into the backing.
    """

    layers: tuple[str, ...]
    offset: int
    layer_stride: int
    block_stride: int


@dataclass(frozen=True)
class BlockFirstView:
    """One group's contiguous per-block span on a Block First layout.

    The group's descriptors tile the block slot back to back (each
    starting where the previous ends, paddings riding inside), so one
    block of this group is one IO span: every layer page of every
    descriptor in it.
    """

    base_ptr: int  # first layer of the first (lowest-offset) descriptor
    block_stride: int
    block_size_bytes: int  # sum of per-descriptor layer_count * layer_stride


class KVCacheGroupLayout:
    """One KV group's addressing facts, distilled once into columns.

    Resolve block IDs and layer selections to physical segment bases.
    StoreLayout chooses ranges and record packing; physical token-to-byte
    conversion lives here with the registered segment geometry.
    """

    def __init__(
        self,
        group_id: int,
        kv_caches: "Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]]",
        *,
        layer_specs: "Sequence[UCMLayerSpec]",
        block_first_layout: bool,
        state_snapshot: bool,
        device_type: str = "npu",
    ) -> None:
        self.group_id = group_id
        self.layer_specs = tuple(
            sorted(layer_specs, key=lambda item: (item.layer_index, item.layer_name))
        )
        self.num_blocks = self.layer_specs[0].num_blocks

        # Walk the group's layers once, in layer order.  One model layer
        # contributes several layer names (attn, indexer.k_cache, swa_cache,
        # compressor states all belong to layer 5), so layer_index is the
        # order key and the name is only a stable tiebreak within the same
        # layer -- the record layout must not depend on the config's
        # enumeration order.
        layer_names: list[str] = []
        layer_ids: list[int] = []
        base_ptrs: list[int] = []
        block_strides: list[int] = []
        states_per_block: list[int] = []
        payload_bytes: list[int] = []
        self.layer_views: dict[str, LayerView] = {}
        for layer in self.layer_specs:
            layer_view = view.build_layer_view(
                kv_caches[layer.layer_name],
                layer,
                state_snapshot=state_snapshot,
                device_type=device_type,
            )
            self.layer_views[layer.layer_name] = layer_view
            segments = layer_view.segments
            if not segments:
                raise ValueError(
                    f"Layer {layer.layer_name} registered no addressable view"
                )
            # Ascend may split K/V or trim allocation padding before registering
            # these views. Only the CUDA/CPU ABI preserves descriptor geometry.
            if device_type != "npu" and layer.descriptor is not None:
                for segment in segments:
                    if segment.block_stride_bytes != layer.descriptor.block_stride:
                        raise ValueError(
                            f"Layer {layer.layer_name}: view block stride "
                            f"{segment.block_stride_bytes} disagrees with the "
                            f"declared {layer.descriptor.block_stride}"
                        )
            for segment in segments:
                layer_names.append(layer.layer_name)
                layer_ids.append(layer.layer_index)
                base_ptrs.append(segment.base_ptr)
                block_strides.append(segment.block_stride_bytes)
                states_per_block.append(segment.states_per_block)
                payload_bytes.append(segment.payload_bytes)

        # Flat columns carry the arithmetic; layer IDs/names select columns.
        self.layer_names: tuple[str, ...] = tuple(layer_names)
        self.layer_ids = np.asarray(layer_ids, dtype=np.uint64)
        self.base_ptrs = np.asarray(base_ptrs, dtype=np.uint64)
        self.block_strides = np.asarray(block_strides, dtype=np.uint64)
        self.states_per_block = np.asarray(states_per_block, dtype=np.uint64)
        self.payload_bytes = np.asarray(payload_bytes, dtype=np.uint64)
        self.supports_partial_tokens = all(
            layer_view.supports_partial_tokens
            for layer_view in self.layer_views.values()
        )
        self.block_first = self._block_first_span() if block_first_layout else None

    def export_metadata(self, tp_size: int) -> GroupLayoutMetadata:
        """Export one complete local block's format/size facts for PP assembly."""
        from .kv_cache import _spec_tp_shards

        tp_shards = lcm(
            *(
                _spec_tp_shards(layer.kv_cache_spec, tp_size)
                for layer in self.layer_specs
            )
        )
        return GroupLayoutMetadata(
            sizes=tuple(map(int, self.payload_bytes)),
            states_per_block=tuple(map(int, self.states_per_block)),
            supports_partial_tokens=self.supports_partial_tokens,
            tp_shards=tp_shards,
        )

    def _block_first_span(self) -> BlockFirstView | None:
        """Use CUDA/CPU BLHNC/BLNHC placement declarations directly.

        The native allocator places a group's descriptors consecutively
        from offset zero in one backing. GLM's special slot descriptors
        have layer-outer strides and do not qualify. Ascend never enters
        this path; its registered components use per-layer segments.
        """
        descriptors: dict[int, TensorDescriptor] = {}
        for layer in self.layer_specs:
            descriptor = layer.descriptor
            if descriptor is None:
                return None
            descriptors[id(descriptor)] = descriptor
        first_descriptor = min(descriptors.values(), key=lambda item: item.offset)
        if any(
            not 0 < descriptor.layer_stride < descriptor.block_stride
            or descriptor.block_stride != first_descriptor.block_stride
            for descriptor in descriptors.values()
        ):
            return None
        return BlockFirstView(
            base_ptr=self.layer_views[first_descriptor.layers[0]].segments[0].base_ptr,
            block_stride=first_descriptor.block_stride,
            block_size_bytes=sum(
                len(descriptor.layers) * descriptor.layer_stride
                for descriptor in descriptors.values()
            ),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def segment_mask(
        self,
        layer_names: "Collection[str] | None" = None,
        layer_ids: Sequence[int] | None = None,
    ) -> np.ndarray | None:
        """Boolean column mask selecting segments (None = every segment).

        Exact names or every cache name of the model layers with these
        IDs (attention and indexer of one layer share the ID).
        """

        if layer_names is not None and layer_ids is not None:
            raise ValueError("Specify either layer_names or layer_ids")
        if layer_ids is not None:
            return np.isin(self.layer_ids, np.asarray(layer_ids, dtype=np.uint64))
        if layer_names is None:
            return None
        wanted_layer_names = set(layer_names)
        return np.asarray(
            [layer_name in wanted_layer_names for layer_name in self.layer_names],
            dtype=np.bool_,
        )

    def resolve_ptrs(
        self,
        vllm_block_ids: Sequence[int] | np.ndarray,
        *,
        segment_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return [block, segment] bases, in caller-supplied block order."""
        vllm_block_ids = np.asarray(vllm_block_ids, dtype=np.uint64)
        segment_columns = slice(None) if segment_mask is None else segment_mask
        return (
            self.base_ptrs[None, segment_columns]
            + vllm_block_ids[:, None] * self.block_strides[None, segment_columns]
        )

    def resolve_block_first_ptrs(
        self, vllm_block_ids: Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """Return one contiguous group-span base per physical block."""
        block_first_view = self.block_first
        assert block_first_view is not None
        vllm_block_ids = np.asarray(vllm_block_ids, dtype=np.uint64)
        return np.uint64(block_first_view.base_ptr) + vllm_block_ids * np.uint64(
            block_first_view.block_stride
        )

    def tokens_to_bytes(
        self,
        tokens: np.ndarray,
        tokens_per_block: int,
        segment_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert token spans/displacements to [range, segment] byte counts.

        The caller supplies the logical block span; no chunk/window metadata is
        retained here. StoreLayout validates physical capability and entry
        alignment once when compiling the fixed ranges.
        """
        segment_columns = slice(None) if segment_mask is None else segment_mask
        return np.multiply.outer(
            tokens, self.payload_bytes[segment_columns]
        ) // np.uint64(tokens_per_block)
