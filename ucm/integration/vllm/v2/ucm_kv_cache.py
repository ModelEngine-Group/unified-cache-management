"""Resolve scheduler dispatch plans against Layout and build proxy transfers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .layout import UCMKVCacheGroupInfo, UCMKVCacheLayout
from .layout.view import LAYOUT_DEBUG, layout_debug
from .ucm_proxy import UCMProxyTransfer
from .ucm_scheduler import dispatch_routes

if TYPE_CHECKING:
    from .ucm_scheduler import UCMConnectorMetadata, UCMGroupDispatchPlan


class UCMTransferBuilder:
    """Interpret plans and associate their keys with HBM/storage byte ranges."""

    def __init__(self, layout: UCMKVCacheLayout) -> None:
        self.layout = layout
        self._routes_by_kind: Mapping[str, tuple[UCMKVCacheGroupInfo, ...]] = dict(
            dispatch_routes(layout.spec)
        )

    def _selected_layer_names(
        self, layer_name: str | None, layer_id: int | None
    ) -> frozenset[str] | None:
        if layer_name is not None and layer_id is not None:
            raise ValueError("Specify either layer_name or layer_id")
        if layer_id is not None:
            return self.layout.layer_names_by_id[layer_id]
        return None if layer_name is None else frozenset((layer_name,))

    def build_load_transfers(
        self,
        metadata: "UCMConnectorMetadata",
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
    ) -> tuple[UCMProxyTransfer, ...]:
        return self._build_transfers(
            (p for r in metadata.requests.values() for p in r.load_plans),
            self._selected_layer_names(layer_name, layer_id),
        )

    def build_dump_transfers(
        self,
        metadata: "UCMConnectorMetadata",
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
    ) -> tuple[UCMProxyTransfer, ...]:
        return self._build_transfers(
            (p for r in metadata.requests.values() for p in r.dump_plans),
            self._selected_layer_names(layer_name, layer_id),
        )

    def _build_transfers(
        self,
        plans: Iterable["UCMGroupDispatchPlan"],
        layer_names: frozenset[str] | None,
    ) -> tuple[UCMProxyTransfer, ...]:
        transfers = []
        for plan in plans:
            groups = [
                g
                for g in self._iter_plan_segments(plan, layer_names)
                if g[2].shape[1]
            ]
            if not groups:
                continue
            ptrs = (
                groups[0][2]
                if len(groups) == 1
                else np.concatenate([g[2] for g in groups], axis=1)
            )
            if len(groups) == 1:
                offsets, sizes = groups[0][1], groups[0][3]
            else:
                offsets = np.broadcast_to(
                    np.concatenate([g[1][0] for g in groups]), ptrs.shape
                )
                sizes = np.broadcast_to(
                    np.concatenate([g[3][0] for g in groups]), ptrs.shape
                )
            # Each plan belongs to one request's hash chain, so keys are unique.
            # Keep requests separate: shared-prefix keys may have distinct targets.
            transfers.append(UCMProxyTransfer(plan.keys, ptrs, sizes, offsets))
        return tuple(transfers)

    def _iter_plan_segments(
        self,
        plan: "UCMGroupDispatchPlan",
        layer_names: frozenset[str] | None,
    ) -> Iterator[tuple[Sequence[bytes], np.ndarray, np.ndarray, np.ndarray]]:
        """One plan's records as per-group arrays, straight off templates.

        The scheduler sends keys and window block ids; each group's
        static template (compiled at layout init -- the offsets/sizes
        grids of one key's window, block-major: block 0's views, block
        1's views, ...) places them inside the key's record, and group
        g's contribution starts at group_record_offset.  All keys of
        a plan resolve in one vectorized pass per group; only the block
        ids (and full-attention sub-span heads) vary per call.  Block
        First groups with whole unfiltered windows take the
        one-span-per-block fast path, paddings riding inside.
        """

        if not plan.keys:
            return
        physical_groups = self._routes_by_kind[plan.hash_group]
        if LAYOUT_DEBUG:
            layout_debug(
                f"plan hash_group={plan.hash_group} "
                f"tokens=[{plan.token_start},{plan.token_end}) "
                f"keys={len(plan.keys)} "
                f"groups={[group.group_id for group in physical_groups]}"
            )
        key_count = len(plan.keys)
        group_record_offset = 0
        for group, blocks in zip(physical_groups, plan.group_block_ids, strict=True):
            if group.group_id not in self.layout.group_layouts:
                continue
            group_layout = self.layout.group_layouts[group.group_id]
            # Normalize at the pickle boundary: int64 ids mixed into the
            # uint64 stride arithmetic below would silently promote.
            blocks = np.asarray(blocks, dtype=np.uint64)
            store_layout = self.layout.store_layouts[group.group_id]
            mask = (
                None
                if layer_names is None
                else group_layout.segment_mask(layer_names=layer_names)
            )
            token_offsets = None
            if store_layout.dynamic_token_offsets:
                first_key = plan.token_start // self.layout.spec.ucm_cache_block_size
                token_offsets = (
                    np.arange(first_key, first_key + key_count, dtype=np.uint64)
                    * self.layout.spec.ucm_cache_block_size
                    % group_layout.token_block_size
                )
            offsets, ptrs, sizes = store_layout.resolve_matrices(
                blocks, key_count, token_offsets=token_offsets, segment_mask=mask
            )
            if group_record_offset:
                offsets = np.broadcast_to(
                    offsets[0] + group_record_offset, offsets.shape
                )
            if LAYOUT_DEBUG:
                layout_debug(
                    f"record group={group.group_id} keys={key_count} "
                    f"blocks={len(blocks)} segments_per_key={ptrs.shape[1]} "
                    f"record_size={group_record_offset + store_layout.store_bytes}"
                )
            yield plan.keys, offsets, ptrs, sizes
            group_record_offset += store_layout.store_bytes
