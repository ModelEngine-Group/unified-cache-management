"""Resolve scheduler dispatch plans against Layout and build proxy transfers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .layout import UCMKVCacheGroupInfo, UCMKVCacheLayout
from .layout.view import LAYOUT_DEBUG, layout_debug
from .ucm_proxy import UCMProxyTransfer
from .ucm_scheduler import AttentionType, groups_by_attention_type

if TYPE_CHECKING:
    from .ucm_scheduler import UCMConnectorMetadata, UCMGroupDispatchPlan


def dump_metadata(
    plan: "UCMGroupDispatchPlan", ucm_cache_block_size: int
) -> tuple[str, ...]:
    """Per-UCM-block JSON for Store dump: attention type and 0-based block index.

    A plan's ucm_block_ids are consecutive UCM blocks starting at its
    token_start.
    """
    first_block = plan.token_start // ucm_cache_block_size
    prefix = f'{{"type":"{plan.attention_type.value}","block_index":'
    return tuple(
        f"{prefix}{first_block + row}}}" for row in range(len(plan.ucm_block_ids))
    )


class UCMTransferBuilder:
    """Interpret plans and map their UCM block ids to HBM/storage byte ranges."""

    def __init__(self, layout: UCMKVCacheLayout) -> None:
        self.layout = layout
        self._groups_by_attention_type: Mapping[
            AttentionType, tuple[UCMKVCacheGroupInfo, ...]
        ] = dict(groups_by_attention_type(layout.spec))

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
            with_metadata=False,
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
            with_metadata=True,
        )

    def _build_transfers(
        self,
        plans: Iterable["UCMGroupDispatchPlan"],
        layer_names: frozenset[str] | None,
        *,
        with_metadata: bool,
    ) -> tuple[UCMProxyTransfer, ...]:
        transfers = []
        for plan in plans:
            groups = [
                g for g in self._iter_plan_segments(plan, layer_names) if g[2].shape[1]
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
            # A plan's ucm_block_ids come from one request, so they are unique.
            # Keep requests separate: shared-prefix ids may have distinct targets.
            key_metadata = (
                dump_metadata(plan, self.layout.spec.ucm_cache_block_size)
                if with_metadata
                else ()
            )
            transfers.append(
                UCMProxyTransfer(
                    plan.ucm_block_ids,
                    np.arange(
                        plan.token_start // self.layout.spec.ucm_cache_block_size,
                        plan.token_start // self.layout.spec.ucm_cache_block_size
                        + len(plan.ucm_block_ids),
                        dtype=np.int64,
                    ),
                    ptrs,
                    sizes,
                    offsets,
                    key_metadata,
                    (
                        np.full(
                            len(plan.ucm_block_ids),
                            sum(
                                self.layout.store_layouts[g.group_id].store_bytes
                                for g in self._groups_by_attention_type[
                                    plan.attention_type
                                ]
                                if g.group_id in self.layout.store_layouts
                            ),
                            dtype=np.uint64,
                        )
                        if with_metadata
                        else None
                    ),
                )
            )
        return tuple(transfers)

    def _iter_plan_segments(
        self,
        plan: "UCMGroupDispatchPlan",
        layer_names: frozenset[str] | None,
    ) -> Iterator[tuple[Sequence[bytes], np.ndarray, np.ndarray, np.ndarray]]:
        """One plan's records as per-group arrays, straight off templates.

        The scheduler sends UCM block ids and window vLLM block ids; each
        group's static template (compiled at layout init -- the offsets/sizes
        grids of one UCM block's window, block-major: block 0's views, block
        1's views, ...) places them inside the UCM block's record, and group
        g's contribution starts at group_record_offset.  All UCM blocks of
        a plan resolve in one vectorized pass per group; only the vLLM block
        ids (and full-attention sub-span heads) vary per call.  Block
        First groups with whole unfiltered windows take the
        one-span-per-block fast path, paddings riding inside.
        """

        if not plan.ucm_block_ids:
            return
        groups = self._groups_by_attention_type[plan.attention_type]
        if LAYOUT_DEBUG:
            layout_debug(
                f"plan attention_type={plan.attention_type.value} "
                f"tokens=[{plan.token_start},{plan.token_end}) "
                f"ucm_block_ids={len(plan.ucm_block_ids)} "
                f"groups={[group.group_id for group in groups]}"
            )
        ucm_block_count = len(plan.ucm_block_ids)
        group_record_offset = 0
        for group, vllm_block_ids in zip(
            groups, plan.vllm_block_ids_by_group, strict=True
        ):
            if group.group_id not in self.layout.group_layouts:
                continue
            group_layout = self.layout.group_layouts[group.group_id]
            # Normalize at the pickle boundary: int64 ids mixed into the
            # uint64 stride arithmetic below would silently promote.
            vllm_block_ids = np.asarray(vllm_block_ids, dtype=np.uint64)
            store_layout = self.layout.store_layouts[group.group_id]
            mask = (
                None
                if layer_names is None
                else group_layout.segment_mask(layer_names=layer_names)
            )
            token_offsets = None
            if store_layout.dynamic_token_offsets:
                first_block = plan.token_start // self.layout.spec.ucm_cache_block_size
                token_offsets = (
                    np.arange(
                        first_block, first_block + ucm_block_count, dtype=np.uint64
                    )
                    * self.layout.spec.ucm_cache_block_size
                    % group_layout.token_block_size
                )
            offsets, ptrs, sizes = store_layout.resolve_matrices(
                vllm_block_ids,
                ucm_block_count,
                token_offsets=token_offsets,
                segment_mask=mask,
            )
            if group_record_offset:
                offsets = np.broadcast_to(
                    offsets[0] + group_record_offset, offsets.shape
                )
            if LAYOUT_DEBUG:
                layout_debug(
                    f"record group={group.group_id} ucm_block_ids={ucm_block_count} "
                    f"vllm_block_ids={len(vllm_block_ids)} "
                    f"segments_per_ucm_block={ptrs.shape[1]} "
                    f"record_size={group_record_offset + store_layout.store_bytes}"
                )
            yield plan.ucm_block_ids, offsets, ptrs, sizes
            group_record_offset += store_layout.store_bytes
