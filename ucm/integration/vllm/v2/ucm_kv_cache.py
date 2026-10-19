"""Resolve scheduler dispatch plans against Layout and build proxy transfers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .layout import UCMKVCacheLayout
from .layout.view import LAYOUT_DEBUG, layout_debug
from .parallel import ParallelLayout
from .ucm_block_id import make_local_ucm_block_ids
from .ucm_proxy import UCMProxyTransfer
from .ucm_sharding import (
    ParallelRank,
    UCMRecordClass,
    UCMSharding,
    groups_by_attention_type,
)

if TYPE_CHECKING:
    from .ucm_scheduler import UCMConnectorMetadata, UCMGroupDispatchPlan

# Returns the rows (UCM blocks) of a plan this rank writes for a record class.
RowSelector = Callable[[UCMRecordClass, Sequence[bytes]], np.ndarray]


def dump_metadata(
    plan: "UCMGroupDispatchPlan",
    ucm_cache_block_size: int,
    selected_rows: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Per-UCM-block JSON for Store dump: attention type and 0-based block index.

    A plan's blocks are consecutive UCM blocks starting at its token_start;
    selected_rows picks some of them (all by default).
    """
    first_block = plan.token_start // ucm_cache_block_size
    prefix = f'{{"type":"{plan.attention_type.value}","block_index":'
    if selected_rows is None:
        selected_rows = range(len(plan.common_ucm_block_ids))
    return tuple(f"{prefix}{first_block + int(row)}}}" for row in selected_rows)


class UCMTransferBuilder:
    """Build this rank's transfers from plans.

    Each record class is a separate record and uses this rank's shard in the
    UCM block id.
    """

    def __init__(
        self,
        layout: UCMKVCacheLayout,
        sharding: UCMSharding | None = None,
        rank: ParallelRank = (0, 0, 0),
    ) -> None:
        self.layout = layout
        if sharding is None:
            sharding = UCMSharding(layout.spec, ParallelLayout())
        self._record_classes = {
            attention_type: sharding.record_classes(attention_type)
            for attention_type, _ in groups_by_attention_type(layout.spec)
        }
        self._shard_keys = {
            record_class: sharding.shard_key(record_class, rank)
            for record_classes in self._record_classes.values()
            for record_class in record_classes
        }

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
            (
                plan
                for request in metadata.requests.values()
                for plan in request.load_plans
            ),
            self._selected_layer_names(layer_name, layer_id),
            row_selector=None,
            with_metadata=False,
        )

    def build_dump_transfers(
        self,
        metadata: "UCMConnectorMetadata",
        layer_name: str | None = None,
        *,
        layer_id: int | None = None,
        row_selector: RowSelector | None = None,
    ) -> tuple[UCMProxyTransfer, ...]:
        return self._build_transfers(
            (
                plan
                for request in metadata.requests.values()
                for plan in request.dump_plans
            ),
            self._selected_layer_names(layer_name, layer_id),
            row_selector=row_selector,
            with_metadata=True,
        )

    def build_empty_markers(
        self,
        metadata: "UCMConnectorMetadata",
        row_selector: RowSelector | None = None,
    ) -> tuple[UCMProxyTransfer, ...]:
        """Empty records for record classes with no group on this rank.

        A pipeline stage may have no layers of a record class. It still
        stores an empty record so lookup finds every shard.
        """
        markers = []
        for plan, record_class, selected_rows in self._dump_rows(
            metadata, row_selector
        ):
            if any(
                group.group_id in self.layout.group_layouts
                for group in record_class.groups
            ):
                continue
            else:
                local_ucm_block_ids, block_indices = self._local_ucm_block_ids(
                    plan, record_class, selected_rows
                )
                empty_matrix = np.empty((len(selected_rows), 0), dtype=np.uint64)
                markers.append(
                    UCMProxyTransfer(
                        local_ucm_block_ids,
                        block_indices,
                        empty_matrix,
                        empty_matrix,
                        empty_matrix,
                        dump_metadata(
                            plan, self.layout.spec.ucm_cache_block_size, selected_rows
                        ),
                        np.zeros(len(selected_rows), dtype=np.uint64),
                    )
                )
        return tuple(markers)

    def dump_targets(
        self,
        metadata: "UCMConnectorMetadata",
        row_selector: RowSelector | None = None,
    ) -> tuple[tuple[tuple[bytes, ...], np.ndarray], ...]:
        """(UCM block ids, block indices) this rank dumps, for commit or abort."""
        return tuple(
            self._local_ucm_block_ids(plan, record_class, selected_rows)
            for plan, record_class, selected_rows in self._dump_rows(
                metadata, row_selector
            )
        )

    def _dump_rows(
        self, metadata: "UCMConnectorMetadata", row_selector: RowSelector | None
    ) -> Iterator[tuple["UCMGroupDispatchPlan", UCMRecordClass, np.ndarray]]:
        for request in metadata.requests.values():
            for plan in request.dump_plans:
                for record_class in self._record_classes[plan.attention_type]:
                    selected_rows = self._selected_rows(
                        plan, record_class, row_selector
                    )
                    if len(selected_rows):
                        yield plan, record_class, selected_rows

    def _selected_rows(
        self,
        plan: "UCMGroupDispatchPlan",
        record_class: UCMRecordClass,
        row_selector: RowSelector | None,
    ) -> np.ndarray:
        if row_selector is None:
            return np.arange(len(plan.common_ucm_block_ids), dtype=np.int64)
        else:
            return np.asarray(
                row_selector(record_class, plan.common_ucm_block_ids), dtype=np.int64
            )

    def _local_ucm_block_ids(
        self,
        plan: "UCMGroupDispatchPlan",
        record_class: UCMRecordClass,
        selected_rows: np.ndarray,
    ) -> tuple[tuple[bytes, ...], np.ndarray]:
        """Local UCM block ids and block indices of the selected rows on this rank."""
        local_ucm_block_ids = make_local_ucm_block_ids(
            [plan.common_ucm_block_ids[row] for row in selected_rows],
            plan.attention_type,
            record_class.class_index,
            self._shard_keys[record_class],
        )
        first_block = plan.token_start // self.layout.spec.ucm_cache_block_size
        return local_ucm_block_ids, first_block + selected_rows

    def _build_transfers(
        self,
        plans: Iterable["UCMGroupDispatchPlan"],
        layer_names: frozenset[str] | None,
        *,
        row_selector: RowSelector | None,
        with_metadata: bool,
    ) -> tuple[UCMProxyTransfer, ...]:
        transfers = []
        for plan in plans:
            for record_class in self._record_classes[plan.attention_type]:
                selected_rows = self._selected_rows(plan, record_class, row_selector)
                if len(selected_rows) == 0:
                    continue
                group_segments = [
                    segments
                    for segments in self._iter_class_segments(
                        plan, record_class, selected_rows, layer_names
                    )
                    if segments[1].shape[1]
                ]
                if not group_segments:
                    continue
                transfers.append(
                    self._transfer(
                        plan, record_class, selected_rows, group_segments, with_metadata
                    )
                )
        return tuple(transfers)

    def _transfer(
        self,
        plan: "UCMGroupDispatchPlan",
        record_class: UCMRecordClass,
        selected_rows: np.ndarray,
        group_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        with_metadata: bool,
    ) -> UCMProxyTransfer:
        """One transfer from each group's (offsets, ptrs, sizes) of a record."""
        if len(group_segments) == 1:
            ((offsets, ptrs, sizes),) = group_segments
        else:
            ptrs = np.concatenate(
                [group_ptrs for _, group_ptrs, _ in group_segments], axis=1
            )
            # Offsets and sizes repeat on every row, so join the first rows.
            offsets = np.broadcast_to(
                np.concatenate(
                    [group_offsets[0] for group_offsets, _, _ in group_segments]
                ),
                ptrs.shape,
            )
            sizes = np.broadcast_to(
                np.concatenate(
                    [group_sizes[0] for _, _, group_sizes in group_segments]
                ),
                ptrs.shape,
            )
        # Ids in one plan are unique because they come from one request.
        # Requests stay in separate transfers: two requests may share a prefix
        # but load it into different blocks.
        local_ucm_block_ids, block_indices = self._local_ucm_block_ids(
            plan, record_class, selected_rows
        )
        if with_metadata:
            block_metadata = dump_metadata(
                plan, self.layout.spec.ucm_cache_block_size, selected_rows
            )
            record_nbytes = np.full(
                len(selected_rows),
                sum(
                    self.layout.store_layouts[group.group_id].store_bytes
                    for group in record_class.groups
                    if group.group_id in self.layout.store_layouts
                ),
                dtype=np.uint64,
            )
        else:
            block_metadata = ()
            record_nbytes = None
        return UCMProxyTransfer(
            local_ucm_block_ids,
            block_indices,
            ptrs,
            sizes,
            offsets,
            block_metadata,
            record_nbytes,
        )

    def _iter_class_segments(
        self,
        plan: "UCMGroupDispatchPlan",
        record_class: UCMRecordClass,
        selected_rows: np.ndarray,
        layer_names: frozenset[str] | None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(offsets, ptrs, sizes) of each group in one record class.

        Each group has a template built at layout init. It is applied to the
        selected rows' vLLM block ids, and the group's data starts at
        group_record_offset in the record. Block First groups copy a whole
        block (padding included) in one piece.
        """

        ucm_block_size = self.layout.spec.ucm_cache_block_size
        if LAYOUT_DEBUG:
            layout_debug(
                f"plan attention_type={plan.attention_type.value} "
                f"record_class={record_class.class_index} "
                f"tokens=[{plan.token_start},{plan.token_end}) rows={len(selected_rows)} "
                f"groups={[group.group_id for group in record_class.groups]}"
            )
        group_record_offset = 0
        for group, position in zip(record_class.groups, record_class.group_positions):
            if group.group_id not in self.layout.group_layouts:
                continue
            group_layout = self.layout.group_layouts[group.group_id]
            # Normalize at the pickle boundary: int64 ids mixed into the
            # uint64 stride arithmetic below would silently promote.
            plan_vllm_block_ids = np.asarray(
                plan.vllm_block_ids_by_group[position], dtype=np.uint64
            )
            # Each row (UCM block) owns an equal slice of the window block ids.
            vllm_block_ids = plan_vllm_block_ids.reshape(
                len(plan.common_ucm_block_ids), -1
            )[selected_rows].reshape(-1)
            store_layout = self.layout.store_layouts[group.group_id]
            segment_mask = (
                None
                if layer_names is None
                else group_layout.segment_mask(layer_names=layer_names)
            )
            token_offsets = None
            if store_layout.dynamic_token_offsets:
                # Slice start of each row inside its vLLM block.
                first_block = plan.token_start // ucm_block_size
                token_offsets = (
                    (first_block + selected_rows).astype(np.uint64)
                    * ucm_block_size
                    % group_layout.token_block_size
                )
            offsets, ptrs, sizes = store_layout.resolve_matrices(
                vllm_block_ids,
                len(selected_rows),
                token_offsets=token_offsets,
                segment_mask=segment_mask,
            )
            if group_record_offset:
                offsets = np.broadcast_to(
                    offsets[0] + group_record_offset, offsets.shape
                )
            if LAYOUT_DEBUG:
                layout_debug(
                    f"record group={group.group_id} rows={len(selected_rows)} "
                    f"vllm_block_ids={len(vllm_block_ids)} "
                    f"segments_per_ucm_block={ptrs.shape[1]} "
                    f"record_size={group_record_offset + store_layout.store_bytes}"
                )
            yield offsets, ptrs, sizes
            group_record_offset += store_layout.store_bytes
