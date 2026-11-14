"""Bind parsed layer declarations to tensors and collect registered layouts.

Native configuration interpretation belongs to kv_cache_spec. This module
constructs group layouts from those declarations and the Worker's real tensors,
then collects distributed Store layouts and registered layer indexes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from .group_layout import GroupLayoutMetadata, KVCacheGroupLayout
from .kv_cache_spec import (
    RestorePolicy,
    UCMKVCacheGroupSpec,
    UCMKVCacheSpec,
    _select_chunk_size,
    groups_by_restore_policy,
    is_block_first_layout,
    parse_layer_specs,
)
from .store_layout import GroupStoreLayout

if TYPE_CHECKING:
    import torch
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from .runtime import ParallelRank, ParallelTopology


def build_group_layouts(
    spec: UCMKVCacheSpec,
    kv_cache_config: "KVCacheConfig",
    kv_caches: Mapping[
        str, "torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]"
    ],
    *,
    num_hidden_layers: int,
) -> dict[int, KVCacheGroupLayout]:
    """Bind previously parsed declarations to actual registered tensor views."""
    block_first_layout = is_block_first_layout(kv_cache_config, spec.device_type)
    layer_specs_by_group = parse_layer_specs(
        spec, kv_cache_config, num_hidden_layers=num_hidden_layers
    )
    return {
        group_id: KVCacheGroupLayout(
            group_id,
            kv_caches,
            layer_specs=layer_specs,
            block_first_layout=block_first_layout,
            state_snapshot=spec.groups[group_id].is_state_snapshot,
            device_type=spec.device_type,
        )
        for group_id, layer_specs in layer_specs_by_group.items()
    }


class UCMKVCacheLayout:
    """Local device ranges and their placement in a DP replica's Store records."""

    def __init__(
        self,
        spec: UCMKVCacheSpec,
        kv_caches: Mapping[
            str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
        ],
        *,
        kv_cache_config: "KVCacheConfig",
        num_hidden_layers: int,
    ) -> None:
        self.spec = spec
        self.group_layouts: Mapping[int, "KVCacheGroupLayout"] = build_group_layouts(
            spec, kv_cache_config, kv_caches, num_hidden_layers=num_hidden_layers
        )
        # A callback may name only attention, while indexer and other caches
        # of the same model layer have distinct registered names/groups.
        self.layer_id_by_name = {
            layer.layer_name: layer.layer_index
            for group_layout in self.group_layouts.values()
            for layer in group_layout.layer_specs
        }
        self.layer_ids = tuple(sorted(set(self.layer_id_by_name.values())))

    def build_store_layouts(
        self,
        topology: ParallelTopology,
        rank: ParallelRank,
        *,
        chunk_size: int | None = None,
    ) -> None:
        local_report = {
            group_id: (
                self.spec.groups[group_id],
                layout.export_metadata(topology.tp_size),
            )
            for group_id, layout in self.group_layouts.items()
            if self.spec.groups[group_id].restore_policy is not None
        }
        reports = [local_report] * topology.pp_size
        if topology.pp_size > 1:
            import torch.distributed as dist
            from vllm.distributed.parallel_state import get_pp_group

            dist.all_gather_object(
                reports, local_report, group=get_pp_group().cpu_group
            )
        groups_by_id: dict[int, dict[int, GroupLayoutMetadata]] = {}
        stored_specs: dict[int, UCMKVCacheGroupSpec] = {}
        for stage, report in enumerate(reports):
            for group_id, (group, metadata) in report.items():
                groups_by_id.setdefault(group_id, {})[stage] = metadata
                previous = stored_specs.setdefault(group.group_id, group)
                if (
                    group.block_token_span != previous.block_token_span
                    or group.restore_policy != previous.restore_policy
                    or group.tail_tokens != previous.tail_tokens
                ):
                    raise ValueError(
                        f"PP stages disagree on restore geometry for KV group {group.group_id}"
                    )
        if topology.pp_size > 1:
            groups = tuple(
                stored_specs.get(group.group_id, replace(group, is_persistent=False))
                for group in self.spec.groups
            )
            self.spec = replace(
                self.spec,
                # GroupStoreLayout checks partial-token alignment against
                # every registered segment across PP stages.
                chunk_size=_select_chunk_size(
                    groups,
                    self.spec.chunk_size,
                    chunk_size,
                    dict.fromkeys(stored_specs, 1),
                    topology.dcp_size,
                ),
                groups=groups,
            )
        # Every stage advances past the groups owned by other stages too.
        self.store_layouts: dict[int, GroupStoreLayout] = {}
        self.store_offsets_by_group: dict[int, int] = {}
        self.store_sizes_by_policy: dict[RestorePolicy, int] = {}
        for restore_policy, group_specs in groups_by_restore_policy(self.spec).items():
            store_size = 0
            for group in group_specs:
                group_id = group.group_id
                layout = GroupStoreLayout.build(
                    self.spec,
                    group_id,
                    self.group_layouts.get(group_id),
                    topology,
                    rank,
                    groups_by_id[group_id],
                )
                self.store_offsets_by_group[group_id] = store_size
                store_size += layout.whole_chunk_nbytes
                if layout.group_layout is not None:
                    self.store_layouts[group_id] = layout
            self.store_sizes_by_policy[restore_policy] = store_size
