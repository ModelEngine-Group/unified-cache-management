"""Bind parsed layer declarations to tensors and collect registered layouts.

Native configuration interpretation belongs to kv_cache_spec. This module
constructs group layouts from those declarations and the Worker's real tensors,
then collects stage-local group/store layouts and registered layer indexes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .group_layout import KVCacheGroupLayout
from .kv_cache_spec import (
    RestorePolicy,
    UCMKVCacheSpec,
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
    """Registered group and object layouts of this Worker's PP stage."""

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
    ) -> None:
        self.store_layouts = {
            group_id: GroupStoreLayout.build(
                self.spec,
                group_id,
                self.group_layouts[group_id],
                topology,
                rank,
            )
            for group_id in sorted(self.group_layouts)
        }
        # Byte ranges in a stage-local Store record, grouped by restore policy.
        self.store_offsets_by_group: dict[int, int] = {}
        self.store_sizes_by_policy: dict[RestorePolicy, int] = {}
        for restore_policy, group_specs in groups_by_restore_policy(self.spec).items():
            local_group_ids = [
                group_spec.group_id
                for group_spec in group_specs
                if group_spec.group_id in self.store_layouts
            ]
            if not local_group_ids:
                continue
            store_size = 0
            for group_id in local_group_ids:
                self.store_offsets_by_group[group_id] = store_size
                store_size += self.store_layouts[group_id].whole_chunk_nbytes
            self.store_sizes_by_policy[restore_policy] = store_size
