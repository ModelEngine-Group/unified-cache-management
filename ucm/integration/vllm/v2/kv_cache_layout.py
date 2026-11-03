"""Bind parsed layer declarations to tensors and collect registered layouts.

Native configuration interpretation belongs to kv_cache_spec. This module
constructs group layouts from those declarations and the Worker's real tensors,
then collects stage-local group/store layouts and registered layer indexes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .group_layout import KVCacheGroupLayout
from .kv_cache_spec import UCMKVCacheSpec, is_block_first_layout, parse_layer_specs
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
        layer_name_lists: dict[int, list[str]] = {}
        for layer_name, layer_id in self.layer_id_by_name.items():
            layer_name_lists.setdefault(layer_id, []).append(layer_name)
        self.layer_names_by_id = {
            layer_id: frozenset(layer_names)
            for layer_id, layer_names in layer_name_lists.items()
        }

    def build_store_layouts(
        self,
        topology: ParallelTopology,
        rank: ParallelRank,
    ) -> None:
        expected = set(self.group_layouts)
        self.store_layouts = {
            group_id: GroupStoreLayout.build(
                self.spec,
                group_id,
                self.group_layouts[group_id],
                topology,
                rank,
            )
            for group_id in sorted(expected)
        }
