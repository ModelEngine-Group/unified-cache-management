"""Build the per-group KV-cache addressing plans for connector v2.

One entry point, :func:`build_group_layouts`: walk every KV group of the
parsed spec and hand the runtime views to :class:`KVCacheGroupLayout`,
which flattens them into addressing columns.  Declarations (0.29)
already ride the layer specs -- ``UCMLayerSpec.descriptor`` -- after
``parse_kv_cache_config`` mirrored them; 0.26 layers carry None and the
groups build the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .group import KVCacheGroupLayout

if TYPE_CHECKING:
    from ..ucm_kv_cache import UCMKVCacheSpec
    from ..ucm_proxy import KVCacheValue


def build_group_layouts(
    spec: "UCMKVCacheSpec",
    kv_caches: Mapping[str, "KVCacheValue"],
) -> dict[int, KVCacheGroupLayout]:
    """One :class:`KVCacheGroupLayout` per KV group of the parsed spec."""

    return {
        group.group_id: KVCacheGroupLayout(group, kv_caches)
        for group in spec.groups
    }


__all__ = ["KVCacheGroupLayout", "build_group_layouts"]
