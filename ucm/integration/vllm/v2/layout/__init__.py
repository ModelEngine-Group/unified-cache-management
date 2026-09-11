"""Build the per-group KV-cache addressing plans for connector v2.

One entry point, :func:`build_group_layouts`: walk every KV group of the
parsed spec and hand the runtime views to :class:`GroupLayout`, which
resolves each component straight off its view.  Declarations (0.29)
already ride the layer specs -- ``UCMLayerSpec.descriptor`` -- after
``parse_kv_cache_config`` mirrored them; 0.26 layers carry None and the
groups build the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .group import GroupLayout

if TYPE_CHECKING:
    from ..ucm_kv_cache import UCMKVCacheSpec
    from ..ucm_proxy import KVCacheValue


def build_group_layouts(
    spec: "UCMKVCacheSpec",
    kv_caches: Mapping[str, "KVCacheValue"],
) -> dict[int, GroupLayout]:
    """One :class:`GroupLayout` per KV group of the parsed spec."""

    return {
        group.group_id: GroupLayout(group, kv_caches) for group in spec.groups
    }


__all__ = ["GroupLayout", "build_group_layouts"]
