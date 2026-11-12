"""GLM-5.2 DSA layout: one FA object per logical page, KV + indexer.

Object fragments are ordered by logical layer, pool, component and physical
buffer. GPU addresses are never part of persistent identity. Draft buffers
attached to a target layer by SGLang are included in that layer's fragments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .runtime import proxy


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class LayerLayout:
    # Each pool has its own row coordinates (KV tokens versus indexer pages).
    pools: tuple
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]


class DsaLayout:
    def __init__(self, group, namespace: str, model_identity: str):
        self.group = group
        self.entries = tuple(sorted(group.entries, key=lambda entry: str(entry.name)))
        names = {str(getattr(e.name, "value", e.name)) for e in self.entries}
        # PoolName values in SGLang are lower-case strings.
        if names != {"kv", "indexer"} or not group.rank_replicated:
            raise ValueError(
                "Store v2 currently supports the replicated DSA KV/indexer layout only"
            )
        self.layers = []
        facts = []
        offset = 0
        covered = {entry.name: set() for entry in self.entries}
        for layer in range(group.num_layers):
            pools, sizes, offsets = [], [], []
            for entry in self.entries:
                if entry.indices_from_pool != "kv" or not entry.packed:
                    raise ValueError("DSA requires packed pools sharing KV indices")
                mapped = entry.layer_mapping.get(layer)
                if mapped is None:
                    raise ValueError(f"DSA pool {entry.name} misses layer {layer}")
                indices = (mapped,) if isinstance(mapped, int) else tuple(mapped)
                items = []
                for component_id, component in enumerate(entry.buffer_meta):
                    for index in indices:
                        fragment = (component_id, index)
                        if fragment in covered[entry.name]:
                            raise ValueError(
                                "A physical buffer is mapped to multiple layers"
                            )
                        covered[entry.name].add(fragment)
                        base, stride, size = component[index]
                        buffer = entry.components[component_id][index]
                        if not buffer.is_contiguous():
                            raise ValueError("DSA requires contiguous physical buffers")
                        items.append((int(base), int(stride)))
                        sizes.append(int(size))
                        offsets.append(offset)
                        facts.append(
                            (
                                layer,
                                str(entry.name),
                                component_id,
                                index,
                                str(buffer.dtype),
                                int(size),
                            )
                        )
                        offset += int(size)
                pools.append((entry, np.asarray(items, dtype=np.uint64)))
            self.layers.append(LayerLayout(tuple(pools), tuple(sizes), tuple(offsets)))
        for entry in self.entries:
            expected = sum(len(component) for component in entry.buffer_meta)
            if len(covered[entry.name]) != expected:
                raise ValueError(
                    "DSA layer mapping does not cover every physical buffer"
                )
        self.total_size = offset
        self.identity = fingerprint((namespace, model_identity, group.page_size, facts))
        self.num_layers = group.num_layers

    def block_ids(self, keys):
        prefix = (self.identity + ":FA:").encode()
        return tuple(
            hashlib.blake2b(prefix + key.encode(), digest_size=16).digest()
            for key in keys
        )

    def expand(self, transfers):
        expanded = self.group.resolve_transfers(transfers)
        if not expanded:
            return ()
        by_name = {transfer.name: transfer for transfer in expanded}
        if set(by_name) != {entry.name for entry in self.entries}:
            raise ValueError("Both DSA KV and indexer transfers are required")
        ordered = tuple(by_name[entry.name] for entry in self.entries)
        keys = tuple(ordered[0].keys)
        if any(tuple(t.keys) != keys for t in ordered):
            raise ValueError("DSA KV and indexer must describe the same logical pages")
        return ordered

    def prepare(self, transfers):
        """Snapshot shared indices once, then interpret each pool's row geometry."""
        snapshots, locations = {}, {}
        keys = tuple(transfers[0].keys)
        for entry, transfer in zip(self.entries, transfers):
            indices = transfer.host_indices
            if indices is None:
                raise ValueError(f"{entry.name} has no device indices")
            source = id(indices)
            if source not in snapshots:
                snapshots[source] = indices.detach().cpu()
            rows = entry.prepare_locations(snapshots[source])
            if len(rows) != len(keys):
                raise ValueError("Device page count does not match object keys")
            locations[entry.name] = np.asarray(rows, dtype=np.uint64)
        return PreparedPages(self, keys, locations, transfers)


@dataclass
class PreparedPages:
    layout: DsaLayout
    keys: tuple[str, ...]
    locations: dict
    # Retain source tensors until the job ends, including a sliced index view.
    transfers: tuple

    def payloads(self, layer=None):
        """Build only this layer on load; dump combines every layer in one job."""
        layouts = self.layout.layers if layer is None else (self.layout.layers[layer],)
        matrices, sizes, offsets = [], [], []
        for layout in layouts:
            for entry, items in layout.pools:
                rows = self.locations[entry.name]
                matrices.append(rows[:, None] * items[None, :, 1] + items[None, :, 0])
            sizes.extend(layout.sizes)
            offsets.extend(layout.offsets)
        pointers = np.concatenate(matrices, axis=1)
        result = []
        for row in pointers:
            payload = proxy().BlockPayload()
            payload.pointers = row.tolist()
            payload.sizes = sizes
            payload.offsets = offsets
            result.append(payload)
        return result
