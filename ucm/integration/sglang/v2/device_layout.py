"""Layout of each Store V2 object for replicated DSA/DeepSeek V4 and Mamba.

Object fragments are ordered by logical layer, pool, component and physical
buffer. GPU addresses are never part of persistent identity. Draft buffers
attached to a target layer by SGLang are included in that layer's fragments.

A sharded section (Mamba state under attention TP) holds a different byte
range on every rank, but its block identity is still derived from the full
unsharded layout so the same logical block maps to the same physical block on
every rank. Each rank then writes only its own shard at its canonical offset
within that block.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch.distributed as dist

from .runtime import proxy


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class LayerLayout:
    # Each pool has its own row coordinates (KV tokens versus indexer pages).
    pools: tuple
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]


class DeviceLayout:
    def __init__(self, group, namespace: str, model_identity: str, tp_group=None):
        self.group = group
        self.tp_group = tp_group
        self.entries = tuple(sorted(group.entries, key=lambda entry: str(entry.name)))
        # SGLang's pool assembler already chose the hybrid layout; here we
        # only group the entries it produced by their index source. Section
        # names are stable block-id namespaces, not layout classifiers.
        section_names = {"kv": "FA", "swa": "SWA", "mamba": "MAMBA"}
        by_source = {}
        for entry in self.entries:
            source = entry.indices_from_pool
            if source not in section_names or not entry.packed:
                raise ValueError(f"Unsupported Store v2 pool layout: {entry.name}")
            by_source.setdefault(source, []).append(entry)
        if "kv" not in by_source:
            raise ValueError("Store v2 requires a full-attention section")

        self.sections = {
            source: SectionLayout(
                section_names[source],
                entries,
                group.num_layers,
                group.page_size,
                namespace,
                model_identity,
                sharded=(
                    not group.rank_replicated
                    and not all(getattr(e, "tp_replicated", False) for e in entries)
                ),
                tp_group=tp_group,
            )
            for source, entries in sorted(by_source.items())
        }
        self.has_shared_shards = any(
            section.shared_shards for section in self.sections.values()
        )
        devices = {section.device for section in self.sections.values()}
        if len(devices) != 1:
            raise ValueError("Store v2 requires all buffers on one device per rank")
        self.device_id = devices.pop().index
        if self.device_id is None:
            raise ValueError("Store v2 requires an indexed accelerator device")
        self.num_layers = group.num_layers
        self.section_sizes = {s.name: s.total_size for s in self.sections.values()}
        self.identity = fingerprint(
            {source: s.identity for source, s in self.sections.items()}
        )

    def expand(self, transfers, **kwargs):
        by_name = {transfer.name: transfer for transfer in self.group.resolve_transfers(
            transfers, **kwargs
        )}
        sections = []
        for section in self.sections.values():
            selected = tuple(by_name[entry.name] for entry in section.entries
                             if entry.name in by_name)
            if not selected:
                continue
            keys = selected[0].keys
            if len(selected) != len(section.entries) or any(t.keys != keys for t in selected):
                raise ValueError(f"Pools in section {section.name} must share logical pages")
            sections.append((section, selected))
        return sections


class SectionLayout:
    def __init__(self, name, entries, num_layers, page_size, namespace, model_identity,
                 *, sharded=False, tp_group=None):
        self.name = name
        self.entries = entries
        self.layers = []
        facts = []
        devices = set()
        offset = 0
        covered = {entry.name: set() for entry in self.entries}
        for layer in range(num_layers):
            pools, sizes, offsets = [], [], []
            for entry in self.entries:
                mapped = entry.layer_mapping.get(layer)
                if mapped is None:
                    continue
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
                        devices.add(buffer.device)
                        if not buffer.is_contiguous():
                            raise ValueError("Store v2 requires contiguous physical buffers")
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
                    "Pool layer mapping does not cover every physical buffer"
                )
        if len(devices) != 1:
            raise ValueError(f"Store v2 requires section {name} on one device")
        self.device = devices.pop()
        self.total_size = offset
        self.identity = fingerprint((namespace, model_identity, page_size, facts))
        self.shared_shards = bool(
            sharded
            and tp_group is not None
            and dist.is_initialized()
            and dist.get_world_size(tp_group) > 1
        )
        if self.shared_shards:
            self._globalize(tp_group, facts, namespace, model_identity, page_size)

    def _globalize(self, tp_group, facts, namespace, model_identity, page_size):
        """Derive one rank-independent block layout for a sharded section.

        A sharded section (Mamba state sharded across attention-TP ranks) has
        one fragment per (layer, entry, component, buffer) on every rank, but
        each rank's fragment covers only its own slice. ``facts`` therefore
        differs per rank only in the final ``size`` element. This gathers the
        per-rank facts and re-derives:

        * ``identity`` from the full unsharded sizes, so ``block_ids`` map the
          same logical block to the same physical block on every rank;
        * ``total_size`` as the sum of every rank's shard;
        * each fragment's offset as its global base plus the preceding ranks'
          shard sizes, so a rank writes its shard at its canonical position.
        """
        if tp_group is None or not dist.is_initialized():
            return
        world = dist.get_world_size(tp_group)
        if world <= 1:
            return
        rank = dist.get_rank(tp_group)
        all_facts = [None] * world
        dist.all_gather_object(all_facts, facts, group=tp_group)
        structure = [fact[:5] for fact in facts]
        for peer in all_facts:
            if len(peer) != len(facts) or [fact[:5] for fact in peer] != structure:
                raise ValueError(
                    f"Store v2 section {self.name} has an inconsistent shard "
                    "layout across ranks"
                )
        global_sizes = [
            sum(all_facts[r][i][5] for r in range(world)) for i in range(len(facts))
        ]
        global_base = []
        offset = 0
        for size in global_sizes:
            global_base.append(offset)
            offset += size
        rank_prefix = [
            sum(all_facts[r][i][5] for r in range(rank)) for i in range(len(facts))
        ]
        local_offsets = [global_base[i] + rank_prefix[i] for i in range(len(facts))]
        global_facts = [structure[i] + (global_sizes[i],) for i in range(len(facts))]
        self.total_size = offset
        self.identity = fingerprint((namespace, model_identity, page_size, global_facts))
        rebuilt = []
        cursor = 0
        for layout in self.layers:
            count = len(layout.sizes)
            rebuilt.append(
                LayerLayout(
                    layout.pools,
                    layout.sizes,
                    tuple(local_offsets[cursor:cursor + count]),
                )
            )
            cursor += count
        self.layers = rebuilt

    def block_ids(self, keys):
        prefix = (self.identity + ":" + self.name + ":").encode()
        return tuple(
            hashlib.blake2b(prefix + key.encode(), digest_size=16).digest()
            for key in keys
        )

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
    layout: SectionLayout
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
        if not matrices:
            return []
        pointers = np.concatenate(matrices, axis=1)
        result = []
        for row in pointers:
            payload = proxy().BlockPayload()
            payload.pointers = row.tolist()
            payload.sizes = sizes
            payload.offsets = offsets
            result.append(payload)
        return result
