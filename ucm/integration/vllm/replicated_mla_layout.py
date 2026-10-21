"""TP-independent records for verified replicated target-model MLA caches.

No GQA head splitting is inferred here. CP/PP and sharded payloads retain the
old isolated namespace. Draft and private tensors are not target prefix data.
"""

import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np

MODEL_TYPES = frozenset({"glm_moe_dsa", "deepseek_v4", "deepseek_v41"})


class CacheFamilyStore:
    """Isolate FA/WA records even when sharing one global remote KV pool.

    Separate Posix directories do not namespace Mooncake keys. Scheduler
    hashes remain logical; translate every keyed Store operation here.
    Task handles and native transfer payloads are unchanged.
    """

    def __init__(self, store, family):
        self.store = store
        self.prefix = f"ucm-fawa-record-v1:{family}:".encode()

    def key_for(self, logical_key):
        return hashlib.md5(self.prefix + logical_key).digest()

    def _keys(self, keys):
        return [self.key_for(key) for key in keys]

    def lookup(self, keys):
        return self.store.lookup(self._keys(keys))

    def lookup_on_prefix(self, keys):
        return self.store.lookup_on_prefix(self._keys(keys))

    def lookup_on_reverse(self, keys):
        return self.store.lookup_on_reverse(self._keys(keys))

    def prefetch(self, keys):
        return self.store.prefetch(self._keys(keys))

    def dump_data(self, keys, shards, ptrs, prerequisite_handle=0):
        return self.store.dump_data(self._keys(keys), shards, ptrs, prerequisite_handle)

    def load_data(self, keys, shards, ptrs):
        return self.store.load_data(self._keys(keys), shards, ptrs)

    def dump(self, keys, shards, tensors):
        return self.store.dump(self._keys(keys), shards, tensors)

    def load(self, keys, shards, tensors):
        return self.store.load(self._keys(keys), shards, tensors)

    def __getattr__(self, name):
        return getattr(self.store, name)


def target_config(config):
    model = config.model_config
    hf = getattr(model, "hf_text_config", None) or getattr(model, "hf_config", None)
    return hf if getattr(hf, "model_type", None) in MODEL_TYPES else None


def supports_replicated_mla(config):
    if target_config(config) is None:
        return False
    parallel = config.parallel_config
    return all(
        getattr(parallel, name, 1) == 1
        for name in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    )


def target_layers(config, cache_config):
    """Keep original group IDs but select only transferable target tensors."""
    hf = target_config(config)
    count = getattr(hf, "num_hidden_layers", 0)
    result = {}
    for gid, group in enumerate(cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not getattr(group, "enable_kv_transfer", True) or not getattr(
            spec, "prefix_cacheable", True
        ):
            continue
        members = getattr(spec, "kv_cache_specs", None) or {}
        selected = []
        for name in group.layer_names:
            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
            if count and match and int(match[1]) >= count:
                continue
            if any(part in name.split(".") for part in ("mtp", "draft_model")):
                continue
            inner = members.get(name, spec)
            if not getattr(inner, "prefix_cacheable", True):
                continue
            if getattr(inner, "num_kv_heads", 1) != 1:
                raise ValueError(f"Target KV is not TP-replicated: {name}")
            selected.append(name)
        if selected:
            result[gid] = tuple(sorted(selected))
    return dict(sorted(result.items(), key=lambda item: item[1]))


def build_namespace(config, cache_config):
    """Versioned schema; excludes local TP, draft options and device addresses."""
    if not supports_replicated_mla(config) or cache_config is None:
        return None
    hf = target_config(config)
    tensors = getattr(cache_config, "kv_cache_tensors", ())
    ratios = set(getattr(hf, "compress_ratios", ()) or ())
    packed_v41 = bool(tensors) and (
        hf.model_type == "deepseek_v41"
        or getattr(hf, "kv_source_layer_ids", None)
        or getattr(hf, "index_source_layer_ids", None)
        or (ratios.issubset({0, 1, 2}) and ratios & {1, 2})
        or any(
            getattr(inner, "model_version", None) == "deepseek_v41"
            for group in cache_config.kv_cache_groups
            for inner in (
                getattr(group.kv_cache_spec, "kv_cache_specs", None)
                or {"": group.kv_cache_spec}
            ).values()
        )
    )
    rows = []
    for gid, names in target_layers(config, cache_config).items():
        spec = cache_config.kv_cache_groups[gid].kv_cache_spec
        members = getattr(spec, "kv_cache_specs", None) or {}
        for name in names:
            inner = members.get(name, spec)
            fields = {
                key: str(getattr(inner, key, None))
                for key in (
                    "block_size",
                    "head_size",
                    "dtype",
                    "compress_ratio",
                    "tokens_per_state",
                    "sliding_window",
                    "model_version",
                    "state_content_size_bytes",
                    "kv_quant_mode",
                    "page_size_bytes",
                )
            }
            if packed_v41:
                # Scheduler merges heterogeneous inner specs. Whole-page
                # records are described by the surviving page descriptors,
                # not a representative inner spec that differs from workers.
                fields = {
                    "block_size": str(spec.block_size),
                    "sliding_window": str(getattr(inner, "sliding_window", None)),
                    "prefix_replay_tokens": str(
                        getattr(inner, "prefix_replay_tokens", 0)
                    ),
                }
            rows.append((name, fields))
    if not rows:
        raise ValueError("Replicated MLA namespace has no persistent target tensors")
    semantic = {
        "model": config.model_config.model.rstrip("/").split("/")[-1],
        "revision": getattr(config.model_config, "revision", None),
        "dtype": str(config.model_config.dtype),
        "cache_dtype": str(config.cache_config.cache_dtype),
        "quantization": str(getattr(config.model_config, "quantization", None)),
        "model_type": hf.model_type,
        "compress_ratios": (
            list(getattr(hf, "compress_ratios", ()) or ())[: hf.num_hidden_layers]
            if getattr(hf, "num_hidden_layers", None)
            else getattr(hf, "compress_ratios", None)
        ),
        "sparse_sfa_c8": (getattr(config, "additional_config", None) or {}).get(
            "enable_sparse_sfa_c8", False
        ),
        "sparse_li_c8": (getattr(config, "additional_config", None) or {}).get(
            "enable_sparse_li_c8", False
        ),
        "rows": sorted(rows),
        "groups": sorted(target_layers(config, cache_config).values()),
    }
    if packed_v41:
        names = {
            name
            for _name_group in target_layers(config, cache_config).values()
            for name in _name_group
        }
        semantic["transfer_format"] = "v41-whole-pages-v1"
        semantic["record_schema"] = sorted(
            (
                sorted(members),
                getattr(tensor, "layer_stride", 0)
                or getattr(tensor, "block_stride", 0),
            )
            for tensor in tensors
            if (members := names.intersection(tensor.layers))
        )
    fingerprint = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, default=str).encode()
    ).hexdigest()
    return f"ucm-replicated-mla-v1:{fingerprint}"


class ReplicatedMLAStoreLayout:
    """One record contains every persistent GLM MLA/indexer group.

    Each component remains in its native byte order. Groups must use the same
    logical page size; FAWA has its own compressed/window segment adapter.
    """

    def __init__(self, caches, config, vllm_config, cache_config, layout_factory):
        self.groups = target_layers(vllm_config, cache_config)
        sizes = {
            cache_config.kv_cache_groups[gid].kv_cache_spec.block_size
            for gid in self.groups
        }
        if len(sizes) != 1:
            raise ValueError("Direct replicated MLA groups need equal logical pages")
        self.layouts = {}
        for gid, names in self.groups.items():
            subconfig = SimpleNamespace(
                num_blocks=cache_config.num_blocks,
                kv_cache_groups=[cache_config.kv_cache_groups[gid]],
            )
            # A group can contain non-consecutive transformer layer IDs.
            # Construct each entry separately instead of the generic layout's
            # contiguous layer-row assumption. Preserve every indexer component.
            for name in names:
                self.layouts[(gid, name)] = layout_factory(
                    {name: caches[name]},
                    dict(config, use_layerwise=False),
                    vllm_config,
                    subconfig,
                )
        self.layer_name_to_id = {
            name: index
            for layout in self.layouts.values()
            for name, index in layout.layer_name_to_id.items()
        }
        self.base_ptrs = np.concatenate([x.base_ptrs for x in self.layouts.values()])
        self.buffer_sizes = np.concatenate(
            [x.buffer_sizes for x in self.layouts.values()]
        )
        self.tensor_size_lists = np.concatenate(
            [x.tensor_size_lists for x in self.layouts.values()]
        )
        self.block_stride_lists = np.concatenate(
            [x.block_stride_lists for x in self.layouts.values()]
        )

    @property
    def tensor_size_list(self):
        return self.tensor_size_lists.tolist()

    @property
    def shard_size(self):
        return int(self.tensor_size_lists.sum())

    @property
    def block_size(self):
        return self.shard_size

    def extract_block_addrs(self, block_ids, layer_first=False):
        return self.extract_block_addrs_for_groups(
            tuple(block_ids for _ in range(max(self.groups) + 1))
        )

    def extract_block_addrs_for_groups(self, groups):
        rows = [
            layout.extract_block_addrs(groups[gid])
            for (gid, _name), layout in self.layouts.items()
        ]
        if len({len(row) for row in rows}) != 1:
            raise ValueError("Incomplete replicated MLA component mapping")
        return np.concatenate(rows, axis=1)
