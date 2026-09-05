"""UCM Trace hit-rate analysis CLI.

Collects trace logs, parses topology, dispatches to the appropriate
simulation engine (standard / mamba / fawa), and reports four-tier
hit-rate scenarios.

Architecture:
    Data structures: BlockPool, ByteLRUPool, ByteCacheEntry, RequestGroups,
                     GroupContext, GroupSpec, TraceRecord, SimTopology
    Context:         SimContext / MambaContext / FAWAContext
    Simulators:      HitRateSimulator(ABC) → Standard / Mamba / FAWA
    CLI:             LogCollector, AnalysisRunner, main()

All code resides in this single file.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import multiprocessing
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from math import ceil
from pathlib import Path
from typing import Any, Iterable

# ============================================================================
# Constants & regexes
# ============================================================================

GIB = 1024**3
PROMPT_TOKENS_TOTAL_METRICS = (
    "vllm:prompt_tokens_total",
    "prompt_tokens_total",
)
PROMPT_TOKENS_CACHE_HIT_METRICS = (
    'vllm:prompt_tokens_by_source_total{source="local_cache_hit"}',
    'prompt_tokens_by_source_total{source="local_cache_hit"}',
)

AVAILABLE_KV_RE = re.compile(
    r"\b(?:available|current)[_\s-]*(?:kv[_\s-]*)?cache[_\s-]*memory\b"
    r"[^0-9]*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kmgt]?i?b|bytes?)?",
    re.IGNORECASE,
)
ASYNC_SCHED_RE = re.compile(r"Asynchronous scheduling is enabled", re.IGNORECASE)
MAX_BATCHED_TOKENS_RE = re.compile(
    r"['\"]?max_num_batched_tokens['\"]?\s*[:=]\s*(?P<value>\d+)",
    re.IGNORECASE,
)
TP_SIZE_RE = re.compile(
    r"(?:['\"]?tensor[_-]parallel[_-]size['\"]?\s*[:=]\s*|"
    r"--tensor[-_]parallel[-_]size\s+)"
    r"(?P<value>\d+)",
    re.IGNORECASE,
)
DP_SIZE_RE = re.compile(
    r"(?:['\"]?data[_-]parallel[_-]size['\"]?\s*[:=]\s*|"
    r"--data[-_]parallel[-_]size\s+)"
    r"(?P<value>\d+)",
    re.IGNORECASE,
)
PROM_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

_UCM_TRACE_META_RE = re.compile(r"UCMTraceMeta:\s*(?P<fields>.+)", re.IGNORECASE)
_UCM_TRACE_RE = re.compile(
    r"UCMTrace:\s*"
    r"timestamp:\s*(?P<timestamp>\d+(?:\.\d+)?),\s*"
    r"(?:request_id:\s*(?P<request_id>[^,]+),\s*)?"
    r"input_length:\s*(?P<input_length>\d+),\s*"
    r"output_length:\s*(?P<output_length>\d+),\s*"
    r"block_hashes:\s*(?P<block_hashes>\[.*?\])"
)
_TRACE_RE = re.compile(
    r"timestamp:\s*(?P<timestamp>\d+(?:\.\d+)?),\s*"
    r"(?:request_id:\s*(?P<request_id>[^,]+),\s*)?"
    r"input_length:\s*(?P<input_length>\d+),\s*"
    r"output_length:\s*(?P<output_length>\d+),\s*"
    r"ucm_block_ids:\s*(?P<ucm_block_ids>\[.*?\])"
)
_SYSTEM_TIME_RE = re.compile(r"^\[(?P<system_time>\d{4}-\d{2}-\d{2} [^\]]+)\]")


# ============================================================================
# Core data structures
# ============================================================================


class BlockPool:
    """Simulates vLLM's HBM BlockPool: alloc/free/touch/rescue lifecycle.

    All block_ids share one pool (vLLM global BlockPool). The free queue is
    ordered by release time: head = oldest released, tail = newest. ``alloc``
    reuses from the head (oldest); ``free`` with hash appends to tail (survives
    longer), without hash prepends to head (immediate reuse).

    Prefix cache entries are keyed by ``(hash, group_id)`` so different groups
    sharing the same content hash map to different physical blocks.

    block_id 0 is reserved as null_block (never allocated, never freed),
    matching vLLM's BlockPool.__init__.
    """

    def __init__(self, capacity_block_ids: int):
        self._capacity = max(0, capacity_block_ids)
        self._free_deque: OrderedDict[int, None] = OrderedDict()
        self._next_block_id = 1
        self._in_use: dict[int, int] = {}
        self._hash_to_block: dict = {}
        self._block_hashes: dict[int, set] = {}

    def num_free(self) -> int:
        return len(self._free_deque) + (
            self._capacity - self._next_block_id + 1
            if self._next_block_id <= self._capacity
            else 0
        )

    def num_in_use(self) -> int:
        return len(self._in_use)

    def alloc(self) -> int | None:
        # Prefer new block IDs (no cached hash to evict) over freed blocks
        # (which may have a cached hash that would be evicted). This mirrors
        # vLLM's pre-populated free queue where unused blocks sit between
        # no-hash freed blocks (head) and with-hash freed blocks (tail).
        if self._next_block_id <= self._capacity:
            block_id = self._next_block_id
            self._next_block_id += 1
        elif self._free_deque:
            block_id, _ = self._free_deque.popitem(last=False)
        else:
            return None
        self._evict_hashes(block_id)
        self._in_use[block_id] = 1
        return block_id

    def touch(self, hash_val, group_id: int) -> bool:
        key = (hash_val, group_id)
        block_id = self._hash_to_block.get(key)
        if block_id is None:
            return False
        if block_id in self._in_use:
            self._in_use[block_id] += 1
            return True
        del self._free_deque[block_id]
        self._in_use[block_id] = 1
        return True

    def touch_get(self, hash_val, group_id: int) -> int | None:
        key = (hash_val, group_id)
        block_id = self._hash_to_block.get(key)
        if block_id is None:
            return None
        if block_id in self._in_use:
            self._in_use[block_id] += 1
            return block_id
        del self._free_deque[block_id]
        self._in_use[block_id] = 1
        return block_id

    def peek(self, hash_val, group_id: int) -> bool:
        return (hash_val, group_id) in self._hash_to_block

    def cache_block(self, block_id: int, hash_val, group_id: int) -> None:
        key = (hash_val, group_id)
        self._hash_to_block[key] = block_id
        self._block_hashes.setdefault(block_id, set()).add(key)

    def free(self, block_id: int, has_hash: bool) -> None:
        self._in_use[block_id] -= 1
        if self._in_use[block_id] > 0:
            return
        del self._in_use[block_id]
        if has_hash:
            self._free_deque[block_id] = None
        else:
            self._free_deque[block_id] = None
            self._free_deque.move_to_end(block_id, last=False)

    def free_reverse(self, blocks: list[tuple[int, bool]]) -> None:
        for block_id, has_hash in reversed(blocks):
            self.free(block_id, has_hash)

    def _evict_hashes(self, block_id: int) -> None:
        keys = self._block_hashes.pop(block_id, None)
        if keys is None:
            return
        for key in keys:
            cached = self._hash_to_block.get(key)
            if cached == block_id:
                del self._hash_to_block[key]


@dataclass(frozen=True)
class ByteCacheEntry:
    producer_index: int


class RequestGroups:
    """Union-find tracking request lifetime (first appearance -> last hit)."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.first_timestamp: list[float] = []
        self.last_hit_timestamp: list[float | None] = []

    def add(self, timestamp: float) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.first_timestamp.append(timestamp)
        self.last_hit_timestamp.append(None)
        return index

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union_roots(self, roots: Iterable[int]) -> int:
        root_set = {self.find(r) for r in roots}
        if not root_set:
            raise ValueError("cannot union empty request group")
        root = min(root_set, key=lambda i: self.first_timestamp[i])
        for item in root_set:
            if item == root:
                continue
            self.parent[item] = root
            item_last = self.last_hit_timestamp[item]
            if item_last is not None:
                root_last = self.last_hit_timestamp[root]
                self.last_hit_timestamp[root] = (
                    item_last if root_last is None else max(root_last, item_last)
                )
        return root

    def record_hit(self, root: int, timestamp: float) -> None:
        root = self.find(root)
        last = self.last_hit_timestamp[root]
        self.last_hit_timestamp[root] = (
            timestamp if last is None else max(last, timestamp)
        )

    def lifetimes(self) -> list[float]:
        values: list[float] = []
        for i, last in enumerate(self.last_hit_timestamp):
            if self.find(i) != i or last is None:
                continue
            values.append(last - self.first_timestamp[i])
        return values


def _nearest_percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = ceil(len(s) * pct / 100) - 1
    return s[max(0, min(idx, len(s) - 1))]


def _lifetime_stats(values: list[float]) -> dict:
    return {
        "request_lifetime_sample_count": len(values),
        "average_request_lifetime_seconds": (
            sum(values) / len(values) if values else 0.0
        ),
        "p90_request_lifetime_seconds": _nearest_percentile(values, 90),
        "p95_request_lifetime_seconds": _nearest_percentile(values, 95),
    }


class ByteLRUPool:
    """Byte-level LRU pool for UCM DRAM/FS stores.

    Unlike BlockPool (block-count capacity), ByteLRUPool tracks bytes. Entries
    can have different sizes (FA ``fa_file_size`` vs WA ``wa_file_size``),
    so eviction is by byte budget, not entry count.
    """

    def __init__(self, capacity_bytes: int):
        self._capacity = max(0, capacity_bytes)
        self._items: OrderedDict = OrderedDict()
        self._used_bytes = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def get(self, key) -> ByteCacheEntry | None:
        if self._capacity <= 0 or key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key][0]

    def peek(self, key) -> ByteCacheEntry | None:
        if self._capacity <= 0 or key not in self._items:
            return None
        return self._items[key][0]

    def put(self, key, size_bytes: int, entry: ByteCacheEntry) -> None:
        if self._capacity <= 0 or size_bytes > self._capacity:
            return
        if key in self._items:
            old_size = self._items[key][1]
            self._used_bytes -= old_size
            self._items[key] = (entry, size_bytes)
            self._items.move_to_end(key)
        else:
            self._items[key] = (entry, size_bytes)
        self._used_bytes += size_bytes
        self._evict(key)

    def _evict(self, keep_key) -> None:
        while self._used_bytes > self._capacity and len(self._items) > 1:
            evict_key, (_, evict_size) = self._items.popitem(last=False)
            self._used_bytes -= evict_size
            if evict_key == keep_key:
                self._items[evict_key] = (ByteCacheEntry(0), evict_size)
                self._used_bytes += evict_size


@dataclass
class GroupContext:
    """Per-group runtime state for HBM simulation.

    One instance per KV cache group, mirroring vLLM's
    ``SingleTypeKVCacheManager``. Tracks block allocation, hash derivation,
    and release rules specific to this group's manager type.
    """

    group_id: int
    logical_block_size: int
    vllm_hash_block_size: int
    manager_type: str
    sliding_window: int | None = None
    compress_ratio: int = 1
    alignment_tokens: int = 0
    block_ids: list[int | None] = field(default_factory=list)
    num_cached: int = 0

    @property
    def scale_factor(self) -> int:
        return self.logical_block_size // self.vllm_hash_block_size

    def derive_block_hash(self, block_idx: int, chain: list) -> object:
        return chain[(block_idx + 1) * self.scale_factor - 1]

    def reachable(self, block_idx: int) -> bool:
        if self.manager_type != "sliding_window" or self.sliding_window is None:
            return True
        need = ceil((self.sliding_window - 1) / self.logical_block_size)
        per_segment = self.alignment_tokens // self.logical_block_size
        if need >= per_segment:
            return True
        return block_idx % per_segment >= per_segment - need

    def reset(self) -> None:
        self.block_ids = []
        self.num_cached = 0


# ============================================================================
# Simulation context (per-request state passed between phases)
# ============================================================================


@dataclass
class SimContext:
    """Per-request context, passed through _lookup -> _forward_sim -> _request_end_free."""

    req_idx: int
    hbm: BlockPool
    dram: ByteLRUPool
    fs: ByteLRUPool
    hit_roots: set[int] = field(default_factory=set)
    fa_track: list[tuple[int, bool]] = field(default_factory=list)
    prefix_hit_count: int = 0


@dataclass
class MambaContext(SimContext):
    gated_tokens: int = 0
    gated_tier: int = -1
    mamba_track: list[tuple[int, bool]] = field(default_factory=list)
    state_dicts: list[dict[int, str]] = field(default_factory=list)
    mamba_remaining: list[tuple[int, bool]] = field(default_factory=list)


@dataclass
class FAWAContext(SimContext):
    wa_dram: ByteLRUPool | None = None
    fa_groups: list[GroupContext] = field(default_factory=list)
    wa_groups: list[GroupContext] = field(default_factory=list)
    hbm_prefix_tokens: float = 0.0
    ucm_prefix: int = 0
    fa_ext_tier: int = 0
    gated_tokens: int = 0
    gated_tier: int = -1
    wa_track: dict[int, list[tuple[int, bool]]] = field(default_factory=dict)
    wa_remaining: list[tuple[int, bool]] = field(default_factory=list)


# ============================================================================
# Topology / Trace data structures & parsing
# ============================================================================


@dataclass
class GroupSpec:
    name: str
    logical_block_size: int
    manager_type: str
    sliding_window: int | None = None
    compress_ratio: int = 1


@dataclass
class SimTopology:
    """Merged model topology + cache-pool topology.

    Model topology fields are parsed from the trace's UCMTraceMeta line.
    Cache-pool topology fields (unified, num_nodes, dp_size, tp_size) are
    filled in by AnalysisRunner._build_topology.
    """

    # Model topology (from trace)
    model_type: str
    is_mla: bool
    vllm_hash_block_size: int
    trace_hash_block_size: int
    hbm_block_data_size: int
    lcm_block_size: int = 0
    mamba_groups: int = 0
    ucm_hash_block_size: int = 0
    fa_file_size: int = 0
    wa_file_size: int = 0
    alignment_tokens: int = 0
    group_specs: list[GroupSpec] = field(default_factory=list)
    # Cache-pool topology (from CLI args + log facts)
    unified: bool = False
    num_nodes: int = 1
    dp_size: int = 1
    tp_size: int = 1

    @property
    def dp_per_node(self) -> int:
        return max(1, self.dp_size // self.num_nodes) if self.num_nodes > 0 else 1

    def num_dram_pools(self) -> int:
        if self.unified:
            return 1
        return self.num_nodes if self.is_mla else self.dp_size

    def select_pool(self, dp_rank: int, pools: list) -> Any:
        """Common pool-selection logic. Standard/Mamba call once (dram);
        FAWA calls twice (fa_dram, wa_dram)."""
        if self.unified:
            return pools[0]
        if self.is_mla:
            node = min(dp_rank // self.dp_per_node, self.num_nodes - 1)
            return pools[node]
        return pools[dp_rank]

    def dram_per_pool_bytes(self, total_dram: int) -> int:
        if self.unified:
            return total_dram if self.is_mla else total_dram // self.tp_size
        if self.is_mla:
            return total_dram // self.num_nodes
        return total_dram // (self.dp_size * self.tp_size)

    def fs_capacity_bytes(self, total_fs: int) -> int:
        return total_fs if self.is_mla else total_fs // self.tp_size

    def build_group_contexts(self) -> list[GroupContext]:
        at = self.alignment_tokens or self.vllm_hash_block_size
        return [
            GroupContext(
                group_id=i,
                logical_block_size=spec.logical_block_size,
                vllm_hash_block_size=self.vllm_hash_block_size,
                manager_type=spec.manager_type,
                sliding_window=spec.sliding_window,
                compress_ratio=spec.compress_ratio,
                alignment_tokens=at,
            )
            for i, spec in enumerate(self.group_specs)
        ]


@dataclass
class TraceRecord:
    timestamp: float
    input_length: int
    output_length: int
    hash_ids: list[str]
    source: str
    request_id: str | None = None
    system_time: str | None = None


def _extract_int(line: str, name: str) -> int:
    m = re.search(rf"\b{name}=(\d+)", line)
    return int(m.group(1)) if m else 0


def _extract_bool(line: str, name: str) -> bool:
    m = re.search(rf"\b{name}=(true|false)", line, re.IGNORECASE)
    return bool(m) and m.group(1).lower() == "true"


def _extract_str(line: str, name: str) -> str:
    m = re.search(rf"\b{name}=(\w+)", line)
    return m.group(1) if m else ""


def _extract_groups(line: str) -> list[GroupSpec]:
    m = re.search(r"\bgroups=(\[.*?\])", line)
    if not m:
        return []
    try:
        raw = ast.literal_eval(m.group(1))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    specs: list[GroupSpec] = []
    for tup in raw:
        if not isinstance(tup, (tuple, list)) or len(tup) < 3:
            continue
        name = str(tup[0])
        lbs = int(tup[1])
        mtype = str(tup[2])
        sw = int(tup[3]) if len(tup) > 3 and tup[3] is not None else None
        cr = int(tup[4]) if len(tup) > 4 and tup[4] is not None else 1
        specs.append(GroupSpec(name, lbs, mtype, sw, cr))
    return specs


def parse_trace_meta(line: str) -> SimTopology | None:
    if not _UCM_TRACE_META_RE.search(line):
        return None
    model_type = _extract_str(line, "type")
    if model_type not in ("standard", "mamba", "fawa"):
        return None
    topo = SimTopology(
        model_type=model_type,
        is_mla=_extract_bool(line, "is_mla"),
        vllm_hash_block_size=_extract_int(line, "vllm_hash_block_size"),
        hbm_block_data_size=_extract_int(line, "hbm_block_data_size"),
        trace_hash_block_size=_extract_int(line, "trace_hash_block_size"),
    )
    if model_type == "mamba":
        topo.lcm_block_size = _extract_int(line, "lcm_block_size")
        topo.mamba_groups = _extract_int(line, "mamba_groups")
        topo.alignment_tokens = topo.lcm_block_size or topo.vllm_hash_block_size
    elif model_type == "fawa":
        topo.ucm_hash_block_size = _extract_int(line, "ucm_hash_block_size")
        topo.fa_file_size = _extract_int(line, "fa_file_size")
        topo.wa_file_size = _extract_int(line, "wa_file_size")
        topo.alignment_tokens = (
            _extract_int(line, "alignment_tokens") or topo.vllm_hash_block_size
        )
        topo.group_specs = _extract_groups(line)
    else:
        topo.alignment_tokens = topo.vllm_hash_block_size
    return topo


def parse_trace_line(line: str, source: str) -> TraceRecord | None:
    m = _UCM_TRACE_RE.search(line)
    field_key = "block_hashes"
    if m is None:
        m = _TRACE_RE.search(line)
        field_key = "ucm_block_ids"
    if m is None:
        return None
    try:
        hash_ids = ast.literal_eval(m.group(field_key))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(hash_ids, list):
        return None
    sys_match = _SYSTEM_TIME_RE.search(line)
    req_id = m.group("request_id")
    return TraceRecord(
        timestamp=float(m.group("timestamp")),
        input_length=int(m.group("input_length")),
        output_length=int(m.group("output_length")),
        hash_ids=[str(h) for h in hash_ids],
        source=source,
        request_id=req_id.strip() if req_id else None,
        system_time=sys_match.group("system_time") if sys_match else None,
    )


# ============================================================================
# Shared helpers
# ============================================================================


def _rate(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _progress(desc: str, current: int, total: int) -> None:
    """Print a progress bar to stderr (doesn't pollute stdout)."""
    pct = current / total * 100 if total else 100
    bar_len = 30
    filled = int(bar_len * current / total) if total else bar_len
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stderr.write(f"\r{desc:42s} {pct:5.1f}% |{bar}| {current}/{total}")
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def _cdiv(a: int, b: int) -> int:
    """Integer ceil division."""
    return -(-a // b)


def block_token_weights(record: TraceRecord) -> list[int]:
    n = len(record.hash_ids)
    if n == 0:
        return []
    base = record.input_length // n
    rem = record.input_length % n
    return [base + 1 if i < rem else base for i in range(n)]


def synthesize_chain(
    coarse_hashes: list[str], trace_bs: int, vllm_hash_bs: int
) -> list[str]:
    """Synthesize a fine-grained hash chain from a coarse-grained one.

    Each coarse hash covers ``trace_bs`` tokens. The fine chain has
    ``trace_bs // vllm_hash_bs`` entries per coarse hash. Since coarse hashes
    are already unique md5 hex digests, we derive fine hashes by appending
    ``_<index>`` — deterministic, unique, and ~10x faster than re-hashing.
    """
    scale = trace_bs // vllm_hash_bs
    if scale <= 1:
        return list(coarse_hashes)
    return [f"{ch}_{i}" for ch in coarse_hashes for i in range(scale)]


def _derive_state_dicts(
    record: TraceRecord, lcm_block: int, vllm_bs: int, num_mamba_groups: int
) -> list[dict[int, str]]:
    out: list[dict[int, str]] = []
    step = lcm_block // vllm_bs
    num_boundaries = len(record.hash_ids) // step
    for g in range(num_mamba_groups):
        d: dict[int, str] = {}
        for k in range(num_boundaries):
            boundary = (k + 1) * lcm_block
            idx = (k + 1) * step - 1
            prefix_hash = record.hash_ids[idx]
            d[boundary] = f"g{g}:B{boundary}:{prefix_hash}"
        out.append(d)
    return out


# ============================================================================
# Simulator hierarchy
# ============================================================================


class HitRateSimulator(ABC):
    """Abstract simulation engine.

    Template method ``_simulate_record`` fixes the three-stage flow:
    _lookup -> _forward_sim -> _request_end_free.
    Subclasses implement the four abstract methods.
    """

    def __init__(
        self,
        topo: SimTopology,
        gpu_cap: int,
        dram_cap: int,
        fs_cap: int,
        *,
        random_seed: int = 0,
        chunk_size: int | None = None,
        async_scheduling: bool = False,
        **sim_params,
    ) -> None:
        if gpu_cap < 0 or dram_cap < 0 or fs_cap < 0:
            raise ValueError("cache capacities must be >= 0")
        self.topo = topo
        self.rng = random.Random(random_seed)
        self.chunk_size = chunk_size
        self.async_scheduling = async_scheduling
        self.hbm_pools = [BlockPool(gpu_cap) for _ in range(topo.dp_size)]
        self._init_pools(dram_cap, fs_cap)
        self._sim_params = sim_params
        self.request_groups = RequestGroups()
        self.producer_map: dict = {}
        self.total_tokens = 0
        self.gpu_hit_tokens = 0
        self.dram_hit_tokens = 0
        self.fs_hit_tokens = 0
        self.miss_tokens = 0

    # --- Init hooks (overridable) ---

    def _init_pools(self, dram_cap: int, fs_cap: int) -> None:
        """Default: single DRAM pool set. FAWA overrides for FA/WA split."""
        self.dram_pools = [
            ByteLRUPool(dram_cap) for _ in range(self.topo.num_dram_pools())
        ]
        self.fs_pool = ByteLRUPool(fs_cap)

    def _get_pools(self, dp_rank: int) -> tuple:
        """Default: (hbm, dram, fs) 3-tuple. FAWA overrides to 4-tuple."""
        hbm = self.hbm_pools[dp_rank]
        dram = self.topo.select_pool(dp_rank, self.dram_pools)
        return hbm, dram, self.fs_pool

    # --- Template method ---

    def simulate(
        self,
        records: Iterable[TraceRecord],
        desc: str = "",
        progress_cb=None,
    ) -> dict:
        """Outer template: iterate records -> _simulate_record -> _build_result."""
        trace_bs = self.topo.trace_hash_block_size or self.topo.vllm_hash_block_size
        vllm_bs = self.topo.vllm_hash_block_size
        need_synth = trace_bs > vllm_bs

        records = list(records)
        total = len(records)
        for i, record in enumerate(records):
            if need_synth:
                record = replace(
                    record,
                    hash_ids=synthesize_chain(record.hash_ids, trace_bs, vllm_bs),
                )
            self.total_tokens += record.input_length
            request_index = self.request_groups.add(record.timestamp)
            if not record.hash_ids:
                self.miss_tokens += record.input_length
            else:
                dp_rank = self._pick_dp_rank()
                self._simulate_record(record, request_index, dp_rank)
            if progress_cb:
                progress_cb(i + 1, total)
            elif desc:
                _progress(desc, i + 1, total)
        return self._build_result()

    def _simulate_record(self, record: TraceRecord, req_idx: int, dp_rank: int) -> None:
        """Inner template: four-stage flow fixed."""
        ctx = self._new_context(req_idx, dp_rank)
        self._lookup(record, ctx)
        self._record_hit_roots(ctx.hit_roots, req_idx, record)
        self._alloc_after_lookup(record, ctx)
        self._forward_sim(record, ctx)
        self._request_end_free(ctx)

    # --- Abstract methods ---

    @abstractmethod
    def _new_context(self, req_idx: int, dp_rank: int) -> SimContext:
        """Create per-request context. Select pools, build group_contexts (FAWA)."""

    @abstractmethod
    def _lookup(self, record: TraceRecord, ctx: SimContext) -> None:
        """1. Lookup (three sub-phases, read-only for HBM):
        Phase 1: HBM peek (FA forward + WA/mamba reverse with need blocks).
        Phase 2: UCM peek (FA forward + WA/mamba reverse).
        Phase 3: UCM heat update (get/put for [0, gated_tokens)).
        Sets ctx.prefix_hit_count / ctx.gated_tokens / ctx.gated_tier / ctx.hit_roots.
        """

    @abstractmethod
    def _alloc_after_lookup(self, record: TraceRecord, ctx: SimContext) -> None:
        """2. HBM allocation for hit range. touch_get first (reuse), miss then alloc+cache."""

    @abstractmethod
    def _forward_sim(self, record: TraceRecord, ctx: SimContext) -> None:
        """3. Forward chunk sim (miss range only): alloc FA blocks (from hit end) +
        dump to UCM + WA/mamba per-chunk alloc/cache/carry-over/free."""

    @abstractmethod
    def _request_end_free(self, ctx: SimContext) -> None:
        """4. Request end: free FA blocks + remaining WA/mamba blocks."""

    # --- Shared helpers ---

    def _pick_dp_rank(self) -> int:
        if self.topo.dp_size > 1:
            return self.rng.randrange(self.topo.dp_size)
        return 0

    def _alloc_and_cache(
        self, hash_id, group_id: int, hbm: BlockPool, req_idx: int
    ) -> int | None:
        """alloc + cache_block + producer_map[hash]=req_idx. Returns bid or None."""
        bid = hbm.alloc()
        if bid is None:
            return None
        hbm.cache_block(bid, hash_id, group_id)
        self.producer_map[hash_id] = req_idx
        return bid

    def _record_hit_roots(
        self, hit_roots: set[int], req_idx: int, record: TraceRecord
    ) -> None:
        if hit_roots:
            root = self.request_groups.union_roots([req_idx, *hit_roots])
            self.request_groups.record_hit(root, record.timestamp)

    def _build_result(self) -> dict:
        total_hit = self.gpu_hit_tokens + self.dram_hit_tokens + self.fs_hit_tokens
        result = {
            "total_tokens": self.total_tokens,
            "gpu_hit_tokens": self.gpu_hit_tokens,
            "dram_hit_tokens": self.dram_hit_tokens,
            "fs_hit_tokens": self.fs_hit_tokens,
            "miss_tokens": self.miss_tokens,
            "total_hit_tokens": total_hit,
            "hit_rate": _rate(total_hit, self.total_tokens),
        }
        result.update(_lifetime_stats(self.request_groups.lifetimes()))
        return result

    # --- Class methods for AnalysisRunner ---

    @classmethod
    @abstractmethod
    def dump_modes(cls) -> list[tuple[str, dict]]:
        """Return [(mode_name, extra_kwargs)]. Default [("", {})]. FAWA overrides."""

    @classmethod
    @abstractmethod
    def entry_byte_size(cls, topo: SimTopology) -> int:
        """Byte size per entry for theoretical_max capacity. FAWA overrides."""


# ============================================================================
# Standard model simulator
# ============================================================================


class StandardSimulator(HitRateSimulator):
    """Standard model: FA prefix (weight-based) + dump + free_reverse."""

    def _new_context(self, req_idx: int, dp_rank: int) -> SimContext:
        hbm, dram, fs = self._get_pools(dp_rank)
        return SimContext(req_idx=req_idx, hbm=hbm, dram=dram, fs=fs)

    def _lookup(self, record: TraceRecord, ctx: SimContext) -> None:
        """Three-phase lookup: HBM peek + UCM peek + UCM heat update."""
        group_id = 0
        entry_size = self.topo.hbm_block_data_size
        weights = block_token_weights(record)
        total = len(record.hash_ids)

        # Phase 1: HBM peek (read-only)
        hbm_prefix = 0
        for i in range(total):
            hash_id = record.hash_ids[i]
            if ctx.hbm.peek(hash_id, group_id):
                hbm_prefix = i + 1
                ctx.hit_roots.add(
                    self.request_groups.find(
                        self.producer_map.get(hash_id, ctx.req_idx)
                    )
                )
                self.gpu_hit_tokens += weights[i]
            else:
                break

        # Phase 2: UCM peek (read-only, no MRU refresh)
        ucm_prefix = hbm_prefix
        for i in range(hbm_prefix, total):
            hash_id = record.hash_ids[i]
            dram_entry = ctx.dram.peek(hash_id)
            if dram_entry is not None:
                ucm_prefix = i + 1
                ctx.hit_roots.add(self.request_groups.find(dram_entry.producer_index))
                self.dram_hit_tokens += weights[i]
                continue
            fs_entry = ctx.fs.peek(hash_id)
            if fs_entry is not None:
                ucm_prefix = i + 1
                ctx.hit_roots.add(self.request_groups.find(fs_entry.producer_index))
                self.fs_hit_tokens += weights[i]
                continue
            break

        ctx.prefix_hit_count = ucm_prefix

        # Phase 3: UCM heat update + FS→DRAM promotion
        # HBM hit area (i < hbm_prefix): get only (MRU refresh, no promotion).
        # UCM hit area (i >= hbm_prefix): fs.get→dram.put (promote).
        for i in range(ucm_prefix):
            hash_id = record.hash_ids[i]
            if i < hbm_prefix:
                ctx.dram.get(hash_id)
                ctx.fs.get(hash_id)
            else:
                fs_entry = ctx.fs.get(hash_id)
                if fs_entry is not None:
                    ctx.dram.put(hash_id, entry_size, fs_entry)
                else:
                    ctx.dram.get(hash_id)

    def _alloc_after_lookup(self, record: TraceRecord, ctx: SimContext) -> None:
        """Allocate HBM blocks for hit range. touch_get first, then alloc+cache."""
        group_id = 0
        for i in range(ctx.prefix_hit_count):
            hash_id = record.hash_ids[i]
            bid = ctx.hbm.touch_get(hash_id, group_id)
            if bid is None:
                bid = self._alloc_and_cache(hash_id, group_id, ctx.hbm, ctx.req_idx)
            if bid is not None:
                ctx.fa_track.append((bid, True))

    def _forward_sim(self, record: TraceRecord, ctx: SimContext) -> None:
        """Per-chunk FA alloc + dump (miss range only)."""
        vllm_bs = self.topo.vllm_hash_block_size
        total = len(record.hash_ids)
        if self.chunk_size and self.chunk_size > 0:
            chunk_step = max(1, self.chunk_size // vllm_bs)
        else:
            chunk_step = total
        entry = ByteCacheEntry(ctx.req_idx)
        entry_size = self.topo.hbm_block_data_size
        group_id = 0
        weights = block_token_weights(record)

        pos = ctx.prefix_hit_count
        fa_dumped = ctx.prefix_hit_count
        while pos < total:
            chunk_end = min(pos + chunk_step, total)

            for i in range(pos, chunk_end):
                hash_id = record.hash_ids[i]
                weight = weights[i]
                bid = self._alloc_and_cache(hash_id, group_id, ctx.hbm, ctx.req_idx)
                if bid is not None:
                    ctx.fa_track.append((bid, True))
                self.miss_tokens += weight

            for i in range(fa_dumped, chunk_end):
                hash_id = record.hash_ids[i]
                ctx.dram.put(hash_id, entry_size, entry)
                ctx.fs.put(hash_id, entry_size, entry)
            fa_dumped = chunk_end

            pos = chunk_end

    def _request_end_free(self, ctx: SimContext) -> None:
        ctx.hbm.free_reverse(ctx.fa_track)

    @classmethod
    def dump_modes(cls) -> list[tuple[str, dict]]:
        return [("", {})]

    @classmethod
    def entry_byte_size(cls, topo: SimTopology) -> int:
        return topo.hbm_block_data_size


# ============================================================================
# Mamba model simulator
# ============================================================================


class MambaSimulator(HitRateSimulator):
    """Linear hybrid model (Qwen3.6 / Kimi-K3):
    FA prefix (block count) + mamba reverse + dump + multi-stage free.
    """

    def _new_context(self, req_idx: int, dp_rank: int) -> MambaContext:
        hbm, dram, fs = self._get_pools(dp_rank)
        return MambaContext(req_idx=req_idx, hbm=hbm, dram=dram, fs=fs)

    def _lookup(self, record: TraceRecord, ctx: MambaContext) -> None:
        """Three-phase lookup: HBM peek + UCM peek + UCM heat update."""
        group_id = 0
        tp = self.topo.tp_size
        derive_mamba = self.topo.is_mla
        lcm_block = self.topo.lcm_block_size
        mamba_groups = self.topo.mamba_groups
        entry_size = self.topo.hbm_block_data_size
        vllm_bs = self.topo.vllm_hash_block_size
        total = len(record.hash_ids)

        # Phase 1: HBM peek
        # Always derive state_dicts (needed by _forward_sim even when no HBM hit)
        if lcm_block > 0:
            ctx.state_dicts = _derive_state_dicts(
                record, lcm_block, vllm_bs, mamba_groups
            )

        # 1a. FA forward
        hbm_prefix = 0
        for i in range(total):
            hash_id = record.hash_ids[i]
            if ctx.hbm.peek(hash_id, group_id):
                hbm_prefix = i + 1
                ctx.hit_roots.add(
                    self.request_groups.find(
                        self.producer_map.get(hash_id, ctx.req_idx)
                    )
                )
            else:
                break

        # 1b. Mamba reverse (HBM only)
        hbm_gated = 0
        hbm_gated_tier = -1
        if lcm_block > 0 and hbm_prefix > 0:
            max_boundary = hbm_prefix * vllm_bs
            all_boundaries: set[int] = set()
            for d in ctx.state_dicts:
                all_boundaries.update(d.keys())
            candidates = sorted(b for b in all_boundaries if b <= max_boundary)
            for boundary in reversed(candidates):
                ok = True
                for d in ctx.state_dicts:
                    rank0 = d.get(boundary)
                    if rank0 is None:
                        ok = False
                        break
                    for cid in self._rank_ids(rank0, tp, derive_mamba):
                        if not ctx.hbm.peek(cid, group_id):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    hbm_gated = boundary
                    hbm_gated_tier = 0
                    for d in ctx.state_dicts:
                        rank0 = d.get(boundary)
                        if rank0:
                            for cid in self._rank_ids(rank0, tp, derive_mamba):
                                ctx.hit_roots.add(
                                    self.request_groups.find(
                                        self.producer_map.get(cid, ctx.req_idx)
                                    )
                                )
                    break

        # Phase 2: UCM peek
        # 2a. FA forward (DRAM/FS peek)
        ucm_prefix = hbm_prefix
        fa_ext_tier = 0
        for i in range(hbm_prefix, total):
            hash_id = record.hash_ids[i]
            dram_entry = ctx.dram.peek(hash_id)
            if dram_entry is not None:
                ucm_prefix = i + 1
                fa_ext_tier = max(fa_ext_tier, 1)
                ctx.hit_roots.add(self.request_groups.find(dram_entry.producer_index))
                continue
            fs_entry = ctx.fs.peek(hash_id)
            if fs_entry is not None:
                ucm_prefix = i + 1
                fa_ext_tier = max(fa_ext_tier, 2)
                ctx.hit_roots.add(self.request_groups.find(fs_entry.producer_index))
                continue
            break

        # 2b. Mamba reverse (DRAM/FS only, from ucm_prefix)
        ucm_gated = 0
        ucm_gated_tier = -1
        if lcm_block > 0 and ucm_prefix > 0:
            max_boundary = ucm_prefix * vllm_bs
            all_boundaries = set()
            for d in ctx.state_dicts:
                all_boundaries.update(d.keys())
            candidates = sorted(b for b in all_boundaries if b <= max_boundary)
            if hbm_gated_tier == 0:
                candidates = [b for b in candidates if b > hbm_gated]
            for boundary in reversed(candidates):
                tier_rank = 0
                ok = True
                for d in ctx.state_dicts:
                    rank0 = d.get(boundary)
                    if rank0 is None:
                        ok = False
                        break
                    for cid in self._rank_ids(rank0, tp, derive_mamba):
                        dram_entry = ctx.dram.peek(cid)
                        if dram_entry is not None:
                            tier_rank = max(tier_rank, 1)
                            continue
                        fs_entry = ctx.fs.peek(cid)
                        if fs_entry is not None:
                            tier_rank = max(tier_rank, 2)
                            continue
                        ok = False
                        break
                    if not ok:
                        break
                if ok:
                    ucm_gated = boundary
                    ucm_gated_tier = tier_rank
                    for d in ctx.state_dicts:
                        rank0 = d.get(boundary)
                        if rank0:
                            for cid in self._rank_ids(rank0, tp, derive_mamba):
                                dram_entry = ctx.dram.peek(cid)
                                if dram_entry is not None:
                                    ctx.hit_roots.add(
                                        self.request_groups.find(
                                            dram_entry.producer_index
                                        )
                                    )
                                else:
                                    fs_entry = ctx.fs.peek(cid)
                                    if fs_entry is not None:
                                        ctx.hit_roots.add(
                                            self.request_groups.find(
                                                fs_entry.producer_index
                                            )
                                        )
                    break

        # Final gated_tokens and tier
        if hbm_gated >= ucm_gated:
            ctx.gated_tokens = hbm_gated
            ctx.gated_tier = hbm_gated_tier
        else:
            ctx.gated_tokens = ucm_gated
            ctx.gated_tier = ucm_gated_tier
        ctx.prefix_hit_count = ucm_prefix

        # Count tokens
        if ctx.gated_tier == 0:
            self.gpu_hit_tokens += ctx.gated_tokens
        elif ctx.gated_tier == 1:
            self.dram_hit_tokens += ctx.gated_tokens
        elif ctx.gated_tier == 2:
            self.fs_hit_tokens += ctx.gated_tokens
        self.miss_tokens += record.input_length - ctx.gated_tokens

        # Phase 3: UCM heat update + FS→DRAM promotion
        if lcm_block > 0:
            # FA entries
            for i in range(ucm_prefix):
                hash_id = record.hash_ids[i]
                if i < hbm_prefix:
                    ctx.dram.get(hash_id)
                    ctx.fs.get(hash_id)
                else:
                    fs_entry = ctx.fs.get(hash_id)
                    if fs_entry is not None:
                        ctx.dram.put(hash_id, entry_size, fs_entry)
                    else:
                        ctx.dram.get(hash_id)
            # Mamba entries: all boundaries in [0, gated_tokens]
            # HBM-hit (<= hbm_gated): get only; UCM non-gated: get only;
            # UCM gated boundary: fs.get→dram.put (promote)
            if ctx.state_dicts:
                hbm_gated_val = hbm_gated if hbm_gated_tier == 0 else 0
                for d in ctx.state_dicts:
                    for boundary in sorted(b for b in d if b <= ctx.gated_tokens):
                        rank0 = d[boundary]
                        is_gated = boundary == ctx.gated_tokens
                        for cid in self._rank_ids(rank0, tp, derive_mamba):
                            if boundary <= hbm_gated_val:
                                ctx.dram.get(cid)
                                ctx.fs.get(cid)
                            elif is_gated and ctx.gated_tier != 0:
                                fs_entry = ctx.fs.get(cid)
                                if fs_entry is not None:
                                    ctx.dram.put(cid, entry_size, fs_entry)
                                else:
                                    ctx.dram.get(cid)
                            else:
                                ctx.dram.get(cid)
                                ctx.fs.get(cid)

    def _alloc_after_lookup(self, record: TraceRecord, ctx: MambaContext) -> None:
        """Allocate HBM blocks for hit range. touch_get first, then alloc+cache."""
        group_id = 0
        tp = self.topo.tp_size
        derive_mamba = self.topo.is_mla

        # FA blocks
        for i in range(ctx.prefix_hit_count):
            hash_id = record.hash_ids[i]
            bid = ctx.hbm.touch_get(hash_id, group_id)
            if bid is None:
                bid = self._alloc_and_cache(hash_id, group_id, ctx.hbm, ctx.req_idx)
            if bid is not None:
                ctx.fa_track.append((bid, True))

        # Mamba blocks at gated boundary
        if ctx.gated_tokens > 0 and ctx.state_dicts:
            for d in ctx.state_dicts:
                rank0 = d.get(ctx.gated_tokens)
                if rank0 is None:
                    continue
                for cid in self._rank_ids(rank0, tp, derive_mamba):
                    bid = ctx.hbm.touch_get(cid, group_id)
                    if bid is None:
                        bid = self._alloc_and_cache(cid, group_id, ctx.hbm, ctx.req_idx)
                    if bid is not None:
                        ctx.mamba_track.append((bid, True))

    def _forward_sim(self, record: TraceRecord, ctx: MambaContext) -> None:
        """Per-chunk: free prev mamba + FA alloc + mamba alloc + dump (miss only)."""
        vllm_bs = self.topo.vllm_hash_block_size
        lcm_block = self.topo.lcm_block_size
        tp = self.topo.tp_size
        derive_mamba = self.topo.is_mla
        total = len(record.hash_ids)
        entry_size = self.topo.hbm_block_data_size
        group_id = 0
        entry = ByteCacheEntry(ctx.req_idx)

        if self.chunk_size and self.chunk_size > 0:
            chunk_step = max(1, (self.chunk_size // lcm_block) * lcm_block // vllm_bs)
        else:
            chunk_step = total

        prev_mamba_running: list[tuple[int, bool]] = []
        pos = ctx.gated_tokens // vllm_bs
        fa_dumped = ctx.prefix_hit_count
        while pos < total:
            chunk_end = min(pos + chunk_step, total)
            chunk_token_end = chunk_end * vllm_bs
            is_lcm = chunk_token_end % lcm_block == 0

            for bid, has_hash in prev_mamba_running:
                ctx.hbm.free(bid, has_hash)
            prev_mamba_running = []

            for i in range(max(pos, ctx.prefix_hit_count), chunk_end):
                hash_id = record.hash_ids[i]
                bid = self._alloc_and_cache(hash_id, group_id, ctx.hbm, ctx.req_idx)
                if bid is not None:
                    ctx.fa_track.append((bid, True))

            if is_lcm and chunk_token_end > ctx.gated_tokens:
                for d in ctx.state_dicts:
                    rank0 = d.get(chunk_token_end)
                    if rank0 is None:
                        continue
                    for cid in self._rank_ids(rank0, tp, derive_mamba):
                        bid = self._alloc_and_cache(cid, group_id, ctx.hbm, ctx.req_idx)
                        if bid is not None:
                            prev_mamba_running.append((bid, True))

            for i in range(fa_dumped, chunk_end):
                hash_id = record.hash_ids[i]
                ctx.dram.put(hash_id, entry_size, entry)
                ctx.fs.put(hash_id, entry_size, entry)
            fa_dumped = chunk_end

            if is_lcm and chunk_token_end > ctx.gated_tokens:
                for d in ctx.state_dicts:
                    rank0 = d.get(chunk_token_end)
                    if rank0 is None:
                        continue
                    for cid in self._rank_ids(rank0, tp, derive_mamba):
                        ctx.dram.put(cid, entry_size, entry)
                        ctx.fs.put(cid, entry_size, entry)

            pos = chunk_end

        ctx.mamba_remaining = prev_mamba_running

    def _request_end_free(self, ctx: MambaContext) -> None:
        ctx.hbm.free_reverse(ctx.fa_track)
        for bid, has_hash in ctx.mamba_track:
            ctx.hbm.free(bid, has_hash)
        for bid, has_hash in ctx.mamba_remaining:
            ctx.hbm.free(bid, has_hash)

    @staticmethod
    def _rank_ids(rank0_hash: str, tp: int, derive: bool) -> list[str]:
        if not derive or tp <= 1:
            return [rank0_hash]
        return [rank0_hash if r == 0 else f"{rank0_hash}:{r}" for r in range(tp)]

    @classmethod
    def dump_modes(cls) -> list[tuple[str, dict]]:
        return [("", {})]

    @classmethod
    def entry_byte_size(cls, topo: SimTopology) -> int:
        return topo.hbm_block_data_size


# ============================================================================
# FAWA (DS V4) simulator
# ============================================================================


class FAWASimulator(HitRateSimulator):
    """DSV4 (FAWA): FA per-group prefix + WA reverse + dual-store dump +
    WA per-chunk alloc/cache/carry-over/free.
    """

    # --- Override init ---

    def _init_pools(self, dram_cap: int, fs_cap: int) -> None:
        if self.topo.unified:
            shared = [ByteLRUPool(dram_cap) for _ in range(self.topo.num_dram_pools())]
            self.fa_dram_pools = shared
            self.wa_dram_pools = shared
        else:
            half = dram_cap // 2
            self.fa_dram_pools = [
                ByteLRUPool(half) for _ in range(self.topo.num_dram_pools())
            ]
            self.wa_dram_pools = [
                ByteLRUPool(half) for _ in range(self.topo.num_dram_pools())
            ]
        self.fs_pool = ByteLRUPool(fs_cap)

    def _get_pools(self, dp_rank: int) -> tuple:
        hbm = self.hbm_pools[dp_rank]
        fa_dram = self.topo.select_pool(dp_rank, self.fa_dram_pools)
        wa_dram = self.topo.select_pool(dp_rank, self.wa_dram_pools)
        return hbm, fa_dram, wa_dram, self.fs_pool

    # --- Three-stage implementation ---

    def _new_context(self, req_idx: int, dp_rank: int) -> FAWAContext:
        hbm, fa_dram, wa_dram, fs = self._get_pools(dp_rank)
        group_contexts = self.topo.build_group_contexts()
        fa_groups = [
            g
            for g in group_contexts
            if g.manager_type in ("compress", "full_attention")
        ]
        wa_groups = [g for g in group_contexts if g.manager_type == "sliding_window"]
        ctx = FAWAContext(req_idx=req_idx, hbm=hbm, dram=fa_dram, fs=fs)
        ctx.wa_dram = wa_dram
        ctx.fa_groups = fa_groups
        ctx.wa_groups = wa_groups
        return ctx

    def _lookup(self, record: TraceRecord, ctx: FAWAContext) -> None:
        """Three-phase lookup: HBM peek + UCM peek + UCM heat update."""
        ucm_hash_bs = self.topo.ucm_hash_block_size
        vllm_hash_bs = self.topo.vllm_hash_block_size
        ucm_scale = ucm_hash_bs // vllm_hash_bs
        fa_size = self.topo.fa_file_size
        wa_size = self.topo.wa_file_size
        chain = record.hash_ids
        num_ucm_blocks = len(chain) // ucm_scale

        # Phase 1: HBM peek
        ctx.hbm_prefix_tokens = self._hbm_fa_lookup(
            chain, ctx.fa_groups, ctx.hbm, ctx.req_idx, ctx.hit_roots
        )
        hbm_gated, hbm_gated_tier = self._hbm_wa_reverse(
            chain,
            ctx,
            int(ctx.hbm_prefix_tokens) // ucm_hash_bs,
            num_ucm_blocks,
            ucm_scale,
            ucm_hash_bs,
        )

        # Phase 2: UCM peek
        ctx.ucm_prefix, ctx.fa_ext_tier = self._ucm_fa_forward(
            chain,
            ctx.hbm_prefix_tokens,
            num_ucm_blocks,
            ucm_scale,
            ctx.dram,
            ctx.fs,
            ctx.req_idx,
            ctx.hit_roots,
        )
        ucm_gated, ucm_gated_tier = self._ucm_wa_reverse(
            chain,
            ctx,
            ctx.ucm_prefix,
            num_ucm_blocks,
            ucm_scale,
            ucm_hash_bs,
            hbm_gated if hbm_gated_tier == 0 else 0,
        )

        if hbm_gated >= ucm_gated:
            ctx.gated_tokens = hbm_gated
            ctx.gated_tier = hbm_gated_tier
        else:
            ctx.gated_tokens = ucm_gated
            ctx.gated_tier = ucm_gated_tier

        self._count_fawa_tokens(
            ctx.gated_tokens,
            ctx.gated_tier,
            ctx.hbm_prefix_tokens,
            ctx.fa_ext_tier,
        )
        self.miss_tokens += record.input_length - ctx.gated_tokens

        # Phase 3: UCM heat update + FS→DRAM promotion
        hbm_prefix_ucm = int(ctx.hbm_prefix_tokens) // ucm_hash_bs
        gated_ucm = ctx.gated_tokens // ucm_hash_bs
        for ucm_idx in range(min(gated_ucm, num_ucm_blocks)):
            ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
            is_hbm_hit = ucm_idx < hbm_prefix_ucm
            is_last = ucm_idx == gated_ucm - 1

            # FA: HBM hit → get only; UCM hit → fs.get→dram.put
            if is_hbm_hit:
                ctx.dram.get(self._fa_key(ucm_hash))
                ctx.fs.get(self._fa_key(ucm_hash))
            else:
                fs_entry = ctx.fs.get(self._fa_key(ucm_hash))
                if fs_entry is not None:
                    ctx.dram.put(self._fa_key(ucm_hash), fa_size, fs_entry)
                else:
                    ctx.dram.get(self._fa_key(ucm_hash))

            # WA: HBM hit → get only; UCM non-last → get only; UCM last → fs.get→dram.put
            if is_hbm_hit or not is_last:
                ctx.wa_dram.get(self._wa_key(ucm_hash))
                ctx.fs.get(self._wa_key(ucm_hash))
            else:
                fs_entry = ctx.fs.get(self._wa_key(ucm_hash))
                if fs_entry is not None:
                    ctx.wa_dram.put(self._wa_key(ucm_hash), wa_size, fs_entry)
                else:
                    ctx.wa_dram.get(self._wa_key(ucm_hash))

    def _alloc_after_lookup(self, record: TraceRecord, ctx: FAWAContext) -> None:
        """Allocate HBM blocks for hit range. touch_get first, then alloc+cache."""
        chain = record.hash_ids

        # FA: per-group, full blocks [0, gated_tokens // lbs)
        for g in ctx.fa_groups:
            sf = g.scale_factor
            total_full = len(chain) // sf
            end_block = min(total_full, ctx.gated_tokens // g.logical_block_size)
            while g.num_cached < end_block:
                block_idx = g.num_cached
                h = g.derive_block_hash(block_idx, chain)
                bid = ctx.hbm.touch_get(h, g.group_id)
                if bid is None:
                    bid = ctx.hbm.alloc()
                    if bid is not None:
                        ctx.hbm.cache_block(bid, h, g.group_id)
                        self.producer_map[h] = ctx.req_idx
                if bid is not None:
                    g.block_ids.append(bid)
                    g.num_cached += 1
                else:
                    break

        # WA: per-group, need blocks at gated boundary
        for g in ctx.wa_groups:
            lbs = g.logical_block_size
            need = ceil((g.sliding_window - 1) / lbs) if g.sliding_window else 1
            base = ctx.gated_tokens // lbs
            ctx.wa_track.setdefault(g.group_id, [])
            for blk_idx in range(max(0, base - need), base):
                if not g.reachable(blk_idx):
                    continue
                h = g.derive_block_hash(blk_idx, chain)
                bid = ctx.hbm.touch_get(h, g.group_id)
                if bid is None:
                    bid = ctx.hbm.alloc()
                    if bid is not None:
                        ctx.hbm.cache_block(bid, h, g.group_id)
                        self.producer_map[h] = ctx.req_idx
                if bid is not None:
                    ctx.wa_track[g.group_id].append((bid, True))

    def _forward_sim(self, record: TraceRecord, ctx: FAWAContext) -> None:
        """Unified per-chunk loop: FA alloc + WA alloc+free + FA/WA dump."""
        vllm_bs = self.topo.vllm_hash_block_size
        ucm_hash_bs = self.topo.ucm_hash_block_size
        ucm_scale = ucm_hash_bs // vllm_bs
        fa_size = self.topo.fa_file_size
        wa_size = self.topo.wa_file_size
        alignment = self.topo.alignment_tokens or ucm_hash_bs
        chain = record.hash_ids
        total = len(chain)
        entry = ByteCacheEntry(ctx.req_idx)
        wa_block_wise = self._sim_params.get("wa_dump_block_wise", True)

        if self.chunk_size and self.chunk_size > 0:
            chunk_step = max(1, (self.chunk_size // alignment) * alignment // vllm_bs)
        else:
            chunk_step = total

        fa_next = {g.group_id: g.num_cached for g in ctx.fa_groups}
        wa_carry: dict[int, list[tuple[int, bool]]] = {
            g.group_id: list(ctx.wa_track.get(g.group_id, [])) for g in ctx.wa_groups
        }
        wa_pos = {
            g.group_id: ctx.gated_tokens // g.logical_block_size for g in ctx.wa_groups
        }
        fa_dumped = ctx.gated_tokens // ucm_hash_bs
        wa_dumped = ctx.gated_tokens // ucm_hash_bs

        async_delay = self.async_scheduling
        pending_prev2: dict[int, list[tuple[int, bool]]] = {}
        pending_prev1: dict[int, list[tuple[int, bool]]] = {}

        pos = ctx.gated_tokens // vllm_bs
        while pos < total:
            chunk_end = min(pos + chunk_step, total)
            is_last = chunk_end == total

            # Free WA non-running from 2 chunks ago (async scheduling delay)
            if async_delay and pending_prev2:
                for blocks in pending_prev2.values():
                    ctx.hbm.free_reverse(blocks)
                pending_prev2 = {}

            for g in ctx.fa_groups:
                sf = g.scale_factor
                total_full = total // sf
                total_blocks = total_full + (1 if total % sf != 0 else 0)
                needed = min(total_blocks, _cdiv(chunk_end, sf))
                while fa_next[g.group_id] < needed:
                    block_idx = fa_next[g.group_id]
                    if block_idx < total_full:
                        h = g.derive_block_hash(block_idx, chain)
                        bid = ctx.hbm.alloc()
                        if bid is not None:
                            ctx.hbm.cache_block(bid, h, g.group_id)
                            self.producer_map[h] = ctx.req_idx
                            g.block_ids.append(bid)
                            g.num_cached += 1
                    else:
                        bid = ctx.hbm.alloc()
                        if bid is not None:
                            g.block_ids.append(bid)
                    fa_next[g.group_id] += 1

            new_pending: dict[int, list[tuple[int, bool]]] = {}
            for g in ctx.wa_groups:
                lbs = g.logical_block_size
                need = ceil((g.sliding_window - 1) / lbs) if g.sliding_window else 1
                wa_sf = g.scale_factor
                wa_total = total // wa_sf
                if chunk_step < total:
                    blocks_per_chunk = max(1, chunk_step * vllm_bs // lbs)
                else:
                    blocks_per_chunk = wa_total
                wa_ce = min(wa_pos[g.group_id] + blocks_per_chunk, wa_total)

                chunk_blocks = list(wa_carry[g.group_id])
                for blk_idx in range(wa_pos[g.group_id], wa_ce):
                    h = g.derive_block_hash(blk_idx, chain)
                    bid = ctx.hbm.alloc()
                    if bid is not None:
                        if g.reachable(blk_idx):
                            ctx.hbm.cache_block(bid, h, g.group_id)
                            self.producer_map[h] = ctx.req_idx
                            chunk_blocks.append((bid, True))
                        else:
                            chunk_blocks.append((bid, False))

                if not is_last:
                    to_free = chunk_blocks[:-need] if len(chunk_blocks) > need else []
                    if async_delay:
                        new_pending.setdefault(g.group_id, []).extend(to_free)
                    else:
                        ctx.hbm.free_reverse(to_free)
                    wa_carry[g.group_id] = (
                        chunk_blocks[-need:]
                        if len(chunk_blocks) >= need
                        else chunk_blocks
                    )
                else:
                    wa_carry[g.group_id] = chunk_blocks
                wa_pos[g.group_id] = wa_ce

            # Roll pending queue for async delay
            if async_delay:
                pending_prev2 = pending_prev1
                pending_prev1 = new_pending

            ucm_completed = chunk_end // ucm_scale
            for ucm_idx in range(fa_dumped, ucm_completed):
                ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
                ctx.dram.put(self._fa_key(ucm_hash), fa_size, entry)
                ctx.fs.put(self._fa_key(ucm_hash), fa_size, entry)

            if wa_block_wise:
                for ucm_idx in range(wa_dumped, ucm_completed):
                    ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
                    ctx.wa_dram.put(self._wa_key(ucm_hash), wa_size, entry)
                    ctx.fs.put(self._wa_key(ucm_hash), wa_size, entry)
            else:
                if is_last or chunk_end * vllm_bs % ucm_hash_bs == 0:
                    if ucm_completed > wa_dumped:
                        ucm_idx = ucm_completed - 1
                        ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
                        ctx.wa_dram.put(self._wa_key(ucm_hash), wa_size, entry)
                        ctx.fs.put(self._wa_key(ucm_hash), wa_size, entry)
            fa_dumped = ucm_completed
            wa_dumped = ucm_completed

            pos = chunk_end

        # Free remaining pending WA (async delay)
        if async_delay:
            for pending in (pending_prev2, pending_prev1):
                for blocks in pending.values():
                    ctx.hbm.free_reverse(blocks)

        ctx.wa_remaining = []
        for g in ctx.wa_groups:
            ctx.wa_remaining.extend(wa_carry[g.group_id])

    def _request_end_free(self, ctx: FAWAContext) -> None:
        # WA chunk-end already freed during _forward_sim
        # FA free_reverse -> before WA track/remaining
        for g in ctx.fa_groups:
            blocks = [
                (b, i < g.num_cached)
                for i, b in enumerate(g.block_ids)
                if b is not None
            ]
            ctx.hbm.free_reverse(blocks)
        # WA track blocks are already managed by _forward_sim (they become
        # wa_carry -> freed per-chunk or survive in wa_remaining). Do NOT
        # free wa_track separately here — that would double-free.
        # WA remaining free_reverse -> last
        ctx.hbm.free_reverse(ctx.wa_remaining)

    # --- FAWA-specific phase methods ---

    def _hbm_fa_lookup(
        self,
        chain: list,
        fa_groups: list[GroupContext],
        hbm: BlockPool,
        req_idx: int,
        hit_roots: set[int],
    ) -> float:
        """Per-group FA prefix peek (read-only, no touch_get/alloc)."""
        hbm_prefix_tokens = float("inf")
        for g in fa_groups:
            g.reset()
            g_num_full = len(chain) // g.scale_factor
            g_prefix = 0
            for block_idx in range(g_num_full):
                h = g.derive_block_hash(block_idx, chain)
                if hbm.peek(h, g.group_id):
                    g_prefix += 1
                    hit_roots.add(
                        self.request_groups.find(self.producer_map.get(h, req_idx))
                    )
                else:
                    break
            g_prefix_tokens = g_prefix * g.logical_block_size
            if g_prefix_tokens < hbm_prefix_tokens:
                hbm_prefix_tokens = g_prefix_tokens
        if hbm_prefix_tokens == float("inf"):
            hbm_prefix_tokens = 0
        return hbm_prefix_tokens

    def _ucm_fa_forward(
        self,
        chain: list,
        hbm_prefix_tokens: float,
        num_ucm_blocks: int,
        ucm_scale: int,
        fa_dram: ByteLRUPool,
        fs: ByteLRUPool,
        req_idx: int,
        hit_roots: set[int],
    ) -> tuple[int, int]:
        """UCM FA DRAM/FS forward peek (read-only, no MRU refresh)."""
        ucm_start = int(hbm_prefix_tokens) // self.topo.ucm_hash_block_size
        ucm_prefix = ucm_start
        fa_ext_tier = 0
        for ucm_idx in range(ucm_start, num_ucm_blocks):
            ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
            dram_entry = fa_dram.peek(self._fa_key(ucm_hash))
            if dram_entry is not None:
                ucm_prefix = ucm_idx + 1
                fa_ext_tier = max(fa_ext_tier, 1)
                hit_roots.add(self.request_groups.find(dram_entry.producer_index))
                continue
            fs_entry = fs.peek(self._fa_key(ucm_hash))
            if fs_entry is not None:
                ucm_prefix = ucm_idx + 1
                fa_ext_tier = max(fa_ext_tier, 2)
                hit_roots.add(self.request_groups.find(fs_entry.producer_index))
                continue
            break
        return ucm_prefix, fa_ext_tier

    def _hbm_wa_reverse(
        self,
        chain: list,
        ctx: FAWAContext,
        hbm_prefix_ucm: int,
        num_ucm_blocks: int,
        ucm_scale: int,
        ucm_hash_bs: int,
    ) -> tuple[int, int]:
        """HBM WA reverse peek: check need blocks per group per boundary."""
        if hbm_prefix_ucm <= 0:
            return 0, -1
        for ucm_idx in range(min(hbm_prefix_ucm, num_ucm_blocks) - 1, -1, -1):
            all_hit = True
            for g in ctx.wa_groups:
                lbs = g.logical_block_size
                need = ceil((g.sliding_window - 1) / lbs) if g.sliding_window else 1
                base = (ucm_idx + 1) * (ucm_hash_bs // lbs)
                found_reachable = False
                for blk_idx in range(max(0, base - need), base):
                    if not g.reachable(blk_idx):
                        continue
                    found_reachable = True
                    h = g.derive_block_hash(blk_idx, chain)
                    if not ctx.hbm.peek(h, g.group_id):
                        all_hit = False
                        break
                if not found_reachable:
                    all_hit = False
                if not all_hit:
                    break
            if all_hit:
                gated = (ucm_idx + 1) * ucm_hash_bs
                for g in ctx.wa_groups:
                    lbs = g.logical_block_size
                    need = ceil((g.sliding_window - 1) / lbs) if g.sliding_window else 1
                    base = (ucm_idx + 1) * (ucm_hash_bs // lbs)
                    for blk_idx in range(max(0, base - need), base):
                        if g.reachable(blk_idx):
                            h = g.derive_block_hash(blk_idx, chain)
                            ctx.hit_roots.add(
                                self.request_groups.find(
                                    self.producer_map.get(h, ctx.req_idx)
                                )
                            )
                return gated, 0
        return 0, -1

    def _ucm_wa_reverse(
        self,
        chain: list,
        ctx: FAWAContext,
        ucm_prefix: int,
        num_ucm_blocks: int,
        ucm_scale: int,
        ucm_hash_bs: int,
        skip_below: int,
    ) -> tuple[int, int]:
        """UCM WA reverse peek: check 1 WA entry per boundary (DRAM/FS only)."""
        upper = min(ucm_prefix, num_ucm_blocks)
        for ucm_idx in range(upper - 1, -1, -1):
            boundary = (ucm_idx + 1) * ucm_hash_bs
            if boundary <= skip_below:
                break
            ucm_hash = chain[(ucm_idx + 1) * ucm_scale - 1]
            wa_entry = ctx.wa_dram.peek(self._wa_key(ucm_hash))
            if wa_entry is not None:
                ctx.hit_roots.add(self.request_groups.find(wa_entry.producer_index))
                return boundary, 1
            fs_entry = ctx.fs.peek(self._wa_key(ucm_hash))
            if fs_entry is not None:
                ctx.hit_roots.add(self.request_groups.find(fs_entry.producer_index))
                return boundary, 2
        return 0, -1

    def _count_fawa_tokens(
        self,
        gated_tokens: int,
        gated_tier: int,
        hbm_prefix_tokens: float,
        fa_ext_tier: int,
    ) -> None:
        fa_tier = 0 if gated_tokens <= int(hbm_prefix_tokens) else fa_ext_tier
        overall_tier = max(fa_tier, gated_tier)
        if overall_tier == 0:
            self.gpu_hit_tokens += gated_tokens
        elif overall_tier == 1:
            self.dram_hit_tokens += gated_tokens
        else:
            self.fs_hit_tokens += gated_tokens

    # --- Static + class methods ---

    @staticmethod
    def _fa_key(hash_val) -> tuple:
        return ("fa", hash_val)

    @staticmethod
    def _wa_key(hash_val) -> tuple:
        return ("wa", hash_val)

    @classmethod
    def dump_modes(cls) -> list[tuple[str, dict]]:
        return [
            ("block_wise", {"wa_dump_block_wise": True}),
            ("chunk_wise", {"wa_dump_block_wise": False}),
        ]

    @classmethod
    def entry_byte_size(cls, topo: SimTopology) -> int:
        return max(topo.hbm_block_data_size, topo.fa_file_size + topo.wa_file_size)


# ============================================================================
# Log collection
# ============================================================================


@dataclass
class LogFacts:
    log_files: list[str]
    records: list[TraceRecord]
    available_kv_cache_memory_bytes: list[int]
    tensor_parallel_sizes: list[int]
    data_parallel_sizes: list[int]
    sim_topology: SimTopology | None = None
    async_scheduling: bool = False
    max_num_batched_tokens: int = 0


class LogCollector:
    """Encapsulates log file collection and parsing."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir

    def collect(self) -> LogFacts:
        if not self.log_dir.exists() or not self.log_dir.is_dir():
            raise ValueError(f"log directory does not exist: {self.log_dir}")

        log_files = self._iter_log_files()
        if not log_files:
            raise ValueError(f"no log files found in log directory: {self.log_dir}")

        records: list[TraceRecord] = []
        available_memory: list[int] = []
        tp_sizes: list[int] = []
        dp_sizes: list[int] = []
        sim_topology: SimTopology | None = None
        async_scheduling = False
        max_num_batched_tokens = 0

        for path in log_files:
            with self._open_log_file(path) as handle:
                for line in handle:
                    record = parse_trace_line(line, str(path))
                    if record is not None:
                        records.append(record)

                    topo = parse_trace_meta(line)
                    if topo is not None:
                        sim_topology = topo

                    for match in AVAILABLE_KV_RE.finditer(line):
                        available_memory.append(
                            self._parse_bytes(match.group("value"), match.group("unit"))
                        )
                    for match in TP_SIZE_RE.finditer(line):
                        tp_sizes.append(int(match.group("value")))
                    for match in DP_SIZE_RE.finditer(line):
                        dp_sizes.append(int(match.group("value")))
                    if ASYNC_SCHED_RE.search(line):
                        async_scheduling = True
                    for match in MAX_BATCHED_TOKENS_RE.finditer(line):
                        max_num_batched_tokens = int(match.group("value"))

        if not records:
            raise ValueError("no trace records found in log files")
        if not available_memory:
            raise ValueError("available kv cache memory was not found in log files")
        if not tp_sizes:
            raise ValueError("tensor_parallel_size was not found in log files")
        if not dp_sizes:
            raise ValueError("data_parallel_size was not found in log files")

        records.sort(key=lambda item: item.timestamp)

        return LogFacts(
            log_files=[str(path) for path in log_files],
            records=records,
            available_kv_cache_memory_bytes=available_memory,
            tensor_parallel_sizes=tp_sizes,
            data_parallel_sizes=dp_sizes,
            sim_topology=sim_topology,
            async_scheduling=async_scheduling,
            max_num_batched_tokens=max_num_batched_tokens,
        )

    def _iter_log_files(self) -> list[Path]:
        patterns = ("*.log", "*.log.*", "*.log.gz")
        files: dict[Path, None] = {}
        for pattern in patterns:
            for path in self.log_dir.rglob(pattern):
                if path.is_file():
                    files[path] = None
        return sorted(files)

    def _open_log_file(self, path: Path):
        if path.name.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        return path.open("r", encoding="utf-8", errors="ignore")

    @staticmethod
    def _parse_bytes(value: str, unit: str | None) -> int:
        number = float(value)
        if not unit:
            return int(number)
        multipliers = {
            "b": 1,
            "byte": 1,
            "bytes": 1,
            "kb": 1024,
            "kib": 1024,
            "mb": 1024**2,
            "mib": 1024**2,
            "gb": 1024**3,
            "gib": 1024**3,
            "tb": 1024**4,
            "tib": 1024**4,
        }
        return int(number * multipliers.get(unit.lower(), 1))


# ============================================================================
# Log helpers (module-level)
# ============================================================================


def _resolve_single(values: list[int], name: str) -> int:
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(
            f"conflicting {name} values: " + ", ".join(str(v) for v in sorted(unique))
        )
    val = next(iter(unique))
    if val <= 0:
        raise ValueError(f"{name} must be > 0")
    return val


def _resolve_gpu_cache_bytes(facts: LogFacts) -> int:
    return min(facts.available_kv_cache_memory_bytes)


def _parse_prometheus(metrics_text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROM_SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        labels = match.group("labels")
        if labels:
            source = ""
            for pair in labels.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() == "source":
                        source = v.strip().strip('"')
            if source:
                key = f'{name}{{source="{source}"}}'
            else:
                key = name
        else:
            key = name
        samples[key] = samples.get(key, 0.0) + float(match.group("value"))
    return samples


def _fetch_service_hit_rate(service_url: str, timeout: float) -> dict:
    normalized = service_url.strip()
    if "://" not in normalized:
        normalized = "http://" + normalized
    metrics_url = normalized.rstrip("/")
    if not metrics_url.endswith("/metrics"):
        metrics_url += "/metrics"
    with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    samples = _parse_prometheus(text)

    total_tokens = next(
        (samples[n] for n in PROMPT_TOKENS_TOTAL_METRICS if n in samples), 0
    )
    cache_hit = next(
        (samples[n] for n in PROMPT_TOKENS_CACHE_HIT_METRICS if n in samples), 0
    )

    hit_rate = cache_hit / total_tokens if total_tokens > 0 else 0.0
    return {
        "service_url": service_url,
        "metrics_url": metrics_url,
        "prefix_cache_hits_total": cache_hit,
        "prefix_cache_queries_total": total_tokens,
        "actual_kv_cache_hit_rate": hit_rate,
    }


# ============================================================================
# CLI: AnalysisRunner + main
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze theoretical UCM KV cache hit-rate from logs."
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--dram-pool-size-gb", type=float, required=True)
    parser.add_argument("--fs-pool-size-gb", type=float, required=True)
    parser.add_argument("--unified-memory-pool", action="store_true", default=False)
    parser.add_argument("--service-url")
    parser.add_argument("--metrics-timeout", type=float, default=5.0)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _run_single_scenario(task, progress_dict, task_id):
    """Worker function for ProcessPoolExecutor."""
    records, sim_cls, topo, gpu_cap, dram_cap, fs_cap, kwargs, desc = task
    progress_dict[task_id] = (desc, 0, len(records))

    def cb(current, total):
        if (
            total == 0
            or current >= total
            or current * 100 // total > (current - 1) * 100 // total
        ):
            progress_dict[task_id] = (desc, current, total)

    sim = sim_cls(topo, gpu_cap, dram_cap, fs_cap, **kwargs)
    return sim.simulate(records, progress_cb=cb)


class AnalysisRunner:
    """Encapsulates scenario running + result formatting + output.
    Uses polymorphism to eliminate all if-model-type branches."""

    SIMULATOR_MAP = {
        "standard": StandardSimulator,
        "mamba": MambaSimulator,
        "fawa": FAWASimulator,
    }

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def run(self) -> dict:
        facts = LogCollector(self.args.log_dir).collect()
        topo = self._build_topology(facts)
        sim_cls = self._pick_simulator(topo.model_type)

        gpu_kv_bytes = _resolve_gpu_cache_bytes(facts)
        gpu_cap_blocks = gpu_kv_bytes // topo.hbm_block_data_size

        dram_bytes = int(self.args.dram_pool_size_gb * GIB)
        fs_bytes = int(self.args.fs_pool_size_gb * GIB)

        dram_per_pool = topo.dram_per_pool_bytes(dram_bytes)
        fs_cap = topo.fs_capacity_bytes(fs_bytes)

        scenario_defs = self._build_scenario_defs(facts, topo, sim_cls)
        base_kwargs = self._build_base_kwargs(topo, facts)

        # Build task list (polymorphic: FAWA runs two modes)
        tasks: list = []
        task_labels: list = []
        for mode_name, mode_extra in sim_cls.dump_modes():
            kwargs = {**base_kwargs, **mode_extra}
            for name, gpu_cap, dram_cap, fs_cap in scenario_defs:
                desc = f"[{mode_name}] {name}" if mode_name else name
                tasks.append(
                    (
                        facts.records,
                        sim_cls,
                        topo,
                        gpu_cap,
                        dram_cap,
                        fs_cap,
                        kwargs,
                        desc,
                    )
                )
                task_labels.append((mode_name, name))

        # Run scenarios in parallel
        scenarios: dict[str, dict] = {}
        num_bars = len(tasks)
        manager = multiprocessing.Manager()
        progress_dict = manager.dict()
        max_workers = min(num_bars, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _run_single_scenario, tasks[i], progress_dict, i
                ): task_labels[i]
                for i in range(num_bars)
            }

            # Multi-line progress bar rendering
            sys.stderr.write("\n" * num_bars + f"\033[{num_bars}A")
            sys.stderr.flush()

            while not all(f.done() for f in futures):
                for i in range(num_bars):
                    if i in progress_dict:
                        d, cur, tot = progress_dict[i]
                        pct = cur / tot * 100 if tot else 100
                        bl = 30
                        filled = int(bl * cur / tot) if tot else bl
                        bar = "█" * filled + "░" * (bl - filled)
                        line = f"  {d:42s} {pct:5.1f}% |{bar}| {cur}/{tot}"
                    else:
                        line = "  " + " " * 42
                    sys.stderr.write(f"\r{line}\033[K\n")
                sys.stderr.write(f"\033[{num_bars}A")
                sys.stderr.flush()
                time.sleep(0.1)

            # Collect results
            for future in as_completed(futures):
                mode_name, name = futures[future]
                key = mode_name or "default"
                scenarios.setdefault(key, {})[name] = future.result()

            # Final render (all 100%)
            for i in range(num_bars):
                d, _, tot = progress_dict.get(i, ("", 0, 0))
                bar = "█" * 30
                line = f"  {d:42s} {100.0:5.1f}% |{bar}| {tot}/{tot}"
                sys.stderr.write(f"\r{line}\033[K\n")
            sys.stderr.flush()

        analysis = self._format_analysis(scenarios, facts)

        service_metrics = (
            _fetch_service_hit_rate(self.args.service_url, self.args.metrics_timeout)
            if self.args.service_url
            else None
        )
        if service_metrics:
            analysis["service_actual_kv_cache_hit_rate_percent"] = (
                service_metrics["actual_kv_cache_hit_rate"] * 100
            )

        model_type = topo.model_type
        return {
            "inputs": {
                "log_dir": str(self.args.log_dir),
                "model_type": model_type,
                "is_mla": topo.is_mla,
                "hbm_block_data_size": topo.hbm_block_data_size,
                "dram_pool_size_gb": self.args.dram_pool_size_gb,
                "fs_pool_size_gb": self.args.fs_pool_size_gb,
                "chunk_size": facts.max_num_batched_tokens or 8192,
                "async_scheduling": facts.async_scheduling,
                "unified_memory_pool": self.args.unified_memory_pool,
                "service_url": self.args.service_url,
            },
            "derived": {
                "log_files": facts.log_files,
                "tp_size": topo.tp_size,
                "dp_size": topo.dp_size,
                "num_nodes": self.args.num_nodes,
                "gpu_kv_cache_bytes": gpu_kv_bytes,
                "gpu_capacity_blocks": gpu_cap_blocks,
                "dram_per_pool_bytes": dram_per_pool,
                "fs_capacity_bytes": fs_cap,
            },
            "analysis": analysis,
            "simulation_details": scenarios,
        }

    def _build_topology(self, facts: LogFacts) -> SimTopology:
        if facts.sim_topology is None:
            raise ValueError(
                "No UCMTraceMeta found in log. Use a Lite connector "
                "(use_lite=true) to produce the topology line."
            )
        topo = facts.sim_topology
        topo.unified = self.args.unified_memory_pool
        topo.num_nodes = self.args.num_nodes
        topo.dp_size = _resolve_single(facts.data_parallel_sizes, "data_parallel_size")
        topo.tp_size = _resolve_single(
            facts.tensor_parallel_sizes, "tensor_parallel_size"
        )
        if topo.hbm_block_data_size <= 0:
            raise ValueError(
                "hbm_block_data_size must be > 0 (from UCMTraceMeta topology line)"
            )
        return topo

    def _pick_simulator(self, model_type: str) -> type[HitRateSimulator]:
        cls = self.SIMULATOR_MAP.get(model_type)
        if cls is None:
            raise ValueError(f"unknown model type: {model_type}")
        return cls

    def _build_scenario_defs(
        self,
        facts: LogFacts,
        topo: SimTopology,
        sim_cls: type[HitRateSimulator],
    ) -> list[tuple[str, int, int, int]]:
        trace_bs = topo.trace_hash_block_size or topo.vllm_hash_block_size
        scale = max(1, trace_bs // topo.vllm_hash_block_size)
        unique_blocks = len({h for r in facts.records for h in r.hash_ids}) * scale
        total_hash_count = sum(len(r.hash_ids) for r in facts.records) * scale
        theoretical_blocks = max(unique_blocks, total_hash_count) * 100
        theoretical_bytes = theoretical_blocks * sim_cls.entry_byte_size(topo)

        gpu_kv_bytes = _resolve_gpu_cache_bytes(facts)
        gpu_cap_blocks = gpu_kv_bytes // topo.hbm_block_data_size

        dram_bytes = int(self.args.dram_pool_size_gb * GIB)
        fs_bytes = int(self.args.fs_pool_size_gb * GIB)
        dram_per_pool = topo.dram_per_pool_bytes(dram_bytes)
        fs_cap = topo.fs_capacity_bytes(fs_bytes)

        return [
            (
                "theoretical_max",
                theoretical_blocks,
                theoretical_bytes,
                theoretical_bytes,
            ),
            ("hbm", gpu_cap_blocks, 0, 0),
            ("hbm_dram", gpu_cap_blocks, dram_per_pool, 0),
            ("hbm_dram_fs", gpu_cap_blocks, dram_per_pool, fs_cap),
        ]

    def _build_base_kwargs(self, topo: SimTopology, facts: LogFacts) -> dict:
        return {
            "random_seed": self.args.random_seed,
            "chunk_size": facts.max_num_batched_tokens or 8192,
            "async_scheduling": facts.async_scheduling,
        }

    def _format_analysis(self, scenarios: dict[str, dict], facts: LogFacts) -> dict:
        def pct(r):
            return r["hit_rate"] * 100

        result: dict = {
            "total_request_count": len(facts.records),
            "total_request_token_count": sum(r.input_length for r in facts.records),
        }

        for mode_name, mode_scenarios in scenarios.items():
            prefix = f"{mode_name}_" if mode_name != "default" else ""
            ms = mode_scenarios
            result[f"{prefix}theoretical_max_percent"] = pct(ms["theoretical_max"])
            result[f"{prefix}hbm_percent"] = pct(ms["hbm"])
            result[f"{prefix}hbm_dram_percent"] = pct(ms["hbm_dram"])
            result[f"{prefix}hbm_dram_fs_percent"] = pct(ms["hbm_dram_fs"])

        # Lifetime stats: take first mode's theoretical_max
        first_mode = next(iter(scenarios.values()))
        tmax = first_mode["theoretical_max"]
        result["request_lifetime_sample_count"] = tmax["request_lifetime_sample_count"]
        result["average_request_lifetime_seconds"] = tmax[
            "average_request_lifetime_seconds"
        ]
        result["p90_request_lifetime_seconds"] = tmax["p90_request_lifetime_seconds"]
        result["p95_request_lifetime_seconds"] = tmax["p95_request_lifetime_seconds"]

        return result

    def print_summary(self, result: dict) -> None:
        a = result["analysis"]
        d = result["derived"]
        i = result["inputs"]
        print("Trace cache hit rate analysis")
        print(f"  Model type: {i['model_type']}")
        print(f"  Total request count: {a['total_request_count']}")
        print(f"  Total request token count: {a['total_request_token_count']}")
        print(f"  HBM available: {d['gpu_kv_cache_bytes'] / GIB:.2f} GiB")
        print(f"  TP={d['tp_size']}  DP={d['dp_size']}  Nodes={d['num_nodes']}")
        print(
            f"  DRAM pool: {i['dram_pool_size_gb']:.2f} GiB  "
            f"FS pool: {i['fs_pool_size_gb']:.2f} GiB"
        )

        sim_details = result["simulation_details"]
        for mode_name, mode_scenarios in sim_details.items():
            if mode_name != "default":
                print(f"  [{mode_name}]")
                indent = "    "
            else:
                indent = "  "
            prefix = f"{mode_name}_" if mode_name != "default" else ""
            print(
                f"{indent}Theoretical max: "
                f"{a[f'{prefix}theoretical_max_percent']:.2f}%"
            )
            if "service_actual_kv_cache_hit_rate_percent" in a:
                print(
                    f"{indent}Service actual:  "
                    f"{a['service_actual_kv_cache_hit_rate_percent']:.2f}%"
                )
            print(f"{indent}HBM:              {a[f'{prefix}hbm_percent']:.2f}%")
            print(f"{indent}HBM+DRAM:         {a[f'{prefix}hbm_dram_percent']:.2f}%")
            print(
                f"{indent}HBM+DRAM+FS:      "
                f"{a[f'{prefix}hbm_dram_fs_percent']:.2f}%"
            )

        print(
            f"  Request lifetime sample count: " f"{a['request_lifetime_sample_count']}"
        )
        print(
            f"  Average request lifetime: "
            f"{a['average_request_lifetime_seconds']:.6f} s"
        )
        print(f"  P90 request lifetime: {a['p90_request_lifetime_seconds']:.6f} s")
        print(f"  P95 request lifetime: {a['p95_request_lifetime_seconds']:.6f} s")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        runner = AnalysisRunner(args)
        result = runner.run()
        if args.trace_output:
            facts = LogCollector(args.log_dir).collect()
            args.trace_output.parent.mkdir(parents=True, exist_ok=True)
            with args.trace_output.open("w", encoding="utf-8") as f:
                for r in facts.records:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": r.timestamp,
                                "input_length": r.input_length,
                                "output_length": r.output_length,
                                "hash_ids": r.hash_ids,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        runner.print_summary(result)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
