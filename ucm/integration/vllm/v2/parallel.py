"""Same-topology shard identity and all-shard visibility for connector v2."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import lcm
from typing import TYPE_CHECKING, Any

from .ucm_proxy import KVCacheValue, UCMProxy

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .layout import UCMKVCacheSpec


@dataclass(frozen=True)
class ParallelLayout:
    tp: int = 1
    pp: int = 1
    pcp: int = 1
    dcp: int = 1
    interleave: int = 1
    pp_partition: str = ""

    @classmethod
    def from_config(cls, config: VllmConfig) -> ParallelLayout:
        p = config.parallel_config
        result = cls(
            tp=p.tensor_parallel_size,
            pp=p.pipeline_parallel_size,
            pcp=p.prefill_context_parallel_size,
            dcp=p.decode_context_parallel_size,
            interleave=p.cp_kv_cache_interleave_size,
        )
        # UCM's shard mapping treats DCP as a TP subgroup. Upstream also
        # permits PCP-spanning DCP, which this implementation does not handle.
        if result.tp % result.dcp:
            raise ValueError("v2 CP requires DCP to be a TP subgroup")
        return replace(result, pp_partition=os.getenv("VLLM_PP_LAYER_PARTITION", ""))

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.pcp

    @property
    def namespace(self) -> str:
        value = (
            f"tp{self.tp}-pp{self.pp}-pcp{self.pcp}-dcp{self.dcp}-i{self.interleave}"
        )
        if self.pp_partition:
            value += (
                "-partition"
                + hashlib.sha256(self.pp_partition.encode()).hexdigest()[:16]
            )
        return value

    def apply_context_parallel(
        self, spec: UCMKVCacheSpec, requested_tokens_per_chunk: int | None = None
    ) -> UCMKVCacheSpec:
        cp = self.pcp * self.dcp
        if cp == 1:
            return spec
        if spec.wa_groups or spec.state_groups:
            raise NotImplementedError("v2 CP currently supports full attention only")
        groups = tuple(
            (
                replace(g, tokens_per_block=g.tokens_per_block * cp)
                if g.is_persistent
                else g
            )
            for g in spec.groups
        )
        persistent_groups = tuple(g for g in groups if g.is_persistent)
        # Only complete distributed pages: a fractional CP page needs explicit
        # striped-token mapping, not proportional source offsets.
        tokens_per_chunk = requested_tokens_per_chunk or lcm(
            *(g.tokens_per_block for g in persistent_groups)
        )
        if any(tokens_per_chunk % g.tokens_per_block for g in persistent_groups):
            raise ValueError("CP UCM blocks must contain whole distributed KV pages")
        return replace(
            spec,
            groups=groups,
            tokens_per_chunk=tokens_per_chunk,
        )


class AllShardLookup:
    """Storage wrapper: a logical key is ready only when every shard is committed.

    Rank-local dump/load still use the unchanged keys/[K,S] protocol. Each
    shard has its own namespace; this intentionally does not reshard TP/PP/CP.
    """

    def __init__(self, local: UCMProxy, shards: Sequence[UCMProxy]) -> None:
        self.local = local
        self.shards = tuple(shards)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.local, name)

    def wait(self, task: object) -> None:
        return self.local.wait(task)

    def register_tensors(self, caches: Mapping[str, KVCacheValue]) -> None:
        method = getattr(self.local, "register_tensors", None)
        if method is not None:
            return method(caches)

    def lookup(self, keys: Sequence[bytes]) -> tuple[bool, ...]:
        hits = [True] * len(keys)
        for shard in self.shards:
            hits = [a and b for a, b in zip(hits, shard.lookup(keys), strict=True)]
        return tuple(hits)

    def lookup_on_prefix(self, keys: Sequence[bytes]) -> int:
        for index, hit in enumerate(self.lookup(keys)):
            if not hit:
                return index - 1
        return len(keys) - 1

    def lookup_on_reverse(self, keys: Sequence[bytes]) -> int:
        hits = self.lookup(keys)
        return next((i for i in range(len(hits) - 1, -1, -1) if hits[i]), -1)
