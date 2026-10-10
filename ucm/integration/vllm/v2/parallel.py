"""Same-topology shard identity and all-shard visibility for connector v2."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import lcm
from typing import TYPE_CHECKING, Any

import numpy as np

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
        if min(result.tp, result.pp, result.pcp, result.dcp, result.interleave) < 1:
            raise ValueError("Parallel dimensions must be positive")
        if result.tp % result.dcp:
            raise ValueError("DCP must divide TP; DCP is a TP subgroup")
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
        self, spec: UCMKVCacheSpec, requested_block_size: int | None = None
    ) -> UCMKVCacheSpec:
        cp = self.pcp * self.dcp
        if cp == 1:
            return spec
        if spec.wa_groups or spec.state_groups:
            raise NotImplementedError("v2 CP currently supports full attention only")
        groups = tuple(
            replace(g, token_block_size=g.token_block_size * cp) for g in spec.groups
        )
        # Only complete distributed pages: a fractional CP page needs explicit
        # striped-token mapping, not proportional source offsets.
        ucm_block_size = requested_block_size or lcm(
            *(g.token_block_size for g in groups)
        )
        if any(ucm_block_size % g.token_block_size for g in groups):
            raise ValueError("CP UCM blocks must contain whole distributed KV pages")
        return replace(
            spec,
            groups=tuple(
                replace(g, tail_blocks=ucm_block_size // g.token_block_size)
                for g in groups
            ),
            ucm_cache_block_size=ucm_block_size,
        )


class AllShardLookup:
    """Storage wrapper: a logical ucm_block_id is ready only when every shard is committed.

    Rank-local dump/load still use the unchanged ucm_block_ids/[K,S] protocol. Each
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

    def lookup(
        self, ucm_block_ids: Sequence[bytes], block_indices: np.ndarray
    ) -> tuple[bool, ...]:
        hits = [True] * len(ucm_block_ids)
        for shard in self.shards:
            hits = [
                a and b
                for a, b in zip(
                    hits, shard.lookup(ucm_block_ids, block_indices), strict=True
                )
            ]
        return tuple(hits)

    def lookup_on_prefix(
        self, ucm_block_ids: Sequence[bytes], block_indices: np.ndarray
    ) -> int:
        for index, hit in enumerate(self.lookup(ucm_block_ids, block_indices)):
            if not hit:
                return index - 1
        return len(ucm_block_ids) - 1

    def lookup_on_reverse(
        self, ucm_block_ids: Sequence[bytes], block_indices: np.ndarray
    ) -> int:
        hits = self.lookup(ucm_block_ids, block_indices)
        return next((i for i in range(len(hits) - 1, -1, -1) if hits[i]), -1)

    def close(self) -> None:
        error = None
        for shard in self.shards:
            try:
                shard.close()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error
