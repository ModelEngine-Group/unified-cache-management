"""Parallel topology for connector v2: hash namespace and CP block sizes."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        parallel_config = config.parallel_config
        tp = parallel_config.tensor_parallel_size
        pp = parallel_config.pipeline_parallel_size
        pcp = parallel_config.prefill_context_parallel_size
        dcp = parallel_config.decode_context_parallel_size
        interleave = parallel_config.cp_kv_cache_interleave_size
        if min(tp, pp, pcp, dcp, interleave) < 1:
            raise ValueError("Parallel dimensions must be positive")
        # Same rule as vLLM: without PCP, DCP must divide TP; with PCP, DCP
        # must be 1, PCP or TP * PCP.
        if pcp == 1:
            is_dcp_valid = tp % dcp == 0
        else:
            is_dcp_valid = dcp in (1, pcp, tp * pcp)
        if is_dcp_valid:
            return cls(
                tp=tp,
                pp=pp,
                pcp=pcp,
                dcp=dcp,
                interleave=interleave,
                pp_partition=os.getenv("VLLM_PP_LAYER_PARTITION", ""),
            )
        else:
            raise ValueError(f"Unsupported DCP={dcp} for TP={tp}, PCP={pcp}")

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.pcp

    @property
    def namespace(self) -> str:
        namespace = (
            f"tp{self.tp}-pp{self.pp}-pcp{self.pcp}-dcp{self.dcp}-i{self.interleave}"
        )
        if self.pp_partition:
            namespace += (
                "-partition"
                + hashlib.sha256(self.pp_partition.encode()).hexdigest()[:16]
            )
        return namespace

    def apply_context_parallel(self, spec: UCMKVCacheSpec) -> None:
        """Multiply block sizes by DCP in place.

        vLLM gives per-rank block sizes. Like resolve_dcp_kv_block_size, only
        dcp_sharded groups cover dcp times more tokens. PCP does not split KV.
        The UCM block size comes from vLLM's scheduler block size, which
        already counts DCP, so it must hold whole DCP-scaled blocks.
        """
        if self.dcp == 1:
            return
        elif spec.wa_groups or spec.state_groups:
            raise NotImplementedError("v2 CP currently supports full attention only")
        else:
            # Transient groups are not stored, so they keep their block size.
            stored_groups = [group for group in spec.groups if not group.is_transient]
            token_block_sizes = [
                group.token_block_size * (self.dcp if group.dcp_sharded else 1)
                for group in stored_groups
            ]
            ucm_block_size = spec.ucm_cache_block_size
            if any(
                ucm_block_size % token_block_size
                for token_block_size in token_block_sizes
            ):
                # This also rejects FA UCM blocks smaller than a vLLM block,
                # which are not supported with DCP.
                raise ValueError(
                    "CP UCM blocks must contain whole distributed KV pages"
                )
            else:
                for group, token_block_size in zip(stored_groups, token_block_sizes):
                    group.token_block_size = token_block_size
                    group.tail_blocks = ucm_block_size // token_block_size
