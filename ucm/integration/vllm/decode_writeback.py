"""Request-scoped Decode offload planning, independent of device/runtime APIs."""

from dataclasses import dataclass, field
from typing import Any

from ucm.integration.vllm.request_hasher import IncrementalRequestHasher


def validate_decode_config(launch_config, vllm_config, kv_cache_config) -> bool:
    enabled = launch_config.get("save_decode_cache", False)
    if not isinstance(enabled, bool):
        raise ValueError("save_decode_cache must be a boolean")
    if not enabled:
        return False
    if kv_cache_config is None:
        raise ValueError("save_decode_cache requires vLLM 0.30 KVCacheConfig")
    if launch_config.get("use_layerwise", True):
        raise ValueError("save_decode_cache requires use_layerwise=false")
    if (
        launch_config.get("use_lite", False)
        or "hit_ratio" in launch_config
        or launch_config.get("use_inference_duration_monitor", False)
    ):
        raise ValueError("Decode writeback requires a real storage connector")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, key, 1) != 1
        for key in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    ):
        raise ValueError("Decode writeback currently requires PP=PCP=DCP=1")
    return True


def window_boundary_available(
    computed_end, hash_size, window_block_sizes, scheduled_end=None
) -> bool:
    """A complete boundary page remains immutable until the next page expires.

    The page containing the first token after the boundary may be partially
    written. Never include it in the checkpoint. With a remainder smaller than
    each window page span, the prior tail's first page is still needed by the
    current window; the block-table validity check guards actual availability.
    Private in-place compressor rings must not be included in these groups.
    """
    boundary = computed_end // hash_size * hash_size
    remainder = (computed_end if scheduled_end is None else scheduled_end) - boundary
    return all(
        hash_size % size == 0 and 0 <= remainder < size for size in window_block_sizes
    )


@dataclass(frozen=True)
class WritebackJob:
    request_id: str
    generation: int
    job_id: int
    start: int
    end: int
    keys: tuple[bytes, ...]
    phase: str
    ranks: tuple[int, ...]


@dataclass
class WritebackResult:
    job: WritebackJob
    completed: set[int] = field(default_factory=set)
    failed: set[int] = field(default_factory=set)

    def merge(self, other: "WritebackResult") -> None:
        if self.job != other.job:
            raise ValueError("Conflicting writeback job identities")
        self.completed.update(other.completed)
        self.failed.update(other.failed)


@dataclass
class DecodeWritebackState:
    request: Any
    generation: int
    prefill_end: int
    hasher: IncrementalRequestHasher
    blocks: list[list[int]] = field(default_factory=list)
    saved: set[bytes] = field(default_factory=set)
    pending: dict[int, WritebackResult] = field(default_factory=dict)
    computed_end: int = 0
    scheduled_end: int = 0
    scheduled_once: bool = False

    def update_blocks(self, groups, *, replace=False) -> None:
        if groups is None:
            return
        if replace or not self.blocks:
            self.blocks = [list(group) for group in groups]
        else:
            if len(groups) != len(self.blocks):
                raise ValueError("KV group count changed within a request generation")
            for dst, src in zip(self.blocks, groups):
                dst.extend(src)

    def refresh_request(self, scheduled_tokens: int) -> tuple[list[bytes], bool]:
        start = self.request.num_computed_tokens
        self.scheduled_end = start + scheduled_tokens
        self.computed_end = min(self.scheduled_end, len(self.request.all_token_ids))
        hashes = self.hasher.update(self.request, self.computed_end)
        return hashes, start >= self.prefill_end

    def plan_save(self, hashes, job_id, ranks, phase):
        # One outstanding job per request bounds metadata and avoids duplicate
        # PUTs while the previous worker completion is still in flight.
        if self.pending:
            return None
        first = next((i for i, key in enumerate(hashes) if key not in self.saved), None)
        if first is None:
            return None
        size = self.hasher.block_size
        job = WritebackJob(
            self.request.request_id,
            self.generation,
            job_id,
            first * size,
            len(hashes) * size,
            tuple(hashes[first:]),
            phase,
            tuple(ranks),
        )
        self.pending[job_id] = WritebackResult(job)
        return job

    def complete_save(self, result: WritebackResult) -> bool:
        if result.job.generation != self.generation:
            return False
        pending = self.pending.get(result.job.job_id)
        if pending is None:
            return False
        pending.merge(result)
        if not set(pending.job.ranks).issubset(pending.completed):
            return False
        if not pending.failed:
            self.saved.update(pending.job.keys)
        del self.pending[result.job.job_id]
        return not pending.failed


def prefill_end_tokens(request, scheduled_new=None) -> int:
    """Capture the re-prefill range once, for both MRV1 and MRV2."""
    ids = getattr(scheduled_new, "prefill_token_ids", None)
    return len(request.all_token_ids if ids is None else ids)


def merge_writeback_results(dst, src) -> None:
    for job_id, result in src.items():
        if job_id not in dst:
            dst[job_id] = WritebackResult(result.job)
        dst[job_id].merge(result)
