"""TTFT analysis model for UCM prefix-cache hits."""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import ToolkitError

_GB = 10**9
_MS_PER_S = 1000.0


class TtftAnalysisError(ToolkitError):
    """Raised when TTFT analysis inputs are invalid."""


@dataclass
class TtftInputs:
    """User-supplied TTFT analysis inputs.

    ``cache_size_bytes`` is the KV cache bytes a single card must load into its
    HBM (already TP-sharded for GQA, full latent for MLA).
    """

    cache_size_bytes: float
    tp: int
    posix_bw_gbps: float
    h2d_bw_gbps: float
    ttft_prefill_ms: float
    ttft_hbm_ms: float


@dataclass
class TtftBreakdown:
    """Per-load-mode TTFT analysis result."""

    storage_read_ms: float
    h2d_ms: float
    t_cache_load_ms: float
    ttft_ucm_ms: float
    gain_vs_prefill: float
    loss_vs_hbm: float
    bottleneck: str


def _validate(inputs: TtftInputs) -> None:
    if inputs.cache_size_bytes <= 0:
        raise TtftAnalysisError("cache size must be positive")
    if inputs.tp < 1:
        raise TtftAnalysisError("tp must be at least 1")
    if inputs.posix_bw_gbps <= 0 or inputs.h2d_bw_gbps <= 0:
        raise TtftAnalysisError("bandwidth must be positive")
    if inputs.ttft_prefill_ms <= 0 or inputs.ttft_hbm_ms <= 0:
        raise TtftAnalysisError("TTFT baselines must be positive")


def t_cache_load(inputs: TtftInputs) -> float:
    """Return the cache load time in milliseconds (storage read + H2D)."""
    per_card_gb = inputs.cache_size_bytes / _GB
    storage_read_ms = per_card_gb * inputs.tp / inputs.posix_bw_gbps * _MS_PER_S
    h2d_ms = per_card_gb / inputs.h2d_bw_gbps * _MS_PER_S
    return storage_read_ms + h2d_ms


def _bottleneck(storage_read_ms: float, h2d_ms: float, compute_ms: float) -> str:
    parts = {"storage-read": storage_read_ms, "H2D": h2d_ms, "compute": compute_ms}
    return max(parts, key=parts.get)


def analyze(inputs: TtftInputs, load_mode: str) -> TtftBreakdown:
    """Compute the TTFT breakdown for a given load mode."""
    _validate(inputs)
    per_card_gb = inputs.cache_size_bytes / _GB
    storage_read_ms = per_card_gb * inputs.tp / inputs.posix_bw_gbps * _MS_PER_S
    h2d_ms = per_card_gb / inputs.h2d_bw_gbps * _MS_PER_S
    load_ms = storage_read_ms + h2d_ms

    if load_mode == "layered":
        ttft_ucm_ms = max(inputs.ttft_hbm_ms, load_ms)
    else:
        ttft_ucm_ms = inputs.ttft_hbm_ms + load_ms

    return TtftBreakdown(
        storage_read_ms=storage_read_ms,
        h2d_ms=h2d_ms,
        t_cache_load_ms=load_ms,
        ttft_ucm_ms=ttft_ucm_ms,
        gain_vs_prefill=inputs.ttft_prefill_ms / ttft_ucm_ms,
        loss_vs_hbm=ttft_ucm_ms / inputs.ttft_hbm_ms,
        bottleneck=_bottleneck(storage_read_ms, h2d_ms, inputs.ttft_hbm_ms),
    )
