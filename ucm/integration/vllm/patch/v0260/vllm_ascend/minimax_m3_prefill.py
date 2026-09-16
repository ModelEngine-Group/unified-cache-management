"""Backport vllm-ascend #16121 without changing its prefill API.

Adapted from commit 12be3d34a1510be8e19542f577e732a12c47a1d1.
Only imports are adjusted to reuse the installed, unchanged Ascend kernels.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm_ascend.models.minimax_m3.ops.msa_m3_triton import (
    PREFILL_SCORE_QUERY_TILE_SIZE,
    SCORE_BLOCK_STRIDE_ALIGNMENT,
    SPARSE_BLOCK_SIZE,
    _as_triton_index_kv_cache,
    _copy_topk_indices,
    _mask_prefill_topk_indices_kernel,
    _prefill_index_score_kernel,
    init_device_properties_triton,
)

try:
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
except ImportError:
    get_vectorcore_num = None

PREFILL_PREPARE_FALLBACK_VECTORCORE_COUNT = 64


@lru_cache(maxsize=1)
def _detected_vectorcore_count() -> int:
    """Returns the detected Ascend Vector Core count, or zero on fallback."""
    if get_vectorcore_num is None or init_device_properties_triton is None:
        return 0
    try:
        detected = int(get_vectorcore_num())
        if detected > 0:
            return detected
    except AssertionError:
        pass
    except Exception:
        return 0
    try:
        init_device_properties_triton()
        return max(0, int(get_vectorcore_num()))
    except Exception:
        return 0


@triton.jit(do_not_specialize=["score_block_count"])
def _prepare_prefill_topk_scores_kernel(
    score_ptr,
    query_start_offsets_ptr,
    prefix_lengths_ptr,
    index_head_count: tl.constexpr,
    init_block_count: tl.constexpr,
    local_block_count: tl.constexpr,
    score_block_count,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    sparse_block_size: tl.constexpr,
    MAX_QUERY_LEN: tl.constexpr,
    PROGRAMS_PER_BATCH_HEAD: tl.constexpr,
    BLOCK_SIZE_FORCE: tl.constexpr,
):
    """Applies init/local priorities with contiguous block stores."""
    query_program_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // index_head_count
    head_id = batch_head_id % index_head_count

    # Programs are divided across each (batch, head) group. Each program owns
    # one contiguous query range; query is a scalar loop dimension, while the
    # vector dimension is always the contiguous score block dimension.
    base_queries: tl.constexpr = MAX_QUERY_LEN // PROGRAMS_PER_BATCH_HEAD
    extra_programs: tl.constexpr = MAX_QUERY_LEN % PROGRAMS_PER_BATCH_HEAD
    query_count = base_queries + tl.where(query_program_id < extra_programs, 1, 0)
    query_start = query_program_id * base_queries + tl.minimum(
        query_program_id, extra_programs
    )

    sequence_start = tl.load(query_start_offsets_ptr + batch_id)
    sequence_end = tl.load(query_start_offsets_ptr + batch_id + 1)
    query_length = sequence_end - sequence_start
    prefix_length = tl.load(prefix_lengths_ptr + batch_id)

    block_offsets = tl.arange(0, BLOCK_SIZE_FORCE)
    safe_last_block = tl.maximum(score_block_count - 1, 0)

    for query_inner in tl.range(0, query_count):
        query_offset = query_start + query_inner
        query_valid = query_offset < query_length
        safe_query_offset = tl.where(query_valid, query_offset, 0)
        token_index = sequence_start + safe_query_offset

        valid_block_count = (
            prefix_length + safe_query_offset + sparse_block_size
        ) // sparse_block_size
        valid_block_count = tl.minimum(
            tl.maximum(valid_block_count, 0), score_block_count
        )
        local_start = tl.maximum(valid_block_count - local_block_count, 0)
        row_base = (
            score_ptr + head_id * score_head_stride + token_index * score_token_stride
        )

        if init_block_count > 0:
            init_count = tl.minimum(init_block_count, valid_block_count)
            init_block_ids = tl.minimum(block_offsets, safe_last_block)
            init_mask = query_valid & (block_offsets < init_count)
            tl.store(
                row_base + init_block_ids * score_block_stride, 1e30, mask=init_mask
            )

        if local_block_count > 0:
            local_count = valid_block_count - local_start
            local_block_ids = local_start + block_offsets
            safe_local_block_ids = tl.minimum(local_block_ids, safe_last_block)
            local_mask = query_valid & (block_offsets < local_count)
            tl.store(
                row_base + safe_local_block_ids * score_block_stride,
                1e29,
                mask=local_mask,
            )


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Computes one block score for every visible prefill KV block.

    ``sm_scale`` is accepted for API compatibility. A positive global scale
    does not change block ordering, so this score-only path intentionally omits
    it.
    """
    del sm_scale
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    assert index_head_count == num_kv_heads, "M3 requires num_idx_heads == num_kv_heads"

    batch_size = cu_seqlens_q.shape[0] - 1
    max_block_count = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(
        max_block_count,
        SCORE_BLOCK_STRIDE_ALIGNMENT,
    )
    score = torch.full(
        (index_head_count, total_query_tokens, score_block_stride),
        float("-inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )

    score_grid = (
        triton.cdiv(max_query_len, PREFILL_SCORE_QUERY_TILE_SIZE),
        batch_size * index_head_count,
    )
    _prefill_index_score_kernel[score_grid](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        index_head_count,
        head_dim,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=PREFILL_SCORE_QUERY_TILE_SIZE,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Finalizes prefill scores and returns zero-based block IDs."""
    assert topk > 0
    index_head_count, total_query_tokens, score_block_count = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    force_tile_size = triton.next_power_of_2(max(1, init_blocks, local_blocks))
    batch_head_count = max(1, batch_size * index_head_count)
    detected_vectorcore_count = _detected_vectorcore_count()
    target_program_count = (
        detected_vectorcore_count
        if detected_vectorcore_count > 0
        else PREFILL_PREPARE_FALLBACK_VECTORCORE_COUNT
    )
    programs_per_batch_head = max(
        1, triton.cdiv(target_program_count, batch_head_count)
    )
    programs_per_batch_head = min(programs_per_batch_head, max(1, max_query_len))
    prepare_grid = (programs_per_batch_head, batch_head_count)
    _prepare_prefill_topk_scores_kernel[prepare_grid](
        score,
        cu_seqlens_q,
        prefix_lens,
        index_head_count,
        init_blocks,
        local_blocks,
        score_block_count,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        sparse_block_size=SPARSE_BLOCK_SIZE,
        MAX_QUERY_LEN=max(1, max_query_len),
        PROGRAMS_PER_BATCH_HEAD=programs_per_batch_head,
        BLOCK_SIZE_FORCE=force_tile_size,
    )

    total_query_rows = max(1, max_query_len * batch_size * index_head_count)
    required_query_tile_size = triton.cdiv(total_query_rows, 128)
    base_query_tile_size = min(64, triton.next_power_of_2(required_query_tile_size))

    selected_count = min(topk, score_block_count)
    score_rows = score[:, :total_query_tokens, :score_block_count]
    raw_indices = torch.topk(
        score_rows,
        k=selected_count,
        dim=-1,
    ).indices
    topk_indices = _copy_topk_indices(raw_indices, topk, out)

    topk_tile_size = triton.next_power_of_2(max(1, topk))
    mask_tile_limit = max(1, 2048 // topk_tile_size)
    mask_query_tile_size = min(
        base_query_tile_size,
        1 << (mask_tile_limit.bit_length() - 1),
    )
    mask_grid = (
        triton.cdiv(max_query_len, mask_query_tile_size),
        batch_size * index_head_count,
    )
    _mask_prefill_topk_indices_kernel[mask_grid](
        topk_indices,
        cu_seqlens_q,
        prefix_lens,
        index_head_count,
        SPARSE_BLOCK_SIZE,
        topk,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        BLOCK_SIZE_Q=mask_query_tile_size,
    )
    return topk_indices
