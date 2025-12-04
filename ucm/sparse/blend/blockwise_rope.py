import torch
import triton
import triton.language as tl


@triton.jit
def _triton_rope_blockwise_kernel(
    k_ptr,  # (total_blocks, seq_len, n_kv_head, hd)
    vllm_ids,  # (bs,) block id for each batch
    positions,  # (bs,) delta angle for each batch
    cos_sin_cache,  # (1, seq_len, hd)
    k_row_stride,
    k_head_stride,
    cos_sin_row_stride,
    sl,
    bs: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_hd: tl.constexpr,
):
    """
    each program/batch process a single head for each token
    programs matrix (batch_idx, seq_idx, head_idx)
    """
    pid = tl.program_id(0)

    heads_per_seq = n_kh
    tokens_per_batch = sl * n_kh
    batch_idx = pid // tokens_per_batch
    seq_head_idx = pid % tokens_per_batch
    seq_idx = seq_head_idx // n_kh
    head_idx = seq_head_idx % n_kh

    # block id & position
    block_id = tl.load(vllm_ids + batch_idx)
    pos_idx = tl.load(positions + batch_idx)

    # k offset
    k_offset = block_id * k_row_stride + seq_idx * (n_kh * hd) + head_idx * hd
    k_ptr = k_ptr + k_offset

    # fetch  cos sin from cos_sin_cache
    cos_base = pos_idx * cos_sin_row_stride
    sin_base = cos_base + hd // 2  # sin just behind cos

    offs = tl.arange(0, pad_hd // 2)
    mask = offs < hd // 2

    cos_row = tl.load(cos_sin_cache + cos_base + offs, mask=mask, other=0)
    sin_row = tl.load(cos_sin_cache + sin_base + offs, mask=mask, other=0)

    k_tile_1 = tl.load(k_ptr + offs, mask=mask, other=0).to(cos_row.dtype)
    k_tile_2 = tl.load(k_ptr + offs + hd // 2, mask=mask, other=0).to(cos_row.dtype)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row

    tl.store(k_ptr + offs, new_k_tile_1, mask=mask)
    tl.store(k_ptr + offs + hd // 2, new_k_tile_2, mask=mask)


def block_wise_rope_forward(
    k_cache: torch.Tensor,
    vllm_ids: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """
    Args：
        k_cache: torch.Tensor (total_blocks, seq_len, n_kv_heads, hd), vllm owned.
        vllm_ids: torch.LongTensor (batch_size,), vllm block id
        positions: torch.LongTensor (batch_size,), delta angle of each block for rope
        cos_sin_cache: torch.Tensor (1, seq_len, hd)，same as the tensor in rotary_emb
    """
    total_blocks, seq_len, n_kv_head, head_dim = k_cache.shape
    batch_size = vllm_ids.shape[0]
    pad_hd = triton.next_power_of_2(head_dim)

    k_cache = k_cache.contiguous()
    vllm_ids = vllm_ids.contiguous()
    positions = positions.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()

    n_row = batch_size * seq_len * n_kv_head

    _triton_rope_blockwise_kernel[(n_row,)](
        k_cache,
        vllm_ids,
        positions,
        cos_sin_cache,
        k_cache.stride(0),
        k_cache.stride(-2),
        cos_sin_cache.stride(-2),
        seq_len,
        batch_size,
        n_kv_head,
        head_dim,
        pad_hd,
    )

    return k_cache


def rope_naive_torch(
    k_cache: torch.Tensor,
    vllm_ids: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    """
    naive torch implementation for accuracy and perf baseline
    Args:
        k_cache: (total_blocks, seq_len, n_heads, hd)
        vllm_ids: (bs,)
        positions: (bs,)
        cos_sin_cache: (1, seq_len, hd)
    Returns:
        rotated_k: same shape as k_cache
    """
    total_blocks, sl, nh, hd = k_cache.shape
    bs = vllm_ids.shape[0]

    # copy to avoid in-place modifying original
    k_out = k_cache.clone()

    half = hd // 2

    # cos_sin_cache shape: (1, seq_len, hd)
    cos_sin_cache = cos_sin_cache.squeeze(0)  # (sl, hd)
    cos_table = cos_sin_cache[:, :half]  # (sl, half)
    sin_table = cos_sin_cache[:, half:]  # (sl, half)

    # Loop in python (slow but clear)
    for b in range(bs):
        blk = vllm_ids[b].item()
        pos = positions[b].item()  # rope offset

        for s in range(sl):
            # cos, sin row for this position
            cos = cos_table[pos]  # (half,)
            sin = sin_table[pos]

            for h in range(nh):
                # read original k
                k_vec = k_out[blk, s, h]  # (hd,)
                k1 = k_vec[:half]  # (half,)
                k2 = k_vec[half:]  # (half,)

                # rope rotate
                new_k1 = k1 * cos - k2 * sin
                new_k2 = k2 * cos + k1 * sin

                # write back
                k_out[blk, s, h, :half] = new_k1
                k_out[blk, s, h, half:] = new_k2

    return k_out


if __name__ == "__main__":
    import time

    torch.manual_seed(42)

    total_blocks = 5120
    num_blocks = 128
    block_size = 128
    max_num_tokens = num_blocks * block_size
    num_heads = 8
    head_size = 128
    dtype = torch.bfloat16

    kcache = torch.randn(
        total_blocks, block_size, num_heads, head_size, device="cuda", dtype=dtype
    )
    vllm_ids = torch.randint(
        0, total_blocks, (num_blocks,), device="cuda", dtype=torch.long
    )
    positions = torch.randint(
        0, max_num_tokens, (num_blocks,), device="cuda", dtype=torch.long
    )
    cos_sin_cache = torch.randn(max_num_tokens, head_size, device="cuda", dtype=dtype)

    # naive torch result
    baseline_rope_kcache = rope_naive_torch(kcache, vllm_ids, positions, cos_sin_cache)

    triton_rope_kcache = block_wise_rope_forward(
        kcache, vllm_ids, positions, cos_sin_cache
    )

    # precision compare
    diff = (triton_rope_kcache[vllm_ids] - baseline_rope_kcache[vllm_ids]).abs()
    mean_err = diff.mean().item()
    print(f"MAE : {mean_err:.6f}. Expected 1e-3")

    def bench(fn, n_iter=50):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iter
        return dt * 1e3  # ms

    ms = bench(
        lambda: block_wise_rope_forward(kcache, vllm_ids, positions, cos_sin_cache)
    )
    print(f"Kernel avg latency: {ms:.3f} ms. Expected 100 us")

    # load K,load cos,sin -> dump K
    bytes_total = (
        num_blocks
        * block_size
        * num_heads
        * (
            head_size * kcache.dtype.itemsize  # K load
            + vllm_ids.dtype.itemsize  # vllm_ids load
            + positions.dtype.itemsize  # positions load
            + head_size * cos_sin_cache.dtype.itemsize  # cos sin load
            + head_size * kcache.dtype.itemsize  # K dump
        )
    )
    bw = bytes_total / (ms / 1e3) / (1024**3)
    print(f"Estimated memory BW: {bw:.1f} GiB/s")
