import torch
from vllm.triton_utils import tl, triton


@triton.jit
def triton_hash_code_kernel(
    x_ptr,
    code_ptr,
    pack_w_ptr,
    hash_out_ptr,
    M,
    K,
    N,
    stride_xm,
    stride_xk,
    stride_codek,
    stride_coden,
    stride_pack_w,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        code = tl.load(
            code_ptr + offs_k[:, None] * stride_codek + offs_n[None, :] * stride_coden,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(x, code)
        offs_k += BLOCK_K

    bits = (acc > 0).to(tl.uint8)
    bits = tl.reshape(bits, (BLOCK_M, BLOCK_N // 8, 8))

    pack_w = tl.load(pack_w_ptr + tl.arange(0, 8) * stride_pack_w)
    packed = tl.sum(bits * pack_w[None, None, :], axis=-1).to(tl.uint8)

    offs_n = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    hash_out_ptrs = (
        hash_out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    tl.store(
        hash_out_ptrs, packed, mask=(offs_m[:, None] < M) & (offs_n[None, :] < (N // 8))
    )


def triton_hash_code(x, code, pack_weight):
    m = x.shape[:-1]
    K = x.shape[-1]
    x = x.reshape(-1, K)
    M = x.shape[0]
    _, N = code.shape
    assert (pack_weight.shape[0] == 8) and (N % 8 == 0)
    hash_out = torch.empty((M, N // 8), dtype=pack_weight.dtype, device=x.device)

    grid = lambda opts: (
        triton.cdiv(M, opts["BLOCK_M"]),
        triton.cdiv(N, opts["BLOCK_N"]),
    )

    triton_hash_code_kernel[grid](
        x,
        code,
        pack_weight,
        hash_out,
        M,
        K,
        N,
        x.stride(0),
        x.stride(1),
        code.stride(0),
        code.stride(1),
        pack_weight.stride(0),
        hash_out.stride(0),
        hash_out.stride(1),
        BLOCK_M=32,
        BLOCK_K=64,
        BLOCK_N=16,
    )
    return hash_out.view((*m, N // 8))


@torch.compile()
def torch_hash_code(x, code, pack_weight):
    x = x @ code
    m = x.shape[:-1]
    x = (x > 0).to(torch.uint8).view(*m, -1, 8)
    # 8bit -> 1bit
    x = torch.sum(x * pack_weight, dim=-1, dtype=torch.uint8)
    return x
