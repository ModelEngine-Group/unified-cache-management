"""
The MIT License

Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch

if hasattr(torch, "npu") and torch.npu.is_available():
    import torch_npu

from ucm.logger import init_logger

logger = init_logger(__name__)

if hasattr(torch, "cuda") and torch.cuda.is_available():
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

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # sample dimension
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # hash_rbits dimension
        offs_k = tl.arange(0, BLOCK_K)  # input_dim dimension

        # Matrix multiplication
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            code = tl.load(
                code_ptr
                + offs_k[:, None] * stride_codek
                + offs_n[None, :] * stride_coden,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(x, code)
            offs_k += BLOCK_K

        # Binarize and pack
        bits = (acc > 0).to(tl.uint8)  # Binarize
        bits = tl.reshape(bits, (BLOCK_M, BLOCK_N // 8, 8))  # Reshape for packing

        # Load the packing weights (ensure it has the correct shape)
        pack_w = tl.load(pack_w_ptr + tl.arange(0, 8) * stride_pack_w)
        packed = tl.sum(bits * pack_w[None, None, :], axis=-1).to(tl.uint8)

        # Store results
        offs_n = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        hash_out_ptrs = (
            hash_out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        )
        tl.store(
            hash_out_ptrs,
            packed,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < (N // 8)),
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

    @triton.jit
    def _reshape_and_cache_khash_kernel(
        k_in_ptr,  # [T, H, W]
        slot_ptr,  # [T]
        k_cache_ptr,  # [B, BS, H, W]
        n_tokens: tl.constexpr,
        H: tl.constexpr,  # H 作为 constexpr 更快（见下方 wrapper 解释）
        W: tl.constexpr,  # W 作为 constexpr 更快
        # strides for k_in: [T, H, W]
        in_stride_t: tl.constexpr,
        in_stride_h: tl.constexpr,
        in_stride_w: tl.constexpr,
        # strides for k_cache: [B, BS, H, W]
        cache_stride_b: tl.constexpr,
        cache_stride_s: tl.constexpr,
        cache_stride_h: tl.constexpr,
        cache_stride_w: tl.constexpr,
        block_size: tl.constexpr,  # BS (e.g. 128)
        cache_num_slots: tl.constexpr,  # B*BS，用于上界检查
        BLOCK: tl.constexpr,  # 每个 program 处理的元素数（1D）
    ):
        pid_t = tl.program_id(0)  # token id
        pid_c = tl.program_id(1)  # chunk id

        if pid_t >= n_tokens:
            return

        # slot mapping
        slot = tl.load(slot_ptr + pid_t).to(tl.int64)
        if slot < 0:
            return
        # 上界检查：避免 slot_mapping 脏值写爆缓存
        if slot >= cache_num_slots:
            return

        b = slot // block_size
        s = slot - b * block_size

        # flatten HW 并按 chunk 拷贝
        n_elems = H * W
        offs = pid_c * BLOCK + tl.arange(0, BLOCK)  # [BLOCK]
        mask = offs < n_elems

        # 由 flatten offset -> (h,w)
        h = offs // W
        w = offs - h * W

        # load k_in[pid_t, h, w]
        in_ptrs = k_in_ptr + pid_t * in_stride_t + h * in_stride_h + w * in_stride_w
        x = tl.load(in_ptrs, mask=mask, other=0)

        # store k_cache[b, s, h, w]
        out_ptrs = (
            k_cache_ptr
            + b * cache_stride_b
            + s * cache_stride_s
            + h * cache_stride_h
            + w * cache_stride_w
        )
        tl.store(out_ptrs, x, mask=mask)

    def reshape_and_cache_khash_triton(
        k_hash_compute: torch.Tensor,  # [T, H, W]
        slot_mapping: torch.Tensor,  # [T]
        k_hash: torch.Tensor,  # [B, BS, H, W]
        block_size: int = 128,
    ):
        assert k_hash_compute.is_cuda and k_hash.is_cuda and slot_mapping.is_cuda
        assert k_hash_compute.ndim == 3, f"expect [T,H,W], got {k_hash_compute.shape}"
        assert k_hash.ndim == 4, f"expect [B,BS,H,W], got {k_hash.shape}"
        assert (
            slot_mapping.ndim == 1 and slot_mapping.shape[0] == k_hash_compute.shape[0]
        )
        assert (
            k_hash.shape[1] == block_size
        ), f"k_hash BS={k_hash.shape[1]} != block_size={block_size}"
        assert (
            k_hash_compute.shape[1:] == k_hash.shape[2:]
        ), f"shape mismatch: compute {k_hash_compute.shape[1:]} vs cache {k_hash.shape[2:]}"

        T, H, W = k_hash_compute.shape
        B = k_hash.shape[0]
        cache_num_slots = B * block_size

        # strides are in elements
        in_stride_t, in_stride_h, in_stride_w = k_hash_compute.stride()
        cache_stride_b, cache_stride_s, cache_stride_h, cache_stride_w = k_hash.stride()

        n_elems = H * W

        # 选一个 BLOCK（必须 constexpr），并用 chunk 维度覆盖 n_elems
        # 你也可以按性能调整这些档位
        if n_elems <= 256:
            BLOCK = 256
            num_warps = 4
        elif n_elems <= 512:
            BLOCK = 512
            num_warps = 8
        else:
            BLOCK = 1024
            num_warps = 8  # 1024 元素一般 8 warps 足够；更大可再调

        n_chunks = triton.cdiv(n_elems, BLOCK)

        grid = (T, n_chunks)

        _reshape_and_cache_khash_kernel[grid](
            k_hash_compute,
            slot_mapping,
            k_hash,
            n_tokens=T,
            H=H,
            W=W,
            in_stride_t=in_stride_t,
            in_stride_h=in_stride_h,
            in_stride_w=in_stride_w,
            cache_stride_b=cache_stride_b,
            cache_stride_s=cache_stride_s,
            cache_stride_h=cache_stride_h,
            cache_stride_w=cache_stride_w,
            block_size=block_size,
            cache_num_slots=cache_num_slots,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return k_hash

    @triton.jit
    def fused_hash_and_cache_kernel(
        x_ptr,
        code_ptr,
        pack_w_ptr,
        slot_ptr,
        k_cache_ptr,
        T,
        H,
        K: tl.constexpr,
        N_BITS: tl.constexpr,
        N_BYTES: tl.constexpr,
        stride_xt,
        stride_xh,
        stride_xk,
        stride_codek,
        stride_coden,
        stride_packw,
        stride_cb,
        stride_cs,
        stride_ch,
        stride_cw,
        block_size: tl.constexpr,
        cache_num_slots: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_h = tl.program_id(2)

        if pid_m * BLOCK_M >= T:
            return
        if pid_n * BLOCK_N >= N_BITS:
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        m_mask = offs_m < T
        n_mask = offs_n < N_BITS
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K

            x_ptrs = (
                x_ptr
                + offs_m[:, None] * stride_xt
                + pid_h * stride_xh
                + offs_k[None, :] * stride_xk
            )
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            code_ptrs = (
                code_ptr
                + offs_k[:, None] * stride_codek
                + offs_n[None, :] * stride_coden
            )
            code = tl.load(code_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(x, code)
            offs_k += BLOCK_K

        bits = (acc > 0).to(tl.uint8)
        bits_2d = tl.reshape(bits, (BLOCK_M, BLOCK_N // 8, 8))

        pack_w = tl.load(pack_w_ptr + tl.arange(0, 8) * stride_packw)
        packed = tl.sum(bits_2d * pack_w[None, None, :], axis=-1).to(tl.uint8)

        offs_byte = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        offs_byte = offs_byte[None, :]

        slot = tl.load(slot_ptr + offs_m, mask=m_mask, other=-1)

        valid_slot = (slot >= 0) & (slot < cache_num_slots)
        valid_row = m_mask & valid_slot
        valid_byte = offs_byte < N_BYTES
        out_mask = valid_row[:, None] & valid_byte
        safe_slot = tl.where(valid_row, slot, tl.zeros((BLOCK_M,), dtype=tl.int64))

        b = (safe_slot // block_size)[:, None]
        s = (safe_slot % block_size)[:, None]

        out_ptrs = (
            k_cache_ptr
            + b * stride_cb
            + s * stride_cs
            + pid_h * stride_ch
            + offs_byte * stride_cw
        )
        tl.store(out_ptrs, packed, mask=out_mask)

    @triton.jit
    def fused_hash_and_cache_mla_kernel(
        x_nope_ptr,
        x_rope_ptr,
        code_nope_ptr,
        code_rope_ptr,
        pack_w_ptr,
        slot_ptr,
        k_cache_ptr,
        T,
        K_nope: tl.constexpr,
        K_rope: tl.constexpr,
        N_BITS_NOPE: tl.constexpr,
        N_BITS_ROPE: tl.constexpr,
        nope_hash_bytes: tl.constexpr,
        stride_xt_nope,
        stride_xk_nope,
        stride_xt_rope,
        stride_xk_rope,
        stride_codek_nope,
        stride_coden_nope,
        stride_codek_rope,
        stride_coden_rope,
        stride_packw,
        stride_cb,
        stride_cs,
        stride_cw,
        block_size: tl.constexpr,
        cache_num_slots: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        if pid_m * BLOCK_M >= T:
            return

        blocks_for_nope = N_BITS_NOPE // BLOCK_N
        is_nope = pid_n < blocks_for_nope

        if is_nope:
            K = K_nope
            N_BITS = N_BITS_NOPE
            x_ptr = x_nope_ptr
            code_ptr = code_nope_ptr
            stride_xt = stride_xt_nope
            stride_xk = stride_xk_nope
            stride_codek = stride_codek_nope
            stride_coden = stride_coden_nope
            byte_offset_base = 0
            rel_pid_n = pid_n
        else:
            K = K_rope
            N_BITS = N_BITS_ROPE
            x_ptr = x_rope_ptr
            code_ptr = code_rope_ptr
            stride_xt = stride_xt_rope
            stride_xk = stride_xk_rope
            stride_codek = stride_codek_rope
            stride_coden = stride_coden_rope
            byte_offset_base = nope_hash_bytes
            rel_pid_n = pid_n - blocks_for_nope

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = rel_pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        m_mask = offs_m < T
        n_mask = offs_n < N_BITS
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K
            x_ptrs = x_ptr + offs_m[:, None] * stride_xt + offs_k[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            code_ptrs = (
                code_ptr
                + offs_k[:, None] * stride_codek
                + offs_n[None, :] * stride_coden
            )
            code = tl.load(code_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(x, code)
            offs_k += BLOCK_K

        bits = (acc > 0).to(tl.uint8)
        bits_2d = tl.reshape(bits, (BLOCK_M, BLOCK_N // 8, 8))
        pack_w = tl.load(pack_w_ptr + tl.arange(0, 8) * stride_packw)
        packed = tl.sum(bits_2d * pack_w[None, None, :], axis=-1).to(tl.uint8)

        offs_byte = rel_pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        slot = tl.load(slot_ptr + offs_m, mask=m_mask, other=-1)

        valid_slot = (slot >= 0) & (slot < cache_num_slots)
        valid_row = m_mask & valid_slot
        valid_byte = offs_byte[None, :] < (N_BITS // 8)
        out_mask = valid_row[:, None] & valid_byte

        safe_slot = tl.where(valid_row, slot, tl.zeros((BLOCK_M,), dtype=tl.int64))
        b = (safe_slot // block_size)[:, None]
        s = (safe_slot % block_size)[:, None]

        out_ptrs = (
            k_cache_ptr
            + b * stride_cb
            + s * stride_cs
            + (byte_offset_base + offs_byte[None, :]) * stride_cw
        )
        tl.store(out_ptrs, packed, mask=out_mask)


@torch.compile()
def torch_hash_code(x, code, pack_weight):
    # [N, hash_bits]
    x = x @ code
    m = x.shape[:-1]
    # [N, hash_bits] -- > [N, hash_bits // 8, 8]
    x = (x > 0).to(torch.uint8).view(*m, -1, 8)
    # 8bit -> 1bit
    # binary_codes * self.bit_masks [N, hash_numbers, 8] * [1, 1, 8] -> [N, hash_numbers, 8]
    # then sum along the last dimension to get [N, hash_numbers]
    x = torch.sum(x * pack_weight, dim=-1, dtype=torch.uint8)
    return x


class HashEncoder:
    """
    HashEncoder converts a float tensor to a binary hash code tensor,
    and it packs every 8 bits into a uint8 number.
    """

    def __init__(
        self,
        input_dim: int,
        hash_bits: int,
        dtype: torch.dtype,
        device: torch.device,
        input_dim_rope: int = 0,
        hash_bits_rope: int = 0,
        is_mla: bool = False,
    ) -> None:
        self.input_dim = input_dim

        if hash_bits % 8 != 0:
            raise ValueError("hash_bits must be a multiple of 8")

        self.hash_bits = hash_bits

        # number of uint8 numbers to store hash_bits bits
        self.hash_numbers = self.hash_bits // 8

        self.dtype = dtype
        self.device = device
        self.is_mla = is_mla

        if self.device.type == "npu":
            if dtype not in [torch.float16, torch.float32, torch.float64]:
                logger.warning(
                    "NPU only supports float16, float32 and float64 for hash_weights"
                )
                logger.warning("automatically using  float16 for hash_weights now")
                self.dtype = torch.float16

        # if self.device.type == "cuda" and dtype == torch.bfloat16:
        #     logger.warning("geqrf_cuda not implemented for BFloat16")
        #     logger.warning("automatically using  float32 for hash_weights now")
        #     self.dtype = torch.float32

        self.hash_weights = self._init_hash_weights(self.input_dim, self.hash_bits)

        if self.device.type == "cuda" or self.device.type == "cpu":
            self._init_bit_masks()

        if self.is_mla:
            self.input_dim_rope = input_dim_rope
            self.hash_bits_rope = hash_bits_rope
            if hash_bits_rope % 8 != 0:
                raise ValueError("hash_bits_rope must be a multiple of 8")
            self.hash_numbers_rope = hash_bits_rope // 8
            self.hash_weights_rope = self._init_hash_weights(
                self.input_dim_rope, self.hash_bits_rope
            )
        else:
            self.hash_numbers_rope = 0
        self.hash_numbers_total = self.hash_numbers + self.hash_numbers_rope

    def _init_hash_weights(self, input_dim, hash_bits):
        # Step 1: 随机高斯矩阵
        random_weights = torch.normal(
            mean=0,
            std=2,
            size=(input_dim, hash_bits),
            dtype=(
                torch.float32
                if (self.device.type == "cuda" and self.dtype == torch.bfloat16)
                else self.dtype
            ),
            device=self.device,
        )
        # Step 2: QR分解
        Q, R = torch.linalg.qr(random_weights)

        # Step 3: 调整符号，保证Haar 分布
        d = torch.sign(torch.diag(R))
        hash_weights = Q * d
        if self.device.type == "cuda" and self.dtype == torch.bfloat16:
            hash_weights = hash_weights.to(self.dtype)
        return hash_weights

    def set_hash_weight(self, hash_weights: torch.Tensor) -> None:
        if hash_weights.shape != (self.input_dim, self.hash_bits):
            raise ValueError(
                f"hash_weights shape {hash_weights.shape} does not match required shape {(self.input_dim, self.hash_bits)}"
            )
        if hash_weights.dtype != self.dtype:
            raise ValueError(
                f"hash_weights dtype {hash_weights.dtype} does not match required dtype {self.dtype}"
            )
        if hash_weights.device != self.device:
            raise ValueError(
                f"hash_weights device {hash_weights.device} does not match required device {self.device}"
            )

        self.hash_weights.copy_(hash_weights)

    def _init_bit_masks(self) -> None:
        self.bit_masks = torch.pow(
            2, torch.arange(8, dtype=torch.uint8, device=self.device)
        )

    def compute_hash(self, x: torch.Tensor, is_rope: bool = False) -> torch.Tensor:
        """
        Compute the hash code for input tensor x.
        Args:
            x: input tensor of shape (..., input_dim)
        Returns:
            A tensor of shape (..., hash_numbers=hash_bits // 8) representing the hash codes.
            Each element is a uint8 number representing 8 bits of the hash code.
        """
        input_dim = self.input_dim_rope if is_rope else self.input_dim
        hash_weights = self.hash_weights_rope if is_rope else self.hash_weights
        hash_numbers = self.hash_numbers_rope if is_rope else self.hash_numbers

        if x.shape[-1] != input_dim:
            raise ValueError(
                f"x must be of shape (..., {input_dim}), but got {x.shape}"
            )
        if x.device != self.device:
            raise ValueError(
                f"x device {x.device} does not match required device {self.device}"
            )

        # original shape without the last dimension
        # e.g. x.shape=[s1,s2,s3,input_dim], orig_shape=[s1,s2,s3]
        orig_shape = x.shape[:-1]

        # [N, input_dim], e.g., N = s1*s2*s3
        x_flat = x.reshape(-1, input_dim)

        if x_flat.dtype != self.dtype:
            x_flat = x_flat.to(self.dtype)

        if self.device.type == "npu":
            # [N, hash_bits]
            xW = torch.matmul(x_flat, hash_weights)
            # [N * hash_bits]
            xW_flat = xW.view(-1)
            # [N*hash_numbers], where hash_numbers = hash_bits // 8
            packed_codes_flat = torch_npu.npu_sign_bits_pack(xW_flat, size=1)

        elif self.device.type == "cuda":
            packed_codes_flat = triton_hash_code(
                x_flat, hash_weights, self.bit_masks
            ).view(
                -1
            )  # [N * hash_numbers]

        elif self.device.type == "cpu":
            packed_codes_flat = torch_hash_code(
                x_flat, hash_weights, self.bit_masks
            ).view(
                -1
            )  # [N * hash_numbers]

        else:
            raise ValueError(f"Unsupported device type: {self.device.type}")

        # e.g., [s1, s2, s3, hash_numbers]
        out_shape = orig_shape + (hash_numbers,)
        packed_codes = packed_codes_flat.view(out_shape)

        return packed_codes

    def _reinterpret_cache_as_u8(self, cache: torch.Tensor) -> torch.Tensor:
        """
        支持:
        - uint8 cache: [B, BS, H, hash_numbers]
        - bf16  cache: [B, BS, H, hash_numbers // 2]
            内部 view 成 uint8 后变成 [B, BS, H, hash_numbers]
        """
        if cache.dtype == torch.uint8:
            if cache.shape[-1] != self.hash_numbers_total:
                raise ValueError(
                    f"uint8 cache last dim mismatch: got {cache.shape[-1]}, "
                    f"expected {self.hash_numbers_total}"
                )
            return cache

        if cache.dtype == torch.bfloat16:
            if self.hash_numbers_total % 2 != 0:
                raise ValueError(
                    f"for bfloat16 cache, hash_numbers must be even, got {self.hash_numbers_total}"
                )
            if cache.shape[-1] != self.hash_numbers_total // 2:
                raise ValueError(
                    f"bfloat16 cache last dim mismatch: got {cache.shape[-1]}, "
                    f"expected {self.hash_numbers_total // 2}"
                )

            cache_u8 = cache.view(torch.uint8)

            if cache_u8.shape[-1] != self.hash_numbers_total:
                raise ValueError(
                    f"reinterpret bf16->u8 failed: got last dim {cache_u8.shape[-1]}, "
                    f"expected {self.hash_numbers_total}"
                )
            return cache_u8

        raise ValueError(
            f"unsupported k_hash_cache dtype {cache.dtype}, only uint8 / bfloat16 are supported"
        )

    def compute_hash_and_cache(
        self,
        x: torch.Tensor,  # [T, H, K]
        slot_mapping: torch.Tensor,  # [T]
        k_hash_cache: torch.Tensor,  # [B, BS, H, ?], uint8 or bf16
        block_size: int = 128,
        BLOCK_M: int = 64,
        BLOCK_K: int = 16,
        BLOCK_N: int = 32,
        num_warps: int = 4,
    ) -> torch.Tensor:
        """
        融合:
            compute_hash(x) + cache write
        只发起一次 Triton kernel。

        支持:
        - uint8 cache: [B, BS, H, hash_numbers]
        - bf16  cache: [B, BS, H, hash_numbers // 2]
            内部 reinterpret 为 uint8 后写 packed hash
        """
        if self.device.type != "cuda":
            raise NotImplementedError(
                "compute_hash_and_cache currently only supports CUDA"
            )

        if x.ndim != 3:
            raise ValueError(f"x must be [T,H,K], got {x.shape}")
        if slot_mapping.ndim != 1:
            raise ValueError(f"slot_mapping must be [T], got {slot_mapping.shape}")
        if k_hash_cache.ndim != 4:
            raise ValueError(f"k_hash_cache must be rank-4, got {k_hash_cache.shape}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"x last dim must be {self.input_dim}, got {x.shape[-1]}")
        if x.device != self.device:
            raise ValueError(
                f"x device {x.device} does not match required device {self.device}"
            )
        if slot_mapping.device != self.device:
            raise ValueError(
                f"slot_mapping device {slot_mapping.device} does not match required device {self.device}"
            )
        if k_hash_cache.device != self.device:
            raise ValueError(
                f"k_hash_cache device {k_hash_cache.device} does not match required device {self.device}"
            )
        if BLOCK_N % 8 != 0:
            raise ValueError(f"BLOCK_N must be multiple of 8, got {BLOCK_N}")

        T, H, K = x.shape
        B, BS, H_cache, _ = k_hash_cache.shape

        if slot_mapping.shape[0] != T:
            raise ValueError(f"slot_mapping length {slot_mapping.shape[0]} != T {T}")
        if BS != block_size:
            raise ValueError(f"k_hash_cache.shape[1]={BS} != block_size={block_size}")
        if H_cache != H:
            raise ValueError(f"k_hash_cache.shape[2]={H_cache} != H={H}")

        # 关键修改：内部统一转成 uint8 view
        k_hash_cache_u8 = self._reinterpret_cache_as_u8(k_hash_cache)

        # 这里的最后一维现在一定是 packed hash bytes
        B2, BS2, H2, W2 = k_hash_cache_u8.shape
        if B2 != B or BS2 != BS or H2 != H:
            raise ValueError(
                f"reinterpret cache shape mismatch: raw={k_hash_cache.shape}, u8={k_hash_cache_u8.shape}"
            )
        if W2 != self.hash_numbers:
            raise ValueError(
                f"u8 cache last dim {W2} != hash_numbers {self.hash_numbers}"
            )

        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        cache_num_slots = B * block_size

        stride_xt, stride_xh, stride_xk = x.stride()
        stride_codek, stride_coden = self.hash_weights.stride()
        (stride_packw,) = self.bit_masks.stride()
        stride_cb, stride_cs, stride_ch, stride_cw = k_hash_cache_u8.stride()

        grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(self.hash_bits, BLOCK_N), H)

        fused_hash_and_cache_kernel[grid](
            x_ptr=x,
            code_ptr=self.hash_weights,
            pack_w_ptr=self.bit_masks,
            slot_ptr=slot_mapping,
            k_cache_ptr=k_hash_cache_u8,
            T=T,
            H=H,
            K=K,
            N_BITS=self.hash_bits,
            N_BYTES=self.hash_numbers,
            stride_xt=stride_xt,
            stride_xh=stride_xh,
            stride_xk=stride_xk,
            stride_codek=stride_codek,
            stride_coden=stride_coden,
            stride_packw=stride_packw,
            stride_cb=stride_cb,
            stride_cs=stride_cs,
            stride_ch=stride_ch,
            stride_cw=stride_cw,
            block_size=block_size,
            cache_num_slots=cache_num_slots,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

        return k_hash_cache

    def compute_hash_and_cache_mla(
        self,
        x: torch.Tensor,  # [T, 1, K_nope]
        x_rope: torch.Tensor,  # [T, 1, K_rope]
        slot_mapping: torch.Tensor,  # [T]
        k_hash_cache: torch.Tensor,  # [B, BS, 1, N], uint8 or bf16
        block_size: int = 128,
        BLOCK_M: int = 64,
        BLOCK_K: int = 64,
        BLOCK_N: int = 64,
        num_warps: int = 4,
    ):
        T, K = x.shape
        _, K_rope = x_rope.shape
        B, _, _ = k_hash_cache.shape

        k_hash_cache_u8 = self._reinterpret_cache_as_u8(k_hash_cache)

        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        if x_rope.dtype != self.dtype:
            x_rope = x_rope.to(self.dtype)

        cache_num_slots = B * block_size

        stride_xt, stride_xk = x.stride()
        stride_xt_rope, stride_xk_rope = x_rope.stride()
        stride_codek, stride_coden = self.hash_weights.stride()
        stride_codek_rope, stride_coden_rope = self.hash_weights_rope.stride()
        (stride_packw,) = self.bit_masks.stride()
        stride_cb, stride_cs, stride_cw = k_hash_cache_u8.stride()

        total_hash_bits = self.hash_bits + self.hash_bits_rope
        grid = (triton.cdiv(T, BLOCK_M), triton.cdiv(total_hash_bits, BLOCK_N))

        fused_hash_and_cache_mla_kernel[grid](
            x_nope_ptr=x,
            x_rope_ptr=x_rope,
            code_nope_ptr=self.hash_weights,
            code_rope_ptr=self.hash_weights_rope,
            pack_w_ptr=self.bit_masks,
            slot_ptr=slot_mapping,
            k_cache_ptr=k_hash_cache_u8,
            T=T,
            K_nope=K,
            K_rope=K_rope,
            N_BITS_NOPE=self.hash_bits,
            N_BITS_ROPE=self.hash_bits_rope,
            nope_hash_bytes=self.hash_numbers,
            stride_xt_nope=stride_xt,
            stride_xk_nope=stride_xk,
            stride_xt_rope=stride_xt_rope,
            stride_xk_rope=stride_xk_rope,
            stride_codek_nope=stride_codek,
            stride_coden_nope=stride_coden,
            stride_codek_rope=stride_codek_rope,
            stride_coden_rope=stride_coden_rope,
            stride_packw=stride_packw,
            stride_cb=stride_cb,
            stride_cs=stride_cs,
            stride_cw=stride_cw,
            block_size=block_size,
            cache_num_slots=cache_num_slots,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )

        return k_hash_cache

    def _unpack_hash(self, packed_codes: torch.Tensor) -> torch.Tensor:
        """
        Unpack the hash codes to +1 or -1 bits.
        Args:
            packed_codes: input tensor of shape (..., hash_numbers), dtype=torch.uint8
        Returns:
            A tensor of shape (..., hash_bits=hash_numbers*8) representing the unpacked bits.
            Each element is either -1 or 1.
        """
        if packed_codes.shape[-1] != self.hash_numbers:
            raise ValueError(
                f"packed_codes must be of shape (..., {self.hash_numbers}), but got {packed_codes.shape}"
            )
        if packed_codes.device != self.device:
            raise ValueError(
                f"packed_codes device {packed_codes.device} does not match required device {self.device}"
            )
        if packed_codes.dtype != torch.uint8:
            raise ValueError(
                f"packed_codes dtype {packed_codes.dtype} is not torch.uint8"
            )

        # e.g., packed_codes.shape=[s1, s2, s3, hash_numbers]
        # orig_shape = [s1, s2, s3]
        orig_shape = packed_codes.shape[:-1]

        # [N * hash_numbers], e.g., N = s1*s2*s3
        packed_codes_flat = packed_codes.view(-1)

        if self.device.type == "npu":
            # [N * hash_bits]
            unpacked_bits_flat = torch_npu.npu_sign_bits_unpack(
                packed_codes_flat, size=1, dtype=torch.float16
            )
        elif self.device.type == "cuda" or self.device.type == "cpu":
            # (TODO) improve performance later on CUDA ops and CPU SIMD instructions
            # [N, hash_numbers]
            packed_codes_2d = packed_codes_flat.view(-1, self.hash_numbers)

            # [N, hash_numbers, 8]
            expanded = packed_codes_2d.unsqueeze(-1).expand(
                -1, -1, 8
            )  # expand last dim to 8

            # (expanded & self.bit_masks) > 0 -> [N, hash_numbers, 8]
            unpacked_bits = (expanded & self.bit_masks.unsqueeze(0).unsqueeze(0)) > 0

            # 0 -> -1, 1 -> 1
            unpacked_bits = unpacked_bits * 2 - 1

            unpacked_bits = unpacked_bits.to(torch.float16)

            # [N, hash_bits]
            unpacked_bits_flat = unpacked_bits.view(-1, self.hash_bits)
        else:
            raise ValueError(f"Unsupported device type: {self.device.type}")

        out_shape = orig_shape + (self.hash_bits,)
        unpacked_bits = unpacked_bits_flat.view(out_shape)

        return unpacked_bits


if __name__ == "__main__":
    torch.manual_seed(42)

    print("test HashEncoder...")
    dtype = torch.float16
    if hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu:0")
    elif hasattr(torch, "cuda") and torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float32
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    encoder = HashEncoder(input_dim=8, hash_bits=8, dtype=dtype, device=device)

    x = torch.randn(2, 8, device=device, dtype=dtype)
    print("x:", x)

    hash_codes = encoder.compute_hash(x)
    print("hash_codes:", hash_codes)
    print("hash_codes shape:", hash_codes.shape)

    unpacked_bits = encoder._unpack_hash(hash_codes)
    print("unpacked_bits:", unpacked_bits)
    print("unpacked_bits shape:", unpacked_bits.shape)

    print(
        f"hash_codes[0].item()={hash_codes[0].item()}, 8-bit binary form:{hash_codes[0].item():08b}"
    )
    print(
        f"hash_codes[1].item()={hash_codes[1].item()}, 8-bit binary form:{hash_codes[1].item():08b}"
    )

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        print("test cuda triton and torch hash code functions...")
        x = torch.randn((1024, 512), device="cuda:0", dtype=torch.bfloat16)
        code = torch.randn((512, 512), device="cuda:0", dtype=torch.bfloat16)
        pack_weight = torch.tensor(
            [128, 64, 32, 16, 8, 4, 2, 1], device="cuda:0", dtype=torch.uint8
        )

        torch_output = torch_hash_code(x, code, pack_weight)
        triton_output = triton_hash_code(x, code, pack_weight)
        assert torch_output.shape == triton_output.shape
        print(f"x_shape: {x.shape}  code_shape: {code.shape}")
        print("torch_output", torch_output)
        print("triton_output", triton_output)
        print(
            f"The maximum difference between Torch and Triton is"
            f" {torch.max(torch.abs(torch_output.to(torch.int32) - triton_output.to(torch.int32)))}"
        )
        # benchmark
        print(
            "torch:",
            triton.testing.do_bench(lambda: torch_hash_code(x, code, pack_weight)),
        )
        print(
            "triton:",
            triton.testing.do_bench(lambda: triton_hash_code(x, code, pack_weight)),
        )
