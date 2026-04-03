import pytest
import torch
from vllm import _custom_ops as ops

from ucm.sparse.gsa_on_device.csrc.cuda.hash_and_cache import fused_mla_hash
from ucm.sparse.gsa_on_device.hash_encoder import (
    HashEncoder,
    reshape_and_cache_khash_triton,
)

torch.manual_seed(42)

warmup_iters = 5
test_iters = 20

num_tokens = 128 * 300  # T
num_heads = 1  # H, MLA直接不构造该维度
head_dim = 512  # K (input_dim)
head_dim_rope = 64  # K_rope (input_dim_rope)
hash_bits = 512  # N (hash_bits)
hash_bits_rope = 64  # N_rope
hash_numbers = hash_bits // 8  # W (hash_numbers)
hash_numbers_rope = hash_bits_rope // 8  # W (hash_numbers)
block_size = 128  # BS
num_blocks = 300  # B


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestCudaHashAndCacheMLA:
    def get_input_data(self):
        device = torch.device("cuda:6")
        torch.cuda.set_device(6)
        dtype = torch.bfloat16

        assert (
            num_tokens <= num_blocks * block_size
        ), "num_blocks is not large enough to contain all tokens."

        # 初始化 HashEncoder
        encoder = HashEncoder(
            input_dim=head_dim,
            hash_bits=hash_bits,
            dtype=dtype,
            device=device,
            input_dim_rope=head_dim_rope,
            hash_bits_rope=hash_bits_rope,
            is_mla=True,
        )

        # key: [T, H, K]
        key = torch.randn((num_tokens, head_dim), device=device, dtype=dtype)
        key_rope = torch.randn((num_tokens, head_dim_rope), device=device, dtype=dtype)

        # slot_mapping: [T], 随机映射到 cache 中的位置
        slot_mapping = torch.randperm(num_blocks * block_size)[:num_tokens].to(
            device, dtype=torch.int64
        )

        # 初始化两个相同的 cache 用于对比
        # k_hash_cache: [B, BS, H, W] 其中 W = hash_bits // 8
        # bf16格式，相比u8的维度减半
        cache_fused = torch.zeros(
            (num_blocks, block_size, (hash_numbers + hash_numbers_rope) // 2),
            device=device,
            dtype=torch.bfloat16,
        )
        cache_ref = torch.zeros_like(cache_fused)

        k_scale = torch.tensor(1.0, dtype=torch.float32)
        return (
            encoder,
            key,
            key_rope,
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            hash_numbers,
            hash_numbers_rope,
            block_size,
            k_scale,
        )

    def test_cuda_hash_and_cache_gqa_accuracy(self):

        (
            encoder,
            key,
            key_rope,
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            hash_numbers,
            hash_numbers_rope,
            block_size,
            k_scale,
        ) = self.get_input_data()

        # 融合算子
        encoder.compute_hash_and_cache_mla(
            key, key_rope, slot_mapping, cache_fused, block_size=block_size
        )
        # 基准计算
        # 1. 计算 Hash Code [T, H, W]
        k_hash_computed = encoder.compute_hash(key).view(torch.bfloat16)
        k_rope_hash_computed = encoder.compute_hash(key_rope, is_rope=True).view(
            torch.bfloat16
        )
        # 2. 写入 Cache
        ops.concat_and_cache_mla(
            k_hash_computed,
            k_rope_hash_computed,
            cache_ref,
            slot_mapping.flatten(),
            kv_cache_dtype="auto",
            scale=k_scale,
        )
        torch.save(
            cache_ref.to("cpu"), "/home/externals/wangwenxin21/fl/data/cache_triton.npy"
        )

        cache_ref.zero_()
        torch.cuda.synchronize()
        block_size = 128
        fused_mla_hash.fused_hash_and_cache_mla(
            key,
            key_rope,
            encoder.hash_weights,
            encoder.hash_weights_rope,
            encoder.bit_masks,
            slot_mapping.flatten(),
            cache_ref.view(torch.uint8),
            block_size,
        )
        torch.save(
            cache_ref.to("cpu"), "/home/externals/wangwenxin21/fl/data/cache_cuda.npy"
        )
        # 验证融合算子的结果与分步计算的结果是否一致
        diff = torch.abs(cache_fused.view(torch.uint8) - cache_ref.view(torch.uint8))
        print(
            f"\nBit flip rate: {diff.nonzero().shape[0]}/{diff.numel()} = {diff.nonzero().shape[0] / diff.numel():.4f}"
        )
        assert (
            diff.nonzero().shape[0] / diff.numel() < 0.01
        ), "More than 1% of the elements differ between fused and reference results."

    def test_cuda_hash_and_cache_gqa_baseline(self):
        (
            encoder,
            key,
            key_rope,
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            hash_numbers,
            hash_numbers_rope,
            block_size,
            k_scale,
        ) = self.get_input_data()

        # 原版：分步计算
        # 预热
        for _ in range(warmup_iters):
            k_hash_computed = encoder.compute_hash(key).view(torch.bfloat16)
            k_rope_hash_computed = encoder.compute_hash(key_rope, is_rope=True).view(
                torch.bfloat16
            )
            # 2. 写入 Cache
            ops.concat_and_cache_mla(
                k_hash_computed,
                k_rope_hash_computed,
                cache_ref,
                slot_mapping.flatten(),
                kv_cache_dtype="auto",
                scale=k_scale,
            )
        torch.cuda.synchronize()

        # 性能测试
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        total_time = 0
        torch.cuda.synchronize()
        start_time.record()
        for _ in range(test_iters):
            k_hash_computed = encoder.compute_hash(key).view(torch.bfloat16)
            k_rope_hash_computed = encoder.compute_hash(key_rope, is_rope=True).view(
                torch.bfloat16
            )
            # 2. 写入 Cache
            ops.concat_and_cache_mla(
                k_hash_computed,
                k_rope_hash_computed,
                cache_ref,
                slot_mapping.flatten(),
                kv_cache_dtype="auto",
                scale=k_scale,
            )
        end_time.record()
        torch.cuda.synchronize()
        total_time += start_time.elapsed_time(end_time)
        avg_time_ms_ref = total_time / test_iters
        print(f"\nAverage time per iteration (Unfused): {avg_time_ms_ref:.2f} ms")

    def test_cuda_hash_and_cache_mla_tritonFused_performance(self):

        (
            encoder,
            key,
            key_rope,
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            hash_numbers,
            hash_numbers_rope,
            block_size,
            k_scale,
        ) = self.get_input_data()

        # 融合算子
        # 预热
        for _ in range(warmup_iters):
            encoder.compute_hash_and_cache_mla(
                key, key_rope, slot_mapping, cache_fused, block_size=block_size
            )
        torch.cuda.synchronize()

        # 性能测试
        total_time = 0
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        for _ in range(test_iters):
            encoder.compute_hash_and_cache_mla(
                key, key_rope, slot_mapping, cache_fused, block_size=block_size
            )
        end_time.record()
        torch.cuda.synchronize()
        total_time += start_time.elapsed_time(end_time)
        avg_time_ms = total_time / test_iters
        print(f"\nAverage time per iteration (Triton fused): {avg_time_ms:.2f} ms")

    def test_cuda_hash_and_cache_mla_cudaFused_performance(self):

        (
            encoder,
            key,
            key_rope,
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            hash_numbers,
            hash_numbers_rope,
            block_size,
            k_scale,
        ) = self.get_input_data()

        # 融合算子
        # 预热
        for _ in range(warmup_iters):
            fused_mla_hash.fused_hash_and_cache_mla(
                key,
                key_rope,
                encoder.hash_weights,
                encoder.hash_weights_rope,
                encoder.bit_masks,
                slot_mapping.flatten(),
                cache_ref.view(torch.uint8),
                block_size,
            )
        torch.cuda.synchronize()

        # 性能测试
        total_time = 0
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        for _ in range(test_iters):
            fused_mla_hash.fused_hash_and_cache_mla(
                key,
                key_rope,
                encoder.hash_weights,
                encoder.hash_weights_rope,
                encoder.bit_masks,
                slot_mapping.flatten(),
                cache_ref.view(torch.uint8),
                block_size,
            )
        end_time.record()
        torch.cuda.synchronize()
        total_time += start_time.elapsed_time(end_time)
        avg_time_ms = total_time / test_iters
        print(f"\nAverage time per iteration (Cuda fused): {avg_time_ms:.2f} ms")


if __name__ == "__main__":
    pytest.main([__file__])
