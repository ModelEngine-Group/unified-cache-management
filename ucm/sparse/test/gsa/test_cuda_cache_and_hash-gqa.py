import pytest
import torch

from ucm.sparse.gsa_on_device.hash_encoder import (
    HashEncoder,
    reshape_and_cache_khash_triton,
)

torch.manual_seed(42)

warmup_iters = 5
test_iters = 20

num_tokens = 1  # T
num_heads = 8  # H
head_dim = 128  # K (input_dim)
hash_bits = (
    128  # N (hash_bits) 压缩后的维度，单位是 bit，所以实际存储时需要除以 8 转换为 byte
)
hash_numbers = hash_bits // 8  # W (hash_numbers) 每个 token
block_size = 128  # BS
num_blocks = 300  # B


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestCudaHashAndCacheGQA:
    def get_input_data(self):
        device = torch.device("cuda:0")
        dtype = torch.bfloat16

        assert (
            num_tokens <= num_blocks * block_size
        ), "num_tokens must be less than or equal to num_blocks * block_size for this test."

        # 初始化 HashEncoder
        encoder = HashEncoder(
            input_dim=head_dim, hash_bits=hash_bits, dtype=dtype, device=device
        )

        # key: [T, H, K]
        key = torch.randn((num_tokens, num_heads, head_dim), device=device, dtype=dtype)

        # slot_mapping: [T], 随机映射到 cache 中的位置
        slot_mapping = torch.randperm(num_blocks * block_size)[:num_tokens].to(
            device, dtype=torch.int32
        )
        # slot_mapping = torch.arange(num_tokens).to(device, dtype=torch.int32)

        # 初始化两个相同的 cache 用于对比
        # k_hash_cache: [B, BS, H, W] 其中 W = hash_bits // 8
        cache_fused = torch.zeros(
            (num_blocks, block_size, num_heads, hash_numbers),
            device=device,
            dtype=torch.uint8,
        )
        cache_ref = torch.zeros_like(cache_fused)
        return (
            encoder,
            key,
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            num_heads,
            hash_numbers,
            block_size,
        )

    def test_cuda_hash_and_cache_gqa_accuracy(self):

        (
            encoder,
            key,
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            num_heads,
            hash_numbers,
            block_size,
        ) = self.get_input_data()

        # 融合算子
        encoder.compute_hash_and_cache(
            key, slot_mapping, cache_fused, block_size=block_size
        )

        # 基准计算
        # 1. 计算 Hash Code [T, H, W]
        k_hash_computed = encoder.compute_hash(key)

        # 2. 写入 Cache
        reshape_and_cache_khash_triton(
            k_hash_computed.view(num_tokens, num_heads, hash_numbers),
            slot_mapping,
            cache_ref,
            block_size=block_size,
        )

        # 验证融合算子的结果与分步计算的结果是否一致
        diff = torch.abs(cache_fused.to(torch.float32) - cache_ref.to(torch.float32))
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
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            num_heads,
            hash_numbers,
            block_size,
        ) = self.get_input_data()

        # 原版：分步计算
        # 预热
        for _ in range(warmup_iters):
            k_hash_computed = encoder.compute_hash(key)
            reshape_and_cache_khash_triton(
                k_hash_computed.view(num_tokens, num_heads, hash_numbers),
                slot_mapping,
                cache_ref,
                block_size=block_size,
            )
        torch.cuda.synchronize()

        # 性能测试
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        total_time = 0
        torch.cuda.synchronize()
        start_time.record()
        for _ in range(test_iters):
            k_hash_computed = encoder.compute_hash(key)
            reshape_and_cache_khash_triton(
                k_hash_computed.view(num_tokens, num_heads, hash_numbers),
                slot_mapping,
                cache_ref,
                block_size=block_size,
            )
        end_time.record()
        torch.cuda.synchronize()
        total_time += start_time.elapsed_time(end_time)
        avg_time_ms_ref = total_time / test_iters
        print(f"\nAverage time per iteration (unfused): {avg_time_ms_ref:.2f} ms")

    def test_cuda_hash_and_cache_gqa_performance(self):

        (
            encoder,
            key,
            slot_mapping,
            cache_fused,
            cache_ref,
            num_tokens,
            num_heads,
            hash_numbers,
            block_size,
        ) = self.get_input_data()

        # 融合算子
        # 预热
        for _ in range(warmup_iters):
            encoder.compute_hash_and_cache(
                key, slot_mapping, cache_fused, block_size=block_size
            )
        torch.cuda.synchronize()

        # 性能测试
        total_time = 0
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        for _ in range(test_iters):
            encoder.compute_hash_and_cache(
                key, slot_mapping, cache_fused, block_size=block_size
            )
        end_time.record()
        torch.cuda.synchronize()
        total_time += start_time.elapsed_time(end_time)
        avg_time_ms = total_time / test_iters
        print(f"\nAverage time per iteration (fused): {avg_time_ms:.2f} ms")


if __name__ == "__main__":
    pytest.main([__file__])
