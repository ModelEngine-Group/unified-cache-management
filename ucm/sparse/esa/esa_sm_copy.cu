#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_TRANS_UNIT_SIZE (sizeof(uint4) * 2)

constexpr __host__ __device__
int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

inline __device__ void CudaCopyUnit(const uint8_t* __restrict__ src,
                                    volatile uint8_t* __restrict__ dst)
{
    uint4 lo, hi;
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w)
                 : "l"(src));
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4+16];"
                 : "=r"(hi.x), "=r"(hi.y), "=r"(hi.z), "=r"(hi.w)
                 : "l"(src));
    asm volatile("st.volatile.global.v4.b32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(dst), "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w));
    asm volatile("st.volatile.global.v4.b32 [%0+16], {%1,%2,%3,%4};"
                 :
                 : "l"(dst), "r"(hi.x), "r"(hi.y), "r"(hi.z), "r"(hi.w));
}

__global__ void CudaCopyKernel(const void** src, void** dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        auto host = ((const uint8_t*)src[idx]) + off;
        auto device = ((uint8_t*)dst[idx]) + off;
        CudaCopyUnit(host, device);
        offset += gridDim.x * blockDim.x * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void CudaCopyKernel(const void** src, void* dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        auto host = ((const uint8_t*)src[idx]) + off;
        auto device = ((uint8_t*)dst) + offset;
        CudaCopyUnit(host, device);
        offset += gridDim.x * blockDim.x * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void CudaCopyKernel(const void* src, void** dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        auto host = ((const uint8_t*)src) + offset;
        auto device = ((uint8_t*)dst[idx]) + off;
        CudaCopyUnit(host, device);
        offset += gridDim.x * blockDim.x * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void CudaCopyKernel(const void* src, void* dst, size_t size)
{
    int num_repeats = (size + gridDim.x * blockDim.x *CUDA_TRANS_UNIT_SIZE - 1) / (gridDim.x * blockDim.x * CUDA_TRANS_UNIT_SIZE);
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i = 0; i < num_repeats; ++i){
        int offset = (i * gridDim.x * blockDim.x + tid) * CUDA_TRANS_UNIT_SIZE;
        auto host = ((const uint8_t*)src) + offset;
        auto device = ((uint8_t*)dst) + offset;
        CudaCopyUnit(host, device);
    }
}


extern "C" void esa_copy(torch::Tensor src, torch::Tensor dst, size_t size)
{
    dim3 numThreads = {1024};
    int totalThreads = ceildiv(size, CUDA_TRANS_UNIT_SIZE);
    dim3 numBlocks = {static_cast<unsigned int>(ceildiv(totalThreads, numThreads.x))};
    CudaCopyKernel<<<numBlocks, numThreads>>>(
        (const void*)src.data_ptr(), (void*)dst.data_ptr(), size);
}


extern "C" void esa_copy_batch(torch::Tensor src_ptrs, torch::Tensor dst_ptrs, int size)
{
    dim3 numThreads = {1024};
    int num = src_ptrs.size(0);
    int totalThreads = ceildiv(size, CUDA_TRANS_UNIT_SIZE);
    dim3 numBlocks = {static_cast<unsigned int>(ceildiv(totalThreads, numThreads.x))};
    CudaCopyKernel<<<numBlocks, numThreads>>>(
        (const void**)src_ptrs.data_ptr(), (void**)dst_ptrs.data_ptr(), size, num);
}

// Scatter-copy rows from pinned host src to device dst using index tables.
// dst[block_table_dst[i]] = src[block_table_src[i]] for i in [0, K)
template <typename index_t>
__global__ void ScatterCopyRowsPinnedKernel(const uint8_t* __restrict__ src,
                                            uint8_t* __restrict__ dst,
                                            const index_t* __restrict__ src_idx,
                                            const index_t* __restrict__ dst_idx,
                                            size_t row_bytes,
                                            size_t K)
{
    const size_t units_per_row = (row_bytes + CUDA_TRANS_UNIT_SIZE - 1) / CUDA_TRANS_UNIT_SIZE;
    const size_t total_units = units_per_row * K;
    size_t gtid = blockIdx.x * blockDim.x + threadIdx.x;

    while (gtid < total_units) {
        size_t i = gtid / units_per_row;          // which pair
        size_t unit_off = gtid % units_per_row;   // which 32B unit within the row
        size_t byte_off = unit_off * CUDA_TRANS_UNIT_SIZE;

        size_t srow = static_cast<size_t>(src_idx[i]);
        size_t drow = static_cast<size_t>(dst_idx[i]);

        const uint8_t* s_ptr = src + srow * row_bytes + byte_off;
        uint8_t* d_ptr = dst + drow * row_bytes + byte_off;

        size_t remaining = row_bytes - byte_off;
        if (remaining >= CUDA_TRANS_UNIT_SIZE) {
            CudaCopyUnit(s_ptr, (volatile uint8_t*)d_ptr);
        } else {
            // tail copy for the last partial unit
            for (size_t b = 0; b < remaining; ++b) {
                d_ptr[b] = s_ptr[b];
            }
        }

        gtid += gridDim.x * blockDim.x;
    }
}

extern "C" void esa_scatter_copy(torch::Tensor src,
                                 torch::Tensor dst,
                                 torch::Tensor block_table_src,
                                 torch::Tensor block_table_dst)
{
    TORCH_CHECK(src.dim() == 2 && dst.dim() == 2, "esa_scatter_copy: src and dst must be 2D [N, dim] and [M, dim]");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "esa_scatter_copy: src and dst must be contiguous");

    TORCH_CHECK(block_table_src.device().is_cuda() && block_table_dst.device().is_cuda(),
                "esa_scatter_copy: block tables must be CUDA tensors");
    TORCH_CHECK(block_table_src.numel() == block_table_dst.numel(),
                "esa_scatter_copy: block_table_src and block_table_dst must have the same length");

    const size_t K = static_cast<size_t>(block_table_src.numel());
    if (K == 0) {
        return;
    }

    const size_t row_bytes = static_cast<size_t>(src.size(1)) * static_cast<size_t>(src.element_size());

    // Launch configuration: one thread processes one 32B unit.
    const size_t units_per_row = (row_bytes + CUDA_TRANS_UNIT_SIZE - 1) / CUDA_TRANS_UNIT_SIZE;
    const size_t total_units = units_per_row * K;

    dim3 threads(1024);
    dim3 blocks(static_cast<unsigned int>(ceildiv(static_cast<int>(total_units), static_cast<int>(threads.x))));

    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src.data_ptr());
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (block_table_src.scalar_type() == at::kLong && block_table_dst.scalar_type() == at::kLong) {
        const int64_t* sidx = block_table_src.data_ptr<int64_t>();
        const int64_t* didx = block_table_dst.data_ptr<int64_t>();
        ScatterCopyRowsPinnedKernel<int64_t><<<blocks, threads, 0, stream>>>(
            src_ptr, dst_ptr, sidx, didx, row_bytes, K);
    } else if (block_table_src.scalar_type() == at::kInt && block_table_dst.scalar_type() == at::kInt) {
        const int32_t* sidx = block_table_src.data_ptr<int32_t>();
        const int32_t* didx = block_table_dst.data_ptr<int32_t>();
        ScatterCopyRowsPinnedKernel<int32_t><<<blocks, threads, 0, stream>>>(
            src_ptr, dst_ptr, sidx, didx, row_bytes, K);
    } else {
        TORCH_CHECK(false, "esa_scatter_copy: block tables must both be int32 or both be int64");
    }
}
