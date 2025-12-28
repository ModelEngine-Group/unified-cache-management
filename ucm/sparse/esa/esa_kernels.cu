#include <cub/cub.cuh>
#include <cstddef>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <atomic>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <thread>
#include <condition_variable>
#include <deque>
#include <vector>
#include <algorithm>
#include <nvtx3/nvtx3.hpp>

#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()

__inline__ __device__ float warpReduceSum(float val)
{
    int warpSize = 32;
    unsigned mask = __activemask();          // ballot of *all* currently active threads
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;                              // only lane 0 holds the total
}

constexpr __host__ __device__
int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}
/**
 * This kernel performs: repre_cache[repre_repre_index[i]] = mean( key_cache[key_repre_index[i]], 0 )
 *
 * @param key_cache: [N, block_size, dim]
 * @param repre_cache: [N, dim]
 * @param key_repre_index: [S]
 * @param repre_repre_index: [S]
 */
__global__ void extract_repre_fp32(const float *key_cache, float *repre_cache, const int *block_table, const int *repre_index, int block_size, int dim, int num_blocks, int key_rows, int repre_rows) {
    int idx = blockIdx.x;
    if (idx >= num_blocks){
        return;
    }
    int index1 = block_table[idx];
    int index2 = repre_index[idx];
    if (index1 < 0 || index1 >= key_rows || index2 < 0 || index2 >= repre_rows) {
        return;
    }
    const float* key_ptr = key_cache + index1 * block_size * dim;
    float* repre_ptr = repre_cache + index2 * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < block_size; ++j) {
            sum += key_ptr[j * dim + d];
        }
        repre_ptr[d] = sum / block_size;
    }
}

__global__ void extract_repre_bf16(const __nv_bfloat16 *key_cache, __nv_bfloat16 *repre_cache, const int *block_table, const int *repre_index, int block_size, int dim, int num_blocks, int key_rows, int repre_rows) {
    int idx = blockIdx.x;
    if (idx >= num_blocks){
        return;
    }
    int index1 = block_table[idx];
    int index2 = repre_index[idx];
    if (index1 < 0 || index1 >= key_rows || index2 < 0 || index2 >= repre_rows) {
        return;
    }
    const __nv_bfloat16* key_ptr = key_cache + index1 * block_size * dim;
    __nv_bfloat16* repre_ptr = repre_cache + index2 * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < block_size; ++j) {
            sum += __bfloat162float(key_ptr[j * dim + d]);
        }
        repre_ptr[d] = __float2bfloat16(sum / block_size);
    }
}

__global__ void extract_repre_fp16(const __half *key_cache, __half *repre_cache, const int *block_table, const int *repre_index, int block_size, int dim, int num_blocks, int key_rows, int repre_rows) {
    int idx = blockIdx.x;
    if (idx >= num_blocks){
        return;
    }
    int index1 = block_table[idx];
    int index2 = repre_index[idx];
    if (index1 < 0 || index1 >= key_rows || index2 < 0 || index2 >= repre_rows) {
        return;
    }
    const __half* key_ptr = key_cache + index1 * block_size * dim;
    __half* repre_ptr = repre_cache + index2 * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < block_size; ++j) {
            sum += __half2float(key_ptr[j * dim + d]);
        }
        repre_ptr[d] = __float2half(sum / block_size);
    }
}

/**
 * This kernel performs: score[i] = queries[query_index[i]] * repre_cache[repre_index[i]]
 *
 * @param queries: a list of tensors. { batch_size * [num_q_heads, dim] }
 * @param repre_cache: [N, num_k_heads, dim]
 * @param score: [S]
 * @param repre_index: [S]
 * @param query_index: [S]
 */

__global__ void retrieval_kernel_fp16(__half *__restrict__ queries, __half *__restrict__ repre_cache, __half *__restrict__ score, int *__restrict__ repre_index, int *__restrict__ query_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries + query_index[blockIdx.x] * num_q_heads * dim;
    auto *k_offset = repre_cache + repre_index[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += __half2float(q_val) * __half2float(k_val);
            }
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = __float2half(sum);
        }
    }
}


__global__ void retrieval_kernel_fp32(float *__restrict__ queries, float *__restrict__ repre_cache, float *__restrict__ score, int *__restrict__ repre_index, int *__restrict__ query_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries + query_index[blockIdx.x] * num_q_heads * dim;
    auto *k_offset = repre_cache + repre_index[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += q_val * k_val;
            }
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = sum;
        }
    }
}

__global__ void retrieval_kernel_bf16(__nv_bfloat16 *__restrict__ queries, __nv_bfloat16 *__restrict__ repre_cache, __nv_bfloat16 *__restrict__ score, int *__restrict__ repre_index, int *__restrict__ query_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries + query_index[blockIdx.x] * num_q_heads * dim;
    auto *k_offset = repre_cache + repre_index[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += __bfloat162float(q_val) * __bfloat162float(k_val);
            }
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = __float2bfloat16(sum);
        }
    }
}


extern "C" void esa_repre(torch::Tensor key_cache, torch::Tensor repre_cache, torch::Tensor block_table, torch::Tensor repre_table){
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
    TORCH_CHECK(repre_cache.is_cuda(), "repre_cache must be a CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
    TORCH_CHECK(repre_table.is_cuda(), "repre_index must be a CUDA tensor");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(repre_cache.is_contiguous(), "repre_cache must be contiguous");

    // Shape validations based on expected contract:
    // key_cache: [N, block_size, dim], repre_cache: [M, dim]
    TORCH_CHECK(key_cache.dim() == 3, "key_cache must be 3D [N, block_size, dim]");
    TORCH_CHECK(repre_cache.dim() == 2, "repre_cache must be 2D [M, dim]");
    TORCH_CHECK(block_table.dim() == 1 && repre_table.dim() == 1, "block_table and repre_index must be 1-D");
    TORCH_CHECK(block_table.size(0) == repre_table.size(0), "block_table and repre_index must have the same length");

    // Indices must be int32 on device and contiguous for the kernel
    if (block_table.scalar_type() != at::kInt || !block_table.is_contiguous()) {
        block_table = block_table.to(at::kInt).contiguous();
    }
    if (repre_table.scalar_type() != at::kInt || !repre_table.is_contiguous()) {
        repre_table = repre_table.to(at::kInt).contiguous();
    }

    int block_size = key_cache.size(1);
    int dim = repre_cache.size(-1);
    int num_blocks = block_table.size(0);
    int key_rows = key_cache.size(0);
    int repre_rows = repre_cache.size(0);

    int threads = dim < 1024 ? dim : 1024;
    int blocks = num_blocks;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, key_cache.scalar_type(), "esa_repre_cuda", ([&] {
        if constexpr (std::is_same_v<scalar_t, float>) {
            extract_repre_fp32<<<blocks, threads>>>(
                key_cache.data_ptr<float>(),
                repre_cache.data_ptr<float>(),
                block_table.data_ptr<int>(),
                repre_table.data_ptr<int>(),
                block_size,
                dim,
                num_blocks,
                key_rows,
                repre_rows);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
            extract_repre_fp16<<<blocks, threads>>>(
                reinterpret_cast<__half*>(key_cache.data_ptr()),
                reinterpret_cast<__half*>(repre_cache.data_ptr()),
                block_table.data_ptr<int>(),
                repre_table.data_ptr<int>(),
                block_size,
                dim,
                num_blocks,
                key_rows,
                repre_rows);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            extract_repre_bf16<<<blocks, threads>>>(
                reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(repre_cache.data_ptr()),
                block_table.data_ptr<int>(),
                repre_table.data_ptr<int>(),
                block_size,
                dim,
                num_blocks,
                key_rows,
                repre_rows);
        }
    }));
}

extern "C" void esa_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, torch::Tensor workspace){
    void* temp_workspace = nullptr;
    size_t temp_bytes = 0;
    size_t B = offsets.size(0) - 1;
    size_t total = score.size(0);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_workspace, temp_bytes,
            score.data_ptr<float>(),  score_out.data_ptr<float>(),
            index.data_ptr<int>(), index_out.data_ptr<int>(),
            total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
    // NOTE: Don't use malloc, just reuse the workspace, but the first call of
    // SortPairsDescending is necesssary to determine the workspace size.
    // CUDA_CHECK(cudaMalloc(&temp_workspace, temp_bytes));
    temp_workspace = workspace.data_ptr<int>();
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_workspace, temp_bytes,
            score.data_ptr<float>(),  score_out.data_ptr<float>(),
            index.data_ptr<int>(), index_out.data_ptr<int>(),
            total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
}


namespace {

// Forward declare SM copy kernel API implemented in esa_sm_copy.cu
extern "C" void esa_copy(torch::Tensor src, torch::Tensor dst, size_t size);

// Async context for CPU argsort per-batch
struct RetrievalCtx {
    std::atomic<int> ready{0};
    int S{0};
    int B{0};

    // Keep tensors alive and provide host-side access
    torch::Tensor score_dev;           // [S] device scores
    torch::Tensor score_cpu;           // [S] pinned CPU buffer (same dtype as score_dev)
    torch::Tensor score_sorted_cpu;    // [S] pinned CPU buffer
    torch::Tensor index_sorted_cpu;    // [S] pinned CPU int32
    torch::Tensor repre_index_dev;     // [S] device int32 (input)
    torch::Tensor repre_index_cpu;     // [S] pinned CPU int32 (copied from device)
    torch::Tensor offsets_cpu;         // [B+1] pinned CPU int32 (copied from device)

    // For stream dependency
    cudaEvent_t ready_event{nullptr};
};

std::mutex g_mutex;
std::unordered_map<int, std::unique_ptr<RetrievalCtx>> g_ctx;
int g_next_handle = 1;

// Worker thread infra
std::mutex w_mutex;
std::condition_variable w_cv;
std::deque<RetrievalCtx*> w_queue;
std::atomic<bool> w_started{false};
std::atomic<bool> w_running{false};
std::thread w_thread;

/* Dedicated stream for host callbacks (to decouple from compute stream) */
static cudaStream_t g_cb_stream = nullptr;

inline void ensure_cb_stream() {
    if (g_cb_stream == nullptr) {
        cudaStreamCreateWithFlags(&g_cb_stream, cudaStreamNonBlocking);
    }
}

void worker_loop() {
    while (w_running.load(std::memory_order_acquire)) {
        RetrievalCtx* job = nullptr;
        {
            std::unique_lock<std::mutex> lk(w_mutex);
            w_cv.wait(lk, [] {
                return !w_queue.empty() || !w_running.load(std::memory_order_acquire);
            });
            if (!w_running.load(std::memory_order_acquire)) break;
            job = w_queue.front();
            w_queue.pop_front();
        }
        if (!job) continue;

        NVTX_PUSH("worker_argmin");
        // Perform segmented argsort on CPU
        const int32_t* offsets = job->offsets_cpu.data_ptr<int32_t>();
        int32_t* out_idx = job->index_sorted_cpu.data_ptr<int32_t>();
        auto st = job->score_cpu.scalar_type();

        auto do_segments = [&](auto* scores_ptr, auto* out_scores_ptr) {
            using T = std::remove_pointer_t<decltype(scores_ptr)>;
            for (int b = 0; b < job->B; ++b) {
                int s = offsets[b];
                int e = offsets[b + 1];
                int n = e - s;
                if (n <= 0) continue;

                std::vector<int> idx(n);
                for (int i = 0; i < n; ++i) idx[i] = i;
                std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                    float va = static_cast<float>(scores_ptr[s + a]);
                    float vb = static_cast<float>(scores_ptr[s + b]);
                    if (va == vb) return a < b; // stable tie-breaker on original position
                    return va > vb; // descending
                });
                for (int i = 0; i < n; ++i) {
                    int src = s + idx[i];
                    out_idx[s + i] = job->repre_index_cpu.data_ptr<int32_t>()[src];
                    out_scores_ptr[s + i] = scores_ptr[src];
                }
            }
        };

        if (st == at::kFloat) {
            do_segments(job->score_cpu.data_ptr<float>(),
                        job->score_sorted_cpu.data_ptr<float>());
        } else if (st == at::kHalf) {
            do_segments(job->score_cpu.data_ptr<at::Half>(),
                        job->score_sorted_cpu.data_ptr<at::Half>());
        } else if (st == at::kBFloat16) {
            do_segments(job->score_cpu.data_ptr<at::BFloat16>(),
                        job->score_sorted_cpu.data_ptr<at::BFloat16>());
        } else {
            // Unsupported dtype; mark ready without doing anything
        }

        job->ready.store(1, std::memory_order_release);
        NVTX_POP();
    }
}

void ensure_worker() {
    bool expected = false;
    if (w_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        w_running.store(true, std::memory_order_release);
        w_thread = std::thread(worker_loop);
        // not detached; joined on shutdown
    }
}

void enqueue_job(RetrievalCtx* ctx) {
    {
        std::lock_guard<std::mutex> lk(w_mutex);
        w_queue.push_back(ctx);
    }
    w_cv.notify_one();
}

void CUDART_CB host_cb(void* userData) {
    auto* ctx = reinterpret_cast<RetrievalCtx*>(userData);
    if (ctx->ready_event) {
        cudaEventDestroy(ctx->ready_event);
        ctx->ready_event = nullptr;
    }
    enqueue_job(ctx);
}

} // anonymous namespace

extern "C" int esa_retrieval_poll(int handle) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_ctx.find(handle);
    if (it == g_ctx.end()) return -1;
    return it->second->ready.load(std::memory_order_acquire) ? 1 : 0;
}

extern "C" int esa_retrieval_cleanup(int handle) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_ctx.find(handle);
    if (it == g_ctx.end()) return 0;
    g_ctx.erase(it);
    return 1;
}

extern "C" int esa_retrieval_pending() {
    std::lock_guard<std::mutex> lk(g_mutex);
    return static_cast<int>(g_ctx.size());
}

extern "C" void esa_retrieval_shutdown() {
    if (w_started.load(std::memory_order_acquire)) {
        w_running.store(false, std::memory_order_release);
        w_cv.notify_all();
        if (w_thread.joinable()) w_thread.join();
        w_started.store(false, std::memory_order_release);
    }
    if (g_cb_stream) {
        cudaStreamDestroy(g_cb_stream);
        g_cb_stream = nullptr;
    }
}

extern "C" int esa_retrieval_launcher(torch::Tensor query, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor repre_index_cpu,
        torch::Tensor batch_offset, torch::Tensor score, torch::Tensor score_cpu, torch::Tensor score_sorted_cpu, torch::Tensor index_sorted_cpu,
        int batch, int s){
    TORCH_CHECK(query.dim() == 3, "query dim must be 3");
    TORCH_CHECK(repre_cache.dim() == 3, "repre_cache dim must be 3");
    TORCH_CHECK(q_index.size(0) == repre_index.size(0), "q_index shape should be same with repre_index");
    TORCH_CHECK(q_index.dtype() == at::kInt, "q_index must be int32 (torch.long)");
    TORCH_CHECK(repre_index.dtype() == at::kInt, "repre_index must be int32 (torch.long)");
    TORCH_CHECK(batch_offset.dtype() == at::kInt, "batch_offset must be int32 (torch.long)");

    // CPU pinned outputs must be provided
    TORCH_CHECK(score_cpu.device().is_cpu() && score_cpu.is_pinned() && score_cpu.dim() == 1, "score_cpu must be pinned CPU [s]");
    TORCH_CHECK(score_sorted_cpu.device().is_cpu() && score_sorted_cpu.is_pinned() && score_sorted_cpu.dim() == 1, "score_sorted_cpu must be pinned CPU [s]");
    TORCH_CHECK(index_sorted_cpu.device().is_cpu() && index_sorted_cpu.is_pinned() && index_sorted_cpu.scalar_type() == at::kInt && index_sorted_cpu.dim() == 1, "index_sorted_cpu must be pinned CPU int32 [s]");
    TORCH_CHECK(score_cpu.scalar_type() == score.scalar_type(), "score_cpu dtype must match score dtype");
    TORCH_CHECK(score_sorted_cpu.scalar_type() == score.scalar_type(), "score_sorted_cpu dtype must match score dtype");

    int num_k_heads = repre_cache.size(1);
    int num_q_heads = query.size(1);
    int dim = repre_cache.size(2);

    dim3 numBlocks = {(unsigned int)(s)};
    dim3 numThreads = {32, 32};
    int numWarps = ceildiv(numThreads.x * numThreads.y, 32);
    size_t bytes = numWarps * sizeof(float);

    // Use current CUDA stream ('main stream' for this context)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    ensure_cb_stream();

    NVTX_PUSH("esa_retrieval: kernel");
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, repre_cache.scalar_type(), "esa_retrieval_cuda", ([&] {
        if constexpr (std::is_same_v<scalar_t, float>) {
            retrieval_kernel_fp32<<<numBlocks, numThreads, bytes, stream>>>(
                reinterpret_cast<float*>(query.data_ptr()),
                reinterpret_cast<float*>(repre_cache.data_ptr()),
                reinterpret_cast<float*>(score.data_ptr()),
                repre_index.data_ptr<int>(),
                q_index.data_ptr<int>(),
                num_q_heads, num_k_heads, dim, s);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
            retrieval_kernel_fp16<<<numBlocks, numThreads, bytes, stream>>>(
                reinterpret_cast<__half*>(query.data_ptr()),
                reinterpret_cast<__half*>(repre_cache.data_ptr()),
                reinterpret_cast<__half*>(score.data_ptr()),
                repre_index.data_ptr<int>(),
                q_index.data_ptr<int>(),
                num_q_heads, num_k_heads, dim, s);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            retrieval_kernel_bf16<<<numBlocks, numThreads, bytes, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(query.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(repre_cache.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                repre_index.data_ptr<int>(),
                q_index.data_ptr<int>(),
                num_q_heads, num_k_heads, dim, s);
        }
    }));
    NVTX_POP();

    // Copy scores to pinned CPU using SM-copy kernel (must use current stream internally).
    // Make esa_copy run on our private compute stream by guarding the current stream.
    size_t score_bytes = static_cast<size_t>(s) * static_cast<size_t>(score.element_size());
    NVTX_PUSH("esa_retrieval: D2H score->cpu (cudaMemcpyAsync)");
    esa_copy(score, score_cpu, score_bytes);
    // cudaError_t memcpy_status = cudaMemcpyAsync(
    //     score_cpu.data_ptr(),            // dst (pinned host)
    //     score.data_ptr(),                // src (device)
    //     score_bytes,
    //     cudaMemcpyDeviceToHost,
    //     stream
    // );
    // TORCH_CHECK(memcpy_status == cudaSuccess, "cudaMemcpyAsync score->cpu failed: ", cudaGetErrorString(memcpy_status));
    NVTX_POP();

    // Prepare ctx and D2H copies of metadata needed by CPU worker
    auto ctx = std::make_unique<RetrievalCtx>();
    ctx->S = s;
    ctx->B = batch;
    ctx->score_dev = score;
    ctx->score_cpu = score_cpu;
    ctx->score_sorted_cpu = score_sorted_cpu;
    ctx->index_sorted_cpu = index_sorted_cpu;
    ctx->repre_index_dev = repre_index;
    ctx->offsets_cpu = batch_offset;
    ctx->repre_index_cpu = repre_index_cpu;

    int handle;
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        handle = g_next_handle++; // 生成任务id
        g_ctx.emplace(handle, std::move(ctx)); // 保证ctx生命周期跨越python调用返回后仍然存在
    }

    // Record event on compute stream after kernel + copies, then trigger host callback on dedicated stream
    NVTX_PUSH("esa_retrieval: record_event");
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, stream);
    NVTX_POP();

    NVTX_PUSH("esa_retrieval: host_callback_enqueue");
    ensure_cb_stream();
    cudaStreamWaitEvent(g_cb_stream, ev, 0); // worker topk之前确保 score 计算完成

    {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto& ptr = g_ctx.find(handle)->second;
        ptr->ready_event = ev;
        ensure_worker();
        cudaLaunchHostFunc(g_cb_stream, host_cb, ptr.get());
    }
    NVTX_POP();

    // Return immediately; caller can poll for readiness and read CPU outputs
    return handle;
}
