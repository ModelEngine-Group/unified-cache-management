#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include "cp_async.cuh"
#include "operator.h"

#define HEAD_SWITCH(val, NumHead, ...)        \
  do {                                        \
    if ((val) == 128) {                        \
      constexpr int NumHead = 128;             \
      {                                       \
        __VA_ARGS__                           \
      }                                       \
    } else if ((val) == 64) {                 \
      constexpr int NumHead = 64;             \
      {                                       \
        __VA_ARGS__                           \
      }                                       \
    } else if ((val) == 32) {                 \
      constexpr int NumHead = 32;             \
      {                                       \
        __VA_ARGS__                           \
      }                                       \
    } else if ((val) == 16) {                 \
      constexpr int NumHead = 16;             \
      {                                       \
        __VA_ARGS__                           \
      }                                       \
    } else if ((val) == 8) {                  \
      constexpr int NumHead = 8;              \
      {                                       \
        __VA_ARGS__                           \
      }                                       \
    } else if ((val) == 4) {                  \
      constexpr int NumHead = 4;              \
      {                                       \
        __VA_ARGS__                           \
      }                                       \
    } else if ((val) == 2) {                  \
      constexpr int NumHead = 2;              \
      {                                       \
        __VA_ARGS__                           \
      }                                       \
    } else if ((val) == 1) {                  \
      constexpr int NumHead = 1;              \
      {                                       \
        __VA_ARGS__                           \
      }                                       \
    } else {                                  \
      LOG(FATAL) << "NumHead is not support"; \
    }                                         \
  } while (0);

#define KVHEAD_SWITCH(val, NumKVHead, ...)      \
  do {                                          \
    if ((val) == 1) {                           \
      constexpr int NumKVHead = 1;              \
      {                                         \
        __VA_ARGS__                             \
      }                                         \
    } else {                                    \
      LOG(FATAL) << "NumKVHead is not support"; \
    }                                           \
  } while (0);

#define NUMCHUNK_SWITCH(val, NumChunk, ...)    \
  do {                                         \
    if ((val) == 18) {                          \
      constexpr int NumChunk = 18;              \
      {                                        \
        __VA_ARGS__                            \
      }                                        \
    } else if ((val) == 16) {                   \
      constexpr int NumChunk = 16;              \
      {                                        \
        __VA_ARGS__                            \
      }                                        \
    } else if ((val) == 8) {                   \
      constexpr int NumChunk = 8;              \
      {                                        \
        __VA_ARGS__                            \
      }                                        \
    } else if ((val) == 4) {                     \
      constexpr int NumChunk = 4;              \
      {                                        \
        __VA_ARGS__                            \
      }                                        \
    } else {                                   \
      LOG(FATAL) << "NumChunk is not support"; \
    }                                          \
  } while (0);

namespace kvlib {

template <typename T, bool USE_INT64, int32_t NumThreads, int32_t ELEMS,
          int32_t NumHead, int32_t NumKVHead, int32_t NumChunk>
__global__ void HammingScoreContiKernel(void* __restrict__ keys_ptr,
                                        void* __restrict__ query_ptr,
                                        half* __restrict__ output_ptr, int32_t BSZ,
                                        int32_t SEQ, int32_t SINK, int32_t RECENT,
                                        const int32_t* __restrict__ block_table_ptr,
                                        int32_t block_size,
                                        int32_t max_num_block_per_seq, int32_t num_blocks,
                                        const int32_t* __restrict__ seq_len_ptr) {
  assert(NumThreads == blockDim.x);
  constexpr int32_t BLOCK_M = (ELEMS / NumChunk / NumKVHead);
  constexpr int32_t KVGroup = NumHead / NumKVHead;

  const int32_t tid = threadIdx.x;
  const int32_t batch_id = blockIdx.y;

  const int32_t actual_seq_len = seq_len_ptr[batch_id];

  const int32_t block_id = blockIdx.x;
  const int32_t block_start_m = block_id * BLOCK_M;
  const int32_t left_m = min(BLOCK_M, SEQ - block_start_m);
  const int32_t left_elem = left_m * NumKVHead * NumChunk;

  extern __shared__ int32_t smem[];
  int32_t* smem_popc = smem;
  T* smem_query_code = (T*)(smem_popc + NumThreads);

  T* q_ptr = (T*)query_ptr + batch_id * 1 * NumHead * NumChunk;
// load q to smem
#pragma unroll
  for (int i = tid; i < NumHead * NumChunk; i += NumThreads) {
    smem_query_code[i] = q_ptr[i];
  }

  __syncthreads();

  int32_t score = 0;
  if (tid < left_elem) {
    const int32_t kv_head_id = (tid / NumChunk) % NumKVHead;
    const int32_t kv_chunk_id = tid % NumChunk;
    const int32_t token_in_block_idx = (tid/ NumChunk) / NumKVHead;

    const int32_t token_global_m = block_start_m + token_in_block_idx;

    const int32_t block_slot = token_global_m / block_size;
    const int32_t offset_in_block = token_global_m % block_size;

    int32_t block_id_from_table = 0;
    if(block_slot >=0 && block_slot < max_num_block_per_seq) {
        block_id_from_table = block_table_ptr[batch_id * max_num_block_per_seq + block_slot];
    }

    bool is_valid_block = (block_start_m < actual_seq_len);

    if (is_valid_block) {
      T* base_block =
        (T*)keys_ptr + (int64_t)block_id_from_table * (int64_t)(NumKVHead * block_size * NumChunk);
      
      T key = base_block[(kv_head_id * block_size + offset_in_block) * NumChunk + kv_chunk_id];

      T* query = smem_query_code + kv_head_id * KVGroup * NumChunk + kv_chunk_id;
#pragma unroll
      for(int i=0; i<KVGroup; i++) {
        T q = *(query + i * NumChunk);
        T tmp = key ^ q;
        score += USE_INT64 ? __popcll(tmp) : __popc(tmp);
      }
    } else {
        score += 0;
    }
  }

  smem_popc[tid] = score;
  __syncthreads();

  // write results to global memory
  if (tid < NumKVHead * BLOCK_M) {
    int kv_head_id = tid / BLOCK_M;
    int m_id = tid % BLOCK_M;
    int pos = m_id + block_start_m;
    bool is_sink_or_recent = (pos < SINK) ||
                             ((pos >= (actual_seq_len - RECENT)) && (pos < actual_seq_len));
    bool is_inf = pos >= actual_seq_len;

    if (m_id < left_m) {
      // transpose
      half* o_ptr =
          (half*)output_ptr + batch_id * NumKVHead * SEQ + block_start_m;
      half* _o_ptr = o_ptr + kv_head_id * SEQ + m_id;
      half sum = (half)(0.);

#pragma unroll
      for (int h = 0; h < NumChunk; h += 1) {
        sum +=
            (half)(smem_popc[(m_id * NumKVHead + kv_head_id) * NumChunk + h]);
      }

      *_o_ptr = is_inf ? half(INFINITY) :
            (is_sink_or_recent ? half(0.) : sum);
    }
  }
}

template <typename T, bool USE_INT64, int32_t NumThreads, int32_t ELEMS,
          int32_t NumHead, int32_t NumKVHead, int32_t NumChunk>
__global__ void HammingScoreKernel(void* __restrict__ keys_ptr,
                                   void* __restrict__ query_ptr,
                                   half* __restrict__ output_ptr, int32_t BSZ,
                                   int32_t SEQ, int32_t batch_key_code_stride,
                                   int32_t SINK, int32_t RECENT) {
  assert(NumThreads == blockDim.x);
  constexpr int32_t BLOCK_M = (ELEMS / NumChunk / NumKVHead);
  constexpr int32_t KVGroup = NumHead / NumKVHead;

  const half RBIT = (half)((int32_t)sizeof(T) * 8 * NumChunk);

  const int32_t tid = threadIdx.x;
  const int32_t batch_id = blockIdx.y;

  const int32_t block_id = blockIdx.x;
  const int32_t block_start_elem = block_id * ELEMS;
  const int32_t block_start_m = block_id * BLOCK_M;
  const int32_t left_m = min(BLOCK_M, SEQ - block_start_m);
  const int32_t left_elem = left_m * NumKVHead * NumChunk;

  extern __shared__ int32_t smem[];
  int32_t* smem_popc = smem;
  T* smem_query_code = (T*)(smem_popc + NumThreads);

  T* q_ptr = (T*)query_ptr + batch_id * 1 * NumHead * NumChunk;
// load q to smem
#pragma unroll
  for (int i = tid; i < NumHead * NumChunk; i += NumThreads) {
    smem_query_code[i] = q_ptr[i];
  }

  __syncthreads();

  int32_t score = 0;
  if (tid < left_elem) {
    T* k_ptr =
        (T*)keys_ptr + batch_id * batch_key_code_stride + block_start_elem;

    const int32_t kv_head_id = (tid / NumChunk) % NumKVHead;
    const int32_t kv_chunk_id = tid % NumChunk;

    T key = k_ptr[tid];
    T* query = smem_query_code + kv_head_id * KVGroup * NumChunk + kv_chunk_id;

#pragma unroll
    for (int i = 0; i < KVGroup; i++) {
      T q = *(query + i * NumChunk);
      T tmp = key ^ q;
      score += USE_INT64 ? __popcll(tmp) : __popc(tmp);
    }
  }

  smem_popc[tid] = score;
  __syncthreads();

  // write results to global memory
  if (tid < NumKVHead * BLOCK_M) {
    int kv_head_id = tid / BLOCK_M;
    int m_id = tid % BLOCK_M;
    bool is_sink_or_recent = (m_id + block_start_m) < SINK ||
                             (m_id + block_start_m) >= (SEQ - RECENT);

    if (m_id < left_m) {
      // transpose
      half* o_ptr =
          (half*)output_ptr + batch_id * NumKVHead * SEQ + block_start_m;
      half* _o_ptr = o_ptr + kv_head_id * SEQ + m_id;
      half sum = (half)(0.);

#pragma unroll
      for (int h = 0; h < NumChunk; h += 1) {
        sum +=
            (half)(smem_popc[(m_id * NumKVHead + kv_head_id) * NumChunk + h]);
      }

      *_o_ptr = is_sink_or_recent ? half(0.) : sum;
    }
  }
}

torch::Tensor HammingScoreContiCUDA(torch::Tensor& key_codes,
                                    torch::Tensor& query_code,
                                    torch::optional<torch::Tensor> block_table_opt,
                                    torch::Tensor& seq_len, int32_t max_seq_len,
  int32_t sink, int32_t recent) {
  // shape for Legacy key_codes is (BATCH_SIZE, SEQ, #NUM_K_HEAD, num_chunk) and dtype
  // is int32 shape for query_code is (BATCH_SIZE, 1, #NUM_HEAD, num_chunk) and
  // dtype is int32
  // shape for Block key_codes is (num_blocks, num_kv_head, block_size, num_chunk)->(num_blocks, block_size, num_kv_head, num_chunk)
  // shape for block_table is (batch_size, num_blocks_per_seq)
  // shape for output is (BATCH_SIZE, #NUM_KV_HEAD, SEQ) [transpose head and seq
  // here] and dtype is float16
 
  bool is_block_mode = block_table_opt.has_value();
 
  int32_t bsz = query_code.size(0);
  int32_t num_kv_head = is_block_mode ? key_codes.size(1) : key_codes.size(2);
  int32_t num_chunk = key_codes.size(3);
 
  int32_t num_head = query_code.size(2);
 
  const int32_t* seq_len_ptr = seq_len.data_ptr<int32_t>();
 
  auto device = key_codes.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);
 
  if(is_block_mode) {
    int32_t num_blocks = key_codes.size(0);
    int32_t block_size = key_codes.size(2);
    const auto& block_table = block_table_opt.value(); // *block_table_opt;
    int32_t max_num_block_per_seq = block_table.size(1);
    TORCH_CHECK(bsz == block_table.size(0), "batch size mismatch between query_code and block_table");
    TORCH_CHECK(key_codes.is_contiguous(), "key_codes must be contiguous, but got non-contiguous tensor");
    torch::Tensor output = torch::empty({bsz, num_kv_head, max_seq_len}, options);
    const int32_t* block_table_ptr = block_table.data_ptr<int32_t>();
    HEAD_SWITCH(num_head, NumHead, {
      KVHEAD_SWITCH(num_kv_head, NumKVHead, {
        NUMCHUNK_SWITCH(num_chunk, NumChunk, {
          constexpr int32_t NumThreads = 512;
          size_t shm_size = 0;
          shm_size += NumThreads * sizeof(int32_t); // for popc results
          shm_size += NumHead * NumChunk * sizeof(int32_t);
 
          if (NumChunk % 2 == 0) { 
            // convert int32 to int64
            constexpr int32_t HalfNumChunk = NumChunk/2 < 1 ? 1 : NumChunk/2; 
            constexpr int32_t NumTokens = NumThreads/(NumKVHead * HalfNumChunk);
            constexpr int32_t ELEMS_PER_BLOCK = 
              NumKVHead * HalfNumChunk * NumTokens; 
 
            dim3 blks(NumThreads);
            dim3 grids(
              (max_seq_len * NumKVHead * HalfNumChunk + ELEMS_PER_BLOCK - 1) / 
                ELEMS_PER_BLOCK, 
                bsz);
 
            HammingScoreContiKernel<int64_t, true, NumThreads, ELEMS_PER_BLOCK, 
                        NumHead, NumKVHead, HalfNumChunk>
                <<<grids, blks, shm_size, stream>>>(
                    key_codes.data_ptr(), query_code.data_ptr(),
                    (half*)(output.data_ptr<at::Half>()), bsz, max_seq_len, sink, recent,
                    block_table_ptr, block_size, max_num_block_per_seq, num_blocks, seq_len_ptr);
 
          } else {
            constexpr int32_t NumTokens = NumThreads / (NumKVHead * NumChunk);
            constexpr int32_t ELEMS_PER_BLOCK = NumKVHead * NumChunk * NumTokens; // = NumThreads
 
            dim3 blks(NumThreads); // blockDim.x=NumThreads
            dim3 grids((max_seq_len * NumKVHead * NumChunk + ELEMS_PER_BLOCK - 1) / 
                        ELEMS_PER_BLOCK, // gridDim.x=
                        bsz); // gridDim.y=bsz
 
            HammingScoreContiKernel<int32_t, false, NumThreads, ELEMS_PER_BLOCK, 
                            NumHead, NumKVHead, NumChunk>
                    <<<grids, blks, shm_size, stream>>>(
                        key_codes.data_ptr(), query_code.data_ptr(),
                        (half*)(output.data_ptr<at::Half>()), bsz, max_seq_len, sink, recent,
                        block_table_ptr, block_size, max_num_block_per_seq, num_blocks, seq_len_ptr);
            }          
        });
      });
    });
 
    return output;
} else {
  torch::Tensor output = torch::empty({bsz, num_kv_head, max_seq_len}, options);
  int32_t batch_key_code_stride = key_codes.stride(0);
  HEAD_SWITCH(num_head, NumHead, {
    KVHEAD_SWITCH(num_kv_head, NumKVHead, {
      NUMCHUNK_SWITCH(num_chunk, NumChunk, {
        constexpr int32_t NumThreads = 512;
        size_t shm_size = 0;
        shm_size += NumThreads * sizeof(int32_t); // for popc results
        shm_size += NumHead * NumChunk * sizeof(int32_t); 
        
        if (NumChunk % 2 == 0) { 
            // convert int32 to int64
            constexpr int32_t HalfNumChunk = NumChunk / 2 < 1 ? 1 : NumChunk / 2;
            constexpr int32_t NumTokens = NumThreads / (NumKVHead * HalfNumChunk);
            constexpr int32_t ELEMS_PER_BLOCK =
                NumKVHead * HalfNumChunk * NumTokens; 
 
            dim3 blks(NumThreads);
            dim3 grids(
                (max_seq_len * NumKVHead * HalfNumChunk + ELEMS_PER_BLOCK - 1) /
                    ELEMS_PER_BLOCK,
                    bsz);
 
            HammingScoreKernel<int64_t, true, NumThreads, ELEMS_PER_BLOCK,
                        NumHead, NumKVHead, HalfNumChunk>
                <<<grids, blks, shm_size, stream>>>(
                    key_codes.data_ptr(), query_code.data_ptr(),
                    (half*)(output.data_ptr<at::Half>()), bsz, max_seq_len,
                    (batch_key_code_stride / 2), sink, recent);
 
        } else {
            constexpr int32_t NumTokens = NumThreads / (NumKVHead * NumChunk);
            constexpr int32_t ELEMS_PER_BLOCK = NumKVHead * NumChunk * NumTokens; // = NumThreads
 
            dim3 blks(NumThreads);
            dim3 grids((max_seq_len * NumKVHead * NumChunk + ELEMS_PER_BLOCK - 1) /
                        ELEMS_PER_BLOCK,
                    bsz);
 
            HammingScoreKernel<int32_t, false, NumThreads, ELEMS_PER_BLOCK,
                                    NumHead, NumKVHead, NumChunk>
                <<<grids, blks, shm_size, stream>>>(
                    key_codes.data_ptr(), query_code.data_ptr(),
                    (half*)(output.data_ptr<at::Half>()), bsz, max_seq_len,
                    batch_key_code_stride, sink, recent);
        }
      });
    });
  });
 
  return output;
 }
}
 
} // namespace kvlib