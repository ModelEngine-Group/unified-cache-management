#pragma once

#ifdef USE_ROCM
// torch keeps the cuda spelling for its public symbols on ROCm; the hipified
// context header provides c10::cuda::getCurrentCUDAStream backed by HIP, while
// the cuda-spelled header pulls in NVIDIA-only cuda_runtime_api.h/cusparse.h.
#include <ATen/hip/HIPContext.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#endif
#include <cuda_runtime.h>
#include <torch/script.h>

namespace kvlib {

torch::Tensor HammingScoreContiCUDA(torch::Tensor& key_codes, torch::Tensor& query_code,
                                    torch::optional<torch::Tensor> block_table_opt,
                                    torch::Tensor& seq_len, int32_t max_seq_len, int32_t sink,
                                    int32_t recent, bool reduce_kvhead);

}  // namespace kvlib
