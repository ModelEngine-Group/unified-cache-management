#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

namespace kvlib {

torch::Tensor HammingScoreContiCUDA(torch::Tensor& key_codes, torch::Tensor& query_code,
                                    torch::optional<torch::Tensor> block_table_opt,
                                    torch::Tensor& seq_len, int32_t max_seq_len, int32_t sink,
                                    int32_t recent);

}  // namespace kvlib
