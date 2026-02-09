// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
// Modified from
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/torch_binding_meta.cpp

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include "utils.h"
/*
 * How to write a meta implementation for a custom operator (meta kernel):
 *
 * Meta implementations are used for shape and dtype inference, tracing, and export.
 * They do NOT perform any real computation or allocate device memory.
 * Instead, they return empty tensors with the correct shapes, dtypes, and device types.
 *
 * Steps to write a meta implementation:
 * 1. The function signature should match the operator's schema, but only use the arguments
 *    necessary to infer output shapes and dtypes.
 * 2. Use input tensor shapes, dtypes, and any relevant arguments to compute the output shapes.
 * 3. Return empty tensors (e.g., at::empty_symint, at::empty_like) with the correct shape and
 * dtype.
 * 4. Do NOT perform any real computation or data movement.
 * 5. Register the meta implementation with the "Meta" dispatch key using TORCH_LIBRARY_IMPL or
 * similar.
 *
 * Example:
 *   std::tuple<at::Tensor, at::Tensor> my_op_meta(
 *       at::Tensor &input, int64_t some_param) {
 *     // Infer output shape based on input and parameters
 *     auto out_shape = ...;
 *     at::Tensor out = at::empty_symint(out_shape, input.options());
 *     // Return empty tensor(s) with correct shape/dtype
 *     return {out, ...};
 *   }
 *
 * See below for real examples.
 */

namespace ucm {
namespace meta {
const int64_t INT4_NUMS_IN_INT32 = 8;

at::Tensor npu_reshape_and_cache_bnsd_meta(const at::Tensor& hashq, const at::Tensor& hashkCache,
                                           const at::Tensor& slotMapping, const at::Tensor& seqLen,
                                           const at::Tensor& hashkCacheOut)
{
    at::Tensor output =
        at::empty(hashkCache.sizes(),
                  hashkCache.options().dtype(hashkCache.dtype()).device(hashkCache.device()));
    return output;
}

at::Tensor npu_hamming_dist_top_k_meta(
    const at::Tensor& hashq, const at::Tensor& hashkCache, const at::Tensor& hashkCacheRope,
    const at::Tensor& topN, const at::Tensor& seqLen, const c10::optional<at::Tensor>& chunkSize,
    const c10::optional<int64_t> maxSeqLen, const c10::optional<int64_t> sink,
    const c10::optional<int64_t> recent, const c10::optional<int64_t> supportOffload,
    const c10::optional<at::Tensor>& blockTable, const c10::optional<at::Tensor>& mask,
    const c10::optional<at::Tensor>& indices)
{
    if (indices.has_value()) { return at::empty_like(indices.value()); }

    auto n_bs = hashq.size(0);
    auto n_kv_heads = hashkCache.size(1);
    auto n_max_kv = 512;  // 设置和hamming_dist_top_k算子实现一致
    at::Tensor out = at::empty({n_bs, n_kv_heads, n_max_kv},
                               torch::TensorOptions().dtype(torch::kInt32).device(hashq.device()));
    return out;
}
}  // namespace meta
}  // namespace ucm

namespace {
// Register the meta implementations of the custom kernels for symbolic tracing, this will also
// the custom kernel been captured into aclgraph
TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _ucm), Meta, ops)
{
    ops.impl("npu_hamming_dist_top_k", &ucm::meta::npu_hamming_dist_top_k_meta);
    // reshape_and_cache_bnsd
    ops.impl("npu_reshape_and_cache_bnsd", &ucm::meta::npu_reshape_and_cache_bnsd_meta);
}
}  // namespace
