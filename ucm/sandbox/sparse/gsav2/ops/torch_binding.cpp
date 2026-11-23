// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
// GSA ops torch binding: 统一用 torch.ops._C_gsa 调用所有算子
//
// 包含两类算子（均注册到 torch.ops._C_gsa 命名空间）:
//   1. 调内置 aclnn 的轻量算子: gsa_scatter_copy / gsa_gather_copy / indexer_gather_qw /
//   gsa_topk_expand_tmp
//   2. 自定义 AscendC 算子: npu_kv_quant_sparse_flash_attention_gsa（需先编译 ascend/ 下的 kernel
//   并安装 .run）

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/npu/Module.h>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "utils.h"  // CONCAT / TORCH_LIBRARY_EXPAND / REGISTER_EXTENSION 宏

namespace gsa {

// ===========================================================================
// 第一类: 调内置 aclnn 的轻量算子
// ===========================================================================

void gsa_scatter_copy(at::Tensor& dst, const at::Tensor& src, const at::Tensor& index)
{
    if (index.numel() == 0) { return; }
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1, "dst must be on NPU");
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "src must be on NPU");
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1, "index must be on NPU");
    TORCH_CHECK(dst.dim() == 2, "dst must be 2D");
    TORCH_CHECK(src.dim() == 2, "src must be 2D");
    TORCH_CHECK(index.dim() == 2, "index must be 2D");
    TORCH_CHECK(dst.size(1) == src.size(1), "the second dimension of dst and src must be equal");

    int64_t dim = 0;
    int64_t reduce = 0;
    EXEC_NPU_CMD(aclnnInplaceScatter, dst, dim, index, src, reduce);
}

void gsa_gather_copy(at::Tensor& dst, const at::Tensor& src, const at::Tensor& index)
{
    if (index.numel() == 0) { return; }
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1, "dst must be on NPU");
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "src must be on NPU");
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1, "index must be on NPU");
    TORCH_CHECK(dst.dim() == 2, "dst must be 2D");
    TORCH_CHECK(src.dim() == 2, "src must be 2D");
    TORCH_CHECK(index.dim() == 2, "index must be 2D");
    TORCH_CHECK(dst.size(1) == src.size(1), "the second dimension of dst and src must be equal");

    int64_t dim = 0;
    EXEC_NPU_CMD(aclnnGather, src, dim, index, dst);
}

void indexer_gather_qw(const at::Tensor& q_src, const at::Tensor& w_src, at::Tensor& q_dst,
                       at::Tensor& w_dst, const at::Tensor& q_index, const at::Tensor& w_index)
{
    TORCH_CHECK(q_src.device().type() == c10::DeviceType::PrivateUse1, "dst must be on NPU");
    TORCH_CHECK(w_src.device().type() == c10::DeviceType::PrivateUse1, "src must be on NPU");
    TORCH_CHECK(q_dst.device().type() == c10::DeviceType::PrivateUse1, "dst must be on NPU");
    TORCH_CHECK(w_dst.device().type() == c10::DeviceType::PrivateUse1, "src must be on NPU");
    TORCH_CHECK(q_index.device().type() == c10::DeviceType::PrivateUse1, "index must be on NPU");
    TORCH_CHECK(w_index.device().type() == c10::DeviceType::PrivateUse1, "index must be on NPU");
    TORCH_CHECK(q_src.dim() == 2, "dst must be 2D");
    TORCH_CHECK(w_src.dim() == 2, "src must be 2D");
    TORCH_CHECK(q_dst.dim() == 2, "dst must be 2D");
    TORCH_CHECK(w_dst.dim() == 2, "src must be 2D");
    TORCH_CHECK(q_index.dim() == 2, "index must be 2D");
    TORCH_CHECK(w_index.dim() == 2, "index must be 2D");
    TORCH_CHECK(q_src.size(1) == q_dst.size(1),
                "the second dimension of dst and src must be equal");
    TORCH_CHECK(w_src.size(1) == w_dst.size(1),
                "the second dimension of dst and src must be equal");

    int64_t dim = 0;
    EXEC_NPU_CMD(aclnnGather, q_src, dim, q_index, q_dst);
    EXEC_NPU_CMD(aclnnGather, w_src, dim, w_index, w_dst);
}

at::Tensor gsa_topk_expand_tmp(at::Tensor& topk_indices, at::Tensor& swap_topk,
                               at::Tensor& swap_index, at::Tensor& tmp_tensor,
                               const at::IntArrayRef& pad, const at::Scalar padValue,
                               int64_t expand_len)
{
    TORCH_CHECK(topk_indices.device().type() == c10::DeviceType::PrivateUse1,
                "topk_indices must be on NPU");
    TORCH_CHECK(swap_topk.device().type() == c10::DeviceType::PrivateUse1,
                "swap_topk must be on NPU");
    TORCH_CHECK(swap_index.device().type() == c10::DeviceType::PrivateUse1,
                "swap_index must be on NPU");
    TORCH_CHECK(tmp_tensor.device().type() == c10::DeviceType::PrivateUse1,
                "tmp_tensor must be on NPU");
    TORCH_CHECK(swap_index.dtype() == torch::kInt64, "index must be torch.int64");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must be 2D");
    TORCH_CHECK(swap_topk.dim() == 2, "swap_topk must be 2D");
    TORCH_CHECK(swap_index.dim() == 2, "swap_index must be 2D");
    TORCH_CHECK(tmp_tensor.dim() == 2, "tmp_tensor must be 2D");
    TORCH_CHECK(tmp_tensor.is_contiguous(), "tmp_tensor must be contiguous");
    TORCH_CHECK(tmp_tensor.dtype() == topk_indices.dtype(),
                "tmp_tensor dtype must match topk_indices");
    TORCH_CHECK(expand_len >= topk_indices.size(1),
                "the second dimension of topk_indices must small than expand_len");
    TORCH_CHECK(tmp_tensor.size(0) >= topk_indices.size(0) + 2,
                "tmp_tensor rows must be no less than topk_indices rows + 2");
    TORCH_CHECK(tmp_tensor.size(1) == expand_len, "tmp_tensor cols must equal expand_len");

    constexpr int32_t DIM_0 = 0;
    int64_t dim = 0;
    int64_t reduce = 0;

    at::Tensor padTensor = tmp_tensor.slice(0, 0, topk_indices.size(DIM_0));
    EXEC_NPU_CMD(aclnnConstantPadNd, topk_indices, pad, padValue, padTensor);
    EXEC_NPU_CMD(aclnnInplaceScatter, tmp_tensor, dim, swap_index, swap_topk, reduce);
    return tmp_tensor.view({tmp_tensor.size(DIM_0), 1, expand_len})
        .slice(DIM_0, 0, topk_indices.size(DIM_0));
}

// IntArrayRef/Scalar 不能直接进 TorchScript def，用 pybind 桥接的 wrapper
at::Tensor gsa_topk_expand_tmp_wrapper(at::Tensor& topk_indices, at::Tensor& swap_topk,
                                       at::Tensor& swap_index, at::Tensor& tmp_tensor,
                                       std::vector<int64_t> pad, const int64_t padValue,
                                       int64_t expand_len)
{
    return gsa_topk_expand_tmp(topk_indices, swap_topk, swap_index, tmp_tensor,
                               at::IntArrayRef(pad), at::Scalar(padValue), expand_len);
}

// ===========================================================================
// 第二类: 自定义 AscendC 算子（需先编译 ascend/kv_quant_sparse_flash_attention_gsa 并安装 .run）
// aclnnKvQuantSparseFlashAttentionGsa 接口由 .run 包安装后提供
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_kv_quant_sparse_flash_attention_gsa(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& sparseIndices, double scaleValue, int64_t keyQuantMode,
    int64_t valueQuantMode, const c10::optional<at::Tensor>& keyDequantScale,
    const c10::optional<at::Tensor>& valueDequantScale, const c10::optional<at::Tensor>& blockTable,
    const c10::optional<at::Tensor>& actualSeqLengthsQuery,
    const c10::optional<at::Tensor>& actualSeqLengthsKV, const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& actualSparseLengths, c10::optional<int64_t> sparseBlockSize,
    c10::string_view layoutQuery, c10::string_view layoutKV, c10::optional<int64_t> sparseMode,
    c10::optional<int64_t> preTokens, c10::optional<int64_t> nextTokens,
    c10::optional<int64_t> attentionMode, c10::optional<int64_t> quantScaleRepoMode,
    c10::optional<int64_t> tileSize, c10::optional<int64_t> ropeHeadDim,
    c10::optional<bool> returnSoftmaxLse)
{
    auto queryShape = query.sizes();
    int64_t ropeDim = ropeHeadDim.value_or(64);
    std::vector<int64_t> outShape(queryShape.begin(), queryShape.end());
    outShape.back() = queryShape.back() - ropeDim;
    at::Tensor attentionOut = at::empty(outShape, query.options());

    at::Tensor softmaxMax = at::empty({0}, query.options().dtype(at::kFloat));
    at::Tensor softmaxSum = at::empty({0}, query.options().dtype(at::kFloat));

    auto&& sparseBlockSize_ = sparseBlockSize.value_or(1);
    auto&& sparseMode_ = sparseMode.value_or(3);
    auto&& preTokens_ = preTokens.value_or(INT64_MAX);
    auto&& nextTokens_ = nextTokens.value_or(INT64_MAX);
    auto&& attentionMode_ = attentionMode.value_or(0);
    auto&& quantScaleRepoMode_ = quantScaleRepoMode.value_or(1);
    auto&& tileSize_ = tileSize.value_or(128);
    auto&& returnSoftmaxLse_ = returnSoftmaxLse.value_or(false);

    // CovertTypes
    std::string layoutQueryStr(layoutQuery);
    std::string layoutKVStr(layoutKV);

    EXEC_NPU_CMD(aclnnKvQuantSparseFlashAttentionGsa, query, key, value, sparseIndices,
                 keyDequantScale, valueDequantScale, blockTable, actualSeqLengthsQuery,
                 actualSeqLengthsKV, sinks, actualSparseLengths, scaleValue, keyQuantMode,
                 valueQuantMode, sparseBlockSize_, layoutQueryStr, layoutKVStr, sparseMode_,
                 preTokens_, nextTokens_, attentionMode_, quantScaleRepoMode_, tileSize_, ropeDim,
                 returnSoftmaxLse_, attentionOut, softmaxMax, softmaxSum);
    return std::make_tuple(attentionOut, softmaxMax, softmaxSum);
}

}  // namespace gsa

// ===========================================================================
// TORCH_LIBRARY 统一注册到 torch.ops._C_gsa
// 所有算子（轻量 + 自定义 AscendC）都通过 torch.ops._C_gsa.xxx 调用
// ===========================================================================
TORCH_LIBRARY_EXPAND(CONCAT(_C, _gsa), ops)
{
    // --- 轻量算子（调内置 aclnn） ---
    ops.def("gsa_scatter_copy(Tensor(a!) dst, Tensor src, Tensor index) -> ()");
    ops.impl("gsa_scatter_copy", torch::kPrivateUse1, &gsa::gsa_scatter_copy);

    ops.def("gsa_gather_copy(Tensor(a!) dst, Tensor src, Tensor index) -> ()");
    ops.impl("gsa_gather_copy", torch::kPrivateUse1, &gsa::gsa_gather_copy);

    ops.def(
        "indexer_gather_qw(Tensor q_src, Tensor w_src, Tensor(a!) q_dst, "
        "Tensor(a!) w_dst, Tensor q_index, Tensor w_index) -> ()");
    ops.impl("indexer_gather_qw", torch::kPrivateUse1, &gsa::indexer_gather_qw);

    // gsa_topk_expand_tmp: pad 是 IntArrayRef，用 SymInt list 表达；padValue 用 int
    // 返回 view 的 tensor
    ops.def(
        "gsa_topk_expand_tmp(Tensor topk_indices, Tensor swap_topk, Tensor swap_index, "
        "Tensor tmp_tensor, int[] pad, int pad_value, int expand_len) -> Tensor");
    ops.impl("gsa_topk_expand_tmp", torch::kPrivateUse1, &gsa::gsa_topk_expand_tmp_wrapper);

    // --- 自定义 AscendC 算子 ---
    ops.def(
        "npu_kv_quant_sparse_flash_attention_gsa("
        "Tensor query, Tensor key, Tensor value, Tensor sparse_indices, "
        "float scale_value, int key_quant_mode, int value_quant_mode, "
        "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, "
        "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, "
        "Tensor? actual_seq_lengths_kv=None, Tensor? sinks=None, "
        "Tensor? actual_sparse_lengths=None, "
        "int? sparse_block_size=1, str layout_query=\"BSND\", str layout_kv=\"BSND\", "
        "int? sparse_mode=3, int? pre_tokens=9223372036854775807, "
        "int? next_tokens=9223372036854775807, int? attention_mode=0, "
        "int? quant_scale_repo_mode=1, int? tile_size=128, int? rope_head_dim=64, "
        "bool? return_softmax_lse=False) -> (Tensor, Tensor, Tensor)");
    ops.impl("npu_kv_quant_sparse_flash_attention_gsa", torch::kPrivateUse1,
             &gsa::npu_kv_quant_sparse_flash_attention_gsa);
}

// 导出 PyInit_gsa_ops，使 "import gsa_ops" 时 TORCH_LIBRARY 静态初始化器运行
REGISTER_EXTENSION(gsa_ops)
