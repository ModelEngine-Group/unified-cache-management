// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
// Modified from
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/torch_binding.cpp

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

namespace gsa {

void gsa_scatter_copy(at::Tensor& dst, const at::Tensor& src, const at::Tensor& index)
{
    // 如果 index 为空（无数据拷贝），直接返回
    if (index.numel() == 0) { return; }
    // ---------- 1. 输入合法性检查 ----------
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1, "dst must be on NPU");
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "src must be on NPU");
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1, "index must be on NPU");

    // TORCH_CHECK(index.dtype() == torch::kInt64, "index must be torch.int64");

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
    // 如果 index 为空（无数据拷贝），直接返回
    if (index.numel() == 0) { return; }
    // ---------- 1. 输入合法性检查 ----------
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1, "dst must be on NPU");
    TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1, "src must be on NPU");
    TORCH_CHECK(index.device().type() == c10::DeviceType::PrivateUse1, "index must be on NPU");

    // TORCH_CHECK(index.dtype() == torch::kInt64, "index must be torch.int64");

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
    // ---------- 1. 输入合法性检查 ----------
    TORCH_CHECK(q_src.device().type() == c10::DeviceType::PrivateUse1, "dst must be on NPU");
    TORCH_CHECK(w_src.device().type() == c10::DeviceType::PrivateUse1, "src must be on NPU");
    TORCH_CHECK(q_dst.device().type() == c10::DeviceType::PrivateUse1, "dst must be on NPU");
    TORCH_CHECK(w_dst.device().type() == c10::DeviceType::PrivateUse1, "src must be on NPU");
    TORCH_CHECK(q_index.device().type() == c10::DeviceType::PrivateUse1, "index must be on NPU");
    TORCH_CHECK(w_index.device().type() == c10::DeviceType::PrivateUse1, "index must be on NPU");

    // TORCH_CHECK(q_index.dtype() == torch::kInt64, "index must be torch.int64");
    // TORCH_CHECK(w_index.dtype() == torch::kInt64, "index must be torch.int64");

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
    // ---------- 1. 输入合法性检查 ----------
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
    constexpr int32_t DIM_1 = 1;

    int64_t dim = 0;
    int64_t reduce = 0;

    at::Tensor padTensor = tmp_tensor.slice(0, 0, topk_indices.size(DIM_0));
    EXEC_NPU_CMD(aclnnConstantPadNd, topk_indices, pad, padValue, padTensor);

    // ③ scatter 覆盖 swap 行
    EXEC_NPU_CMD(aclnnInplaceScatter, tmp_tensor, dim, swap_index, swap_topk, reduce);
    return tmp_tensor.view({tmp_tensor.size(DIM_0), 1, expand_len})
        .slice(DIM_0, 0, topk_indices.size(DIM_0));
}

}  // namespace gsa

at::Tensor gsa_topk_expand_tmp_pybind(at::Tensor& topk_indices, at::Tensor& swap_topk,
                                      at::Tensor& swap_index, at::Tensor& tmp_tensor,
                                      std::vector<int64_t> pad, const int64_t padValue,
                                      int64_t expand_len)
{
    return gsa::gsa_topk_expand_tmp(topk_indices, swap_topk, swap_index, tmp_tensor,
                                    at::IntArrayRef(pad), at::Scalar(padValue), expand_len);
}

PYBIND11_MODULE(TORCH_EXTENSION, m)
{
    m.def("gsa_scatter_copy", &gsa::gsa_scatter_copy, "GSA scatter copy");
    m.def("gsa_gather_copy", &gsa::gsa_gather_copy, "GSA gather copy");
    m.def("indexer_gather_qw", &gsa::indexer_gather_qw, "GSA indexer gather q/w");
    m.def("gsa_topk_expand_tmp", &gsa::gsa_topk_expand_tmp, "GSA topk expand into tmp");
}
