// SPDX-License-Identifier: Apache-2.0
// GSA ops Meta 实现（用于 aclgraph 捕获 / tracing）
// 所有算子的 Meta impl，确保 torch.ops._C_gsa.xxx 在 tracing/aclgraph 下可用

#include <torch/extension.h>
#include <torch/library.h>
#include "utils.h"

namespace gsa {

// 轻量算子 Meta：不实际执行，仅做 shape 校验/透传
void meta_gsa_scatter_copy(at::Tensor& dst, const at::Tensor& src, const at::Tensor& index) {}
void meta_gsa_gather_copy(at::Tensor& dst, const at::Tensor& src, const at::Tensor& index) {}
void meta_indexer_gather_qw(const at::Tensor& q_src, const at::Tensor& w_src, at::Tensor& q_dst,
                            at::Tensor& w_dst, const at::Tensor& q_index, const at::Tensor& w_index)
{
}

at::Tensor meta_gsa_topk_expand_tmp(at::Tensor& topk_indices, at::Tensor& swap_topk,
                                    at::Tensor& swap_index, at::Tensor& tmp_tensor,
                                    std::vector<int64_t> pad, int64_t padValue, int64_t expand_len)
{
    // 返回 view，shape = [topk_rows, 1, expand_len]
    return tmp_tensor.view({tmp_tensor.size(0), 1, expand_len}).slice(0, 0, topk_indices.size(0));
}

// 自定义 AscendC 算子 Meta
std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_kv_quant_sparse_flash_attention_gsa(
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

    bool returnLse = returnSoftmaxLse.value_or(false);
    at::Tensor softmaxMax = at::empty({0}, query.options().dtype(at::kFloat));
    at::Tensor softmaxSum = at::empty({0}, query.options().dtype(at::kFloat));
    if (returnLse) {
        std::vector<int64_t> lseShape;
        if (std::string(layoutQuery) == "BSND") {
            lseShape = {queryShape[0], 1, queryShape[1], queryShape[2]};
        } else {
            lseShape = {1, queryShape[0], queryShape[1]};
        }
        softmaxMax = at::empty(lseShape, query.options().dtype(at::kFloat));
        softmaxSum = at::empty(lseShape, query.options().dtype(at::kFloat));
    }
    return std::make_tuple(attentionOut, softmaxMax, softmaxSum);
}

}  // namespace gsa

TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _gsa), Meta, ops)
{
    ops.impl("gsa_scatter_copy", &gsa::meta_gsa_scatter_copy);
    ops.impl("gsa_gather_copy", &gsa::meta_gsa_gather_copy);
    ops.impl("indexer_gather_qw", &gsa::meta_indexer_gather_qw);
    ops.impl("gsa_topk_expand_tmp", &gsa::meta_gsa_topk_expand_tmp);
    ops.impl("npu_kv_quant_sparse_flash_attention_gsa",
             &gsa::meta_kv_quant_sparse_flash_attention_gsa);
}
