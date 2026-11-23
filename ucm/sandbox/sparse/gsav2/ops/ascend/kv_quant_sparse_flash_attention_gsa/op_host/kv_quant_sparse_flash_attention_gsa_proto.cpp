/*!
 * \file kv_quant_sparse_flash_attention_gsa_proto.cpp
 * \brief GSA 版 InferShape / InferDataType，含 softmax_max / softmax_sum 可选输出。
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 4;  // 与 v1 def 属性顺序一致
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 12;
constexpr uint32_t RETURN_SOFTMAX_LSE_ATTR_INDEX = 13;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;

ge::graphStatus InferShapeKvQuantSparseFlashAttentionGsa(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("KvQuantSparseFlashAttentionGsa", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(LAYOUT_QUERY_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    const int64_t ropeHeadDim = *attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);

    // attention_out: 与 query 同 shape，最后一维减去 ropeHeadDim
    *attentionOutShape = *queryShape;
    if (inputLayoutQueryPtrStr == "BSND") {
        attentionOutShape->SetDimNum(DIM_NUM_4);
        attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
        attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
        attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2));
        if (queryShape->GetDim(DIM_INDEX_3) != -1) {
            attentionOutShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_3) - ropeHeadDim);
        }
    } else {  // TND
        attentionOutShape->SetDimNum(DIM_NUM_3);
        attentionOutShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
        attentionOutShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_1));
        if (queryShape->GetDim(DIM_INDEX_2) != -1) {
            attentionOutShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_2) - ropeHeadDim);
        }
    }

    // softmax_max / softmax_sum: 仅 return_softmax_lse=true 时有效
    // shape: BSND -> [B, KV_N, Q_S, Q_N/KV_N]; TND -> [KV_N, Q_T, Q_N/KV_N]
    // 当 return_softmax_lse=false 时返回空占位 (0,)
    gert::Shape *softmaxMaxShape = context->GetOutputShape(1);
    gert::Shape *softmaxSumShape = context->GetOutputShape(2);
    const bool returnLse = *attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);
    if (returnLse && softmaxMaxShape != nullptr && softmaxSumShape != nullptr) {
        // 精确 shape 由 tiling 阶段按 actual_seq_len 计算，这里给一个基于 query 的保守上界
        // 实际运行时由 kernel 按实际有效长度填充
        if (inputLayoutQueryPtrStr == "BSND") {
            softmaxMaxShape->SetDimNum(DIM_NUM_4);
            softmaxMaxShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));  // B
            softmaxMaxShape->SetDim(DIM_INDEX_1, 1);  // KV_N (运行时由 tiling 修正)
            softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));  // Q_S
            softmaxMaxShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2));  // Q_N
            *softmaxSumShape = *softmaxMaxShape;
        } else {  // TND
            softmaxMaxShape->SetDimNum(DIM_NUM_3);
            softmaxMaxShape->SetDim(DIM_INDEX_0, 1);                                // KV_N
            softmaxMaxShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));  // Q_T
            softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));  // Q_N
            *softmaxSumShape = *softmaxMaxShape;
        }
    } else {
        if (softmaxMaxShape != nullptr) { softmaxMaxShape->SetDimNum(0); }
        if (softmaxSumShape != nullptr) { softmaxSumShape->SetDimNum(0); }
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvQuantSparseFlashAttentionGsa(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("KvQuantSparseFlashAttentionGsa", "InferDataTypeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    // softmax_max / softmax_sum 始终 FP32
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvQuantSparseFlashAttentionGsa)
    .InferShape(InferShapeKvQuantSparseFlashAttentionGsa)
    .InferDataType(InferDataTypeKvQuantSparseFlashAttentionGsa);
}  // namespace ops
