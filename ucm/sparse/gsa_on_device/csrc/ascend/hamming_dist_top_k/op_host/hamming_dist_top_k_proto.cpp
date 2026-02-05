/*!
 * \file hamming_dist_top_k_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeHammingDistTopK(gert::InferShapeContext *context)
{
    gert::Shape *outShape = context->GetOutputShape(0);
    const gert::Shape *inputShape = context->GetInputShape(6);
    *outShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeHammingDistTopK(gert::InferDataTypeContext *context)
{
    ge::DataType outputType = context->GetInputDataType(ge::DT_INT32);
    context->SetOutputDataType(0, outputType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HammingDistTopK)
    .InferShape(InferShapeHammingDistTopK)
    .InferDataType(InferDataTypeHammingDistTopK);
}  // namespace ops