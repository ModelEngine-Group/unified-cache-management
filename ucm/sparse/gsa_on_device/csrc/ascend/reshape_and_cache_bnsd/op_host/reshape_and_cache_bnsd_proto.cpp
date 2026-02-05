
/*!
 * \file reshape_and_cache_bnsd_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeReshapeAndCacheBnsd(gert::InferShapeContext* context)
{
    gert::Shape* outShape = context->GetOutputShape(0);
    const gert::Shape* inputShape = context->GetInputShape(1);
    *outShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeReshapeAndCacheBnsd(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(1);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReshapeAndCacheBnsd)
    .InferShape(InferShapeReshapeAndCacheBnsd)
    .InferDataType(InferDataTypeReshapeAndCacheBnsd);
}  // namespace ops