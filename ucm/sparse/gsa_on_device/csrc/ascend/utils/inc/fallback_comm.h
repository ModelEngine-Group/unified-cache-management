// from vllm-ascend, see
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/utils/inc/fallback_comm.h

/*!
 * \file fallback_comm.h
 * \brief
 */

#ifndef INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_
#define INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_

#include "aclnn/aclnn_base.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/tensor.h"
#include "register/op_impl_registry.h"
#include "runtime/base.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {

aclDataType ToAclDataType(ge::DataType dtype);
}  // namespace fallback

#ifdef __cplusplus
}
#endif

#endif  // INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_
