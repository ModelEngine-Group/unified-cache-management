// from vllm-ascend, see
// https://github.com/vllm-project/vllm-ascend/blob/main/csrc/utils/inc/error/ops_error.h

/*!
 * \file ops_error.h
 * \brief
 */

#pragma once

#include "log/ops_log.h"

/* 基础报错 */
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E89999", OPS_DESC, __VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E69999", OPS_DESC, __VA_ARGS__)

/* 条件报错 */
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR) OPS_LOG_STUB_IF(COND, LOG_FUNC, EXPR)
