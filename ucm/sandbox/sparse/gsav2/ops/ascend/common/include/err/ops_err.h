/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use the License except in compliance with the License.
 * You can obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file ops_err.h
 * \brief 兼容头: CANN 9.x 安装包内 OP_LOGE_FOR_INVALID 系列与 OP_LOGI 等宏已收敛到
 *        include/op_common/log/log.h(其内部已含 base/err_msg.h 的 REPORT_INNER_ERR_MSG)，
 *        老代码中的 err/ops_err.h 由本兼容头提供。
 *        与 vllm-ascend / ops-transformer 的 common/include/err/ops_err.h 保持一致。
 */

#ifndef GSA_COMMON_OPS_ERR_H
#define GSA_COMMON_OPS_ERR_H

#include "log/log.h"

constexpr int32_t OP_TRANSFORMER_MODULE_ID = 63;

#define OPS_INNER_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)                                                           \
    do {                                                                                                               \
        OpLogSub(OP_TRANSFORMER_MODULE_ID, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__);                                  \
        REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ##__VA_ARGS__);                                                        \
    } while (0)


/* 基础报错 */
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E89999", OPS_DESC, __VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E69999", OPS_DESC, __VA_ARGS__)

#endif // GSA_COMMON_OPS_ERR_H
