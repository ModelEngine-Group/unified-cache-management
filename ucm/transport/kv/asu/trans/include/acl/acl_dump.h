/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INC_EXTERNAL_ACL_DUMP_H_
#define INC_EXTERNAL_ACL_DUMP_H_

#include <stdint.h>
#include "acl_base.h"

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define ACL_DUMP_API __declspec(dllexport)
#else
#define ACL_DUMP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ACL_DUMP_MAX_FILE_PATH_LENGTH    4096
typedef struct acldumpChunk  {
    char       fileName[ACL_DUMP_MAX_FILE_PATH_LENGTH];   // file name, absolute path
    uint32_t   bufLen;                                    // dataBuf length
    uint32_t   isLastChunk;                               // is last chunk. 0: not 1: yes
    int64_t    offset;                                    // Offset in file. -1: append write
    int32_t    flag;                                      // flag
    uint8_t    dataBuf[0];                                // data buffer
} acldumpChunk;

ACL_DUMP_API aclError acldumpRegCallback(int32_t (* const messageCallback)(const acldumpChunk *, int32_t),
    int32_t flag);
ACL_DUMP_API void acldumpUnregCallback();

#define ACL_OP_DUMP_OP_AICORE_ARGS 0x00000001U

/**
 * @ingroup AscendCL
 * @brief Enable the dump function of the corresponding dump type.
 *
 * @param dumpType [IN]       type of dump
 * @param path     [IN]       dump path
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopStartDumpArgs(uint32_t dumpType, const char *path);

/**
 * @ingroup AscendCL
 * @brief Disable the dump function of the corresponding dump type.
 *
 * @param dumpType [IN]       type of dump
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
ACL_FUNC_VISIBILITY aclError aclopStopDumpArgs(uint32_t dumpType);

typedef enum acldumpType {
    AIC_ERR_BRIEF_DUMP = 1,         // lite exception dump
    AIC_ERR_NORM_DUMP = 2,          // normal exception dump
    AIC_ERR_DETAIL_DUMP = 3,        // coredump mode
    DATA_DUMP = 4,                  // tensor data dump
    OVERFLOW_DUMP = 5               // overflow dump
} acldumpType;

/**
 * @ingroup AscendCL
 * @brief Get dump path.
 *
 * @param dumpType [IN]   type of dump path
 *
 * @retval path for success
 * @retval NULL for failed
 */
ACL_FUNC_VISIBILITY const char* acldumpGetPath(acldumpType dumpType);

#ifdef __cplusplus
}
#endif

#endif
