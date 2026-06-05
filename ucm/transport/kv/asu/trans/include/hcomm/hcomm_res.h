/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCOMM_RES_H
#define HCOMM_RES_H
 
#include "hcomm_res_defs.h"
 
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

extern HcommResult HcommEndpointCreate(const EndpointDesc *endpoint, EndpointHandle *endpointHandle);

extern HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle);

extern HcommResult HcommMemReg(EndpointHandle endpointHandle, const char *memTag, const CommMem *mem, HcommMemHandle *memHandle);

extern HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle);

extern HcommResult HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void **memDesc, uint32_t *memDescLen);

extern HcommResult HcommMemImport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen, CommMem *outMem);

extern HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void *memDesc, uint32_t descLen);

extern HcommResult HcommChannelCreate(EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc *channelDescs,
    uint32_t channelNum, ChannelHandle *channels);

extern HcommResult HcommChannelGetStatus(const ChannelHandle *channelList, uint32_t listNum, int32_t *statusList);

extern HcommResult HcommChannelDestroy(const ChannelHandle *channels, uint32_t channelNum);

extern HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t *notifyNumPerThread, ThreadHandle *threads);

extern HcommResult HcommThreadFree(const ThreadHandle *threads, uint32_t threadNum);

extern HcommResult HcommChannelGetPtrByHandle(const ChannelHandle *channelList, uint32_t listNum, uint64_t *channelPtr);

extern HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t *port);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOMM_RES_H
