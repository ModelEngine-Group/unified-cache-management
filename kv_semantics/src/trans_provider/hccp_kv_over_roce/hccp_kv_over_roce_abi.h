/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */

/**
 * C ABI exported by the external HCCP KV-over-RoCE library loaded at runtime by
 * HccpKvOverRoceTransProvider.
 *
 * The library must export CreateConnection, DeleteConnections and RegisterMemory.
 * All other symbols are optional: when present the provider forwards the matching
 * TransProvider operation, otherwise the provider keeps the documented fallback.
 *
 * Every function returns 0 on success and a non-zero HccpKvRoceReturnCode (or an
 * implementation-defined positive value) on failure. Strings are passed as
 * NUL-terminated const char* instead of C++ std::string to keep the ABI stable.
 */
#ifndef KV_SEMANTICS_HCCP_KV_OVER_ROCE_ABI_H
#define KV_SEMANTICS_HCCP_KV_OVER_ROCE_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HCCP_KV_ROCE_ABI_VERSION 1u

typedef enum HccpKvRoceReturnCode {
    HCCP_KV_ROCE_OK = 0,
    HCCP_KV_ROCE_INVALID_ARGUMENT = 1,
    HCCP_KV_ROCE_UNSUPPORTED = 2,
    HCCP_KV_ROCE_RESOURCE_BUSY = 3,
    HCCP_KV_ROCE_CONNECTION_ERROR = 4,
    HCCP_KV_ROCE_IO_ERROR = 5,
    HCCP_KV_ROCE_BUFFER_NOT_REGISTERED = 6,
    HCCP_KV_ROCE_INTERNAL_ERROR = 7,
} HccpKvRoceReturnCode;

typedef enum HccpKvRoceMemoryType {
    HCCP_KV_ROCE_MEM_HOST = 0,
    HCCP_KV_ROCE_MEM_DEVICE = 1,
} HccpKvRoceMemoryType;

typedef struct HccpKvRoceRegisterMemoryDesc {
    int32_t memoryType;
    uint64_t addr;
    uint64_t size;
    uint64_t localAddr;
} HccpKvRoceRegisterMemoryDesc;

typedef struct HccpKvRoceBindMemoryDesc {
    int32_t memoryType;
    uint64_t addr;
    uint64_t size;
    uint32_t tokenId;
} HccpKvRoceBindMemoryDesc;

typedef struct HccpKvRoceSendIoBatch {
    void* connectionHandle;
    void* sendBuffer;
    void* flagBuffer;
    uint64_t len;
} HccpKvRoceSendIoBatch;

typedef struct HccpKvRoceServerCapabilities {
    uint32_t queueNum;
    uint32_t ioQueueDepth;
    uint32_t ioQueueKeyConcurrency;
    uint32_t connectionKeyConcurrency;
    uint64_t singleValueMaxBytes;
    uint64_t batchValueMaxBytes;
    uint32_t batchStoreKeys;
    uint32_t batchLoadKeys;
    uint32_t deleteKeys;
    uint32_t queryKeys;
    uint32_t keyLength;
    uint32_t kvCapabilities;
} HccpKvRoceServerCapabilities;

/* Required symbols. connectionHandles must point to qpNum writable slots;
 * RegisterMemory must fill descNum writable mrHandles. */
int32_t CreateConnection(const char* localIp, const char* remoteIp, uint32_t port, uint32_t qpNum,
                         uint32_t timeout, void** connectionHandles);
int32_t DeleteConnections(void* const* connectionHandles, uint32_t handleNum);
int32_t RegisterMemory(const HccpKvRoceRegisterMemoryDesc* descs, uint32_t descNum,
                       uint64_t* mrHandles);

/* Optional symbols.
 * BindMemory returns provider-local handles for an existing registration; those
 * handles must be released through UnregisterMemory.
 * Send and FreeThread fill one statuses entry per input batch/thread. */
uint32_t GetAbiVersion(void);
int32_t BindMemory(const HccpKvRoceBindMemoryDesc* descs, uint32_t descNum, uint64_t* mrHandles);
int32_t UnregisterMemory(const uint64_t* mrHandles, uint32_t handleNum);
int32_t Send(const HccpKvRoceSendIoBatch* batches, uint32_t batchNum, uint32_t kernelCount,
             uint32_t quietCount, int32_t* statuses);
int32_t AllocThread(uint32_t threadNum, const uint32_t* notifyNumPerThread, void** threads);
int32_t FreeThread(void* const* threads, uint32_t threadNum, int32_t* statuses);
int32_t GetMemTokenId(uint64_t mrHandle, uint32_t* tokenId);
int32_t GetServerCapabilities(void* connectionHandle, HccpKvRoceServerCapabilities* capabilities);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // KV_SEMANTICS_HCCP_KV_OVER_ROCE_ABI_H
