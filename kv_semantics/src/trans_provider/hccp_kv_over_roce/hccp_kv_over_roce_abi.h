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
 * The library must export Init, CreateConnection, DeleteConnections and
 * RegisterMemory. All other symbols are optional: when present the provider
 * forwards the matching TransProvider operation, otherwise the provider keeps
 * the documented fallback. Init carries the runtime configuration explicitly;
 * the library does not read config files or environment variables.
 *
 * Memory registration is descriptor based: RegisterMemory only registers the MR
 * itself and reports one HccpKvRoceMemoryInfo per descriptor. The provider owns
 * the MR handle space and keeps a one-to-one mapping between its own handle and
 * the reported memory info, so no handle ever crosses this ABI.
 *
 * Connection creation is QPC based: the provider allocates one HccpKvRoceQpc per
 * QP, hands its address to the library through connectionHandles and the
 * library fills the QPC in place. The library also returns one manager for the
 * whole IP pair; the provider keeps that manager in its connection record and
 * uses it for DeleteConnections and GetServerCapabilities. After creation the
 * provider copies each QPC into its HBM connection slot and passes that slot
 * address back during Send, so the library reads the QPC from the slot instead
 * of keeping a host-side handle.
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
#define HCCP_KV_ROCE_INIT_CONFIG_VERSION 1u

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

typedef enum HccpKvRoceOobTransport {
    HCCP_KV_ROCE_OOB_TCP = 0,
    HCCP_KV_ROCE_OOB_RASOCKET = 1,
} HccpKvRoceOobTransport;

/* Init: provider-supplied runtime configuration. The provider must fill every
 * field (zero-initialize then assign); structSize/version allow future
 * extension. The library does not read config files or environment variables.
 */
typedef struct HccpKvRoceInitConfig {
    uint32_t structSize; /* sizeof(HccpKvRoceInitConfig) */
    uint32_t version;    /* HCCP_KV_ROCE_INIT_CONFIG_VERSION */

    uint64_t memSize;    /* 671088640 */
    uint64_t sendMrSize; /* 0 */
    uint64_t flagMrSize; /* 0 */

    uint32_t oobTransport;       /* HccpKvRoceOobTransport */
    uint32_t sdkIoqDepth;        /* 128 */
    uint32_t timeoutMs;          /* 60000 */
    uint32_t kvnsid;             /* 0 */
    uint32_t serverBatchLimit;   /* 32 */
    uint32_t serverDeleteLimit;  /* 254 */
    uint32_t serverExistLimit;   /* 256 */
    uint32_t maxInflightKvPerIp; /* 2048 */
    uint32_t maxInflightKvPerQp; /* 256 */
    uint32_t sendAivCount;       /* 1 */
    uint32_t waitAivCount;       /* 1 */
    uint32_t cacheCount;         /* 4 */
    uint32_t stripeCount;        /* 64 */
    uint32_t maxFlagsPerStripe;  /* 20480 */
    uint32_t npuCachelineSize;   /* 256 */
    uint32_t quietInterval;      /* 8 */
    uint32_t sendCqWait;         /* 1 */
    uint32_t virtualNodes;       /* 5 */
    uint32_t postConnectDelayMs; /* 0 */
    uint32_t waitPollYieldUs;    /* 0 */
    uint32_t waitTaskTimeoutMs;  /* 2000 */
    uint32_t maxAivRetry;        /* 3 */
    uint32_t iterations;         /* 1 */
    uint32_t totalKvCount;       /* 4096 */
    uint32_t blockSize;          /* 131072 */
    uint32_t opType;             /* read */
    uint32_t pipelineNumLayers;  /* 64 */
    uint32_t compDelayUs;        /* 400000 */
    uint32_t mtNumThreads;       /* 4 */
    uint32_t mtOpsPerThread;     /* 10 */
    uint32_t debug;              /* 0 */

    uint8_t negMtu;          /* 5 */
    uint8_t negSl;           /* 0 */
    uint8_t negTrafficClass; /* 3 */
    uint8_t negRnrTimer;     /* 12 */
    uint8_t negRnrRetryCnt;  /* 7 */
    uint8_t negTimeout;      /* 14 */
    uint8_t negRetryCnt;     /* 7 */
    uint8_t negQpRdAtom;     /* 16 */
    uint32_t negStartPsn;    /* 0 */

    uint32_t reserved[8];
} HccpKvRoceInitConfig;

typedef struct HccpKvRoceRegisterMemoryDesc {
    int32_t memoryType;
    uint64_t addr;
    uint64_t size;
    uint64_t localAddr;
} HccpKvRoceRegisterMemoryDesc;

/* One registered MR as reported by the external library. rkey is the token id
 * the client uses to address the region. */
typedef struct HccpKvRoceMemoryInfo {
    uint64_t addr;
    uint64_t size;
    uint32_t access;
    uint32_t rkey;
} HccpKvRoceMemoryInfo;

/* Provider-owned QP context payload. The provider allocates one entry per
 * connection and passes its address through connectionHandles; the library
 * fills the struct in place. No allocation happens on the library side.
 * This size is an ABI contract with the library: the provider copies exactly
 * HCCP_KV_ROCE_QPC_BYTES bytes into one HBM slot, so the slot size and this
 * value must stay equal. */
#define HCCP_KV_ROCE_QPC_BYTES 512u

typedef struct HccpKvRoceQpc {
    uint8_t data[HCCP_KV_ROCE_QPC_BYTES];
} HccpKvRoceQpc;

typedef struct HccpKvRoceSendIoBatch {
    /* HBM slot that holds the QPC of this QP. */
    void* slotAddress;
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

/*
 * Required symbols
 * Init Must be called before the first CreateConnection. The first successful
 * call installs the process wide config; later calls return OK without changing
 * it. CreateConnection returns HCCP_KV_ROCE_INVALID_ARGUMENT before init.
 */
int32_t Init(const HccpKvRoceInitConfig* config);

/* Required symbols.
 * CreateConnection: connectionHandles is in/out. The caller pre-fills qpNum
 * entries with provider-owned HccpKvRoceQpc addresses; the library fills the
 * QPC payloads and leaves the pointers valid. connectionHandleNum reports how
 * many connections were actually created and must never exceed qpNum.
 * connectionManager receives the library-owned manager for this IP pair. One
 * manager owns every QP created by this call and is the only handle accepted by
 * DeleteConnections and GetServerCapabilities.
 * RegisterMemory must fill descNum writable memoryInfos entries. */
int32_t CreateConnection(const char* localIp, const char* remoteIp, uint32_t port, uint32_t qpNum,
                         uint32_t timeout, void** connectionHandles, uint32_t* connectionHandleNum,
                         void** connectionManager);

/* Deletes whole IP-pair connection sets. connectionManagers holds managers
 * returned by CreateConnection; the library releases every QP owned by each
 * manager, so callers must delete a manager as a unit. On a nonzero return the
 * managers are not consumed: each one remains valid and may be passed to a
 * later DeleteConnections call. */
int32_t DeleteConnections(void* const* connectionManagers, uint32_t managerNum);
int32_t RegisterMemory(const HccpKvRoceRegisterMemoryDesc* descs, uint32_t descNum,
                       HccpKvRoceMemoryInfo* memoryInfos);

/* Optional symbols.
 * UnregisterMemory deregisters MRs identified by the memory info reported by
 * RegisterMemory; the provider owns no cross-ABI handles.
 * Send fills one statuses entry per input batch. Thread bookkeeping is not part
 * of this ABI: the operator owns no per-thread resources, so the provider
 * always reports success without calling the library. */
uint32_t GetAbiVersion(void);
int32_t UnregisterMemory(const HccpKvRoceMemoryInfo* memoryInfos, uint32_t infoNum);
int32_t Send(const HccpKvRoceSendIoBatch* batches, uint32_t batchNum, uint32_t kernelCount,
             uint32_t quietCount, int32_t* statuses);
int32_t GetServerCapabilities(void* connectionManager, HccpKvRoceServerCapabilities* capabilities);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // KV_SEMANTICS_HCCP_KV_OVER_ROCE_ABI_H
