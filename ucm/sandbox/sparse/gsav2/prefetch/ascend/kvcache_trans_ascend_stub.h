/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#ifndef KV_CACHE_TRANS_ASCEND_STUB_H
#define KV_CACHE_TRANS_ASCEND_STUB_H

/*
 * Test-only stub for the Ascend ACL SDK.  Pulled in automatically by
 * kvcache_trans_ascend.h when KV_CACHE_PRE_TEST_PRIVATE_AS_PUBLIC is defined,
 * so the prefetch unit/module/end-to-end tests can be compiled and executed on
 * a plain Linux host without an Ascend NPU or CANN toolkit installed.
 *
 * The stub mirrors the small subset of ACL symbols consumed by the prefetch
 * source tree and additionally exposes a `StubControl` struct with injectable
 * failure switches so exception/early-return branches in the production code can
 * be exercised deterministically.
 */

#include <ATen/Tensor.h>
#include <atomic>
#include <c10/util/SmallVector.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include "kvcache_log.h"

namespace acl_stub {

/* Injectable failure switches.  Reset to defaults by calling Reset(). */
struct StubControl {
    std::atomic<bool> setDeviceFails{false};
    std::atomic<bool> createContextFails{false};
    std::atomic<bool> createStreamFails{false};
    std::atomic<bool> getCurrentContextFails{false};
    std::atomic<bool> setCurrentContextFails{false};
    std::atomic<bool> createStreamStatusFails{false};
    std::atomic<bool> memcpyAsyncFails{false};
    std::atomic<bool> memcpyFails{false};
    std::atomic<bool> syncStreamFails{false};
    std::atomic<bool> streamQueryFails{false};
    std::atomic<bool> destroyStreamFails{false};
    std::atomic<bool> mallocFails{false};
    std::atomic<bool> freeFails{false};
    std::atomic<bool> inplaceScatterWsfFails{false};
    std::atomic<bool> inplaceScatterFails{false};
    std::atomic<bool> createTensorFails{false};
    std::atomic<bool> destroyTensorFails{false};
    std::atomic<bool> returnNullErrMsg{false};

    /* Call counters for assertions in tests. */
    std::atomic<int> setDeviceCount{0};
    std::atomic<int> createContextCount{0};
    std::atomic<int> createStreamCount{0};
    std::atomic<int> getCurrentContextCount{0};
    std::atomic<int> setCurrentContextCount{0};
    std::atomic<int> memcpyAsyncCount{0};
    std::atomic<int> memcpyCount{0};
    std::atomic<int> syncStreamCount{0};
    std::atomic<int> streamQueryCount{0};
    std::atomic<int> destroyStreamCount{0};
    std::atomic<int> mallocCount{0};
    std::atomic<int> freeCount{0};
    std::atomic<int> inplaceScatterWsfCount{0};
    std::atomic<int> inplaceScatterCount{0};
    std::atomic<int> createTensorCount{0};
    std::atomic<int> destroyTensorCount{0};

    void Reset()
    {
        setDeviceFails = false;
        createContextFails = false;
        createStreamFails = false;
        getCurrentContextFails = false;
        setCurrentContextFails = false;
        createStreamStatusFails = false;
        memcpyAsyncFails = false;
        memcpyFails = false;
        syncStreamFails = false;
        streamQueryFails = false;
        destroyStreamFails = false;
        mallocFails = false;
        freeFails = false;
        inplaceScatterWsfFails = false;
        inplaceScatterFails = false;
        createTensorFails = false;
        destroyTensorFails = false;
        returnNullErrMsg = false;

        setDeviceCount = 0;
        createContextCount = 0;
        createStreamCount = 0;
        getCurrentContextCount = 0;
        setCurrentContextCount = 0;
        memcpyAsyncCount = 0;
        memcpyCount = 0;
        syncStreamCount = 0;
        streamQueryCount = 0;
        destroyStreamCount = 0;
        mallocCount = 0;
        freeCount = 0;
        inplaceScatterWsfCount = 0;
        inplaceScatterCount = 0;
        createTensorCount = 0;
        destroyTensorCount = 0;
    }
};

inline StubControl& Stub()
{
    static StubControl s;
    return s;
}

}  // namespace acl_stub

/* ---- ACL type / constant surface expected by the prefetch sources ---- */

using aclError = int32_t;
using aclrtStream = void*;
using aclrtContext = void*;

enum AclrtStreamStatus {
    ACL_STREAM_STATUS_RESERVED = 0,
    ACL_STREAM_STATUS_COMPLETE = 1,
    ACL_STREAM_STATUS_NOT_READY = 2,
};
using aclrtStreamStatus = int;

enum AclrtMemcpyKind {
    ACL_MEMCPY_HOST_TO_DEVICE = 1,
    ACL_MEMCPY_DEVICE_TO_HOST = 2,
    ACL_MEMCPY_DEVICE_TO_DEVICE = 3,
    ACL_MEMCPY_DEVICE_TO_DEVICE_INNER = 4,
};
using aclrtMemcpyKind = int;

enum AclrtMallocAllocPolicy {
    ACL_MEM_MALLOC_HUGE_FIRST = 0,
    ACL_MEM_MALLOC_HUGE_ONLY = 1,
    ACL_MEM_MALLOC_NORMAL_FIRST = 2,
    ACL_MEM_MALLOC_NORMAL_ONLY = 3,
};
using aclrtMallocAllocPolicy = AclrtMallocAllocPolicy;

constexpr aclError ACL_SUCCESS = 0;
constexpr aclError ACL_ERROR_INVALID_PARAM = -1;
constexpr aclError ACL_ERROR_INTERNAL_ERROR = -2;
constexpr aclError ACL_ERROR_STREAM_NOT_READY = 100000;

enum aclDataType {
    ACL_DT_UNDEFINED = -1,
    ACL_FLOAT = 0,
    ACL_FLOAT16 = 1,
    ACL_INT8 = 2,
    ACL_INT32 = 3,
    ACL_UINT32 = 4,
    ACL_INT64 = 9,
    ACL_UINT64 = 10,
    ACL_DOUBLE = 11,
    ACL_BOOL = 12,
    ACL_BF16 = 13,
};
enum aclFormat {
    ACL_FORMAT_UNDEFINED = -1,
    ACL_FORMAT_ND = 2,
};

struct aclTensor;
struct aclOpExecutor;

/* ---- ACL functions used by kvcache_trans_ascend.cpp ---- */

inline const char* aclGetRecentErrMsg()
{
    if (acl_stub::Stub().returnNullErrMsg.load()) { return nullptr; }
    return "stub-error-message";
}

inline aclError aclrtSetDevice(int32_t deviceId)
{
    (void)deviceId;
    acl_stub::Stub().setDeviceCount.fetch_add(1);
    return acl_stub::Stub().setDeviceFails.load() ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

inline aclError aclrtCreateContext(aclrtContext* ctx, int32_t deviceId)
{
    (void)deviceId;
    acl_stub::Stub().createContextCount.fetch_add(1);
    if (acl_stub::Stub().createContextFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    *ctx = reinterpret_cast<aclrtContext>(0xABADCAFEu);
    return ACL_SUCCESS;
}

inline aclError aclrtDestroyContext(aclrtContext ctx)
{
    (void)ctx;
    return ACL_SUCCESS;
}

inline aclError aclrtResetDevice(int32_t deviceId)
{
    (void)deviceId;
    return ACL_SUCCESS;
}

inline aclError aclrtGetCurrentContext(aclrtContext* ctx)
{
    acl_stub::Stub().getCurrentContextCount.fetch_add(1);
    if (acl_stub::Stub().getCurrentContextFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    *ctx = reinterpret_cast<aclrtContext>(0xC0FFEEu);
    return ACL_SUCCESS;
}

inline aclError aclrtSetCurrentContext(aclrtContext ctx)
{
    (void)ctx;
    acl_stub::Stub().setCurrentContextCount.fetch_add(1);
    return acl_stub::Stub().setCurrentContextFails.load() ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

inline aclError aclrtCreateStream(aclrtStream* stream)
{
    acl_stub::Stub().createStreamCount.fetch_add(1);
    if (acl_stub::Stub().createStreamFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    *stream = reinterpret_cast<aclrtStream>(0x5F12EA4u);
    return ACL_SUCCESS;
}

inline aclError aclrtDestroyStream(aclrtStream stream)
{
    (void)stream;
    acl_stub::Stub().destroyStreamCount.fetch_add(1);
    return acl_stub::Stub().destroyStreamFails.load() ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

inline aclError aclrtSynchronizeStream(aclrtStream stream)
{
    (void)stream;
    acl_stub::Stub().syncStreamCount.fetch_add(1);
    return acl_stub::Stub().syncStreamFails.load() ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

inline aclError aclrtStreamQuery(aclrtStream stream, aclrtStreamStatus* status)
{
    (void)stream;
    acl_stub::Stub().streamQueryCount.fetch_add(1);
    if (acl_stub::Stub().streamQueryFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    *status = ACL_STREAM_STATUS_COMPLETE;
    return ACL_SUCCESS;
}

inline aclError aclrtMemcpyAsync(void* dst, size_t dstMax, const void* src, size_t count,
                                 aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)dstMax;
    (void)kind;
    (void)stream;
    acl_stub::Stub().memcpyAsyncCount.fetch_add(1);
    if (acl_stub::Stub().memcpyAsyncFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    if (dst != nullptr && src != nullptr && count > 0) { std::memcpy(dst, src, count); }
    return ACL_SUCCESS;
}

inline aclError aclrtMemcpy(void* dst, size_t dstMax, const void* src, size_t count,
                            aclrtMemcpyKind kind)
{
    (void)dstMax;
    (void)kind;
    acl_stub::Stub().memcpyCount.fetch_add(1);
    if (acl_stub::Stub().memcpyFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    if (dst != nullptr && src != nullptr && count > 0) { std::memcpy(dst, src, count); }
    return ACL_SUCCESS;
}

inline aclError aclrtMalloc(void** addr, size_t size, aclrtMallocAllocPolicy policy)
{
    (void)policy;
    acl_stub::Stub().mallocCount.fetch_add(1);
    if (acl_stub::Stub().mallocFails.load() || size == 0) {
        *addr = nullptr;
        return ACL_ERROR_INTERNAL_ERROR;
    }
    *addr = std::malloc(size);
    return *addr == nullptr ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

inline aclError aclrtFree(void* addr)
{
    acl_stub::Stub().freeCount.fetch_add(1);
    if (acl_stub::Stub().freeFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    if (addr != nullptr) { std::free(addr); }
    return ACL_SUCCESS;
}

/* ---- opapi surface used by the scatter path ---- */

inline aclError aclnnInplaceScatterGetWorkspaceSize(aclTensor* dst, int64_t dim, aclTensor* index,
                                                    aclTensor* src, int64_t reduce,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor)
{
    (void)dst;
    (void)dim;
    (void)index;
    (void)src;
    (void)reduce;
    acl_stub::Stub().inplaceScatterWsfCount.fetch_add(1);
    if (acl_stub::Stub().inplaceScatterWsfFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    if (workspaceSize != nullptr) { *workspaceSize = 0; }
    if (executor != nullptr) { *executor = reinterpret_cast<aclOpExecutor*>(0xE7ECu); }
    return ACL_SUCCESS;
}

inline aclError aclnnInplaceScatter(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor, aclrtStream stream)
{
    (void)workspace;
    (void)workspaceSize;
    (void)executor;
    (void)stream;
    acl_stub::Stub().inplaceScatterCount.fetch_add(1);
    return acl_stub::Stub().inplaceScatterFails.load() ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

inline aclTensor* aclCreateTensor(const int64_t* dims, uint64_t dimNum, aclDataType dtype,
                                  const int64_t* stride, int64_t offset, aclFormat format,
                                  const int64_t* storageDims, uint64_t storageDimNum, void* data)
{
    (void)dims;
    (void)dimNum;
    (void)dtype;
    (void)stride;
    (void)offset;
    (void)format;
    (void)storageDims;
    (void)storageDimNum;
    (void)data;
    acl_stub::Stub().createTensorCount.fetch_add(1);
    if (acl_stub::Stub().createTensorFails.load()) { return nullptr; }
    return reinterpret_cast<aclTensor*>(0xABCD1234u);
}

inline aclError aclDestroyTensor(const aclTensor* tensor)
{
    (void)tensor;
    acl_stub::Stub().destroyTensorCount.fetch_add(1);
    return acl_stub::Stub().destroyTensorFails.load() ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

/* ---- dlsym-style opapi resolver expected by kvcache_trans_ascend.cpp ----
 * The production header resolves these via dlopen/dlsym against libopapi; the
 * stub routes them to the inline functions above so the same .cpp compiles and
 * runs in a host test build without libopapi installed. */
using _aclCreateTensor = aclTensor* (*)(const int64_t* view_dims, uint64_t view_dims_num,
                                        aclDataType data_type, const int64_t* stride,
                                        int64_t offset, aclFormat format,
                                        const int64_t* storage_dims, uint64_t storage_dims_num,
                                        void* tensor_data);
using _aclDestroyTensor = int (*)(const aclTensor* tensor);

inline void* GetOpApiLibHandler(const char*) { return reinterpret_cast<void*>(0x1); }
inline const char* GetOpApiLibName(void) { return "stub-libopapi"; }
inline const char* GetCustOpApiLibName(void) { return "stub-libcustopapi"; }
inline void* GetOpApiFuncAddrInLib(void*, const char*, const char*) { return nullptr; }
inline void* GetCustOpApiFuncAddr(const char*) { return nullptr; }
inline void* GetOpApiFuncAddr(const char* apiName)
{
    if (apiName == nullptr) { return nullptr; }
    if (std::strcmp(apiName, "aclCreateTensor") == 0) {
        return reinterpret_cast<void*>(&aclCreateTensor);
    }
    if (std::strcmp(apiName, "aclDestroyTensor") == 0) {
        return reinterpret_cast<void*>(&aclDestroyTensor);
    }
    return nullptr;
}
#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

/* ---- Declarations of the wrapper functions defined in
 * ascend/kvcache_trans_ascend.cpp.  The non-test header declares these in the
 * `#else` branch; the test build needs the same declarations visible so both
 * the .cpp and the test files see a consistent prototype. ---- */
class Logger;  // forward decl (kvcache_log.h is included by the .cpp)
int32_t CopyTrans(void** host_ptrs_addr, void** device_ptrs_addr, size_t size_bytes, size_t number,
                  std::string type, void* stream_);
void ObtainStream(void** outputStream, void** outputContext, int32_t deviceID, int32_t mStreamNum);
int32_t Synchronize(void* context, void* stream, int streamNum, Logger& mLogger);
bool IsStreamIdle(void* stream);
void DestroyStream(void* stream, void* context, int32_t deviceID, int32_t mStreamNum);
aclTensor* AtTensorToAclTensorShared(at::Tensor& tensor);
aclDataType GetAclDataType(at::ScalarType dtype);
void ReleaseAclTensor(aclTensor* tensor);

#endif  // KV_CACHE_TRANS_ASCEND_STUB_H
