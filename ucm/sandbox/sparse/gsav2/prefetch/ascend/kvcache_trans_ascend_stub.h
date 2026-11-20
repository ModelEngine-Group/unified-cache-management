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
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "kvcache_log.h"

namespace acl_stub {

/* Host-side scatter emulator (defined after the ACL type surface below).
 * aclnnInplaceScatter becomes a real in-memory scatter when realScatter is
 * set, letting end-to-end tests verify the HBM KV cache bytes the way the
 * real ACL op would write them. */
struct ScatterEmulator;
inline ScatterEmulator& ScatterEmu();
inline void ResetScatterEmulator();

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

    /* Off by default: history is a count-only stub. When enabled, tensor
     * handles become registry keys and aclnnInplaceScatter applies
     * dst[index[i][j], j] = src[i][j] in host memory (dim 0 only). */
    bool realScatter{false};

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
        realScatter = false;
        ResetScatterEmulator();

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

/* ---- Host-side scatter emulator ------------------------------------------
 * Opt-in (Stub().realScatter). Mirrors the semantics of the production
 * KVCacheInplaceScatter call chain: AtTensorToAclTensorShared hands ACL the
 * real storage base + storage offset + sizes/strides, so the emulator can
 * resolve every (dst, index, src) triple to host memory and apply
 * dst[index[i][j], j] = src[i][j] (dim 0, stride-aware including broadcast
 * index strides of 0). Handles are registry keys and only exist while
 * realScatter is on, so the default count-only stub path is untouched. */
namespace acl_stub {

struct ScatterEmulator {
    struct TensorDesc {
        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
        aclDataType dtype = ACL_DT_UNDEFINED;
        int64_t offset = 0;  // storage offset, in elements
        void* data = nullptr;  // storage base
    };

    struct PendingScatter {
        TensorDesc dst;
        TensorDesc index;
        TensorDesc src;
        int64_t dim = 0;
    };

    std::map<void*, TensorDesc> registry;
    std::deque<PendingScatter> pending;
    std::mutex mu;
    uintptr_t nextHandle = 0x100000;

    static size_t ElemSize(aclDataType dtype)
    {
        switch (dtype) {
            case ACL_BOOL:
            case ACL_INT8: return 1;
            case ACL_FLOAT16:
            case ACL_BF16: return 2;
            case ACL_FLOAT:
            case ACL_INT32:
            case ACL_UINT32: return 4;
            case ACL_INT64:
            case ACL_UINT64:
            case ACL_DOUBLE: return 8;
            default: return 0;
        }
    }

    aclTensor* Register(const int64_t* dims, uint64_t dimNum, aclDataType dtype,
                        const int64_t* strides, int64_t offset, void* data)
    {
        std::lock_guard<std::mutex> lock(mu);
        TensorDesc desc;
        desc.dims.assign(dims, dims + dimNum);
        desc.strides.assign(strides, strides + dimNum);
        desc.dtype = dtype;
        desc.offset = offset;
        desc.data = data;
        void* handle = reinterpret_cast<void*>(nextHandle);
        nextHandle += 0x10;
        registry[handle] = std::move(desc);
        return reinterpret_cast<aclTensor*>(handle);
    }

    void Unregister(const aclTensor* tensor)
    {
        std::lock_guard<std::mutex> lock(mu);
        registry.erase(const_cast<void*>(reinterpret_cast<const void*>(tensor)));
    }

    /* Records (dst, dim, index, src) at workspace-size time; the op runs at
     * aclnnInplaceScatter time. Returns false when a handle is unknown
     * (e.g. failure-injection tests passing sentinel pointers). */
    bool TryQueueScatter(aclTensor* dst, int64_t dim, aclTensor* index, aclTensor* src)
    {
        std::lock_guard<std::mutex> lock(mu);
        auto dstIt = registry.find(dst);
        auto idxIt = registry.find(index);
        auto srcIt = registry.find(src);
        if (dstIt == registry.end() || idxIt == registry.end() ||
            srcIt == registry.end()) {
            return false;
        }
        PendingScatter op;
        op.dst = dstIt->second;
        op.index = idxIt->second;
        op.src = srcIt->second;
        op.dim = dim;
        pending.push_back(std::move(op));
        return true;
    }

    void ApplyOne()
    {
        PendingScatter op;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (pending.empty()) { return; }
            op = std::move(pending.front());
            pending.pop_front();
        }
        Apply(op);
    }

    void Clear()
    {
        std::lock_guard<std::mutex> lock(mu);
        registry.clear();
        pending.clear();
    }

    /* dst[index[i][j], j] = src[i][j] over every element of the index shape. */
    static void Apply(const PendingScatter& op)
    {
        size_t dstElem = 0, srcElem = 0, idxElem = 0, copyBytes = 0;
        if (!ValidateScatterOp(op, dstElem, srcElem, idxElem, copyBytes)) { return; }
        int64_t total = 1;
        for (int64_t d : op.index.dims) { total *= d; }
        std::vector<int64_t> coords(op.index.dims.size(), 0);
        for (int64_t flat = 0; flat < total; ++flat) {
            ScatterOneElement(op, coords, dstElem, srcElem, idxElem, copyBytes);
            /* odometer increment over the index shape */
            for (size_t d = coords.size(); d-- > 0;) {
                if (++coords[d] < op.index.dims[d]) { break; }
                coords[d] = 0;
            }
        }
    }

    static bool ValidateScatterOp(const PendingScatter& op, size_t& dstElem, size_t& srcElem,
                                  size_t& idxElem, size_t& copyBytes)
    {
        if (op.dim != 0 || op.index.dims.empty() || op.index.dims.size() != op.src.dims.size() ||
            op.index.dims.size() != op.dst.dims.size()) {
            return false;
        }
        dstElem = ElemSize(op.dst.dtype);
        srcElem = ElemSize(op.src.dtype);
        idxElem = ElemSize(op.index.dtype);
        if (dstElem == 0 || srcElem == 0 || idxElem != sizeof(int64_t)) { return false; }
        copyBytes = (dstElem < srcElem) ? dstElem : srcElem;
        return true;
    }

    static void ScatterOneElement(const PendingScatter& op, const std::vector<int64_t>& coords,
                                  size_t dstElem, size_t srcElem, size_t idxElem, size_t copyBytes)
    {
        auto* dstBase = static_cast<uint8_t*>(op.dst.data);
        auto* srcBase = static_cast<uint8_t*>(op.src.data);
        auto* idxBase = static_cast<uint8_t*>(op.index.data);
        int64_t idxOffset = op.index.offset;
        int64_t srcOffset = op.src.offset;
        for (size_t d = 0; d < coords.size(); ++d) {
            idxOffset += coords[d] * op.index.strides[d];
            srcOffset += coords[d] * op.src.strides[d];
        }
        const int64_t idxVal =
            *reinterpret_cast<const int64_t*>(idxBase + idxOffset * idxElem);
        if (idxVal < 0 || idxVal >= op.dst.dims[op.dim]) {
            /* Out-of-range scatter: real ACL errors; the emulator skips
             * the element so a broken index cannot corrupt the dump. */
            return;
        }
        int64_t dstOffset = op.dst.offset;
        for (size_t d = 0; d < coords.size(); ++d) {
            dstOffset += (d == op.dim ? idxVal : coords[d]) * op.dst.strides[d];
        }
        std::memcpy(dstBase + dstOffset * dstElem, srcBase + srcOffset * srcElem, copyBytes);
    }
};

inline ScatterEmulator& ScatterEmu()
{
    static ScatterEmulator emu;
    return emu;
}

inline void ResetScatterEmulator()
{
    ScatterEmu().Clear();
}

}  // namespace acl_stub

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
    (void)reduce;
    acl_stub::Stub().inplaceScatterWsfCount.fetch_add(1);
    if (acl_stub::Stub().inplaceScatterWsfFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    if (acl_stub::Stub().realScatter) {
        (void)acl_stub::ScatterEmu().TryQueueScatter(dst, dim, index, src);
    }
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
    if (acl_stub::Stub().realScatter) { acl_stub::ScatterEmu().ApplyOne(); }
    acl_stub::Stub().inplaceScatterCount.fetch_add(1);
    return acl_stub::Stub().inplaceScatterFails.load() ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

inline aclTensor* aclCreateTensor(const int64_t* dims, uint64_t dimNum, aclDataType dtype,
                                  const int64_t* stride, int64_t offset, aclFormat format,
                                  const int64_t* storageDims, uint64_t storageDimNum, void* data)
{
    (void)format;
    (void)storageDims;
    (void)storageDimNum;
    acl_stub::Stub().createTensorCount.fetch_add(1);
    if (acl_stub::Stub().createTensorFails.load()) { return nullptr; }
    if (acl_stub::Stub().realScatter) {
        return acl_stub::ScatterEmu().Register(dims, dimNum, dtype, stride, offset, data);
    }
    return reinterpret_cast<aclTensor*>(0xABCD1234u);
}

inline aclError aclDestroyTensor(const aclTensor* tensor)
{
    (void)tensor;
    acl_stub::Stub().destroyTensorCount.fetch_add(1);
    if (acl_stub::Stub().destroyTensorFails.load()) { return ACL_ERROR_INTERNAL_ERROR; }
    if (acl_stub::Stub().realScatter) { acl_stub::ScatterEmu().Unregister(tensor); }
    return ACL_SUCCESS;
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
int32_t ObtainStream(void** outputStream, void** outputContext, int32_t deviceID, int32_t mStreamNum);
int32_t Synchronize(void* context, void* stream, int streamNum, Logger& mLogger);
bool IsStreamIdle(void* stream);
void DestroyStream(void* stream, void* context, int32_t deviceID, int32_t mStreamNum);
aclTensor* AtTensorToAclTensorShared(at::Tensor& tensor);
aclDataType GetAclDataType(at::ScalarType dtype);
void ReleaseAclTensor(aclTensor* tensor);

#endif  // KV_CACHE_TRANS_ASCEND_STUB_H
