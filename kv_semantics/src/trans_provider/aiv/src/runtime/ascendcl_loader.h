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

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include "src/ub/status.h"

namespace umc::kv::dl {

enum AclrtMemcpyKind : int32_t {
    ACL_MEMCPY_HOST_TO_HOST = 0,
    ACL_MEMCPY_HOST_TO_DEVICE = 1,
    ACL_MEMCPY_DEVICE_TO_HOST = 2,
    ACL_MEMCPY_DEVICE_TO_DEVICE = 3,
};

enum AclrtMallocPolicy : int32_t {
    ACL_MEM_MALLOC_HUGE_FIRST = 0,
    ACL_MEM_MALLOC_HUGE_ONLY = 1,
    ACL_MEM_MALLOC_NORMAL_ONLY = 2,
};

class DlAscendcl {
public:
    static ::umc::comm::UbStatus LoadLibrary();
    static void CleanUpLibrary();
    static bool IsLoaded();

    static ::umc::comm::UbStatus AclrtMalloc(void** devPtr, std::size_t size,
                                             AclrtMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
    static ::umc::comm::UbStatus AclrtFree(void* devPtr);
    static ::umc::comm::UbStatus AclrtMemcpy(void* dst, std::size_t dstMax, const void* src,
                                             std::size_t count, AclrtMemcpyKind kind);
    static ::umc::comm::UbStatus AclrtSynchronizeStream(void* stream);
    static ::umc::comm::UbStatus AclrtSynchronizeStreamWithTimeout(void* stream, int32_t timeoutMs);
    static ::umc::comm::UbStatus AclrtCreateStream(void** stream);
    static ::umc::comm::UbStatus AclrtDestroyStream(void* stream);

    static ::umc::comm::UbStatus AclrtBinaryLoadFromFile(const char* binPath, void** binHandle);
    static ::umc::comm::UbStatus AclrtBinaryLoadFromData(const void* data, std::size_t len,
                                                         void** binHandle);
    static ::umc::comm::UbStatus AclrtBinaryGetFunction(void* binHandle, const char* kernelName,
                                                        void** funcHandle);
    static ::umc::comm::UbStatus AclrtBinaryUnload(void* binHandle);
    static ::umc::comm::UbStatus AclrtLaunchKernelWithHostArgs(void* funcHandle, uint32_t blockDim,
                                                               void* stream, void* hostArgs,
                                                               std::size_t argsSize);
    static ::umc::comm::UbStatus AclrtLaunchKernelWithDeviceArgs(void* funcHandle,
                                                                 uint32_t blockDim, void* stream,
                                                                 const void* deviceArgs,
                                                                 std::size_t argsSize);

private:
    static std::mutex& Mu();
    static void*& Handle();  // dlopen handle
    static std::atomic_bool& Loaded();
    using MallocFunc = int (*)(void**, std::size_t, int);
    using FreeFunc = int (*)(void*);
    using MemcpyFunc = int (*)(void*, std::size_t, const void*, std::size_t, int);
    using SyncFunc = int (*)(void*);
    using SyncTimeoutFunc = int (*)(void*, int32_t);
    using CreateStreamFunc = int (*)(void**);
    using DestroyStreamFunc = int (*)(void*);
    using BinLoadFileFunc = int (*)(const char*, void*, void**);
    using BinLoadDataFunc = int (*)(const void*, std::size_t, void*, void**);
    using BinGetFuncFunc = int (*)(void*, const char*, void**);
    using BinUnloadFunc = int (*)(void*);
    using LaunchKernelV2Func = int (*)(void*, uint32_t, const void*, std::size_t, void*, void*);
    using LaunchHostArgsFunc = int (*)(void*, uint32_t, void*, void*, void*, std::size_t, void*,
                                       std::size_t);
    static MallocFunc& MallocSlot();
    static FreeFunc& FreeSlot();
    static MemcpyFunc& MemcpySlot();
    static SyncFunc& SyncSlot();
    static SyncTimeoutFunc& SyncTimeoutSlot();
    static CreateStreamFunc& CreateStreamSlot();
    static DestroyStreamFunc& DestroyStreamSlot();
    static BinLoadFileFunc& BinLoadFileSlot();
    static BinLoadDataFunc& BinLoadDataSlot();
    static BinGetFuncFunc& BinGetFuncSlot();
    static BinUnloadFunc& BinUnloadSlot();
    static LaunchKernelV2Func& LaunchKernelV2Slot();
    static LaunchHostArgsFunc& LaunchHostArgsSlot();
};

}  // namespace umc::kv::dl
