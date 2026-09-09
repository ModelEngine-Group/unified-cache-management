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
#include "src/protocol/ub_error_code.h"
#include "src/ub/status.h"

namespace umc::comm::acl {

enum class MemcpyKind : int32_t {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
};

enum class MallocPolicy : int32_t {
    HugeFirst = 0,
    HugeOnly = 1,
    NormalOnly = 2,
};

class DlAclRt {
public:
    static UbStatus LoadLibrary();
    static void CleanUpLibrary();
    static bool IsLoaded();

    static UbStatus SetDevice(int32_t deviceId);
    static UbStatus ResetDevice(int32_t deviceId);
    static UbStatus Finalize();

    static UbStatus GetCurrentContext(void** ctx);
    static UbStatus SetCurrentContext(void* ctx);
    static bool HasContextApi();

    static UbStatus GetPhyDevIdByLogicDevId(int32_t logicDevId, int32_t* phyDevId);

    static UbStatus Malloc(void** devPtr, std::size_t size,
                           MallocPolicy policy = MallocPolicy::HugeFirst);
    static UbStatus Free(void* devPtr);
    static UbStatus Memset(void* devPtr, std::size_t maxCount, int32_t value, std::size_t count);
    static UbStatus Memcpy(void* dst, std::size_t dstMax, const void* src, std::size_t count,
                           MemcpyKind kind);

private:
    using MallocFunc = int (*)(void**, std::size_t, int);
    using FreeFunc = int (*)(void*);
    using MemsetFunc = int (*)(void*, std::size_t, int32_t, std::size_t);
    using MemcpyFunc = int (*)(void*, std::size_t, const void*, std::size_t, int);
    using SetDeviceFunc = int (*)(int32_t);
    using InitFunc = int (*)(const char*);
    using FinalizeFunc = int (*)();
    using GetPhyIdFunc = int (*)(int32_t, int32_t*);
    using GetCtxFunc = int (*)(void**);
    using SetCtxFunc = int (*)(void*);

    static std::mutex& Mu();
    static void*& Handle();
    static std::atomic_bool& Loaded();
    static MallocFunc& MallocSlot();
    static FreeFunc& FreeSlot();
    static MemsetFunc& MemsetSlot();
    static MemcpyFunc& MemcpySlot();
    static SetDeviceFunc& SetDeviceSlot();
    static SetDeviceFunc& ResetDeviceSlot();
    static InitFunc& InitSlot();
    static FinalizeFunc& FinalizeSlot();
    static GetPhyIdFunc& GetPhyIdSlot();
    static GetCtxFunc& GetCtxSlot();
    static SetCtxFunc& SetCtxSlot();
};

}  // namespace umc::comm::acl
