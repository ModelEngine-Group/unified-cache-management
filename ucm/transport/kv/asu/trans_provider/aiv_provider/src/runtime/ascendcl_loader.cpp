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

#include "src/runtime/ascendcl_loader.h"
#include <dlfcn.h>
#include "src/ub/log.h"

namespace umc::kv::dl {

using ::umc::comm::UbErrorCode;
using ::umc::comm::UbStatus;

std::mutex& DlAscendcl::Mu()
{
    static std::mutex m;
    return m;
}
void*& DlAscendcl::Handle()
{
    static void* h = nullptr;
    return h;
}
std::atomic_bool& DlAscendcl::Loaded()
{
    static std::atomic_bool b{false};
    return b;
}
DlAscendcl::MallocFunc& DlAscendcl::MallocSlot()
{
    static MallocFunc f = nullptr;
    return f;
}
DlAscendcl::FreeFunc& DlAscendcl::FreeSlot()
{
    static FreeFunc f = nullptr;
    return f;
}
DlAscendcl::MemcpyFunc& DlAscendcl::MemcpySlot()
{
    static MemcpyFunc f = nullptr;
    return f;
}
DlAscendcl::SyncFunc& DlAscendcl::SyncSlot()
{
    static SyncFunc f = nullptr;
    return f;
}
DlAscendcl::SyncTimeoutFunc& DlAscendcl::SyncTimeoutSlot()
{
    static SyncTimeoutFunc f = nullptr;
    return f;
}
DlAscendcl::CreateStreamFunc& DlAscendcl::CreateStreamSlot()
{
    static CreateStreamFunc f = nullptr;
    return f;
}
DlAscendcl::DestroyStreamFunc& DlAscendcl::DestroyStreamSlot()
{
    static DestroyStreamFunc f = nullptr;
    return f;
}
DlAscendcl::BinLoadFileFunc& DlAscendcl::BinLoadFileSlot()
{
    static BinLoadFileFunc f = nullptr;
    return f;
}
DlAscendcl::BinLoadDataFunc& DlAscendcl::BinLoadDataSlot()
{
    static BinLoadDataFunc f = nullptr;
    return f;
}
DlAscendcl::BinGetFuncFunc& DlAscendcl::BinGetFuncSlot()
{
    static BinGetFuncFunc f = nullptr;
    return f;
}
DlAscendcl::BinUnloadFunc& DlAscendcl::BinUnloadSlot()
{
    static BinUnloadFunc f = nullptr;
    return f;
}
DlAscendcl::LaunchKernelV2Func& DlAscendcl::LaunchKernelV2Slot()
{
    static LaunchKernelV2Func f = nullptr;
    return f;
}
DlAscendcl::LaunchHostArgsFunc& DlAscendcl::LaunchHostArgsSlot()
{
    static LaunchHostArgsFunc f = nullptr;
    return f;
}

bool DlAscendcl::IsLoaded() { return Loaded().load(std::memory_order_acquire); }

UbStatus DlAscendcl::LoadLibrary()
{
    std::lock_guard<std::mutex> lk(Mu());
    if (Loaded().load(std::memory_order_relaxed)) return UbStatus::Ok();

    void* h = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) {
        const char* error = dlerror();
        UB_LOG_WARN(
            "dlopen libascendcl.so failed: {}; "
            "device operations are unavailable",
            error != nullptr ? error : "(no info)");
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "libascendcl.so not available");
    }
    auto resolve = [&](const char* name, void** outPtr) -> bool {
        void* sym = dlsym(h, name);
        if (sym == nullptr) {
            const char* error = dlerror();
            UB_LOG_ERROR("dlsym {} failed: {}", name, error != nullptr ? error : "(no info)");
            return false;
        }
        *outPtr = sym;
        return true;
    };
    void *m = nullptr, *f = nullptr, *cp = nullptr;
    bool ok = resolve("aclrtMalloc", &m) && resolve("aclrtFree", &f) && resolve("aclrtMemcpy", &cp);
    if (!ok) {
        dlclose(h);
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "missing ascendcl symbols");
    }
    Handle() = h;
    MallocSlot() = reinterpret_cast<MallocFunc>(m);
    FreeSlot() = reinterpret_cast<FreeFunc>(f);
    MemcpySlot() = reinterpret_cast<MemcpyFunc>(cp);
    SyncSlot() = reinterpret_cast<SyncFunc>(dlsym(h, "aclrtSynchronizeStream"));
    SyncTimeoutSlot() =
        reinterpret_cast<SyncTimeoutFunc>(dlsym(h, "aclrtSynchronizeStreamWithTimeout"));
    CreateStreamSlot() = reinterpret_cast<CreateStreamFunc>(dlsym(h, "aclrtCreateStream"));
    DestroyStreamSlot() = reinterpret_cast<DestroyStreamFunc>(dlsym(h, "aclrtDestroyStream"));
    BinLoadFileSlot() = reinterpret_cast<BinLoadFileFunc>(dlsym(h, "aclrtBinaryLoadFromFile"));
    BinLoadDataSlot() = reinterpret_cast<BinLoadDataFunc>(dlsym(h, "aclrtBinaryLoadFromData"));
    BinGetFuncSlot() = reinterpret_cast<BinGetFuncFunc>(dlsym(h, "aclrtBinaryGetFunction"));
    void* binUnload = dlsym(h, "aclrtBinaryUnLoad");
    if (binUnload == nullptr) { binUnload = dlsym(h, "aclrtBinaryUnload"); }
    BinUnloadSlot() = reinterpret_cast<BinUnloadFunc>(binUnload);
    LaunchKernelV2Slot() = reinterpret_cast<LaunchKernelV2Func>(dlsym(h, "aclrtLaunchKernelV2"));
    LaunchHostArgsSlot() =
        reinterpret_cast<LaunchHostArgsFunc>(dlsym(h, "aclrtLaunchKernelWithHostArgs"));
    Loaded().store(true, std::memory_order_release);
    UB_LOG_DEBUG("DlAscendcl: libascendcl.so loaded ok");
    return UbStatus::Ok();
}

void DlAscendcl::CleanUpLibrary()
{
    // Calls use resolved function pointers without a per-call lock. Keep the
    // successfully loaded library resident for the process lifetime.
}

UbStatus DlAscendcl::AclrtMalloc(void** devPtr, std::size_t size, AclrtMallocPolicy policy)
{
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    if (MallocSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "ascendcl not loaded");
    }
    int rc = MallocSlot()(devPtr, size, static_cast<int>(policy));
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtMalloc failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtFree(void* devPtr)
{
    if (devPtr == nullptr) { return UbStatus::Ok(); }
    if (!Loaded() || FreeSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtFree unavailable: ascendcl not loaded");
    }
    int rc = FreeSlot()(devPtr);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtFree failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtMemcpy(void* dst, std::size_t dstMax, const void* src, std::size_t count,
                                 AclrtMemcpyKind kind)
{
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    if (MemcpySlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "ascendcl not loaded");
    }
    int rc = MemcpySlot()(dst, dstMax, src, count, static_cast<int>(kind));
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtMemcpy failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtCreateStream(void** stream)
{
    if (stream == nullptr) { return UbStatus(UbErrorCode::InvalidArgument, "stream out ptr null"); }
    *stream = nullptr;
    UB_LOG_DEBUG("DlAscendcl::AclrtCreateStream enter: loaded={} slot={}", Loaded() ? 1 : 0,
                 reinterpret_cast<void*>(CreateStreamSlot()));
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    if (CreateStreamSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "aclrtCreateStream symbol missing");
    }
    UB_LOG_DEBUG("DlAscendcl::AclrtCreateStream call aclrtCreateStream");
    int rc = CreateStreamSlot()(stream);
    UB_LOG_DEBUG("DlAscendcl::AclrtCreateStream returned: rc={} stream={}", rc, *stream);
    if (rc != 0) {
        *stream = nullptr;
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtCreateStream failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtDestroyStream(void* stream)
{
    if (stream == nullptr) { return UbStatus::Ok(); }
    if (!Loaded() || DestroyStreamSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "aclrtDestroyStream unavailable");
    }
    int rc = DestroyStreamSlot()(stream);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtDestroyStream failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtBinaryLoadFromFile(const char* binPath, void** binHandle)
{
    if (binPath == nullptr || binHandle == nullptr) {
        return UbStatus(UbErrorCode::InvalidArgument, "binPath/binHandle null");
    }
    *binHandle = nullptr;
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    if (BinLoadFileSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtBinaryLoadFromFile symbol missing");
    }
    int rc = BinLoadFileSlot()(binPath, nullptr, binHandle);
    if (rc != 0 || *binHandle == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        std::string("aclrtBinaryLoadFromFile(") + binPath +
                            ") failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtBinaryLoadFromData(const void* data, std::size_t len, void** binHandle)
{
    if (data == nullptr || len == 0 || binHandle == nullptr) {
        return UbStatus(UbErrorCode::InvalidArgument, "data/len/binHandle invalid");
    }
    *binHandle = nullptr;
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    if (BinLoadDataSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtBinaryLoadFromData symbol missing");
    }
    int rc = BinLoadDataSlot()(data, len, nullptr, binHandle);
    if (rc != 0 || *binHandle == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtBinaryLoadFromData(len=" + std::to_string(len) +
                            ") failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtBinaryGetFunction(void* binHandle, const char* kernelName,
                                            void** funcHandle)
{
    if (binHandle == nullptr || kernelName == nullptr || funcHandle == nullptr) {
        return UbStatus(UbErrorCode::InvalidArgument, "binHandle/kernelName/funcHandle null");
    }
    *funcHandle = nullptr;
    if (!Loaded() || BinGetFuncSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtBinaryGetFunction symbol missing");
    }
    int rc = BinGetFuncSlot()(binHandle, kernelName, funcHandle);
    if (rc != 0 || *funcHandle == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        std::string("aclrtBinaryGetFunction('") + kernelName +
                            "') failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtBinaryUnload(void* binHandle)
{
    if (binHandle == nullptr) { return UbStatus::Ok(); }
    if (!Loaded() || BinUnloadSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "aclrtBinaryUnload unavailable");
    }
    int rc = BinUnloadSlot()(binHandle);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtBinaryUnload failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtLaunchKernelWithHostArgs(void* funcHandle, uint32_t blockDim,
                                                   void* stream, void* hostArgs,
                                                   std::size_t argsSize)
{
    if (funcHandle == nullptr) { return UbStatus(UbErrorCode::InvalidArgument, "funcHandle null"); }
    if (!Loaded() || LaunchHostArgsSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtLaunchKernelWithHostArgs symbol missing");
    }
    struct LaunchKernelAttr {
        int32_t id;  // aclrtLaunchKernelAttrId
        union {
            uint8_t schemMode;   // ATTR_SCHEM_MODE
            int32_t engineType;  // ATTR_ENGINE_TYPE：ACL_RT_ENGINE_TYPE_AIV=1
            struct {
                uint32_t timeoutLow;
                uint32_t timeoutHigh;
            } timeoutUs;  // ATTR_TIMEOUT_US
            uint32_t rsv[4];
        } value;
    };
    static_assert(sizeof(LaunchKernelAttr) == 20, "aclrtLaunchKernelAttr ABI drift");
    constexpr int32_t kAttrSchemMode = 1;   // ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE
    constexpr int32_t kAttrTimeoutUs = 8;   // ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US
    constexpr int32_t kAttrEngineType = 3;  // ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE
    constexpr int32_t kEngineAiv = 1;       // ACL_RT_ENGINE_TYPE_AIV
    constexpr uint32_t kAivTimeoutUs = 1091u * 1000000u;
    LaunchKernelAttr attrs[3]{};
    attrs[0].id = kAttrSchemMode;
    attrs[0].value.schemMode = 1;
    attrs[1].id = kAttrTimeoutUs;
    attrs[1].value.timeoutUs.timeoutLow = kAivTimeoutUs;
    attrs[1].value.timeoutUs.timeoutHigh = 0;
    attrs[2].id = kAttrEngineType;
    attrs[2].value.engineType = kEngineAiv;
    struct {
        void* attrs;
        std::size_t numAttrs;
    } cfg{attrs, 3};
    UB_LOG_DEBUG(
        "DlAscendcl launch: api=aclrtLaunchKernelWithHostArgs func={} blockDim={} stream={} "
        "hostArgs={} argsSize={} attrs=[scheme=1,timeoutUs={},engine=AIV]",
        funcHandle, blockDim, stream, hostArgs, argsSize, kAivTimeoutUs);
    int rc = LaunchHostArgsSlot()(funcHandle, blockDim, stream, &cfg, hostArgs, argsSize,
                                  /*ph=*/nullptr, /*phNum=*/0);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtLaunchKernelWithHostArgs failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtLaunchKernelWithDeviceArgs(void* funcHandle, uint32_t blockDim,
                                                     void* stream, const void* deviceArgs,
                                                     std::size_t argsSize)
{
    if (funcHandle == nullptr) { return UbStatus(UbErrorCode::InvalidArgument, "funcHandle null"); }
    if (deviceArgs == nullptr || argsSize == 0) {
        return UbStatus(UbErrorCode::InvalidArgument, "deviceArgs null/empty");
    }
    if (!Loaded() || LaunchKernelV2Slot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "aclrtLaunchKernelV2 symbol missing");
    }
    struct LaunchKernelAttr {
        int32_t id;
        union {
            uint8_t schemMode;
            int32_t engineType;
            struct {
                uint32_t timeoutLow;
                uint32_t timeoutHigh;
            } timeoutUs;
            uint32_t rsv[4];
        } value;
    };
    static_assert(sizeof(LaunchKernelAttr) == 20, "aclrtLaunchKernelAttr ABI drift");
    constexpr int32_t kAttrSchemMode = 1;
    constexpr int32_t kAttrTimeoutUs = 8;
    constexpr int32_t kAttrEngineType = 3;
    constexpr int32_t kEngineAiv = 1;
    constexpr uint32_t kAivTimeoutUs = 1091u * 1000000u;
    LaunchKernelAttr attrs[3]{};
    attrs[0].id = kAttrSchemMode;
    attrs[0].value.schemMode = 1;
    attrs[1].id = kAttrTimeoutUs;
    attrs[1].value.timeoutUs.timeoutLow = kAivTimeoutUs;
    attrs[1].value.timeoutUs.timeoutHigh = 0;
    attrs[2].id = kAttrEngineType;
    attrs[2].value.engineType = kEngineAiv;
    struct {
        void* attrs;
        std::size_t numAttrs;
    } cfg{attrs, 3};
    UB_LOG_DEBUG(
        "DlAscendcl launch: api=aclrtLaunchKernelV2 func={} blockDim={} stream={} "
        "deviceArgs={} argsSize={} attrs=[scheme=1,timeoutUs={},engine=AIV]",
        funcHandle, blockDim, stream, deviceArgs, argsSize, kAivTimeoutUs);
    int rc = LaunchKernelV2Slot()(funcHandle, blockDim, deviceArgs, argsSize, &cfg, stream);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtLaunchKernelV2 failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtSynchronizeStream(void* stream)
{
    if (!Loaded() || SyncSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "aclrtSynchronizeStream unavailable");
    }
    int rc = SyncSlot()(stream);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtSynchronizeStream failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAscendcl::AclrtSynchronizeStreamWithTimeout(void* stream, int32_t timeoutMs)
{
    if (!Loaded()) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtSynchronizeStreamWithTimeout unavailable");
    }
    if (SyncTimeoutSlot() != nullptr) {
        int rc = SyncTimeoutSlot()(stream, timeoutMs);
        if (rc != 0) {
            return UbStatus(UbErrorCode::UdmaCqPollTimeout,
                            "aclrtSynchronizeStreamWithTimeout(timeoutMs=" +
                                std::to_string(timeoutMs) + ") failed rc=" + std::to_string(rc) +
                                " (device kernel may be hung; reset the device)");
        }
        return UbStatus::Ok();
    }
    return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                    "aclrtSynchronizeStreamWithTimeout symbol missing");
}

}  // namespace umc::kv::dl
