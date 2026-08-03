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

#include "src/runtime/acl_rt_loader.h"
#include <dlfcn.h>
#include <string>
#include "src/ub/log.h"

namespace umc::comm::acl {

std::mutex& DlAclRt::Mu()
{
    static std::mutex m;
    return m;
}
void*& DlAclRt::Handle()
{
    static void* h = nullptr;
    return h;
}
std::atomic_bool& DlAclRt::Loaded()
{
    static std::atomic_bool b{false};
    return b;
}
DlAclRt::MallocFunc& DlAclRt::MallocSlot()
{
    static MallocFunc f = nullptr;
    return f;
}
DlAclRt::FreeFunc& DlAclRt::FreeSlot()
{
    static FreeFunc f = nullptr;
    return f;
}
DlAclRt::MemsetFunc& DlAclRt::MemsetSlot()
{
    static MemsetFunc f = nullptr;
    return f;
}
DlAclRt::MemcpyFunc& DlAclRt::MemcpySlot()
{
    static MemcpyFunc f = nullptr;
    return f;
}
DlAclRt::SetDeviceFunc& DlAclRt::SetDeviceSlot()
{
    static SetDeviceFunc f = nullptr;
    return f;
}
DlAclRt::SetDeviceFunc& DlAclRt::ResetDeviceSlot()
{
    static SetDeviceFunc f = nullptr;
    return f;
}
DlAclRt::InitFunc& DlAclRt::InitSlot()
{
    static InitFunc f = nullptr;
    return f;
}
DlAclRt::FinalizeFunc& DlAclRt::FinalizeSlot()
{
    static FinalizeFunc f = nullptr;
    return f;
}
DlAclRt::GetPhyIdFunc& DlAclRt::GetPhyIdSlot()
{
    static GetPhyIdFunc f = nullptr;
    return f;
}
DlAclRt::GetCtxFunc& DlAclRt::GetCtxSlot()
{
    static GetCtxFunc f = nullptr;
    return f;
}
DlAclRt::SetCtxFunc& DlAclRt::SetCtxSlot()
{
    static SetCtxFunc f = nullptr;
    return f;
}

bool DlAclRt::IsLoaded() { return Loaded().load(std::memory_order_acquire); }

UbStatus DlAclRt::LoadLibrary()
{
    std::lock_guard<std::mutex> lk(Mu());
    if (Loaded().load(std::memory_order_relaxed)) return UbStatus::Ok();

    void* h = dlopen("libascendcl.so", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) {
        const char* error = dlerror();
        UB_LOG_WARN(
            "DlAclRt: dlopen libascendcl.so failed: {}; "
            "device-side AivInfo allocation needs CANN runtime",
            error != nullptr ? error : "(no info)");
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "libascendcl.so not available");
    }
    auto resolve = [&](const char* name, void** outPtr) -> bool {
        void* sym = dlsym(h, name);
        if (sym == nullptr) {
            const char* error = dlerror();
            UB_LOG_ERROR("DlAclRt: dlsym {} failed: {}", name,
                         error != nullptr ? error : "(no info)");
            return false;
        }
        *outPtr = sym;
        return true;
    };
    void *m = nullptr, *f = nullptr, *ms = nullptr, *cp = nullptr, *sd = nullptr, *rd = nullptr,
         *ai = nullptr, *af = nullptr, *gp = nullptr, *gc = nullptr, *sc = nullptr;
    bool ok = resolve("aclrtMalloc", &m) && resolve("aclrtFree", &f) &&
              resolve("aclrtMemset", &ms) && resolve("aclrtMemcpy", &cp) &&
              resolve("aclrtSetDevice", &sd) && resolve("aclrtResetDevice", &rd);
    (void)resolve("aclInit", &ai);
    (void)resolve("aclFinalize", &af);
    (void)resolve("aclrtGetPhyDevIdByLogicDevId", &gp);
    (void)resolve("aclrtGetCurrentContext", &gc);
    (void)resolve("aclrtSetCurrentContext", &sc);
    if (!ok) {
        dlclose(h);
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "missing ascendcl symbols");
    }
    Handle() = h;
    MallocSlot() = reinterpret_cast<MallocFunc>(m);
    FreeSlot() = reinterpret_cast<FreeFunc>(f);
    MemsetSlot() = reinterpret_cast<MemsetFunc>(ms);
    MemcpySlot() = reinterpret_cast<MemcpyFunc>(cp);
    SetDeviceSlot() = reinterpret_cast<SetDeviceFunc>(sd);
    ResetDeviceSlot() = reinterpret_cast<SetDeviceFunc>(rd);
    InitSlot() = reinterpret_cast<InitFunc>(ai);
    FinalizeSlot() = reinterpret_cast<FinalizeFunc>(af);
    GetPhyIdSlot() = reinterpret_cast<GetPhyIdFunc>(gp);
    GetCtxSlot() = reinterpret_cast<GetCtxFunc>(gc);
    SetCtxSlot() = reinterpret_cast<SetCtxFunc>(sc);
    Loaded().store(true, std::memory_order_release);
    UB_LOG_DEBUG("DlAclRt: libascendcl.so loaded ok");
    return UbStatus::Ok();
}

void DlAclRt::CleanUpLibrary()
{
    // Calls use resolved function pointers without a per-call lock. Keep the
    // successfully loaded library resident for the process lifetime.
}

UbStatus DlAclRt::SetDevice(int32_t deviceId)
{
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    SetDeviceFunc fn = SetDeviceSlot();
    if (fn == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "aclrtSetDevice not loaded");
    }
    int rc = fn(deviceId);
    if (rc != 0 && InitSlot() != nullptr) {
        (void)InitSlot()(nullptr);
        rc = fn(deviceId);
    }
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtSetDevice failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAclRt::ResetDevice(int32_t deviceId)
{
    if (!Loaded() || ResetDeviceSlot() == nullptr) return UbStatus::Ok();
    int rc = ResetDeviceSlot()(deviceId);
    if (rc != 0) {
        UB_LOG_ERROR("DlAclRt: aclrtResetDevice({}) rc={}", deviceId, rc);
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtResetDevice failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAclRt::Finalize()
{
    if (!Loaded()) return UbStatus::Ok();
    FinalizeFunc fn = FinalizeSlot();
    if (fn == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "aclFinalize not loaded");
    }
    int rc = fn();
    if (rc != 0) {
        UB_LOG_ERROR("DlAclRt: aclFinalize rc={}", rc);
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclFinalize failed rc=" + std::to_string(rc));
    }
    UB_LOG_DEBUG("DlAclRt: aclFinalize done");
    return UbStatus::Ok();
}

bool DlAclRt::HasContextApi()
{
    if (!Loaded()) {
        if (LoadLibrary().IsError()) return false;
    }
    return GetCtxSlot() != nullptr && SetCtxSlot() != nullptr;
}

UbStatus DlAclRt::GetCurrentContext(void** ctx)
{
    if (ctx == nullptr) return UbStatus(UbErrorCode::InvalidArgument, "ctx null");
    *ctx = nullptr;
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    GetCtxFunc fn = GetCtxSlot();
    if (fn == nullptr) {
        return UbStatus(UbErrorCode::NotImplemented, "aclrtGetCurrentContext not available");
    }
    int rc = fn(ctx);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtGetCurrentContext failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAclRt::SetCurrentContext(void* ctx)
{
    if (ctx == nullptr) return UbStatus(UbErrorCode::InvalidArgument, "ctx null");
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    SetCtxFunc fn = SetCtxSlot();
    if (fn == nullptr) {
        return UbStatus(UbErrorCode::NotImplemented, "aclrtSetCurrentContext not available");
    }
    int rc = fn(ctx);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtSetCurrentContext failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAclRt::GetPhyDevIdByLogicDevId(int32_t logicDevId, int32_t* phyDevId)
{
    if (phyDevId == nullptr) return UbStatus(UbErrorCode::InvalidArgument, "phyDevId null");
    if (!Loaded()) {
        auto st = LoadLibrary();
        if (st.IsError()) return st;
    }
    GetPhyIdFunc fn = GetPhyIdSlot();
    if (fn == nullptr) {
        *phyDevId = logicDevId;
        UB_LOG_WARN(
            "DlAclRt: aclrtGetPhyDevIdByLogicDevId is unavailable; using logicId={} as phyId. "
            "HCCP RaInit may return -ENODEV for multi-device or non-contiguous phyId layouts",
            logicDevId);
        return UbStatus::Ok();
    }
    int32_t phy = -1;
    int rc = fn(logicDevId, &phy);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtGetPhyDevIdByLogicDevId(logicId=" + std::to_string(logicDevId) +
                            ") failed rc=" + std::to_string(rc));
    }
    *phyDevId = phy;
    return UbStatus::Ok();
}

UbStatus DlAclRt::Malloc(void** devPtr, std::size_t size, MallocPolicy policy)
{
    if (devPtr == nullptr) return UbStatus(UbErrorCode::InvalidArgument, "devPtr null");
    if (!Loaded() || MallocSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "acl rt not loaded");
    }
    int rc = MallocSlot()(devPtr, size, static_cast<int>(policy));
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtMalloc failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAclRt::Free(void* devPtr)
{
    if (devPtr == nullptr) return UbStatus::Ok();
    if (!Loaded() || FreeSlot() == nullptr) return UbStatus::Ok();
    int rc = FreeSlot()(devPtr);
    if (rc != 0) {
        UB_LOG_ERROR("DlAclRt: aclrtFree rc={}", rc);
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtFree failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAclRt::Memset(void* devPtr, std::size_t maxCount, int32_t value, std::size_t count)
{
    if (!Loaded() || MemsetSlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "acl rt not loaded");
    }
    int rc = MemsetSlot()(devPtr, maxCount, value, count);
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtMemset failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

UbStatus DlAclRt::Memcpy(void* dst, std::size_t dstMax, const void* src, std::size_t count,
                         MemcpyKind kind)
{
    if (!Loaded() || MemcpySlot() == nullptr) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed, "acl rt not loaded");
    }
    int rc = MemcpySlot()(dst, dstMax, src, count, static_cast<int>(kind));
    if (rc != 0) {
        return UbStatus(UbErrorCode::HccpV2LoadLibraryFailed,
                        "aclrtMemcpy failed rc=" + std::to_string(rc));
    }
    return UbStatus::Ok();
}

}  // namespace umc::comm::acl
