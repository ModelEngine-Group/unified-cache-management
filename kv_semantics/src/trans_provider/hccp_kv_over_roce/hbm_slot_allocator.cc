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
#include "hbm_slot_allocator.h"

#ifdef UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER

#include <acl/acl.h>
#include <limits>
#include <string>
#include <utility>
#include "logger.h"

namespace kv {
namespace {

Status AclError(const std::string& operation, aclError ret)
{
    std::string message = operation + " failed ret=" + std::to_string(static_cast<int>(ret));
    if (const char* recent = aclGetRecentErrMsg(); recent != nullptr && recent[0] != '\0') {
        message += " msg=";
        message += recent;
    }
    KV_ERROR("HbmSlotAllocator: {}", message);
    return Status::Error(StatusCode::INTERNAL_ERROR, std::move(message));
}

// Switches to the provider's ACL device for one HBM operation and restores the
// caller's context (or device) on exit. Every allocation happens on one device,
// but the caller may invoke the provider while another device is current.
class ScopedAclDevice final {
public:
    explicit ScopedAclDevice(std::int32_t requestedDevice)
    {
        std::int32_t currentDevice = -1;
        if (aclrtGetDevice(&currentDevice) == ACL_SUCCESS) { callerDevice_ = currentDevice; }

        aclrtContext currentContext = nullptr;
        if (aclrtGetCurrentContext(&currentContext) == ACL_SUCCESS && currentContext != nullptr) {
            callerContext_ = currentContext;
        }

        targetDevice_ = requestedDevice >= 0
                            ? requestedDevice
                            : (callerDevice_ >= 0 ? callerDevice_ : kDefaultAclDevice);
        if (callerDevice_ == targetDevice_) { return; }

        const auto initRet = aclInit(nullptr);
        if (initRet != ACL_SUCCESS && initRet != ACL_ERROR_REPEAT_INITIALIZE) {
            status_ = AclError("aclInit", initRet);
            return;
        }

        const auto setRet = aclrtSetDevice(targetDevice_);
        if (setRet != ACL_SUCCESS) {
            status_ = AclError("aclrtSetDevice device_id=" + std::to_string(targetDevice_), setRet);
            return;
        }
        deviceChanged_ = true;
    }

    ~ScopedAclDevice()
    {
        if (!deviceChanged_) { return; }

        // Prefer restoring the caller's exact context: aclrtSetDevice switches to
        // the device's default context, which would silently replace a caller
        // context created outside this provider.
        if (callerContext_ != nullptr) {
            const auto ret = aclrtSetCurrentContext(callerContext_);
            if (ret != ACL_SUCCESS) {
                KV_WARN("HbmSlotAllocator: failed to restore ACL context context={} ret={}",
                        static_cast<const void*>(callerContext_), static_cast<int>(ret));
            }
            return;
        }

        if (callerDevice_ < 0) { return; }
        const auto ret = aclrtSetDevice(callerDevice_);
        if (ret != ACL_SUCCESS) {
            KV_WARN("HbmSlotAllocator: failed to restore ACL device_id={} ret={}", callerDevice_,
                    static_cast<int>(ret));
        }
    }

    ScopedAclDevice(const ScopedAclDevice&) = delete;
    ScopedAclDevice& operator=(const ScopedAclDevice&) = delete;
    ScopedAclDevice(ScopedAclDevice&&) = delete;
    ScopedAclDevice& operator=(ScopedAclDevice&&) = delete;

    const Status& status() const { return status_; }
    std::int32_t targetDevice() const { return targetDevice_; }

private:
    static constexpr std::int32_t kDefaultAclDevice = 0;

    std::int32_t callerDevice_{-1};
    aclrtContext callerContext_{nullptr};
    std::int32_t targetDevice_{kDefaultAclDevice};
    bool deviceChanged_{false};
    Status status_;
};

}  // namespace

HbmSlotAllocator::HbmSlotAllocator(std::int32_t deviceId, std::size_t slotBytes,
                                   std::uint32_t capacity)
    : deviceId_(deviceId), slotBytes_(slotBytes), capacity_(capacity)
{
}

HbmSlotAllocator::~HbmSlotAllocator()
{
    if (base_ == nullptr) { return; }

    ScopedAclDevice device(deviceId_);
    if (!device.status().ok()) {
        KV_WARN("HbmSlotAllocator: freeing HBM without device binding: {}",
                device.status().message);
    }
    (void)aclrtFree(base_);
    base_ = nullptr;
}

Status HbmSlotAllocator::Init()
{
    if (initialized_) { return Status::OK(); }
    if (slotBytes_ == 0 || capacity_ == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HbmSlotAllocator: slot size and capacity must be non-zero");
    }
    if (slotBytes_ > std::numeric_limits<std::size_t>::max() / capacity_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HbmSlotAllocator: HBM region size overflows");
    }

    ScopedAclDevice device(deviceId_);
    if (!device.status().ok()) { return device.status(); }

    const auto bytes = slotBytes_ * capacity_;
    void* base = nullptr;
    const auto mallocRet = aclrtMalloc(&base, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (mallocRet != ACL_SUCCESS || base == nullptr) {
        if (base != nullptr) { (void)aclrtFree(base); }
        return AclError("aclrtMalloc bytes=" + std::to_string(bytes), mallocRet);
    }

    const auto memsetRet = aclrtMemset(base, bytes, 0, bytes);
    if (memsetRet != ACL_SUCCESS) {
        (void)aclrtFree(base);
        return AclError("aclrtMemset bytes=" + std::to_string(bytes), memsetRet);
    }

    base_ = base;
    deviceId_ = device.targetDevice();
    indexPool_.Setup(capacity_);
    initialized_ = true;
    KV_INFO(
        "HbmSlotAllocator: HBM region ready device_id={} base={} slot_bytes={} capacity={} "
        "total_bytes={}",
        deviceId_, base_, slotBytes_, capacity_, bytes);
    return Status::OK();
}

Status HbmSlotAllocator::Acquire(std::uint32_t& slotId)
{
    if (!initialized_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HbmSlotAllocator: HBM region is not initialized");
    }

    const auto index = indexPool_.Acquire();
    if (index == IndexPool::npos) {
        return Status::Error(StatusCode::RESOURCE_BUSY,
                             "HbmSlotAllocator: no free HBM connection slot");
    }
    slotId = index;
    return Status::OK();
}

Status HbmSlotAllocator::Release(std::uint32_t slotId)
{
    if (!initialized_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HbmSlotAllocator: HBM region is not initialized");
    }
    if (!ValidSlot(slotId)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HbmSlotAllocator: invalid slot id " + std::to_string(slotId));
    }
    indexPool_.Release(slotId);
    return Status::OK();
}

Status HbmSlotAllocator::Write(std::uint32_t slotId, const void* data, std::size_t size)
{
    if (!initialized_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HbmSlotAllocator: HBM region is not initialized");
    }
    if (!ValidSlot(slotId) || data == nullptr || size == 0 || size > slotBytes_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HbmSlotAllocator: invalid write slot_id=" + std::to_string(slotId) +
                                 " size=" + std::to_string(size));
    }

    ScopedAclDevice device(deviceId_);
    if (!device.status().ok()) { return device.status(); }

    const auto ret = aclrtMemcpy(SlotAddress(slotId), slotBytes_, data, size,
                                 ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return AclError("aclrtMemcpy slot_id=" + std::to_string(slotId) +
                            " size=" + std::to_string(size),
                        ret);
    }
    return Status::OK();
}

Status HbmSlotAllocator::Clear(std::uint32_t slotId)
{
    if (!initialized_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "HbmSlotAllocator: HBM region is not initialized");
    }
    if (!ValidSlot(slotId)) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "HbmSlotAllocator: invalid slot id " + std::to_string(slotId));
    }

    ScopedAclDevice device(deviceId_);
    if (!device.status().ok()) { return device.status(); }

    const auto ret = aclrtMemset(SlotAddress(slotId), slotBytes_, 0, slotBytes_);
    if (ret != ACL_SUCCESS) {
        return AclError("aclrtMemset slot_id=" + std::to_string(slotId), ret);
    }
    return Status::OK();
}

bool HbmSlotAllocator::ValidSlot(std::uint32_t slotId) const noexcept
{
    return slotId < capacity_;
}

void* HbmSlotAllocator::SlotAddress(std::uint32_t slotId) const noexcept
{
    return static_cast<void*>(static_cast<std::uint8_t*>(base_) +
                              static_cast<std::size_t>(slotId) * slotBytes_);
}

}  // namespace kv

#endif  // UCM_ASU_ENABLE_HCCP_KV_OVER_ROCE_PROVIDER
