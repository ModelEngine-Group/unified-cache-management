/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "worker/device_allocator.h"

#include <cstdlib>
#include <cstring>

#include "kv_client.h"
#include "log/logger_manager.h"

namespace {

/** 值向上对齐到 alignment 的整数倍（alignment 须为正的 2 的幂） */
std::size_t AlignUp(std::size_t value, std::size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

HostAllocator::HostAllocator(bool deviceSemantics) : deviceSemantics_(deviceSemantics) {}

bool HostAllocator::Allocate(std::size_t size, std::uint64_t &addr, std::string &err)
{
    if (size == 0) {
        err = "host memory allocate size must be positive";
        return false;
    }
    // device 语义内存须对齐到协议对齐粒度（真实设备注册约束，模拟时保持同一对齐语义）
    void *ptr = nullptr;
    if (deviceSemantics_) {
        ptr = std::aligned_alloc(kv::kAlignmentBytes, AlignUp(size, kv::kAlignmentBytes));
    } else {
        ptr = std::malloc(size);
    }
    if (ptr == nullptr) {
        err = "host memory allocate failed for " + std::to_string(size) + " bytes";
        return false;
    }
    addr = reinterpret_cast<std::uint64_t>(ptr);
    return true;
}

void HostAllocator::Deallocate(std::uint64_t addr)
{
    if (addr == 0) {
        return;
    }
    std::free(reinterpret_cast<void *>(static_cast<std::uintptr_t>(addr)));
}

bool HostAllocator::Write(std::uint64_t addr, const void *hostData, std::size_t size,
                          std::string &err)
{
    if (addr == 0) {
        err = "host memory write address is null";
        return false;
    }
    if (size != 0) {
        std::memcpy(reinterpret_cast<void *>(static_cast<std::uintptr_t>(addr)), hostData, size);
    }
    return true;
}

bool HostAllocator::Read(void *hostData, std::uint64_t addr, std::size_t size, std::string &err)
{
    if (addr == 0) {
        err = "host memory read address is null";
        return false;
    }
    if (size != 0) {
        std::memcpy(hostData, reinterpret_cast<const void *>(static_cast<std::uintptr_t>(addr)),
                    size);
    }
    return true;
}

bool HostAllocator::IsDeviceMemory() const
{
    return deviceSemantics_;
}

std::unique_ptr<DeviceAllocator> CreateDeviceAllocator(const std::string &bufferType,
                                                       std::int32_t deviceId, std::string &err)
{
    if (bufferType == "host") {
        return std::make_unique<HostAllocator>(false);
    }
    if (bufferType == "device") {
        if (deviceId < 0) {
            err = "deviceid must be >= 0 when buffertype=device";
            return nullptr;
        }
        // 真设备实现挂点：编入 TESTTOOL_ENABLE_ASCEND 后经 ACL 分配设备内存
        // （aclrtMalloc/aclrtMemcpy/aclrtFree）；未编入时回退主机内存模拟 device 语义，
        // memoryType 仍按 DEVICE 注册走完整链路，便于无设备环境先行验证
        LOG_WARN("DeviceAllocator: no device SDK compiled in, buffertype=device falls back to "
                 "host memory simulating device semantics");
        return std::make_unique<HostAllocator>(true);
    }
    err = "invalid buffertype '" + bufferType + "', expect host or device";
    return nullptr;
}
