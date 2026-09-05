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
#include "ascend_buffer.h"
#include <acl/acl.h>
#include <cstdint>
#include <limits>
#include <string>
#include "logger.h"
#include "types.h"

namespace kv::runtime {

namespace {

constexpr std::uintptr_t HOST_REGISTER_PAGE_SIZE = 4096;

Status RegisterHostBuffer(void* host, size_t size, void** pDevice);
void UnregisterHostBuffer(void* host);

void FreeHostMemory(void* host)
{
    auto ret = aclrtFreeHost(host);
    if (ret != ACL_SUCCESS) { UC_ERROR("Failed to free host memory addr={} ret={}", host, ret); }
}

void* AlignUp(void* ptr, std::uintptr_t alignment)
{
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    return reinterpret_cast<void*>((addr + alignment - 1) / alignment * alignment);
}

void ReleaseHostMappedDeviceMemory(void* registeredHost, void* allocatedHost)
{
    UnregisterHostBuffer(registeredHost);
    FreeHostMemory(allocatedHost);
}

void ReleaseDeviceMappedHostMemory(void* mappedAddress, aclrtDrvMemHandle handle)
{
    auto ret = aclrtUnmapMem(mappedAddress);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed to unmap device-mapped host memory addr={} ret={}", mappedAddress, ret);
    }
    ret = aclrtReleaseMemAddress(mappedAddress);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed to release device-mapped host address addr={} ret={}", mappedAddress, ret);
    }
    ret = aclrtFreePhysical(handle);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed to free physical device memory handle={} ret={}", handle, ret);
    }
}

Status RegisterHostBuffer(void* host, size_t size, void** pDevice)
{
    void* device = nullptr;
#if ASCEND_SUPPORTS_REGISTER_PIN
    auto ret = aclrtHostRegisterV2(host, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
    if (ret != ACL_SUCCESS) [[unlikely]] {
        return Status::Error(StatusCode::INTERNAL_ERROR, std::to_string(ret));
    }
    if (pDevice) { ret = aclrtHostGetDevicePointer(host, &device, 0); }
#else
    auto ret = aclrtHostRegister(host, size, ACL_HOST_REGISTER_MAPPED, &device);
#endif
    if (ret != ACL_SUCCESS) [[unlikely]] {
        return Status::Error(StatusCode::INTERNAL_ERROR, std::to_string(ret));
    }
    if (pDevice) { *pDevice = device; }
    return Status::OK();
}

void UnregisterHostBuffer(void* host) { aclrtHostUnregister(host); }

}  // namespace

std::shared_ptr<void> AscendBuffer::MakeDeviceBuffer(size_t size)
{
    void* device = nullptr;
    auto ret = aclrtMalloc(&device, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(device, aclrtFree); }
    return nullptr;
}

std::shared_ptr<void> AscendBuffer::MakeHostBuffer(size_t size)
{
    void* host = nullptr;
    auto ret = aclrtMallocHost(&host, size);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(host, aclrtFreeHost); }
    return nullptr;
}

std::shared_ptr<void> AscendBuffer::MakeDeviceMappedHostBuffer(size_t size)
{
    int32_t deviceId = 0;
    auto ret = aclrtGetDevice(&deviceId);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("aclrtGetDevice failed, size={} ret={}", size, ret);
        return nullptr;
    }

    aclrtPhysicalMemProp prop{};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_NORMAL;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = static_cast<uint32_t>(deviceId);

    size_t granularity = 0;
    ret =
        aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);
    constexpr auto maxSize = std::numeric_limits<size_t>::max();
    if (ret != ACL_SUCCESS || granularity == 0 || size > maxSize - (granularity - 1)) {
        UC_ERROR(
            "Invalid device memory allocation granularity, deviceId={} size={} granularity={} "
            "maxSize={} ret={}",
            deviceId, size, granularity, maxSize, ret);
        return nullptr;
    }
    const auto allocationSize = (size + granularity - 1) / granularity * granularity;

    aclrtDrvMemHandle handle = nullptr;
    ret = aclrtMallocPhysical(&handle, allocationSize, &prop, 0);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("aclrtMallocPhysical failed, deviceId={} size={} ret={}", deviceId, allocationSize,
                 ret);
        return nullptr;
    }

    void* mappedAddress = nullptr;
    ret = aclrtReserveMemAddress(&mappedAddress, allocationSize, 0, nullptr, 0);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("aclrtReserveMemAddress failed, size={} ret={}", allocationSize, ret);
        aclrtFreePhysical(handle);
        return nullptr;
    }

    ret = aclrtMapMem(mappedAddress, allocationSize, 0, handle, 0);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("aclrtMapMem failed, addr={} size={} ret={}", mappedAddress, allocationSize, ret);
        aclrtReleaseMemAddress(mappedAddress);
        aclrtFreePhysical(handle);
        return nullptr;
    }

#if ACL_MAJOR_VERSION > 1 || (ACL_MAJOR_VERSION == 1 && ACL_MINOR_VERSION >= 16)
    aclrtMemAccessDesc accessDesc{};
    accessDesc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    accessDesc.location.type = ACL_MEM_LOCATION_TYPE_HOST;
    accessDesc.location.id = 0;
    ret = aclrtMemSetAccess(mappedAddress, allocationSize, &accessDesc, 1);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed to grant host access to device memory addr={} size={} ret={}",
                 mappedAddress, allocationSize, ret);
        ReleaseDeviceMappedHostMemory(mappedAddress, handle);
        return nullptr;
    }
#else
    UC_ERROR(
        "Unsupported ACL version {}.{}: device-mapped host memory requires "
        "aclrtMemSetAccess, minimum supported ACL version is 1.16",
        ACL_MAJOR_VERSION, ACL_MINOR_VERSION);
    ReleaseDeviceMappedHostMemory(mappedAddress, handle);
    return nullptr;
#endif
    return std::shared_ptr<void>(
        mappedAddress, [handle](void* address) { ReleaseDeviceMappedHostMemory(address, handle); });
}

std::shared_ptr<void> AscendBuffer::MakeHostMappedDeviceBuffer(size_t size, void** pDevice)
{
    if (pDevice) { *pDevice = nullptr; }

    constexpr auto kMaxSize = std::numeric_limits<size_t>::max();
    if (size > kMaxSize - (HOST_REGISTER_PAGE_SIZE - 1)) { return nullptr; }

    void* allocatedHost = nullptr;
    const auto allocationSize = size + HOST_REGISTER_PAGE_SIZE - 1;
    auto ret = aclrtMallocHost(&allocatedHost, allocationSize);
    if (ret != ACL_SUCCESS) { return nullptr; }

    void* host = AlignUp(allocatedHost, HOST_REGISTER_PAGE_SIZE);

    void* device = nullptr;
    auto status = RegisterHostBuffer(host, size, &device);
    if (!status.ok()) {
        UC_ERROR("Failed to register host-mapped device memory addr={} size={} status={}", host,
                 size, status.message);
        FreeHostMemory(allocatedHost);
        return nullptr;
    }

    if (pDevice) { *pDevice = device; }
    return std::shared_ptr<void>(host, [allocatedHost](void* registeredHost) {
        ReleaseHostMappedDeviceMemory(registeredHost, allocatedHost);
    });
}

Status Memset(void* ptr, std::size_t size, std::int32_t value)
{
    if (aclrtMemset(ptr, size, value, size) != ACL_SUCCESS) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "aclrtMemset failed");
    }
    return Status::OK();
}

}  // namespace kv::runtime
