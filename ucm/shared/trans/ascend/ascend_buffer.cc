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
#include "ascend_buffer.h"
#include <acl/acl.h>
#include <limits>
#include <sys/mman.h>
#include "logger/logger.h"

namespace UC::Trans {

namespace {

constexpr std::uintptr_t HOST_REGISTER_PAGE_SIZE = 4096;

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

void ReleaseHostPinnedMemory(void* registeredHost, void* allocatedHost)
{
    Buffer::UnregisterHostBuffer(registeredHost);
    FreeHostMemory(allocatedHost);
}

void ReleaseCpuAccessibleDeviceMemory(void* mappedAddress, aclrtDrvMemHandle handle)
{
    auto ret = aclrtUnmapMem(mappedAddress);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed to unmap CPU-accessible device memory addr={} ret={}", mappedAddress, ret);
    }
    ret = aclrtReleaseMemAddress(mappedAddress);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed to release CPU-accessible device address addr={} ret={}", mappedAddress,
                 ret);
    }
    ret = aclrtFreePhysical(handle);
    if (ret != ACL_SUCCESS) {
        UC_ERROR("Failed to free physical device memory handle={} ret={}", handle, ret);
    }
}

}  // namespace

class HostHugePages : public std::enable_shared_from_this<HostHugePages> {
    struct ConstructorKey {};
    static constexpr auto HUGE_PAGE_SIZE = 2UL << 20;
    static constexpr auto GIGANTIC_PAGE_SIZE = 1UL << 30;
    static constexpr auto HUGE_PAGE_FLAG = 21 << MAP_HUGE_SHIFT;
    static constexpr auto GIGANTIC_PAGE_FLAG = 30 << MAP_HUGE_SHIFT;
    size_t size_;
    void* buffer_;

    static void* MMapWithTLB(size_t& size, bool useGiganticPages)
    {
        const auto pageSize = useGiganticPages ? GIGANTIC_PAGE_SIZE : HUGE_PAGE_SIZE;
        const auto alignedSize = (size + pageSize - 1) / pageSize * pageSize;
        const auto pageFlag = useGiganticPages ? GIGANTIC_PAGE_FLAG : HUGE_PAGE_FLAG;
        const auto prot = PROT_WRITE | PROT_READ;
        const auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | pageFlag;
        void* ptr = mmap(nullptr, alignedSize, prot, flags, -1, 0);
        if (ptr == MAP_FAILED) {
            UC_WARN("Mmap({}) with TLB({}) return: {}.", alignedSize, pageSize, errno);
            return ptr;
        }
        size = alignedSize;
        return ptr;
    }
    static void* MMapWithAdvice(size_t& size)
    {
        const auto pageSize = HUGE_PAGE_SIZE;
        const auto alignedSize = (size + pageSize - 1) / pageSize * pageSize;
        const auto prot = PROT_WRITE | PROT_READ;
        const auto flags = MAP_PRIVATE | MAP_ANONYMOUS;
        void* ptr = mmap(nullptr, alignedSize, prot, flags, -1, 0);
        if (ptr == MAP_FAILED) {
            UC_WARN("Mmap({}) with advice({}) return: {}.", alignedSize, pageSize, errno);
            return ptr;
        }
        madvise(ptr, alignedSize, MADV_HUGEPAGE);
        size = alignedSize;
        return ptr;
    }

public:
    HostHugePages(size_t size, ConstructorKey) : size_(size), buffer_(MAP_FAILED) {}
    static std::shared_ptr<HostHugePages> Create(size_t size)
    {
        return std::make_shared<HostHugePages>(size, ConstructorKey{});
    }
    ~HostHugePages()
    {
        if (buffer_ == MAP_FAILED) { return; }
        Buffer::UnregisterHostBuffer(buffer_);
        munlock(buffer_, size_);
        munmap(buffer_, size_);
    }
    std::shared_ptr<void> Data()
    {
        if (buffer_ != MAP_FAILED) {
            return std::shared_ptr<void>(buffer_, [self = shared_from_this()](auto) {});
        }
        const auto useGiganticPages = size_ >= GIGANTIC_PAGE_SIZE;
        buffer_ = MMapWithTLB(size_, useGiganticPages);
        if (buffer_ == MAP_FAILED && useGiganticPages) { buffer_ = MMapWithTLB(size_, false); }
        if (buffer_ == MAP_FAILED) { buffer_ = MMapWithAdvice(size_); }
        if (buffer_ == MAP_FAILED) {
            UC_ERROR("Failed to make host buffer({}).", size_);
            return nullptr;
        }
        std::memset(buffer_, 0, size_);
        mlock(buffer_, size_);
        auto s = Buffer::RegisterHostBuffer(buffer_, size_);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to register buffer({}).", s, size_);
            munlock(buffer_, size_);
            munmap(buffer_, size_);
            buffer_ = MAP_FAILED;
            return nullptr;
        }
        return std::shared_ptr<void>(buffer_, [self = shared_from_this()](auto) {});
    }
};

std::shared_ptr<void> Trans::AscendBuffer::MakeDeviceBuffer(size_t size)
{
    void* device = nullptr;
    auto ret = aclrtMalloc(&device, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(device, aclrtFree); }
    return nullptr;
}

std::shared_ptr<void> Trans::AscendBuffer::MakeCpuAccessibleDeviceBuffer(size_t size)
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
        ReleaseCpuAccessibleDeviceMemory(mappedAddress, handle);
        return nullptr;
    }
#else
    UC_ERROR(
        "Unsupported ACL version {}.{}: CPU-accessible device memory requires "
        "aclrtMemSetAccess, minimum supported ACL version is 1.16",
        ACL_MAJOR_VERSION, ACL_MINOR_VERSION);
    ReleaseCpuAccessibleDeviceMemory(mappedAddress, handle);
    return nullptr;
#endif
    return std::shared_ptr<void>(mappedAddress, [handle](void* address) {
        ReleaseCpuAccessibleDeviceMemory(address, handle);
    });
}

std::shared_ptr<void> Trans::AscendBuffer::MakeHostBuffer(size_t size)
{
    void* host = nullptr;
    auto ret = aclrtMallocHost(&host, size);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(host, aclrtFreeHost); }
    return nullptr;
}

std::shared_ptr<void> Trans::AscendBuffer::MakeHostPinnedBuffer(size_t size, void** pDevice)
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
    auto status = Buffer::RegisterHostBuffer(host, size, &device);
    if (status.Failure()) {
        UC_ERROR("Failed to register host-pinned memory addr={} size={} status={}", host, size,
                 status);
        FreeHostMemory(allocatedHost);
        return nullptr;
    }

    if (pDevice) { *pDevice = device; }
    return std::shared_ptr<void>(host, [allocatedHost](void* registeredHost) {
        ReleaseHostPinnedMemory(registeredHost, allocatedHost);
    });
}

std::shared_ptr<void> Trans::AscendBuffer::MakeHostBuffer4DirectIo(size_t size)
{
    try {
        return HostHugePages::Create(size)->Data();
    } catch (...) {
        return nullptr;
    }
}

Status Buffer::RegisterHostBuffer(void* host, size_t size, void** pDevice)
{
    void* device = nullptr;
#if ASCEND_SUPPORTS_REGISTER_PIN
    auto ret = aclrtHostRegisterV2(host, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    if (pDevice) { ret = aclrtHostGetDevicePointer(host, &device, 0); }
#else
    auto ret = aclrtHostRegister(host, size, ACL_HOST_REGISTER_MAPPED, &device);
#endif
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    if (pDevice) { *pDevice = device; }
    return Status::OK();
}

void Buffer::UnregisterHostBuffer(void* host) { aclrtHostUnregister(host); }

Status ZeroDeviceMemory(void* ptr, std::size_t size)
{
    if (aclrtMemset(ptr, size, 0, size) != ACL_SUCCESS) {
        return Status::Error("aclrtMemset failed to zero device memory");
    }
    return Status::OK();
}

}  // namespace UC::Trans
