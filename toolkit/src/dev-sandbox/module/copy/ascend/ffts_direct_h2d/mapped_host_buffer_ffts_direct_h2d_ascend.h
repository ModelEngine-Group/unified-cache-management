/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef MAPPED_HOST_BUFFER_FFTS_DIRECT_H2D_ASCEND_H
#define MAPPED_HOST_BUFFER_FFTS_DIRECT_H2D_ASCEND_H

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include "ascend/error_handle_ascend.h"
#include "copy_buffer.h"
#include "error_handle.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

class FftsDirectMappedHostBuffer {
public:
    virtual ~FftsDirectMappedHostBuffer() = default;
    virtual void* MappedAt(size_t i) const = 0;
};

inline size_t FftsDirectCheckedTotalBytes(size_t size, size_t number)
{
    ASSERT(number == 0 || size <= std::numeric_limits<size_t>::max() / number);
    const auto total = size * number;
    ASSERT(total > 0);
    return total;
}

inline size_t FftsDirectRoundUpToPageSize(size_t bytes)
{
    constexpr size_t kPageSize = 4096;
    const auto remainder = bytes % kPageSize;
    if (remainder == 0) { return bytes; }
    const auto padding = kPageSize - remainder;
    ASSERT(bytes <= std::numeric_limits<size_t>::max() - padding);
    return bytes + padding;
}

inline bool FftsDirectIsPageAligned(const void* ptr)
{
    constexpr size_t kPageSize = 4096;
    return reinterpret_cast<std::uintptr_t>(ptr) % kPageSize == 0;
}

inline size_t FftsDirectRoundUp(size_t bytes, size_t alignment)
{
    const auto remainder = bytes % alignment;
    if (remainder == 0) { return bytes; }
    const auto padding = alignment - remainder;
    ASSERT(bytes <= std::numeric_limits<size_t>::max() - padding);
    return bytes + padding;
}

inline void* FftsDirectMmapHugeTlb(size_t& bytes, bool useGiganticPages)
{
    constexpr size_t kHugePageSize = 2ull * 1024ull * 1024ull;
    constexpr size_t kGiganticPageSize = 1ull * 1024ull * 1024ull * 1024ull;
    constexpr int kHugePageFlag = 21 << MAP_HUGE_SHIFT;
    constexpr int kGiganticPageFlag = 30 << MAP_HUGE_SHIFT;

    const auto pageSize = useGiganticPages ? kGiganticPageSize : kHugePageSize;
    const auto alignedBytes = FftsDirectRoundUp(bytes, pageSize);
    const auto pageFlag = useGiganticPages ? kGiganticPageFlag : kHugePageFlag;
    constexpr auto prot = PROT_READ | PROT_WRITE;
    const auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | pageFlag;
    auto* ptr = mmap(nullptr, alignedBytes, prot, flags, -1, 0);
    if (ptr != MAP_FAILED) { bytes = alignedBytes; }
    return ptr;
}

inline void* FftsDirectMmapWithTransparentHugePage(size_t& bytes)
{
    constexpr size_t kHugePageSize = 2ull * 1024ull * 1024ull;
    const auto alignedBytes = FftsDirectRoundUp(bytes, kHugePageSize);
    constexpr auto prot = PROT_READ | PROT_WRITE;
    constexpr auto flags = MAP_PRIVATE | MAP_ANONYMOUS;
    auto* ptr = mmap(nullptr, alignedBytes, prot, flags, -1, 0);
    if (ptr != MAP_FAILED) {
        madvise(ptr, alignedBytes, MADV_HUGEPAGE);
        bytes = alignedBytes;
    }
    return ptr;
}

inline void* FftsDirectMmapODirectHostBuffer(size_t& bytes)
{
    constexpr size_t kGiganticPageSize = 1ull * 1024ull * 1024ull * 1024ull;
    const bool useGiganticPages = bytes >= kGiganticPageSize;
    auto* ptr = FftsDirectMmapHugeTlb(bytes, useGiganticPages);
    if (ptr == MAP_FAILED && useGiganticPages) { ptr = FftsDirectMmapHugeTlb(bytes, false); }
    if (ptr == MAP_FAILED) { ptr = FftsDirectMmapWithTransparentHugePage(bytes); }
    return ptr;
}

class FftsMappedHostCopyBuffer : public CopyBuffer, public FftsDirectMappedHostBuffer {
public:
    FftsMappedHostCopyBuffer(size_t device, size_t size, size_t number)
        : CopyBuffer{device, size, number}
    {
        const auto total = FftsDirectCheckedTotalBytes(size, number);
        mappedBytes_ = FftsDirectRoundUpToPageSize(total);
        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(aclrtMallocHost(&addr_, mappedBytes_));
        ASSERT(FftsDirectIsPageAligned(addr_));
        std::memset(addr_, 'h', total);
        ASCEND_ASSERT(aclrtHostRegisterV2(addr_, mappedBytes_, ACL_HOST_REG_MAPPED));
        registered_ = true;
        ASCEND_ASSERT(aclrtHostGetDevicePointer(addr_, &mappedAddr_, 0));
    }

    ~FftsMappedHostCopyBuffer() override
    {
        if (addr_ != nullptr) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            if (registered_) { ASCEND_ASSERT(aclrtHostUnregister(addr_)); }
            ASCEND_ASSERT(aclrtFreeHost(addr_));
            addr_ = nullptr;
        }
    }

    void* MappedAt(size_t i) const override
    {
        ASSERT(i < number_);
        return static_cast<void*>(static_cast<char*>(mappedAddr_) + i * size_);
    }

    std::string Name() const override
    {
        return "acl::host_mapped::" + std::to_string(device_);
    }

private:
    void* mappedAddr_ = nullptr;
    size_t mappedBytes_ = 0;
    bool registered_ = false;
};

class FftsODirectMappedHostCopyBuffer : public CopyBuffer, public FftsDirectMappedHostBuffer {
public:
    FftsODirectMappedHostCopyBuffer(size_t device, size_t size, size_t number)
        : CopyBuffer{device, size, number}
    {
        const auto total = FftsDirectCheckedTotalBytes(size, number);
        mappedBytes_ = total;
        addr_ = FftsDirectMmapODirectHostBuffer(mappedBytes_);
        ASSERT(addr_ != MAP_FAILED);
        ASSERT(FftsDirectIsPageAligned(addr_));
        std::memset(addr_, 'o', total);
        locked_ = (mlock(addr_, mappedBytes_) == 0);

        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(
            aclrtHostRegisterV2(addr_, mappedBytes_, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
        registered_ = true;
        ASCEND_ASSERT(aclrtHostGetDevicePointer(addr_, &mappedAddr_, 0));
    }

    ~FftsODirectMappedHostCopyBuffer() override
    {
        if (addr_ != nullptr && addr_ != MAP_FAILED) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            if (registered_) { ASCEND_ASSERT(aclrtHostUnregister(addr_)); }
            if (locked_) { munlock(addr_, mappedBytes_); }
            munmap(addr_, mappedBytes_);
            addr_ = nullptr;
        }
    }

    void* MappedAt(size_t i) const override
    {
        ASSERT(i < number_);
        return static_cast<void*>(static_cast<char*>(mappedAddr_) + i * size_);
    }

    std::string Name() const override
    {
        return "acl::odirect_mmap::" + std::to_string(device_);
    }

private:
    void* mappedAddr_ = nullptr;
    size_t mappedBytes_ = 0;
    bool registered_ = false;
    bool locked_ = false;
};

class FftsMappedSharedHostRegion : public CopyBuffer {
public:
    FftsMappedSharedHostRegion(std::string tag, size_t device, size_t size, size_t number)
        : CopyBuffer{device, size, number}
    {
        const auto total = FftsDirectCheckedTotalBytes(size, number);
        mappedBytes_ = FftsDirectRoundUpToPageSize(total);
        shmName_ = "/copy_ascend_ffts_direct_" + std::to_string(getpid()) + "_" + tag + "_" +
                   std::to_string(reinterpret_cast<std::uintptr_t>(this));
        const auto fd = shm_open(shmName_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        ASSERT(fd != -1);
        ASSERT(ftruncate(fd, mappedBytes_) == 0);
        constexpr auto prot = PROT_READ | PROT_WRITE;
        constexpr auto flags = MAP_SHARED | MAP_POPULATE;
        addr_ = mmap(nullptr, mappedBytes_, prot, flags, fd, 0);
        const auto closeStatus = close(fd);
        ASSERT(closeStatus == 0);
        ASSERT(addr_ != MAP_FAILED);
        std::memset(addr_, 's', total);
    }

    ~FftsMappedSharedHostRegion() override
    {
        if (addr_ != nullptr) {
            munmap(addr_, mappedBytes_);
            addr_ = nullptr;
        }
        if (!shmName_.empty()) { shm_unlink(shmName_.c_str()); }
    }

    const std::string& ShmName() const { return shmName_; }
    size_t MappedBytes() const { return mappedBytes_; }
    std::string Name() const override { return "acl::shm::0"; }

private:
    std::string shmName_;
    size_t mappedBytes_ = 0;
};

class FftsMappedSharedHostCopyBuffer : public CopyBuffer, public FftsDirectMappedHostBuffer {
public:
    FftsMappedSharedHostCopyBuffer(std::string shmName, size_t mappedBytes, size_t device,
                                   size_t size, size_t number)
        : CopyBuffer{device, size, number}, shmName_{std::move(shmName)}, mappedBytes_{mappedBytes}
    {
        ASSERT(FftsDirectCheckedTotalBytes(size, number) <= mappedBytes_);
        const auto fd = shm_open(shmName_.c_str(), O_RDWR, 0600);
        ASSERT(fd != -1);
        constexpr auto prot = PROT_READ | PROT_WRITE;
        constexpr auto flags = MAP_SHARED | MAP_POPULATE;
        addr_ = mmap(nullptr, mappedBytes_, prot, flags, fd, 0);
        const auto closeStatus = close(fd);
        ASSERT(closeStatus == 0);
        ASSERT(addr_ != MAP_FAILED);

        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(
            aclrtHostRegisterV2(addr_, mappedBytes_, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
        registered_ = true;
        ASCEND_ASSERT(aclrtHostGetDevicePointer(addr_, &mappedAddr_, 0));
    }

    ~FftsMappedSharedHostCopyBuffer() override
    {
        if (addr_ != nullptr) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            if (registered_) { ASCEND_ASSERT(aclrtHostUnregister(addr_)); }
            munmap(addr_, mappedBytes_);
            addr_ = nullptr;
        }
    }

    void* MappedAt(size_t i) const override
    {
        ASSERT(i < number_);
        return static_cast<void*>(static_cast<char*>(mappedAddr_) + i * size_);
    }

    std::string Name() const override { return "acl::shm::0"; }

private:
    std::string shmName_;
    void* mappedAddr_ = nullptr;
    size_t mappedBytes_ = 0;
    bool registered_ = false;
};

#endif  // MAPPED_HOST_BUFFER_FFTS_DIRECT_H2D_ASCEND_H
