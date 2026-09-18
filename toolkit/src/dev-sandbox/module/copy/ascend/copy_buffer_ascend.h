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
#ifndef COPY_BUFFER_ASCEND_H
#define COPY_BUFFER_ASCEND_H

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include "copy_buffer.h"
#include "error_handle_ascend.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

inline size_t CheckedTotalBytes(size_t size, size_t number)
{
    ASSERT(number == 0 || size <= std::numeric_limits<size_t>::max() / number);
    const auto total = size * number;
    ASSERT(total > 0);
    return total;
}

inline size_t RoundUpToAlignment(size_t bytes, size_t alignment)
{
    ASSERT(alignment > 0);
    const auto remainder = bytes % alignment;
    if (remainder == 0) { return bytes; }
    const auto padding = alignment - remainder;
    ASSERT(bytes <= std::numeric_limits<size_t>::max() - padding);
    return bytes + padding;
}

inline bool IsPageAligned(const void* ptr)
{
    constexpr size_t kPageSize = 4096;
    return reinterpret_cast<std::uintptr_t>(ptr) % kPageSize == 0;
}

inline void* MmapODirectHugeTlb(size_t& bytes, bool useGiganticPages)
{
    constexpr size_t kHugePageSize = 2ull * 1024ull * 1024ull;
    constexpr size_t kGiganticPageSize = 1ull * 1024ull * 1024ull * 1024ull;
    constexpr int kHugePageFlag = 21 << MAP_HUGE_SHIFT;
    constexpr int kGiganticPageFlag = 30 << MAP_HUGE_SHIFT;

    const auto pageSize = useGiganticPages ? kGiganticPageSize : kHugePageSize;
    const auto alignedBytes = RoundUpToAlignment(bytes, pageSize);
    const auto pageFlag = useGiganticPages ? kGiganticPageFlag : kHugePageFlag;
    constexpr auto prot = PROT_READ | PROT_WRITE;
    const auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | pageFlag;
    auto* ptr = mmap(nullptr, alignedBytes, prot, flags, -1, 0);
    if (ptr != MAP_FAILED) { bytes = alignedBytes; }
    return ptr;
}

inline void* MmapODirectWithTransparentHugePage(size_t& bytes)
{
    constexpr size_t kHugePageSize = 2ull * 1024ull * 1024ull;
    const auto alignedBytes = RoundUpToAlignment(bytes, kHugePageSize);
    constexpr auto prot = PROT_READ | PROT_WRITE;
    constexpr auto flags = MAP_PRIVATE | MAP_ANONYMOUS;
    auto* ptr = mmap(nullptr, alignedBytes, prot, flags, -1, 0);
    if (ptr != MAP_FAILED) {
        madvise(ptr, alignedBytes, MADV_HUGEPAGE);
        bytes = alignedBytes;
    }
    return ptr;
}

inline void* MmapODirectHostBuffer(size_t& bytes)
{
    constexpr size_t kGiganticPageSize = 1ull * 1024ull * 1024ull * 1024ull;
    const bool useGiganticPages = bytes >= kGiganticPageSize;
    auto* ptr = MmapODirectHugeTlb(bytes, useGiganticPages);
    if (ptr == MAP_FAILED && useGiganticPages) { ptr = MmapODirectHugeTlb(bytes, false); }
    if (ptr == MAP_FAILED) { ptr = MmapODirectWithTransparentHugePage(bytes); }
    return ptr;
}

class HostCopyBuffer : public CopyBuffer {
public:
    HostCopyBuffer(size_t device, size_t size, size_t number) : CopyBuffer{device, size, number}
    {
        const auto total = size * number;
        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(aclrtMallocHost(&addr_, total));
        std::memset(addr_, 'h', total);
    }
    ~HostCopyBuffer() override
    {
        if (addr_) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            ASCEND_ASSERT(aclrtFreeHost(addr_));
        }
    }
    std::string Name() const override { return "acl::host::" + std::to_string(device_); }
};

class AnonymousCopyBuffer : public CopyBuffer {
public:
    AnonymousCopyBuffer(size_t device, size_t size, size_t number)
        : CopyBuffer{device, size, number}
    {
        const auto total = size * number;
        ASCEND_ASSERT(aclrtSetDevice(device_));
        constexpr auto prot = PROT_READ | PROT_WRITE;
        constexpr auto flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE;
        addr_ = mmap(nullptr, total, prot, flags, -1, 0);
        ASSERT(addr_ != MAP_FAILED);
        std::memset(addr_, 'a', total);
        ASCEND_ASSERT(aclrtHostRegisterV2(addr_, total, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
    }
    ~AnonymousCopyBuffer() override
    {
        if (addr_) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            ASCEND_ASSERT(aclrtHostUnregister(addr_));
            munmap(addr_, size_ * number_);
        }
    }
    std::string Name() const override { return "acl::anon::" + std::to_string(device_); }
};

class ODirectHostCopyBuffer : public CopyBuffer {
public:
    ODirectHostCopyBuffer(size_t device, size_t size, size_t number)
        : CopyBuffer{device, size, number}
    {
        const auto total = CheckedTotalBytes(size, number);
        mappedBytes_ = total;
        addr_ = MmapODirectHostBuffer(mappedBytes_);
        ASSERT(addr_ != MAP_FAILED);
        ASSERT(IsPageAligned(addr_));
        std::memset(addr_, 'o', total);
        locked_ = (mlock(addr_, mappedBytes_) == 0);

        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(
            aclrtHostRegisterV2(addr_, mappedBytes_, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
        registered_ = true;
    }

    ~ODirectHostCopyBuffer() override
    {
        if (addr_ != nullptr && addr_ != MAP_FAILED) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            if (registered_) { ASCEND_ASSERT(aclrtHostUnregister(addr_)); }
            if (locked_) { munlock(addr_, mappedBytes_); }
            munmap(addr_, mappedBytes_);
            addr_ = nullptr;
        }
    }

    std::string Name() const override { return "acl::odirect_mmap::" + std::to_string(device_); }

private:
    size_t mappedBytes_ = 0;
    bool registered_ = false;
    bool locked_ = false;
};

class SharedHostRegion : public CopyBuffer {
public:
    SharedHostRegion(std::string tag, size_t device, size_t size, size_t number)
        : CopyBuffer{device, size, number}
    {
        shmName_ = "/copy_ascend_" + std::to_string(getpid()) + "_" + tag + "_" +
                   std::to_string(reinterpret_cast<std::uintptr_t>(this));
        const auto total = CheckedTotalBytes(size, number);
        const auto fd = shm_open(shmName_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        ASSERT(fd != -1);
        ASSERT(ftruncate(fd, total) == 0);
        constexpr auto prot = PROT_READ | PROT_WRITE;
        constexpr auto flags = MAP_SHARED | MAP_POPULATE;
        addr_ = mmap(nullptr, total, prot, flags, fd, 0);
        const auto closeStatus = close(fd);
        ASSERT(closeStatus == 0);
        ASSERT(addr_ != MAP_FAILED);
        std::memset(addr_, 's', total);
    }

    ~SharedHostRegion() override
    {
        if (addr_) { munmap(addr_, size_ * number_); }
        if (!shmName_.empty()) { shm_unlink(shmName_.c_str()); }
    }

    const std::string& ShmName() const { return shmName_; }
    std::string Name() const override { return "acl::shm::0"; }

private:
    std::string shmName_;
};

class SharedHostCopyBuffer : public CopyBuffer {
public:
    SharedHostCopyBuffer(std::string shmName, size_t device, size_t size, size_t number)
        : CopyBuffer{device, size, number}, shmName_{std::move(shmName)}
    {
        const auto total = CheckedTotalBytes(size, number);
        const auto fd = shm_open(shmName_.c_str(), O_RDWR, 0600);
        ASSERT(fd != -1);
        constexpr auto prot = PROT_READ | PROT_WRITE;
        constexpr auto flags = MAP_SHARED | MAP_POPULATE;
        addr_ = mmap(nullptr, total, prot, flags, fd, 0);
        const auto closeStatus = close(fd);
        ASSERT(closeStatus == 0);
        ASSERT(addr_ != MAP_FAILED);
        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(aclrtHostRegisterV2(addr_, total, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
    }

    ~SharedHostCopyBuffer() override
    {
        if (addr_) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            ASCEND_ASSERT(aclrtHostUnregister(addr_));
            munmap(addr_, size_ * number_);
        }
    }

    std::string Name() const override { return "acl::shm::0"; }

private:
    std::string shmName_;
};

class DeviceCopyBuffer : public CopyBuffer {
public:
    DeviceCopyBuffer(size_t device, size_t size, size_t number) : CopyBuffer{device, size, number}
    {
        const auto total = size * number;
        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(aclrtMalloc(&addr_, total, ACL_MEM_MALLOC_HUGE_FIRST));
        ASCEND_ASSERT(aclrtMemset(addr_, total, 'd', total));
    }
    ~DeviceCopyBuffer() override
    {
        if (addr_) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            ASCEND_ASSERT(aclrtFree(addr_));
        }
    }
    std::string Name() const override { return "acl::device::" + std::to_string(device_); }
};

#endif  // COPY_BUFFER_ASCEND_H
