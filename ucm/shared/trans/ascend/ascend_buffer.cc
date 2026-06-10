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
#include <sys/mman.h>
#include "logger/logger.h"

namespace UC::Trans {

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
        UC_DEBUG("Mmap({}) with TLB({}) success.", alignedSize, pageSize);
        size = alignedSize;
        return ptr;
    }
    static void* MMapAnonymous(size_t& size, bool useHugePage)
    {
        const auto pageSize = HUGE_PAGE_SIZE;
        const auto alignedSize = (size + pageSize - 1) / pageSize * pageSize;
        const auto prot = PROT_WRITE | PROT_READ;
        const auto flags = MAP_PRIVATE | MAP_ANONYMOUS;
        void* ptr = mmap(nullptr, alignedSize, prot, flags, -1, 0);
        if (ptr == MAP_FAILED) {
            UC_WARN("Mmap({}) anonymous return: {}.", alignedSize, errno);
            return ptr;
        }
        auto ret = madvise(ptr, alignedSize, useHugePage ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
        if (ret != 0) {
            UC_WARN("Madvise {} on mmap({}) return: {}.",
                    useHugePage ? "hugepage" : "nohugepage", alignedSize, errno);
        } else {
            UC_DEBUG("Mmap({}) anonymous success, hugepage advice: {}.", alignedSize, useHugePage);
        }
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
    std::shared_ptr<void> Data(bool useHugePage)
    {
        if (buffer_ != MAP_FAILED) {
            return std::shared_ptr<void>(buffer_, [self = shared_from_this()](auto) {});
        }
        const auto requestedSize = size_;
        const auto useGiganticPages = useHugePage && size_ >= GIGANTIC_PAGE_SIZE;
        UC_DEBUG("Make direct-io host buffer start, requested size: {}, use hugepage: {}, use 1GiB hugepage: {}.",
                 requestedSize, useHugePage, useGiganticPages);
        if (useHugePage) {
            buffer_ = MMapWithTLB(size_, useGiganticPages);
            if (buffer_ == MAP_FAILED && useGiganticPages) {
                UC_DEBUG("Fallback direct-io host buffer mmap from 1GiB hugepage to 2MiB hugepage, requested size: {}.",
                         requestedSize);
                buffer_ = MMapWithTLB(size_, false);
            }
        }
        if (buffer_ == MAP_FAILED) {
            UC_DEBUG("Make direct-io host buffer mmap with anonymous mmap, requested size: {}, use hugepage advice: {}.",
                     requestedSize, useHugePage);
            buffer_ = MMapAnonymous(size_, useHugePage);
        }
        if (buffer_ == MAP_FAILED) {
            UC_ERROR("Failed to make direct-io host buffer, requested size: {}, aligned size: {}.",
                     requestedSize, size_);
            return nullptr;
        }
        UC_DEBUG("Make direct-io host buffer mmap success, requested size: {}, actual size: {}, addr: {}.",
                 requestedSize, size_, buffer_);
        UC_DEBUG("Zero direct-io host buffer start, size: {}, addr: {}.", size_, buffer_);
        std::memset(buffer_, 0, size_);
        UC_DEBUG("Zero direct-io host buffer success, size: {}, addr: {}.", size_, buffer_);
        auto ret = mlock(buffer_, size_);
        if (ret != 0) {
            UC_WARN("Mlock direct-io host buffer failed, size: {}, errno: {}.", size_, errno);
        } else {
            UC_DEBUG("Mlock direct-io host buffer success, size: {}.", size_);
        }
        UC_DEBUG("Register direct-io host buffer start, size: {}, addr: {}.", size_, buffer_);
        auto s = Buffer::RegisterHostBuffer(buffer_, size_);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to register direct-io host buffer, requested size: {}, actual size: {}, addr: {}.",
                     s, requestedSize, size_, buffer_);
            munlock(buffer_, size_);
            munmap(buffer_, size_);
            buffer_ = MAP_FAILED;
            return nullptr;
        }
        UC_DEBUG("Register direct-io host buffer success, size: {}, addr: {}.", size_, buffer_);
        return std::shared_ptr<void>(buffer_, [self = shared_from_this()](auto) {});
    }
};

std::shared_ptr<void> Trans::AscendBuffer::MakeDeviceBuffer(size_t size)
{
    void* device = nullptr;
    auto ret = aclrtMalloc(&device, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(device, aclrtFree); }
    UC_ERROR("Aclrt malloc device buffer failed, size: {}, ret: {}.", size, ret);
    return nullptr;
}

std::shared_ptr<void> Trans::AscendBuffer::MakeHostBuffer(size_t size)
{
    void* host = nullptr;
    auto ret = aclrtMallocHost(&host, size);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(host, aclrtFreeHost); }
    UC_ERROR("Aclrt malloc host buffer failed, size: {}, ret: {}.", size, ret);
    return nullptr;
}

std::shared_ptr<void> Trans::AscendBuffer::MakeHostBuffer4DirectIo(size_t size, bool useHugePage)
{
    try {
        return HostHugePages::Create(size)->Data(useHugePage);
    } catch (...) {
        UC_ERROR("Make direct-io host buffer failed with exception, size: {}.", size);
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

}  // namespace UC::Trans
