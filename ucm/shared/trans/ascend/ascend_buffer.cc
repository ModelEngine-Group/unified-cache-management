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

std::shared_ptr<void> Trans::AscendBuffer::MakeHostBuffer(size_t size)
{
    void* host = nullptr;
    auto ret = aclrtMallocHost(&host, size);
    if (ret == ACL_SUCCESS) { return std::shared_ptr<void>(host, aclrtFreeHost); }
    return nullptr;
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

}  // namespace UC::Trans
