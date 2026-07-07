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

#include <cstring>
#include <sys/mman.h>
#include "copy_buffer.h"
#include "error_handle_ascend.h"

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
