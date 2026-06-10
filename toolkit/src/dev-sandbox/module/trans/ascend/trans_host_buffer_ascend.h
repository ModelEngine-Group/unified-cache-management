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
#ifndef TRANS_HOST_BUFFER_ASCEND_H
#define TRANS_HOST_BUFFER_ASCEND_H

#include <sys/mman.h>
#include "trans_assert_ascend.h"
#include "trans_buffer.h"

class TransHostNormalBuffer : public TransBuffer {
public:
    TransHostNormalBuffer(std::size_t device, std::size_t size, std::size_t number)
        : TransBuffer{device, size, number}
    {
        const auto total = size * number;
        ASCEND_ASSERT(aclrtSetDevice(device));
        ASCEND_ASSERT(aclrtMallocHost(&addr_, total));
        std::memset(addr_, 'n', total);
    }
    ~TransHostNormalBuffer() override
    {
        ASCEND_ASSERT(aclrtSetDevice(device_));
        if (addr_) { ASCEND_ASSERT(aclrtFreeHost(addr_)); }
        addr_ = nullptr;
    }
    std::string Name() const override { return "host::normal::" + std::to_string(device_); }
};

class TransHostAnonymousBuffer : public TransBuffer {
public:
    TransHostAnonymousBuffer(std::size_t device, std::size_t size, std::size_t number)
        : TransBuffer{device, size, number}
    {
        const auto total = size * number;
        ASCEND_ASSERT(aclrtSetDevice(device));
        const auto prot = PROT_WRITE | PROT_READ;
        const auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
        addr_ = mmap(nullptr, total, prot, flags, -1, 0);
        std::memset(addr_, 'a', total);
    }
    ~TransHostAnonymousBuffer() override
    {
        ASCEND_ASSERT(aclrtSetDevice(device_));
        const auto total = size_ * number_;
        if (addr_) { munmap(addr_, total); }
        addr_ = nullptr;
    }
    std::string Name() const override { return "host::anon::" + std::to_string(device_); }
};

class TransHostRegisteredBuffer : public TransBuffer {
public:
    TransHostRegisteredBuffer(std::size_t device, std::size_t size, std::size_t number)
        : TransBuffer{device, size, number}
    {
        const auto total = size * number;
        ASCEND_ASSERT(aclrtSetDevice(device));
        const auto prot = PROT_WRITE | PROT_READ;
        const auto flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE;
        addr_ = mmap(nullptr, total, prot, flags, -1, 0);
        std::memset(addr_, 'a', total);
        ASCEND_ASSERT(aclrtHostRegisterV2(addr_, total, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
    }
    ~TransHostRegisteredBuffer() override
    {
        ASCEND_ASSERT(aclrtSetDevice(device_));
        const auto total = size_ * number_;
        if (addr_) {
            ASCEND_ASSERT(aclrtHostUnregister(addr_));
            munmap(addr_, total);
        }
        addr_ = nullptr;
    }
    std::string Name() const override { return "host::reg::" + std::to_string(device_); }
};

#endif
