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
#ifndef TRANS_DEVICE_BUFFER_ASCEND_H
#define TRANS_DEVICE_BUFFER_ASCEND_H

#include "trans_assert_ascend.h"
#include "trans_buffer.h"

class TransDeviceNormalBuffer : public TransBuffer {
public:
    TransDeviceNormalBuffer(std::size_t device, std::size_t size, std::size_t number)
        : TransBuffer{device, size, number}
    {
        const auto total = size * number;
        ASCEND_ASSERT(aclrtSetDevice(device));
        ASCEND_ASSERT(aclrtMalloc(&addr_, total, ACL_MEM_MALLOC_HUGE_FIRST));
        ASCEND_ASSERT(aclrtMemset(addr_, total, 'N', total));
    }
    ~TransDeviceNormalBuffer() override
    {
        ASCEND_ASSERT(aclrtSetDevice(device_));
        if (addr_) { ASCEND_ASSERT(aclrtFree(addr_)); }
        addr_ = nullptr;
    }
    std::string Name() const override { return "device::normal::" + std::to_string(device_); }
};

#endif
