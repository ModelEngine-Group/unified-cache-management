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
#ifndef TRANS_DEVICE_BUFFER_CUDA_H
#define TRANS_DEVICE_BUFFER_CUDA_H

#include "trans_assert_cuda.h"
#include "trans_buffer.h"

class TransDeviceNormalBuffer : public TransBuffer {
public:
    TransDeviceNormalBuffer(std::size_t device, std::size_t size, std::size_t number)
        : TransBuffer{device, size, number}
    {
        const auto total = size * number;
        CUDA_ASSERT(cudaSetDevice(device));
        CUDA_ASSERT(cudaMalloc(&addr_, total));
        CUDA_ASSERT(cudaMemset(addr_, 'N', total));
    }
    ~TransDeviceNormalBuffer() override
    {
        CUDA_ASSERT(cudaSetDevice(device_));
        if (addr_) { CUDA_ASSERT(cudaFree(addr_)); }
        addr_ = nullptr;
    }
    std::string Name() const override { return "device::normal::" + std::to_string(device_); }
};

#endif
