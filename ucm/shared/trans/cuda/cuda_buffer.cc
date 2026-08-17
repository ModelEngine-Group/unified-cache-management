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
#include "cuda_buffer.h"
#include <cuda_runtime.h>

namespace UC::Trans {

std::shared_ptr<void> CudaBuffer::MakeDeviceBuffer(size_t size)
{
    void* device = nullptr;
    auto ret = cudaMalloc(&device, size);
    if (ret == cudaSuccess) { return std::shared_ptr<void>(device, cudaFree); }
    return nullptr;
}

std::shared_ptr<void> CudaBuffer::MakeHostBuffer(size_t size)
{
    void* host = nullptr;
    auto ret = cudaMallocHost(&host, size);
    if (ret == cudaSuccess) { return std::shared_ptr<void>(host, cudaFreeHost); }
    return nullptr;
}

std::shared_ptr<void> CudaBuffer::MakeHostMappedDeviceBuffer(size_t size, void** pDevice)
{
    if (pDevice) { *pDevice = nullptr; }

    void* host = nullptr;
    auto ret = cudaHostAlloc(&host, size, cudaHostAllocMapped);
    if (ret != cudaSuccess) { return nullptr; }

    if (pDevice) {
        void* device = nullptr;
        ret = cudaHostGetDevicePointer(&device, host, 0);
        if (ret != cudaSuccess) {
            cudaFreeHost(host);
            return nullptr;
        }
        *pDevice = device;
    }
    return std::shared_ptr<void>(host, cudaFreeHost);
}

Status Buffer::RegisterHostBuffer(void* host, size_t size, void** pDevice)
{
    if (pDevice) { *pDevice = nullptr; }

    const auto flags = pDevice ? cudaHostRegisterMapped : cudaHostRegisterDefault;
    auto ret = cudaHostRegister(host, size, flags);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    if (pDevice) {
        void* device = nullptr;
        ret = cudaHostGetDevicePointer(&device, host, 0);
        if (ret != cudaSuccess) [[unlikely]] {
            cudaHostUnregister(host);
            return Status{ret, cudaGetErrorString(ret)};
        }
        *pDevice = device;
    }
    return Status::OK();
}

void Buffer::UnregisterHostBuffer(void* host) { cudaHostUnregister(host); }

Status Memset(void* ptr, std::size_t size, std::int32_t value)
{
    auto ret = cudaMemset(ptr, value, size);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }

    ret = cudaStreamSynchronize(nullptr);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

}  // namespace UC::Trans
