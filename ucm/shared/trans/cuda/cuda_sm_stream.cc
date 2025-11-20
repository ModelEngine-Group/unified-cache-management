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
#include "cuda_sm_stream.h"
#include "cuda_sm_kernel.h"

namespace UC::Trans {

Status CudaSmStream::DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number)
{
    auto ret = CudaSMCopyAsync(device, host, size, number, stream_);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

Status CudaSmStream::DeviceToHostAsync(void* device[], void* host, size_t size, size_t number)
{
    auto ret = CudaSMCopyAsync(device, host, size, number, stream_);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

Status CudaSmStream::HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number)
{
    auto ret = CudaSMCopyAsync(host, device, size, number, stream_);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

Status CudaSmStream::HostToDeviceAsync(void* host, void* device[], size_t size, size_t number)
{
    auto ret = CudaSMCopyAsync(host, device, size, number, stream_);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

} // namespace UC::Trans
