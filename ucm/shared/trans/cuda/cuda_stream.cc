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
#include "cuda_stream.h"

namespace UC::Trans {

Status CudaStream::Setup()
{
    auto ret = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (ret != cudaSuccess) { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

Status CudaStream::DeviceToHost(void* device, void* host, size_t size)
{
    auto ret = cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

Status CudaStream::DeviceToHost(void* device[], void* host[], size_t size, size_t number)
{
    auto s = DeviceToHostAsync(device, host, size, number);
    if (s.Failure()) [[unlikely]] { return s; }
    return Synchronized();
}

Status CudaStream::DeviceToHost(void* device[], void* host, size_t size, size_t number)
{
    auto s = DeviceToHostAsync(device, host, size, number);
    if (s.Failure()) [[unlikely]] { return s; }
    return Synchronized();
}

Status CudaStream::DeviceToHostAsync(void* device, void* host, size_t size)
{
    auto ret = cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost, stream_);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

Status CudaStream::DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto s = DeviceToHostAsync(device[i], host[i], size);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

Status CudaStream::DeviceToHostAsync(void* device[], void* host, size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto pHost = (void*)(((int8_t*)host) + size * i);
        auto s = DeviceToHostAsync(device[i], pHost, size);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

Status CudaStream::HostToDevice(void* host, void* device, size_t size)
{
    auto ret = cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

Status CudaStream::HostToDevice(void* host[], void* device[], size_t size, size_t number)
{
    auto s = HostToDeviceAsync(host, device, size, number);
    if (s.Failure()) [[unlikely]] { return s; }
    return Synchronized();
}

Status CudaStream::HostToDevice(void* host, void* device[], size_t size, size_t number)
{
    auto s = HostToDeviceAsync(host, device, size, number);
    if (s.Failure()) [[unlikely]] { return s; }
    return Synchronized();
}

Status CudaStream::HostToDeviceAsync(void* host, void* device, size_t size)
{
    auto ret = cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice, stream_);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

Status CudaStream::HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto s = HostToDeviceAsync(host[i], device[i], size);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

Status Trans::CudaStream::HostToDeviceAsync(void* host, void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto pHost = (void*)(((int8_t*)host) + size * i);
        auto s = HostToDeviceAsync(pHost, device[i], size);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

using Closure = std::function<void(bool)>;

static void Trampoline(cudaStream_t stream, cudaError_t err, void* data)
{
    (void)stream;
    auto c = static_cast<Closure*>(data);
    (*c)(err == cudaSuccess);
    delete c;
}

Status Trans::CudaStream::AppendCallback(std::function<void(bool)> cb)
{
    auto c = new (std::nothrow) Closure{std::move(cb)};
    if (!c) [[unlikely]] { return Status::Error("out of memory for appending callback"); }
    auto ret = cudaStreamAddCallback(stream_, Trampoline, c, 0);
    if (ret != cudaSuccess) [[unlikely]] {
        delete c;
        return Status{ret, cudaGetErrorString(ret)};
    }
    return Status::OK();
}

Status Trans::CudaStream::Synchronized()
{
    auto ret = cudaStreamSynchronize(stream_);
    if (ret != cudaSuccess) [[unlikely]] { return Status{ret, cudaGetErrorString(ret)}; }
    return Status::OK();
}

} // namespace UC::Trans
