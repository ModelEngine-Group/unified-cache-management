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
#include "cuda_device.h"
#include <cuda_runtime.h>
#include <spdlog/fmt/fmt.h>
#include "logger/logger.h"

template <>
struct fmt::formatter<cudaError_t> : formatter<int32_t> {
    auto format(cudaError_t err, format_context& ctx) const -> format_context::iterator
    {
        return formatter<int32_t>::format(err, ctx);
    }
};

namespace UCM {

template <typename Api, typename... Args>
Status CudaApi(const char* caller, const char* file, const size_t line, const char* name, Api&& api, Args&&... args)
{
    auto ret = api(args...);
    if (ret != cudaSuccess) {
        UCM_ERROR("CUDA ERROR: api={}, code={}, err={}, caller={},{}:{}.", name, ret, cudaGetErrorString(ret), caller,
                 basename(file), line);
        return Status::OsApiError();
    }
    return Status::OK();
}
#define CUDA_API(api, ...) CudaApi(__FUNCTION__, __FILE__, __LINE__, #api, api, __VA_ARGS__)

Status CudaDevice::Setup()
{
    auto status = Status::OK();
    if ((status = CUDA_API(cudaSetDevice, this->_deviceId)).Failure()) { return status; }
    if ((status = IBufferedDevice::Setup()).Failure()) { return status; }
    if ((status = CUDA_API(cudaStreamCreate, (cudaStream_t*)&this->_stream)).Failure()) { return status; }
    return status;
}

Status CudaDevice::H2DAsync(std::byte* dst, const std::byte* src, const size_t count)
{
    return CUDA_API(cudaMemcpyAsync, dst, src, count, cudaMemcpyHostToDevice, (cudaStream_t)this->_stream);
}

Status CudaDevice::D2HAsync(std::byte* dst, const std::byte* src, const size_t count)
{
    return CUDA_API(cudaMemcpyAsync, dst, src, count, cudaMemcpyDeviceToHost, (cudaStream_t)this->_stream);
}

struct Closure {
    std::function<void(bool)> cb;
    explicit Closure(std::function<void(bool)> cb) : cb{cb} {}
};

static void Trampoline(cudaStream_t stream, cudaError_t ret, void* data)
{
    (void)stream;
    auto c = (Closure*)data;
    c->cb(ret == cudaSuccess);
    delete c;
}

Status CudaDevice::AppendCallback(std::function<void(bool)> cb)
{
    auto* c = new (std::nothrow) Closure(cb);
    if (!c) {
        UCM_ERROR("Failed to make closure for append cb.");
        return Status::OutOfMemory();
    }
    auto status = CUDA_API(cudaStreamAddCallback, (cudaStream_t)this->_stream, Trampoline, (void*)c, 0);
    if (status.Failure()) { delete c; }
    return status;
}

std::shared_ptr<std::byte> CudaDevice::MakeBuffer(const size_t size)
{
    std::byte* host = nullptr;
    auto ret = cudaMallocHost((void**)&host, size);
    if (ret != cudaSuccess) {
        UCM_ERROR("CUDA ERROR: api=cudaMallocHost, code={}.", ret);
        return nullptr;
    }
    return std::shared_ptr<std::byte>(host, cudaFreeHost);
}

} // namespace UCM
