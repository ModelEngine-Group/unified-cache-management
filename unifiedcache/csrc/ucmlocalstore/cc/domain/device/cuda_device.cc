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
#include "logger/logger.h"

template <>
struct fmt::formatter<cudaError_t> : formatter<int32_t> {
    auto format(cudaError_t err, format_context& ctx) const -> format_context::iterator
    {
        return formatter<int32_t>::format(err, ctx);
    }
};

namespace UCM {

CudaDevice::~CudaDevice()
{
    if (!this->_stream) { return; }
    this->_stream = nullptr;
}

Status CudaDevice::Setup()
{
    auto status = IBufferedDevice::Setup();
    if (status.Failure()){ return status; }
    auto ret = cudaSetDevice(this->_deviceId);
    if (ret != cudaSuccess) {
        UCM_ERROR("Failed({}) to run cudaSetDevice with device({}).", ret, this->_deviceId);
        return Status::OsApiError();
    }
    ret = cudaStreamCreate(&this->_stream);
    if (ret != cudaSuccess) {
        UCM_ERROR("Failed({}) to run cudaStreamCreate.", ret);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status CudaDevice::H2DAsync(void* dst, size_t dstMax, const void* src, const size_t count)
{
    auto ret = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, this->_stream);
    if (ret != cudaSuccess) {
        UCM_ERROR("Failed({}) to copy data from H({}) to D({}).", ret, count, dstMax);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status CudaDevice::D2HAsync(void* dst, size_t dstMax, const void* src, const size_t count)
{
    auto ret = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, this->_stream);
    if (ret != cudaSuccess) {
        UCM_ERROR("Failed({}) to copy data from D({}) to H({}).", ret, count, dstMax);
        return Status::OsApiError();
    }
    return Status::OK();
}

Status CudaDevice::WaitFinish()
{
    auto ret = cudaStreamSynchronize(this->_stream);
    if (ret != cudaSuccess) {
        UCM_ERROR("Failed({}) to synchronize device stream.", ret);
        return Status::OsApiError();
    }
    this->ResetHostBufferIndex();
    return Status::OK();
}

} // namespace UCM
