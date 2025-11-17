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
#include "helper.h"
#include "stream.h"

namespace UC {

Stream::~Stream()
{
    auto ret = cudaStreamDestroy((cudaStream_t)this->stream_);
    if (ret != cudaSuccess) { CUDA_ERROR(ret); }
}

Status Stream::Setup()
{
    auto ret = cudaStreamCreate((cudaStream_t*)&this->stream_);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    return Status::OK();
}

Status Stream::H2DBatchSync(uintptr_t hostAddr, uintptr_t deviceAddrs[], const size_t size,
                            const size_t number)
{
    auto ret = cudaSuccess;
    for (size_t i = 0; i < number; i++) {
        void* src = (void*)(((std::byte*)hostAddr) + size * i);
        void* dst = (void*)deviceAddrs[i];
        ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)this->stream_);
        if (ret != cudaSuccess) {
            CUDA_ERROR(ret);
            return Status::Error();
        }
    }
    ret = cudaStreamSynchronize((cudaStream_t)this->stream_);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    return Status::OK();
}

Status Stream::D2HBatchSync(uintptr_t deviceAddrs[], uintptr_t hostAddr, const size_t size,
                            const size_t number)
{
    auto s = this->D2HBatchAsync(deviceAddrs, hostAddr, size, number);
    if (s.Failure()) { return s; }
    return this->Synchronize();
}

Status Stream::D2HBatchAsync(uintptr_t deviceAddrs[], uintptr_t hostAddr, const size_t size,
                             const size_t number)
{
    auto ret = cudaSuccess;
    for (size_t i = 0; i < number; i++) {
        void* src = (void*)deviceAddrs[i];
        void* dst = (void*)(((std::byte*)hostAddr) + size * i);
        ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)this->stream_);
        if (ret != cudaSuccess) {
            CUDA_ERROR(ret);
            return Status::Error();
        }
    }
    return Status::OK();
}

Status Stream::Synchronize()
{
    auto ret = cudaStreamSynchronize((cudaStream_t)this->stream_);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    return Status::OK();
}

} // namespace UC
