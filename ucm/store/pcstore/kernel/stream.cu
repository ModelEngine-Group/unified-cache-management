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

#define CUDA_TRANS_UNIT_SIZE (sizeof(uint4) * 2)
#define CUDA_TRANS_BLOCK_NUMBER (32)
#define CUDA_TRANS_BLOCK_SIZE (256)
#define CUDA_TRANS_THREAD_NUMBER (CUDA_TRANS_BLOCK_NUMBER * CUDA_TRANS_BLOCK_SIZE)

inline __device__ void MemcpyUnit(const uint8_t* __restrict__ src,
                                  volatile uint8_t* __restrict__ dst)
{
    uint4 lo, hi;
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w)
                 : "l"(src));
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4+16];"
                 : "=r"(hi.x), "=r"(hi.y), "=r"(hi.z), "=r"(hi.w)
                 : "l"(src));
    asm volatile("st.volatile.global.v4.b32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(dst), "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w));
    asm volatile("st.volatile.global.v4.b32 [%0+16], {%1,%2,%3,%4};"
                 :
                 : "l"(dst), "r"(hi.x), "r"(hi.y), "r"(hi.z), "r"(hi.w));
}

__global__ void H2DKernel(const volatile void* src, void** dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        MemcpyUnit(((const uint8_t*)src) + offset, ((uint8_t*)dst[idx]) + off);
        offset += CUDA_TRANS_THREAD_NUMBER * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void D2HKernel(const void** src, volatile void* dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        MemcpyUnit(((const uint8_t*)src[idx]) + off, ((uint8_t*)dst) + offset);
        offset += CUDA_TRANS_THREAD_NUMBER * CUDA_TRANS_UNIT_SIZE;
    }
}

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
    void* hostAddrOnDevice = nullptr;
    auto ret = cudaHostGetDevicePointer(&hostAddrOnDevice, (void*)hostAddr, 0);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    auto devicePtrs = this->MakeDevicePtrs(deviceAddrs, number);
    if (!devicePtrs) { return Status::Error(); }
    H2DKernel<<<CUDA_TRANS_BLOCK_NUMBER, CUDA_TRANS_BLOCK_SIZE, 0, (cudaStream_t)this->stream_>>>(
        hostAddrOnDevice, (void**)devicePtrs, size, number);
    ret = cudaStreamSynchronize((cudaStream_t)this->stream_);
    cudaFree(devicePtrs);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    return Status::OK();
}

Status Stream::D2HBatchSync(uintptr_t deviceAddrs[], uintptr_t hostAddr, const size_t size,
                            const size_t number)
{
    void* hostAddrOnDevice = nullptr;
    auto ret = cudaHostGetDevicePointer(&hostAddrOnDevice, (void*)hostAddr, 0);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    auto devicePtrs = this->MakeDevicePtrs(deviceAddrs, number);
    if (!devicePtrs) { return Status::Error(); }
    D2HKernel<<<CUDA_TRANS_BLOCK_NUMBER, CUDA_TRANS_BLOCK_SIZE, 0, (cudaStream_t)this->stream_>>>(
        (const void**)devicePtrs, hostAddrOnDevice, size, number);
    ret = cudaStreamSynchronize((cudaStream_t)this->stream_);
    cudaFree(devicePtrs);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    return Status::OK();
}

void* Stream::MakeDevicePtrs(uintptr_t deviceAddrs[], const size_t number)
{
    auto size = sizeof(uintptr_t) * number;
    void* deviceArray = nullptr;
    auto ret = cudaMalloc(&deviceArray, size);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return nullptr;
    }
    ret = cudaMemcpy(deviceArray, deviceAddrs, size, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        cudaFree(deviceArray);
        CUDA_ERROR(ret);
        return nullptr;
    }
    return deviceArray;
}

} // namespace UC
