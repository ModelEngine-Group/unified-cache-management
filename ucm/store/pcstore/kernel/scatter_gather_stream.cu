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
#include "scatter_gather_stream.h"

#define CUDA_TRANS_UNIT_SIZE (sizeof(uint4) * 2)
#define CUDA_TRANS_BLOCK_NUMBER (32)
#define CUDA_TRANS_BLOCK_SIZE (256)
#define CUDA_TRANS_THREAD_NUMBER (CUDA_TRANS_BLOCK_NUMBER * CUDA_TRANS_BLOCK_SIZE)

inline __device__ void CudaCopyUnit(const uint8_t* __restrict__ src,
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

__global__ void CudaCopyH2DKernel(const void* src, void** dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        auto host = ((const uint8_t*)src) + offset;
        auto device = ((uint8_t*)dst[idx]) + off;
        CudaCopyUnit(host, device);
        offset += CUDA_TRANS_THREAD_NUMBER * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void CudaCopyD2HKernel(const void* const* src, void* dst, size_t size, size_t num)
{
    auto length = size * num;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        auto device = static_cast<const uint8_t*>(src[idx]) + off;
        auto host = static_cast<uint8_t*>(dst) + offset;
        CudaCopyUnit(device, host);
        offset += CUDA_TRANS_THREAD_NUMBER * CUDA_TRANS_UNIT_SIZE;
    }
}

namespace UC {

static void ReleasePtrArrOnDevice(void** deviceArray)
{
    auto ret = cudaFree((void*)deviceArray);
    if (ret != cudaSuccess) { CUDA_ERROR(ret); }
}

static void** MakePtrArrOnDevice(const void* const* ptrArrOnHost, const size_t number)
{
    auto size = sizeof(void*) * number;
    void** ptrArrOnDevice = nullptr;
    auto ret = cudaMalloc(&ptrArrOnDevice, size);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return nullptr;
    }
    ret = cudaMemcpy(ptrArrOnDevice, ptrArrOnHost, size, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        ReleasePtrArrOnDevice(ptrArrOnDevice);
        return nullptr;
    }
    return ptrArrOnDevice;
}

Status ScatterGatherStream::D2HBatchAsync(uintptr_t deviceAddrs[], uintptr_t hostAddr,
                                          const size_t size, const size_t number)
{
    auto ret = cudaSuccess;
    auto stream = (cudaStream_t)this->stream_;
    void* hostAddrOnDevice = nullptr;
    ret = cudaHostGetDevicePointer(&hostAddrOnDevice, (void*)hostAddr, 0);
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    auto devicePtrs = MakePtrArrOnDevice((const void**)deviceAddrs, number);
    if (!devicePtrs) { return Status::OutOfMemory(); }
    CudaCopyD2HKernel<<<CUDA_TRANS_BLOCK_NUMBER, CUDA_TRANS_BLOCK_SIZE, 0, stream>>>(
        (const void**)devicePtrs, hostAddrOnDevice, size, number);
    ret = cudaGetLastError();
    if (ret != cudaSuccess) {
        CUDA_ERROR(ret);
        return Status::Error();
    }
    // ret = cudaStreamSynchronize(stream);
    // ReleasePtrArrOnDevice(devicePtrs);
    // if (ret != cudaSuccess) {
    //     CUDA_ERROR(ret);
    //     return Status::Error();
    // }
    return Status::OK();
}

} // namespace UC
