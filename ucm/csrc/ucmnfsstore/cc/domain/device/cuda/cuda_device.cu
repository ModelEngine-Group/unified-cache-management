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
#include <cuda_runtime.h>
#include "idevice.h"
#include "logger/logger.h"

namespace UC {

#define CUDA_TRANS_UNIT_SIZE (sizeof(uint64_t) * 2)
#define CUDA_TRANS_BLOCK_NUMBER (32)
#define CUDA_TRANS_BLOCK_SIZE (256)
#define CUDA_TRANS_THREAD_NUMBER (CUDA_TRANS_BLOCK_NUMBER * CUDA_TRANS_BLOCK_SIZE)

inline __device__ void H2DUnit(uint8_t* __restrict__ dst, const volatile uint8_t* __restrict__ src)
{
    uint64_t a, b;
    asm volatile("ld.global.cs.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(src));
    asm volatile("st.global.cg.v2.u64 [%0], {%1, %2};" ::"l"(dst), "l"(a), "l"(b));
}

inline __device__ void D2HUnit(volatile uint8_t* __restrict__ dst, const uint8_t* __restrict__ src)
{
    uint64_t a, b;
    asm volatile("ld.global.cs.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(src));
    asm volatile("st.volatile.global.v2.u64 [%0], {%1, %2};" ::"l"(dst), "l"(a), "l"(b));
}

__global__ void H2DKernel(const volatile uintptr_t* src, uintptr_t* dst, size_t number, size_t size)
{
    auto length = number * size;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        H2DUnit(((uint8_t*)dst[idx]) + off, ((const uint8_t*)src[idx]) + off);
        offset += CUDA_TRANS_THREAD_NUMBER * CUDA_TRANS_UNIT_SIZE;
    }
}

__global__ void D2HKernel(const uintptr_t* src, volatile uintptr_t* dst, size_t number, size_t size)
{
    auto length = number * size;
    auto offset = (blockIdx.x * blockDim.x + threadIdx.x) * CUDA_TRANS_UNIT_SIZE;
    while (offset + CUDA_TRANS_UNIT_SIZE <= length) {
        auto idx = offset / size;
        auto off = offset % size;
        D2HUnit(((uint8_t*)dst[idx]) + off, ((const uint8_t*)src[idx]) + off);
        offset += CUDA_TRANS_THREAD_NUMBER * CUDA_TRANS_UNIT_SIZE;
    }
}

inline __host__ void H2D(const volatile uintptr_t* src, uintptr_t* dst, size_t number, size_t size, cudaStream_t stream)
{
    H2DKernel<<<CUDA_TRANS_BLOCK_NUMBER, CUDA_TRANS_BLOCK_SIZE, 0, stream>>>(src, dst, number, size);
}

inline __host__ void D2H(const uintptr_t* src, volatile uintptr_t* dst, size_t number, size_t size, cudaStream_t stream)
{
    D2HKernel<<<CUDA_TRANS_BLOCK_NUMBER, CUDA_TRANS_BLOCK_SIZE, 0, stream>>>(src, dst, number, size);
}

class CudaDevice : public IDevice {
    void* _addr;
    cudaStream_t _stream;
    void* AllocHost(const size_t size)
    {
        void* ptr = nullptr;
        auto ret = cudaMallocHost(&ptr, size);
        if (ret != cudaSuccess) {
            UC_ERROR("Failed({}) to alloc host memory({}): {}.", fmt::underlying(ret), size, cudaGetErrorString(ret));
            return nullptr;
        }
        return ptr;
    }
    void FreeHost(void* ptr) { (void)cudaFreeHost(ptr); }
    void* AllocDevice(const size_t size)
    {
        void* ptr = nullptr;
        auto ret = cudaMalloc(&ptr, size);
        if (ret != cudaSuccess) {
            UC_ERROR("Failed({}) to alloc device memory({}): {}.", fmt::underlying(ret), size, cudaGetErrorString(ret));
            return nullptr;
        }
        return ptr;
    }
    void FreeDevice(void* ptr) { (void)cudaFree(ptr); }
    void* HostArray2DeviceArray(const uintptr_t* hostArray, const size_t number)
    {
        auto size = sizeof(uintptr_t) * number;
        auto deviceArray = this->AllocDevice(size);
        if (!deviceArray) { return nullptr; }
        auto ret = cudaMemcpy(deviceArray, hostArray, size, cudaMemcpyHostToDevice);
        if (ret != cudaSuccess) {
            this->FreeDevice(deviceArray);
            UC_ERROR("Failed({}) to memcpy({}): {}.", fmt::underlying(ret), size, cudaGetErrorString(ret));
            return nullptr;
        }
        return deviceArray;
    }

public:
    CudaDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : IDevice{deviceId, bufferSize, bufferNumber}, _addr{nullptr}, _stream{nullptr}
    {
    }
    ~CudaDevice() override
    {
        if (this->_addr) {
            this->FreeHost(this->_addr);
            this->_addr = nullptr;
        }
    }
    Status Setup() override
    {
        if (this->deviceId < 0) {
            UC_ERROR("Invalid xpu id({}).", this->deviceId);
            return Status::InvalidParam();
        }
        auto ret = cudaSetDevice(this->deviceId);
        if (ret != cudaSuccess) {
            UC_ERROR("Failed({}) to set device({}): {}.", fmt::underlying(ret), this->deviceId,
                     cudaGetErrorString(ret));
            return Status::Error();
        }
        ret = cudaStreamCreate(&this->_stream);
        if (ret != cudaSuccess) {
            UC_ERROR("Failed({}) to create stream: {}.", fmt::underlying(ret), cudaGetErrorString(ret));
            return Status::Error();
        }
        auto reservedMemSize = this->bufferNumber * this->bufferSize;
        if (reservedMemSize != 0) {
            this->_addr = this->AllocHost(reservedMemSize);
            if (!this->_addr) { return Status::OutOfMemory(); }
        }
        return Status::OK();
    }
    void* GetBuffer(const size_t idx) override
    {
        if (idx < this->bufferNumber) { return ((uint8_t*)this->_addr) + this->bufferSize * idx; }
        return this->AllocHost(this->bufferSize);
    }
    void PutBuffer(const size_t idx, void* ptr) override
    {
        if (idx < this->bufferNumber) { return; }
        this->FreeHost(ptr);
    }
    Status H2DBatch(const uintptr_t* from, uintptr_t* to, const size_t number, const size_t size) override
    {
        auto src = this->HostArray2DeviceArray(from, number);
        if (!src) { return Status::OutOfMemory(); }
        auto dst = this->HostArray2DeviceArray(to, number);
        if (!dst) {
            this->FreeDevice(src);
            return Status::OutOfMemory();
        }
        H2D((const volatile uintptr_t*)src, (uintptr_t*)dst, number, size, this->_stream);
        auto ret = cudaStreamSynchronize(this->_stream);
        this->FreeDevice(src);
        this->FreeDevice(dst);
        if (ret != cudaSuccess) {
            UC_ERROR("Stream error({}): {}.", fmt::underlying(ret), cudaGetErrorString(ret));
            return Status::Error();
        }
        return Status::OK();
    }
    Status D2HBatch(const uintptr_t* from, uintptr_t* to, const size_t number, const size_t size) override
    {
        auto src = this->HostArray2DeviceArray(from, number);
        if (!src) { return Status::OutOfMemory(); }
        auto dst = this->HostArray2DeviceArray(to, number);
        if (!dst) {
            this->FreeDevice(src);
            return Status::OutOfMemory();
        }
        D2H((const uintptr_t*)src, (volatile uintptr_t*)dst, number, size, this->_stream);
        auto ret = cudaStreamSynchronize(this->_stream);
        this->FreeDevice(src);
        this->FreeDevice(dst);
        if (ret != cudaSuccess) {
            UC_ERROR("Stream error({}): {}.", fmt::underlying(ret), cudaGetErrorString(ret));
            return Status::Error();
        }
        return Status::OK();
    }
};

std::unique_ptr<IDevice> DeviceFactory::Make(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
{
    try {
        return std::make_unique<CudaDevice>(deviceId, bufferSize, bufferNumber);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to make cuda device({},{},{}).", e.what(), deviceId, bufferSize, bufferNumber);
        return nullptr;
    }
}

} // namespace UC
