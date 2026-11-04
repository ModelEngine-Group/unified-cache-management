/**
 * MIT License
 *
 * Copyright (c) 2026 relat-ivity
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
#ifndef COPY_BUFFER_GDR_H
#define COPY_BUFFER_GDR_H

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <cstring>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "copy_buffer.h"
#include "error_handle_gdr.h"
#include "rdma_channel_gdr.h"

class GdrCopyBuffer : public CopyBuffer {
protected:
    std::unordered_map<size_t, ibv_mr*> memoryRegions_;

    void RegisterHostMemoryForAllChannels()
    {
        auto& manager = ChannelManager::Instance();
        ASSERT(manager.IsInitialized());
        for (auto targetDevice : manager.DeviceIds()) {
            const auto device = static_cast<size_t>(targetDevice);
            memoryRegions_.emplace(
                device, manager.Get(targetDevice).RegisterHostMemory(addr_, TotalBytes()));
        }
    }

    void RegisterDeviceMemoryForOwnerChannel()
    {
        auto& manager = ChannelManager::Instance();
        ASSERT(manager.IsInitialized());
        const auto device = static_cast<int32_t>(device_);
        memoryRegions_.emplace(device_, manager.Get(device).RegisterGpuMemory(addr_, TotalBytes()));
    }

    void ReleaseMemoryRegions() noexcept
    {
        for (auto& entry : memoryRegions_) {
            if (entry.second != nullptr) { ibv_dereg_mr(entry.second); }
        }
        memoryRegions_.clear();
    }

    ibv_mr* FindMemoryRegion(size_t targetDevice) const
    {
        const auto it = memoryRegions_.find(targetDevice);
        return it == memoryRegions_.end() ? nullptr : it->second;
    }

public:
    GdrCopyBuffer(size_t device, size_t size, size_t number) : CopyBuffer(device, size, number) {}
    ~GdrCopyBuffer() override = default;

    size_t TotalBytes() const { return size_ * number_; }
    uint64_t Address(size_t index) const { return reinterpret_cast<uint64_t>((*this)[index]); }
    bool HasMR(size_t targetDevice) const { return FindMemoryRegion(targetDevice) != nullptr; }
    uint32_t LKey(size_t targetDevice) const
    {
        auto* memoryRegion = FindMemoryRegion(targetDevice);
        ASSERT(memoryRegion != nullptr);
        return memoryRegion->lkey;
    }
    uint32_t RKey(size_t targetDevice) const
    {
        auto* memoryRegion = FindMemoryRegion(targetDevice);
        ASSERT(memoryRegion != nullptr);
        return memoryRegion->rkey;
    }
};

class GdrHostCopyBuffer : public GdrCopyBuffer {
public:
    GdrHostCopyBuffer(size_t device, size_t size, size_t number)
        : GdrCopyBuffer(device, size, number)
    {
        const auto total = TotalBytes();
        CUDA_ASSERT(cudaSetDevice(device_));
        CUDA_ASSERT(cudaMallocHost(&addr_, total));
        std::memset(addr_, 'h', total);
        RegisterHostMemoryForAllChannels();
    }

    ~GdrHostCopyBuffer() override
    {
        ReleaseMemoryRegions();
        if (addr_ != nullptr) {
            CUDA_ASSERT(cudaSetDevice(device_));
            CUDA_ASSERT(cudaFreeHost(addr_));
        }
    }

    std::string Name() const override { return "gdr::host::" + std::to_string(device_); }
};

class GdrDeviceCopyBuffer : public GdrCopyBuffer {
public:
    GdrDeviceCopyBuffer(size_t device, size_t size, size_t number)
        : GdrCopyBuffer(device, size, number)
    {
        const auto total = TotalBytes();
        CUDA_ASSERT(cudaSetDevice(device_));
        CUDA_ASSERT(cudaMalloc(&addr_, total));
        CUDA_ASSERT(cudaMemset(addr_, 'd', total));
        RegisterDeviceMemoryForOwnerChannel();
    }

    ~GdrDeviceCopyBuffer() override
    {
        ReleaseMemoryRegions();
        if (addr_ != nullptr) {
            CUDA_ASSERT(cudaSetDevice(device_));
            CUDA_ASSERT(cudaFree(addr_));
        }
    }

    std::string Name() const override { return "gdr::device::" + std::to_string(device_); }
};

#endif  // COPY_BUFFER_GDR_H
