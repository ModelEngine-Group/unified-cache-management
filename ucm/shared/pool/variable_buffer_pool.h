/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "pool/buffer_region.h"
#include "pool/detail/offset_allocator.h"
#include "status/status.h"

namespace UC {

/**
 * @brief Variable-size allocator backed by one contiguous buffer region.
 *
 * Each pool owns one region and one offset allocator. Callers must pair each
 * successful Allocate with exactly one successful Free and must not call Free
 * concurrently for the same handle. If memory clearing fails, the allocation
 * remains live and the caller must retry Free or Reset the pool. Init and Reset
 * must not run concurrently with allocation methods.
 */
class VariableBufferPool {
    static constexpr std::size_t kDefaultAllocationAlignment = 64;

public:
    using MemoryType = BufferMemoryType;

    class BufferHandle {
    public:
        BufferHandle() = default;
        BufferHandle(const BufferHandle&) = default;
        BufferHandle& operator=(const BufferHandle&) = default;
        BufferHandle(BufferHandle&&) = default;
        BufferHandle& operator=(BufferHandle&&) = default;

        std::size_t GetRequestedSize() const { return requestedSize_; }
        std::size_t GetAllocatedSize() const { return allocatedSize_; }
        std::size_t GetOffset() const { return offset_; }
        void* GetLocalAddr() const { return localAddr_; }
        void* GetDeviceAddr() const { return deviceAddr_; }

    private:
        friend class VariableBufferPool;

        const VariableBufferPool* owner_{nullptr};
        OffsetAllocator::Allocation allocation_;
        // Offset and node index required by OffsetAllocator to release this block.

        std::size_t requestedSize_{0};
        std::size_t allocatedSize_{0};
        // Number of bytes reserved after alignment.

        std::size_t offset_{0};
        // Byte offset from both pool base addresses.

        void* localAddr_{nullptr};
        void* deviceAddr_{nullptr};
    };

    VariableBufferPool() = default;
    ~VariableBufferPool() = default;

    VariableBufferPool(const VariableBufferPool&) = delete;
    VariableBufferPool& operator=(const VariableBufferPool&) = delete;

    // allocationAlignment controls allocation sizes and offsets, not the region base address.
    Status Init(std::string name, MemoryType memoryType, std::size_t totalCapacity,
                std::uint32_t metadataNodeCapacity, bool enableZero = false,
                std::size_t allocationAlignment = kDefaultAllocationAlignment);

    Status Allocate(std::size_t requestedSize, BufferHandle& handle);
    Status Free(const BufferHandle& handle);
    void Reset();

    bool IsInitialized() const { return static_cast<bool>(region_) && allocator_ != nullptr; }

    const std::string& GetName() const { return name_; }
    MemoryType GetMemoryType() const { return memoryType_; }
    std::size_t GetTotalSize() const { return totalCapacity_; }
    void* GetLocalAddr() const { return region_.localAddr; }
    void* GetDeviceAddr() const { return region_.deviceAddr; }

private:
    static bool ComputeAllocationLayout(std::size_t requestedSize, std::size_t allocationAlignment,
                                        std::size_t& allocatedSize, std::uint32_t& requiredUnits);

    Status ZeroMemory(void* address, std::size_t size) const;

    std::string name_;
    MemoryType memoryType_{MemoryType::Host};
    std::size_t totalCapacity_{0};
    std::size_t allocationAlignment_{kDefaultAllocationAlignment};
    bool enableZero_{false};
    // Whether released memory is cleared before it becomes reusable.

    BufferRegion region_;
    // One contiguous backing region owned by this pool.

    std::unique_ptr<OffsetAllocator::Allocator> allocator_;
    // Manages variable-size offsets after Init provides the region size.
};

}  // namespace UC
