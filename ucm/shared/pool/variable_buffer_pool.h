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
#include <mutex>
#include <string>
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
    enum class MemoryType {
        HOST = 0,
        HOST_PINNED = 1,
        ASCEND_DEVICE = 2,
    };

    struct AllocationHandle {
        OffsetAllocator::Allocation allocation;
        // Metadata required by OffsetAllocator to release this block.

        std::size_t requested_size{0};
        std::size_t allocated_size{0};
        // Number of bytes reserved after alignment.

        std::size_t offset{0};
        // Byte offset from both pool base addresses.

        void* local_addr{nullptr};
        void* device_addr{nullptr};
    };

    VariableBufferPool() = default;
    ~VariableBufferPool() = default;

    VariableBufferPool(const VariableBufferPool&) = delete;
    VariableBufferPool& operator=(const VariableBufferPool&) = delete;

    // allocation_alignment controls allocation sizes and offsets, not the region base address.
    Status Init(std::string name, MemoryType memory_type, std::size_t total_capacity,
                std::uint32_t metadata_node_capacity, bool enable_zero = false,
                std::size_t allocation_alignment = kDefaultAllocationAlignment);

    Status Allocate(std::size_t requested_size, AllocationHandle& handle);
    Status Free(const AllocationHandle& handle);
    void Reset();

    bool IsInitialized() const { return static_cast<bool>(region_) && allocator_ != nullptr; }

    const std::string& GetName() const { return name_; }
    MemoryType GetMemoryType() const { return memory_type_; }
    std::size_t GetTotalSize() const { return total_capacity_; }
    void* GetLocalAddr() const { return region_.local_addr; }
    void* GetDeviceAddr() const { return region_.device_addr; }

private:
    struct BufferRegion {
        static Status Create(MemoryType memory_type, std::size_t size, BufferRegion& region);

        explicit operator bool() const { return owner != nullptr; }
        void Reset();

        std::shared_ptr<void> owner;
        void* local_addr{nullptr};
        void* device_addr{nullptr};
    };

    static bool ComputeAllocationLayout(std::size_t requested_size,
                                        std::size_t allocation_alignment,
                                        std::size_t& allocated_size, std::uint32_t& required_units);

    Status ZeroMemory(void* address, std::size_t size) const;

    std::string name_;
    MemoryType memory_type_{MemoryType::HOST};
    std::size_t total_capacity_{0};
    std::size_t allocation_alignment_{kDefaultAllocationAlignment};
    bool enable_zero_{false};
    // Whether released memory is cleared before it becomes reusable.

    BufferRegion region_;
    // One contiguous backing region owned by this pool.

    std::unique_ptr<OffsetAllocator::Allocator> allocator_;
    // Manages variable-size offsets after Init provides the region size.

    std::mutex mutex_;
    // Serializes OffsetAllocator metadata updates within this pool.
};

}  // namespace UC
