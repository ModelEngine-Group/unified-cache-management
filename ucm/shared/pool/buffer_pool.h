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
 * */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include "pool/buffer_region.h"
#include "status/status.h"
#include "thread/index_pool.h"

namespace UC {

class BufferPool {
    static constexpr std::size_t kDefaultSlotAlignment = 64;

public:
    using MemoryType = BufferMemoryType;

    struct Slot {
        void* local_addr{nullptr};
        void* device_addr{nullptr};
        std::size_t length{0};
        std::uint32_t slot_index{UINT32_MAX};
        std::size_t offset{0};  // Byte offset from both pool base addresses.
    };

    BufferPool() = default;
    ~BufferPool() = default;

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    // slot_alignment applies to the slot stride and offsets from the pool base, not to base
    // addresses.
    Status Init(std::string name, MemoryType type, std::size_t slot_capacity, std::size_t slot_num,
                bool enable_zero = false, std::size_t slot_alignment = kDefaultSlotAlignment);
    Status Allocate(Slot& slot);
    Status Free(std::uint32_t slot_index);
    void Reset();

    bool IsInitialized() const { return static_cast<bool>(region_); }
    bool IsValidPointer(const void* ptr) const;

    const std::string& GetName() const { return name_; }
    void* GetLocalAddr() const { return region_.local_addr; }
    void* GetDeviceAddr() const { return region_.device_addr; }
    std::size_t GetTotalSize() const { return slot_stride_ * slot_num_; }
    std::size_t GetSlotCount() const { return slot_num_; }
    MemoryType GetMemoryType() const { return memory_type_; }

private:
    static bool ComputeSlotStride(std::size_t capacity, std::size_t alignment, std::size_t& stride);
    Status ZeroMemory(void* ptr, std::size_t size) const;

    std::string name_;
    std::size_t slot_capacity_{0};
    std::size_t slot_stride_{0};
    std::size_t slot_num_{0};
    MemoryType memory_type_{MemoryType::HOST};
    bool enable_zero_{false};

    PoolDetail::BufferRegion region_;
    IndexPool index_pool_;
};

}  // namespace UC
