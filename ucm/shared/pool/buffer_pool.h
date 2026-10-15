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
        void* localAddr{nullptr};
        void* deviceAddr{nullptr};
        std::size_t length{0};
        std::uint32_t slotIndex{UINT32_MAX};
        std::size_t offset{0};  // Byte offset from both pool base addresses.
    };

    BufferPool() = default;
    ~BufferPool() = default;

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    // slotAlignment applies to the slot stride and offsets from the pool base, not to base
    // addresses.
    Status Init(std::string name, MemoryType type, std::size_t slotCapacity, std::size_t slotNum,
                bool enableZero = false, std::size_t slotAlignment = kDefaultSlotAlignment);
    Status Allocate(Slot& slot);
    Status Free(std::uint32_t slotIndex);
    void Reset();

    bool IsInitialized() const { return static_cast<bool>(region_); }
    bool IsValidPointer(const void* ptr) const;

    const std::string& GetName() const { return name_; }
    void* GetLocalAddr() const { return region_.localAddr; }
    void* GetDeviceAddr() const { return region_.deviceAddr; }
    std::size_t GetTotalSize() const { return slotStride_ * slotNum_; }
    std::size_t GetSlotCount() const { return slotNum_; }
    MemoryType GetMemoryType() const { return memoryType_; }

private:
    static bool ComputeSlotStride(std::size_t capacity, std::size_t alignment, std::size_t& stride);
    Status ZeroMemory(void* ptr, std::size_t size) const;

    std::string name_;
    std::size_t slotCapacity_{0};
    std::size_t slotStride_{0};
    std::size_t slotNum_{0};
    MemoryType memoryType_{MemoryType::Host};
    bool enableZero_{false};

    BufferRegion region_;
    IndexPool indexPool_;
};

}  // namespace UC
