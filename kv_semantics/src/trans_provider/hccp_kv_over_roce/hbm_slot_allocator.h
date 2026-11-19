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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "index_pool.h"
#include "kv_types.h"

namespace kv {

// Fixed per-connection slot size in HBM. Each connection stores its operator QPC
// in one slot; the HBM region is carved into slots of this size. A slot holds
// exactly the QPC payload, with no padding, so this must match
// HCCP_KV_ROCE_QPC_BYTES in hccp_kv_over_roce_abi.h.
constexpr std::size_t kHccpKvRoceConnSlotBytes = 256;
// Default slot capacity: 384 x 8, the connection budget of one non-shared
// provider.
constexpr std::uint32_t kHccpKvRoceDefaultSlotCapacity = 384 * 8;

// Owns one contiguous NPU/HBM region and hands out fixed-size slots. Each slot
// is tracked with an IndexPool index (the "slotid" recorded by the provider) and
// an explicit FREE/IN_USE state. Write, Clear and Release are only valid for an
// acquired slot: writing or releasing a slot twice is rejected with
// INVALID_ARGUMENT instead of corrupting the free list. The class is
// thread-safe: Acquire/Release are lock-free, and the slot base address never
// changes after Init(). Every ACL operation runs on the caller's bound
// device/context; the provider owns that binding.
class HbmSlotAllocator {
public:
    HbmSlotAllocator(std::int32_t deviceId, std::size_t slotBytes, std::uint32_t capacity);
    ~HbmSlotAllocator();

    HbmSlotAllocator(const HbmSlotAllocator&) = delete;
    HbmSlotAllocator& operator=(const HbmSlotAllocator&) = delete;
    HbmSlotAllocator(HbmSlotAllocator&&) = delete;
    HbmSlotAllocator& operator=(HbmSlotAllocator&&) = delete;

    // Allocates the HBM region on the caller-bound ACL device/context.
    Status Init();

    // Acquire marks a free slot IN_USE. Release only accepts an IN_USE slot and
    // rejects a second release of the same id. Write and Clear require IN_USE.
    Status Acquire(std::uint32_t& slotId);
    Status Release(std::uint32_t slotId);
    Status Write(std::uint32_t slotId, const void* data, std::size_t size);
    Status Clear(std::uint32_t slotId);

    // Device address of one slot, or nullptr when the slot is out of range.
    // The caller must have bound the provider ACL context first. This accessor
    // intentionally ignores the slot state so error paths can still resolve the
    // address of a slot that is being torn down.
    void* SlotAddress(std::uint32_t slotId) const noexcept;

    std::int32_t deviceId() const noexcept { return deviceId_; }
    std::size_t slotBytes() const noexcept { return slotBytes_; }
    std::uint32_t capacity() const noexcept { return capacity_; }

private:
    bool ValidSlot(std::uint32_t slotId) const noexcept;

    std::int32_t deviceId_;
    std::size_t slotBytes_;
    std::uint32_t capacity_;
    void* base_{nullptr};
    bool initialized_{false};
    IndexPool indexPool_;
    // Per-slot FREE/IN_USE flag. Atomic elements are not zero-initialized by the
    // vector constructor, so Init() resets every entry explicitly.
    std::vector<std::atomic<std::uint8_t>> slotStates_;
};

}  // namespace kv
