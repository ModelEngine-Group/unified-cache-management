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
#include "pool/buffer_pool.h"
#include <limits>
#include <utility>
#include "trans/detail/reserved_buffer.h"
#include "trans/device.h"

namespace UC {

bool BufferPool::ComputeSlotStride(std::size_t capacity, std::size_t alignment, std::size_t& stride)
{
    constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();
    if (capacity == 0 || alignment == 0 || capacity > kMaxSize - (alignment - 1)) { return false; }

    stride = (capacity + alignment - 1) / alignment * alignment;
    return true;
}

Status BufferPool::Init(std::string name, MemoryType type, std::size_t slotCapacity,
                        std::size_t slotNum, bool enableZero, std::size_t slotAlignment)
{
    if (region_) { return Status::InvalidParam(name + " already initialized"); }
    if (slotCapacity == 0 || slotNum == 0 || slotAlignment == 0) {
        return Status::InvalidParam(
            name + ": slot_capacity, slot_num and slot_alignment must be non-zero");
    }

    std::size_t slotStride = 0;
    if (!ComputeSlotStride(slotCapacity, slotAlignment, slotStride) ||
        slotNum > std::numeric_limits<std::size_t>::max() / slotStride ||
        slotNum >= std::numeric_limits<IndexPool::Index>::max()) {
        return Status::InvalidParam(name + ": slot layout size overflow");
    }

    BufferRegion region;
    const auto total = slotStride * slotNum;
    auto status = BufferRegion::Create(type, total, region);
    if (status.Failure()) { return status; }

    name_ = std::move(name);
    slotCapacity_ = slotCapacity;
    slotStride_ = slotStride;
    slotNum_ = slotNum;
    memoryType_ = type;
    enableZero_ = enableZero;
    region_ = std::move(region);

    if (enableZero_) {
        status = ZeroMemory(region_.localAddr, total);
        if (status.Failure()) {
            Reset();
            return status;
        }
    }

    indexPool_.Setup(static_cast<IndexPool::Index>(slotNum_));
    return Status::OK();
}

Status BufferPool::Allocate(Slot& slot)
{
    if (!region_) { return Status::Error("buffer pool not initialized"); }

    const auto index = indexPool_.Acquire();
    if (index == IndexPool::npos) {
        return Status(Status::NoSpace().Underlying(), name_ + ": no free slots");
    }

    const auto offset = static_cast<std::size_t>(index) * slotStride_;
    slot.localAddr = static_cast<char*>(region_.localAddr) + offset;
    slot.deviceAddr = static_cast<char*>(region_.deviceAddr) + offset;
    slot.length = slotCapacity_;
    slot.slotIndex = index;
    slot.offset = offset;
    return Status::OK();
}

Status BufferPool::Free(std::uint32_t slotIndex)
{
    if (!region_) { return Status::Error("buffer pool not initialized"); }
    if (slotIndex >= slotNum_) { return Status::InvalidParam(name_ + ": slot_index out of range"); }

    if (enableZero_) {
        auto* slot = static_cast<char*>(region_.localAddr) + slotIndex * slotStride_;
        auto status = ZeroMemory(slot, slotStride_);
        if (status.Failure()) { return status; }
    }

    indexPool_.Release(static_cast<IndexPool::Index>(slotIndex));
    return Status::OK();
}

void BufferPool::Reset()
{
    region_.Reset();
    name_.clear();
    slotCapacity_ = 0;
    slotStride_ = 0;
    slotNum_ = 0;
    memoryType_ = MemoryType::Host;
    enableZero_ = false;
}

bool BufferPool::IsValidPointer(const void* ptr) const
{
    if (!ptr || !region_) { return false; }
    const auto base = reinterpret_cast<std::uintptr_t>(region_.localAddr);
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    if (address < base) { return false; }

    const auto offset = address - base;
    return offset < GetTotalSize() && offset % slotStride_ == 0;
}

Status BufferPool::ZeroMemory(void* ptr, std::size_t size) const
{
    const auto status = UC::Trans::Memset(ptr, size, 0);
    if (status.Failure()) { return Status::Error(name_ + ": failed to memset buffer"); }
    return Status::OK();
}

}  // namespace UC
