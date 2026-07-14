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
#include <acl/acl.h>
#include <cstring>
#include <limits>
#include <utility>
#include "trans/ascend/ascend_buffer.h"

namespace UC {
namespace {

constexpr std::size_t kSlotAddressAlignment = 64;

}  // namespace

bool BufferPool::ComputeSlotStride(std::size_t capacity, std::size_t& stride)
{
    constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();
    if (capacity == 0 || capacity > kMaxSize - (kSlotAddressAlignment - 1)) { return false; }

    stride = (capacity + kSlotAddressAlignment - 1) / kSlotAddressAlignment *
             kSlotAddressAlignment;
    return true;
}

Status BufferPool::BufferRegion::Create(MemoryType type, std::size_t size,
                                        BufferRegion& region)
{
    Trans::AscendBuffer ascendBuffer;
    switch (type) {
        case MemoryType::HOST: {
            auto owner = ascendBuffer.MakeHostBuffer(size);
            if (!owner) { return Status::Error("failed to allocate host memory"); }
            region.owner = std::move(owner);
            region.local_addr = region.owner.get();
            region.device_addr = region.owner.get();
            return Status::OK();
        }
        case MemoryType::HOST_PINNED: {
            void* deviceAddr = nullptr;
            auto owner = ascendBuffer.MakeHostPinnedBuffer(size, &deviceAddr);
            if (!owner || !deviceAddr) {
                return Status::Error("failed to allocate host-pinned memory");
            }
            region.owner = std::move(owner);
            region.local_addr = region.owner.get();
            region.device_addr = deviceAddr;
            return Status::OK();
        }
        case MemoryType::ASCEND_DEVICE: {
            auto owner = ascendBuffer.MakeDeviceBuffer(size);
            if (!owner) { return Status::Error("failed to allocate device memory"); }
            region.owner = std::move(owner);
            region.local_addr = region.owner.get();
            region.device_addr = region.owner.get();
            return Status::OK();
        }
        default: return Status::InvalidParam("unsupported memory type");
    }
}

void BufferPool::BufferRegion::Reset()
{
    owner.reset();
    local_addr = nullptr;
    device_addr = nullptr;
}

Status BufferPool::Init(std::string name, MemoryType type, std::size_t slot_capacity,
                        std::size_t slot_num)
{
    if (region_) { return Status::InvalidParam(name + " already initialized"); }
    if (slot_capacity == 0 || slot_num == 0) {
        return Status::InvalidParam(name + ": slot_capacity and slot_num must be non-zero");
    }

    std::size_t slotStride = 0;
    if (!ComputeSlotStride(slot_capacity, slotStride) ||
        slot_num > std::numeric_limits<std::size_t>::max() / slotStride ||
        slot_num >= std::numeric_limits<IndexPool::Index>::max()) {
        return Status::InvalidParam(name + ": slot layout size overflow");
    }

    BufferRegion region;
    const auto total = slotStride * slot_num;
    auto status = BufferRegion::Create(type, total, region);
    if (status.Failure()) { return status; }

    name_ = std::move(name);
    slot_capacity_ = slot_capacity;
    slot_stride_ = slotStride;
    slot_num_ = slot_num;
    memory_type_ = type;
    region_ = std::move(region);

    status = ZeroMemory(region_.local_addr, total);
    if (status.Failure()) {
        Reset();
        return status;
    }

    index_pool_.Setup(static_cast<IndexPool::Index>(slot_num_));
    return Status::OK();
}

Status BufferPool::Allocate(std::size_t size, Slot& slot)
{
    if (!region_) { return Status::Error("buffer pool not initialized"); }
    if (size == 0) { return Status::InvalidParam(name_ + ": size must be non-zero"); }
    if (size > slot_capacity_) {
        return Status::InvalidParam(name_ + ": size exceeds slot_capacity");
    }

    const auto idx = index_pool_.Acquire();
    if (idx == IndexPool::npos) {
        return Status(Status::Retry().Underlying(), name_ + ": no free slots");
    }

    const auto offset = static_cast<std::size_t>(idx) * slot_stride_;
    slot.local_addr = static_cast<char*>(region_.local_addr) + offset;
    slot.device_addr = static_cast<char*>(region_.device_addr) + offset;
    slot.length = size;
    slot.slot_index = idx;
    return Status::OK();
}

Status BufferPool::Free(std::uint32_t slot_index)
{
    if (!region_) { return Status::Error("buffer pool not initialized"); }
    if (slot_index >= slot_num_) {
        return Status::InvalidParam(name_ + ": slot_index out of range");
    }

    auto* slot = static_cast<char*>(region_.local_addr) + slot_index * slot_stride_;
    auto status = ZeroMemory(slot, slot_stride_);
    if (status.Failure()) { return status; }

    index_pool_.Release(static_cast<IndexPool::Index>(slot_index));
    return Status::OK();
}

void BufferPool::Reset()
{
    region_.Reset();
    name_.clear();
    slot_capacity_ = 0;
    slot_stride_ = 0;
    slot_num_ = 0;
    memory_type_ = MemoryType::HOST;
}

bool BufferPool::IsValidPointer(const void* ptr) const
{
    if (!ptr || !region_) { return false; }
    const auto base = reinterpret_cast<std::uintptr_t>(region_.local_addr);
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    if (address < base) { return false; }

    const auto offset = address - base;
    return offset < GetTotalSize() && offset % slot_stride_ == 0;
}

Status BufferPool::ZeroMemory(void* ptr, std::size_t size) const
{
    if (memory_type_ == MemoryType::ASCEND_DEVICE) {
        if (aclrtMemset(ptr, size, 0, size) != ACL_SUCCESS) {
            return Status::Error(name_ + ": failed to zero device memory");
        }
    } else {
        std::memset(ptr, 0, size);
    }
    return Status::OK();
}

}  // namespace UC
