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

bool AlignAddress(void* address, std::size_t alignment, void*& aligned, std::size_t& offset)
{
    if (address == nullptr || alignment == 0) { return false; }

    const auto value = reinterpret_cast<std::uintptr_t>(address);
    const auto remainder = value % alignment;
    offset = remainder == 0 ? 0 : alignment - remainder;
    if (value > std::numeric_limits<std::uintptr_t>::max() - offset) { return false; }

    aligned = reinterpret_cast<void*>(value + offset);
    return true;
}

}  // namespace

bool BufferPool::ComputeSlotStride(std::size_t capacity, std::size_t alignment, std::size_t& stride)
{
    constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();
    if (capacity == 0 || alignment == 0 || capacity > kMaxSize - (alignment - 1)) { return false; }

    stride = (capacity + alignment - 1) / alignment * alignment;
    return true;
}

Status BufferPool::BufferRegion::Create(MemoryType type, std::size_t size, std::size_t alignment,
                                        BufferRegion& region)
{
    constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();
    if (alignment == 0 || size > kMaxSize - (alignment - 1)) {
        return Status::InvalidParam("buffer region size overflow");
    }

    const auto allocationSize = size + alignment - 1;
    auto setRegion = [&region, alignment](std::shared_ptr<void> owner, void* localAddr,
                                          void* deviceAddr) -> Status {
        // Align up local addr
        void* alignedLocalAddr = nullptr;
        std::size_t offset = 0;
        if (!owner || !AlignAddress(localAddr, alignment, alignedLocalAddr, offset)) {
            return Status::Error("failed to align buffer region");
        }

        // Align up device addr
        void* alignedDeviceAddr = nullptr;
        std::size_t deviceOffset = 0;
        if (!AlignAddress(deviceAddr, alignment, alignedDeviceAddr, deviceOffset) ||
            deviceOffset != offset) {
            return Status::Error("local and device addresses cannot share requested alignment");
        }

        region.owner = std::move(owner);
        region.local_addr = alignedLocalAddr;
        region.device_addr = alignedDeviceAddr;
        return Status::OK();
    };

    Trans::AscendBuffer ascendBuffer;
    switch (type) {
        case MemoryType::HOST: {
            auto owner = ascendBuffer.MakeHostBuffer(allocationSize);
            if (!owner) { return Status::Error("failed to allocate host memory"); }
            auto* address = owner.get();
            return setRegion(std::move(owner), address, address);
        }
        case MemoryType::HOST_PINNED: {
            void* deviceAddr = nullptr;
            auto owner = ascendBuffer.MakeHostPinnedBuffer(allocationSize, &deviceAddr);
            if (!owner || !deviceAddr) {
                return Status::Error("failed to allocate host-pinned memory");
            }
            auto* localAddr = owner.get();
            return setRegion(std::move(owner), localAddr, deviceAddr);
        }
        case MemoryType::ASCEND_DEVICE: {
            auto owner = ascendBuffer.MakeDeviceBuffer(allocationSize);
            if (!owner) { return Status::Error("failed to allocate device memory"); }
            auto* address = owner.get();
            return setRegion(std::move(owner), address, address);
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
                        std::size_t slot_num, bool enable_zero, std::size_t slot_alignment)
{
    if (region_) { return Status::InvalidParam(name + " already initialized"); }
    if (slot_capacity == 0 || slot_num == 0 || slot_alignment == 0) {
        return Status::InvalidParam(
            name + ": slot_capacity, slot_num and slot_alignment must be non-zero");
    }

    std::size_t slotStride = 0;
    if (!ComputeSlotStride(slot_capacity, slot_alignment, slotStride) ||
        slot_num > std::numeric_limits<std::size_t>::max() / slotStride ||
        slot_num >= std::numeric_limits<IndexPool::Index>::max()) {
        return Status::InvalidParam(name + ": slot layout size overflow");
    }

    BufferRegion region;
    const auto total = slotStride * slot_num;
    auto status = BufferRegion::Create(type, total, slot_alignment, region);
    if (status.Failure()) { return status; }

    name_ = std::move(name);
    slot_capacity_ = slot_capacity;
    slot_stride_ = slotStride;
    slot_num_ = slot_num;
    memory_type_ = type;
    enable_zero_ = enable_zero;
    region_ = std::move(region);

    if (enable_zero_) {
        status = ZeroMemory(region_.local_addr, total);
        if (status.Failure()) {
            Reset();
            return status;
        }
    }

    index_pool_.Setup(static_cast<IndexPool::Index>(slot_num_));
    return Status::OK();
}

Status BufferPool::Allocate(Slot& slot)
{
    if (!region_) { return Status::Error("buffer pool not initialized"); }

    const auto idx = index_pool_.Acquire();
    if (idx == IndexPool::npos) {
        return Status(Status::NoSpace().Underlying(), name_ + ": no free slots");
    }

    const auto offset = static_cast<std::size_t>(idx) * slot_stride_;
    slot.local_addr = static_cast<char*>(region_.local_addr) + offset;
    slot.device_addr = static_cast<char*>(region_.device_addr) + offset;
    slot.length = slot_capacity_;
    slot.slot_index = idx;
    return Status::OK();
}

Status BufferPool::Free(std::uint32_t slot_index)
{
    if (!region_) { return Status::Error("buffer pool not initialized"); }
    if (slot_index >= slot_num_) {
        return Status::InvalidParam(name_ + ": slot_index out of range");
    }

    if (enable_zero_) {
        auto* slot = static_cast<char*>(region_.local_addr) + slot_index * slot_stride_;
        auto status = ZeroMemory(slot, slot_stride_);
        if (status.Failure()) { return status; }
    }

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
    enable_zero_ = false;
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
