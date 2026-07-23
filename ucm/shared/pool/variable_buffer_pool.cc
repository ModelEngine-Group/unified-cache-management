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
#include "pool/variable_buffer_pool.h"
#include <acl/acl.h>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include "trans/ascend/ascend_buffer.h"

namespace UC {

bool VariableBufferPool::ComputeAllocationLayout(std::size_t requested_size,
                                                 std::size_t allocation_alignment,
                                                 std::size_t& allocated_size,
                                                 std::uint32_t& required_units)
{
    constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();
    if (requested_size == 0 || allocation_alignment == 0 ||
        requested_size > kMaxSize - (allocation_alignment - 1)) {
        return false;
    }

    allocated_size =
        (requested_size + allocation_alignment - 1) / allocation_alignment * allocation_alignment;
    const auto units = allocated_size / allocation_alignment;
    if (units > std::numeric_limits<std::uint32_t>::max()) { return false; }

    required_units = static_cast<std::uint32_t>(units);
    return true;
}

Status VariableBufferPool::BufferRegion::Create(MemoryType memory_type, std::size_t size,
                                                BufferRegion& region)
{
    Trans::AscendBuffer ascendBuffer;
    switch (memory_type) {
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

void VariableBufferPool::BufferRegion::Reset()
{
    owner.reset();
    local_addr = nullptr;
    device_addr = nullptr;
}

Status VariableBufferPool::Init(std::string name, MemoryType memory_type,
                                std::size_t total_capacity, std::uint32_t metadata_node_capacity,
                                bool enable_zero, std::size_t allocation_alignment)
{
    if (IsInitialized()) { return Status::InvalidParam(name + " already initialized"); }

    std::size_t alignedCapacity = 0;
    std::uint32_t totalUnits = 0;
    if (!ComputeAllocationLayout(total_capacity, allocation_alignment, alignedCapacity,
                                 totalUnits)) {
        return Status::InvalidParam(
            name + ": total_capacity or allocation_alignment is invalid or too large");
    }
    if (metadata_node_capacity < 3 ||
        metadata_node_capacity >
            static_cast<std::uint32_t>(std::numeric_limits<OffsetAllocator::NodeIndex>::max())) {
        return Status::InvalidParam(name + ": metadata_node_capacity is out of range");
    }

    BufferRegion region;
    auto status = BufferRegion::Create(memory_type, alignedCapacity, region);
    if (status.Failure()) { return status; }

    std::unique_ptr<OffsetAllocator::Allocator> allocator;
    try {
        allocator =
            std::make_unique<OffsetAllocator::Allocator>(totalUnits, metadata_node_capacity);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }

    name_ = std::move(name);
    memory_type_ = memory_type;
    total_capacity_ = alignedCapacity;
    allocation_alignment_ = allocation_alignment;
    enable_zero_ = enable_zero;
    region_ = std::move(region);
    allocator_ = std::move(allocator);

    if (enable_zero_) {
        status = ZeroMemory(region_.local_addr, total_capacity_);
        if (status.Failure()) {
            Reset();
            return status;
        }
    }
    return Status::OK();
}

Status VariableBufferPool::Allocate(std::size_t requested_size, AllocationHandle& handle)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!IsInitialized()) { return Status::Error("buffer pool not initialized"); }

    std::size_t allocatedSize = 0;
    std::uint32_t requiredUnits = 0;
    if (!ComputeAllocationLayout(requested_size, allocation_alignment_, allocatedSize,
                                 requiredUnits)) {
        return Status::InvalidParam(name_ + ": requested_size is invalid or too large");
    }

    const auto allocation = allocator_->allocate(requiredUnits);
    if (allocation.offset == OffsetAllocator::Allocation::NO_SPACE) {
        return Status(Status::NoSpace().Underlying(), name_ + ": no suitable free region");
    }

    const auto byteOffset = static_cast<std::size_t>(allocation.offset) * allocation_alignment_;
    AllocationHandle result;
    result.allocation = allocation;
    result.requested_size = requested_size;
    result.allocated_size = allocatedSize;
    result.offset = byteOffset;
    result.local_addr = static_cast<char*>(region_.local_addr) + byteOffset;
    result.device_addr = static_cast<char*>(region_.device_addr) + byteOffset;
    handle = result;
    return Status::OK();
}

Status VariableBufferPool::Free(const AllocationHandle& handle)
{
    std::size_t allocatedSize = 0;
    void* localAddress = nullptr;
    bool enableZero = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!IsInitialized()) { return Status::Error("buffer pool not initialized"); }

        const auto allocatedUnits = allocator_->allocationSize(handle.allocation);
        if (allocatedUnits == 0) {
            return Status::InvalidParam(name_ + ": invalid allocation handle");
        }

        const auto byteOffset =
            static_cast<std::size_t>(handle.allocation.offset) * allocation_alignment_;
        allocatedSize = static_cast<std::size_t>(allocatedUnits) * allocation_alignment_;
        if (byteOffset > total_capacity_ || allocatedSize > total_capacity_ - byteOffset) {
            return Status::InvalidParam(name_ + ": allocation is outside the buffer region");
        }

        localAddress = static_cast<char*>(region_.local_addr) + byteOffset;
        enableZero = enable_zero_;
        if (!enableZero) {
            if (!allocator_->free(handle.allocation)) {
                return Status::InvalidParam(name_ + ": invalid allocation handle");
            }
            return Status::OK();
        }
    }

    // Keep the block allocated while clearing. On failure, the caller retains the live handle.
    auto status = ZeroMemory(localAddress, allocatedSize);
    if (status.Failure()) { return status; }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!allocator_->free(handle.allocation)) {
        return Status::InvalidParam(name_ + ": invalid allocation handle");
    }
    return Status::OK();
}

void VariableBufferPool::Reset()
{
    allocator_.reset();
    region_.Reset();
    name_.clear();
    memory_type_ = MemoryType::HOST;
    total_capacity_ = 0;
    allocation_alignment_ = kDefaultAllocationAlignment;
    enable_zero_ = false;
}

Status VariableBufferPool::ZeroMemory(void* address, std::size_t size) const
{
    if (memory_type_ == MemoryType::ASCEND_DEVICE) {
        if (aclrtMemset(address, size, 0, size) != ACL_SUCCESS) {
            return Status::Error(name_ + ": failed to zero device memory");
        }
    } else {
        std::memset(address, 0, size);
    }
    return Status::OK();
}

}  // namespace UC
