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
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include "trans/detail/reserved_buffer.h"

namespace UC {

bool VariableBufferPool::ComputeAllocationLayout(std::size_t requestedSize,
                                                 std::size_t allocationAlignment,
                                                 std::size_t& allocatedSize,
                                                 std::uint32_t& requiredUnits)
{
    constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();
    if (requestedSize == 0 || allocationAlignment == 0 ||
        requestedSize > kMaxSize - (allocationAlignment - 1)) {
        return false;
    }

    allocatedSize =
        (requestedSize + allocationAlignment - 1) / allocationAlignment * allocationAlignment;
    const auto units = allocatedSize / allocationAlignment;
    if (units > std::numeric_limits<std::uint32_t>::max()) { return false; }

    requiredUnits = static_cast<std::uint32_t>(units);
    return true;
}

Status VariableBufferPool::Init(std::string name, MemoryType memoryType,
                                std::size_t totalCapacity, std::uint32_t metadataNodeCapacity,
                                bool enableZero, std::size_t allocationAlignment)
{
    if (IsInitialized()) { return Status::InvalidParam(name + " already initialized"); }

    std::size_t alignedCapacity = 0;
    std::uint32_t totalUnits = 0;
    if (!ComputeAllocationLayout(totalCapacity, allocationAlignment, alignedCapacity,
                                 totalUnits)) {
        return Status::InvalidParam(
            name + ": total_capacity or allocation_alignment is invalid or too large");
    }
    if (metadataNodeCapacity < 3 ||
        metadataNodeCapacity >
            static_cast<std::uint32_t>(std::numeric_limits<OffsetAllocator::NodeIndex>::max())) {
        return Status::InvalidParam(name + ": metadata_node_capacity is out of range");
    }

    BufferRegion region;
    auto status = BufferRegion::Create(memoryType, alignedCapacity, region);
    if (status.Failure()) { return status; }

    std::unique_ptr<OffsetAllocator::Allocator> allocator;
    try {
        allocator =
            std::make_unique<OffsetAllocator::Allocator>(totalUnits, metadataNodeCapacity);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }

    name_ = std::move(name);
    memoryType_ = memoryType;
    totalCapacity_ = alignedCapacity;
    allocationAlignment_ = allocationAlignment;
    enableZero_ = enableZero;
    region_ = std::move(region);
    allocator_ = std::move(allocator);

    if (enableZero_) {
        status = ZeroMemory(region_.localAddr, totalCapacity_);
        if (status.Failure()) {
            Reset();
            return status;
        }
    }
    return Status::OK();
}

Status VariableBufferPool::Allocate(std::size_t requestedSize, BufferHandle& handle)
{
    if (!IsInitialized()) { return Status::Error("buffer pool not initialized"); }

    std::size_t allocatedSize = 0;
    std::uint32_t requiredUnits = 0;
    if (!ComputeAllocationLayout(requestedSize, allocationAlignment_, allocatedSize,
                                 requiredUnits)) {
        return Status::InvalidParam(name_ + ": requested_size is invalid or too large");
    }

    const auto allocation = allocator_->Allocate(requiredUnits);
    if (allocation.offset == OffsetAllocator::NO_SPACE) {
        return Status(Status::NoSpace().Underlying(), name_ + ": no suitable free region");
    }

    const auto byteOffset = static_cast<std::size_t>(allocation.offset) * allocationAlignment_;
    BufferHandle result;
    result.owner_ = this;
    result.allocation_ = allocation;
    result.requestedSize_ = requestedSize;
    result.allocatedSize_ = allocatedSize;
    result.offset_ = byteOffset;
    result.localAddr_ = static_cast<char*>(region_.localAddr) + byteOffset;
    result.deviceAddr_ = static_cast<char*>(region_.deviceAddr) + byteOffset;
    handle = result;
    return Status::OK();
}

Status VariableBufferPool::Free(const BufferHandle& handle)
{
    if (!IsInitialized()) { return Status::Error("buffer pool not initialized"); }
    if (handle.owner_ != this) {
        return Status::InvalidParam(name_ + ": allocation handle belongs to another pool");
    }

    if (!enableZero_) {
        if (!allocator_->Free(handle.allocation_)) {
            return Status::InvalidParam(name_ + ": invalid allocation handle");
        }
        return Status::OK();
    }

    // Keep the block allocated while clearing. On failure, the caller retains the live handle.
    auto status = ZeroMemory(handle.localAddr_, handle.allocatedSize_);
    if (status.Failure()) { return status; }

    if (!allocator_->Free(handle.allocation_)) {
        return Status::InvalidParam(name_ + ": invalid allocation handle");
    }
    return Status::OK();
}

void VariableBufferPool::Reset()
{
    allocator_.reset();
    region_.Reset();
    name_.clear();
    memoryType_ = MemoryType::Host;
    totalCapacity_ = 0;
    allocationAlignment_ = kDefaultAllocationAlignment;
    enableZero_ = false;
}

Status VariableBufferPool::ZeroMemory(void* address, std::size_t size) const
{
    if (memoryType_ == MemoryType::Device) {
        const auto status = Trans::Memset(address, size);
        if (status.Failure()) { return Status::Error(name_ + ": failed to zero device memory"); }
    } else {
        std::memset(address, 0, size);
    }
    return Status::OK();
}

}  // namespace UC
