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
#include "buffer_manager.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include "logger.h"
#include "runtime/buffer.h"
#include "runtime/device.h"

namespace kv {

constexpr std::size_t kSlotAddressAlignment = 64;

bool GetSlotStride(std::size_t capacity, std::size_t& stride)
{
    // NOTE: Ascend ACL documents an aclrtMallocHost large-block suballocation
    // layout of ALIGN_UP(len, 32) + 32 bytes with 64-byte-aligned segment
    // starts. Current HCOMM/RDMA validation did not reproduce failures without
    // the extra 32-byte tail room, so ASU keeps only the 64-byte slot-start
    // alignment for now.
    // Keep one layout for every memory type by aligning each slot start to a
    // 64-byte boundary.
    constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();
    if (capacity > kMaxSize - (kSlotAddressAlignment - 1)) { return false; }

    stride = (capacity + kSlotAddressAlignment - 1) / kSlotAddressAlignment * kSlotAddressAlignment;
    return true;
}

Status BufferManager::BufferRegion::Create(MemoryType type, std::size_t size, BufferRegion& region)
{
    auto buffer = runtime::Device{}.MakeBuffer();
    if (!buffer) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "failed to create runtime buffer");
    }

    switch (type) {
        case MemoryType::HOST: {
            auto owner = buffer->MakeHostBuffer(size);
            if (!owner) {
                return Status::Error(StatusCode::INTERNAL_ERROR, "failed to allocate host memory");
            }
            // HOST has one CPU-visible address, which is also passed to the
            // provider when it registers the region as MEM_HOST.
            region = {owner, owner.get(), owner.get(), TransProvider::MemType::MEM_HOST};
            return Status::OK();
        }
        case MemoryType::HOST_PINNED: {
            if (!buffer->SupportsHostMappedDeviceBuffer()) {
                return Status::Error(StatusCode::UNSUPPORTED,
                                     "host-mapped device buffer not supported by runtime");
            }
            void* deviceAddr = nullptr;
            auto owner = buffer->MakeHostMappedDeviceBuffer(size, &deviceAddr);
            if (!owner) {
                return Status::Error(StatusCode::INTERNAL_ERROR,
                                     "failed to allocate host-pinned memory");
            }
            region = {owner, owner.get(), deviceAddr, TransProvider::MemType::MEM_DEVICE};
            return Status::OK();
        }
        case MemoryType::DEVICE: {
            auto owner = buffer->MakeDeviceBuffer(size);
            if (!owner) {
                return Status::Error(StatusCode::INTERNAL_ERROR,
                                     "failed to allocate device memory");
            }
            region = {owner, owner.get(), owner.get(), TransProvider::MemType::MEM_DEVICE};
            return Status::OK();
        }
        default: return Status::Error(StatusCode::INVALID_ARGUMENT, "unsupported memory type");
    }
}

void BufferManager::BufferRegion::Reset()
{
    owner.reset();
    localAddr = nullptr;
    deviceAddr = nullptr;
    providerMemType = TransProvider::MemType::MEM_HOST;
}

BufferManager::~BufferManager() { Shutdown(); }

void BufferManager::Shutdown()
{
    tokenId_ = 0;
    region_.Reset();
    slot_capacity_ = 0;
    slot_stride_ = 0;
    slot_num_ = 0;
}

Status BufferManager::Init(std::string name, MemoryType type, std::size_t slot_capacity,
                           std::size_t slot_num)
{
    if (region_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name + " already initialized");
    }
    if (slot_capacity == 0 || slot_num == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             name + ": slot_capacity and slot_num must be non-zero");
    }
    std::size_t slotStride = 0;
    if (!GetSlotStride(slot_capacity, slotStride) ||
        slot_num > std::numeric_limits<std::size_t>::max() / slotStride) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name + ": slot layout size overflow");
    }

    name_ = std::move(name);
    memory_type_ = type;
    slot_capacity_ = slot_capacity;
    slot_stride_ = slotStride;
    slot_num_ = slot_num;

    std::size_t total = slot_stride_ * slot_num_;

    auto allocStatus = BufferRegion::Create(memory_type_, total, region_);
    if (!allocStatus.ok()) { return allocStatus; }

    if (memory_type_ == MemoryType::DEVICE) {
        if (const auto memStatus = runtime::Memset(region_.localAddr, total, 0);
            !memStatus.ok()) {
            region_.Reset();
            return Status::Error(StatusCode::INTERNAL_ERROR, name_ + ": failed to zero memory");
        }
    } else {
        std::memset(region_.localAddr, 0, total);
    }

    index_pool_.Setup(static_cast<IndexPool::Index>(slot_num));
    tokenId_ = 0;

    return Status::OK();
}

Status BufferManager::GetRegisterMemoryDesc(TransProvider::RegisterMemoryDesc& desc) const
{
    if (!region_) { return Status::Error(StatusCode::NOT_INITIALIZED, name_ + " not initialized"); }
    desc = {region_.providerMemType, reinterpret_cast<uintptr_t>(region_.deviceAddr),
            slot_stride_ * slot_num_, reinterpret_cast<uintptr_t>(region_.localAddr)};
    return Status::OK();
}

Status BufferManager::Allocate(std::size_t size, ScatterGatherEntry& sge)
{
    if (!region_) { return Status::Error(StatusCode::NOT_INITIALIZED, name_ + " not initialized"); }
    if (size == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name_ + ": size must be non-zero");
    }
    if (size > slot_capacity_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name_ + ": size exceeds slot_capacity");
    }

    auto idx = index_pool_.Acquire();
    if (idx == IndexPool::npos) {
        return Status::Error(StatusCode::RESOURCE_BUSY, name_ + ": no free slots");
    }
    const auto offset = idx * slot_stride_;
    sge.local_addr =
        reinterpret_cast<std::uint64_t>(static_cast<char*>(region_.localAddr) + offset);
    sge.device_addr =
        reinterpret_cast<std::uint64_t>(static_cast<char*>(region_.deviceAddr) + offset);
    sge.length = static_cast<std::uint32_t>(size);
    sge.tokenId = tokenId_;
    sge.slot_index = idx;
    sge.memory_type = memory_type_;
    return Status::OK();
}

Status BufferManager::Free(std::uint32_t slot_index)
{
    if (!region_) { return Status::Error(StatusCode::NOT_INITIALIZED, name_ + " not initialized"); }
    if (slot_index >= slot_num_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name_ + ": slot_index out of range");
    }
    auto* p = static_cast<char*>(region_.localAddr) + slot_index * slot_stride_;
    if (memory_type_ == MemoryType::DEVICE) {
        if (const auto memStatus = runtime::Memset(p, slot_stride_, 0); !memStatus.ok()) {
            return Status::Error(StatusCode::INTERNAL_ERROR, name_ + ": failed to zero memory");
        }
    } else {
        std::memset(p, 0, slot_stride_);
    }
    index_pool_.Release(static_cast<IndexPool::Index>(slot_index));
    return Status::OK();
}

bool BufferManager::IsValidPointer(const void* ptr) const
{
    if (!ptr || !region_) { return false; }
    auto* base = static_cast<const char*>(region_.localAddr);
    auto* p = static_cast<const char*>(ptr);
    if (p < base || p >= base + slot_stride_ * slot_num_) { return false; }
    auto offset = static_cast<std::size_t>(p - base);
    return (offset % slot_stride_) == 0;
}

}  // namespace kv
