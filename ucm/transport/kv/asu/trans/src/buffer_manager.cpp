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
#include <acl/acl.h>
#include <cstdlib>
#include <cstring>
#include "trans/ascend/ascend_buffer.h"

namespace UC::ASU {

BufferManager::~BufferManager()
{
    memory_.reset();
    slot_size_ = 0;
    slot_num_ = 0;
}

Status BufferManager::Init(std::string name, MemoryType type, std::size_t slot_size,
                           std::size_t slot_num)
{
    if (memory_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name + " already initialized");
    }
    if (slot_size == 0 || slot_num == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             name + ": slot_size and slot_num must be non-zero");
    }

    name_ = std::move(name);
    memory_type_ = type;
    slot_size_ = slot_size;
    slot_num_ = slot_num;

    std::size_t total = slot_size * slot_num;

    Trans::AscendBuffer allocator;
    switch (memory_type_) {
        case MemoryType::HOST: memory_ = allocator.MakeHostBuffer(total); break;
        case MemoryType::HOST_PINNED: memory_ = allocator.MakeHostBuffer4DirectIo(total); break;
        case MemoryType::ASCEND_DEVICE: memory_ = allocator.MakeDeviceBuffer(total); break;
        default:
            return Status::Error(StatusCode::INVALID_ARGUMENT, name_ + ": unsupported memory type");
    }

    if (!memory_) {
        return Status::Error(StatusCode::INTERNAL_ERROR, name_ + ": failed to allocate memory");
    }

    if (memory_type_ == MemoryType::ASCEND_DEVICE) {
        if (aclrtMemset(memory_.get(), total, 0, total) != ACL_SUCCESS) {
            memory_.reset();
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 name_ + ": failed to zero device memory");
        }
    } else {
        std::memset(memory_.get(), 0, total);
    }

    index_pool_.Setup(static_cast<IndexPool::Index>(slot_num));

    return Status::OK();
}

Status BufferManager::Allocate(std::size_t size, ScatterGatherEntry& sge)
{
    if (!memory_) { return Status::Error(StatusCode::NOT_INITIALIZED, name_ + " not initialized"); }
    if (size == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name_ + ": size must be non-zero");
    }
    if (size > slot_size_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name_ + ": size exceeds slot_size");
    }

    auto idx = index_pool_.Acquire();
    if (idx == IndexPool::npos) {
        return Status::Error(StatusCode::RESOURCE_BUSY, name_ + ": no free slots");
    }
    void* addr = static_cast<char*>(memory_.get()) + idx * slot_size_;
    sge.addr = reinterpret_cast<std::uint64_t>(addr);
    sge.length = static_cast<std::uint32_t>(size);
    sge.lkey = static_cast<std::uint32_t>(idx + 1);
    sge.slot_index = idx;
    return Status::OK();
}

Status BufferManager::Free(std::uint32_t slot_index)
{
    if (!memory_) { return Status::Error(StatusCode::NOT_INITIALIZED, name_ + " not initialized"); }
    if (slot_index >= slot_num_) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, name_ + ": slot_index out of range");
    }
    auto* p = static_cast<char*>(memory_.get()) + slot_index * slot_size_;
    if (memory_type_ == MemoryType::ASCEND_DEVICE) {
        if (aclrtMemset(p, slot_size_, 0, slot_size_) != ACL_SUCCESS) {
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 name_ + ": failed to zero device memory");
        }
    } else {
        std::memset(p, 0, slot_size_);
    }
    index_pool_.Release(static_cast<IndexPool::Index>(slot_index));
    return Status::OK();
}

bool BufferManager::IsValidPointer(const void* ptr) const
{
    if (!ptr || !memory_) { return false; }
    auto* base = static_cast<const char*>(memory_.get());
    auto* p = static_cast<const char*>(ptr);
    if (p < base || p >= base + slot_size_ * slot_num_) { return false; }
    auto offset = static_cast<std::size_t>(p - base);
    return (offset % slot_size_) == 0;
}

}  // namespace UC::ASU
