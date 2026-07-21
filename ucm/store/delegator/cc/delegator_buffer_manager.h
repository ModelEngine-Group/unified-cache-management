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
 */
#pragma once

#include <cstddef>
#include "pool/buffer_pool.h"

namespace UC::Delegator {

class BufferManager {
public:
    static constexpr std::size_t kAlignmentBytes = 16 * 1024;

    Status Init(std::size_t shard_size, std::size_t slot_num);
    Status Acquire(BufferPool::Slot& slot) { return pool_.Allocate(slot); }
    Status Release(const BufferPool::Slot& slot) { return pool_.Free(slot.slot_index); }

    bool IsInitialized() const { return pool_.IsInitialized(); }
    std::size_t AlignedSize() const { return aligned_size_; }
    std::size_t Size() const { return pool_.GetTotalSize(); }
    std::size_t SlotCount() const { return pool_.GetSlotCount(); }
    void* DeviceAddress() const { return pool_.GetDeviceAddr(); }
    std::size_t Offset(const BufferPool::Slot& slot) const;

private:
    BufferPool pool_;
    std::size_t aligned_size_{0};
};

}  // namespace UC::Delegator
