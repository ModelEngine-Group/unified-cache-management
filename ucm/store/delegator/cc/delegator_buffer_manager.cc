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
#include "delegator_buffer_manager.h"
#include <cstdint>
#include <limits>

namespace UC::Delegator {
namespace {

constexpr std::size_t kAlignmentMask = BufferManager::kAlignmentBytes - 1;

bool AlignUp(std::size_t value, std::size_t& aligned)
{
    if (value == 0 || value > std::numeric_limits<std::size_t>::max() - kAlignmentMask) {
        return false;
    }
    aligned = (value + kAlignmentMask) & ~kAlignmentMask;
    return true;
}

}  // namespace

Status BufferManager::Init(std::size_t shard_size, std::size_t slot_num)
{
    // Compute aligned shard size
    std::size_t alignedSize = 0;
    if (!AlignUp(shard_size, alignedSize)) {
        return Status::InvalidParam("invalid delegator shard size");
    }

    auto status = pool_.Init("delegator_buffer_pool", BufferPool::MemoryType::ASCEND_DEVICE,
                             alignedSize, slot_num, false, kAlignmentBytes);
    if (status.Failure()) { return status; }

    aligned_size_ = alignedSize;
    return Status::OK();
}

std::size_t BufferManager::Offset(const BufferPool::Slot& slot) const
{
    const auto baseAddress = reinterpret_cast<std::uintptr_t>(pool_.GetDeviceAddr());
    const auto slotAddress = reinterpret_cast<std::uintptr_t>(slot.device_addr);
    return static_cast<std::size_t>(slotAddress - baseAddress);
}

}  // namespace UC::Delegator
