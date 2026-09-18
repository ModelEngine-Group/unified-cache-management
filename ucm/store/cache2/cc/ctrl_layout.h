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
#include <limits>
#include "mutex/shared_mutex.h"
#include "status/status.h"

namespace UC::Cache2 {

inline constexpr size_t kInvalid{std::numeric_limits<size_t>::max()};

class CtrlLayout {
public:
    using BucketLock = SharedMutex;
    struct SlotMeta {};
    struct RankDataDesc {
        std::atomic<size_t> handle{kInvalid};

        RankDataDesc() = default;
        RankDataDesc(const RankDataDesc& o) : handle(o.handle.load(std::memory_order_relaxed)) {}
        RankDataDesc& operator=(const RankDataDesc& o)
        {
            handle.store(o.handle.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };

public:
    void InitHeader(size_t slotSize) {}
    void InitSlotRange(size_t rank) {}
    void MarkReady() {}
    bool WaitReady(size_t timeoutMs) const { return false; }
    Status SetRankDesc(size_t rank, const RankDataDesc& d) { return Status::Unsupported(); }
    Expected<RankDataDesc> GetRankDesc(size_t rank) const { return Status::Unsupported(); }
    std::atomic<size_t>* Buckets() const { return nullptr; }
    BucketLock* LockOf(size_t iBucket) const { return nullptr; }
    SlotMeta* SlotMetaArr() const { return nullptr; }
    size_t BucketCount() const { return 0; }
    size_t SlotCount() const { return 0; }
};

}  // namespace UC::Cache2
