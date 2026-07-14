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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_ENTRY_H
#define UNIFIEDCACHE_DRAM_STORE_CC_ENTRY_H

#include <chrono>
#include <cstdint>
#include <memory>
#include "thread/lock.h"
#include "type/types.h"

namespace UC::DramStore {

using Spinlock = UC::SpinLock;
using RwLock = UC::RwLock;
using BlockId = UC::Detail::BlockId;

enum class EntryStatus {
    INITIALIZED = 0,
    READY,
    DELETING,
};

struct Entry {
    BlockId key;
    uint32_t refCnt{0};
    uint32_t shard{0};
    uint32_t slot{0};
    void* addr{nullptr};
    std::size_t size{0};
    std::chrono::system_clock::time_point leaseTimeout{};
    EntryStatus status{EntryStatus::INITIALIZED};
    Spinlock lock;
    std::chrono::system_clock::time_point lifeTimeout{};
    uint32_t position{0};

    bool IsInitial()
    {
        SpinLockGuard guard(lock);
        return status == EntryStatus::INITIALIZED && refCnt == 0;
    }

    bool TryMarkEvicting(std::chrono::system_clock::time_point now)
    {
        SpinLockGuard guard(lock);
        if (status != EntryStatus::READY) { return false; }
        if (refCnt != 0) { return false; }
        if (leaseTimeout > now) { return false; }
        status = EntryStatus::DELETING;
        return true;
    }

    bool TryMarkReady()
    {
        SpinLockGuard guard(lock);
        if (status != EntryStatus::INITIALIZED) { return false; }
        status = EntryStatus::READY;
        return true;
    }

    bool TryMarkHit(std::chrono::system_clock::time_point timeout)
    {
        SpinLockGuard guard(lock);
        if (status != EntryStatus::READY) { return false; }
        leaseTimeout = timeout;
        return true;
    }

    bool TryIncRef()
    {
        SpinLockGuard guard(lock);
        if (status != EntryStatus::READY) { return false; }
        ++refCnt;
        return true;
    }

    bool TryDecRef()
    {
        SpinLockGuard guard(lock);
        if (status != EntryStatus::READY) { return false; }
        if (refCnt == 0) { return false; }
        --refCnt;
        return true;
    }
};

using EntryPtr = std::shared_ptr<Entry>;

}  // namespace UC::DramStore

#endif
