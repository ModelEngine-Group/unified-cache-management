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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_TTL_EVICTION_POLICY_H
#define UNIFIEDCACHE_DRAM_STORE_CC_TTL_EVICTION_POLICY_H

#include <chrono>
#include <vector>
#include "entry.h"
#include "eviction_policy.h"
#include "logger/logger.h"

namespace UC::DramPool {

struct TtlEntryCmp {
    bool operator()(const EntryPtr& a, const EntryPtr& b) const noexcept
    {
        return a->lifeTimeout < b->lifeTimeout;
    }
};

/**
 * @brief Eviction policy that removes all entries whose lifetime has expired.
 *
 * Entries are ordered by lifeTimeout ascending. GetEvictionResults iterates
 * from the earliest-expiring entry and breaks at the first entry whose
 * lifeTimeout is still in the future.
 */
class TtlEvictionPolicy : public OrderedEvictionPolicy<TtlEntryCmp> {
public:
    std::vector<EntryPtr> GetEvictionResults(double /*evict_ratio*/) override
    {
        std::vector<EntryPtr> victims;
        const auto now = std::chrono::system_clock::now();
        for (const auto& entry : entries_) {
            if (entry->lifeTimeout > now) { break; }
            if (!entry->TryMarkEvicting(now)) { continue; }
            victims.push_back(entry);
        }
        if (!victims.empty()) {
            UC_INFO("TtlEvictionPolicy evict {} of {} entries.", victims.size(), entries_.size());
        }
        return victims;
    }
};

}  // namespace UC::DramPool

#endif
