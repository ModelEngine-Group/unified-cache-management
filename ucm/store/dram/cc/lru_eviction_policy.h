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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_LRU_EVICTION_POLICY_H
#define UNIFIEDCACHE_DRAM_STORE_CC_LRU_EVICTION_POLICY_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>
#include "entry.h"
#include "eviction_policy.h"
#include "logger/logger.h"

namespace UC::DramStore {

/**
 * @brief Eviction policy that removes least-recently-used entries first.
 *
 * The front of lruList_ is the most recently used entry, and the back is the
 * least recently used entry. GetEvictionResults scans from the back and marks
 * eligible entries as DELETING through Entry::TryMarkEvicting().
 */
class LruEvictionPolicy : public EvictionPolicy {
public:
    Status AddKey(const BlockId& key, EntryPtr entry) override
    {
        if (entry == nullptr) { return Status::InvalidParam(); }
        if (index_.find(key) != index_.end()) { return Status::DuplicateKey(); }

        lruList_.push_front(key);
        index_.emplace(key, LruNode{std::move(entry), lruList_.begin()});
        return Status::OK();
    }

    Status DeleteKey(const BlockId& key) override
    {
        auto mapIt = index_.find(key);
        if (mapIt == index_.end()) { return Status::NotFound(); }

        lruList_.erase(mapIt->second.iter);
        index_.erase(mapIt);
        return Status::OK();
    }

    Status AccessKey(const BlockId& key) override
    {
        auto mapIt = index_.find(key);
        if (mapIt == index_.end()) { return Status::NotFound(); }

        lruList_.splice(lruList_.begin(), lruList_, mapIt->second.iter);
        mapIt->second.iter = lruList_.begin();
        return Status::OK();
    }

    std::vector<BlockId> GetEvictionResults(double evictRatio) override
    {
        std::vector<BlockId> victims;
        if (!std::isfinite(evictRatio) || evictRatio <= 0.0 || index_.empty()) { return victims; }

        const double boundedRatio = std::min(evictRatio, 1.0);
        const auto target =
            static_cast<std::size_t>(static_cast<double>(index_.size()) * boundedRatio);
        const auto now = std::chrono::system_clock::now();

        for (auto it = lruList_.rbegin(); it != lruList_.rend() && victims.size() < target; ++it) {
            auto mapIt = index_.find(*it);
            if (mapIt == index_.end()) { continue; }
            if (!mapIt->second.entry->TryMarkEvicting(now)) { continue; }
            victims.push_back(*it);
        }

        if (!victims.empty()) {
            UC_INFO("LruEvictionPolicy evict {} of {} entries.", victims.size(), index_.size());
        }
        return victims;
    }

private:
    using ListIter = std::list<BlockId>::iterator;

    struct LruNode {
        EntryPtr entry;
        ListIter iter;
    };

    std::list<BlockId> lruList_;
    std::unordered_map<BlockId, LruNode, UC::Detail::BlockIdHasher> index_;
};

}  // namespace UC::DramStore

#endif
