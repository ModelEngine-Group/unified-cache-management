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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_EVICTION_POLICY_H
#define UNIFIEDCACHE_DRAM_STORE_CC_EVICTION_POLICY_H

#include <set>
#include <unordered_map>
#include <vector>
#include "entry.h"
#include "status/status.h"

namespace UC::DramStore {

enum class EvictionPolicyType {
    TTL = 0,
    POSITION,
};

/**
 * @brief Abstract interface for cache eviction policies.
 *
 * An EvictionPolicy tracks which cached blocks are eligible for removal
 * and produces a list of victim block identifiers when eviction is
 * requested.
 *
 * Thread safety: The interface does not provide internal synchronization.
 * Callers must ensure that concurrent access to the same instance is
 * externally synchronized when required.
 */
class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;

    /**
     * @brief Registers a new entry under the given key.
     *
     * @param key Block identifier of the entry.
     * @param entry Shared pointer to the entry to register.
     * @return Status
     *   - Status::OK() on success.
     *   - Status::DuplicateKey() if the key is already registered.
     *   - Status::InvalidParam() if entry is nullptr.
     */
    virtual Status AddKey(const BlockId& key, EntryPtr entry) = 0;

    /**
     * @brief Remove the entry registered under the given key.
     *
     * @param key Block identifier to remove.
     * @return Status
     *   - Status::OK() on success.
     *   - Status::NotFound() if the key is not registered.
     */
    virtual Status DeleteKey(const BlockId& key) = 0;

    /**
     * @brief Notify the policy that the given key was accessed.
     *
     * Used by recency- or frequency-based policies (e.g. LRU, LFU) to
     * update ordering. Policies that do not track access patterns may
     * implement this as a no-op.
     *
     * @param key Block identifier that was accessed.
     * @return Status indicating the result of the access notification.
     */
    virtual Status AccessKey(const BlockId& key) = 0;

    /**
     * @brief Compute the list of blocks to evict.
     *
     * @param evict_ratio Hint ratio in [0.0, 1.0] indicating the fraction
     *                     of entries to consider for eviction. Concrete
     *                     policies may ignore this hint.
     * @return Vector of block identifiers selected as eviction victims.
     *         The ordering is policy-defined.
     */
    virtual std::vector<BlockId> GetEvictionResults(double evict_ratio) = 0;
};

/**
 * @brief Common base for eviction policies backed by a std::multiset ordered
 *        by a policy-specific comparator.
 *
 * Provides shared implementations of AddKey, DeleteKey, and AccessKey.
 * Subclasses must supply their own comparator type as the template parameter
 * and implement GetEvictionResults().
 *
 * @tparam Cmp Comparator type for ordering EntryPtr in the multiset.
 */
template <typename Cmp>
class OrderedEvictionPolicy : public EvictionPolicy {
public:
    Status AddKey(const BlockId& key, EntryPtr entry) override
    {
        if (entry == nullptr) { return Status::InvalidParam(); }
        if (index_.find(key) != index_.end()) { return Status::DuplicateKey(); }
        auto it = entries_.insert(std::move(entry));
        index_.emplace(key, it);
        return Status::OK();
    }

    Status DeleteKey(const BlockId& key) override
    {
        auto mapIt = index_.find(key);
        if (mapIt == index_.end()) { return Status::NotFound(); }
        entries_.erase(mapIt->second);
        index_.erase(mapIt);
        return Status::OK();
    }

    Status AccessKey(const BlockId& /*key*/) override { return Status::OK(); }

protected:
    using EntrySet = std::multiset<EntryPtr, Cmp>;
    using EntryIter = typename EntrySet::iterator;

    EntrySet entries_;
    std::unordered_map<BlockId, EntryIter, UC::Detail::BlockIdHasher> index_;
};

}  // namespace UC::DramStore

#endif
