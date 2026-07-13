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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_METADATA_H
#define UNIFIEDCACHE_DRAM_STORE_CC_METADATA_H

#include <chrono>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>
#include "entry.h"
#include "eviction_policy.h"
#include "status/status.h"

namespace UC::DramStore {

struct MetadataConfig {
    EvictionPolicyType periodicType;
    EvictionPolicyType deepType;
    std::chrono::milliseconds leaseTime;
    double defaultEvictRatio;
};

/**
 * @brief Per-shard metadata container owning the key→Entry lookup map and
 *        eviction coordination logic.
 *
 * ShardMetadata is the single entry-lifecycle surface for a shard. All public
 * operations take an RwLock for read-only or read-write guard.
 */
class ShardMetadata {
public:
    explicit ShardMetadata(const MetadataConfig& config);

    Status AddKey(const BlockId& key, EntryPtr entry);
    Status AccessKey(const BlockId& key);
    Status DeleteKey(const BlockId& key);
    bool QueryKey(const BlockId& key) const;
    bool QueryKey(const BlockId& key, EntryPtr& entry) const;
    std::size_t GetKeyCnt() const noexcept;
    std::vector<BlockId> EvictPeriodic();
    std::vector<BlockId> EvictDeep();
    std::vector<BlockId> EvictDeep(double evict_ratio);

private:
    mutable RwLock mtx_;
    std::unordered_map<BlockId, EntryPtr, UC::Detail::BlockIdHasher> metadata_;
    std::unique_ptr<EvictionPolicy> periodicEvictor_;
    std::unique_ptr<EvictionPolicy> deepEvictor_;
    std::chrono::milliseconds leaseTime_;
    double defaultEvictRatio_;
};

}  // namespace UC::DramStore

#endif
