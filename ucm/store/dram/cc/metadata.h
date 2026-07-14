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

    /**
     * @brief Begin storing a new key. Write-locks the shard.
     * @param key BlockId to register.
     * @param entry Shared ownership of the Entry to associate with the key.
     * @return Status::OK() on success; Status::InvalidParam() for a nullptr or
     *         non-initial entry.
     */
    Status StoreBegin(const BlockId& key, EntryPtr entry);

    /**
     * @brief Finalize storage: transition the entry from INITIALIZED to READY.
     *        Read-locks the shard.
     * @return Status::OK() on success; Status::NotFound() if the key is
     *         missing; Status::Error() if the entry is not INITIALIZED.
     */
    Status StoreEnd(const BlockId& key);

    /**
     * @brief Begin a load: increment the entry's refCnt and notify both
     *        eviction policies of the access. Read-locks the shard.
     * @return Status::OK() on success; Status::NotFound() if the key is
     *         missing; Status::Error() if the entry is not READY.
     */
    Status LoadBegin(const BlockId& key);

    /**
     * @brief End a load: decrement the entry's refCnt. Read-locks the shard.
     * @return Status::OK() on success; Status::NotFound() if the key is
     *         missing; Status::Error() if the entry is not READY or refCnt
     *         is already 0.
     */
    Status LoadEnd(const BlockId& key);

    /**
     * @brief Check whether the key is present and READY, refreshing its lease
     *        timeout on a hit. Read-locks the shard.
     * @return true if the key exists and the entry is READY (lease refreshed);
     *         false otherwise (missing or not READY).
     */
    bool Exist(const BlockId& key);

    /**
     * @brief Check whether the key is present in the lookup map. Read-locks
     *        the shard.
     * @return true if the key exists, false otherwise.
     */
    bool Query(const BlockId& key) const;

    /**
     * @brief Delete a key and its entry. Write-locks the shard. Removes the
     *        key from both eviction policies and the lookup map.
     * @return Status::OK() on success; Status::NotFound() if the key is
     *         missing.
     */
    Status Delete(const BlockId& key);

    /**
     * @brief Return the number of keys tracked by this shard. Read-locks.
     */
    std::size_t GetKeyCnt() const noexcept;

    /**
     * @brief Run the eviction policy and return the victim keys.
     *        Read-locks the shard.
     * @param evict_ratio Fraction of eligible entries to evict, in [0, 1].
     * @return Keys selected for eviction.
     */
    std::vector<BlockId> EvictPeriodic(double evict_ratio);
    std::vector<BlockId> EvictDeep(double evict_ratio);

private:
    mutable RwLock mtx_;
    std::unordered_map<BlockId, EntryPtr, UC::Detail::BlockIdHasher> metadata_;
    std::unique_ptr<EvictionPolicy> periodicEvictor_;
    std::unique_ptr<EvictionPolicy> deepEvictor_;
    std::chrono::milliseconds leaseTime_;
};

}  // namespace UC::DramStore

#endif
