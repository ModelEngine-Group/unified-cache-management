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
#include "metadata.h"
#include <cstdlib>
#include "logger/logger.h"
#include "pos_eviction_policy.h"
#include "ttl_eviction_policy.h"

namespace UC::DramStore {
namespace {
std::unique_ptr<EvictionPolicy> CreateEvictionPolicy(EvictionPolicyType type)
{
    switch (type) {
        case EvictionPolicyType::TTL: return std::make_unique<TtlEvictionPolicy>();
        case EvictionPolicyType::POSITION: return std::make_unique<PosEvictionPolicy>();
        default: break;
    }
    UC_ERROR("CreateEvictionPolicy: invalid EvictionPolicyType {}, fallback to TTL.",
             static_cast<int>(type));
    return std::make_unique<TtlEvictionPolicy>();
}
}  // namespace

ShardMetadata::ShardMetadata(const MetadataConfig& config)
    : periodicEvictor_(CreateEvictionPolicy(config.periodicType)),
      deepEvictor_(CreateEvictionPolicy(config.deepType)),
      leaseTime_(config.leaseTime)
{
}

Status ShardMetadata::StoreBegin(const BlockId& key, EntryPtr entry)
{
    ReadWriteGuard lock(mtx_);
    if (metadata_.find(key) != metadata_.end()) {
        UC_INFO("ShardMetadata StoreBegin: key already exists, skip.");
        return Status::OK();
    }
    if (entry == nullptr || !entry->IsInitial()) {
        UC_ERROR("ShardMetadata StoreBegin: entry not in initial state.");
        return Status::InvalidParam();
    }
    auto st = periodicEvictor_->AddKey(key, entry);
    if (!st.Success()) {
        UC_ERROR("ShardMetadata StoreBegin: periodicEvictor AddKey failed.");
        return st;
    }
    st = deepEvictor_->AddKey(key, entry);
    if (!st.Success()) {
        UC_ERROR("ShardMetadata StoreBegin: deepEvictor AddKey failed, rollback periodicEvictor.");
        periodicEvictor_->DeleteKey(key);
        return st;
    }
    metadata_.emplace(key, std::move(entry));
    return Status::OK();
}

Status ShardMetadata::StoreEnd(const BlockId& key)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    auto& entry = it->second;
    return entry->TryMarkReady() ? Status::OK() : Status::Error();
}

Status ShardMetadata::LoadBegin(const BlockId& key)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    auto& entry = it->second;
    if (!entry->TryIncRef()) { return Status::Error(); }
    periodicEvictor_->AccessKey(key);
    deepEvictor_->AccessKey(key);
    return Status::OK();
}

Status ShardMetadata::LoadEnd(const BlockId& key)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    auto& entry = it->second;
    return entry->TryDecRef() ? Status::OK() : Status::Error();
}

bool ShardMetadata::Exist(const BlockId& key)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return false; }
    auto& entry = it->second;
    return entry->TryMarkHit(std::chrono::system_clock::now() + leaseTime_);
}

bool ShardMetadata::Query(const BlockId& key) const
{
    ReadOnlyGuard lock(mtx_);
    return metadata_.find(key) != metadata_.end();
}

Status ShardMetadata::Delete(const BlockId& key)
{
    ReadWriteGuard lock(mtx_);
    if (metadata_.find(key) == metadata_.end()) { return Status::NotFound(); }
    periodicEvictor_->DeleteKey(key);
    deepEvictor_->DeleteKey(key);
    metadata_.erase(key);
    return Status::OK();
}

std::size_t ShardMetadata::GetKeyCnt() const noexcept
{
    ReadOnlyGuard lock(mtx_);
    return metadata_.size();
}

std::vector<BlockId> ShardMetadata::EvictPeriodic(double evict_ratio)
{
    ReadOnlyGuard lock(mtx_);
    return periodicEvictor_->GetEvictionResults(evict_ratio);
}

std::vector<BlockId> ShardMetadata::EvictDeep(double evict_ratio)
{
    ReadOnlyGuard lock(mtx_);
    return deepEvictor_->GetEvictionResults(evict_ratio);
}

Status MetadataManager::StoreBegin(const BlockId& key, EntryPtr entry)
{
    auto idx = ShardIdx(key);
    entry->shard = static_cast<uint32_t>(idx);
    // TODO: integrate BufferManager allocation for entry.
    Status st = Status::OK();
    if (!st.Success()) {
        EvictOneShard(*shards_[rand() % kShardCnt]);
        // TODO: retry BufferManager allocation, update st.
        if (!st.Success()) { EvictOneShard(*shards_[rand() % kShardCnt], true); }
    }
    return shards_[idx]->StoreBegin(key, std::move(entry));
}

void MetadataManager::EvictLoop()
{
    while (true) {
        std::unique_lock<std::mutex> lock(cvMtx_);
        cv_.wait_for(lock, evictPeriod_, [this] { return stop_.load(); });
        if (stop_.load()) { return; }
        for (auto& s : shards_) { EvictOneShard(*s); }
    }
}

void MetadataManager::EvictOneShard(ShardMetadata& s, bool deep)
{
    auto victims = deep ? s.EvictDeep(defaultEvictRatio_) : s.EvictPeriodic(defaultEvictRatio_);
    // TODO: integrate BufferManager release for victim entries.
    for (const auto& k : victims) { s.Delete(k); }
}

}  // namespace UC::DramStore
