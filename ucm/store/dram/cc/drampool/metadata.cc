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
#include <stdexcept>
#include "logger/logger.h"
#include "pos_eviction_policy.h"
#include "ttl_eviction_policy.h"

namespace UC::DramPool {
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
        UC_INFO("ShardMetadata StoreBegin: key already exists.");
        return Status::DuplicateKey();
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

Status ShardMetadata::LoadBegin(const BlockId& key, EntryPtr& entry)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    auto& existingEntry = it->second;
    if (!existingEntry->TryIncRef()) { return Status::Error(); }
    periodicEvictor_->AccessKey(key);
    deepEvictor_->AccessKey(key);
    entry = existingEntry;
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

std::vector<EntryPtr> ShardMetadata::EvictPeriodic(double evict_ratio)
{
    ReadOnlyGuard lock(mtx_);
    return periodicEvictor_->GetEvictionResults(evict_ratio);
}

std::vector<EntryPtr> ShardMetadata::EvictDeep(double evict_ratio)
{
    ReadOnlyGuard lock(mtx_);
    return deepEvictor_->GetEvictionResults(evict_ratio);
}

Status MetadataManager::StoreBegin(const BlockId& key, EntryPtr entry)
{
    auto idx = ShardIdx(key);
    entry->shard = static_cast<uint32_t>(idx);
    auto st = bufferManager_.Allocate(entry->size, entry->buffer);
    if (st == Status::NoSpace()) {
        // TODO: Maybe the random shard doesn't have the data of current size
        EvictOneShard(*shards_[rand() % kShardCnt]);
        st = bufferManager_.Allocate(entry->size, entry->buffer);
    }
    if (st == Status::NoSpace()) {
        EvictOneShard(*shards_[rand() % kShardCnt], true);
        st = bufferManager_.Allocate(entry->size, entry->buffer);
    }
    if (!st.Success()) {
        UC_ERROR("StoreBegin: Allocate for size {} failed, status {}.", entry->size, st.ToString());
        return Status::Error();
    }
    const auto bufSize = entry->size;
    const auto bufSlot = entry->buffer.slot;
    st = shards_[idx]->StoreBegin(key, std::move(entry));
    if (!st.Success()) {
        auto freeSt = bufferManager_.Free(bufSize, bufSlot);
        if (!freeSt.Success()) {
            UC_ERROR("StoreBegin: Free slot {} failed, status {}.", bufSlot, freeSt.ToString());
        }
    }
    return st;
}

void MetadataManager::EvictOneShard(ShardMetadata& s, bool deep)
{
    auto victims = deep ? s.EvictDeep(defaultEvictRatio_) : s.EvictPeriodic(defaultEvictRatio_);
    for (const auto& entry : victims) {
        auto st = bufferManager_.Free(entry->size, entry->buffer.slot);
        if (!st.Success()) {
            UC_ERROR("EvictOneShard: Free slot {} failed, status {}.", entry->buffer.slot,
                     st.ToString());
        }
        s.Delete(entry->key);
    }
}

}  // namespace UC::DramPool
