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
    UC_ERROR("CreateEvictionPolicy: invalid EvictionPolicyType {}.", static_cast<int>(type));
    return nullptr;
}
}  // namespace

ShardMetadata::ShardMetadata(const MetadataConfig& config)
    : periodicEvictor_(CreateEvictionPolicy(config.periodicType)),
      deepEvictor_(CreateEvictionPolicy(config.deepType)),
      leaseTime_(config.leaseTime),
      defaultEvictRatio_(config.defaultEvictRatio)
{
}

Status ShardMetadata::AddKey(const BlockId& key, EntryPtr entry)
{
    ReadWriteGuard lock(mtx_);
    if (metadata_.find(key) != metadata_.end()) {
        UC_INFO("ShardMetadata AddKey: key already exists, skip.");
        return Status::OK();
    }
    auto st = periodicEvictor_->AddKey(key, entry);
    if (!st.Success()) {
        UC_ERROR("ShardMetadata AddKey: periodicEvictor AddKey failed.");
        return st;
    }
    st = deepEvictor_->AddKey(key, entry);
    if (!st.Success()) {
        UC_ERROR("ShardMetadata AddKey: deepEvictor AddKey failed, rollback periodicEvictor.");
        periodicEvictor_->DeleteKey(key);
        return st;
    }
    metadata_.emplace(key, std::move(entry));
    return Status::OK();
}

Status ShardMetadata::AccessKey(const BlockId& key)
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) { return Status::NotFound(); }
    it->second->SetLeaseTimeout(std::chrono::system_clock::now() + leaseTime_);
    periodicEvictor_->AccessKey(key);
    deepEvictor_->AccessKey(key);
    return Status::OK();
}

Status ShardMetadata::DeleteKey(const BlockId& key)
{
    ReadWriteGuard lock(mtx_);
    if (metadata_.find(key) == metadata_.end()) { return Status::NotFound(); }
    periodicEvictor_->DeleteKey(key);
    deepEvictor_->DeleteKey(key);
    metadata_.erase(key);
    return Status::OK();
}

bool ShardMetadata::QueryKey(const BlockId& key) const
{
    ReadOnlyGuard lock(mtx_);
    return metadata_.find(key) != metadata_.end();
}

bool ShardMetadata::QueryKey(const BlockId& key, EntryPtr& entry) const
{
    ReadOnlyGuard lock(mtx_);
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        entry = nullptr;
        return false;
    }
    entry = it->second;
    return true;
}

std::size_t ShardMetadata::GetKeyCnt() const noexcept
{
    ReadOnlyGuard lock(mtx_);
    return metadata_.size();
}

std::vector<BlockId> ShardMetadata::EvictPeriodic()
{
    ReadOnlyGuard lock(mtx_);
    return periodicEvictor_->GetEvictionResults(defaultEvictRatio_);
}

std::vector<BlockId> ShardMetadata::EvictDeep() { return EvictDeep(defaultEvictRatio_); }

std::vector<BlockId> ShardMetadata::EvictDeep(double evict_ratio)
{
    ReadOnlyGuard lock(mtx_);
    return deepEvictor_->GetEvictionResults(evict_ratio);
}

}  // namespace UC::DramStore
