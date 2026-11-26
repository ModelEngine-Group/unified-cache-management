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
#include <algorithm>
#include <chrono>
#include <cstdint>
#include "connection_internal.h"
#include "logger.h"

namespace kv {

ConnectionManager::ConnectionManager(TransProvider& provider, const std::string& localIp,
                                     std::uint32_t timeout, std::uint32_t maxErrorCount)
    : provider_(provider), localIp_(localIp), timeout_(timeout), maxErrorCount_(maxErrorCount)
{
}

ConnectionManager::~ConnectionManager() { Shutdown(); }

Status ConnectionManager::AddGroup(const NodeEndpoint& endpoint, std::uint32_t qp_num)
{
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "connection manager shutting down");
    }
    KV_DEBUG("ConnectionManager::AddGroup endpoint={} qp_num={}", endpoint.ip, qp_num);

    std::vector<TransProvider::ConnectionHandle> handles;
    auto createStatus =
        provider_.CreateConnection(localIp_, endpoint.ip, endpoint.port, qp_num, timeout_, handles);
    if (!createStatus.ok()) { return createStatus; }
    if (handles.empty()) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "provider returned no connection handles");
    }

    ServerKvCapabilities capabilities;
    auto capabilityStatus = provider_.GetServerCapabilities(handles.front(), capabilities);
    if (!capabilityStatus.ok() && capabilityStatus.code != StatusCode::UNSUPPORTED) {
        const auto deleteStatuses = provider_.DeleteConnections(handles);
        for (const auto& deleteStatus : deleteStatuses) {
            if (!deleteStatus.ok()) {
                KV_WARN(
                    "ConnectionManager::AddGroup cleanup failed after capability query failure: "
                    "code={} message={}",
                    static_cast<int>(deleteStatus.code), deleteStatus.message);
            }
        }
        return capabilityStatus;
    }
    if (capabilityStatus.code == StatusCode::UNSUPPORTED) {
        capabilities = {};
        KV_DEBUG(
            "ConnectionManager::AddGroup server capability query is not supported. Use default "
            "configurations instead");
    }
    if (capabilities.ioQueueDepth != 0) {
        maxInflightPerChannel_ = std::min(maxInflightPerChannel_, capabilities.ioQueueDepth);
    }

    {
        std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
        std::unique_lock<std::shared_mutex> lock(structureMu_);
        auto gid = static_cast<std::uint32_t>(groups_.size());
        auto group = std::make_unique<ConnectionGroup>(gid, endpoint, capabilities);
        for (auto handle : handles) { group->AddChannel(handle, &provider_); }
        groups_.push_back(std::move(group));

        for (const auto& channel : groups_.back()->GetChannels()) {
            channelCache_.push_back(channel);
        }
    }
    cacheDirty_.store(false, std::memory_order_release);
    KV_DEBUG("ConnectionManager::AddGroup OK");
    return Status::OK();
}

Status ConnectionManager::Shutdown()
{
    KV_DEBUG("ConnectionManager::Shutdown start");
    shuttingDown_.store(true, std::memory_order_release);
    StopRecoverLoop();

    channelCache_.clear();
    {
        std::unique_lock<std::shared_mutex> lock(drainMu_);
        drainList_.clear();
    }

    // Now safe to destroy ConnectionGroup objects
    {
        std::unique_lock<std::shared_mutex> lock(structureMu_);
        groups_.clear();
    }

    KV_DEBUG("ConnectionManager::Shutdown done");
    return Status::OK();
}

std::shared_ptr<ConnectionChannel> ConnectionManager::SelectConnection(
    const std::vector<ConnectionChannel*>& excludedChannels)
{
    if (shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }

    std::vector<std::shared_ptr<ConnectionChannel>> candidates;
    if (excludedChannels.empty()) {
        auto channel = routingPolicy_ == RoutingPolicy::LEAST_LOADED ? SelectByLeastLoaded()
                                                                      : SelectByRoundRobin();
        if (channel) {
            KV_DEBUG("ConnectionManager::SelectConnection ch_id={} group_id={} inflight={}",
                     channel->GetChannelId(), channel->GetGroup()->GetGroupId(),
                     channel->GetInflightCount());
        }
        return channel;
    }

    // Two-level failover preference: channels in groups outside the excluded
    // channels' groups first, then same-group non-excluded channels as the
    // fallback. Exclusion itself uses channel object identity.
    std::vector<std::uint32_t> excludedGroups;
    for (const auto* channel : excludedChannels) {
        if (channel == nullptr) { continue; }
        excludedGroups.push_back(channel->GetGroup()->GetGroupId());
    }

    {
        std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
        if (cacheDirty_.load(std::memory_order_acquire)) { RebuildChannelCache(); }
        std::vector<std::shared_ptr<ConnectionChannel>> crossGroup;
        std::vector<std::shared_ptr<ConnectionChannel>> sameGroup;
        for (const auto& channel : channelCache_) {
            if (channel->GetState() != ChannelState::ACTIVE) { continue; }
            if (std::find(excludedChannels.begin(), excludedChannels.end(), channel.get()) !=
                excludedChannels.end()) {
                continue;
            }
            const bool groupExcluded = std::find(excludedGroups.begin(), excludedGroups.end(),
                                                 channel->GetGroup()->GetGroupId()) != excludedGroups.end();
            if (groupExcluded) {
                sameGroup.push_back(channel);
            } else {
                crossGroup.push_back(channel);
            }
        }
        candidates = std::move(crossGroup);
        candidates.insert(candidates.end(), std::make_move_iterator(sameGroup.begin()),
                         std::make_move_iterator(sameGroup.end()));
    }

    auto selected = SelectByPolicyLocked(candidates);
    if (selected) {
        KV_DEBUG("ConnectionManager::SelectConnection failover ch_id={} group_id={} "
                 "excluded={}",
                 selected->GetChannelId(), selected->GetGroup()->GetGroupId(), excludedChannels.size());
    } else {
        KV_DEBUG("ConnectionManager::SelectConnection failover NO available channel "
                 "excluded={}",
                 excludedChannels.size());
    }
    return selected;
}

std::shared_ptr<ConnectionChannel> ConnectionManager::AcquireChannel(
    const std::shared_ptr<ConnectionChannel>& preferred)
{
    if (shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }
    if (!preferred || preferred->GetState() != ChannelState::ACTIVE) { return nullptr; }
    if (preferred->GetInflightCount() >= maxInflightPerChannel_) { return nullptr; }
    preferred->IncrementInflight();
    return preferred;
}

std::shared_ptr<ConnectionChannel> ConnectionManager::GetActiveConnection()
{
    if (shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }

    std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
    if (cacheDirty_.load(std::memory_order_acquire)) { RebuildChannelCache(); }

    for (const auto& channel : channelCache_) {
        if (channel->GetState() == ChannelState::ACTIVE) { return channel; }
    }
    return nullptr;
}

void ConnectionManager::SetRoutingPolicy(RoutingPolicy policy) { routingPolicy_ = policy; }

void ConnectionManager::ReportFailure(const std::shared_ptr<ConnectionChannel>& channel)
{
    if (channel == nullptr) { return; }
    if (shuttingDown_.load(std::memory_order_acquire)) { return; }
    auto oldCount = channel->FetchAddErrorCount(1);
    KV_DEBUG("ConnectionManager::ReportFailure ch_id={} group_id={} error_count={} threshold={}",
             channel->GetChannelId(), channel->GetGroup()->GetGroupId(), oldCount + 1,
             maxErrorCount_);
    if (oldCount + 1 < maxErrorCount_) {
        KV_DEBUG("ConnectionManager::ReportFailure below threshold, skip drain");
        return;
    }

    if (!channel->MarkForDrain()) {
        KV_DEBUG(
            "ConnectionManager::ReportFailure MarkForDrain CAS failed "
            "(already draining/failed)");
        return;
    }

    KV_DEBUG("ConnectionManager::ReportFailure MarkForDrain OK, ch_id={} state=DRAINING",
             channel->GetChannelId());
    cacheDirty_.store(true, std::memory_order_release);

    {
        std::unique_lock<std::shared_mutex> lock(drainMu_);
        for (const auto& existing : drainList_) {
            if (existing.get() == channel.get()) return;
        }
        drainList_.push_back(channel);
    }
}

void ConnectionManager::ReportSuccess(const std::shared_ptr<ConnectionChannel>& channel)
{
    if (channel == nullptr) { return; }
    channel->ResetErrorCount();
}

void ConnectionManager::StartRecoverLoop()
{
    KV_DEBUG("ConnectionManager::StartRecoverLoop start");
    if (recoverWorker_.joinable()) { return; }
    stopRecover_.store(false, std::memory_order_release);
    recoverWorker_ = std::thread(&ConnectionManager::RecoverLoop, this);
}

void ConnectionManager::StopRecoverLoop()
{
    KV_DEBUG("ConnectionManager::StopRecoverLoop start");
    stopRecover_.store(true, std::memory_order_release);
    if (recoverWorker_.joinable()) { recoverWorker_.join(); }
}

void ConnectionManager::RecoverLoop()
{
    KV_DEBUG("ConnectionManager::RecoverLoop started");
    while (!stopRecover_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kRecoverIntervalMs));
        if (stopRecover_.load(std::memory_order_acquire)) { break; }

        std::vector<std::shared_ptr<ConnectionChannel>> to_recover;
        {
            std::unique_lock<std::shared_mutex> lock(drainMu_);
            to_recover.swap(drainList_);
        }

        for (auto& channel : to_recover) {
            ConnectionGroup* grp = channel->GetGroup();
            NodeEndpoint ep = grp->GetEndpoint();
            std::uint32_t gid = grp->GetGroupId();

            std::vector<TransProvider::ConnectionHandle> new_handles;
            auto createStatus =
                provider_.CreateConnection(localIp_, ep.ip, ep.port, 1, timeout_, new_handles);

            if (!createStatus.ok()) {
                KV_DEBUG(
                    "ConnectionManager::RecoverLoop RebuildChannel FAILED, keep in drain_list for "
                    "retry group_id={}",
                    gid);
                std::unique_lock<std::shared_mutex> lock(drainMu_);
                drainList_.push_back(std::move(channel));
                continue;
            }

            {
                std::unique_lock<std::shared_mutex> lock(structureMu_);
                grp->RemoveChannel(channel.get());
                auto new_ch = grp->AddChannel(new_handles[0], &provider_);
                KV_DEBUG(
                    "ConnectionManager::RecoverLoop RebuildChannel OK: new_ch_id={} group_id={}",
                    new_ch->GetChannelId(), gid);
                cacheDirty_.store(true, std::memory_order_release);
            }
        }
    }
    KV_DEBUG("ConnectionManager::RecoverLoop stopped");
}

void ConnectionManager::RebuildChannelCache()
{
    if (!cacheDirty_.load(std::memory_order_acquire)) { return; }
    std::shared_lock<std::shared_mutex> struct_lock(structureMu_);
    std::vector<std::shared_ptr<ConnectionChannel>> new_cache;
    for (const auto& g : groups_) {
        for (const auto& channel : g->GetChannels()) {
            if (channel->GetState() == ChannelState::ACTIVE) { new_cache.push_back(channel); }
        }
    }
    channelCache_ = std::move(new_cache);
    cacheDirty_.store(false, std::memory_order_release);
}

std::shared_ptr<ConnectionChannel> ConnectionManager::SelectByPolicyLocked(
    const std::vector<std::shared_ptr<ConnectionChannel>>& candidates)
{
    if (candidates.empty()) { return nullptr; }

    if (routingPolicy_ == RoutingPolicy::LEAST_LOADED) {
        std::shared_ptr<ConnectionChannel> selected;
        std::uint32_t minInflight = std::numeric_limits<std::uint32_t>::max();
        for (const auto& channel : candidates) {
            if (channel->GetState() != ChannelState::ACTIVE) { continue; }
            const auto inflight = channel->GetInflightCount();
            if (inflight < maxInflightPerChannel_ && inflight < minInflight) {
                minInflight = inflight;
                selected = channel;
                if (minInflight == 0) { break; }
            }
        }
        if (selected) {
            selected->IncrementInflight();
            return selected;
        }
        return nullptr;
    }

    // Round robin over the candidate list.
    auto idx = rrIndex_.fetch_add(1, std::memory_order_relaxed);
    const std::size_t total = candidates.size();
    const std::size_t start = idx % total;
    for (std::size_t i = 0; i < total; ++i) {
        const auto& channel = candidates[(start + i) % total];
        if (channel->GetState() != ChannelState::ACTIVE) { continue; }
        if (channel->GetInflightCount() < maxInflightPerChannel_) {
            channel->IncrementInflight();
            return channel;
        }
    }
    KV_DEBUG("ConnectionManager no available channel: all candidates reached maximum "
             "inflight limit, max_inflight_per_channel={}",
             maxInflightPerChannel_);
    return nullptr;
}

std::shared_ptr<ConnectionChannel> ConnectionManager::SelectByRoundRobin()
{
    std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
    if (cacheDirty_.load(std::memory_order_acquire)) { RebuildChannelCache(); }
    if (channelCache_.empty()) { return nullptr; }
    return SelectByPolicyLocked(channelCache_);
}

std::shared_ptr<ConnectionChannel> ConnectionManager::SelectByLeastLoaded()
{
    std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
    if (cacheDirty_.load(std::memory_order_acquire)) { RebuildChannelCache(); }
    if (channelCache_.empty()) { return nullptr; }
    return SelectByPolicyLocked(channelCache_);
}

std::int64_t ConnectionManager::TotalInflightCount()
{
    if (shuttingDown_.load(std::memory_order_acquire)) { return 0; }
    std::int64_t sum = 0;
    std::shared_lock<std::shared_mutex> lock(structureMu_);
    for (const auto& group : groups_) {
        for (const auto& channel : group->GetChannels()) { sum += channel->GetInflightCount(); }
    }
    return sum;
}

std::vector<ServerKvCapabilities> ConnectionManager::GetServerCapabilities()
{
    std::vector<ServerKvCapabilities> capabilities;
    std::shared_lock<std::shared_mutex> lock(structureMu_);
    capabilities.reserve(groups_.size());
    for (const auto& group : groups_) { capabilities.push_back(group->GetServerCapabilities()); }
    return capabilities;
}

}  // namespace kv
