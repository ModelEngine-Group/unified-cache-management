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

Status ConnectionManager::AddGroup(const AsuEndpoint& endpoint, std::uint32_t qp_num)
{
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "connection manager shutting down");
    }
    UC_DEBUG("ConnectionManager::AddGroup endpoint={} qp_num={}", endpoint.ip, qp_num);

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
                UC_WARN(
                    "ConnectionManager::AddGroup cleanup failed after capability query failure: "
                    "code={} message={}",
                    static_cast<int>(deleteStatus.code), deleteStatus.message);
            }
        }
        return capabilityStatus;
    }
    if (capabilityStatus.code == StatusCode::UNSUPPORTED) {
        capabilities = {};
        UC_DEBUG(
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
    UC_DEBUG("ConnectionManager::AddGroup OK");
    return Status::OK();
}

Status ConnectionManager::Shutdown()
{
    UC_DEBUG("ConnectionManager::Shutdown start");
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

    UC_DEBUG("ConnectionManager::Shutdown done");
    return Status::OK();
}

std::shared_ptr<ConnectionChannel> ConnectionManager::SelectConnection()
{
    if (shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }
    auto policy = routingPolicy_;
    auto channel =
        policy == RoutingPolicy::LEAST_LOADED ? SelectByLeastLoaded() : SelectByRoundRobin();
    if (channel) {
        UC_DEBUG("ConnectionManager::SelectConnection policy={} ch_id={} group_id={} inflight={}",
                 policy == RoutingPolicy::LEAST_LOADED ? "LEAST_LOADED" : "ROUND_ROBIN",
                 channel->GetChannelId(), channel->GetGroup()->GetGroupId(),
                 channel->GetInflightCount());
    } else {
        UC_DEBUG("ConnectionManager::SelectConnection policy={} NO available channel",
                 policy == RoutingPolicy::LEAST_LOADED ? "LEAST_LOADED" : "ROUND_ROBIN");
    }
    return channel;
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
    UC_DEBUG("ConnectionManager::ReportFailure ch_id={} group_id={} error_count={} threshold={}",
             channel->GetChannelId(), channel->GetGroup()->GetGroupId(), oldCount + 1,
             maxErrorCount_);
    if (oldCount + 1 < maxErrorCount_) {
        UC_DEBUG("ConnectionManager::ReportFailure below threshold, skip drain");
        return;
    }

    if (!channel->MarkForDrain()) {
        UC_DEBUG(
            "ConnectionManager::ReportFailure MarkForDrain CAS failed "
            "(already draining/failed)");
        return;
    }

    UC_DEBUG("ConnectionManager::ReportFailure MarkForDrain OK, ch_id={} state=DRAINING",
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
    UC_DEBUG("ConnectionManager::StartRecoverLoop start");
    if (recoverWorker_.joinable()) { return; }
    stopRecover_.store(false, std::memory_order_release);
    recoverWorker_ = std::thread(&ConnectionManager::RecoverLoop, this);
}

void ConnectionManager::StopRecoverLoop()
{
    UC_DEBUG("ConnectionManager::StopRecoverLoop start");
    stopRecover_.store(true, std::memory_order_release);
    if (recoverWorker_.joinable()) { recoverWorker_.join(); }
}

void ConnectionManager::RecoverLoop()
{
    UC_DEBUG("ConnectionManager::RecoverLoop started");
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
            AsuEndpoint ep = grp->GetEndpoint();
            std::uint32_t gid = grp->GetGroupId();

            std::vector<TransProvider::ConnectionHandle> new_handles;
            auto createStatus =
                provider_.CreateConnection(localIp_, ep.ip, ep.port, 1, timeout_, new_handles);

            if (!createStatus.ok()) {
                UC_DEBUG(
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
                UC_DEBUG(
                    "ConnectionManager::RecoverLoop RebuildChannel OK: new_ch_id={} group_id={}",
                    new_ch->GetChannelId(), gid);
                cacheDirty_.store(true, std::memory_order_release);
            }
        }
    }
    UC_DEBUG("ConnectionManager::RecoverLoop stopped");
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

std::shared_ptr<ConnectionChannel> ConnectionManager::SelectByRoundRobin()
{
    std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
    if (cacheDirty_.load(std::memory_order_acquire)) { RebuildChannelCache(); }

    if (channelCache_.empty()) { return nullptr; }

    auto idx = rrIndex_.fetch_add(1, std::memory_order_relaxed);
    std::size_t total = channelCache_.size();
    std::size_t start = idx % total;
    const auto maxInflightPerChannel = maxInflightPerChannel_;

    for (std::size_t i = 0; i < total; ++i) {
        std::size_t pos = (start + i) % total;
        const auto& channel = channelCache_[pos];
        if (channel->GetState() != ChannelState::ACTIVE) { continue; }
        if (channel->GetInflightCount() < maxInflightPerChannel) {
            channel->IncrementInflight();
            return channel;
        }
    }
    UC_DEBUG(
        "ConnectionManager::SelectByRoundRobin no available channel: all ACTIVE channels "
        "reached maximum inflight limit, max_inflight_per_channel={}",
        maxInflightPerChannel);
    return nullptr;
}

std::shared_ptr<ConnectionChannel> ConnectionManager::SelectByLeastLoaded()
{
    std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
    if (cacheDirty_.load(std::memory_order_acquire)) { RebuildChannelCache(); }

    if (channelCache_.empty()) { return nullptr; }

    std::shared_ptr<ConnectionChannel> selected;
    std::uint32_t min_inflight = std::numeric_limits<std::uint32_t>::max();
    const auto maxInflightPerChannel = maxInflightPerChannel_;

    for (const auto& channel : channelCache_) {
        auto inflight = channel->GetInflightCount();
        if (channel->GetState() == ChannelState::ACTIVE && inflight < maxInflightPerChannel &&
            inflight < min_inflight) {
            min_inflight = inflight;
            selected = channel;
            if (min_inflight == 0) break;
        }
    }

    if (selected) {
        selected->IncrementInflight();
        return selected;
    }
    UC_DEBUG(
        "ConnectionManager::SelectByLeastLoaded no available channel: all ACTIVE channels "
        "reached maximum inflight limit, max_inflight_per_channel={}",
        maxInflightPerChannel);
    return nullptr;
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
