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

namespace UC::ASU {

ConnectionManager::ConnectionManager() = default;

ConnectionManager::~ConnectionManager() { Shutdown(); }

void ConnectionManager::SetConnectionOps(CreateConnectionFunc create_fn,
                                         DeleteConnectionsFunc delete_fn)
{
    createFn_ = std::move(create_fn);
    deleteFn_ = std::move(delete_fn);
}

Status ConnectionManager::AddGroup(const AsuEndpoint& endpoint, std::uint32_t qp_num)
{
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "connection manager shutting down");
    }
    UC_DEBUG("ConnectionManager::AddGroup endpoint={} qp_num={}", endpoint.ip, qp_num);
    if (!createFn_) { return Status::Error(StatusCode::NOT_INITIALIZED, "Connection ops not set"); }
    auto handles = createFn_(endpoint, qp_num);
    if (handles.size() != qp_num) {
        UC_DEBUG("ConnectionManager::AddGroup FAILED: got {} handles, expected {}", handles.size(),
                 qp_num);
        return Status::Error(StatusCode::CONNECTION_ERROR,
                             "CreateConnection returned wrong number of handles");
    }

    {
        std::unique_lock<std::shared_mutex> lock(structureMu_);
        auto gid = static_cast<std::uint32_t>(groups_.size());
        auto group = std::make_unique<ConnectionGroup>(gid, endpoint);
        for (auto& handle : handles) { group->AddChannel(handle); }
        groups_.push_back(std::move(group));

        for (const auto& channel : groups_.back()->GetChannels()) {
            channelCache_.push_back(channel.get());
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
    {
        std::unique_lock<std::shared_mutex> lock(structureMu_);
        for (const auto& group : groups_) {
            for (const auto& channel : group->GetChannels()) {
                if (channel->GetNativeQp() && deleteFn_) { deleteFn_({channel->GetNativeQp()}); }
            }
        }
        groups_.clear();
    }
    channelCache_.clear();
    {
        std::unique_lock<std::shared_mutex> lock(drainMu_);
        drainList_.clear();
    }
    UC_DEBUG("ConnectionManager::Shutdown done");
    return Status::OK();
}

ConnectionChannel* ConnectionManager::SelectConnection()
{
    if (shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }
    auto policy = routingPolicy_;
    auto* channel =
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

void ConnectionManager::SetRoutingPolicy(RoutingPolicy policy) { routingPolicy_ = policy; }

void ConnectionManager::ReportFailure(ConnectionChannel* channel)
{
    if (shuttingDown_.load(std::memory_order_acquire)) { return; }
    auto old_count = channel->FetchAddErrorCount(1);
    UC_DEBUG("ConnectionManager::ReportFailure ch_id={} group_id={} error_count={} threshold={}",
             channel->GetChannelId(), channel->GetGroup()->GetGroupId(), old_count + 1,
             kFailureThreshold);
    if (old_count + 1 < kFailureThreshold) {
        UC_DEBUG("ConnectionManager::ReportFailure below threshold, skip drain");
        return;
    }

    if (!channel->BeginDrain()) {
        UC_DEBUG(
            "ConnectionManager::ReportFailure BeginDrain CAS failed (already draining/failed)");
        return;
    }

    UC_DEBUG("ConnectionManager::ReportFailure BeginDrain OK, ch_id={} state=DRAINING",
             channel->GetChannelId());
    cacheDirty_.store(true, std::memory_order_release);

    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
    channel->SetDrainStartTime(static_cast<std::uint64_t>(now_ms));

    {
        std::unique_lock<std::shared_mutex> lock(drainMu_);
        for (auto* existing : drainList_) {
            if (existing == channel) return;  // Already in drain list
        }
        drainList_.push_back(channel);
    }
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

        std::vector<ConnectionChannel*> completed;
        {
            std::unique_lock<std::shared_mutex> lock(drainMu_);
            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
            for (auto* channel : drainList_) {
                auto inflight = channel->GetInflightCount();
                if (inflight == 0) {
                    UC_DEBUG("ConnectionManager::RecoverLoop ch_id={} inflight=0, drain complete",
                             channel->GetChannelId());
                    completed.push_back(channel);
                } else {
                    auto elapsed = now_ms - static_cast<std::int64_t>(channel->GetDrainStartTime());
                    if (elapsed >= static_cast<std::int64_t>(kDrainTimeoutMs)) {
                        UC_DEBUG(
                            "ConnectionManager::RecoverLoop ch_id={} inflight={} elapsed={}ms, "
                            "drain timeout",
                            channel->GetChannelId(), inflight, elapsed);
                        completed.push_back(channel);
                    }
                }
            }
        }

        for (auto* channel : completed) {
            // Phase 1: Capture info (no lock needed — all immutable or atomic)
            // old_qp will not be used after drained
            void* old_qp = channel->GetNativeQp();
            ConnectionGroup* grp = channel->GetGroup();
            AsuEndpoint ep = grp->GetEndpoint();
            std::uint32_t gid = grp->GetGroupId();
            std::uint32_t ch_id = channel->GetChannelId();

            channel->FinishDrain();  // Sets state=FAILED (atomic), native_qp=nullptr
            UC_DEBUG("ConnectionManager::RecoverLoop FinishDrain ch_id={} state=FAILED", ch_id);

            // Phase 2: Perform blocking IO operations OUTSIDE the lock
            if (old_qp && deleteFn_) { deleteFn_({old_qp}); }

            std::vector<ConnectionHandle> new_handles;
            if (createFn_) { new_handles = createFn_(ep, 1); }

            if (new_handles.empty()) {
                UC_DEBUG(
                    "ConnectionManager::RecoverLoop RebuildChannel FAILED, keep in drain_list for "
                    "retry group_id={}",
                    gid);
                continue;
            }

            {
                std::unique_lock<std::shared_mutex> lock(drainMu_);
                drainList_.erase(std::remove(drainList_.begin(), drainList_.end(), channel),
                                 drainList_.end());
            }

            {
                std::unique_lock<std::shared_mutex> lock(structureMu_);
                grp->RemoveChannel(channel);
                auto* new_ch = grp->AddChannel(new_handles[0]);
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
    std::vector<ConnectionChannel*> new_cache;
    for (const auto& g : groups_) {
        for (const auto& channel : g->GetChannels()) {
            if (channel->GetState() == ChannelState::ACTIVE) { new_cache.push_back(channel.get()); }
        }
    }
    channelCache_ = std::move(new_cache);
    cacheDirty_.store(false, std::memory_order_release);
}

// No cache lock is added, so only one thread can call it and it cannot be used concurrently
ConnectionChannel* ConnectionManager::SelectByRoundRobin()
{
    if (cacheDirty_.load(std::memory_order_acquire)) { RebuildChannelCache(); }

    if (channelCache_.empty()) return nullptr;

    auto idx = rrIndex_.fetch_add(1, std::memory_order_relaxed);
    std::size_t total = channelCache_.size();
    std::size_t start = idx % total;

    for (std::size_t i = 0; i < total; ++i) {
        std::size_t pos = (start + i) % total;
        auto* channel = channelCache_[pos];
        if (channel->GetState() == ChannelState::ACTIVE &&
            channel->GetInflightCount() < kMaxInflightPerChannel) {
            channel->IncrementInflight();
            return channel;
        }
    }
    return nullptr;
}

// No cache lock is added, so only one thread can call it and it cannot be used concurrently
ConnectionChannel* ConnectionManager::SelectByLeastLoaded()
{
    if (cacheDirty_.load(std::memory_order_acquire)) { RebuildChannelCache(); }

    if (channelCache_.empty()) return nullptr;

    ConnectionChannel* selected = nullptr;
    std::uint32_t min_inflight = std::numeric_limits<std::uint32_t>::max();

    for (auto* channel : channelCache_) {
        auto inflight = channel->GetInflightCount();
        if (channel->GetState() == ChannelState::ACTIVE && inflight < kMaxInflightPerChannel &&
            inflight < min_inflight) {
            min_inflight = inflight;
            selected = channel;
            if (min_inflight == 0) break;
        }
    }

    if (selected) { selected->IncrementInflight(); }
    return selected;
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

}  // namespace UC::ASU