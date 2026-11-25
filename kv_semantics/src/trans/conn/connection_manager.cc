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
#include <limits>
#include <thread>
#include "connection_internal.h"
#include "logger.h"

namespace kv {
namespace {

bool EndpointsFullyEqual(const NodeEndpoint& a, const NodeEndpoint& b)
{
    return a.ip == b.ip && a.port == b.port && a.protocol == b.protocol &&
           a.numaNode == b.numaNode && a.hcaName == b.hcaName && a.hcaPort == b.hcaPort &&
           a.attrs == b.attrs;
}

bool IsValidUpdateEndpoint(const NodeEndpoint& endpoint)
{
    return !endpoint.ip.empty() && endpoint.port != 0;
}

}  // namespace

ConnectionManager::ConnectionManager(TransProvider& provider, const std::string& localIp,
                                     const TransportConfig& config)
    : routingPolicy_(config.connection.selectionPolicy == SelectionPolicy::LEAST_LOADED
                         ? RoutingPolicy::LEAST_LOADED
                         : RoutingPolicy::ROUND_ROBIN),
      recoverCheckIntervalMs_(config.recovery.recoverCheckIntervalMs),
      provider_(provider),
      localIp_(localIp),
      connectTimeoutMs_(config.recovery.connectTimeoutMs),
      maxErrorCount_(config.connection.maxErrorCount),
      maxInflightPerChannel_(config.flowControl.maxInflightPerChannel),
      connectRetryPerRound_(config.recovery.connectRetryPerRound),
      recoverInitialBackoffMs_(config.recovery.recoverInitialBackoffMs),
      recoverMaxBackoffMs_(config.recovery.recoverMaxBackoffMs),
      taskTimeoutMs_(config.retry.taskTimeoutMs),
      config_(config)
{
}

ConnectionManager::ConnectionManager(TransProvider& provider, const std::string& localIp,
                                     std::uint32_t timeout, std::uint32_t maxErrorCount)
    : provider_(provider),
      localIp_(localIp),
      connectTimeoutMs_(timeout),
      maxErrorCount_(maxErrorCount)
{
}

ConnectionManager::~ConnectionManager() { Shutdown(); }

Status ConnectionManager::CreateGroupConnections(
    const NodeEndpoint& endpoint, std::uint32_t qpNum,
    std::vector<TransProvider::ConnectionHandle>& handles, ServerKvCapabilities& capabilities)
{
    handles.clear();
    capabilities = {};
    // One round = 1 + K attempts. Every attempt uses fresh connection objects;
    // handles from a failed attempt are never reused (provider contract: on
    // failure the handle list is empty, so nothing leaks).
    const auto attempts = 1u + connectRetryPerRound_;
    Status lastStatus = Status::Error(StatusCode::CONNECTION_ERROR, "no create attempt made");
    for (std::uint32_t attempt = 0; attempt < attempts; ++attempt) {
        if (shuttingDown_.load(std::memory_order_acquire)) { break; }
        lastStatus = provider_.CreateConnection(localIp_, endpoint.ip, endpoint.port, qpNum,
                                                 connectTimeoutMs_, handles);
        if (lastStatus.ok() && !handles.empty()) { break; }
        if (lastStatus.ok()) {
            lastStatus = Status::Error(StatusCode::INTERNAL_ERROR,
                                       "provider returned no connection handles");
        }
        handles.clear();
        KV_WARN("ConnectionManager create connection failed endpoint_ip={} port={} "
                "attempt={} code={} message={}",
                endpoint.ip, endpoint.port, attempt + 1, static_cast<int>(lastStatus.code),
                lastStatus.message);
    }
    if (handles.empty()) { return lastStatus; }

    auto capabilityStatus = provider_.GetServerCapabilities(handles.front(), capabilities);
    if (!capabilityStatus.ok() && capabilityStatus.code != StatusCode::UNSUPPORTED) {
        const auto deleteStatuses = provider_.DeleteConnections(handles);
        for (const auto& deleteStatus : deleteStatuses) {
            if (!deleteStatus.ok()) {
                KV_WARN("ConnectionManager::CreateGroupConnections cleanup failed: code={} "
                        "message={}",
                        static_cast<int>(deleteStatus.code), deleteStatus.message);
            }
        }
        handles.clear();
        return capabilityStatus;
    }
    if (capabilityStatus.code == StatusCode::UNSUPPORTED) {
        capabilities = {};
        KV_DEBUG(
            "ConnectionManager::CreateGroupConnections capability query unsupported. Use "
            "default configurations instead");
    }
    return Status::OK();
}

Status ConnectionManager::AddGroup(const NodeEndpoint& endpoint, std::uint32_t qp_num)
{
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "connection manager shutting down");
    }
    if (qp_num == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "qp_num must be greater than zero");
    }
    KV_DEBUG("ConnectionManager::AddGroup endpoint={} qp_num={}", endpoint.ip, qp_num);

    ServerKvCapabilities capabilities;
    std::vector<TransProvider::ConnectionHandle> handles;
    auto createStatus = CreateGroupConnections(endpoint, qp_num, handles, capabilities);
    if (!createStatus.ok()) { return createStatus; }

    if (capabilities.ioQueueDepth != 0) {
        maxInflightPerChannel_ = std::min(maxInflightPerChannel_, capabilities.ioQueueDepth);
    }

    {
        std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
        std::unique_lock<std::shared_mutex> lock(structureMu_);
        auto gid = nextGroupId_.fetch_add(1, std::memory_order_relaxed);
        auto group = std::make_shared<ConnectionGroup>(gid, endpoint, capabilities, qp_num);
        group->BindSelf(group);
        for (auto handle : handles) { group->AddChannel(handle, &provider_); }
        // Group 动态流控：启用时挂接控制器；busyFloor 不小于实际 ioQueueDepth
        // 收紧后的每通道上限则 Init 拒绝（INVALID_ARGUMENT）。
        if (config_.flowControl.flowControlEnabled) {
            std::unique_ptr<GroupFlowController> controller;
            const auto flowStatus = GroupFlowController::Create(
                config_.flowControl, qp_num, capabilities.ioQueueDepth, controller);
            if (!flowStatus.ok()) { return flowStatus; }
            group->SetFlowController(std::move(controller));
        }
        groups_.push_back(std::move(group));

        for (const auto& channel : groups_.back()->GetChannels()) {
            channelCache_.push_back(channel);
        }
    }
    cacheDirty_.store(false, std::memory_order_release);
    KV_INFO("ConnectionManager::AddGroup OK endpoint={} qp_num={} group_id={}", endpoint.ip,
            qp_num, static_cast<int>(groups_.size()) - 1);
    return Status::OK();
}

Status ConnectionManager::Shutdown()
{
    KV_DEBUG("ConnectionManager::Shutdown start");
    shuttingDown_.store(true, std::memory_order_release);
    StopRecoverLoop();

    {
        std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
        channelCache_.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(drainMu_);
        drainList_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(recoverMu_);
        recoverStates_.clear();
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
                     channel->GetChannelId(), channel->GetGroupId(),
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
        excludedGroups.push_back(channel->GetGroupId());
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
                                                 channel->GetGroupId()) != excludedGroups.end();
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
                 selected->GetChannelId(), selected->GetGroupId(), excludedChannels.size());
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
    if (auto group = preferred->LockGroup();
        group && !CanAcquireGroup(*group)) {
        return nullptr;
    }
    preferred->IncrementInflight();
    return preferred;
}

bool ConnectionManager::CanAcquireGroup(const ConnectionGroup& group) const
{
    // Group 动态流控静态准入：actualInflight < currentLimit 才允许，达到即
    // fail-fast；未启用流控的组直接放行。
    const auto* flowController = group.GetFlowController();
    return flowController == nullptr || flowController->CanAcquire(group.GetInflightCount());
}

std::shared_ptr<ConnectionChannel> ConnectionManager::AcquireProbe(
    const std::shared_ptr<ConnectionChannel>& channel)
{
    if (!channel || shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }
    // Recovery window: the whole group (FAILED channels included) is skipped;
    // no new probe and no revival.
    if (auto group = channel->LockGroup(); group && group->InRecoveryWindow()) { return nullptr; }
    const auto state = channel->GetState();
    // Probe whitelist: ACTIVE with zero business inflight, or FAILED;
    // DRAINING is excluded.
    if (state == ChannelState::DRAINING) { return nullptr; }
    if (state == ChannelState::ACTIVE && channel->GetInflightCount() != 0) { return nullptr; }
    if (!channel->TryStartProbe()) { return nullptr; }
    return channel;
}

void ConnectionManager::VisitChannels(
    const std::function<void(const std::shared_ptr<ConnectionChannel>&)>& visitor)
{
    std::vector<std::shared_ptr<ConnectionChannel>> snapshot;
    {
        std::shared_lock<std::shared_mutex> lock(structureMu_);
        for (const auto& group : groups_) {
            for (const auto& channel : group->GetChannels()) { snapshot.push_back(channel); }
        }
    }
    for (const auto& channel : snapshot) { visitor(channel); }
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
             channel->GetChannelId(), channel->GetGroupId(), oldCount + 1, maxErrorCount_);
    if (oldCount + 1 < maxErrorCount_) {
        KV_DEBUG("ConnectionManager::ReportFailure below threshold, skip drain");
        return;
    }

    if (!channel->MarkForDrain()) {
        KV_DEBUG("ConnectionManager::ReportFailure MarkForDrain CAS failed "
                 "(already draining/failed)");
        return;
    }

    KV_INFO("ConnectionManager::ReportFailure drain ch_id={} group_id={} state=DRAINING",
            channel->GetChannelId(), channel->GetGroupId());
    cacheDirty_.store(true, std::memory_order_release);

    // X=1 drain: nothing in flight means the channel reaches FAILED at once.
    if (channel->GetInflightCount() == 0) { channel->MarkFailed(); }

    {
        std::unique_lock<std::shared_mutex> lock(drainMu_);
        for (const auto& existing : drainList_) {
            if (existing.get() == channel.get()) { return; }
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
    KV_DEBUG("ConnectionManager::RecoverLoop started interval_ms={}", recoverCheckIntervalMs_);
    while (!stopRecover_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(recoverCheckIntervalMs_));
        if (stopRecover_.load(std::memory_order_acquire)) { break; }
        RunRecoverySweep();
        SweepDrainList();
    }
    KV_DEBUG("ConnectionManager::RecoverLoop stopped");
}

void ConnectionManager::RunRecoverySweep()
{
    std::vector<std::shared_ptr<ConnectionGroup>> snapshot;
    {
        std::shared_lock<std::shared_mutex> lock(structureMu_);
        snapshot = groups_;
    }

    for (const auto& group : snapshot) {
        const auto gid = group->GetGroupId();
        if (group->HasActiveChannel()) { continue; }
        if (group->GetChannels().empty()) { continue; }

        RecoverState state;
        bool inWindow = false;
        {
            std::lock_guard<std::mutex> lock(recoverMu_);
            auto it = recoverStates_.find(gid);
            if (it != recoverStates_.end()) {
                state = it->second;
                inWindow = true;
            }
        }
        if (!inWindow) {
            // Trigger condition: every channel of the group is non-ACTIVE.
            // Opening the window is one-way until recovery succeeds or the
            // group is removed; probes cannot cancel it.
            std::lock_guard<std::mutex> lock(recoverMu_);
            if (recoverStates_.count(gid) != 0) { continue; }
            RecoverState fresh;
            fresh.backoffMs = recoverInitialBackoffMs_;
            fresh.nextRetryAt = std::chrono::steady_clock::now();
            fresh.windowStartedAt = fresh.nextRetryAt;
            recoverStates_[gid] = fresh;
            group->SetRecoveryWindow(true);
            KV_INFO("ConnectionManager recovery window OPEN group_id={} endpoint_ip={} "
                    "channels={}",
                    gid, group->GetEndpoint().ip, group->GetChannels().size());
            continue;
        }

        const auto succeeded = RecoverGroup(group, state);
        if (succeeded) {
            EraseRecoverState(gid);
            group->SetRecoveryWindow(false);
            continue;
        }
        // Write the round result back (unless the group was removed in
        // between, which also erased the state).
        std::lock_guard<std::mutex> lock(recoverMu_);
        auto it = recoverStates_.find(gid);
        if (it != recoverStates_.end()) { it->second = state; }
    }
}

bool ConnectionManager::RecoverGroup(const std::shared_ptr<ConnectionGroup>& group,
                                     RecoverState& state)
{
    const auto now = std::chrono::steady_clock::now();
    if (now < state.nextRetryAt) { return false; }

    // Bounded wait for the group's inflight to reach zero; after the bound the
    // teardown forces the disconnect anyway (in-flight attempts finish through
    // their own CQE_IO_TIMEOUT adjudication).
    const auto drainUntil =
        state.windowStartedAt +
        std::chrono::milliseconds(std::max<std::uint64_t>(taskTimeoutMs_,
                                                          recoverInitialBackoffMs_));
    if (group->GetInflightCount() > 0 && now < drainUntil) { return false; }

    const auto gid = group->GetGroupId();

    // Whole-group teardown: claim the handles under the structure lock, then
    // disconnect outside every lock.
    std::vector<TransProvider::ConnectionHandle> oldHandles;
    {
        std::shared_lock<std::shared_mutex> lock(structureMu_);
        if (std::find_if(groups_.begin(), groups_.end(),
                         [&group](const std::shared_ptr<ConnectionGroup>& g) {
                             return g.get() == group.get();
                         }) == groups_.end()) {
            EraseRecoverState(gid);
            group->SetRecoveryWindow(false);
            return false;
        }
        for (const auto& channel : group->GetChannels()) {
            oldHandles.push_back(channel->DetachHandle());
        }
    }
    oldHandles.erase(std::remove(oldHandles.begin(), oldHandles.end(), nullptr),
                     oldHandles.end());
    if (!oldHandles.empty()) {
        const auto statuses = provider_.DeleteConnections(oldHandles);
        for (const auto& status : statuses) {
            if (!status.ok()) {
                KV_WARN("ConnectionManager::RecoverGroup old handle disconnect failed code={}",
                        static_cast<int>(status.code));
            }
        }
    }
    KV_INFO("ConnectionManager::RecoverGroup teardown done group_id={} handles={}", gid,
            oldHandles.size());

    // Rebuild round: 1 + K fresh attempts against the same endpoint.
    std::vector<TransProvider::ConnectionHandle> newHandles;
    ServerKvCapabilities capabilities;
    const auto createStatus =
        CreateGroupConnections(group->GetEndpoint(), group->GetQpNum(), newHandles, capabilities);
    if (!createStatus.ok()) {
        // backoff x2 capped at recoverMaxBackoffMs; retry next round.
        state.backoffMs =
            std::min(recoverMaxBackoffMs_, state.backoffMs == 0 ? recoverInitialBackoffMs_
                                                                 : state.backoffMs * 2);
        state.nextRetryAt = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(state.backoffMs);
        KV_WARN("ConnectionManager::RecoverGroup rebuild FAILED group_id={} backoff_ms={} "
                 "code={} message={}",
                gid, state.backoffMs, static_cast<int>(createStatus.code),
                createStatus.message);
        return false;
    }

    // Publish: re-verify membership under structureMu_ before swapping the
    // channel set; a group removed by UpdateEndpoints is never resurrected.
    bool published = false;
    {
        std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
        std::unique_lock<std::shared_mutex> lock(structureMu_);
        const bool isMember = std::find_if(groups_.begin(), groups_.end(),
                                           [&group](const std::shared_ptr<ConnectionGroup>& g) {
                                               return g.get() == group.get();
                                           }) != groups_.end();
        if (isMember) {
            group->ClearChannels();
            for (auto handle : newHandles) { group->AddChannel(handle, &provider_); }
            cacheDirty_.store(true, std::memory_order_release);
            published = true;
        }
    }
    if (published) {
        KV_INFO("ConnectionManager::RecoverGroup rebuilt group_id={} new_channels={} "
                "backoff_reset_ms={}",
                gid, newHandles.size(), recoverInitialBackoffMs_);
        return true;
    }

    // Discard the rebuilt result (group was摘表-removed meanwhile).
    provider_.DeleteConnections(newHandles);
    KV_WARN("ConnectionManager::RecoverGroup group_id={} removed during recovery; rebuilt "
            "handles discarded",
            gid);
    EraseRecoverState(gid);
    group->SetRecoveryWindow(false);
    return false;
}

void ConnectionManager::SweepDrainList()
{
    std::vector<std::shared_ptr<ConnectionChannel>> pending;
    {
        std::unique_lock<std::shared_mutex> lock(drainMu_);
        pending.swap(drainList_);
    }
    for (const auto& channel : pending) {
        const auto group = channel->LockGroup();
        if (!group) { continue; }  // group gone: drop the work item
        if (channel->GetState() == ChannelState::DRAINING && channel->GetInflightCount() == 0) {
            channel->MarkFailed();
        }
        if (channel->GetState() == ChannelState::DRAINING && channel->GetInflightCount() > 0) {
            std::unique_lock<std::shared_mutex> lock(drainMu_);
            drainList_.push_back(channel);
        }
    }
}

void ConnectionManager::EraseRecoverState(std::uint32_t groupId)
{
    std::lock_guard<std::mutex> lock(recoverMu_);
    recoverStates_.erase(groupId);
}

bool ConnectionManager::HasRecoverState(std::uint32_t groupId) const
{
    std::lock_guard<std::mutex> lock(recoverMu_);
    return recoverStates_.count(groupId) != 0;
}

void ConnectionManager::GetHealthSnapshot(NodeHealthSnapshot& snapshot)
{
    snapshot.health = NodeHealth::RED;
    snapshot.updating = updating_.load(std::memory_order_acquire);

    if (shuttingDown_.load(std::memory_order_acquire)) { return; }

    bool hasChannel = false;
    bool anyActive = false;
    bool allActive = true;
    {
        std::shared_lock<std::shared_mutex> lock(structureMu_);
        for (const auto& group : groups_) {
            for (const auto& channel : group->GetChannels()) {
                hasChannel = true;
                if (channel->GetState() == ChannelState::ACTIVE) {
                    anyActive = true;
                } else {
                    allActive = false;
                }
            }
        }
    }

    // Read-only derivation: channel states aggregate per group first, then
    // across groups; no GlobalView mutation is driven from here.
    if (!hasChannel) {
        snapshot.health = NodeHealth::RED;
    } else if (allActive) {
        snapshot.health = NodeHealth::GREEN;
    } else if (anyActive) {
        snapshot.health = NodeHealth::YELLOW;
    } else {
        snapshot.health = NodeHealth::RED;
    }
}

std::shared_ptr<ConnectionGroup> ConnectionManager::FindGroupByEndpointLocked(
    const NodeEndpoint& endpoint)
{
    for (const auto& group : groups_) {
        const auto& ep = group->GetEndpoint();
        if (ep.ip == endpoint.ip && ep.port == endpoint.port) { return group; }
    }
    return nullptr;
}

Status ConnectionManager::RemoveGroup(const NodeEndpoint& endpoint, bool& removed)
{
    removed = false;
    std::shared_ptr<ConnectionGroup> victim;
    {
        std::unique_lock<std::shared_mutex> lock(structureMu_);
        victim = FindGroupByEndpointLocked(endpoint);
        if (!victim) { return Status::OK(); }  // no match: skip, idempotent
        // 摘表 first: leave groups_ traversal before disconnecting so the
        // recovery loop cannot re-publish into a removed group.
        groups_.erase(std::find_if(groups_.begin(), groups_.end(),
                                   [&victim](const std::shared_ptr<ConnectionGroup>& g) {
                                       return g.get() == victim.get();
                                   }));
        cacheDirty_.store(true, std::memory_order_release);
    }

    // Clean drainList_ entries of this group; structureMu_ and drainMu_ are
    // never nested.
    {
        std::unique_lock<std::shared_mutex> lock(drainMu_);
        drainList_.erase(
            std::remove_if(drainList_.begin(), drainList_.end(),
                           [&victim](const std::shared_ptr<ConnectionChannel>& channel) {
                               const auto group = channel->LockGroup();
                               return !group || group.get() == victim.get();
                           }),
            drainList_.end());
    }

    EraseRecoverState(victim->GetGroupId());
    victim->SetRecoveryWindow(false);

    // The摘表者 exclusively claims the whole-group disconnect; DetachHandle is
    // idempotent so a concurrent recovery round cannot double-delete.
    std::vector<TransProvider::ConnectionHandle> handles;
    for (const auto& channel : victim->GetChannels()) {
        handles.push_back(channel->DetachHandle());
    }
    handles.erase(std::remove(handles.begin(), handles.end(), nullptr), handles.end());
    if (!handles.empty()) {
        const auto statuses = provider_.DeleteConnections(handles);
        for (const auto& status : statuses) {
            if (!status.ok()) {
                KV_WARN("ConnectionManager::RemoveGroup disconnect failed endpoint_ip={} "
                        "code={} message={}",
                        endpoint.ip, static_cast<int>(status.code), status.message);
            }
        }
    }
    removed = true;
    KV_INFO("ConnectionManager::RemoveGroup removed endpoint_ip={} port={} group_id={} "
            "handles={}",
            endpoint.ip, endpoint.port, victim->GetGroupId(), handles.size());
    return Status::OK();
}

Status ConnectionManager::UpdateGroups(const std::vector<NodeEndpoint>& toAdd,
                                       const std::vector<NodeEndpoint>& toRemove,
                                       UpdateResult& result)
{
    result = UpdateResult{};

    // Fail-fast atomic rejection: nothing is modified when the patch is empty
    // or any endpoint is invalid.
    if (toAdd.empty() && toRemove.empty()) {
        result.status = Status::Error(StatusCode::INVALID_ARGUMENT,
                                      "UpdateEndpoints requires a non-empty patch");
        return result.status;
    }
    for (const auto& endpoint : toAdd) {
        if (!IsValidUpdateEndpoint(endpoint)) {
            result.status =
                Status::Error(StatusCode::INVALID_ARGUMENT, "invalid endpoint in toAdd");
            return result.status;
        }
    }
    for (const auto& endpoint : toRemove) {
        if (!IsValidUpdateEndpoint(endpoint)) {
            result.status =
                Status::Error(StatusCode::INVALID_ARGUMENT, "invalid endpoint in toRemove");
            return result.status;
        }
    }
    if (shuttingDown_.load(std::memory_order_acquire)) {
        result.status = Status::Error(StatusCode::NOT_INITIALIZED, "connection manager is down");
        return result.status;
    }

    std::lock_guard<std::mutex> updateLock(endpointUpdateMu_);
    updating_.store(true, std::memory_order_release);
    result.updating = true;
    KV_INFO("ConnectionManager::UpdateGroups window OPEN to_add={} to_remove={}", toAdd.size(),
            toRemove.size());

    // Phase 1: remove first (先删后加).
    for (const auto& endpoint : toRemove) {
        bool removed = false;
        const auto status = RemoveGroup(endpoint, removed);
        EndpointUpdateOutcome outcome;
        outcome.endpoint = endpoint;
        outcome.action = removed ? "removed" : "skipped";
        outcome.status = status;
        result.removeResults.push_back(outcome);
        KV_INFO("ConnectionManager::UpdateGroups remove endpoint_ip={} port={} action={}",
                endpoint.ip, endpoint.port, outcome.action);
    }

    // Phase 2: add on the post-removal group set, matched by ip:port.
    for (const auto& endpoint : toAdd) {
        std::shared_ptr<ConnectionGroup> existing;
        {
            std::shared_lock<std::shared_mutex> lock(structureMu_);
            existing = FindGroupByEndpointLocked(endpoint);
        }
        const std::string action = existing ? "replaced" : "created";
        if (existing) {
            if (EndpointsFullyEqual(existing->GetEndpoint(), endpoint)) {
                // Fully equal: skip and keep state/backoff/flow-control.
                EndpointUpdateOutcome outcome;
                outcome.endpoint = endpoint;
                outcome.action = "skipped";
                result.addResults.push_back(outcome);
                KV_INFO("ConnectionManager::UpdateGroups add endpoint_ip={} port={} "
                        "action=skipped",
                        endpoint.ip, endpoint.port);
                continue;
            }
            // Attribute-different replacement: old group walks the remove path
            // first so the endpoint never lives in two groups at once.
            bool removed = false;
            (void)RemoveGroup(endpoint, removed);
        }

        ServerKvCapabilities capabilities;
        std::vector<TransProvider::ConnectionHandle> handles;
        auto status = CreateGroupConnections(endpoint, config_.qpNum, handles, capabilities);
        if (!status.ok()) {
            // Best-effort: no rollback of already-successful entries.
            EndpointUpdateOutcome outcome;
            outcome.endpoint = endpoint;
            outcome.action = "failed";
            outcome.status = status;
            result.addResults.push_back(outcome);
            result.failedEndpoints.push_back(endpoint);
            KV_WARN("ConnectionManager::UpdateGroups add FAILED endpoint_ip={} port={} code={}",
                    endpoint.ip, endpoint.port, static_cast<int>(status.code));
            continue;
        }

        {
            std::lock_guard<std::mutex> cacheLock(channelCacheMu_);
            std::unique_lock<std::shared_mutex> lock(structureMu_);
            auto shadow = FindGroupByEndpointLocked(endpoint);
            if (shadow != nullptr) {
                status = Status::Error(StatusCode::INTERNAL_ERROR,
                                       "endpoint appeared concurrently during update");
            } else {
                auto gid = nextGroupId_.fetch_add(1, std::memory_order_relaxed);
                auto group = std::make_shared<ConnectionGroup>(gid, endpoint, capabilities,
                                                              config_.qpNum);
                group->BindSelf(group);
                for (auto handle : handles) { group->AddChannel(handle, &provider_); }
                // 动态 Endpoint 新建组同样挂接流控（best-effort：非法关系按失败计）。
                if (config_.flowControl.flowControlEnabled) {
                    std::unique_ptr<GroupFlowController> controller;
                    const auto flowStatus = GroupFlowController::Create(
                        config_.flowControl, config_.qpNum, capabilities.ioQueueDepth,
                        controller);
                    if (!flowStatus.ok()) {
                        status = flowStatus;
                    } else {
                        group->SetFlowController(std::move(controller));
                    }
                }
                if (status.ok()) {
                    groups_.push_back(std::move(group));
                    cacheDirty_.store(true, std::memory_order_release);
                }
            }
        }
        if (!status.ok()) {
            const auto statuses = provider_.DeleteConnections(handles);
            (void)statuses;
            EndpointUpdateOutcome outcome;
            outcome.endpoint = endpoint;
            outcome.action = "failed";
            outcome.status = status;
            result.addResults.push_back(outcome);
            result.failedEndpoints.push_back(endpoint);
            continue;
        }
        EndpointUpdateOutcome outcome;
        outcome.endpoint = endpoint;
        outcome.action = action;
        result.addResults.push_back(outcome);
        KV_INFO("ConnectionManager::UpdateGroups add endpoint_ip={} port={} action={} "
                "qp_num={}",
                endpoint.ip, endpoint.port, action, config_.qpNum);
    }

    updating_.store(false, std::memory_order_release);
    result.updating = false;
    KV_INFO("ConnectionManager::UpdateGroups window CLOSED failed={}",
            result.failedEndpoints.size());
    return Status::OK();
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

    // 候选分组准入（Group 动态流控：达到 currentLimit 即 fail-fast）。
    const auto candidateAdmissible = [this](const std::shared_ptr<ConnectionChannel>& channel) {
        const auto group = channel->LockGroup();
        return group == nullptr || CanAcquireGroup(*group);
    };

    if (routingPolicy_ == RoutingPolicy::LEAST_LOADED) {
        std::shared_ptr<ConnectionChannel> selected;
        std::uint32_t minInflight = std::numeric_limits<std::uint32_t>::max();
        for (const auto& channel : candidates) {
            if (channel->GetState() != ChannelState::ACTIVE) { continue; }
            const auto inflight = channel->GetInflightCount();
            if (inflight < maxInflightPerChannel_ && inflight < minInflight &&
                candidateAdmissible(channel)) {
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
        if (channel->GetInflightCount() < maxInflightPerChannel_ && candidateAdmissible(channel)) {
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
