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
#include "runtime/device.h"

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
      maxInflightPerChannel_(256),  // static per-channel admission (ioQueueDepth-tightened)
      connectRetryPerRound_(config.recovery.connectRetryPerRound),
      recoverInitialBackoffMs_(config.recovery.recoverInitialBackoffMs),
      recoverMaxBackoffMs_(config.recovery.recoverMaxBackoffMs),
      taskTimeoutMs_(config.timeoutMs),
      qpNum_(config.qpNum),
      deviceId_(config.deviceId)
{
}

ConnectionManager::ConnectionManager(TransProvider& provider, const std::string& localIp,
                                     std::uint32_t timeout, std::uint32_t maxErrorCount,
                                     std::int32_t deviceId)
    : provider_(provider),
      localIp_(localIp),
      connectTimeoutMs_(timeout),
      maxErrorCount_(maxErrorCount),
      deviceId_(deviceId)
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
        for (auto handle : handles) { group->AddChannel(handle, &provider_); }
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

std::shared_ptr<ConnectionChannel> ConnectionManager::SelectConnection()
{
    if (shuttingDown_.load(std::memory_order_acquire)) { return nullptr; }
    auto policy = routingPolicy_;
    auto channel =
        policy == RoutingPolicy::LEAST_LOADED ? SelectByLeastLoaded() : SelectByRoundRobin();
    if (channel) {
        KV_DEBUG("ConnectionManager::SelectConnection policy={} ch_id={} group_id={} inflight={}",
                 policy == RoutingPolicy::LEAST_LOADED ? "LEAST_LOADED" : "ROUND_ROBIN",
                 channel->GetChannelId(), channel->GetGroup()->GetGroupId(),
                 channel->GetInflightCount());
    } else {
        KV_DEBUG("ConnectionManager::SelectConnection policy={} NO available channel",
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
    // Enqueue is allowed only while the worker is alive; set the flag before
    // spawn so no submission is lost in between.
    recoverRunning_.store(true, std::memory_order_release);
    recoverWorker_ = std::thread(&ConnectionManager::RecoverLoop, this);
}

void ConnectionManager::StopRecoverLoop()
{
    KV_DEBUG("ConnectionManager::StopRecoverLoop start");
    // While stopping, submissions are rejected (never executed inline
    // concurrently with the still-running worker); after the join they fall
    // back to inline execution.
    stopRecover_.store(true, std::memory_order_release);
    patchCv_.notify_all();
    if (recoverWorker_.joinable()) { recoverWorker_.join(); }
    recoverRunning_.store(false, std::memory_order_release);
}

void ConnectionManager::RecoverLoop()
{
    // Channel rebuilding runs on this thread and calls back into the provider,
    // so bind the provider device before the first CreateConnection. Otherwise a
    // provider whose API needs the caller's device/context fails here.
    if (deviceId_ >= 0) {
        runtime::Device device;
        const auto setupStatus = device.Setup(deviceId_);
        if (!setupStatus.ok()) {
            KV_WARN(
                "ConnectionManager::RecoverLoop device binding failed: device_id={} code={} "
                "message={}",
                deviceId_, static_cast<int>(setupStatus.code), setupStatus.message);
        }
    } else {
        KV_DEBUG(
            "ConnectionManager::RecoverLoop no device id configured; provider calls run on the "
            "current ACL device");
    }

    KV_DEBUG("ConnectionManager::RecoverLoop started interval_ms={}", recoverCheckIntervalMs_);
    while (!stopRecover_.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lock(patchMu_);
            if (!HasPendingUpdateWorkLocked()) {
                // Sleep becomes an interruptible wait: a pending patch wakes
                // the worker immediately, otherwise it ticks at
                // recoverCheckIntervalMs.
                patchCv_.wait_for(lock,
                                  std::chrono::milliseconds(recoverCheckIntervalMs_),
                                  [this] {
                                      return stopRecover_.load(std::memory_order_acquire) ||
                                             HasPendingUpdateWorkLocked();
                                  });
            }
        }
        if (stopRecover_.load(std::memory_order_acquire)) { break; }
        // Dynamic-endpoint patches first (bounded: at most one add per tick),
        // then the recovery sweep keeps its per-tick cadence.
        ProcessPendingUpdateTick();
        RunRecoverySweep();
        SweepDrainList();
    }
    // Un-settled patches (active or queued) are canceled so no future hangs.
    CancelRemainingPatches();
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
            KV_INFO("ConnectionManager recovery window OPEN group_id={} endpoint_ip={} "
                    "channels={}",
                    gid, group->GetEndpoint().ip, group->GetChannels().size());
            continue;
        }

        const auto succeeded = RecoverGroup(group, state);
        if (succeeded) {
            EraseRecoverState(gid);
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
                               return channel->GetGroup() == victim.get();
                           }),
            drainList_.end());
    }

    EraseRecoverState(victim->GetGroupId());

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
    // Synchronous wrapper: submit-and-wait (worker path) or inline execution
    // when the recover loop is not running. Never call while holding any
    // ConnectionManager lock - the wait would deadlock.
    auto future = UpdateGroupsAsync(toAdd, toRemove);
    result = future.get();
    return result.status;
}

namespace {

// Helper: a ready future carrying just a status.
std::future<UpdateResult> ReadyUpdateFuture(Status status)
{
    std::promise<UpdateResult> completion;
    UpdateResult result;
    result.status = status;
    completion.set_value(std::move(result));
    return completion.get_future();
}

}  // namespace

std::future<UpdateResult> ConnectionManager::UpdateGroupsAsync(
    const std::vector<NodeEndpoint>& toAdd, const std::vector<NodeEndpoint>& toRemove)
{
    // Fail-fast atomic rejection: nothing is modified (and nothing enqueued)
    // when the patch is empty or any endpoint is invalid.
    if (toAdd.empty() && toRemove.empty()) {
        return ReadyUpdateFuture(
            Status::Error(StatusCode::INVALID_ARGUMENT,
                          "UpdateEndpoints requires a non-empty patch"));
    }
    for (const auto& endpoint : toAdd) {
        if (!IsValidUpdateEndpoint(endpoint)) {
            return ReadyUpdateFuture(
                Status::Error(StatusCode::INVALID_ARGUMENT, "invalid endpoint in toAdd"));
        }
    }
    for (const auto& endpoint : toRemove) {
        if (!IsValidUpdateEndpoint(endpoint)) {
            return ReadyUpdateFuture(
                Status::Error(StatusCode::INVALID_ARGUMENT, "invalid endpoint in toRemove"));
        }
    }
    if (shuttingDown_.load(std::memory_order_acquire)) {
        return ReadyUpdateFuture(
            Status::Error(StatusCode::NOT_INITIALIZED, "connection manager is down"));
    }

    // Path decision under patchMu_: enqueue only while the worker will live
    // long enough to consume the patch.
    bool enqueue = false;
    auto patch = std::make_shared<PendingPatch>();
    {
        std::lock_guard<std::mutex> lock(patchMu_);
        const bool running = recoverRunning_.load(std::memory_order_acquire);
        if (running && !stopRecover_.load(std::memory_order_acquire)) {
            patch->toAdd = toAdd;
            patch->toRemove = toRemove;
            patchQueue_.push_back(patch);
            updating_.store(true, std::memory_order_release);
            enqueue = true;
        } else if (running) {
            // Transient stop window: reject instead of racing the worker.
            return ReadyUpdateFuture(
                Status::Error(StatusCode::NOT_INITIALIZED,
                              "recover worker is stopping; retry after stop completes"));
        }
    }
    if (enqueue) {
        KV_INFO("ConnectionManager::UpdateGroupsAsync enqueued to_add={} to_remove={} "
                "queued={}",
                toAdd.size(), toRemove.size(), patchQueue_.size());
        patchCv_.notify_one();
        return patch->completion.get_future();
    }
    return ExecutePatchInline(toAdd, toRemove);
}

std::future<UpdateResult> ConnectionManager::ExecutePatchInline(
    const std::vector<NodeEndpoint>& toAdd, const std::vector<NodeEndpoint>& toRemove)
{
    auto patch = std::make_shared<PendingPatch>();
    patch->toAdd = toAdd;
    patch->toRemove = toRemove;
    auto future = patch->completion.get_future();

    std::lock_guard<std::mutex> updateLock(endpointUpdateMu_);
    updating_.store(true, std::memory_order_release);
    KV_INFO("ConnectionManager::UpdateGroups window OPEN to_add={} to_remove={}", toAdd.size(),
            toRemove.size());
    while (!ProcessPatchTick(patch)) {
        // Inline runs ticks back to back: all removals, then one add per tick,
        // reproducing the original synchronous remove-then-add semantics.
    }
    return future;
}

bool ConnectionManager::HasPendingUpdateWorkLocked() const
{
    return activePatch_ != nullptr || !patchQueue_.empty();
}

void ConnectionManager::ProcessPendingUpdateTick()
{
    std::shared_ptr<PendingPatch> patch;
    {
        std::lock_guard<std::mutex> lock(patchMu_);
        if (activePatch_ == nullptr) {
            if (patchQueue_.empty()) { return; }
            activePatch_ = patchQueue_.front();
            patchQueue_.pop_front();
        }
        patch = activePatch_;
    }
    // Provider calls stay outside patchMu_.
    (void)ProcessPatchTick(patch);
}

bool ConnectionManager::ProcessPatchTick(const std::shared_ptr<PendingPatch>& patch)
{
    // Phase 1: removals are lock/teardown work only (no connection setup) -
    // the whole removal phase settles within one tick.
    while (patch->nextRemoveIndex < patch->toRemove.size()) {
        const auto& endpoint = patch->toRemove[patch->nextRemoveIndex++];
        bool removed = false;
        const auto status = RemoveGroup(endpoint, removed);
        EndpointUpdateOutcome outcome;
        outcome.endpoint = endpoint;
        outcome.action = removed ? "removed" : "skipped";
        outcome.status = status;
        patch->removeResults.push_back(outcome);
        KV_INFO("ConnectionManager::UpdateGroups remove endpoint_ip={} port={} action={}",
                endpoint.ip, endpoint.port, outcome.action);
    }

    // Phase 2: at most ONE add endpoint per tick (bounded head-of-line so the
    // recovery sweep in the same tick is never starved). Single 1+K round;
    // failures land in failedEndpoints for caller re-push (retry rounds are a
    // future iteration).
    if (patch->nextAddIndex < patch->toAdd.size()) {
        ApplySingleAdd(patch->toAdd[patch->nextAddIndex++], *patch);
    }

    if (patch->nextAddIndex < patch->toAdd.size()) { return false; }
    FinalizePatch(patch);
    return true;
}

void ConnectionManager::ApplySingleAdd(const NodeEndpoint& endpoint, PendingPatch& patch)
{
    std::shared_ptr<ConnectionGroup> existing;
    {
        std::shared_lock<std::shared_mutex> lock(structureMu_);
        existing = FindGroupByEndpointLocked(endpoint);
    }
    const std::string action = existing ? "replaced" : "created";
    if (existing) {
        if (EndpointsFullyEqual(existing->GetEndpoint(), endpoint)) {
            // Fully equal: skip and keep state/backoff.
            EndpointUpdateOutcome outcome;
            outcome.endpoint = endpoint;
            outcome.action = "skipped";
            patch.addResults.push_back(outcome);
            KV_INFO("ConnectionManager::UpdateGroups add endpoint_ip={} port={} action=skipped",
                    endpoint.ip, endpoint.port);
            return;
        }
        // Attribute-different replacement: the old group walks the remove path
        // first so the endpoint never lives in two groups at once.
        bool removed = false;
        (void)RemoveGroup(endpoint, removed);
    }

    ServerKvCapabilities capabilities;
    std::vector<TransProvider::ConnectionHandle> handles;
    auto status = CreateGroupConnections(endpoint, qpNum_, handles, capabilities);
    if (!status.ok()) {
        // Best-effort: no rollback of already-successful entries.
        EndpointUpdateOutcome outcome;
        outcome.endpoint = endpoint;
        outcome.action = "failed";
        outcome.status = status;
        patch.addResults.push_back(outcome);
        patch.failedEndpoints.push_back(endpoint);
        KV_WARN("ConnectionManager::UpdateGroups add FAILED endpoint_ip={} port={} code={}",
                endpoint.ip, endpoint.port, static_cast<int>(status.code));
        return;
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
            auto group =
                std::make_shared<ConnectionGroup>(gid, endpoint, capabilities, qpNum_);
            for (auto handle : handles) { group->AddChannel(handle, &provider_); }
            groups_.push_back(std::move(group));
            cacheDirty_.store(true, std::memory_order_release);
        }
    }
    if (!status.ok()) {
        const auto statuses = provider_.DeleteConnections(handles);
        (void)statuses;
        EndpointUpdateOutcome outcome;
        outcome.endpoint = endpoint;
        outcome.action = "failed";
        outcome.status = status;
        patch.addResults.push_back(outcome);
        patch.failedEndpoints.push_back(endpoint);
        return;
    }
    EndpointUpdateOutcome outcome;
    outcome.endpoint = endpoint;
    outcome.action = action;
    patch.addResults.push_back(outcome);
    KV_INFO("ConnectionManager::UpdateGroups add endpoint_ip={} port={} action={} qp_num={}",
            endpoint.ip, endpoint.port, action, qpNum_);
}

void ConnectionManager::FinalizePatch(const std::shared_ptr<PendingPatch>& patch)
{
    const auto failedCount = patch->failedEndpoints.size();
    UpdateResult result;
    result.status = Status::OK();
    result.removeResults = std::move(patch->removeResults);
    result.addResults = std::move(patch->addResults);
    result.failedEndpoints = std::move(patch->failedEndpoints);
    result.updating = false;
    KV_INFO("ConnectionManager::UpdateGroups window CLOSED failed={}", failedCount);
    patch->completion.set_value(std::move(result));

    std::lock_guard<std::mutex> lock(patchMu_);
    if (activePatch_ == patch) { activePatch_.reset(); }
    if (patchQueue_.empty()) { updating_.store(false, std::memory_order_release); }
}

void ConnectionManager::CancelRemainingPatches()
{
    std::vector<std::shared_ptr<PendingPatch>> victims;
    {
        std::lock_guard<std::mutex> lock(patchMu_);
        if (activePatch_ != nullptr) {
            victims.push_back(activePatch_);
            activePatch_.reset();
        }
        while (!patchQueue_.empty()) {
            victims.push_back(patchQueue_.front());
            patchQueue_.pop_front();
        }
        if (!victims.empty()) { updating_.store(false, std::memory_order_release); }
    }
    for (auto& patch : victims) {
        KV_INFO("ConnectionManager::CancelRemainingPatches canceled patch to_add={} "
                "to_remove={}",
                patch->toAdd.size(), patch->toRemove.size());
        UpdateResult canceled;
        canceled.status =
            Status::Error(StatusCode::CANCELED, "endpoint update canceled during shutdown");
        patch->completion.set_value(std::move(canceled));
    }
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
