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
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "kv_transport.h"
#include "kv_types.h"
#include "proto/kv_protocol.h"
#include "trans_provider.h"

namespace kv {

using ConnectionHandle = void*;

enum class RoutingPolicy : std::uint8_t {
    ROUND_ROBIN,
    LEAST_LOADED,
};

class ConnectionChannel;
class ConnectionGroup;
struct ScatterGatherEntry;

// One queued dynamic-endpoint patch with its completion contract. The
// recover worker consumes at most one patch per tick and at most one add
// endpoint per tick (bounded head-of-line), so a patch with A adds settles
// over successive ticks; the future completes once every endpoint has a
// terminal outcome. Patches are strictly FIFO - no coalescing.
struct PendingPatch {
    std::vector<NodeEndpoint> toAdd;
    std::vector<NodeEndpoint> toRemove;
    std::promise<UpdateResult> completion;
    std::size_t nextRemoveIndex{0};
    std::size_t nextAddIndex{0};
    std::vector<EndpointUpdateOutcome> removeResults;
    std::vector<EndpointUpdateOutcome> addResults;
    std::vector<NodeEndpoint> failedEndpoints;
};

// Backoff bookkeeping for one endpoint-level recovery window. Existence of a
// RecoverState means the group is inside the recovery window.
struct RecoverState {
    std::uint32_t backoffMs{100};
    std::chrono::steady_clock::time_point nextRetryAt{};
    std::chrono::steady_clock::time_point windowStartedAt{};
};

class ConnectionManager {
public:
    // Config-driven constructor used by AsuTransportImpl.
    ConnectionManager(TransProvider& provider, const std::string& localIp,
                      const TransportConfig& config);
    // Legacy constructor kept for direct users/tests; X defaults to 1.
    ConnectionManager(TransProvider& provider, const std::string& localIp,
                      std::uint32_t timeout = 5000, std::uint32_t maxErrorCount = 1);
    ~ConnectionManager();

    Status AddGroup(const NodeEndpoint& endpoint, std::uint32_t qp_num);
    Status Shutdown();

    std::shared_ptr<ConnectionChannel> SelectConnection();

    std::shared_ptr<ConnectionChannel> GetActiveConnection();
    void SetRoutingPolicy(RoutingPolicy policy);
    void ReportFailure(const std::shared_ptr<ConnectionChannel>& channel);
    void ReportSuccess(const std::shared_ptr<ConnectionChannel>& channel);

    void StartRecoverLoop();
    void StopRecoverLoop();

    // Three-state snapshot derived from channel states; updating is set while
    // UpdateGroups is mutating the configuration window.
    void GetHealthSnapshot(NodeHealthSnapshot& snapshot);

    // Incremental dynamic endpoint update, synchronous wrapper: submits the
    // patch and waits for the worker (or runs inline when the recover loop is
    // not running). Existing callers and tests keep synchronous semantics.
    Status UpdateGroups(const std::vector<NodeEndpoint>& toAdd,
                        const std::vector<NodeEndpoint>& toRemove, UpdateResult& result);

    // Asynchronous external contract: fail-fast validation rejects empty or
    // invalid patches synchronously with a ready INVALID_ARGUMENT future.
    // When the recover loop runs, the patch is enqueued (strict FIFO) and the
    // future completes after the worker has settled every endpoint (at most
    // one add per tick, single 1+K round per endpoint; failures land in
    // failedEndpoints for caller re-push). When the loop is not running the
    // patch executes inline and the returned future is already complete.
    // Shutdown/stopping completes pending futures with CANCELED.
    // Note: never call the synchronous wrapper while holding any
    // ConnectionManager lock (submit-and-wait would deadlock).
    std::future<UpdateResult> UpdateGroupsAsync(const std::vector<NodeEndpoint>& toAdd,
                                                const std::vector<NodeEndpoint>& toRemove);

    // True while the group is inside an endpoint-level recovery window.
    bool HasRecoverState(std::uint32_t groupId) const;

    std::int64_t TotalInflightCount();
    std::vector<ServerKvCapabilities> GetServerCapabilities();

private:
    std::vector<std::shared_ptr<ConnectionGroup>> groups_;
    std::shared_mutex structureMu_;  // shared_mutex allows concurrent reads

    std::vector<std::shared_ptr<ConnectionChannel>> channelCache_;
    std::mutex channelCacheMu_;
    std::atomic<bool> cacheDirty_{false};

    std::atomic<bool> shuttingDown_{false};

    std::atomic<std::uint32_t> rrIndex_{0};
    RoutingPolicy routingPolicy_{RoutingPolicy::ROUND_ROBIN};
    std::uint32_t recoverCheckIntervalMs_{100};

    std::thread recoverWorker_;
    std::atomic<bool> stopRecover_{false};

    std::shared_mutex drainMu_;
    std::vector<std::shared_ptr<ConnectionChannel>> drainList_;

    // Serialized inline fallback path (recover loop not running); guards the
    // config window for the inline execution.
    std::mutex endpointUpdateMu_;
    std::atomic<bool> updating_{false};

    // Pending update patches consumed by the recover worker (strict FIFO,
    // at most one patch active at a time; patchMu_ is a leaf lock).
    std::mutex patchMu_;
    std::condition_variable patchCv_;
    std::deque<std::shared_ptr<PendingPatch>> patchQueue_;
    std::shared_ptr<PendingPatch> activePatch_;
    std::atomic<bool> recoverRunning_{false};

    // gid -> recovery window state; guarded by recoverMu_.
    mutable std::mutex recoverMu_;
    std::unordered_map<std::uint32_t, RecoverState> recoverStates_;
    std::atomic<std::uint32_t> nextGroupId_{0};

    TransProvider& provider_;
    std::string localIp_;
    std::uint32_t connectTimeoutMs_{5000};
    std::uint32_t maxErrorCount_{1};
    std::uint32_t maxInflightPerChannel_{256};
    std::uint32_t connectRetryPerRound_{2};
    std::uint32_t recoverInitialBackoffMs_{100};
    std::uint32_t recoverMaxBackoffMs_{5000};
    std::uint64_t taskTimeoutMs_{100};
    TransportConfig config_;

    void RecoverLoop();
    void RunRecoverySweep();
    // Dynamic-endpoint patch processing on the worker: pops/continues the
    // active patch, applies all removals in one tick and at most one add
    // endpoint per tick (single 1+K round; failures go to failedEndpoints
    // immediately - retry rounds are a future iteration).
    void ProcessPendingUpdateTick();
    // One processing tick for a single patch; returns true when the patch is
    // settled and its future has been completed.
    bool ProcessPatchTick(const std::shared_ptr<PendingPatch>& patch);
    // Applies one add endpoint (skip/replace/create with membership-checked
    // publish) and records the outcome into the patch.
    void ApplySingleAdd(const NodeEndpoint& endpoint, PendingPatch& patch);
    // Completes the patch future with the accumulated UpdateResult and clears
    // the updating window when no work remains.
    void FinalizePatch(const std::shared_ptr<PendingPatch>& patch);
    // Inline synchronous execution when the recover loop is not running.
    std::future<UpdateResult> ExecutePatchInline(const std::vector<NodeEndpoint>& toAdd,
                                                 const std::vector<NodeEndpoint>& toRemove);
    // StopRecoverLoop tail: completes every un-settled patch future with a
    // CANCELED result so callers never hang on get().
    void CancelRemainingPatches();
    bool HasPendingUpdateWorkLocked() const;
    // One recovery round for a group inside the window: bounded drain, whole
    // group teardown, 1+K rebuild, membership-checked publish. Returns true
    // when the round succeeded and published fresh channels.
    bool RecoverGroup(const std::shared_ptr<ConnectionGroup>& group, RecoverState& state);
    void SweepDrainList();
    void RebuildChannelCache();
    std::shared_ptr<ConnectionChannel> SelectByPolicyLocked(
        const std::vector<std::shared_ptr<ConnectionChannel>>& candidates);
    std::shared_ptr<ConnectionChannel> SelectByRoundRobin();
    std::shared_ptr<ConnectionChannel> SelectByLeastLoaded();
    std::shared_ptr<ConnectionGroup> FindGroupByEndpointLocked(const NodeEndpoint& endpoint);
    Status CreateGroupConnections(const NodeEndpoint& endpoint, std::uint32_t qpNum,
                                   std::vector<TransProvider::ConnectionHandle>& handles,
                                   ServerKvCapabilities& capabilities);
    Status RemoveGroup(const NodeEndpoint& endpoint, bool& removed);
    void EraseRecoverState(std::uint32_t groupId);
};

}  // namespace kv
