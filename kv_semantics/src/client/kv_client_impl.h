/**
 * MIT License
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "kv_client.h"
#include "kv_types.h"
#include "spsc_ring_queue.h"
#include "task/task_manager.h"
#include "transport/asu_transport_pool.h"
#include "view/refresh_scheduler.h"
#include "view/view_server.h"
#include "view/view_universe.h"

namespace kv {
class Router;
class AsuTransportPool;
}  // namespace kv

namespace kv {

using TransportFactory = std::function<std::unique_ptr<AsuTransport>()>;
using TransProviderFactory =
    std::function<Status(const TransportConfig&, std::shared_ptr<TransProvider>&)>;

std::unique_ptr<KvClient> CreateKvClient(TransportFactory transportFactory);
std::unique_ptr<KvClient> CreateKvClient(TransportFactory transportFactory,
                                         TransProviderFactory transProviderFactory);

// ViewSnapshot is the immutable routing state used by foreground IO and submitted tasks.
struct ViewSnapshot {
    KvNsId nsId{kDefaultKvNsId};
    std::shared_ptr<kv::Router> router;
    std::vector<NodeId> asuIds;
    GlobalView view;
    std::unordered_map<NodeId, std::shared_ptr<AsuTransport>> transports;
};

using ViewSnapshotPtr = std::shared_ptr<ViewSnapshot>;
// Published snapshots keyed by namespace.
using ViewSnapshotMap = std::unordered_map<KvNsId, ViewSnapshotPtr>;

class KvClientImpl;

// KvClientImpl coordinates routing, transports, and aggregate task tracking.
class KvClientImpl final : public KvClient {
public:
    // Builds a client with the provided transport factory.
    explicit KvClientImpl(TransportFactory transportFactory,
                          ViewServerFactory viewServerFactory = nullptr,
                          TransProviderFactory transProviderFactory = nullptr);
    // Shuts down the client during destruction.
    ~KvClientImpl() override;

    // Initializes routing and transport resources from an already parsed config.
    Status Init(const KvClientConfig& config) override;
    // Gracefully drains tracked client tasks and releases resources.
    Status Shutdown() override;

    // Submits query operations to routed transports.
    Status QueryAsync(const std::vector<CacheKey>& keys, TaskId& taskId) override;
    // Submits load operations to routed transports.
    Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) override;
    // Submits store operations to routed transports.
    Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId,
                      std::function<Status()> prerequisite = {}) override;
    // Submits delete operations to routed transports.
    Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId) override;

    // Namespace-aware submissions. They select the snapshot of the given namespace
    // and are the entry point used by the public API once it carries KvNsId.
    Status QueryAsync(KvNsId nsId, const std::vector<CacheKey>& keys, TaskId& taskId);
    Status LoadAsync(KvNsId nsId, const std::vector<KVBuffer>& entries, TaskId& taskId);
    Status StoreAsync(KvNsId nsId, const std::vector<KVBuffer>& entries, TaskId& taskId,
                      std::function<Status()> prerequisite = {});
    Status DeleteAsync(KvNsId nsId, const std::vector<CacheKey>& keys, TaskId& taskId);

    // Checks an aggregate task.
    bool Check(TaskId taskId) override;
    // Waits for an aggregate task.
    Status Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result) override;

    // Registers regions and remembers successful resources for future views.
    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisteredMemory>& registeredRegions) override;
    // Unregisters regions and forgets successful resources.
    Status UnregisterRegions(const std::vector<MRHandle>& handles) override;

private:
    // Rolls a partially initialized client back to the pre-Init state.
    Status FailInit(Status status);

    struct ProviderMemoryState {
        std::shared_ptr<TransProvider> provider;
        std::unordered_map<MRHandle, MRHandle> regionHandles;
    };

    // Creates and queues one entry-based client task.
    Status SubmitAsync(AsuOpType opType, const std::vector<KVBuffer>& entries, TaskId& taskId,
                       std::function<Status()> prerequisite = {});
    // Creates and queues one key-based client task.
    Status SubmitAsync(AsuOpType opType, const std::vector<CacheKey>& keys, TaskId& taskId);
    // Namespace-aware task creation; the namespace is forwarded to every transport task.
    Status SubmitAsync(AsuOpType opType, KvNsId nsId, const std::vector<KVBuffer>& entries,
                       TaskId& taskId, std::function<Status()> prerequisite = {});
    Status SubmitAsync(AsuOpType opType, KvNsId nsId, const std::vector<CacheKey>& keys,
                       TaskId& taskId);

    // Runs queued tasks until shutdown and the queue are both complete.
    void WorkerLoop();

    // Performs one register operation on the current snapshot.
    Status RegisterRegionsOnce(const std::vector<MemoryRegion>& regions,
                               std::vector<RegisteredMemory>& registeredRegions, bool& needRefresh);
    // Builds a complete immutable snapshot for one namespace view. Transports always come
    // from the global transport pool.
    Status BuildSnapshot(KvNsId nsId, const GlobalView& view,
                         std::shared_ptr<ViewSnapshot>& snapshot);
    // Creates and initializes a transport from a ready transport config.
    Status CreateTransportForNode(const TransportConfig& config,
                                  std::shared_ptr<AsuTransport>& transport);
    // Builds the transport config for a node: exact config first, template clone otherwise.
    Status BuildTransportConfigForNode(NodeId nodeId, const NodeInfo& asuInfo,
                                       TransportConfig& config) const;
    // Returns the pooled transport for a node, creating and pooling it when missing.
    Status GetOrCreateTransport(NodeId nodeId, const NodeInfo& asuInfo,
                                std::shared_ptr<AsuTransport>& transport);
    Status BindProviderRegions(const std::shared_ptr<TransProvider>& transProvider,
                               const std::vector<RegisteredMemory>& registeredRegions,
                               std::vector<MRHandle>& localHandles);
    Status UnregisterProviderRegions(const std::shared_ptr<TransProvider>& transProvider,
                                     const std::vector<MRHandle>& handles);
    // Returns the current immutable snapshot of a namespace, or nullptr.
    ViewSnapshotPtr GetSnapshot(KvNsId nsId) const;
    // Returns the published snapshot when exactly one namespace is published, else nullptr.
    ViewSnapshotPtr GetSingleSnapshot() const;
    // Resolves the snapshot for a submission. Callers that pass kDefaultKvNsId have no
    // namespace of their own and resolve to the only published namespace.
    Status ResolveSnapshot(KvNsId nsId, std::shared_ptr<ViewSnapshot>& snapshot) const;
    // Returns any current snapshot; used by resource paths that are namespace agnostic.
    std::shared_ptr<ViewSnapshot> GetAnySnapshot() const;

    // Scheduler callback: fetch a candidate universe, validate it, build the changed
    // namespace snapshots and publish everything atomically. A failed refresh leaves the
    // published state unchanged.
    Status OnViewUniverseUpdated();
    // Fetches the next complete view universe. M800 slaves deliver every namespace; legacy
    // single-view servers are adapted to the default namespace.
    Status FetchCandidateUniverse(ViewServer& viewServer, ViewUniverse& candidate);
    // Validates a candidate universe against the published global epoch.
    Status ValidateCandidateUniverse(const ViewUniverse& candidate,
                                     std::uint64_t publishedEpoch) const;
    // Builds the next snapshot table for a candidate universe. Only namespaces whose view
    // must be republished are rebuilt; the rest reuse the published snapshot.
    Status BuildSnapshots(const ViewUniverse& candidate, ViewSnapshotMap& nextSnapshots);
    // Publishes a validated candidate: universe first, then the snapshot table, in one
    // critical section. Failures leave the published state untouched. The snapshot table
    // is taken by rvalue reference: the caller hands it over and must not reuse it.
    Status ApplyUniverse(const ViewUniverse& candidate, ViewSnapshotMap&& nextSnapshots,
                         bool initial);
    // Triggers a refresh through the injected scheduler. Without a scheduler the client
    // performs no refresh.
    void TriggerRefresh(const Status& status);
    // Stops the scheduler and waits for a refresh in progress.
    void StopRefresh();
    // Refresh interval from configuration; defaults to 30 seconds, minimum 1 second.
    // TODO: agree the default value with the view provider owners.
    static std::chrono::milliseconds GetRefreshInterval(const KvClientConfig& config);

    // Returns whether a status suggests the published snapshot should be refreshed.
    bool IsRefreshNeeded(const Status& status) const;
    // Extracts sorted ASU ids from a view.
    static std::vector<NodeId> GetSortedAsuIds(const GlobalView& view);
    // Adds context to a status message.
    static Status WithContext(Status status, const std::string& context);
    // Builds the standard not-initialized status.
    static Status NotInitialized();

    // Tracks aggregate client tasks returned through public TaskId values.
    ClientTaskManager taskManager_;
    // Serializes producers and protects the shutdown acceptance boundary.
    std::mutex producerMu_;
    std::condition_variable workerCv_;
    kv::SpscRingQueue<ClientTaskPtr> taskQueue_;
    std::atomic_bool stopWorker_{true};
    std::thread worker_;
    // Creates ASU transports; tests inject fake transports through this hook.
    TransportFactory transportFactory_;
    // Creates providers; tests inject fake providers through this hook.
    TransProviderFactory transProviderFactory_;
    // Creates the external view server during Init.
    ViewServerFactory viewServerFactory_;
    // mutex_ protects background refresh state and resource/view caches.
    mutable std::mutex mutex_;
    // memoryMu_ serializes provider creation with business-memory register/bind/unregister.
    std::mutex memoryMu_;
    // Tracks whether Init has published a usable snapshot.
    bool initialized_{false};
    // Last accepted initialization config.
    KvClientConfig config_;
    // Source for dynamic global views; may be backed by viewServiceAddrs.
    std::shared_ptr<ViewServer> viewServer_;
    // Transport configs indexed by ASU id for snapshot construction.
    std::unordered_map<NodeId, TransportConfig> transportConfigs_;
    // Provider-local handles indexed by the canonical handles returned by this client.
    std::vector<ProviderMemoryState> providerMemoryStates_;
    // Provider selected for all client business-memory registration operations.
    std::shared_ptr<TransProvider> memoryProvider_;
    // Regions registered by the client and rebound to newly added providers.
    std::unordered_map<MRHandle, RegisteredMemory> registeredRegions_;
    // Current immutable routing and transport snapshots, keyed by namespace.
    ViewSnapshotMap snapshots_;
    // Published view universe; namespaces and global epoch of the current snapshots.
    std::shared_ptr<ViewUniverse> viewUniverse_;
    // Resident refresh scheduler; created by Init and stopped by Shutdown.
    std::unique_ptr<RefreshScheduler> refreshScheduler_;
    // Owns every AsuTransport of this client; closed by Shutdown.
    AsuTransportPool transportPool_;
};

}  // namespace kv
