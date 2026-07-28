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

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "asu_client/asu_client.h"
#include "asu_transport/types.h"
#include "client_task_manager.h"
#include "view_server.h"

namespace UC::KV {
class Router;
}  // namespace UC::KV

namespace UC::ASU {

// ViewSnapshot is the immutable routing state used by foreground IO and submitted tasks.
struct ViewSnapshot {
    std::shared_ptr<UC::KV::Router> router;
    std::vector<AsuId> asuIds;
    GlobalView view;
    std::unordered_map<AsuId, std::shared_ptr<AsuTransport>> transports;
};

class AsuClientImpl;

// AsuClientImpl coordinates routing, transports, and aggregate task tracking.
class AsuClientImpl final : public AsuClient {
public:
    // Builds a client with the provided transport factory.
    explicit AsuClientImpl(TransportFactory transportFactory,
                           ViewServerFactory viewServerFactory = nullptr);
    // Shuts down the client during destruction.
    ~AsuClientImpl() override;

    // Initializes routing and transport resources.
    Status Init(const std::string& configPath) override;
    // Initializes from an already parsed config; intended for internal tests and adapters.
    Status Init(const AsuClientConfig& config) override;
    // Gracefully drains tracked client tasks and releases resources.
    Status Shutdown() override;

    // Queries key existence and schedules background refresh on refreshable failures.
    Status Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                 QueryResult& result) override;

    // Submits load operations to routed transports.
    Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) override;
    // Submits store operations to routed transports.
    Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) override;
    // Submits delete operations to routed transports.
    Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId) override;

    // Checks an aggregate task.
    Status Check(TaskId taskId, TaskResult& result) override;
    // Waits for an aggregate task.
    Status Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result) override;

    // Registers regions and remembers successful resources for future views.
    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisteredMemory>& registeredRegions) override;
    // Unregisters regions and forgets successful resources.
    Status UnregisterRegions(const std::vector<MRHandle>& handles) override;

private:
    using ClientTaskContextPtr = std::shared_ptr<ClientTaskContext>;
    struct PendingQuery {
        AsuId asuId{0};
        std::shared_ptr<AsuTransport> transport;
        std::vector<CacheKey> keys;
        std::vector<std::size_t> originalIndices;
        TaskId taskId{kInvalidTaskId};
    };

    // Creates and queues one entry-based client task.
    Status SubmitAsync(ClientOpType opType, const std::vector<KVBuffer>& entries, TaskId& taskId);
    // Creates and queues one key-based client task.
    Status SubmitAsync(ClientOpType opType, const std::vector<CacheKey>& keys, TaskId& taskId);

    // Routes and dispatches one queued client task.
    void ProcessTask(const ClientTaskContextPtr& ctx);
    // Builds transport subtasks for one queued client task.
    Status BuildSubTasks(const ClientTaskContextPtr& ctx);
    // Runs queued tasks until shutdown and the queue are both complete.
    void WorkerLoop();
    // Completes a task that failed before transport dispatch.
    static void CompleteTaskWithError(const ClientTaskContextPtr& ctx, const Status& status);
    // Sends each subtask to its routed transport and records transport task ids.
    Status DispatchTask(const ClientTaskContextPtr& ctx);
    // Writes one transport callback result into its client subtask.
    static void CompleteSubTask(const ClientTaskContextPtr& ctx, std::size_t subTaskIndex,
                                TaskResult result);
    // Aggregates subtask state after the shared completion count reaches zero.
    static void FinalizeTask(const ClientTaskContextPtr& ctx);
    // Converts a client task context into the public task result shape.
    Status BuildResult(const ClientTaskContextPtr& ctx, TaskResult& result);
    // Waits for one client task context until completion or timeout.
    Status WaitTaskContext(const ClientTaskContextPtr& ctx, std::uint64_t timeoutMs,
                           TaskResult& result);

    // Performs one query attempt on the current snapshot.
    Status QueryOnce(const std::vector<CacheKey>& keys, const QueryOptions& options,
                     QueryResult& result, bool& needRefresh);
    // Performs one register operation on the current snapshot.
    Status RegisterRegionsOnce(const std::vector<MemoryRegion>& regions,
                               std::vector<RegisteredMemory>& registeredRegions, bool& needRefresh);
    // Performs one unregister operation on the current snapshot.
    Status UnregisterRegionsOnce(const std::vector<MRHandle>& handles, bool& needRefresh);

    // Builds a complete immutable snapshot for a view.
    Status BuildSnapshot(const GlobalView& view, const std::shared_ptr<ViewSnapshot>& oldSnapshot,
                         std::shared_ptr<ViewSnapshot>& snapshot);
    // Creates and initializes a transport for one ASU.
    Status BuildTransport(AsuId asuId, const AsuInfo& asuInfo,
                          std::shared_ptr<AsuTransport>& transport);
    // Binds remembered registered regions to a transport.
    Status BindRegisteredRegions(AsuId asuId, const std::shared_ptr<AsuTransport>& transport);
    // Returns the current immutable snapshot if initialized.
    std::shared_ptr<ViewSnapshot> GetSnapshot() const;

    // Refreshes the view and publishes it if it is newer.
    Status RefreshView();
    // Starts a non-blocking refresh after a status indicates stale routing or transport state.
    void RequestBackgroundRefresh();
    // Waits for the background refresh worker to finish.
    void JoinBackgroundRefresh();

    // Shuts down transports owned by a snapshot.
    Status ShutdownSnapshotTransports(const std::shared_ptr<ViewSnapshot>& snapshot);
    // Waits for tracked client tasks before transport shutdown.
    Status DrainTasksBeforeShutdown(std::uint64_t waitTimeoutMs);

    // Marks whether a status suggests the published snapshot should be refreshed.
    void MarkRefreshIfNeeded(const Status& status, bool& needRefresh) const;
    // Extracts sorted ASU ids from a view.
    static std::vector<AsuId> GetSortedAsuIds(const GlobalView& view);
    // Parses client config from a file path supplied through the public interface.
    static Status LoadConfig(const std::string& configPath, AsuClientConfig& config);
    // Adds context to a status message.
    static Status WithContext(Status status, const std::string& context);
    // Builds the standard not-initialized status.
    static Status NotInitialized();

    // Tracks aggregate client tasks returned through public TaskId values.
    ClientTaskManager taskManager_;
    // Protects the client worker queue and shutdown acceptance boundary.
    std::mutex taskQueueMu_;
    std::condition_variable taskQueueCv_;
    std::deque<ClientTaskContextPtr> taskQueue_;
    bool stopWorker_{true};
    std::thread worker_;
    // Creates ASU transports; tests inject fake transports through this hook.
    TransportFactory transportFactory_;
    // Creates the external view server during Init.
    ViewServerFactory viewServerFactory_;
    // mutex_ protects background refresh state and resource/view caches.
    mutable std::mutex mutex_;
    // Tracks whether Init has published a usable snapshot.
    bool initialized_{false};
    // Prevents duplicate background refresh workers.
    bool refreshInProgress_{false};
    // Last accepted initialization config.
    AsuClientConfig config_;
    // Source for dynamic global views; may be backed by viewServiceAddrs.
    std::shared_ptr<ViewServer> viewServer_;
    // Transport configs indexed by ASU id for snapshot construction.
    std::unordered_map<AsuId, TransportConfig> transportConfigs_;
    // Regions registered on the current view and rebound to newly added transports.
    std::vector<RegisteredMemory> registeredRegions_;
    // Current immutable routing and transport snapshot.
    std::shared_ptr<ViewSnapshot> snapshot_;
    // Transports removed from the active snapshot but still needed by old tasks.
    std::vector<std::shared_ptr<AsuTransport>> retiredTransports_;
    // Worker used for non-blocking refresh requests.
    std::thread refreshThread_;
};

}  // namespace UC::ASU
