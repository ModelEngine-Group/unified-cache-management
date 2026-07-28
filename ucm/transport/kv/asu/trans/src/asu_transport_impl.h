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
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "asu_transport/asu_transport.h"
#include "buffer_manager.h"
#include "connection_manager.h"
#include "io_scheduler.h"
#include "kv_protocol.h"
#include "template/spsc_ring_queue.h"
#include "trans_provider.h"
#include "transport_task_manager.h"

namespace UC::ASU {

using TransportTaskContextPtr = std::shared_ptr<TransportTaskContext>;

inline KvOpcode ToKvOpcode(TransportOpType opType)
{
    switch (opType) {
        case TransportOpType::LOAD: return KvOpcode::Retrieve;
        case TransportOpType::STORE: return KvOpcode::Store;
        case TransportOpType::BATCH_LOAD: return KvOpcode::BatchRetrieve;
        case TransportOpType::BATCH_STORE: return KvOpcode::BatchStore;
        case TransportOpType::DELETE: return KvOpcode::Delete;
        case TransportOpType::QUERY: return KvOpcode::Exist;
        case TransportOpType::KEEP_ALIVE: return KvOpcode::KeepAlive;
    }
    return KvOpcode::KeepAlive;
}

class AsuTransportImpl final : public AsuTransport {
public:
    AsuTransportImpl() = default;
    ~AsuTransportImpl() override;

    Status Init(const TransportConfig& config) override;
    Status Init(const std::string& configPath) override;
    Status Shutdown() override;

    Status CheckHealth() override;

    Status Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                 QueryResult& result) override;
    Status QueryAsync(const std::vector<CacheKey>& keys, const QueryOptions& options,
                      TaskId& taskId) override;
    Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId,
                     TaskCompletionCallback onComplete) override;
    Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId,
                      TaskCompletionCallback onComplete) override;
    Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId,
                       TaskCompletionCallback onComplete) override;

    Status Cancel(TaskId taskId) override;
    Status Check(TaskId taskId, TaskResult& result) override;
    Status Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result) override;

    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisteredMemory>& registeredRegions) override;

    Status BindRegisteredRegions(const std::vector<RegisteredMemory>& regions) override;

    Status UnregisterRegions(const std::vector<MRHandle>& handles) override;

private:
    std::uint16_t AllocateRequestCid();
    Status SubmitAsync(std::unique_ptr<TransportTaskContext> ctx, TaskId& taskId);
    void WorkerLoop();
    void CompletionLoop();
    void ProcessTask(const TransportTaskContextPtr& ctx);
    Status AssignSubBatchConnections(std::vector<TransportSubBatchContext>& subBatchContexts);
    Status SubmitTaskRequests(const TransportTaskContext& ctx,
                              std::vector<TransportSubBatchContext>& subBatchContexts);
    Status BuildSubBatchSendBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                                    std::vector<TransProvider::SendIoBatch>& ioBatches,
                                    std::vector<std::size_t>& subBatchIndexes);
    Status SendSubBatchBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                               const std::vector<TransProvider::SendIoBatch>& ioBatches,
                               const std::vector<std::size_t>& subBatchIndexes);
    Status SubmitEntrySubBatchRequest(TransportOpType opType,
                                      const IoScheduler::ScheduledIoBatch& subBatch,
                                      TransportSubBatchContext& subBatchContext);
    Status SubmitKeySubBatchRequest(TransportOpType opType,
                                    const IoScheduler::ScheduledKeyBatch& subBatch,
                                    TransportSubBatchContext& subBatchContext);
    Status SubmitKeepAliveRequest(TransportSubBatchContext& subBatchContext);
    void ReleaseSubBatchResources(TransportSubBatchContext& subBatchContext);
    void ReleaseAllSubBatchResources(std::vector<TransportSubBatchContext>& subBatchContexts);
    void CompleteSubBatch(TransportTaskContext& ctx, TransportSubBatchContext& subBatchContext,
                          const Status& status);

    void PollTaskCompletions(const TransportTaskContextPtr& ctx);
    void BuildResult(const TransportTaskContext& ctx, TaskResult& result);
    void NotifyTaskCompletion(const TransportTaskContextPtr& ctx);

    void SetTransProvider(std::unique_ptr<TransProvider> provider);
    Status UnregisterOwnedRegionHandles(const std::vector<MRHandle>& handles);

    TransportConfig config_;
    IoScheduler ioScheduler_;
    std::unique_ptr<TransProvider> transProvider_;
    BufferManager sendBufferManager_;
    BufferManager flagBufferManager_;
    std::unique_ptr<ProtocolManager> protocolManager_;

    std::unique_ptr<ConnectionManager> connManager_;
    TransportTaskManager taskManager_;
    UC::SpscRingQueue<TransportTaskContextPtr> executeQueue_;
    std::mutex producerMu_;

    std::thread worker_;
    std::thread completionWorker_;
    std::atomic_bool stop_{false};
    std::atomic<std::uint16_t> nextRequestCid_{1};

    std::mutex registeredRegionsMu_;
    std::unordered_map<MRHandle, RegisteredMemory> registeredRegions_;
    bool ownsRegisteredRegionHandles_{false};
};

}  // namespace UC::ASU
