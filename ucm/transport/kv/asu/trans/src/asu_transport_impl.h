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
#include "connection_manager.h"
#include "template/spsc_ring_queue.h"
#include "transport_task_manager.h"

namespace UC::ASU {

using TransportTaskContextPtr = std::shared_ptr<TransportTaskContext>;

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
    Status LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) override;
    Status StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId) override;
    Status DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId) override;

    Status Cancel(TaskId taskId) override;
    Status Check(TaskId taskId, TaskResult& result) override;
    Status Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result) override;

    Status StubCheck(TaskId task_id,
                     TaskResult& result);  // Stub for testing, remove after real implementation
    Status StubWait(TaskId task_id, std::uint64_t timeout_ms,
                    TaskResult& result);  // Stub for testing, remove after real implementation

    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisterResult>& results) override;

    Status BindRegisteredRegions(const std::vector<RegisteredMemory>& regions,
                                 std::vector<RegisterResult>& results) override;

    Status UnregisterRegions(const std::vector<MRHandle>& handles) override;

#ifdef ASU_BUILD_TESTS
    friend class AsuSmokeTest_ConcurrentAll8InterfacesWithChannelRebuild_Test;
    friend class AsuSmokeTest_SequentialChannelDrainAndRebuild_Test;
    friend class AsuSmokeTest_DrainUnderHeavyConcurrentLoad_Test;
    friend class AsuSmokeTest_ClientAsyncTasksCompleteEndToEnd_Test;
    std::atomic<bool> useStubCompleteTask_{true};
#endif

private:
    using TransportTaskContextPtr = std::shared_ptr<TransportTaskContext>;
    Status SubmitAsync(std::unique_ptr<TransportTaskContext> ctx, TaskId& taskId);
    void WorkerLoop();
    void CompleteTask(const TransportTaskContextPtr& ctx);

    // Stub for testing, remove after real implementation
    Status StubSend(ConnectionChannel* channel, TransportTaskContext* ctx);
    std::vector<ConnectionHandle> StubCreateConnection(const AsuEndpoint& endpoint,
                                                       std::uint32_t qp_num);
    std::vector<Status> StubDeleteConnections(const std::vector<ConnectionHandle>& handles);
    void StubCompleteTask(const TransportTaskContextPtr& ctx);

    void BuildResult(const TransportTaskContext& ctx, TaskResult& result);

    TransportConfig config_;

    ConnectionManager connManager_;
    TransportTaskManager taskManager_;
    // TODO: optimize spsc pattern or just submit to RDMA/UB directly ?
    UC::SpscRingQueue<TransportTaskContextPtr> executeQueue_;
    std::mutex producerMu_;

    std::thread worker_;
    std::atomic_bool stop_{false};

    std::mutex registeredRegionsMu_;
    std::atomic<MRHandle> nextMrHandle_{1};
    std::unordered_map<MRHandle, MemoryRegion> registeredRegions_;
};

}  // namespace UC::ASU
