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
#include "asu_transport/trans_provider.h"
#include "buffer_manager.h"
#include "connection_manager.h"
#include "io_scheduler.h"
#include "kv_protocol.h"
#include "template/spsc_ring_queue.h"
#include "transport_task_executor.h"
#include "transport_task_manager.h"

namespace UC::ASU {

class AsuTransportImpl final : public AsuTransport {
public:
    AsuTransportImpl();
    ~AsuTransportImpl() override;

    Status Init(const TransportConfig& config) override;
    Status Init(const std::string& configPath) override;
    Status Shutdown() override;

    Status CheckHealth() override;

    Status Submit(const TransportTaskPtr& task) override;

    Status Cancel(TaskId taskId) override;

    Status RegisterRegions(const std::vector<MemoryRegion>& regions,
                           std::vector<RegisteredMemory>& registeredRegions) override;

    Status BindRegisteredRegions(const std::vector<RegisteredMemory>& regions) override;

    Status UnregisterRegions(const std::vector<MRHandle>& handles) override;

private:
    Status SubmitTask(const TransportTaskPtr& task);
    void WorkerLoop();
    void CompletionLoop();

    void SetTransProvider(std::shared_ptr<TransProvider> provider);
    Status UnregisterOwnedRegionHandles(const std::vector<MRHandle>& handles);

    TransportConfig config_;
    IoScheduler ioScheduler_;
    std::shared_ptr<TransProvider> transProvider_;
    BufferManager sendBufferManager_;
    BufferManager flagBufferManager_;
    std::unique_ptr<ProtocolManager> protocolManager_;

    std::unique_ptr<ConnectionManager> connManager_;
    std::atomic<std::uint16_t> nextRequestCid_{1};

    std::mutex registeredRegionsMu_;
    std::unordered_map<MRHandle, RegisteredMemory> registeredRegions_;
    bool ownsRegisteredRegionHandles_{false};

    std::unique_ptr<TransportTaskExecutor> taskExecutor_;
    TransportTaskManager taskManager_;
    UC::SpscRingQueue<TransportTaskPtr> executeQueue_;
    std::mutex producerMu_;

    std::thread worker_;
    std::thread completionWorker_;
    std::atomic_bool stopWorker_{false};
    std::atomic_bool stopCompletionWorker_{false};
};

}  // namespace UC::ASU
