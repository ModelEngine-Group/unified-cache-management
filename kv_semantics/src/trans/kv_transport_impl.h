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
#include <vector>
#include "kv_transport.h"
#include "trans_provider.h"
#include "conn/connection_manager.h"
#include "spsc_ring_queue.h"
#include "task/trans_task_executor.h"
#include "task/trans_task_manager.h"

namespace kv {

class AsuTransportImpl final : public AsuTransport {
public:
    AsuTransportImpl();
    ~AsuTransportImpl() override;

    Status Init(const TransportConfig& config,
                std::shared_ptr<TransProvider> transProvider) override;
    Status Init(const std::string& configPath,
                std::shared_ptr<TransProvider> transProvider) override;
    Status Shutdown() override;

    Status CheckHealth() override;

    Status Submit(const TransportTaskPtr& task) override;

    Status Cancel(TaskId taskId) override;

private:
    Status SubmitTask(const TransportTaskPtr& task);
    void WorkerLoop();
    void CompletionLoop();

    void SetTransProvider(std::shared_ptr<TransProvider> provider);

    TransportConfig config_;
    std::shared_ptr<TransProvider> transProvider_;
    std::unique_ptr<ConnectionManager> connManager_;

    std::unique_ptr<TransportTaskExecutor> taskExecutor_;
    TransportTaskManager taskManager_;
    kv::SpscRingQueue<TransportTaskPtr> executeQueue_;
    std::mutex producerMu_;

    std::thread worker_;
    std::thread completionWorker_;
    std::atomic_bool stopWorker_{false};
    std::atomic_bool stopCompletionWorker_{false};
};

}  // namespace kv
