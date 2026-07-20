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
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "completion_poller.h"
#include "drampool_config.h"
#include "drampool_types.h"
#include "kv_protocol.h"
#include "status/status.h"

namespace transport {
class TcpMessageChannel;
}

namespace UC::DramPool {

class TaskWorker;

class DramPoolServer {
public:
    DramPoolServer();
    ~DramPoolServer();

    DramPoolServer(const DramPoolServer&) = delete;
    DramPoolServer& operator=(const DramPoolServer&) = delete;

    Status Init();
    Status Start();
    void Stop();

private:
    enum class ServerState {
        New = 0,
        Initialized,
        Running,
        Stopped,
    };

    Status InitializeAclRuntime();
    Status InitMemoryPool();
    Status InitMetadata();
    Status InitProtocol();
    Status InitQueues();
    Status InitTransportManager();
    Status CreateRuntimeContext();

    Status StartTransportService();
    Status RegisterBufferPools();
    Status StartTcpMessageChannel();
    Status StartCompletionPoller();
    Status StartTaskWorker();
    Status StartRequestReceiver();
    Status StartGCThread();

    void StopTcpMessageChannel();
    void StopRequestReceiver();
    void StopTaskWorker();
    void MarkInflightTransportsFailed();
    void StopCompletionPoller();
    void StopGCThread();
    void UnregisterBufferPools();
    void StopTransportService();

    void RequestReceiveLoop();
    bool WaitForChannelReady();
    void TaskWorkerLoop();
    void CompletionPollerLoop();
    void GCThreadLoop();
    void ResetInitializedComponents();

    std::atomic_bool requestReceiverStop_{true};
    std::atomic_bool taskWorkerStop_{true};
    std::atomic_bool completionPollerStop_{true};
    std::atomic_bool gcThreadStop_{true};
    bool tcpMessageChannelReady_{false};

    std::thread requestReceiverThread_;
    std::thread taskWorkerThread_;
    std::thread completionPollerThread_;
    std::thread gcThread_;

    RequestQueue requestQueue_;
    CompletionQueue completionQueue_;
    std::unique_ptr<transport::TransportManager> transportManager_;
    std::unique_ptr<transport::TcpMessageChannel> tcpMessageChannel_;
    BufferPoolList bufferPools_;
    std::vector<transport::MemoryHandle> bufferPoolMemoryHandles_;
    std::unique_ptr<UC::DramPool::MetadataManager> metadataManager_;
    std::unique_ptr<ProtocolManager> protocolManager_;
    std::unique_ptr<DramPoolRuntime> runtime_;
    std::unique_ptr<TaskWorker> taskWorker_;
    std::unique_ptr<CompletionPoller> completionPoller_;
    bool aclRuntimeOwned_{false};

    ServerState state_{ServerState::New};
    std::mutex requestReceiverWaitMutex_;
    std::condition_variable requestReceiverWaitCv_;
    std::mutex stopWaitMutex_;
    std::condition_variable stopWaitCv_;
};

}  // namespace UC::DramPool
