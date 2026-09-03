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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "trans_provider.h"
#include "buffer_manager.h"
#include "connection_manager.h"
#include "io_scheduler.h"
#include "kv_protocol.h"
#include "transport_task_manager.h"

namespace UC::ASU {

inline KvOpcode ToKvOpcode(AsuOpType opType)
{
    switch (opType) {
        case AsuOpType::LOAD: return KvOpcode::Retrieve;
        case AsuOpType::STORE: return KvOpcode::Store;
        case AsuOpType::BATCH_LOAD: return KvOpcode::BatchRetrieve;
        case AsuOpType::BATCH_STORE: return KvOpcode::BatchStore;
        case AsuOpType::DELETE: return KvOpcode::Delete;
        case AsuOpType::QUERY: return KvOpcode::Exist;
        case AsuOpType::KEEP_ALIVE: return KvOpcode::KeepAlive;
    }
    return KvOpcode::KeepAlive;
}

// Owns the transport data path for one task and all of its sub-batches.
// Task lookup, waiting, and callback delivery remain TransportTaskManager concerns.
class TransportTaskExecutor {
public:
    TransportTaskExecutor(const TransportConfig& config,
                          const std::shared_ptr<TransProvider>& transProvider,
                          const std::unique_ptr<ConnectionManager>& connectionManager);
    ~TransportTaskExecutor();

    Status Init();
    Status Shutdown();

    bool Execute(const TransportTaskPtr& task);
    bool Poll(const TransportTaskPtr& task);
    bool Cancel(const TransportTaskPtr& task);

private:
    Status RegisterBufferMemory(BufferManager& bufferManager, MRHandle& mrHandle);
    Status UnregisterBufferMemory(MRHandle& mrHandle);
    std::uint16_t AllocateRequestCid();

    Status PrepareTaskSubBatches(const TransportTask& task,
                                 std::vector<TransportSubBatchContext>& subBatchContexts);
    Status BuildEntrySubBatchRequest(AsuOpType opType,
                                     const IoScheduler::ScheduledIoBatch& subBatch,
                                     TransportSubBatchContext& subBatchContext);
    Status SubmitKeySubBatchRequest(AsuOpType opType,
                                    const IoScheduler::ScheduledKeyBatch& subBatch,
                                    TransportSubBatchContext& subBatchContext);
    Status SubmitKeepAliveRequest(TransportSubBatchContext& subBatchContext);

    Status AssignSubBatchConnections(std::vector<TransportSubBatchContext>& subBatchContexts);
    void BuildSubBatchSendBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                                  std::vector<TransProvider::SendIoBatch>& ioBatches);
    void SendSubBatchBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                             const std::vector<TransProvider::SendIoBatch>& ioBatches);

    void AbortSubBatchesBeforeSend(TransportTask& task,
                                   std::vector<TransportSubBatchContext>& subBatchContexts);
    void CompleteSubBatch(TransportTask& task, TransportSubBatchContext& subBatchContext,
                          const Status& status);
    void ReleaseSubBatchResources(TransportSubBatchContext& subBatchContext);
    void ReleaseAllSubBatchResources(std::vector<TransportSubBatchContext>& subBatchContexts);

    const TransportConfig& config_;
    IoScheduler ioScheduler_;
    const std::shared_ptr<TransProvider>& transProvider_;
    BufferManager sendBufferManager_;
    BufferManager flagBufferManager_;
    MRHandle sendBufferMrHandle_{kInvalidMRHandle};
    MRHandle flagBufferMrHandle_{kInvalidMRHandle};
    ProtocolManager protocolManager_;
    const std::unique_ptr<ConnectionManager>& connManager_;
    std::atomic<std::uint16_t> nextRequestCid_{1};
};

}  // namespace UC::ASU
