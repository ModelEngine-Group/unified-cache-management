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
#include "asu_transport/trans_provider.h"
#include "buffer_manager.h"
#include "connection_manager.h"
#include "io_scheduler.h"
#include "kv_protocol.h"
#include "transport_task_manager.h"

namespace UC::ASU {

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

// Owns the transport data path for one task and all of its sub-batches.
// Task lookup, waiting, and callback delivery remain TransportTaskManager concerns.
class TransportTaskExecutor {
public:
    TransportTaskExecutor(const TransportConfig& config, IoScheduler& ioScheduler,
                          const std::shared_ptr<TransProvider>& transProvider,
                          BufferManager& sendBufferManager, BufferManager& flagBufferManager,
                          const std::unique_ptr<ProtocolManager>& protocolManager,
                          const std::unique_ptr<ConnectionManager>& connectionManager,
                          std::atomic<std::uint16_t>& nextRequestCid);

    bool Execute(const TransportTaskPtr& task);
    bool Poll(const TransportTaskPtr& task);
    bool Cancel(const TransportTaskPtr& task);

private:
    std::uint16_t AllocateRequestCid();

    Status PrepareTaskSubBatches(const TransportTask& task,
                                 std::vector<TransportSubBatchContext>& subBatchContexts);
    Status BuildEntrySubBatchRequest(TransportOpType opType,
                                     const IoScheduler::ScheduledIoBatch& subBatch,
                                     TransportSubBatchContext& subBatchContext);
    Status SubmitKeySubBatchRequest(TransportOpType opType,
                                    const IoScheduler::ScheduledKeyBatch& subBatch,
                                    TransportSubBatchContext& subBatchContext);
    Status SubmitKeepAliveRequest(TransportSubBatchContext& subBatchContext);

    Status AssignSubBatchConnections(std::vector<TransportSubBatchContext>& subBatchContexts);
    Status BuildSubBatchSendBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                                    std::vector<TransProvider::SendIoBatch>& ioBatches,
                                    std::vector<std::size_t>& subBatchIndexes);
    Status SendSubBatchBuffers(std::vector<TransportSubBatchContext>& subBatchContexts,
                               const std::vector<TransProvider::SendIoBatch>& ioBatches,
                               const std::vector<std::size_t>& subBatchIndexes);

    void CompleteSubBatch(TransportTask& task, TransportSubBatchContext& subBatchContext,
                          const Status& status);
    void ReleaseSubBatchResources(TransportSubBatchContext& subBatchContext);
    void ReleaseAllSubBatchResources(std::vector<TransportSubBatchContext>& subBatchContexts);

    const TransportConfig& config_;
    IoScheduler& ioScheduler_;
    const std::shared_ptr<TransProvider>& transProvider_;
    BufferManager& sendBufferManager_;
    BufferManager& flagBufferManager_;
    const std::unique_ptr<ProtocolManager>& protocolManager_;
    const std::unique_ptr<ConnectionManager>& connManager_;
    std::atomic<std::uint16_t>& nextRequestCid_;
};

}  // namespace UC::ASU
