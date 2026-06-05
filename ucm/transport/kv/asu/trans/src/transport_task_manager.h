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
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "asu_transport/types.h"
#include "buffer_manager.h"
#include "connection_manager.h"
#include "task_manager_base.h"

namespace UC::ASU {

enum class TransportOpType {
    QUERY = 0,
    LOAD = 1,
    STORE = 2,
    BATCH_LOAD = 3,
    BATCH_STORE = 4,
    DELETE = 5,
    KEEP_ALIVE = 6,
};

enum class TransportTaskState {
    PENDING = 0,
    INFLIGHT = 1,
    COMPLETED = 2,
    CANCELED = 3,
};

enum class TransportSubBatchState {
    PENDING = 0,
    COMPLETED = 1,
};

constexpr std::size_t kAsuBatchLoadMaxIoSize = 110;
constexpr std::size_t kAsuBatchStoreMaxIoSize = 110;
constexpr std::size_t kAsuDeleteMaxIoSize = 254;
constexpr std::size_t kAsuQueryMaxIoSize = 256;

inline std::size_t GetAsuMaxIoSize(TransportOpType opType)
{
    switch (opType) {
        case TransportOpType::BATCH_LOAD: return kAsuBatchLoadMaxIoSize;
        case TransportOpType::BATCH_STORE: return kAsuBatchStoreMaxIoSize;
        case TransportOpType::DELETE: return kAsuDeleteMaxIoSize;
        case TransportOpType::QUERY: return kAsuQueryMaxIoSize;
        default: return 0;
    }
}

inline bool IsEntryBatchOp(TransportOpType opType)
{
    return opType == TransportOpType::BATCH_LOAD || opType == TransportOpType::BATCH_STORE;
}

inline bool IsKeyBatchOp(TransportOpType opType)
{
    return opType == TransportOpType::DELETE || opType == TransportOpType::QUERY;
}

template <typename T>
struct BatchView {
    const T* data{nullptr};
    std::size_t size{0};

    const T& operator[](std::size_t i) const noexcept { return data[i]; }
    bool empty() const noexcept { return size == 0; }
};

struct TransportSubBatchContext {
    std::uint16_t cid{0};
    TransportOpType opType{TransportOpType::QUERY};
    TransportSubBatchState state{TransportSubBatchState::PENDING};
    Status status{Status::OK()};
    ConnectionChannel* channel{nullptr};
    bool useSeekControl{false};
    ScatterGatherEntry sendSge;
    ScatterGatherEntry flagBuffer;
    std::vector<Status> entryStatus;
};

struct TransportTaskContext {
    TaskId taskId{kInvalidTaskId};
    TransportOpType opType{TransportOpType::QUERY};
    BatchView<CacheKey> keys;
    BatchView<KVBuffer> entries;
    QueryOptions queryOptions;
    QueryResult queryResult;
    std::vector<Status> entryStatus;
    std::vector<TransportSubBatchContext> subBatchContexts;
    std::uint32_t completedSubBatchCount{0};

    std::atomic<TransportTaskState> state{TransportTaskState::PENDING};
    Status finalStatus{Status::OK()};

    std::vector<MRHandle> mrHandles;

    std::mutex waitMu;
    std::condition_variable cv;

    bool Done() const;
    void InitializeTerminalSubBatchCount();
    void TryFinalizeFromSubBatches();
};

class TransportTaskManager : public TaskManagerBase<TransportTaskContext, TransportTaskState> {
public:
    TransportTaskManager() : TaskManagerBase(TransportTaskState::PENDING, "transport") {}

    void CancelAll();
};

}  // namespace UC::ASU
