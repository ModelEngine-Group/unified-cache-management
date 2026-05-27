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
#include <memory>
#include <mutex>
#include <vector>
#include "asu_transport/types.h"
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
    FAILED = 3,
    CANCELED = 4,
};

template <typename T>
struct BatchView {
    const T* data{nullptr};
    std::size_t size{0};

    const T& operator[](std::size_t i) const noexcept { return data[i]; }
    bool empty() const noexcept { return size == 0; }
};

struct TransportTaskContext {
    TaskId taskId{kInvalidTaskId};
    TransportOpType opType{TransportOpType::QUERY};
    BatchView<CacheKey> keys;
    BatchView<KVBuffer> entries;
    QueryOptions queryOptions;
    QueryResult queryResult;
    std::vector<Status> entryStatus;

    std::atomic<TransportTaskState> state{TransportTaskState::PENDING};
    Status finalStatus{Status::OK()};

    std::vector<MRHandle> mrHandles;

    std::uint32_t notifyIdx{0};
    volatile uint32_t* flagBuffer{nullptr};  // NPU 注册的内存地址

    std::mutex waitMu;
    std::condition_variable cv;

    bool Done() const;
};

class TransportTaskManager : public TaskManagerBase<TransportTaskContext, TransportTaskState> {
public:
    TransportTaskManager() : TaskManagerBase(TransportTaskState::PENDING, "transport") {}

    void Shutdown();
};

}  // namespace UC::ASU
