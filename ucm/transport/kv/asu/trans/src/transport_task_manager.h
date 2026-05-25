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
#include <mutex>
#include <vector>
#include "asu_transport/types.h"
#include "task_manager_base.h"

namespace UC::ASU {

enum class TransportOpType {
    QUERY = 0,
    LOAD = 1,
    STORE = 2,
    DELETE = 3,
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
    TaskId task_id{kInvalidTaskId};
    TransportOpType op_type{TransportOpType::QUERY};
    BatchView<CacheKey> keys;
    BatchView<KVBuffer> entries;
    QueryOptions query_options;
    QueryResult query_result;
    std::vector<Status> entry_status;

    std::atomic<TransportTaskState> state{TransportTaskState::PENDING};
    Status final_status{StatusCode::OK};

    std::vector<MRHandle> mr_handles;

    std::mutex wait_mu;
    std::condition_variable cv;

    bool Done() const
    {
        auto s = state.load(std::memory_order_acquire);
        return s == TransportTaskState::COMPLETED || s == TransportTaskState::FAILED ||
               s == TransportTaskState::CANCELED;
    }
};

class TransportTaskManager : public TaskManagerBase<TransportTaskContext, TransportTaskState> {
public:
    TransportTaskManager() : TaskManagerBase(TransportTaskState::PENDING, "transport") {}
};

}  // namespace UC::ASU
