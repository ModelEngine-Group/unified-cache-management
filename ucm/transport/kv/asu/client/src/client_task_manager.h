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

// ViewSnapshot is owned by the client and captured by submitted tasks.
struct ViewSnapshot;

struct ClientSubTask {
    AsuId asuId;
    TaskId transTaskId{kInvalidTaskId};
    bool completed{false};
    bool failed{false};

    // TODO: optimize by zero-copy ?
    std::vector<KVBuffer> entries;
    std::vector<CacheKey> keys;
    std::vector<std::size_t> originalIndices;
};

enum class ClientTaskState {
    PENDING = 0,
    INFLIGHT = 1,
    COMPLETED = 2,
    FAILED = 3,
    CANCELED = 4,
};

enum class ClientOpType {
    LOAD = 0,
    STORE = 1,
    DELETE = 2,
};

struct ClientTaskContext {
    TaskId taskId{kInvalidTaskId};
    ClientOpType opType{ClientOpType::LOAD};
    std::shared_ptr<ViewSnapshot> viewSnapshot;
    std::vector<ClientSubTask> subTasks;
    std::vector<Status> entryStatus;

    std::atomic<ClientTaskState> state{ClientTaskState::PENDING};
    Status finalStatus{Status::OK()};

    // TODO: move to transport completion-driven notification when transport exposes callbacks.
    std::mutex waitMu;
    std::condition_variable cv;

    bool Done() const
    {
        auto s = state.load(std::memory_order_acquire);
        return s == ClientTaskState::COMPLETED || s == ClientTaskState::FAILED ||
               s == ClientTaskState::CANCELED;
    }
};

class ClientTaskManager : public TaskManagerBase<ClientTaskContext, ClientTaskState> {
public:
    ClientTaskManager() : TaskManagerBase(ClientTaskState::PENDING, "client") {}
};

}  // namespace UC::ASU
