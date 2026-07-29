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
    Status status{Status::OK()};

    // TODO: optimize by zero-copy ?
    std::vector<KVBuffer> entries;
    std::vector<CacheKey> keys;
    std::vector<std::size_t> originalIndices;
};

enum class ClientTaskState {
    PENDING = 0,
    INFLIGHT = 1,
    COMPLETED = 2,
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
    std::vector<KVBuffer> entries;
    std::vector<CacheKey> keys;
    std::vector<ClientSubTask> subTasks;
    std::vector<Status> entryStatus;

    std::atomic<std::size_t> remainingSubTasks{0};
    std::atomic<ClientTaskState> state{ClientTaskState::PENDING};
    Status finalStatus{Status::OK()};

    std::mutex waitMu;
    std::condition_variable cv;

    bool Done() const;
    bool AllSubTasksCompleted() const;
};

using ClientTaskContextPtr = std::shared_ptr<ClientTaskContext>;

class ClientTaskManager : public TaskManagerBase<ClientTaskContext, ClientTaskState> {
public:
    ClientTaskManager() : TaskManagerBase(ClientTaskState::PENDING, "client") {}

    Status Check(TaskId taskId, TaskResult& result);
    Status Wait(TaskId taskId, std::uint64_t waitTimeoutMs, TaskResult& result);
    Status Drain(std::uint64_t waitTimeoutMs);
    Status Process(const ClientTaskContextPtr& task);

    static void CompleteWithError(const ClientTaskContextPtr& task, const Status& status);
    static void CompleteSubTask(const ClientTaskContextPtr& task, std::size_t subTaskIndex,
                                TaskResult result);
    static void CompleteUndispatchedSubTasks(const ClientTaskContextPtr& task,
                                             std::size_t firstSubTaskIndex,
                                             const Status& dispatchStatus);
    static void Finalize(const ClientTaskContextPtr& task);

private:
    static Status BuildSubTasks(const ClientTaskContextPtr& task);
    static Status DispatchTask(const ClientTaskContextPtr& task);
    static Status BuildResult(const ClientTaskContextPtr& task, TaskResult& result);
    static Status WaitContext(const ClientTaskContextPtr& task, std::uint64_t waitTimeoutMs,
                              TaskResult& result);
};

}  // namespace UC::ASU
