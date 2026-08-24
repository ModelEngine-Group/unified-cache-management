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
#ifndef UNIFIEDCACHE_DRAM_STORE_CC_TASK_MANAGER_H
#define UNIFIEDCACHE_DRAM_STORE_CC_TASK_MANAGER_H

#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>
#include "bounded_queue.h"
#include "messages.h"
#include "status/status.h"

namespace UC::KV {
class Router;
}

namespace UC::Dram {

struct TaskManagerConfig {
    std::vector<std::uint64_t> tensorSizes;
    std::size_t maxIoEntries{0};
    std::size_t requestBatchEntries{0};
    TimeoutConfig timeouts;
};

struct TaskManagerDependencies {
    std::shared_ptr<const UC::KV::Router> router;
    RequestSubmitter submitRequest;
};

// A single-worker task actor. Submission planning and completion aggregation are
// serialized by the worker; public methods only own admission and task observation.
class TaskManager final {
public:
    TaskManager(TaskManagerConfig config, TaskManagerDependencies dependencies);
    ~TaskManager();

    TaskManager(const TaskManager&) = delete;
    TaskManager& operator=(const TaskManager&) = delete;

    Status Start();
    void Shutdown();

    Expected<TaskId> SubmitLookup(const Detail::BlockId* blocks, std::size_t num);
    Expected<TaskId> SubmitTransfer(OpType op, Detail::TaskDesc task);
    Expected<bool> Check(TaskId taskId);
    Expected<std::vector<std::uint8_t>> WaitLookup(TaskId taskId);
    Status WaitTransfer(TaskId taskId);

    // Completions are reliable while running and are discarded after shutdown begins.
    void Publish(std::vector<RequestCompleted>& events);

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using TaskInput = std::variant<Detail::TaskDesc, std::vector<Detail::BlockId>>;

    struct TaskResult {
        Status status;
        std::vector<std::uint8_t> lookupResults;
    };

    struct TaskTiming {
        std::uint64_t enqueuedUs{0};
        std::uint64_t processSubmissionStartedUs{0};
        std::uint64_t requestsStartedUs{0};
        std::uint64_t completedUs{0};
        std::uint64_t enqueuedTsUs{0};
        std::uint64_t requestsStartedTsUs{0};
        std::uint64_t completedTsUs{0};
    };

    struct Submission {
        TaskId taskId{0};
        OpType op{OpType::LOOKUP};
        TimePoint deadline;
        TaskInput input;
        std::promise<TaskResult> promise;
        TaskTiming timing;
    };

    struct ActiveTask {
        OpType op{OpType::LOOKUP};
        std::size_t remainingRequests{0};
        std::size_t requestCount{0};
        std::size_t entryCount{0};
        std::optional<Status> failure;
        std::vector<std::uint8_t> lookupResults;
        std::promise<TaskResult> promise;
        TaskTiming timing;
    };

    TaskId AllocateTaskIdLocked() noexcept;
    Expected<TaskId> EnqueueTask(OpType op, TaskInput input);
    Expected<TaskResult> WaitResult(TaskId taskId);

    std::vector<IoEntry> NormalizeLookup(std::vector<Detail::BlockId> blocks) const;
    std::vector<IoEntry> NormalizeTransfer(const Detail::TaskDesc& task) const;
    std::vector<Request> BuildRequests(OpType op, std::vector<IoEntry> entries,
                                       TimePoint deadline) const;

    void Run() noexcept;
    void ProcessSubmission(Submission submission);
    void ProcessCompletion(RequestCompleted event);
    void CompleteRequest(TaskId taskId, Status status, std::vector<EntryResult> results = {});
    void ApplyLookupResults(ActiveTask& task, const std::vector<EntryResult>& results) const;
    static void LogTaskDone(TaskId taskId, OpType op, std::size_t entryCount,
                            std::size_t requestCount, const Status& status, TaskTiming timing);

    TaskManagerConfig config_;
    TaskManagerDependencies dependencies_;

    std::mutex taskMutex_;
    std::unordered_map<TaskId, std::future<TaskResult>> taskResults_;
    TaskId nextTaskId_{1};

    std::mutex workMutex_;
    std::condition_variable workReady_;
    bool accepting_{false};
    BoundedQueue<Submission> submissions_;
    BoundedQueue<RequestCompleted> completions_;

    std::thread worker_;

    // Worker-only execution state.
    std::unordered_map<TaskId, ActiveTask> activeTasks_;
    std::size_t usedIoEntries_{0};
    RequestId nextRequestId_{1};
};

}  // namespace UC::Dram

#endif  // UNIFIEDCACHE_DRAM_STORE_CC_TASK_MANAGER_H
