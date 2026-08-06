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
#include "task_manager.h"
#include <algorithm>
#include <fmt/format.h>
#include <string>
#include <system_error>
#include "kv_common/router.h"
#include "logger/logger.h"

namespace UC::Dram {

TaskManager::TaskManager(TaskManagerConfig config, TaskManagerDependencies dependencies)
    : config_(std::move(config)),
      dependencies_(std::move(dependencies)),
      submissions_(config_.maxIoEntries),
      completions_(config_.maxIoEntries)
{
}

TaskManager::~TaskManager() { Shutdown(); }

TaskId TaskManager::AllocateTaskIdLocked() noexcept { return nextTaskId_++; }

Status TaskManager::Start()
{
    std::lock_guard lock(workMutex_);
    accepting_ = true;
    try {
        worker_ = std::thread([this] { Run(); });
        return Status::OK();
    } catch (const std::system_error& error) {
        accepting_ = false;
        return Status::Error(fmt::format("failed to start TaskManager: {}", error.what()));
    }
}

void TaskManager::Shutdown()
{
    {
        std::lock_guard lock(workMutex_);
        accepting_ = false;
    }
    workReady_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

Expected<TaskId> TaskManager::SubmitLookup(const Detail::BlockId* blocks, std::size_t num)
{
    std::vector<Detail::BlockId> lookup(blocks, blocks + num);
    return EnqueueTask(OpType::LOOKUP, std::move(lookup));
}

Expected<TaskId> TaskManager::SubmitTransfer(OpType op, Detail::TaskDesc task)
{
    return EnqueueTask(op, std::move(task));
}

Expected<TaskId> TaskManager::EnqueueTask(OpType op, TaskInput input)
{
    std::promise<TaskResult> promise;
    auto future = promise.get_future();
    const auto timeout = op == OpType::LOOKUP ? config_.timeouts.lookup
                         : op == OpType::DUMP ? config_.timeouts.dump
                                              : config_.timeouts.load;
    const auto deadline = Clock::now() + timeout;

    TaskId taskId = 0;
    {
        std::lock_guard lock(taskMutex_);
        taskId = AllocateTaskIdLocked();
        taskResults_.emplace(taskId, std::move(future));
    }

    Submission submission{taskId, op, deadline, std::move(input), std::move(promise)};
    auto enqueued = Status::OK();
    {
        std::lock_guard lock(workMutex_);
        if (!accepting_) {
            enqueued = Status::Error("TaskManager is stopping");
        } else if (!submissions_.Push(submission)) {
            enqueued = Status::NoSpace();
        }
    }
    if (enqueued.Failure()) {
        std::lock_guard lock(taskMutex_);
        taskResults_.erase(taskId);
        return enqueued;
    }

    workReady_.notify_one();
    return taskId;
}

Expected<bool> TaskManager::Check(TaskId taskId)
{
    std::lock_guard lock(taskMutex_);
    const auto found = taskResults_.find(taskId);
    if (found == taskResults_.end()) { return Status::NotFound(); }
    return found->second.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
}

Expected<TaskManager::TaskResult> TaskManager::WaitResult(TaskId taskId)
{
    std::future<TaskResult> future;
    {
        std::lock_guard lock(taskMutex_);
        const auto found = taskResults_.find(taskId);
        if (found == taskResults_.end()) { return Status::NotFound(); }
        future = std::move(found->second);
        taskResults_.erase(found);
    }
    try {
        return future.get();
    } catch (...) {
        return Status::Error("TaskManager task result invariant violated");
    }
}

Expected<std::vector<std::uint8_t>> TaskManager::WaitLookup(TaskId taskId)
{
    auto waited = WaitResult(taskId);
    if (!waited) { return waited.Error(); }
    auto result = std::move(waited).Value();
    if (result.status.Failure()) { return result.status; }
    return std::move(result.lookupResults);
}

Status TaskManager::WaitTransfer(TaskId taskId)
{
    auto waited = WaitResult(taskId);
    return waited ? std::move(waited).Value().status : waited.Error();
}

void TaskManager::Publish(std::vector<RequestCompleted>& events)
{
    if (events.empty()) { return; }
    {
        std::lock_guard lock(workMutex_);
        if (!accepting_) { return; }
        for (auto& completion : events) { completions_.Push(completion); }
    }
    workReady_.notify_one();
}

std::vector<IoEntry> TaskManager::NormalizeLookup(std::vector<Detail::BlockId> blocks) const
{
    std::vector<IoEntry> entries;
    entries.reserve(blocks.size());
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        IoEntry entry;
        entry.blockId = std::move(blocks[index]);
        entry.originalIndex = index;
        entries.push_back(std::move(entry));
    }
    return entries;
}

std::vector<IoEntry> TaskManager::NormalizeTransfer(const Detail::TaskDesc& task) const
{
    const auto tensorCount = config_.tensorSizes.size();
    std::vector<IoEntry> entries;
    entries.reserve(task.size() * tensorCount);
    for (const auto& shard : task) {
        for (std::size_t tensorIndex = 0; tensorIndex < tensorCount; ++tensorIndex) {
            IoEntry entry;
            entry.blockId = shard.owner;
            entry.shardId = static_cast<std::uint32_t>(shard.index * tensorCount + tensorIndex);
            entry.buffer = BufferRef{reinterpret_cast<std::uintptr_t>(shard.addrs[tensorIndex]),
                                     config_.tensorSizes[tensorIndex]};
            entries.push_back(std::move(entry));
        }
    }
    return entries;
}

std::vector<Request> TaskManager::BuildRequests(OpType op, std::vector<IoEntry> entries,
                                                TimePoint deadline) const
{
    std::vector<std::string> keys;
    keys.reserve(entries.size());
    for (const auto& entry : entries) {
        keys.emplace_back(reinterpret_cast<const char*>(entry.blockId.data()),
                          entry.blockId.size());
    }

    const auto routed = dependencies_.router->RouteKeys(keys);
    const auto batchLimit = std::min(config_.requestBatchEntries, kMaxProtocolBatchEntries);
    std::vector<Request> requests;
    requests.reserve(routed.size());
    for (const auto& [node, indexes] : routed) {
        for (std::size_t begin = 0; begin < indexes.size(); begin += batchLimit) {
            const auto end = std::min(indexes.size(), begin + batchLimit);
            Request request;
            request.nodeId = node;
            request.op = op;
            request.entries.reserve(end - begin);
            for (auto offset = begin; offset < end; ++offset) {
                request.entries.push_back(std::move(entries[indexes[offset]]));
            }
            request.deadline = deadline;
            requests.push_back(std::move(request));
        }
    }
    return requests;
}

void TaskManager::ProcessSubmission(Submission submission)
{
    if (submission.deadline <= Clock::now()) {
        submission.promise.set_value(TaskResult{Status::Timeout(), {}});
        return;
    }

    auto entries =
        std::holds_alternative<Detail::TaskDesc>(submission.input)
            ? NormalizeTransfer(std::get<Detail::TaskDesc>(submission.input))
            : NormalizeLookup(std::get<std::vector<Detail::BlockId>>(std::move(submission.input)));
    const auto entryCount = entries.size();
    if (entryCount > config_.maxIoEntries - usedIoEntries_) {
        submission.promise.set_value(TaskResult{Status::NoSpace(), {}});
        return;
    }
    auto requests = BuildRequests(submission.op, std::move(entries), submission.deadline);
    usedIoEntries_ += entryCount;

    ActiveTask task;
    task.op = submission.op;
    task.remainingRequests = requests.size();
    task.entryCount = entryCount;
    task.promise = std::move(submission.promise);
    if (task.op == OpType::LOOKUP) { task.lookupResults.resize(entryCount); }

    for (auto& request : requests) {
        request.taskId = submission.taskId;
        request.requestId = nextRequestId_++;
    }

    const auto taskId = submission.taskId;
    activeTasks_.emplace(taskId, std::move(task));

    for (auto& request : requests) {
        const auto status = dependencies_.submitRequest(request);
        if (status.Failure()) { CompleteRequest(taskId, status); }
    }
}

void TaskManager::ApplyLookupResults(ActiveTask& task,
                                     const std::vector<EntryResult>& results) const
{
    for (const auto& result : results) {
        task.lookupResults[result.originalIndex] = static_cast<std::uint8_t>(result.found);
    }
}

void TaskManager::CompleteRequest(TaskId taskId, Status status, std::vector<EntryResult> results)
{
    const auto found = activeTasks_.find(taskId);
    if (found == activeTasks_.end()) { return; }
    auto& task = found->second;
    --task.remainingRequests;

    if (status.Failure()) {
        if (!task.failure.has_value()) { task.failure.emplace(std::move(status)); }
    } else if (!task.failure.has_value() && task.op == OpType::LOOKUP) {
        ApplyLookupResults(task, results);
    }
    if (task.remainingRequests != 0) { return; }

    auto promise = std::move(task.promise);
    auto taskStatus = task.failure.has_value() ? std::move(*task.failure) : Status::OK();
    auto lookupResults =
        taskStatus.Success() ? std::move(task.lookupResults) : std::vector<std::uint8_t>{};
    usedIoEntries_ -= task.entryCount;
    activeTasks_.erase(found);
    promise.set_value(TaskResult{std::move(taskStatus), std::move(lookupResults)});
}

void TaskManager::ProcessCompletion(RequestCompleted event)
{
    CompleteRequest(event.taskId, std::move(event.status), std::move(event.entryResults));
}

void TaskManager::Run() noexcept
{
    try {
        for (;;) {
            std::optional<Submission> submission;
            std::optional<RequestCompleted> completion;

            {
                std::unique_lock lock(workMutex_);
                const auto workReady = [this] {
                    return !accepting_ || !completions_.Empty() || !submissions_.Empty();
                };
                workReady_.wait(lock, workReady);

                if (!accepting_) { return; }
                if (!completions_.Empty()) {
                    completion.emplace(completions_.Pop());
                } else {
                    submission.emplace(submissions_.Pop());
                }
            }

            if (completion.has_value()) {
                ProcessCompletion(std::move(*completion));
            } else {
                ProcessSubmission(std::move(*submission));
            }
        }
    } catch (...) {
        UC_ERROR("TaskManager worker stopped unexpectedly");
        {
            std::lock_guard lock(workMutex_);
            accepting_ = false;
            while (!submissions_.Empty()) {
                auto submission = submissions_.Pop();
                submission.promise.set_value(
                    TaskResult{Status::Error("TaskManager stopped unexpectedly"), {}});
            }
        }
        for (auto& [taskId, task] : activeTasks_) {
            task.promise.set_value(
                TaskResult{Status::Error("TaskManager stopped unexpectedly"), {}});
        }
        activeTasks_.clear();
        usedIoEntries_ = 0;
    }
}

}  // namespace UC::Dram
