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
#include "client_task_manager.h"
#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include "asu_client_impl.h"
#include "kv_common/router.h"
#include "logger/logger.h"

namespace UC::ASU {

namespace {

const char* ClientOpTypeName(ClientOpType opType)
{
    switch (opType) {
        case ClientOpType::LOAD: return "load";
        case ClientOpType::STORE: return "store";
        case ClientOpType::DELETE: return "delete";
        default: return "unknown";
    }
}

std::size_t SubTaskItemCount(const ClientSubTask& subTask)
{
    return subTask.entries.empty() ? subTask.keys.size() : subTask.entries.size();
}

std::string SubTaskContext(const ClientTaskContext& task, const ClientSubTask& subTask)
{
    return "client_task_id=" + std::to_string(task.taskId) +
           " op=" + ClientOpTypeName(task.opType) + " asuId=" + std::to_string(subTask.asuId) +
           " trans_task_id=" + std::to_string(subTask.transTaskId) +
           " item_count=" + std::to_string(SubTaskItemCount(subTask));
}

std::string FirstFailedSubTaskContext(const ClientTaskContext& task)
{
    for (const auto& subTask : task.subTasks) {
        if (!subTask.failed) { continue; }

        return SubTaskContext(task, subTask) +
               " code=" + std::to_string(static_cast<int>(subTask.status.code)) +
               " message=" + subTask.status.message;
    }
    return "client_task_id=" + std::to_string(task.taskId) + " op=" + ClientOpTypeName(task.opType);
}

std::vector<UC::KV::CacheKey> ToRouterKeys(const std::vector<CacheKey>& keys)
{
    std::vector<UC::KV::CacheKey> routerKeys;
    routerKeys.reserve(keys.size());
    for (const auto& key : keys) { routerKeys.emplace_back(std::string(CacheKeyView(key))); }
    return routerKeys;
}

std::vector<UC::KV::CacheKey> ExtractEntryKeys(const std::vector<KVBuffer>& entries)
{
    std::vector<UC::KV::CacheKey> keys;
    keys.reserve(entries.size());
    for (const auto& entry : entries) { keys.emplace_back(std::string(CacheKeyView(entry.key))); }
    return keys;
}

Status AddContext(Status status, const std::string& context)
{
    if (context.empty()) { return status; }
    if (status.message.empty()) {
        status.message = context;
    } else {
        status.message += ", " + context;
    }
    return status;
}

}  // namespace

bool ClientTaskContext::Done() const
{
    return state.load(std::memory_order_acquire) == ClientTaskState::COMPLETED;
}

bool ClientTaskContext::AllSubTasksCompleted() const
{
    return remainingSubTasks.load(std::memory_order_acquire) == 0;
}

Status ClientTaskManager::Check(TaskId taskId, TaskResult& result)
{
    auto task = Get(taskId);
    if (!task) { return Status::Error(StatusCode::TASK_NOT_FOUND, "task not found"); }

    Status status;
    bool done = false;
    {
        std::lock_guard<std::mutex> lock(task->waitMu);
        status = BuildResult(task, result);
        done = task->Done();
    }
    if (done) { (void)Remove(taskId); }
    return status;
}

Status ClientTaskManager::Wait(TaskId taskId, std::uint64_t waitTimeoutMs, TaskResult& result)
{
    auto task = Get(taskId);
    if (!task) { return Status::Error(StatusCode::TASK_NOT_FOUND, "task not found"); }

    auto status = WaitContext(task, waitTimeoutMs, result);
    if (status.code != StatusCode::TIMEOUT) { (void)Remove(taskId); }
    return status;
}

Status ClientTaskManager::Drain(std::uint64_t waitTimeoutMs)
{
    Status finalStatus = Status::OK();
    for (const auto& task : GetAll()) {
        if (!task) { continue; }

        if (!task->Done()) {
            TaskResult result;
            auto status = WaitContext(task, waitTimeoutMs, result);
            if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
        }
        (void)Remove(task->taskId);
    }
    return finalStatus;
}

Status ClientTaskManager::Process(const ClientTaskContextPtr& task)
{
    if (!task) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "client task context is null");
    }
    task->state.store(ClientTaskState::INFLIGHT, std::memory_order_release);

    auto status = BuildSubTasks(task);
    if (!status.ok()) {
        CompleteWithError(task, status);
        return status;
    }
    return DispatchTask(task);
}

void ClientTaskManager::CompleteWithError(const ClientTaskContextPtr& task, const Status& status)
{
    std::lock_guard<std::mutex> lock{task->waitMu};
    std::fill(task->entryStatus.begin(), task->entryStatus.end(), status);
    task->finalStatus = status;
    task->state.store(ClientTaskState::COMPLETED, std::memory_order_release);
    task->cv.notify_all();
}

void ClientTaskManager::CompleteSubTask(const ClientTaskContextPtr& task, std::size_t subTaskIndex,
                                        TaskResult result)
{
    std::lock_guard<std::mutex> lock(task->waitMu);
    auto& subTask = task->subTasks[subTaskIndex];
    if (subTask.completed) { return; }

    subTask.completed = true;
    subTask.failed = !result.status.ok();
    subTask.status = result.status;
    for (std::size_t index = 0; index < subTask.originalIndices.size(); ++index) {
        task->entryStatus[subTask.originalIndices[index]] =
            index < result.entryStatus.size() ? result.entryStatus[index] : result.status;
    }

    if (task->remainingSubTasks.fetch_sub(1, std::memory_order_acq_rel) == 1) { Finalize(task); }
}

void ClientTaskManager::CompleteUndispatchedSubTasks(const ClientTaskContextPtr& task,
                                                     std::size_t firstSubTaskIndex,
                                                     const Status& dispatchStatus)
{
    std::lock_guard<std::mutex> lock(task->waitMu);
    for (std::size_t index = firstSubTaskIndex; index < task->subTasks.size(); ++index) {
        auto& failedSubTask = task->subTasks[index];
        failedSubTask.completed = true;
        failedSubTask.failed = true;
        failedSubTask.status =
            index == firstSubTaskIndex
                ? dispatchStatus
                : Status::Error(StatusCode::CANCELED,
                                "subtask not dispatched after a dispatch failure");
        for (auto originalIndex : failedSubTask.originalIndices) {
            task->entryStatus[originalIndex] = failedSubTask.status;
        }
        task->remainingSubTasks.fetch_sub(1, std::memory_order_acq_rel);
    }

    if (task->AllSubTasksCompleted()) { Finalize(task); }
}

void ClientTaskManager::Finalize(const ClientTaskContextPtr& task)
{
    const bool anyFailed = std::any_of(task->subTasks.begin(), task->subTasks.end(),
                                       [](const ClientSubTask& subTask) { return subTask.failed; });
    task->finalStatus =
        anyFailed ? Status::Error(StatusCode::PARTIAL_FAILED, "client task partially failed: " +
                                                                  FirstFailedSubTaskContext(*task))
                  : Status::OK();
    task->state.store(ClientTaskState::COMPLETED, std::memory_order_release);
    task->cv.notify_all();
}

Status ClientTaskManager::BuildSubTasks(const ClientTaskContextPtr& task)
{
    auto snapshot = task == nullptr ? nullptr : task->viewSnapshot;
    if (!snapshot || !snapshot->router || snapshot->transports.empty()) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    const auto routes = task->opType == ClientOpType::DELETE
                            ? snapshot->router->RouteKeys(ToRouterKeys(task->keys))
                            : snapshot->router->RouteKeys(ExtractEntryKeys(task->entries));
    for (const auto& route : routes) {
        if (snapshot->transports.find(route.first) == snapshot->transports.end()) {
            return AddContext(
                Status::Error(StatusCode::NOT_FOUND, "routed asu transport not found"),
                "asuId=" + std::to_string(route.first));
        }
    }

    task->subTasks.reserve(routes.size());
    for (const auto& route : routes) {
        ClientSubTask subTask;
        subTask.asuId = route.first;
        subTask.originalIndices.reserve(route.second.size());
        if (task->opType == ClientOpType::DELETE) {
            subTask.keys.reserve(route.second.size());
            for (auto index : route.second) {
                subTask.keys.push_back(std::move(task->keys[index]));
                subTask.originalIndices.push_back(index);
            }
        } else {
            subTask.entries.reserve(route.second.size());
            for (auto index : route.second) {
                subTask.entries.push_back(std::move(task->entries[index]));
                subTask.originalIndices.push_back(index);
            }
        }
        task->subTasks.push_back(std::move(subTask));
    }
    std::vector<KVBuffer>{}.swap(task->entries);
    std::vector<CacheKey>{}.swap(task->keys);
    task->remainingSubTasks.store(task->subTasks.size(), std::memory_order_release);
    return Status::OK();
}

Status ClientTaskManager::DispatchTask(const ClientTaskContextPtr& task)
{
    auto snapshot = task == nullptr ? nullptr : task->viewSnapshot;
    if (!snapshot) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "client view is not ready");
    }
    if (task->subTasks.empty()) {
        std::lock_guard<std::mutex> lock(task->waitMu);
        Finalize(task);
        return Status::OK();
    }

    for (std::size_t subTaskIndex = 0; subTaskIndex < task->subTasks.size(); ++subTaskIndex) {
        auto& subTask = task->subTasks[subTaskIndex];
        auto transIter = snapshot->transports.find(subTask.asuId);
        if (transIter == snapshot->transports.end()) {
            return Status::Error(StatusCode::NOT_FOUND, "routed ASU transport not found");
        }

        auto onComplete = [task, subTaskIndex](TaskResult result) {
            CompleteSubTask(task, subTaskIndex, std::move(result));
        };
        Status status;
        if (task->opType == ClientOpType::LOAD) {
            status = transIter->second->LoadAsync(subTask.entries, subTask.transTaskId,
                                                  std::move(onComplete));
        } else if (task->opType == ClientOpType::STORE) {
            status = transIter->second->StoreAsync(subTask.entries, subTask.transTaskId,
                                                   std::move(onComplete));
        } else {
            status = transIter->second->DeleteAsync(subTask.keys, subTask.transTaskId,
                                                    std::move(onComplete));
        }
        if (!status.ok()) {
            for (std::size_t index = 0; index < subTaskIndex; ++index) {
                auto& dispatchedSubTask = task->subTasks[index];
                if (dispatchedSubTask.transTaskId == kInvalidTaskId) { continue; }

                auto dispatchedTransIter = snapshot->transports.find(dispatchedSubTask.asuId);
                if (dispatchedTransIter == snapshot->transports.end()) { continue; }
                (void)dispatchedTransIter->second->Cancel(dispatchedSubTask.transTaskId);
            }

            const auto dispatchStatus =
                AddContext(status, "asuId=" + std::to_string(subTask.asuId));
            CompleteUndispatchedSubTasks(task, subTaskIndex, dispatchStatus);
            return dispatchStatus;
        }
    }
    return Status::OK();
}

Status ClientTaskManager::BuildResult(const ClientTaskContextPtr& task, TaskResult& result)
{
    result.status = task->Done()
                        ? task->finalStatus
                        : Status::Error(StatusCode::IN_PROGRESS, "client task in progress");
    result.entryStatus = task->entryStatus;
    result.queryResult.reset();
    return result.status;
}

Status ClientTaskManager::WaitContext(const ClientTaskContextPtr& task, std::uint64_t waitTimeoutMs,
                                      TaskResult& result)
{
    if (!task) { return Status::Error(StatusCode::TASK_NOT_FOUND, "client task not found"); }

    std::unique_lock<std::mutex> lock(task->waitMu);
    const bool done = task->cv.wait_for(lock, std::chrono::milliseconds(waitTimeoutMs),
                                        [task] { return task->Done(); });
    BuildResult(task, result);
    if (!done) {
        result.status = Status::Error(
            StatusCode::TIMEOUT,
            "client task wait timeout: client_task_id=" + std::to_string(task->taskId) + " op=" +
                ClientOpTypeName(task->opType) + " wait_ms=" + std::to_string(waitTimeoutMs));
        UC_ERROR("ASU client task wait timeout: client_task_id={} op={} wait_ms={}.", task->taskId,
                 ClientOpTypeName(task->opType), waitTimeoutMs);
    }
    return result.status;
}

}  // namespace UC::ASU
