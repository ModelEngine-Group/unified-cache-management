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
#include <chrono>
#include <string>
#include <utility>
#include "kv_client_impl.h"
#include "logger.h"
#include "router/router.h"

namespace kv {

namespace {

const char* AsuOpTypeName(AsuOpType opType)
{
    switch (opType) {
        case AsuOpType::QUERY: return "query";
        case AsuOpType::LOAD: return "load";
        case AsuOpType::STORE: return "store";
        case AsuOpType::BATCH_LOAD: return "batch_load";
        case AsuOpType::BATCH_STORE: return "batch_store";
        case AsuOpType::DELETE: return "delete";
        case AsuOpType::KEEP_ALIVE: return "keep_alive";
        default: return "unknown";
    }
}

std::vector<kv::Key> ToRouterKeys(const std::vector<CacheKey>& keys)
{
    std::vector<kv::Key> routerKeys;
    routerKeys.reserve(keys.size());
    for (const auto& key : keys) { routerKeys.emplace_back(std::string(CacheKeyView(key))); }
    return routerKeys;
}

std::vector<kv::Key> ExtractEntryKeys(const std::vector<KVBuffer>& entries)
{
    std::vector<kv::Key> keys;
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

bool ClientTask::Done() const
{
    return state.load(std::memory_order_acquire) == ClientTaskState::COMPLETED;
}

bool ClientTask::AllTransportTasksCompleted() const
{
    return remainingTransportTasks.load(std::memory_order_acquire) == 0;
}

bool ClientTaskManager::Check(TaskId taskId)
{
    auto task = Get(taskId);
    return !task || task->Done();
}

Status ClientTaskManager::Wait(TaskId taskId, std::uint64_t waitTimeoutMs, TaskResult& result)
{
    auto task = Get(taskId);
    if (!task) { return Status::Error(StatusCode::TASK_NOT_FOUND, "task not found"); }

    auto status = WaitContext(task, waitTimeoutMs, result);
    (void)Remove(taskId);
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

Status ClientTaskManager::Process(const ClientTaskPtr& task)
{
    if (!task) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "client task context is null");
    }
    task->state.store(ClientTaskState::INFLIGHT, std::memory_order_release);

    auto status = BuildTransportTasks(task);
    if (!status.ok()) {
        CompleteWithError(task, status);
        return status;
    }
    return DispatchTask(task);
}

void ClientTaskManager::CompleteWithError(const ClientTaskPtr& task, const Status& status)
{
    std::lock_guard<std::mutex> lock{task->waitMu};
    std::fill(task->entryStatus.begin(), task->entryStatus.end(), status);
    task->finalStatus = status;
    KV_ERROR("ASU client task failed: client_task_id={} op={} code={} message={}.", task->taskId,
             AsuOpTypeName(task->opType), static_cast<int>(status.code), status.message);
    task->state.store(ClientTaskState::COMPLETED, std::memory_order_release);
    task->cv.notify_all();
}

void ClientTaskManager::CompleteTransportTask(const ClientTaskPtr& task,
                                              std::size_t transportTaskIndex, TaskResult result)
{
    std::lock_guard<std::mutex> lock(task->waitMu);
    auto& transportTask = task->transportTasks[transportTaskIndex];
    if (transportTask->clientCompleted) { return; }

    auto completionStatus = result.status;
    bool invalidQueryResult = false;
    if (task->opType == AsuOpType::QUERY && completionStatus.ok()) {
        if (!result.queryResult.has_value()) {
            completionStatus =
                Status::Error(StatusCode::INTERNAL_ERROR, "transport query result is missing");
            invalidQueryResult = true;
        } else if (result.queryResult->exists.size() != transportTask->originalIndices.size()) {
            completionStatus =
                Status::Error(StatusCode::INTERNAL_ERROR, "transport query result size mismatch");
            invalidQueryResult = true;
        } else {
            for (std::size_t index = 0; index < transportTask->originalIndices.size(); ++index) {
                task->queryResult.exists[transportTask->originalIndices[index]] =
                    result.queryResult->exists[index];
            }
        }
    }

    transportTask->clientCompleted = true;
    transportTask->finalStatus = completionStatus;
    for (std::size_t index = 0; index < transportTask->originalIndices.size(); ++index) {
        task->entryStatus[transportTask->originalIndices[index]] =
            !invalidQueryResult && index < result.entryStatus.size() ? result.entryStatus[index]
                                                                     : completionStatus;
    }

    if (task->remainingTransportTasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        Finalize(task);
    }
}

void ClientTaskManager::CompleteUndispatchedTransportTasks(const ClientTaskPtr& task,
                                                           std::size_t firstTransportTaskIndex,
                                                           const Status& dispatchStatus)
{
    std::lock_guard<std::mutex> lock(task->waitMu);
    for (std::size_t index = firstTransportTaskIndex; index < task->transportTasks.size();
         ++index) {
        auto& failedTask = task->transportTasks[index];
        failedTask->clientCompleted = true;
        failedTask->finalStatus =
            index == firstTransportTaskIndex
                ? dispatchStatus
                : Status::Error(StatusCode::CANCELED,
                                "transport task not dispatched after a dispatch failure");
        for (auto originalIndex : failedTask->originalIndices) {
            task->entryStatus[originalIndex] = failedTask->finalStatus;
        }
        task->remainingTransportTasks.fetch_sub(1, std::memory_order_acq_rel);
    }

    if (task->AllTransportTasksCompleted()) { Finalize(task); }
}

void ClientTaskManager::Finalize(const ClientTaskPtr& task)
{
    if (task->opType == AsuOpType::QUERY) {
        task->queryResult.prefixHitKeys = 0;
        for (auto exists : task->queryResult.exists) {
            if (exists == 0) { break; }
            ++task->queryResult.prefixHitKeys;
        }
    }

    std::size_t failedTransportTasks = 0;
    for (const auto& transportTask : task->transportTasks) {
        if (transportTask->finalStatus.ok()) { continue; }

        ++failedTransportTasks;
        const auto itemCount = transportTask->entries.empty() ? transportTask->keys.size()
                                                              : transportTask->entries.size();
        KV_ERROR(
            "ASU client transport task failed: client_task_id={} op={} asuId={} "
            "transport_task_id={} item_count={} code={} message={}.",
            task->taskId, AsuOpTypeName(task->opType), transportTask->asuId, transportTask->taskId,
            itemCount, static_cast<int>(transportTask->finalStatus.code),
            transportTask->finalStatus.message);
    }
    task->finalStatus = failedTransportTasks == 0 ? Status::OK()
                                                  : Status::Error(StatusCode::PARTIAL_FAILED,
                                                                  "client task partially failed");
    if (task->finalStatus.ok()) {
        KV_DEBUG("ASU client task completed: client_task_id={} op={} transport_tasks={}.",
                 task->taskId, AsuOpTypeName(task->opType), task->transportTasks.size());
    }
    task->state.store(ClientTaskState::COMPLETED, std::memory_order_release);
    task->cv.notify_all();
}

Status ClientTaskManager::BuildTransportTasks(const ClientTaskPtr& task)
{
    auto snapshot = task->viewSnapshot;
    if (!snapshot || !snapshot->router || snapshot->transports.empty()) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    const auto routes = task->opType == AsuOpType::QUERY || task->opType == AsuOpType::DELETE
                            ? snapshot->router->RouteKeys(ToRouterKeys(task->keys))
                            : snapshot->router->RouteKeys(ExtractEntryKeys(task->entries));
    for (const auto& [asuId, indices] : routes) {
        if (snapshot->transports.find(asuId) == snapshot->transports.end()) {
            return AddContext(
                Status::Error(StatusCode::NOT_FOUND, "routed asu transport not found"),
                "asuId=" + std::to_string(asuId));
        }
    }

    task->transportTasks.reserve(routes.size());
    for (const auto& [asuId, indices] : routes) {
        auto transportTask = std::make_shared<TransportTask>();
        transportTask->asuId = asuId;
        transportTask->transport = snapshot->transports.at(asuId);
        transportTask->originalIndices.reserve(indices.size());
        if (task->opType == AsuOpType::QUERY || task->opType == AsuOpType::DELETE) {
            transportTask->keys.reserve(indices.size());
            for (auto index : indices) {
                transportTask->keys.push_back(std::move(task->keys[index]));
                transportTask->originalIndices.push_back(index);
            }
        } else {
            transportTask->entries.reserve(indices.size());
            for (auto index : indices) {
                transportTask->entries.push_back(std::move(task->entries[index]));
                transportTask->originalIndices.push_back(index);
            }
        }
        task->transportTasks.push_back(std::move(transportTask));
    }
    std::vector<KVBuffer>{}.swap(task->entries);
    std::vector<CacheKey>{}.swap(task->keys);
    task->remainingTransportTasks.store(task->transportTasks.size(), std::memory_order_release);
    return Status::OK();
}

Status ClientTaskManager::DispatchTask(const ClientTaskPtr& task)
{
    if (task->transportTasks.empty()) {
        std::lock_guard<std::mutex> lock(task->waitMu);
        Finalize(task);
        return Status::OK();
    }

    for (std::size_t taskIndex = 0; taskIndex < task->transportTasks.size(); ++taskIndex) {
        auto& transportTask = task->transportTasks[taskIndex];
        auto transport = transportTask->transport.lock();

        std::weak_ptr<ClientTask> clientTask = task;
        transportTask->onComplete = [clientTask, taskIndex](TaskResult result) {
            auto task = clientTask.lock();
            if (!task) { return; }
            CompleteTransportTask(task, taskIndex, std::move(result));
        };
        transportTask->opType = task->opType;
        auto status = transport->Submit(transportTask);
        if (!status.ok()) {
            const auto dispatchStatus =
                AddContext(status, "asuId=" + std::to_string(transportTask->asuId));
            CompleteUndispatchedTransportTasks(task, taskIndex, dispatchStatus);
            return dispatchStatus;
        }
    }
    return Status::OK();
}

Status ClientTaskManager::BuildResult(const ClientTaskPtr& task, TaskResult& result)
{
    result.status = task->Done()
                        ? task->finalStatus
                        : Status::Error(StatusCode::IN_PROGRESS, "client task in progress");
    result.entryStatus = task->entryStatus;
    if (task->opType == AsuOpType::QUERY) {
        result.queryResult = task->queryResult;
    } else {
        result.queryResult.reset();
    }
    return result.status;
}

Status ClientTaskManager::WaitContext(const ClientTaskPtr& task, std::uint64_t waitTimeoutMs,
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
            "client task wait timeout: client_task_id=" + std::to_string(task->taskId) +
                " op=" + AsuOpTypeName(task->opType) + " wait_ms=" + std::to_string(waitTimeoutMs));
        KV_ERROR("ASU client task wait timeout: client_task_id={} op={} wait_ms={}.", task->taskId,
                 AsuOpTypeName(task->opType), waitTimeoutMs);
    }
    return result.status;
}

}  // namespace kv
