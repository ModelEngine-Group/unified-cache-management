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
#include "asu_client_impl.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <thread>
#include <utility>
#include "asu_transport/types.h"
#include "client_config_parser.h"
#include "client_router_config.h"
#include "kv_common/router.h"
#include "logger/logger.h"

namespace UC::ASU {

constexpr std::uint32_t kMaxShutdownDrainAttempts = 64;

Status PartialFailed(const std::string& message)
{
    return Status::Error(StatusCode::PARTIAL_FAILED, message);
}

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

std::string SubTaskContext(const ClientTaskContext& ctx, const ClientSubTask& subTask)
{
    return "client_task_id=" + std::to_string(ctx.taskId) + " op=" + ClientOpTypeName(ctx.opType) +
           " asuId=" + std::to_string(subTask.asuId) +
           " trans_task_id=" + std::to_string(subTask.transTaskId) +
           " item_count=" + std::to_string(SubTaskItemCount(subTask));
}

std::string FirstFailedSubTaskContext(const ClientTaskContext& ctx)
{
    for (const auto& subTask : ctx.subTasks) {
        if (!subTask.failed) { continue; }

        return SubTaskContext(ctx, subTask) +
               " code=" + std::to_string(static_cast<int>(subTask.status.code)) +
               " message=" + subTask.status.message;
    }
    return "client_task_id=" + std::to_string(ctx.taskId) + " op=" + ClientOpTypeName(ctx.opType);
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

AsuClientImpl::AsuClientImpl(TransportFactory transportFactory, ViewServerFactory viewServerFactory)
    : transportFactory_(std::move(transportFactory)),
      viewServerFactory_(std::move(viewServerFactory))
{
    if (!transportFactory_) { transportFactory_ = CreateAsuTransport; }
    if (!viewServerFactory_) { viewServerFactory_ = CreateDefaultViewServer; }
}

AsuClientImpl::~AsuClientImpl() { Shutdown(); }

Status AsuClientImpl::Init(const std::string& configPath)
{
    AsuClientConfig config;
    auto status = LoadConfig(configPath, config);
    if (!status.ok()) { return status; }
    return Init(config);
}

Status AsuClientImpl::Init(const AsuClientConfig& config)
{
    if (initialized_) {
        return Status::Error(StatusCode::RESOURCE_BUSY, "asu client has already been initialized");
    }

    config_ = config;
    viewServer_ = viewServerFactory_(config);
    if (viewServer_ == nullptr) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "view server factory returned null");
    }
    transportConfigs_.clear();
    for (const auto& transportConfig : config.transportConfigs) {
        transportConfigs_[transportConfig.asuId] = transportConfig;
    }

    GlobalView view;
    auto status = viewServer_->GetGlobalView(view);
    if (!status.ok()) { return status; }

    std::shared_ptr<ViewSnapshot> nextSnapshot;
    status = BuildSnapshot(view, nullptr, nextSnapshot);
    if (!status.ok()) { return status; }

    {
        std::lock_guard<std::mutex> lock{taskQueueMu_};
        stopWorker_ = false;
    }
    worker_ = std::thread(&AsuClientImpl::WorkerLoop, this);
    snapshot_ = std::move(nextSnapshot);
    initialized_ = true;
    return Status::OK();
}

Status AsuClientImpl::Shutdown()
{
    std::uint64_t waitTimeoutMs = 0;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        initialized_ = false;
        waitTimeoutMs = config_.defaultWaitTimeoutMs;
    }
    JoinBackgroundRefresh();

    {
        std::lock_guard<std::mutex> lock{taskQueueMu_};
        stopWorker_ = true;
    }
    taskQueueCv_.notify_all();
    if (worker_.joinable()) { worker_.join(); }

    std::shared_ptr<ViewSnapshot> snapshot;
    std::vector<std::shared_ptr<AsuTransport>> retiredTransports;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        snapshot = std::move(snapshot_);
        retiredTransports = std::move(retiredTransports_);
        config_ = AsuClientConfig{};
        viewServer_.reset();
        transportConfigs_.clear();
        registeredRegions_.clear();
    }

    Status finalStatus = Status::OK();
    auto drainStatus = DrainTasksBeforeShutdown(waitTimeoutMs);
    if (!drainStatus.ok()) { finalStatus = drainStatus; }
    if (snapshot) {
        auto shutdownStatus = ShutdownSnapshotTransports(snapshot);
        if (!shutdownStatus.ok() && finalStatus.ok()) { finalStatus = shutdownStatus; }
    }
    for (auto& transport : retiredTransports) {
        if (transport == nullptr) { continue; }
        auto status = transport->Shutdown();
        if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
    }
    return finalStatus;
}

Status AsuClientImpl::Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                            QueryResult& result)
{
    bool needRefresh = false;
    auto status = QueryOnce(keys, options, result, needRefresh);
    if (needRefresh) { RequestBackgroundRefresh(); }
    return status;
}

Status AsuClientImpl::QueryOnce(const std::vector<CacheKey>& keys, const QueryOptions& options,
                                QueryResult& result, bool& needRefresh)
{
    result.exists.assign(keys.size(), 0);
    result.prefixHitKeys = 0;

    auto snapshot = GetSnapshot();
    if (!snapshot) { return NotInitialized(); }

    const auto timeoutMs =
        options.timeoutMs == 0 ? config_.defaultWaitTimeoutMs : options.timeoutMs;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    auto transportOptions = options;
    transportOptions.mode = QueryMode::PER_KEY;
    auto routes = snapshot->router->RouteKeys(ToRouterKeys(keys));
    std::vector<PendingQuery> pendingQueries;
    pendingQueries.reserve(routes.size());
    bool anyFailed = false;

    for (const auto& route : routes) {
        auto transportIter = snapshot->transports.find(route.first);
        if (transportIter == snapshot->transports.end()) {
            auto status = Status::Error(StatusCode::NOT_FOUND, "routed asu transport not found");
            MarkRefreshIfNeeded(status, needRefresh);
            UC_ERROR("ASU client query dispatch failed: asuId={} key_count={} code={} message={}.",
                     route.first, route.second.size(), static_cast<int>(status.code),
                     status.message);
            anyFailed = true;
            continue;
        }

        PendingQuery pending;
        pending.asuId = route.first;
        pending.transport = transportIter->second;
        pending.originalIndices = route.second;
        pending.keys.reserve(route.second.size());
        for (auto index : route.second) { pending.keys.emplace_back(keys[index]); }

        auto status = pending.transport->QueryAsync(pending.keys, transportOptions, pending.taskId);
        if (!status.ok()) {
            MarkRefreshIfNeeded(status, needRefresh);
            UC_ERROR("ASU client query dispatch failed: asuId={} key_count={} code={} message={}.",
                     pending.asuId, pending.keys.size(), static_cast<int>(status.code),
                     status.message);
            anyFailed = true;
            continue;
        }

        pendingQueries.emplace_back(std::move(pending));
    }

    for (auto& pending : pendingQueries) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            UC_ERROR(
                "ASU client query wait timed out before transport wait: asuId={} key_count={}.",
                pending.asuId, pending.keys.size());
            anyFailed = true;
            continue;
        }

        const auto remainingMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const auto waitMs = static_cast<std::uint64_t>(std::max<std::int64_t>(1, remainingMs));
        TaskResult taskResult;
        auto status = pending.transport->Wait(pending.taskId, waitMs, taskResult);
        if (!status.ok()) {
            MarkRefreshIfNeeded(status, needRefresh);
            UC_ERROR("ASU client query wait failed: asuId={} key_count={} code={} message={}.",
                     pending.asuId, pending.keys.size(), static_cast<int>(status.code),
                     status.message);
            anyFailed = true;
            continue;
        }
        if (!taskResult.status.ok()) {
            MarkRefreshIfNeeded(taskResult.status, needRefresh);
            UC_ERROR("ASU client query result failed: asuId={} key_count={} code={} message={}.",
                     pending.asuId, pending.keys.size(), static_cast<int>(taskResult.status.code),
                     taskResult.status.message);
            anyFailed = true;
            continue;
        }
        if (!taskResult.queryResult.has_value()) {
            UC_ERROR("ASU client query result is missing: asuId={} key_count={}.", pending.asuId,
                     pending.keys.size());
            anyFailed = true;
            continue;
        }

        const auto& childResult = *taskResult.queryResult;
        if (childResult.exists.size() != pending.keys.size()) {
            UC_ERROR("ASU client query result size mismatch: asuId={} expected={} actual={}.",
                     pending.asuId, pending.keys.size(), childResult.exists.size());
            anyFailed = true;
            continue;
        }

        for (std::size_t index = 0; index < pending.originalIndices.size(); ++index) {
            result.exists[pending.originalIndices[index]] = childResult.exists[index];
        }
        result.prefixHitKeys += childResult.prefixHitKeys;
    }

    return anyFailed ? PartialFailed("one or more asu queries failed") : Status::OK();
}

Status AsuClientImpl::LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    return SubmitAsync(ClientOpType::LOAD, entries, taskId);
}

Status AsuClientImpl::StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    return SubmitAsync(ClientOpType::STORE, entries, taskId);
}

Status AsuClientImpl::DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId)
{
    return SubmitAsync(ClientOpType::DELETE, keys, taskId);
}

Status AsuClientImpl::Check(TaskId taskId, TaskResult& result)
{
    auto ctx = taskManager_.Get(taskId);
    if (ctx != nullptr) {
        Status status;
        bool done = false;
        {
            std::lock_guard<std::mutex> lock(ctx->waitMu);
            status = BuildResult(ctx, result);
            done = ctx->Done();
        }
        if (done) { (void)taskManager_.Remove(taskId); }
        if (viewServer_ != nullptr &&
            (viewServer_->ShouldRefreshView(status) || viewServer_->ShouldRefreshView(result))) {
            RequestBackgroundRefresh();
        }
        return status;
    }

    return Status::Error(StatusCode::TASK_NOT_FOUND, "task not found");
}

Status AsuClientImpl::Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result)
{
    auto ctx = taskManager_.Get(taskId);
    if (ctx != nullptr) {
        auto status = WaitTaskContext(ctx, timeoutMs, result);
        if (status.code != StatusCode::TIMEOUT) { (void)taskManager_.Remove(taskId); }
        if (viewServer_ != nullptr &&
            (viewServer_->ShouldRefreshView(status) || viewServer_->ShouldRefreshView(result))) {
            RequestBackgroundRefresh();
        }
        return status;
    }

    return Status::Error(StatusCode::TASK_NOT_FOUND, "task not found");
}

Status AsuClientImpl::RegisterRegions(const std::vector<MemoryRegion>& regions,
                                      std::vector<RegisteredMemory>& registeredRegions)
{
    bool needRefresh = false;
    auto status = RegisterRegionsOnce(regions, registeredRegions, needRefresh);
    if (needRefresh) { RequestBackgroundRefresh(); }
    return status;
}

Status AsuClientImpl::RegisterRegionsOnce(const std::vector<MemoryRegion>& regions,
                                          std::vector<RegisteredMemory>& registeredRegions,
                                          bool& needRefresh)
{
    auto snapshot = GetSnapshot();
    if (!snapshot) { return NotInitialized(); }

    registeredRegions.clear();
    if (snapshot->transports.empty()) { return Status::OK(); }

    auto firstIter = snapshot->transports.find(snapshot->asuIds.front());
    if (firstIter == snapshot->transports.end()) {
        auto status = Status::Error(StatusCode::NOT_FOUND, "first asu transport not found");
        MarkRefreshIfNeeded(status, needRefresh);
        return WithContext(status, "asuIndex=0 asuId=" + std::to_string(snapshot->asuIds.front()));
    }

    auto status = firstIter->second->RegisterRegions(regions, registeredRegions);
    if (!status.ok()) {
        MarkRefreshIfNeeded(status, needRefresh);
        return WithContext(status, "asuIndex=0 asuId=" + std::to_string(snapshot->asuIds.front()) +
                                       " region_count=" + std::to_string(regions.size()));
    }
    if (registeredRegions.size() != regions.size()) {
        return WithContext(Status::Error(StatusCode::INTERNAL_ERROR,
                                         "register result count does not match region count"),
                           "asuIndex=0 asuId=" + std::to_string(snapshot->asuIds.front()) +
                               " region_count=" + std::to_string(regions.size()) +
                               " result_count=" + std::to_string(registeredRegions.size()));
    }

    Status finalStatus = Status::OK();
    for (std::size_t asuIndex = 1; asuIndex < snapshot->asuIds.size(); ++asuIndex) {
        auto iter = snapshot->transports.find(snapshot->asuIds[asuIndex]);
        if (iter == snapshot->transports.end()) {
            auto status = Status::Error(StatusCode::NOT_FOUND, "bound asu transport not found");
            MarkRefreshIfNeeded(status, needRefresh);
            finalStatus = WithContext(PartialFailed("one or more asu region bindings failed"),
                                      "asuIndex=" + std::to_string(asuIndex) +
                                          " asuId=" + std::to_string(snapshot->asuIds[asuIndex]));
            continue;
        }

        status = iter->second->BindRegisteredRegions(registeredRegions);
        if (!status.ok() && finalStatus.ok()) {
            MarkRefreshIfNeeded(status, needRefresh);
            finalStatus =
                WithContext(PartialFailed("one or more asu region bindings failed"),
                            "asuIndex=" + std::to_string(asuIndex) +
                                " asuId=" + std::to_string(snapshot->asuIds[asuIndex]) +
                                " region_count=" + std::to_string(registeredRegions.size()));
        }
    }

    // Remember registered regions for future transport bindings.
    if (finalStatus.ok()) {
        std::lock_guard<std::mutex> lock{mutex_};
        registeredRegions_.insert(registeredRegions_.end(), registeredRegions.begin(),
                                  registeredRegions.end());
    }
    return finalStatus;
}

Status AsuClientImpl::SubmitAsync(ClientOpType opType, const std::vector<KVBuffer>& entries,
                                  TaskId& taskId)
{
    auto snapshot = GetSnapshot();
    if (!snapshot || !snapshot->router || snapshot->transports.empty()) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    if (opType != ClientOpType::LOAD && opType != ClientOpType::STORE) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "entries submit only supports load/store");
    }

    auto ctx = std::make_unique<ClientTaskContext>();
    ctx->opType = opType;
    ctx->viewSnapshot = snapshot;
    ctx->entries = entries;
    ctx->entryStatus.assign(entries.size(), Status::OK());

    auto status = taskManager_.Submit(std::move(ctx), taskId);
    if (!status.ok()) { return status; }

    auto rawCtx = taskManager_.Get(taskId);
    if (!rawCtx) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "client task disappeared after submit");
    }

    {
        std::lock_guard<std::mutex> lock{taskQueueMu_};
        if (stopWorker_) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return NotInitialized();
        }
        taskQueue_.emplace_back(std::move(rawCtx));
    }
    taskQueueCv_.notify_one();
    return Status::OK();
}

Status AsuClientImpl::SubmitAsync(ClientOpType opType, const std::vector<CacheKey>& keys,
                                  TaskId& taskId)
{
    auto snapshot = GetSnapshot();
    if (!snapshot || !snapshot->router || snapshot->transports.empty()) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    if (opType != ClientOpType::DELETE) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INVALID_ARGUMENT, "keys submit only supports delete");
    }

    auto ctx = std::make_unique<ClientTaskContext>();
    ctx->opType = opType;
    ctx->viewSnapshot = snapshot;
    ctx->keys = keys;
    ctx->entryStatus.assign(keys.size(), Status::OK());

    auto status = taskManager_.Submit(std::move(ctx), taskId);
    if (!status.ok()) { return status; }

    auto rawCtx = taskManager_.Get(taskId);
    if (!rawCtx) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "client task disappeared after submit");
    }

    {
        std::lock_guard<std::mutex> lock{taskQueueMu_};
        if (stopWorker_) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return NotInitialized();
        }
        taskQueue_.emplace_back(std::move(rawCtx));
    }
    taskQueueCv_.notify_one();
    return Status::OK();
}

void AsuClientImpl::ProcessTask(const ClientTaskContextPtr& ctx)
{
    if (ctx == nullptr) { return; }
    ctx->state.store(ClientTaskState::INFLIGHT, std::memory_order_release);

    auto status = BuildSubTasks(ctx);
    if (!status.ok()) {
        CompleteTaskWithError(ctx, status);
    } else {
        status = DispatchTask(ctx);
    }

    bool needRefresh = false;
    MarkRefreshIfNeeded(status, needRefresh);
    if (needRefresh) { RequestBackgroundRefresh(); }
}

Status AsuClientImpl::BuildSubTasks(const ClientTaskContextPtr& ctx)
{
    auto snapshot = ctx == nullptr ? nullptr : ctx->viewSnapshot;
    if (!snapshot || !snapshot->router || snapshot->transports.empty()) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    const auto routes = ctx->opType == ClientOpType::DELETE
                            ? snapshot->router->RouteKeys(ToRouterKeys(ctx->keys))
                            : snapshot->router->RouteKeys(ExtractEntryKeys(ctx->entries));
    for (const auto& route : routes) {
        if (snapshot->transports.find(route.first) == snapshot->transports.end()) {
            return WithContext(
                Status::Error(StatusCode::NOT_FOUND, "routed asu transport not found"),
                "asuId=" + std::to_string(route.first));
        }
    }

    ctx->subTasks.reserve(routes.size());
    for (const auto& route : routes) {
        ClientSubTask subTask;
        subTask.asuId = route.first;
        subTask.originalIndices.reserve(route.second.size());
        if (ctx->opType == ClientOpType::DELETE) {
            subTask.keys.reserve(route.second.size());
            for (auto index : route.second) {
                subTask.keys.push_back(std::move(ctx->keys[index]));
                subTask.originalIndices.push_back(index);
            }
        } else {
            subTask.entries.reserve(route.second.size());
            for (auto index : route.second) {
                subTask.entries.push_back(std::move(ctx->entries[index]));
                subTask.originalIndices.push_back(index);
            }
        }
        ctx->subTasks.push_back(std::move(subTask));
    }
    std::vector<KVBuffer>{}.swap(ctx->entries);
    std::vector<CacheKey>{}.swap(ctx->keys);
    ctx->remainingSubTasks.store(ctx->subTasks.size(), std::memory_order_release);
    return Status::OK();
}

void AsuClientImpl::WorkerLoop()
{
    while (true) {
        ClientTaskContextPtr ctx;
        {
            std::unique_lock<std::mutex> lock{taskQueueMu_};
            taskQueueCv_.wait(lock, [this] { return stopWorker_ || !taskQueue_.empty(); });
            if (taskQueue_.empty()) {
                if (stopWorker_) { return; }
                continue;
            }
            ctx = std::move(taskQueue_.front());
            taskQueue_.pop_front();
        }
        ProcessTask(ctx);
    }
}

void AsuClientImpl::CompleteTaskWithError(const ClientTaskContextPtr& ctx, const Status& status)
{
    std::lock_guard<std::mutex> lock{ctx->waitMu};
    std::fill(ctx->entryStatus.begin(), ctx->entryStatus.end(), status);
    ctx->finalStatus = status;
    ctx->state.store(ClientTaskState::COMPLETED, std::memory_order_release);
    ctx->cv.notify_all();
}

Status AsuClientImpl::DispatchTask(const ClientTaskContextPtr& ctx)
{
    auto snapshot = ctx == nullptr ? nullptr : ctx->viewSnapshot;
    if (!snapshot) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "client view is not ready");
    }
    if (ctx->subTasks.empty()) {
        std::lock_guard<std::mutex> lock(ctx->waitMu);
        FinalizeTask(ctx);
        return Status::OK();
    }

    for (std::size_t subTaskIndex = 0; subTaskIndex < ctx->subTasks.size(); ++subTaskIndex) {
        auto& subTask = ctx->subTasks[subTaskIndex];
        auto transIter = snapshot->transports.find(subTask.asuId);
        if (transIter == snapshot->transports.end()) {
            return Status::Error(StatusCode::NOT_FOUND, "routed ASU transport not found");
        }

        auto onComplete = [ctx, subTaskIndex](TaskResult result) {
            CompleteSubTask(ctx, subTaskIndex, std::move(result));
        };
        Status status;
        if (ctx->opType == ClientOpType::LOAD) {
            status = transIter->second->LoadAsync(subTask.entries, subTask.transTaskId,
                                                  std::move(onComplete));
        } else if (ctx->opType == ClientOpType::STORE) {
            status = transIter->second->StoreAsync(subTask.entries, subTask.transTaskId,
                                                   std::move(onComplete));
        } else {
            status = transIter->second->DeleteAsync(subTask.keys, subTask.transTaskId,
                                                    std::move(onComplete));
        }
        if (!status.ok()) {
            for (std::size_t index = 0; index < subTaskIndex; ++index) {
                auto& dispatchedSubTask = ctx->subTasks[index];
                if (dispatchedSubTask.transTaskId == kInvalidTaskId) { continue; }

                auto dispatchedTransIter = snapshot->transports.find(dispatchedSubTask.asuId);
                if (dispatchedTransIter == snapshot->transports.end()) { continue; }
                (void)dispatchedTransIter->second->Cancel(dispatchedSubTask.transTaskId);
            }

            const auto dispatchStatus =
                WithContext(status, "asuId=" + std::to_string(subTask.asuId));
            {
                std::lock_guard<std::mutex> lock(ctx->waitMu);
                for (std::size_t index = subTaskIndex; index < ctx->subTasks.size(); ++index) {
                    auto& failedSubTask = ctx->subTasks[index];
                    failedSubTask.completed = true;
                    failedSubTask.failed = true;
                    failedSubTask.status =
                        index == subTaskIndex
                            ? dispatchStatus
                            : Status::Error(StatusCode::CANCELED,
                                            "subtask not dispatched after a dispatch failure");
                    for (auto originalIndex : failedSubTask.originalIndices) {
                        ctx->entryStatus[originalIndex] = failedSubTask.status;
                    }
                    ctx->remainingSubTasks.fetch_sub(1, std::memory_order_acq_rel);
                }

                if (ctx->AllSubTasksCompleted()) { FinalizeTask(ctx); }
            }
            return dispatchStatus;
        }
    }
    return Status::OK();
}

void AsuClientImpl::CompleteSubTask(const ClientTaskContextPtr& ctx, std::size_t subTaskIndex,
                                    TaskResult result)
{
    std::lock_guard<std::mutex> lock(ctx->waitMu);
    auto& subTask = ctx->subTasks[subTaskIndex];
    if (subTask.completed) { return; }

    subTask.completed = true;
    subTask.failed = !result.status.ok();
    subTask.status = result.status;
    for (std::size_t index = 0; index < subTask.originalIndices.size(); ++index) {
        ctx->entryStatus[subTask.originalIndices[index]] =
            index < result.entryStatus.size() ? result.entryStatus[index] : result.status;
    }

    if (ctx->remainingSubTasks.fetch_sub(1, std::memory_order_acq_rel) == 1) { FinalizeTask(ctx); }
}

void AsuClientImpl::FinalizeTask(const ClientTaskContextPtr& ctx)
{
    const bool anyFailed = std::any_of(ctx->subTasks.begin(), ctx->subTasks.end(),
                                       [](const ClientSubTask& subTask) { return subTask.failed; });
    ctx->finalStatus =
        anyFailed ? Status::Error(StatusCode::PARTIAL_FAILED, "client task partially failed: " +
                                                                  FirstFailedSubTaskContext(*ctx))
                  : Status::OK();
    ctx->state.store(ClientTaskState::COMPLETED, std::memory_order_release);
    ctx->cv.notify_all();
}

Status AsuClientImpl::BuildResult(const ClientTaskContextPtr& ctx, TaskResult& result)
{
    result.status = ctx->Done() ? ctx->finalStatus
                                : Status::Error(StatusCode::IN_PROGRESS, "client task in progress");
    result.entryStatus = ctx->entryStatus;
    result.queryResult.reset();
    return result.status;
}

Status AsuClientImpl::WaitTaskContext(const ClientTaskContextPtr& ctx, std::uint64_t timeoutMs,
                                      TaskResult& result)
{
    if (ctx == nullptr) {
        return Status::Error(StatusCode::TASK_NOT_FOUND, "client task not found");
    }

    const auto waitMs = timeoutMs == 0 ? config_.defaultWaitTimeoutMs : timeoutMs;
    std::unique_lock<std::mutex> lock(ctx->waitMu);
    const bool done =
        ctx->cv.wait_for(lock, std::chrono::milliseconds(waitMs), [ctx] { return ctx->Done(); });
    BuildResult(ctx, result);
    if (!done) {
        result.status = Status::Error(
            StatusCode::TIMEOUT,
            "client task wait timeout: client_task_id=" + std::to_string(ctx->taskId) +
                " op=" + ClientOpTypeName(ctx->opType) + " wait_ms=" + std::to_string(waitMs));
        UC_ERROR("ASU client task wait timeout: client_task_id={} op={} wait_ms={}.", ctx->taskId,
                 ClientOpTypeName(ctx->opType), waitMs);
    }
    return result.status;
}

Status AsuClientImpl::UnregisterRegions(const std::vector<MRHandle>& handles)
{
    bool needRefresh = false;
    auto status = UnregisterRegionsOnce(handles, needRefresh);
    if (needRefresh) { RequestBackgroundRefresh(); }
    return status;
}

Status AsuClientImpl::UnregisterRegionsOnce(const std::vector<MRHandle>& handles, bool& needRefresh)
{
    auto snapshot = GetSnapshot();
    if (!snapshot) { return NotInitialized(); }

    Status finalStatus = Status::OK();
    for (const auto& item : snapshot->transports) {
        auto status = item.second->UnregisterRegions(handles);
        if (!status.ok() && finalStatus.ok()) {
            MarkRefreshIfNeeded(status, needRefresh);
            finalStatus =
                WithContext(status, "asuId=" + std::to_string(item.first) +
                                        " handle_count=" + std::to_string(handles.size()));
        }
    }
    if (finalStatus.ok()) {
        std::lock_guard<std::mutex> lock{mutex_};
        registeredRegions_.erase(
            std::remove_if(registeredRegions_.begin(), registeredRegions_.end(),
                           [&handles](const RegisteredMemory& region) {
                               return std::find(handles.begin(), handles.end(), region.handle) !=
                                      handles.end();
                           }),
            registeredRegions_.end());
    }
    return finalStatus;
}

Status AsuClientImpl::BuildSnapshot(const GlobalView& view,
                                    const std::shared_ptr<ViewSnapshot>& oldSnapshot,
                                    std::shared_ptr<ViewSnapshot>& snapshot)
{
    auto nextSnapshot = std::make_shared<ViewSnapshot>();
    auto asuIds = GetSortedAsuIds(view);
    nextSnapshot->view = view;

    for (std::size_t asuIndex = 0; asuIndex < asuIds.size(); ++asuIndex) {
        auto asuId = asuIds[asuIndex];
        std::shared_ptr<AsuTransport> transport;
        if (oldSnapshot != nullptr) {
            auto oldIter = oldSnapshot->transports.find(asuId);
            if (oldIter != oldSnapshot->transports.end()) { transport = oldIter->second; }
        }

        if (transport == nullptr) {
            auto viewIter = view.asuMap.find(asuId);
            auto asuInfo = viewIter == view.asuMap.end() ? AsuInfo{} : viewIter->second;
            auto status = BuildTransport(asuId, asuInfo, transport);
            if (!status.ok()) {
                return WithContext(status, "asuIndex=" + std::to_string(asuIndex) +
                                               " asuId=" + std::to_string(asuId));
            }

            status = BindRegisteredRegions(asuId, transport);
            if (!status.ok()) {
                transport->Shutdown();
                return WithContext(
                    status, "bind registered regions during view refresh, asuIndex=" +
                                std::to_string(asuIndex) + " asuId=" + std::to_string(asuId));
            }
        }

        nextSnapshot->transports.emplace(asuId, std::move(transport));
    }

    UC::KV::RouterConfig routerConfig;
    auto status = BuildRouterConfigFromAttrs(config_.attrs, routerConfig);
    if (!status.ok()) {
        UC_ERROR("BuildSnapshot build router config failed: {}", status.message);
        return status;
    }

    std::vector<UC::KV::NodeId> nodeIds(asuIds.begin(), asuIds.end());
    nextSnapshot->router = UC::KV::CreateRouter(nodeIds, UC::KV::HashFunction{}, routerConfig);
    nextSnapshot->asuIds = std::move(asuIds);
    snapshot = std::move(nextSnapshot);
    return Status::OK();
}

Status AsuClientImpl::BuildTransport(AsuId asuId, const AsuInfo& asuInfo,
                                     std::shared_ptr<AsuTransport>& transport)
{
    TransportConfig config;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto configIter = transportConfigs_.find(asuId);
        if (configIter == transportConfigs_.end()) {
            return Status::Error(StatusCode::NOT_FOUND,
                                 "transport config not found, asuId=" + std::to_string(asuId));
        }
        config = configIter->second;
    }
    ApplyAsuInfoToTransportConfig(asuInfo, config);

    auto nextTransport = transportFactory_();
    if (!nextTransport) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "transport factory returned null, asuId=" + std::to_string(asuId));
    }

    auto status = nextTransport->Init(config);
    if (!status.ok()) {
        return WithContext(status, "init transport failed, asuId=" + std::to_string(asuId));
    }

    transport = std::shared_ptr<AsuTransport>(std::move(nextTransport));
    return Status::OK();
}

Status AsuClientImpl::BindRegisteredRegions(AsuId asuId,
                                            const std::shared_ptr<AsuTransport>& transport)
{
    std::vector<RegisteredMemory> registeredRegions;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        registeredRegions = registeredRegions_;
    }
    if (registeredRegions.empty()) { return Status::OK(); }

    auto status = transport->BindRegisteredRegions(registeredRegions);
    if (!status.ok()) {
        return WithContext(status, "asuId=" + std::to_string(asuId) +
                                       " region_count=" + std::to_string(registeredRegions.size()));
    }
    return Status::OK();
}

Status AsuClientImpl::RefreshView()
{
    AsuClientConfig config;
    std::shared_ptr<ViewServer> viewServer;
    std::shared_ptr<ViewSnapshot> oldSnapshot;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_) { return NotInitialized(); }
        config = config_;
        viewServer = viewServer_;
        oldSnapshot = snapshot_;
    }
    if (viewServer == nullptr) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "view server is not initialized");
    }

    GlobalView view;
    auto status = viewServer->GetGlobalView(view);
    if (!status.ok()) { return status; }
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_) { return NotInitialized(); }
        if (snapshot_ != nullptr && !viewServer->ShouldPublishView(snapshot_->view, view)) {
            return Status::OK();
        }
    }

    std::shared_ptr<ViewSnapshot> nextSnapshot;
    status = BuildSnapshot(view, oldSnapshot, nextSnapshot);
    if (!status.ok()) { return status; }

    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_) { return NotInitialized(); }
        if (snapshot_ != nullptr && !viewServer->ShouldPublishView(snapshot_->view, view)) {
            return Status::OK();
        }
        if (oldSnapshot != nullptr) {
            for (const auto& item : oldSnapshot->transports) {
                if (nextSnapshot->transports.find(item.first) == nextSnapshot->transports.end()) {
                    retiredTransports_.emplace_back(item.second);
                }
            }
        }
        snapshot_ = std::move(nextSnapshot);
    }

    return Status::OK();
}

void AsuClientImpl::RequestBackgroundRefresh()
{
    bool shouldStart = false;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_ || refreshInProgress_) { return; }
        refreshInProgress_ = true;
        shouldStart = true;
    }

    if (!shouldStart) { return; }
    if (refreshThread_.joinable()) { refreshThread_.join(); }

    refreshThread_ = std::thread([this] {
        (void)RefreshView();
        std::lock_guard<std::mutex> lock{mutex_};
        refreshInProgress_ = false;
    });
}

void AsuClientImpl::JoinBackgroundRefresh()
{
    if (refreshThread_.joinable()) { refreshThread_.join(); }
}

Status AsuClientImpl::ShutdownSnapshotTransports(const std::shared_ptr<ViewSnapshot>& snapshot)
{
    if (!snapshot) { return Status::OK(); }
    Status finalStatus = Status::OK();
    for (auto& item : snapshot->transports) {
        auto status = item.second->Shutdown();
        if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
    }
    return finalStatus;
}

Status AsuClientImpl::DrainTasksBeforeShutdown(std::uint64_t waitTimeoutMs)
{
    Status finalStatus = Status::OK();
    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx == nullptr) { continue; }

        if (!ctx->Done()) {
            TaskResult result;
            auto status = WaitTaskContext(ctx, waitTimeoutMs, result);
            if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
        }
        (void)taskManager_.Remove(ctx->taskId);
    }
    return finalStatus;
}

std::shared_ptr<ViewSnapshot> AsuClientImpl::GetSnapshot() const
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (!initialized_) { return nullptr; }
    return snapshot_;
}

void AsuClientImpl::MarkRefreshIfNeeded(const Status& status, bool& needRefresh) const
{
    if (viewServer_ != nullptr && viewServer_->ShouldRefreshView(status)) { needRefresh = true; }
}

std::vector<AsuId> AsuClientImpl::GetSortedAsuIds(const GlobalView& view)
{
    std::vector<AsuId> asuIds;
    asuIds.reserve(view.asuMap.size());
    for (const auto& item : view.asuMap) {
        if (item.first != static_cast<AsuId>(UC::KV::kInvalidNodeId)) {
            asuIds.emplace_back(item.first);
        }
    }
    std::sort(asuIds.begin(), asuIds.end());
    return asuIds;
}

Status AsuClientImpl::LoadConfig(const std::string& configPath, AsuClientConfig& config)
{
    return LoadAsuClientConfig(configPath, config);
}

Status AsuClientImpl::WithContext(Status status, const std::string& context)
{
    if (context.empty()) { return status; }
    if (status.message.empty()) {
        status.message = context;
    } else {
        status.message += ", " + context;
    }
    return status;
}

Status AsuClientImpl::NotInitialized()
{
    return Status::Error(StatusCode::NOT_INITIALIZED, "asu client is not initialized");
}

std::unique_ptr<AsuClient> CreateAsuClient(TransportFactory transportFactory)
{
    return std::make_unique<AsuClientImpl>(std::move(transportFactory), nullptr);
}

extern "C" std::unique_ptr<AsuClient> UcmAsuCreateAsuClient(
    const TransportFactory* transportFactory)
{
    if (transportFactory == nullptr) { return CreateAsuClient(); }
    return CreateAsuClient(*transportFactory);
}

extern "C" Status UcmAsuLoadAsuClientConfig(const char* configPath, AsuClientConfig* config)
{
    if (configPath == nullptr || config == nullptr) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "UcmAsuLoadAsuClientConfig received null argument");
    }
    return LoadAsuClientConfig(configPath, *config);
}

}  // namespace UC::ASU
