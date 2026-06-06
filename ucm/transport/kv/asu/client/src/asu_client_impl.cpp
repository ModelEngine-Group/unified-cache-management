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
#include <cctype>
#include <chrono>
#include <functional>
#include <limits>
#include <thread>
#include <utility>
#include "asu_transport/types.h"
#include "client_config_parser.h"
#include "kv_common/router.h"
#include "logger/logger.h"

namespace UC::ASU {

constexpr std::uint32_t kMaxShutdownDrainAttempts = 64;

Status PartialFailed(const std::string& message)
{
    return Status::Error(StatusCode::PARTIAL_FAILED, message);
}

Status LogErrorStatus(StatusCode code, const std::string& message)
{
    UC_ERROR("{}", message);
    return Status::Error(code, message);
}

std::vector<UC::KV::CacheKey> ExtractEntryKeys(const std::vector<KVBuffer>& entries)
{
    std::vector<UC::KV::CacheKey> keys;
    keys.reserve(entries.size());
    for (const auto& entry : entries) { keys.emplace_back(entry.key); }
    return keys;
}

std::string NormalizeAttrValue(std::string value)
{
    std::replace(value.begin(), value.end(), '-', '_');
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    return value;
}

bool GetAttr(const std::unordered_map<std::string, std::string>& attrs, const std::string& key,
             std::string& value)
{
    auto iter = attrs.find(key);
    if (iter == attrs.end()) { return false; }
    value = iter->second;
    return true;
}

Status GetUint64Attr(const std::unordered_map<std::string, std::string>& attrs,
                     const std::string& key, std::uint64_t& value)
{
    std::string text;
    if (!GetAttr(attrs, key, text)) { return Status::OK(); }

    try {
        std::size_t parsed{0};
        const auto parsedValue = std::stoull(text, &parsed, 0);
        if (parsed != text.size()) {
            return LogErrorStatus(StatusCode::INVALID_ARGUMENT,
                                  "invalid router config " + key + "=" + text);
        }
        value = parsedValue;
        return Status::OK();
    } catch (const std::exception&) {
        return LogErrorStatus(StatusCode::INVALID_ARGUMENT,
                              "invalid router config " + key + "=" + text);
    }
}

Status GetBoolAttr(const std::unordered_map<std::string, std::string>& attrs,
                   const std::string& key, bool& value)
{
    std::string text;
    if (!GetAttr(attrs, key, text)) { return Status::OK(); }

    const auto normalized = NormalizeAttrValue(text);
    if (normalized == "1" || normalized == "TRUE" || normalized == "ON" || normalized == "YES") {
        value = true;
        return Status::OK();
    }
    if (normalized == "0" || normalized == "FALSE" || normalized == "OFF" || normalized == "NO") {
        value = false;
        return Status::OK();
    }
    return LogErrorStatus(StatusCode::INVALID_ARGUMENT,
                          "invalid router config " + key + "=" + text);
}

UC::KV::HashTableType ParseHashTableType(const std::string& value, UC::KV::HashTableType fallback)
{
    const auto type = NormalizeAttrValue(value);
    if (type == "RING_HASH" || type == "RING_HASH_FULL_SPREAD") {
        return UC::KV::HashTableType::RING_HASH_FULL_SPREAD;
    }
    if (type == "MAGLEV" || type == "MAGLEV_FULL_SPREAD") {
        return UC::KV::HashTableType::MAGLEV_FULL_SPREAD;
    }
    if (type == "CONTIGUOUS_BLOCK_AFFINITY") {
        return UC::KV::HashTableType::CONTIGUOUS_BLOCK_AFFINITY;
    }
    if (type == "BATCH_TOPK_AFFINITY") { return UC::KV::HashTableType::BATCH_TOPK_AFFINITY; }
    return fallback;
}

Status BuildHashTableConfig(const std::unordered_map<std::string, std::string>& attrs,
                            UC::KV::HashTableConfig& config)
{
    config = UC::KV::HashTableConfig{};

    std::string type;
    if (GetAttr(attrs, "hash_table.type", type)) {
        config.type = ParseHashTableType(type, config.type);
    }

    auto status =
        GetUint64Attr(attrs, "ring_hash.virtual_node_count", config.ringHash.virtualNodeCount);
    if (!status.ok()) { return status; }
    status = GetUint64Attr(attrs, "maglev.table_size", config.maglev.tableSize);
    if (!status.ok()) { return status; }
    status = GetUint64Attr(attrs, "contiguous_block_affinity.block_count",
                           config.contiguousBlockAffinity.blockCount);
    if (!status.ok()) { return status; }
    status = GetUint64Attr(attrs, "batch_topk_affinity.top_k", config.batchTopKAffinity.topK);
    if (!status.ok()) { return status; }

    std::string fullSpreadType;
    if (GetAttr(attrs, "contiguous_block_affinity.full_spread_type", fullSpreadType)) {
        config.contiguousBlockAffinity.fullSpreadType =
            ParseHashTableType(fullSpreadType, config.contiguousBlockAffinity.fullSpreadType);
    }

    status = GetBoolAttr(attrs, "contiguous_block_affinity.dynamic_adjust_enabled",
                         config.contiguousBlockAffinity.dynamicAdjustEnabled);
    if (!status.ok()) { return status; }
    status = GetBoolAttr(attrs, "batch_topk_affinity.dynamic_adjust_enabled",
                         config.batchTopKAffinity.dynamicAdjustEnabled);
    if (!status.ok()) { return status; }
    return Status::OK();
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

    snapshot_ = std::move(nextSnapshot);
    initialized_ = true;
    return Status::OK();
}

Status AsuClientImpl::Shutdown()
{
    JoinBackgroundRefresh();

    std::shared_ptr<ViewSnapshot> snapshot;
    std::vector<std::shared_ptr<AsuTransport>> retiredTransports;
    std::uint64_t waitTimeoutMs = 0;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        snapshot = std::move(snapshot_);
        retiredTransports = std::move(retiredTransports_);
        waitTimeoutMs = config_.defaultWaitTimeoutMs;
        config_ = AsuClientConfig{};
        viewServer_.reset();
        transportConfigs_.clear();
        registeredResources_.clear();
        initialized_ = false;
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

    if (options.mode == QueryMode::PREFIX) {
        Status finalStatus = Status::OK();
        for (const auto& item : snapshot->transports) {
            QueryResult childResult;
            auto status = item.second->Query(keys, options, childResult);
            if (!status.ok()) {
                MarkRefreshIfNeeded(status, needRefresh);
                finalStatus = WithContext(PartialFailed("one or more asu prefix queries failed"),
                                          "asuId=" + std::to_string(item.first));
                continue;
            }
            if (childResult.exists.size() != keys.size()) {
                return Status::Error(
                    StatusCode::INTERNAL_ERROR,
                    "prefix query result size mismatch, asuId=" + std::to_string(item.first) +
                        " expected=" + std::to_string(keys.size()) +
                        " actual=" + std::to_string(childResult.exists.size()));
            }
            result.prefixHitKeys += childResult.prefixHitKeys;
            for (std::size_t index = 0;
                 index < result.exists.size() && index < childResult.exists.size(); ++index) {
                result.exists[index] = result.exists[index] || childResult.exists[index];
            }
        }
        return finalStatus;
    }

    auto routes = snapshot->router->RouteKeys(keys);
    for (const auto& route : routes) {
        auto transportIter = snapshot->transports.find(route.first);
        if (transportIter == snapshot->transports.end()) {
            auto status = Status::Error(StatusCode::NOT_FOUND, "routed asu transport not found");
            MarkRefreshIfNeeded(status, needRefresh);
            return WithContext(status, "asuId=" + std::to_string(route.first));
        }

        std::vector<CacheKey> childKeys;
        childKeys.reserve(route.second.size());
        for (auto index : route.second) { childKeys.emplace_back(keys[index]); }

        QueryResult childResult;
        auto status = transportIter->second->Query(childKeys, options, childResult);
        if (!status.ok()) {
            MarkRefreshIfNeeded(status, needRefresh);
            return WithContext(status, "asuId=" + std::to_string(route.first) +
                                           " key_count=" + std::to_string(childKeys.size()));
        }
        if (childResult.exists.size() != childKeys.size()) {
            return Status::Error(
                StatusCode::INTERNAL_ERROR,
                "query result size mismatch, asuId=" + std::to_string(route.first) +
                    " expected=" + std::to_string(childKeys.size()) +
                    " actual=" + std::to_string(childResult.exists.size()));
        }

        for (std::size_t index = 0; index < route.second.size(); ++index) {
            result.exists[route.second[index]] = childResult.exists[index];
        }
        result.prefixHitKeys += childResult.prefixHitKeys;
    }

    return Status::OK();
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
        PollTask(ctx);
        auto status = BuildResult(ctx, result);
        if (IsTaskComplete(result)) { (void)taskManager_.Remove(taskId); }
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
        if (IsTaskComplete(result)) { (void)taskManager_.Remove(taskId); }
        if (viewServer_ != nullptr &&
            (viewServer_->ShouldRefreshView(status) || viewServer_->ShouldRefreshView(result))) {
            RequestBackgroundRefresh();
        }
        return status;
    }

    return Status::Error(StatusCode::TASK_NOT_FOUND, "task not found");
}

Status AsuClientImpl::RegisterRegions(const std::vector<MemoryRegion>& regions,
                                      std::vector<RegisterResult>& results)
{
    bool needRefresh = false;
    auto status = RegisterRegionsOnce(regions, results, needRefresh);
    if (needRefresh) { RequestBackgroundRefresh(); }
    return status;
}

Status AsuClientImpl::RegisterRegionsOnce(const std::vector<MemoryRegion>& regions,
                                          std::vector<RegisterResult>& results, bool& needRefresh)
{
    auto snapshot = GetSnapshot();
    if (!snapshot) { return NotInitialized(); }

    results.clear();
    if (snapshot->transports.empty()) { return Status::OK(); }

    auto firstIter = snapshot->transports.find(snapshot->asuIds.front());
    if (firstIter == snapshot->transports.end()) {
        auto status = Status::Error(StatusCode::NOT_FOUND, "first asu transport not found");
        MarkRefreshIfNeeded(status, needRefresh);
        return WithContext(status, "asuIndex=0 asuId=" + std::to_string(snapshot->asuIds.front()));
    }

    auto status = firstIter->second->RegisterRegions(regions, results);
    if (!status.ok()) {
        MarkRefreshIfNeeded(status, needRefresh);
        return WithContext(status, "asuIndex=0 asuId=" + std::to_string(snapshot->asuIds.front()) +
                                       " region_count=" + std::to_string(regions.size()));
    }

    std::vector<RegisteredMemory> registeredRegions;
    registeredRegions.reserve(regions.size());
    for (std::size_t index = 0; index < regions.size(); ++index) {
        RegisteredMemory registeredRegion;
        registeredRegion.region = regions[index];
        if (index < results.size()) { registeredRegion.handle = results[index].handle; }
        registeredRegions.emplace_back(registeredRegion);
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

        std::vector<RegisterResult> childResults;
        status = iter->second->BindRegisteredRegions(registeredRegions, childResults);
        if (!status.ok() && finalStatus.ok()) {
            MarkRefreshIfNeeded(status, needRefresh);
            finalStatus =
                WithContext(PartialFailed("one or more asu region bindings failed"),
                            "asuIndex=" + std::to_string(asuIndex) +
                                " asuId=" + std::to_string(snapshot->asuIds[asuIndex]) +
                                " region_count=" + std::to_string(registeredRegions.size()));
        }
    }

    if (finalStatus.ok()) {
        std::lock_guard<std::mutex> lock{mutex_};
        for (std::size_t index = 0; index < regions.size() && index < results.size(); ++index) {
            registeredResources_.emplace_back(RegisteredResource{regions[index], results[index]});
        }
    }
    return finalStatus;
}

Status AsuClientImpl::SubmitAsync(ClientOpType opType, const std::vector<KVBuffer>& entries,
                                  TaskId& taskId)
{
    bool needRefresh = false;
    auto status = SubmitAsyncOnce(opType, entries, taskId, needRefresh);
    if (needRefresh) { RequestBackgroundRefresh(); }
    return status;
}

Status AsuClientImpl::SubmitAsyncOnce(ClientOpType opType, const std::vector<KVBuffer>& entries,
                                      TaskId& taskId, bool& needRefresh)
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
    const auto count = entries.size();
    ctx->entryStatus.assign(count, Status::OK());

    auto routes = snapshot->router->RouteKeys(ExtractEntryKeys(entries));
    for (const auto& route : routes) {
        if (snapshot->transports.find(route.first) == snapshot->transports.end()) {
            auto status = Status::Error(StatusCode::NOT_FOUND, "routed asu transport not found");
            MarkRefreshIfNeeded(status, needRefresh);
            taskId = kInvalidTaskId;
            return WithContext(status, "asuId=" + std::to_string(route.first));
        }

        ClientSubTask subTask;
        subTask.asuId = route.first;
        subTask.entries.reserve(route.second.size());
        subTask.originalIndices.reserve(route.second.size());
        for (auto index : route.second) {
            subTask.entries.push_back(entries[index]);
            subTask.originalIndices.push_back(index);
        }
        ctx->subTasks.push_back(std::move(subTask));
    }

    auto status = taskManager_.Submit(std::move(ctx), taskId);
    if (!status.ok()) { return status; }

    auto rawCtx = taskManager_.Get(taskId);
    if (!rawCtx) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "client task disappeared after submit");
    }

    status = DispatchTask(rawCtx);
    if (!status.ok()) {
        MarkRefreshIfNeeded(status, needRefresh);
        taskManager_.Remove(taskId);
        taskId = kInvalidTaskId;
        return status;
    }

    rawCtx->state.store(ClientTaskState::INFLIGHT, std::memory_order_release);
    return Status::OK();
}

Status AsuClientImpl::SubmitAsync(ClientOpType opType, const std::vector<CacheKey>& keys,
                                  TaskId& taskId)
{
    bool needRefresh = false;
    auto status = SubmitAsyncOnce(opType, keys, taskId, needRefresh);
    if (needRefresh) { RequestBackgroundRefresh(); }
    return status;
}

Status AsuClientImpl::SubmitAsyncOnce(ClientOpType opType, const std::vector<CacheKey>& keys,
                                      TaskId& taskId, bool& needRefresh)
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
    ctx->entryStatus.assign(keys.size(), Status::OK());

    auto routes = snapshot->router->RouteKeys(keys);
    for (const auto& route : routes) {
        if (snapshot->transports.find(route.first) == snapshot->transports.end()) {
            auto status = Status::Error(StatusCode::NOT_FOUND, "routed asu transport not found");
            MarkRefreshIfNeeded(status, needRefresh);
            taskId = kInvalidTaskId;
            return WithContext(status, "asuId=" + std::to_string(route.first));
        }

        ClientSubTask subTask;
        subTask.asuId = route.first;
        subTask.keys.reserve(route.second.size());
        subTask.originalIndices.reserve(route.second.size());
        for (auto index : route.second) {
            subTask.keys.push_back(keys[index]);
            subTask.originalIndices.push_back(index);
        }
        ctx->subTasks.push_back(std::move(subTask));
    }

    auto status = taskManager_.Submit(std::move(ctx), taskId);
    if (!status.ok()) { return status; }

    auto rawCtx = taskManager_.Get(taskId);
    if (!rawCtx) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "client task disappeared after submit");
    }

    status = DispatchTask(rawCtx);
    if (!status.ok()) {
        MarkRefreshIfNeeded(status, needRefresh);
        taskManager_.Remove(taskId);
        taskId = kInvalidTaskId;
        return status;
    }

    rawCtx->state.store(ClientTaskState::INFLIGHT, std::memory_order_release);
    return Status::OK();
}

Status AsuClientImpl::DispatchTask(const ClientTaskContextPtr& ctx)
{
    auto snapshot = ctx == nullptr ? nullptr : ctx->viewSnapshot;
    if (!snapshot) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "client view is not ready");
    }

    for (auto& subTask : ctx->subTasks) {
        auto transIter = snapshot->transports.find(subTask.asuId);
        if (transIter == snapshot->transports.end()) {
            return Status::Error(StatusCode::NOT_FOUND, "routed ASU transport not found");
        }

        Status status;
        if (ctx->opType == ClientOpType::LOAD) {
            status = transIter->second->LoadAsync(subTask.entries, subTask.transTaskId);
        } else if (ctx->opType == ClientOpType::STORE) {
            status = transIter->second->StoreAsync(subTask.entries, subTask.transTaskId);
        } else {
            status = transIter->second->DeleteAsync(subTask.keys, subTask.transTaskId);
        }
        if (!status.ok()) {
            for (auto& dispatchedSubTask : ctx->subTasks) {
                if (&dispatchedSubTask == &subTask) { break; }
                if (dispatchedSubTask.transTaskId == kInvalidTaskId) { continue; }

                auto dispatchedTransIter = snapshot->transports.find(dispatchedSubTask.asuId);
                if (dispatchedTransIter == snapshot->transports.end()) { continue; }
                (void)dispatchedTransIter->second->Cancel(dispatchedSubTask.transTaskId);
                dispatchedSubTask.completed = true;
                dispatchedSubTask.failed = true;
            }
            return WithContext(status, "asuId=" + std::to_string(subTask.asuId));
        }
    }
    return Status::OK();
}
bool AsuClientImpl::PollTask(const ClientTaskContextPtr& ctx)
{
    auto snapshot = ctx == nullptr ? nullptr : ctx->viewSnapshot;
    if (!ctx || ctx->Done()) { return true; }
    std::lock_guard<std::mutex> lock(ctx->waitMu);
    if (ctx->Done()) { return true; }
    if (!snapshot || ctx->state.load(std::memory_order_acquire) != ClientTaskState::INFLIGHT) {
        return false;
    }

    bool allDone = true;
    bool anyFailed = false;
    for (auto& subTask : ctx->subTasks) {
        if (subTask.completed) {
            anyFailed = anyFailed || subTask.failed;
            continue;
        }

        auto transIter = snapshot->transports.find(subTask.asuId);
        if (transIter == snapshot->transports.end()) {
            subTask.completed = true;
            subTask.failed = true;
            anyFailed = true;
            continue;
        }

        TaskResult subResult;
        auto status = transIter->second->Check(subTask.transTaskId, subResult);
        if (!status.ok()) {
            subTask.completed = true;
            subTask.failed = true;
            anyFailed = true;
            continue;
        }
        if (subResult.status.code == StatusCode::IN_PROGRESS) {
            allDone = false;
            continue;
        }
        subTask.completed = true;
        if (!subResult.status.ok()) {
            subTask.failed = true;
            anyFailed = true;
        }

        const auto& originalIndices = subTask.originalIndices;
        for (std::size_t i = 0; i < originalIndices.size() && i < subResult.entryStatus.size();
             ++i) {
            ctx->entryStatus[originalIndices[i]] = subResult.entryStatus[i];
        }
    }

    if (allDone) {
        ctx->finalStatus =
            anyFailed ? Status::Error(StatusCode::PARTIAL_FAILED, "client task partially failed")
                      : Status::OK();
        ctx->state.store(ClientTaskState::COMPLETED, std::memory_order_release);
        ctx->cv.notify_all();
        return true;
    }
    return false;
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
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(waitMs);
    auto snapshot = ctx->viewSnapshot;

    std::unique_lock<std::mutex> lock(ctx->waitMu);
    while (!ctx->Done()) {
        if (!snapshot || ctx->state.load(std::memory_order_acquire) != ClientTaskState::INFLIGHT) {
            if (std::chrono::steady_clock::now() >= deadline) {
                BuildResult(ctx, result);
                result.status = Status::Error(StatusCode::TIMEOUT, "client task wait timeout");
                return result.status;
            }
            ctx->cv.wait_until(lock, deadline);
            continue;
        }

        bool allDone = true;
        bool anyFailed = false;
        for (auto& subTask : ctx->subTasks) {
            anyFailed = anyFailed || subTask.failed;
            if (subTask.completed) { continue; }

            auto transIter = snapshot->transports.find(subTask.asuId);
            if (transIter == snapshot->transports.end()) {
                subTask.completed = true;
                subTask.failed = true;
                anyFailed = true;
                continue;
            }

            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                BuildResult(ctx, result);
                result.status = Status::Error(StatusCode::TIMEOUT, "client task wait timeout");
                return result.status;
            }
            const auto remainingMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            const auto subTimeoutMs =
                static_cast<std::uint64_t>(std::max<std::int64_t>(1, remainingMs));

            TaskResult subResult;
            auto status = transIter->second->Wait(subTask.transTaskId, subTimeoutMs, subResult);

            if (status.code == StatusCode::TIMEOUT) {
                BuildResult(ctx, result);
                result.status = Status::Error(StatusCode::TIMEOUT, "client task wait timeout");
                return result.status;
            }
            if (status.code == StatusCode::IN_PROGRESS ||
                subResult.status.code == StatusCode::IN_PROGRESS) {
                allDone = false;
                continue;
            }
            subTask.completed = true;
            if (!status.ok() || !subResult.status.ok()) {
                subTask.failed = true;
                anyFailed = true;
            }

            const auto& originalIndices = subTask.originalIndices;
            for (std::size_t i = 0; i < originalIndices.size() && i < subResult.entryStatus.size();
                 ++i) {
                ctx->entryStatus[originalIndices[i]] = subResult.entryStatus[i];
            }
        }

        for (const auto& subTask : ctx->subTasks) {
            allDone = allDone && subTask.completed;
            anyFailed = anyFailed || subTask.failed;
        }
        if (allDone) {
            ctx->finalStatus = anyFailed ? Status::Error(StatusCode::PARTIAL_FAILED,
                                                         "client task partially failed")
                                         : Status::OK();
            ctx->state.store(ClientTaskState::COMPLETED, std::memory_order_release);
            ctx->cv.notify_all();
            break;
        }
    }

    return BuildResult(ctx, result);
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
        registeredResources_.erase(
            std::remove_if(registeredResources_.begin(), registeredResources_.end(),
                           [&handles](const RegisteredResource& resource) {
                               return std::find(handles.begin(), handles.end(),
                                                resource.result.handle) != handles.end();
                           }),
            registeredResources_.end());
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

            status = BindRegisteredResources(asuId, transport);
            if (!status.ok()) {
                transport->Shutdown();
                return WithContext(
                    status, "bind registered resources during view refresh, asuIndex=" +
                                std::to_string(asuIndex) + " asuId=" + std::to_string(asuId));
            }
        }

        nextSnapshot->transports.emplace(asuId, std::move(transport));
    }

    UC::KV::HashTableConfig hashTableConfig;
    auto status = BuildHashTableConfig(config_.attrs, hashTableConfig);
    if (!status.ok()) {
        UC_ERROR("BuildSnapshot build router config failed: {}", status.message);
        return status;
    }

    std::vector<UC::KV::NodeId> nodeIds(asuIds.begin(), asuIds.end());
    nextSnapshot->router = UC::KV::CreateRouter(nodeIds, UC::KV::HashFunction{}, hashTableConfig);
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

Status AsuClientImpl::BindRegisteredResources(AsuId asuId,
                                              const std::shared_ptr<AsuTransport>& transport)
{
    std::vector<RegisteredResource> resources;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        resources = registeredResources_;
    }
    if (resources.empty()) { return Status::OK(); }

    std::vector<RegisteredMemory> registeredRegions;
    registeredRegions.reserve(resources.size());
    for (const auto& resource : resources) {
        RegisteredMemory registeredRegion;
        registeredRegion.region = resource.region;
        registeredRegion.handle = resource.result.handle;
        registeredRegions.emplace_back(registeredRegion);
    }

    std::vector<RegisterResult> results;
    auto status = transport->BindRegisteredRegions(registeredRegions, results);
    if (!status.ok()) {
        return WithContext(status, "asuId=" + std::to_string(asuId) +
                                       " resource_count=" + std::to_string(resources.size()));
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

bool AsuClientImpl::IsTaskComplete(const TaskResult& result)
{
    if (!IsTaskStatusComplete(result.status)) { return false; }
    return std::all_of(result.entryStatus.begin(), result.entryStatus.end(),
                       [](const Status& status) { return IsTaskStatusComplete(status); });
}

bool AsuClientImpl::IsTaskStatusComplete(const Status& status)
{
    return status.code != StatusCode::IN_PROGRESS && status.code != StatusCode::TIMEOUT;
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

}  // namespace UC::ASU
