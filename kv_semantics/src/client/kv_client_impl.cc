/**
 * MIT License
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "kv_client_impl.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <iterator>
#include <limits>
#include <thread>
#include <utility>
#include "device.h"
#include "kv_metrics/metrics.h"
#include "kv_types.h"
#include "logger.h"
#include "router/config.h"
#include "router/router.h"
#include "transport/asu_transport_pool.h"
#include "utils/config_utils.h"
#include "view/m800_view_slave_server.h"
#include "view/threaded_refresh_scheduler.h"

namespace kv {

namespace {

struct SubmitMetrics {
    metrics::CachedMetric* requests;
    metrics::CachedMetric* entries;
    metrics::CachedMetric* errors;
};

bool GetSubmitMetrics(AsuOpType opType, SubmitMetrics& ids)
{
    switch (opType) {
        case AsuOpType::QUERY:
            ids = {&KV_METRIC("kv_client_query_requests_total"),
                   &KV_METRIC("kv_client_query_entries_total"),
                   &KV_METRIC("kv_client_query_errors_total")};
            return true;
        case AsuOpType::LOAD:
        case AsuOpType::BATCH_LOAD:
            ids = {&KV_METRIC("kv_client_load_requests_total"),
                   &KV_METRIC("kv_client_load_entries_total"),
                   &KV_METRIC("kv_client_load_errors_total")};
            return true;
        case AsuOpType::STORE:
        case AsuOpType::BATCH_STORE:
            ids = {&KV_METRIC("kv_client_store_requests_total"),
                   &KV_METRIC("kv_client_store_entries_total"),
                   &KV_METRIC("kv_client_store_errors_total")};
            return true;
        case AsuOpType::DELETE:
            ids = {&KV_METRIC("kv_client_delete_requests_total"),
                   &KV_METRIC("kv_client_delete_entries_total"),
                   &KV_METRIC("kv_client_delete_errors_total")};
            return true;
        default: return false;
    }
}

void RecordSubmit(AsuOpType opType, std::size_t entryCount, const Status& status)
{
    if (!metrics::IsEnabled()) { return; }
    SubmitMetrics ids{};
    if (!GetSubmitMetrics(opType, ids)) { return; }
    metrics::MetricUpdate updates[] = {
        {ids.requests, 1.0                            },
        {ids.entries,  static_cast<double>(entryCount)},
        {ids.errors,   1.0                            },
    };
    metrics::UpdateStats(updates, status.ok() ? std::size(updates) - 1 : std::size(updates));
}

}  // namespace

constexpr std::uint32_t kMaxShutdownDrainAttempts = 64;

KvClientImpl::KvClientImpl(TransportFactory transportFactory, ViewServerFactory viewServerFactory,
                           TransProviderFactory transProviderFactory)
    : transportFactory_(std::move(transportFactory)),
      transProviderFactory_(std::move(transProviderFactory)),
      viewServerFactory_(std::move(viewServerFactory)),
      transportPool_(
          [this](const TransportConfig& config, std::shared_ptr<AsuTransport>& transport) {
              return CreateTransportForNode(config, transport);
          })
{
    if (!transportFactory_) { transportFactory_ = CreateAsuTransport; }
    if (!transProviderFactory_) { transProviderFactory_ = CreateTransProvider; }
    if (!viewServerFactory_) { viewServerFactory_ = CreateDefaultViewServer; }
}

KvClientImpl::~KvClientImpl() { Shutdown(); }

Status KvClientImpl::Init(const KvClientConfig& config)
{
    if (initialized_) {
        return Status::Error(StatusCode::RESOURCE_BUSY, "asu client has already been initialized");
    }
    if (config.sharedProviderMode != SharedProviderMode::INDEPENDENT &&
        config.sharedProviderMode != SharedProviderMode::SHARED) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "sharedProviderMode must be 0 or 1");
    }
    // Validate the config before touching any state, so an invalid config leaves nothing
    // to roll back.
    if (config.transportConfigs.empty()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "at least one transport config is required");
    }

    config_ = config;
    transportConfigs_.clear();
    for (const auto& transportConfig : config.transportConfigs) {
        transportConfigs_[transportConfig.nodeId] = transportConfig;
    }

    viewServer_ = viewServerFactory_(config);
    if (viewServer_ == nullptr) {
        return FailInit(
            Status::Error(StatusCode::NOT_INITIALIZED, "view server factory returned null"));
    }

    const auto queueDepth =
        std::max<std::size_t>(2, static_cast<std::size_t>(config.maxInflightTasks));

    auto universe = CreateViewUniverse();
    auto status = FetchCandidateUniverse(*viewServer_, *universe);
    if (!status.ok()) { return FailInit(status); }
    if (universe->Empty()) {
        return FailInit(
            Status::Error(StatusCode::NOT_INITIALIZED, "view server returned no namespace view"));
    }

    std::shared_ptr<TransProvider> memoryProvider;
    status = transProviderFactory_(config.transportConfigs.front(), memoryProvider);
    if (!status.ok()) {
        return FailInit(WithContext(status, "create business-memory provider failed"));
    }
    if (!memoryProvider) {
        return FailInit(Status::Error(StatusCode::INTERNAL_ERROR,
                                      "business-memory provider factory returned null"));
    }
    {
        std::lock_guard<std::mutex> memoryLock{memoryMu_};
        memoryProvider_ = std::move(memoryProvider);
        providerMemoryStates_.push_back({memoryProvider_, {}});
    }

    viewUniverse_ = CreateViewUniverse();
    ViewSnapshotMap initialSnapshots;
    status = BuildSnapshots(*universe, initialSnapshots);
    if (status.ok()) { status = ApplyUniverse(*universe, std::move(initialSnapshots), true); }
    if (!status.ok()) { return FailInit(status); }

    taskQueue_.Setup(queueDepth + 1);
    stopWorker_.store(false, std::memory_order_release);
    worker_ = std::thread(&KvClientImpl::WorkerLoop, this);
    initialized_ = true;

    // TODO: a factory indirection is only worth adding when a second scheduler
    // implementation appears; the resident threaded scheduler is the only one today.
    refreshScheduler_ = std::make_unique<ThreadedRefreshScheduler>();
    const auto interval = GetRefreshInterval(config);
    const auto startStatus =
        refreshScheduler_->Start(interval, [this] { return OnViewUniverseUpdated(); });
    if (!startStatus.ok()) {
        // Without a scheduler the client cannot refresh, so a half-initialized client is
        // not acceptable: roll back and report the failure.
        KV_ERROR("Refresh scheduler start failed: code={} message={}",
                 static_cast<int>(startStatus.code), startStatus.message);
        return FailInit(startStatus);
    }
    return Status::OK();
}

Status KvClientImpl::FailInit(Status status)
{
    // Reuse the normal shutdown path so a failed Init leaves no view server, transport
    // pool or provider state behind, exactly like a successfully shut down client.
    (void)Shutdown();
    return status;
}

Status KvClientImpl::Shutdown()
{
    std::uint64_t waitTimeoutMs = 0;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        initialized_ = false;
        waitTimeoutMs = config_.defaultWaitTimeoutMs;
    }
    {
        std::lock_guard<std::mutex> lock{producerMu_};
        stopWorker_.store(true, std::memory_order_release);
    }
    workerCv_.notify_one();
    StopRefresh();
    if (worker_.joinable()) { worker_.join(); }

    std::vector<ProviderMemoryState> providerMemoryStates;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        snapshots_.clear();
        viewUniverse_.reset();
        config_ = KvClientConfig{};
        viewServer_.reset();
        transportConfigs_.clear();
    }
    {
        std::lock_guard<std::mutex> lock{memoryMu_};
        providerMemoryStates = std::move(providerMemoryStates_);
        memoryProvider_.reset();
        registeredRegions_.clear();
    }

    Status finalStatus = Status::OK();
    auto drainStatus = taskManager_.Drain(waitTimeoutMs);
    if (!drainStatus.ok()) { finalStatus = drainStatus; }
    // The pool owns every transport, so closing it once releases all connections.
    auto poolStatus = transportPool_.ShutdownAll();
    if (!poolStatus.ok() && finalStatus.ok()) { finalStatus = poolStatus; }
    const auto unregisterState = [this, &finalStatus](const ProviderMemoryState& state) {
        std::vector<MRHandle> localHandles;
        localHandles.reserve(state.regionHandles.size());
        for (const auto& item : state.regionHandles) { localHandles.emplace_back(item.second); }
        auto status = UnregisterProviderRegions(state.provider, localHandles);
        if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
    };
    for (auto iter = providerMemoryStates.rbegin(); iter != providerMemoryStates.rend(); ++iter) {
        unregisterState(*iter);
    }
    return finalStatus;
}

Status KvClientImpl::QueryAsync(const std::vector<CacheKey>& keys, TaskId& taskId)
{
    return QueryAsync(kDefaultKvNsId, keys, taskId);
}

Status KvClientImpl::QueryAsync(KvNsId nsId, const std::vector<CacheKey>& keys, TaskId& taskId)
{
    // A submission failure means the caller passed something the published view cannot
    // serve (unknown namespace, missing namespace for a multi-namespace client, queue
    // full). Refreshing on it would turn caller errors into refresh traffic; the refresh
    // is driven by execution results instead (Wait and the worker path), like the other
    // operations.
    auto status = SubmitAsync(AsuOpType::QUERY, nsId, keys, taskId);
    RecordSubmit(AsuOpType::QUERY, keys.size(), status);
    return status;
}

Status KvClientImpl::LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    return LoadAsync(kDefaultKvNsId, entries, taskId);
}

Status KvClientImpl::LoadAsync(KvNsId nsId, const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    const auto useBatchIo =
        entries.size() > 1 || (entries.size() == 1 && config_.useBatchIoForSingleKey);
    const auto opType = useBatchIo ? AsuOpType::BATCH_LOAD : AsuOpType::LOAD;
    auto status = SubmitAsync(opType, nsId, entries, taskId);
    RecordSubmit(opType, entries.size(), status);
    return status;
}

Status KvClientImpl::StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId,
                                std::function<Status()> prerequisite)
{
    return StoreAsync(kDefaultKvNsId, entries, taskId, std::move(prerequisite));
}

Status KvClientImpl::StoreAsync(KvNsId nsId, const std::vector<KVBuffer>& entries, TaskId& taskId,
                                std::function<Status()> prerequisite)
{
    const auto useBatchIo =
        entries.size() > 1 || (entries.size() == 1 && config_.useBatchIoForSingleKey);
    const auto opType = useBatchIo ? AsuOpType::BATCH_STORE : AsuOpType::STORE;
    auto status = SubmitAsync(opType, nsId, entries, taskId, std::move(prerequisite));
    RecordSubmit(opType, entries.size(), status);
    return status;
}

Status KvClientImpl::DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId)
{
    return DeleteAsync(kDefaultKvNsId, keys, taskId);
}

Status KvClientImpl::DeleteAsync(KvNsId nsId, const std::vector<CacheKey>& keys, TaskId& taskId)
{
    auto status = SubmitAsync(AsuOpType::DELETE, nsId, keys, taskId);
    RecordSubmit(AsuOpType::DELETE, keys.size(), status);
    return status;
}

bool KvClientImpl::Check(TaskId taskId) { return taskManager_.Check(taskId); }

Status KvClientImpl::Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result)
{
    const auto waitMs = timeoutMs == 0 ? config_.defaultWaitTimeoutMs : timeoutMs;
    auto status = taskManager_.Wait(taskId, waitMs, result);
    if (metrics::IsEnabled()) {
        metrics::MetricUpdate updates[] = {
            {KV_METRIC("kv_client_wait_requests_total"), 1.0},
            {KV_METRIC("kv_client_wait_errors_total"),   1.0},
        };
        const bool success = status.ok() && result.status.ok();
        metrics::UpdateStats(updates, success ? std::size(updates) - 1 : std::size(updates));
    }
    if (status.code == StatusCode::TASK_NOT_FOUND) { return status; }
    if (viewServer_ != nullptr &&
        (viewServer_->ShouldRefreshView(status) || viewServer_->ShouldRefreshView(result))) {
        TriggerRefresh(status.ok() ? result.status : status);
    }
    return status;
}

Status KvClientImpl::RegisterRegions(const std::vector<MemoryRegion>& regions,
                                     std::vector<RegisteredMemory>& registeredRegions)
{
    bool needRefresh = false;
    auto status = RegisterRegionsOnce(regions, registeredRegions, needRefresh);
    if (needRefresh) {
        TriggerRefresh(
            Status::Error(StatusCode::NOT_FOUND, "registered memory requires a refreshed view"));
    }
    return status;
}

Status KvClientImpl::RegisterRegionsOnce(const std::vector<MemoryRegion>& regions,
                                         std::vector<RegisteredMemory>& registeredRegions,
                                         bool& needRefresh)
{
    auto snapshot = GetAnySnapshot();
    if (!snapshot) { return NotInitialized(); }

    registeredRegions.clear();
    if (snapshot->transports.empty()) { return Status::OK(); }
    if (regions.empty()) { return Status::OK(); }

    std::lock_guard<std::mutex> memoryLock{memoryMu_};
    if (!memoryProvider_) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "client business-memory provider is not initialized");
    }

    std::vector<TransProvider::RegisterMemoryDesc> registerDescs;
    registerDescs.reserve(regions.size());
    for (const auto& region : regions) {
        const auto memType = region.memoryType == MemoryType::DEVICE
                                 ? TransProvider::MemType::MEM_DEVICE
                                 : TransProvider::MemType::MEM_HOST;
        registerDescs.push_back({memType, static_cast<std::uintptr_t>(region.addr),
                                 static_cast<std::size_t>(region.size)});
    }

    std::vector<MRHandle> mrHandles;
    auto status = memoryProvider_->RegisterMemory(registerDescs, mrHandles);
    if (!status.ok()) {
        needRefresh |= IsRefreshNeeded(status);
        const auto cleanupStatus = UnregisterProviderRegions(memoryProvider_, mrHandles);
        return Status::Error(StatusCode::PARTIAL_FAILED,
                             cleanupStatus.ok()
                                 ? "one or more memory regions failed to register"
                                 : "memory region registration failed and cleanup was incomplete");
    }
    if (mrHandles.size() != regions.size()) {
        const auto cleanupStatus = UnregisterProviderRegions(memoryProvider_, mrHandles);
        return Status::Error(StatusCode::PARTIAL_FAILED,
                             cleanupStatus.ok()
                                 ? "register result count does not match region count"
                                 : "register result count mismatch and cleanup was incomplete");
    }

    std::vector<std::uint32_t> tokenIds(regions.size());
    for (std::size_t index = 0; index < mrHandles.size(); ++index) {
        status = memoryProvider_->GetMemTokenId(mrHandles[index], tokenIds[index]);
        if (status.ok()) { continue; }

        const auto cleanupStatus = UnregisterProviderRegions(memoryProvider_, mrHandles);
        return Status::Error(StatusCode::PARTIAL_FAILED,
                             cleanupStatus.ok()
                                 ? "one or more memory region token lookups failed"
                                 : "memory region token lookup failed and cleanup was incomplete");
    }

    for (std::size_t index = 0; index < regions.size(); ++index) {
        registeredRegions.emplace_back(
            RegisteredMemory{regions[index], mrHandles[index], tokenIds[index]});
    }

    std::vector<std::vector<MRHandle>> providerHandles(providerMemoryStates_.size());
    for (std::size_t index = 0; index < providerMemoryStates_.size(); ++index) {
        auto& state = providerMemoryStates_[index];
        if (state.provider == memoryProvider_) {
            providerHandles[index] = mrHandles;
            continue;
        }
        status = BindProviderRegions(state.provider, registeredRegions, providerHandles[index]);
        if (status.ok()) { continue; }

        for (std::size_t rollback = index; rollback > 0; --rollback) {
            const auto stateIndex = rollback - 1;
            if (providerHandles[stateIndex].empty()) { continue; }
            (void)UnregisterProviderRegions(providerMemoryStates_[stateIndex].provider,
                                            providerHandles[stateIndex]);
        }
        registeredRegions.clear();
        return WithContext(Status::Error(StatusCode::PARTIAL_FAILED,
                                         "one or more providers failed to bind memory"),
                           "providerIndex=" + std::to_string(index) + " message=" + status.message);
    }

    for (std::size_t providerIndex = 0; providerIndex < providerMemoryStates_.size();
         ++providerIndex) {
        auto& handleMap = providerMemoryStates_[providerIndex].regionHandles;
        for (std::size_t regionIndex = 0; regionIndex < registeredRegions.size(); ++regionIndex) {
            handleMap.emplace(registeredRegions[regionIndex].handle,
                              providerHandles[providerIndex][regionIndex]);
        }
    }
    for (const auto& registeredRegion : registeredRegions) {
        registeredRegions_.emplace(registeredRegion.handle, registeredRegion);
    }
    return Status::OK();
}

Status KvClientImpl::SubmitAsync(AsuOpType opType, const std::vector<KVBuffer>& entries,
                                 TaskId& taskId, std::function<Status()> prerequisite)
{
    return SubmitAsync(opType, kDefaultKvNsId, entries, taskId, std::move(prerequisite));
}

Status KvClientImpl::SubmitAsync(AsuOpType opType, KvNsId nsId,
                                 const std::vector<KVBuffer>& entries, TaskId& taskId,
                                 std::function<Status()> prerequisite)
{
    const auto taskStart = std::chrono::steady_clock::now();
    std::shared_ptr<ViewSnapshot> snapshot;
    auto resolveStatus = ResolveSnapshot(nsId, snapshot);
    if (!resolveStatus.ok()) {
        taskId = kInvalidTaskId;
        return resolveStatus;
    }
    if (!snapshot->router || snapshot->transports.empty()) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    if (opType != AsuOpType::LOAD && opType != AsuOpType::STORE &&
        opType != AsuOpType::BATCH_LOAD && opType != AsuOpType::BATCH_STORE) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "entries submit only supports load/store");
    }

    auto ctx = std::make_unique<ClientTask>();
    ctx->submittedAt = taskStart;
    ctx->opType = opType;
    ctx->prerequisite = std::move(prerequisite);
    ctx->viewSnapshot = snapshot;
    // The namespace is the snapshot's; kDefaultKvNsId is used when the caller has none.
    ctx->nsId = snapshot->nsId;
    ctx->entries = entries;
    {
        std::lock_guard<std::mutex> memoryLock{memoryMu_};
        for (auto& entry : ctx->entries) {
            // mrKey is client-owned metadata. Ignore any value supplied by the caller.
            entry.mrKey.reset();
            auto iter = registeredRegions_.find(entry.buffer.handle);
            if (iter != registeredRegions_.end()) { entry.mrKey = iter->second.tokenId; }
        }
    }
    ctx->entryStatus.assign(entries.size(), Status::OK());

    auto status = taskManager_.Submit(std::move(ctx), taskId);
    if (!status.ok()) { return status; }

    auto rawCtx = taskManager_.Get(taskId);
    if (!rawCtx) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "client task disappeared after submit");
    }

    std::chrono::steady_clock::time_point enqueuedAt;
    {
        std::lock_guard<std::mutex> lock{producerMu_};
        if (stopWorker_.load(std::memory_order_acquire)) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return NotInitialized();
        }
        enqueuedAt = std::chrono::steady_clock::now();
        rawCtx->enqueuedAt = enqueuedAt;
        if (!taskQueue_.TryPush(std::move(rawCtx))) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::RESOURCE_BUSY, "client task queue is full");
        }
    }
    workerCv_.notify_one();
    metrics::UpdateStats(KV_METRIC("kv_client_task_enqueue_duration_seconds"),
                         std::chrono::duration<double>(enqueuedAt - taskStart).count());
    return Status::OK();
}

Status KvClientImpl::SubmitAsync(AsuOpType opType, const std::vector<CacheKey>& keys,
                                 TaskId& taskId)
{
    return SubmitAsync(opType, kDefaultKvNsId, keys, taskId);
}

Status KvClientImpl::SubmitAsync(AsuOpType opType, KvNsId nsId, const std::vector<CacheKey>& keys,
                                 TaskId& taskId)
{
    const auto taskStart = std::chrono::steady_clock::now();
    std::shared_ptr<ViewSnapshot> snapshot;
    auto resolveStatus = ResolveSnapshot(nsId, snapshot);
    if (!resolveStatus.ok()) {
        taskId = kInvalidTaskId;
        return resolveStatus;
    }
    if (!snapshot->router || snapshot->transports.empty()) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    if (opType != AsuOpType::QUERY && opType != AsuOpType::DELETE) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "keys submit only supports query/delete");
    }

    auto ctx = std::make_unique<ClientTask>();
    ctx->submittedAt = taskStart;
    ctx->opType = opType;
    ctx->viewSnapshot = snapshot;
    // The namespace is the snapshot's; kDefaultKvNsId is used when the caller has none.
    ctx->nsId = snapshot->nsId;
    ctx->keys = keys;
    ctx->entryStatus.assign(keys.size(), Status::OK());
    ctx->queryResult.exists.assign(opType == AsuOpType::QUERY ? keys.size() : 0, 0);

    auto status = taskManager_.Submit(std::move(ctx), taskId);
    if (!status.ok()) { return status; }

    auto rawCtx = taskManager_.Get(taskId);
    if (!rawCtx) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "client task disappeared after submit");
    }

    std::chrono::steady_clock::time_point enqueuedAt;
    {
        std::lock_guard<std::mutex> lock{producerMu_};
        if (stopWorker_.load(std::memory_order_acquire)) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return NotInitialized();
        }
        enqueuedAt = std::chrono::steady_clock::now();
        rawCtx->enqueuedAt = enqueuedAt;
        if (!taskQueue_.TryPush(std::move(rawCtx))) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::RESOURCE_BUSY, "client task queue is full");
        }
    }
    workerCv_.notify_one();
    metrics::UpdateStats(KV_METRIC("kv_client_task_enqueue_duration_seconds"),
                         std::chrono::duration<double>(enqueuedAt - taskStart).count());
    return Status::OK();
}

void KvClientImpl::WorkerLoop()
{
    runtime::Device device;
    const auto deviceId = config_.transportConfigs.front().deviceId;
    const auto deviceStatus = deviceId >= 0 ? device.Setup(deviceId) : Status::OK();
    auto processTask = [this, &deviceStatus](ClientTaskPtr ctx) {
        ctx->processingStartedAt = std::chrono::steady_clock::now();
        metrics::UpdateStats(
            KV_METRIC("kv_client_task_queue_duration_seconds"),
            std::chrono::duration<double>(ctx->processingStartedAt - ctx->enqueuedAt).count());
        auto status = deviceStatus;
        if (status.ok() && ctx->prerequisite) {
            try {
                status = ctx->prerequisite();
            } catch (const std::exception& error) {
                status = Status::Error(StatusCode::INTERNAL_ERROR, error.what());
            } catch (...) {
                status = Status::Error(StatusCode::INTERNAL_ERROR, "prerequisite callback threw");
            }
        }
        if (status.ok()) {
            status = taskManager_.Process(ctx);
        } else {
            ClientTaskManager::CompleteWithError(ctx, status);
        }
        if (IsRefreshNeeded(status)) { TriggerRefresh(status); }
    };
    taskQueue_.ConsumerLoop(stopWorker_, producerMu_, workerCv_, processTask);

    ClientTaskPtr ctx;
    while (taskQueue_.TryPop(ctx)) { processTask(std::move(ctx)); }
}

Status KvClientImpl::UnregisterRegions(const std::vector<MRHandle>& handles)
{
    std::lock_guard<std::mutex> memoryLock{memoryMu_};
    std::vector<MRHandle> canonicalHandles;
    canonicalHandles.reserve(handles.size());
    for (auto handle : handles) {
        if (handle == kInvalidMRHandle) { continue; }
        if (registeredRegions_.count(handle) != 0) { canonicalHandles.emplace_back(handle); }
    }

    Status finalStatus = Status::OK();
    const auto unregisterState = [this, &canonicalHandles,
                                  &finalStatus](ProviderMemoryState& state) {
        std::vector<MRHandle> localHandles;
        std::vector<MRHandle> mappedCanonicalHandles;
        for (auto canonicalHandle : canonicalHandles) {
            auto iter = state.regionHandles.find(canonicalHandle);
            if (iter == state.regionHandles.end()) { continue; }
            mappedCanonicalHandles.emplace_back(canonicalHandle);
            localHandles.emplace_back(iter->second);
        }
        auto status = UnregisterProviderRegions(state.provider, localHandles);
        if (!status.ok()) {
            if (finalStatus.ok()) { finalStatus = status; }
            return;
        }
        for (auto canonicalHandle : mappedCanonicalHandles) {
            state.regionHandles.erase(canonicalHandle);
        }
    };

    for (auto iter = providerMemoryStates_.rbegin(); iter != providerMemoryStates_.rend(); ++iter) {
        unregisterState(*iter);
    }

    std::vector<MRHandle> removedHandles;
    for (auto canonicalHandle : canonicalHandles) {
        const auto isStillRegistered =
            std::any_of(providerMemoryStates_.begin(), providerMemoryStates_.end(),
                        [canonicalHandle](const ProviderMemoryState& state) {
                            return state.regionHandles.count(canonicalHandle) != 0;
                        });
        if (!isStillRegistered) { removedHandles.emplace_back(canonicalHandle); }
    }
    for (auto handle : removedHandles) { registeredRegions_.erase(handle); }

    if (!finalStatus.ok()) {
        return WithContext(finalStatus, "handle_count=" + std::to_string(canonicalHandles.size()));
    }
    return finalStatus;
}

Status KvClientImpl::BuildSnapshot(KvNsId nsId, const GlobalView& view,
                                   std::shared_ptr<ViewSnapshot>& snapshot)
{
    auto nextSnapshot = std::make_shared<ViewSnapshot>();
    auto asuIds = GetSortedAsuIds(view);
    nextSnapshot->view = view;
    nextSnapshot->nsId = nsId;

    for (std::size_t asuIndex = 0; asuIndex < asuIds.size(); ++asuIndex) {
        auto nodeId = asuIds[asuIndex];
        auto viewIter = view.asuMap.find(nodeId);
        auto asuInfo = viewIter == view.asuMap.end() ? NodeInfo{} : viewIter->second;
        std::shared_ptr<AsuTransport> transport;
        // The pool owns the transport; the snapshot only keeps a reference to it.
        auto status = GetOrCreateTransport(nodeId, asuInfo, transport);
        if (!status.ok()) {
            return WithContext(status, "asuIndex=" + std::to_string(asuIndex) +
                                           " nodeId=" + std::to_string(nodeId));
        }

        nextSnapshot->transports.emplace(nodeId, std::move(transport));
    }

    kv::RouterConfig routerConfig;
    auto status = BuildRouterConfigFromAttrs(config_.attrs, routerConfig);
    if (!status.ok()) {
        KV_ERROR("BuildSnapshot build router config failed: {}", status.message);
        return status;
    }

    std::vector<kv::NodeId> nodeIds(asuIds.begin(), asuIds.end());
    nextSnapshot->router = kv::CreateRouter(nodeIds, kv::HashFunction{}, routerConfig);
    nextSnapshot->asuIds = std::move(asuIds);
    snapshot = std::move(nextSnapshot);
    return Status::OK();
}

Status KvClientImpl::CreateTransportForNode(const TransportConfig& config,
                                            std::shared_ptr<AsuTransport>& transport)
{
    const auto nodeId = config.nodeId;
    std::lock_guard<std::mutex> memoryLock{memoryMu_};
    std::shared_ptr<TransProvider> transProvider;
    if (config_.sharedProviderMode == SharedProviderMode::SHARED) {
        transProvider = memoryProvider_;
    }
    const bool createdProvider = !transProvider;
    if (createdProvider) {
        auto status = transProviderFactory_(config, transProvider);
        if (!status.ok()) {
            return WithContext(
                status, "create transport provider failed, nodeId=" + std::to_string(nodeId));
        }
        if (!transProvider) {
            return Status::Error(
                StatusCode::INTERNAL_ERROR,
                "transport provider factory returned null, nodeId=" + std::to_string(nodeId));
        }
    }

    auto nextTransport = transportFactory_();
    if (!nextTransport) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "transport factory returned null, nodeId=" + std::to_string(nodeId));
    }

    auto status = nextTransport->Init(config, transProvider);
    if (!status.ok()) {
        return WithContext(status, "init transport failed, nodeId=" + std::to_string(nodeId));
    }
    if (!createdProvider) {
        transport = std::shared_ptr<AsuTransport>(std::move(nextTransport));
        return Status::OK();
    }

    std::vector<RegisteredMemory> regionsToBind;
    regionsToBind.reserve(registeredRegions_.size());
    for (const auto& item : registeredRegions_) { regionsToBind.emplace_back(item.second); }

    std::vector<MRHandle> localHandles;
    if (!regionsToBind.empty()) {
        status = BindProviderRegions(transProvider, regionsToBind, localHandles);
        if (!status.ok()) {
            (void)nextTransport->Shutdown();
            return WithContext(status,
                               "bind provider memory failed, nodeId=" + std::to_string(nodeId));
        }
    }
    ProviderMemoryState providerState;
    providerState.provider = transProvider;
    for (std::size_t index = 0; index < regionsToBind.size(); ++index) {
        providerState.regionHandles.emplace(regionsToBind[index].handle, localHandles[index]);
    }
    providerMemoryStates_.emplace_back(std::move(providerState));
    transport = std::shared_ptr<AsuTransport>(std::move(nextTransport));
    return Status::OK();
}

Status KvClientImpl::BuildTransportConfigForNode(NodeId nodeId, const NodeInfo& asuInfo,
                                                 TransportConfig& config) const
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        auto configIter = transportConfigs_.find(nodeId);
        if (configIter != transportConfigs_.end()) {
            config = configIter->second;
        } else if (!config_.transportConfigs.empty()) {
            // Dynamic node: clone the first configured entry, its nodeId is a placeholder.
            config = config_.transportConfigs.front();
        } else {
            return Status::Error(
                StatusCode::NOT_FOUND,
                "transport config template not found, nodeId=" + std::to_string(nodeId));
        }
    }

    config.nodeId = nodeId;
    ApplyAsuInfoToTransportConfig(asuInfo, config);
    // The namespace id belongs to each transport task, never to a shared transport config.
    config.attrs.erase("kv_ns_id");
    return Status::OK();
}

Status KvClientImpl::GetOrCreateTransport(NodeId nodeId, const NodeInfo& asuInfo,
                                          std::shared_ptr<AsuTransport>& transport)
{
    TransportConfig config;
    auto status = BuildTransportConfigForNode(nodeId, asuInfo, config);
    if (!status.ok()) { return status; }

    status = transportPool_.GetOrCreate(config, transport);
    if (status.code != StatusCode::ENDPOINT_CHANGED) { return status; }

    // The viewed endpoints moved; keep IO running by rebuilding this node's connection.
    return transportPool_.Replace(config, transport);
}

Status KvClientImpl::BindProviderRegions(const std::shared_ptr<TransProvider>& transProvider,
                                         const std::vector<RegisteredMemory>& registeredRegions,
                                         std::vector<MRHandle>& localHandles)
{
    localHandles.clear();
    if (registeredRegions.empty()) { return Status::OK(); }
    if (!transProvider) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "transport provider is not initialized");
    }

    std::vector<TransProvider::BindMemoryDesc> bindDescs;
    bindDescs.reserve(registeredRegions.size());
    for (const auto& registeredRegion : registeredRegions) {
        const auto memType = registeredRegion.region.memoryType == MemoryType::DEVICE
                                 ? TransProvider::MemType::MEM_DEVICE
                                 : TransProvider::MemType::MEM_HOST;
        bindDescs.push_back({memType, static_cast<std::uintptr_t>(registeredRegion.region.addr),
                             static_cast<std::size_t>(registeredRegion.region.size),
                             registeredRegion.tokenId});
    }

    auto status = transProvider->BindMemory(bindDescs, localHandles);
    if (!status.ok() || localHandles.size() != registeredRegions.size()) {
        (void)UnregisterProviderRegions(transProvider, localHandles);
        localHandles.clear();
        return status.ok() ? Status::Error(StatusCode::INTERNAL_ERROR,
                                           "bind result count does not match region count")
                           : status;
    }

    for (std::size_t index = 0; index < localHandles.size(); ++index) {
        std::uint32_t tokenId = 0;
        status = transProvider->GetMemTokenId(localHandles[index], tokenId);
        if (status.ok() && tokenId == registeredRegions[index].tokenId) { continue; }

        (void)UnregisterProviderRegions(transProvider, localHandles);
        localHandles.clear();
        return status.ok() ? Status::Error(StatusCode::INTERNAL_ERROR,
                                           "bound memory token does not match registered token")
                           : status;
    }
    return Status::OK();
}

Status KvClientImpl::UnregisterProviderRegions(const std::shared_ptr<TransProvider>& transProvider,
                                               const std::vector<MRHandle>& handles)
{
    if (handles.empty()) { return Status::OK(); }
    if (!transProvider) {
        return Status::Error(StatusCode::NOT_INITIALIZED,
                             "client business-memory provider is not initialized");
    }

    std::vector<TransProvider::UnregisterMemoryDesc> unregisterDescs;
    unregisterDescs.reserve(handles.size());
    for (auto handle : handles) {
        unregisterDescs.emplace_back(TransProvider::UnregisterMemoryDesc{handle});
    }

    const auto statuses = transProvider->UnregisterMemory(unregisterDescs);
    Status failure = statuses.size() == handles.size()
                         ? Status::OK()
                         : Status::Error(StatusCode::INTERNAL_ERROR,
                                         "unregister result count does not match handle count");
    for (std::size_t index = 0; index < handles.size(); ++index) {
        if ((index >= statuses.size() || !statuses[index].ok()) && failure.ok()) {
            failure = index < statuses.size()
                          ? statuses[index]
                          : Status::Error(StatusCode::INTERNAL_ERROR,
                                          "unregister result count does not match handle count");
        }
    }
    return failure;
}

Status KvClientImpl::ValidateCandidateUniverse(const ViewUniverse& candidate,
                                               std::uint64_t publishedEpoch) const
{
    if (candidate.Empty()) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "candidate view has no namespace");
    }

    const auto candidateEpoch = candidate.GetGlobalEpoch();
    if (candidateEpoch < publishedEpoch) {
        return Status::Error(
            StatusCode::INTERNAL_ERROR,
            "candidate globalEpoch is older: candidate=" + std::to_string(candidateEpoch) +
                " published=" + std::to_string(publishedEpoch));
    }

    for (auto nsId : candidate.GetAllNsIds()) {
        GlobalView view;
        auto status = candidate.GetView(nsId, view);
        if (!status.ok()) { return WithContext(status, "nsId=" + std::to_string(nsId)); }
        if (GetSortedAsuIds(view).empty()) {
            return Status::Error(
                StatusCode::INTERNAL_ERROR,
                "candidate namespace has no routable ASU, nsId=" + std::to_string(nsId));
        }
        if (view.viewEpoch > candidateEpoch) {
            return Status::Error(
                StatusCode::INTERNAL_ERROR,
                "namespace viewEpoch exceeds globalEpoch, nsId=" + std::to_string(nsId) +
                    " viewEpoch=" + std::to_string(view.viewEpoch) +
                    " globalEpoch=" + std::to_string(candidateEpoch));
        }
    }
    return Status::OK();
}

Status KvClientImpl::BuildSnapshots(const ViewUniverse& candidate, ViewSnapshotMap& nextSnapshots)
{
    // The published table is only read for reuse; the refresh thread is the only writer.
    ViewSnapshotMap published;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        published = snapshots_;
    }

    for (auto nsId : candidate.GetAllNsIds()) {
        GlobalView view;
        auto status = candidate.GetView(nsId, view);
        if (!status.ok()) { return WithContext(status, "nsId=" + std::to_string(nsId)); }

        ViewSnapshotPtr publishedSnapshot;
        auto publishedIter = published.find(nsId);
        if (publishedIter != published.end()) { publishedSnapshot = publishedIter->second; }

        if (publishedSnapshot != nullptr && viewServer_ != nullptr &&
            !viewServer_->ShouldPublishView(publishedSnapshot->view, view)) {
            // Same or older epoch: the published snapshot of this namespace is kept.
            nextSnapshots.emplace(nsId, publishedSnapshot);
            continue;
        }

        ViewSnapshotPtr nextSnapshot;
        status = BuildSnapshot(nsId, view, nextSnapshot);
        if (!status.ok()) { return WithContext(status, "nsId=" + std::to_string(nsId)); }
        nextSnapshots.emplace(nsId, std::move(nextSnapshot));
    }
    return Status::OK();
}

Status KvClientImpl::FetchCandidateUniverse(ViewServer& viewServer, ViewUniverse& candidate)
{
    candidate.Clear();

    // M800 slaves deliver a complete universe. Legacy file/config servers expose a single
    // global view, which is adapted to the default namespace.
    if (auto* slave = dynamic_cast<M800ViewSlaveServer*>(&viewServer)) {
        return slave->GetAllViews(candidate);
    }

    GlobalView view;
    auto status = viewServer.GetGlobalView(view);
    if (!status.ok()) { return status; }

    const auto globalEpoch = view.viewEpoch;
    std::unordered_map<KvNsId, GlobalView> views;
    views.emplace(kDefaultKvNsId, std::move(view));
    auto single = CreateViewUniverse(std::move(views), globalEpoch);
    if (single == nullptr) {
        return Status::Error(StatusCode::INTERNAL_ERROR, "view universe factory returned null");
    }
    return candidate.ReplaceAll(*single);
}

Status KvClientImpl::ApplyUniverse(const ViewUniverse& candidate, ViewSnapshotMap&& nextSnapshots,
                                   bool initial)
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (viewUniverse_ == nullptr) { viewUniverse_ = CreateViewUniverse(); }

    auto status =
        initial ? viewUniverse_->ReplaceAll(candidate) : viewUniverse_->PublishAll(candidate);
    if (!status.ok()) { return status; }

    // Snapshots only reference pooled transports, so dropping a snapshot never closes a
    // connection: in-flight tasks keep their snapshot and the pool keeps the transport.
    snapshots_ = std::move(nextSnapshots);
    return Status::OK();
}

Status KvClientImpl::OnViewUniverseUpdated()
{
    std::shared_ptr<ViewServer> viewServer;
    std::shared_ptr<ViewUniverse> published;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_) { return NotInitialized(); }
        viewServer = viewServer_;
        published = viewUniverse_;
    }
    if (viewServer == nullptr) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "view server is not initialized");
    }

    auto candidate = CreateViewUniverse();
    auto status = FetchCandidateUniverse(*viewServer, *candidate);
    if (!status.ok()) { return status; }

    const auto publishedEpoch = published == nullptr ? 0 : published->GetGlobalEpoch();
    status = ValidateCandidateUniverse(*candidate, publishedEpoch);
    if (!status.ok()) { return status; }

    if (published != nullptr && candidate->GetGlobalEpoch() == publishedEpoch) {
        // The contract treats an equal epoch as "nothing changed". A different namespace
        // set at the same epoch is inconsistent and must be rejected instead of applied.
        auto publishedNs = published->GetAllNsIds();
        auto candidateNs = candidate->GetAllNsIds();
        std::sort(publishedNs.begin(), publishedNs.end());
        std::sort(candidateNs.begin(), candidateNs.end());
        if (publishedNs != candidateNs) {
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "namespace set changed without a globalEpoch advance");
        }
        return Status::OK();
    }

    ViewSnapshotMap nextSnapshots;
    status = BuildSnapshots(*candidate, nextSnapshots);
    if (!status.ok()) { return status; }

    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_) { return NotInitialized(); }
    }
    // A newer global epoch is always published, even when no namespace snapshot changed:
    // publishing advances the baseline so the next refresh is not a no-op fetch.
    return ApplyUniverse(*candidate, std::move(nextSnapshots), false);
}

std::chrono::milliseconds KvClientImpl::GetRefreshInterval(const KvClientConfig& config)
{
    // TODO: the interval stays client-side and is handed to RefreshScheduler::Start, which
    // only accepts values greater than zero. Open questions: the production default (it may
    // need to follow the agent refresh period) and whether a timer-only opt-out is needed,
    // since a zero interval would also disable IO-failure refreshes.
    constexpr std::uint64_t kDefaultIntervalMs = 30000;
    constexpr std::uint64_t kMinIntervalMs = 1000;
    std::uint64_t intervalMs = kDefaultIntervalMs;
    auto iter = config.attrs.find("view.refresh_interval_ms");
    if (iter != config.attrs.end()) {
        try {
            intervalMs = std::stoull(iter->second);
        } catch (const std::exception&) {
            intervalMs = kDefaultIntervalMs;
        }
    }
    if (intervalMs < kMinIntervalMs) { intervalMs = kMinIntervalMs; }
    return std::chrono::milliseconds(intervalMs);
}

void KvClientImpl::TriggerRefresh(const Status& status)
{
    // The scheduler call is short and never re-enters the client, so holding mutex_ keeps
    // the pointer lifetime simple.
    std::lock_guard<std::mutex> lock{mutex_};
    if (refreshScheduler_ == nullptr) { return; }
    refreshScheduler_->TriggerByIoFailure(status);
}

void KvClientImpl::StopRefresh()
{
    // Stop() joins the worker and must run without mutex_: the callback takes that lock.
    std::unique_ptr<RefreshScheduler> scheduler;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        scheduler = std::move(refreshScheduler_);
    }
    if (scheduler != nullptr) { scheduler->Stop(); }
}

std::shared_ptr<ViewSnapshot> KvClientImpl::GetSnapshot(KvNsId nsId) const
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (!initialized_) { return nullptr; }
    auto iter = snapshots_.find(nsId);
    return iter == snapshots_.end() ? nullptr : iter->second;
}

std::shared_ptr<ViewSnapshot> KvClientImpl::GetSingleSnapshot() const
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (!initialized_ || snapshots_.size() != 1) { return nullptr; }
    return snapshots_.begin()->second;
}

Status KvClientImpl::ResolveSnapshot(KvNsId nsId, std::shared_ptr<ViewSnapshot>& snapshot) const
{
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_) { return NotInitialized(); }
    }

    snapshot = GetSnapshot(nsId);
    if (snapshot) { return Status::OK(); }

    if (nsId != kDefaultKvNsId) {
        return Status::Error(StatusCode::NOT_FOUND,
                             "unknown namespace, nsId=" + std::to_string(nsId));
    }

    // The caller has no namespace of its own. Accept the only published namespace so a
    // single-namespace deployment keeps working.
    snapshot = GetSingleSnapshot();
    if (snapshot) { return Status::OK(); }
    return Status::Error(StatusCode::INVALID_ARGUMENT,
                         "namespace is required for multi-namespace clients");
}

std::shared_ptr<ViewSnapshot> KvClientImpl::GetAnySnapshot() const
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (!initialized_) { return nullptr; }
    for (const auto& item : snapshots_) {
        if (item.second != nullptr) { return item.second; }
    }
    return nullptr;
}

bool KvClientImpl::IsRefreshNeeded(const Status& status) const
{
    return viewServer_ != nullptr && viewServer_->ShouldRefreshView(status);
}

std::vector<NodeId> KvClientImpl::GetSortedAsuIds(const GlobalView& view)
{
    std::vector<NodeId> asuIds;
    asuIds.reserve(view.asuMap.size());
    for (const auto& item : view.asuMap) {
        if (item.first != static_cast<NodeId>(kv::kInvalidNodeId)) {
            asuIds.emplace_back(item.first);
        }
    }
    std::sort(asuIds.begin(), asuIds.end());
    return asuIds;
}

Status KvClientImpl::WithContext(Status status, const std::string& context)
{
    if (context.empty()) { return status; }
    if (status.message.empty()) {
        status.message = context;
    } else {
        status.message += ", " + context;
    }
    return status;
}

Status KvClientImpl::NotInitialized()
{
    return Status::Error(StatusCode::NOT_INITIALIZED, "asu client is not initialized");
}

std::unique_ptr<KvClient> CreateKvClient()
{
    return std::make_unique<KvClientImpl>(nullptr, nullptr, nullptr);
}

std::unique_ptr<KvClient> CreateKvClient(TransportFactory transportFactory)
{
    return std::make_unique<KvClientImpl>(std::move(transportFactory), nullptr, nullptr);
}

std::unique_ptr<KvClient> CreateKvClient(TransportFactory transportFactory,
                                         TransProviderFactory transProviderFactory)
{
    return std::make_unique<KvClientImpl>(std::move(transportFactory), nullptr,
                                          std::move(transProviderFactory));
}

extern "C" std::unique_ptr<KvClient> UcmAsuCreateKvClient(const TransportFactory* transportFactory)
{
    if (transportFactory == nullptr) { return CreateKvClient(); }
    return CreateKvClient(*transportFactory);
}

extern "C" Status UcmAsuLoadKvClientConfig(const char* configPath, KvClientConfig* config)
{
    if (configPath == nullptr || config == nullptr) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "UcmAsuLoadKvClientConfig received null argument");
    }
    return KvClientConfig::LoadFromFile(configPath, *config);
}

}  // namespace kv
