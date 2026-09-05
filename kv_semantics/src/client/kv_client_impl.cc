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
#include "kv_client_impl.h"
#include <algorithm>
#include <limits>
#include <thread>
#include <utility>
#include "types.h"
#include "utils/config_utils.h"
#include "router/config.h"
#include "logger.h"
#include "router/router.h"

namespace kv {

constexpr std::uint32_t kMaxShutdownDrainAttempts = 64;

AsuClientImpl::AsuClientImpl(TransportFactory transportFactory, ViewServerFactory viewServerFactory,
                             TransProviderFactory transProviderFactory)
    : transportFactory_(std::move(transportFactory)),
      transProviderFactory_(std::move(transProviderFactory)),
      viewServerFactory_(std::move(viewServerFactory))
{
    if (!transportFactory_) { transportFactory_ = CreateAsuTransport; }
    if (!transProviderFactory_) { transProviderFactory_ = CreateTransProvider; }
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
    if (config.sharedProviderMode != SharedProviderMode::INDEPENDENT &&
        config.sharedProviderMode != SharedProviderMode::SHARED) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "sharedProviderMode must be 0 or 1");
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
    if (config.transportConfigs.empty()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "at least one transport config is required");
    }
    const auto queueDepth =
        std::max<std::size_t>(2, static_cast<std::size_t>(config.maxInflightTasks));

    GlobalView view;
    auto status = viewServer_->GetGlobalView(view);
    if (!status.ok()) { return status; }

    std::shared_ptr<TransProvider> memoryProvider;
    status = transProviderFactory_(config.transportConfigs.front(), memoryProvider);
    if (!status.ok()) { return WithContext(status, "create business-memory provider failed"); }
    if (!memoryProvider) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "business-memory provider factory returned null");
    }
    {
        std::lock_guard<std::mutex> memoryLock{memoryMu_};
        memoryProvider_ = std::move(memoryProvider);
        providerMemoryStates_.push_back({memoryProvider_, {}});
    }

    std::shared_ptr<ViewSnapshot> nextSnapshot;
    status = BuildSnapshot(view, nullptr, nextSnapshot);
    if (!status.ok()) {
        std::lock_guard<std::mutex> memoryLock{memoryMu_};
        providerMemoryStates_.clear();
        memoryProvider_.reset();
        return status;
    }

    taskQueue_.Setup(queueDepth + 1);
    stopWorker_.store(false, std::memory_order_release);
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
    {
        std::lock_guard<std::mutex> lock{producerMu_};
        stopWorker_.store(true, std::memory_order_release);
    }
    JoinBackgroundRefresh();
    if (worker_.joinable()) { worker_.join(); }

    std::shared_ptr<ViewSnapshot> snapshot;
    std::vector<std::shared_ptr<AsuTransport>> retiredTransports;
    std::vector<ProviderMemoryState> providerMemoryStates;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        snapshot = std::move(snapshot_);
        retiredTransports = std::move(retiredTransports_);
        config_ = AsuClientConfig{};
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
    if (snapshot) {
        auto shutdownStatus = ShutdownSnapshotTransports(snapshot);
        if (!shutdownStatus.ok() && finalStatus.ok()) { finalStatus = shutdownStatus; }
    }
    for (auto& transport : retiredTransports) {
        if (transport == nullptr) { continue; }
        auto status = transport->Shutdown();
        if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
    }
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

Status AsuClientImpl::QueryAsync(const std::vector<CacheKey>& keys, TaskId& taskId)
{
    auto status = SubmitAsync(AsuOpType::QUERY, keys, taskId);
    if (IsRefreshNeeded(status)) { RequestBackgroundRefresh(); }
    return status;
}

Status AsuClientImpl::LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    return SubmitAsync(AsuOpType::LOAD, entries, taskId);
}

Status AsuClientImpl::StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    return SubmitAsync(AsuOpType::STORE, entries, taskId);
}

Status AsuClientImpl::BatchLoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    return SubmitAsync(AsuOpType::BATCH_LOAD, entries, taskId);
}

Status AsuClientImpl::BatchStoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    return SubmitAsync(AsuOpType::BATCH_STORE, entries, taskId);
}

Status AsuClientImpl::DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId)
{
    return SubmitAsync(AsuOpType::DELETE, keys, taskId);
}

bool AsuClientImpl::Check(TaskId taskId) { return taskManager_.Check(taskId); }

Status AsuClientImpl::Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result)
{
    const auto waitMs = timeoutMs == 0 ? config_.defaultWaitTimeoutMs : timeoutMs;
    auto status = taskManager_.Wait(taskId, waitMs, result);
    if (status.code == StatusCode::TASK_NOT_FOUND) { return status; }
    if (viewServer_ != nullptr &&
        (viewServer_->ShouldRefreshView(status) || viewServer_->ShouldRefreshView(result))) {
        RequestBackgroundRefresh();
    }
    return status;
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

Status AsuClientImpl::SubmitAsync(AsuOpType opType, const std::vector<KVBuffer>& entries,
                                  TaskId& taskId)
{
    auto snapshot = GetSnapshot();
    if (!snapshot || !snapshot->router || snapshot->transports.empty()) {
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
    ctx->opType = opType;
    ctx->viewSnapshot = snapshot;
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

    {
        std::lock_guard<std::mutex> lock{producerMu_};
        if (stopWorker_.load(std::memory_order_acquire)) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return NotInitialized();
        }
        if (!taskQueue_.TryPush(std::move(rawCtx))) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::RESOURCE_BUSY, "client task queue is full");
        }
    }
    return Status::OK();
}

Status AsuClientImpl::SubmitAsync(AsuOpType opType, const std::vector<CacheKey>& keys,
                                  TaskId& taskId)
{
    auto snapshot = GetSnapshot();
    if (!snapshot || !snapshot->router || snapshot->transports.empty()) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    if (opType != AsuOpType::QUERY && opType != AsuOpType::DELETE) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "keys submit only supports query/delete");
    }

    auto ctx = std::make_unique<ClientTask>();
    ctx->opType = opType;
    ctx->viewSnapshot = snapshot;
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

    {
        std::lock_guard<std::mutex> lock{producerMu_};
        if (stopWorker_.load(std::memory_order_acquire)) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return NotInitialized();
        }
        if (!taskQueue_.TryPush(std::move(rawCtx))) {
            (void)taskManager_.Remove(taskId);
            taskId = kInvalidTaskId;
            return Status::Error(StatusCode::RESOURCE_BUSY, "client task queue is full");
        }
    }
    return Status::OK();
}

void AsuClientImpl::WorkerLoop()
{
    auto processTask = [this](ClientTaskPtr ctx) {
        auto status = taskManager_.Process(ctx);
        if (IsRefreshNeeded(status)) { RequestBackgroundRefresh(); }
    };
    taskQueue_.ConsumerLoop(stopWorker_, processTask);

    ClientTaskPtr ctx;
    while (taskQueue_.TryPop(ctx)) { processTask(std::move(ctx)); }
}

Status AsuClientImpl::UnregisterRegions(const std::vector<MRHandle>& handles)
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
        }

        nextSnapshot->transports.emplace(asuId, std::move(transport));
    }

    kv::RouterConfig routerConfig;
    auto status = BuildRouterConfigFromAttrs(config_.attrs, routerConfig);
    if (!status.ok()) {
        KV_ERROR("BuildSnapshot build router config failed: {}", status.message);
        return status;
    }

    std::vector<kv::NodeId> nodeIds(asuIds.begin(), asuIds.end());
    nextSnapshot->router =
        kv::CreateRouter(nodeIds, kv::HashFunction{}, routerConfig);
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

    std::lock_guard<std::mutex> memoryLock{memoryMu_};
    std::shared_ptr<TransProvider> transProvider;
    if (config_.sharedProviderMode == SharedProviderMode::SHARED) {
        transProvider = memoryProvider_;
    }
    const bool createdProvider = !transProvider;
    if (createdProvider) {
        auto status = transProviderFactory_(config, transProvider);
        if (!status.ok()) {
            return WithContext(status,
                               "create transport provider failed, asuId=" + std::to_string(asuId));
        }
        if (!transProvider) {
            return Status::Error(
                StatusCode::INTERNAL_ERROR,
                "transport provider factory returned null, asuId=" + std::to_string(asuId));
        }
    }

    auto nextTransport = transportFactory_();
    if (!nextTransport) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "transport factory returned null, asuId=" + std::to_string(asuId));
    }

    auto status = nextTransport->Init(config, transProvider);
    if (!status.ok()) {
        return WithContext(status, "init transport failed, asuId=" + std::to_string(asuId));
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
                               "bind provider memory failed, asuId=" + std::to_string(asuId));
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

Status AsuClientImpl::BindProviderRegions(const std::shared_ptr<TransProvider>& transProvider,
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

Status AsuClientImpl::UnregisterProviderRegions(const std::shared_ptr<TransProvider>& transProvider,
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

Status AsuClientImpl::RefreshView()
{
    std::shared_ptr<ViewServer> viewServer;
    std::shared_ptr<ViewSnapshot> oldSnapshot;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_ && !refreshInProgress_) { return NotInitialized(); }
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
        if (!initialized_ && !refreshInProgress_) { return NotInitialized(); }
        if (snapshot_ != nullptr && !viewServer->ShouldPublishView(snapshot_->view, view)) {
            return Status::OK();
        }
    }

    std::shared_ptr<ViewSnapshot> nextSnapshot;
    status = BuildSnapshot(view, oldSnapshot, nextSnapshot);
    if (!status.ok()) { return status; }

    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_ && !refreshInProgress_) { return NotInitialized(); }
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
    std::thread completedThread;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_ || refreshInProgress_) { return; }
        completedThread = std::move(refreshThread_);
        refreshInProgress_ = true;
        refreshThread_ = std::thread([this] {
            const auto status = RefreshView();
            if (!status.ok()) {
                KV_WARN("Background view refresh failed: code={} message={}",
                        static_cast<int>(status.code), status.message);
            }
            std::lock_guard<std::mutex> lock{mutex_};
            refreshInProgress_ = false;
        });
    }

    if (completedThread.joinable()) { completedThread.join(); }
}

void AsuClientImpl::JoinBackgroundRefresh()
{
    std::thread refreshThread;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        refreshThread = std::move(refreshThread_);
    }
    if (refreshThread.joinable()) { refreshThread.join(); }
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

std::shared_ptr<ViewSnapshot> AsuClientImpl::GetSnapshot() const
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (!initialized_) { return nullptr; }
    return snapshot_;
}

bool AsuClientImpl::IsRefreshNeeded(const Status& status) const
{
    return viewServer_ != nullptr && viewServer_->ShouldRefreshView(status);
}

std::vector<AsuId> AsuClientImpl::GetSortedAsuIds(const GlobalView& view)
{
    std::vector<AsuId> asuIds;
    asuIds.reserve(view.asuMap.size());
    for (const auto& item : view.asuMap) {
        if (item.first != static_cast<AsuId>(kv::kInvalidNodeId)) {
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

std::unique_ptr<AsuClient> CreateAsuClient()
{
    return std::make_unique<AsuClientImpl>(nullptr, nullptr, nullptr);
}

std::unique_ptr<AsuClient> CreateAsuClient(TransportFactory transportFactory)
{
    return std::make_unique<AsuClientImpl>(std::move(transportFactory), nullptr, nullptr);
}

std::unique_ptr<AsuClient> CreateAsuClient(TransportFactory transportFactory,
                                           TransProviderFactory transProviderFactory)
{
    return std::make_unique<AsuClientImpl>(std::move(transportFactory), nullptr,
                                           std::move(transProviderFactory));
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

}  // namespace kv
