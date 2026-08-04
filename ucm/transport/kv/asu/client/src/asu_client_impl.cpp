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
    return finalStatus;
}

Status AsuClientImpl::QueryAsync(const std::vector<CacheKey>& keys, const QueryOptions& options,
                                 TaskId& taskId)
{
    auto status = SubmitAsync(ClientOpType::QUERY, keys, options.timeoutMs, taskId);
    if (IsRefreshNeeded(status)) { RequestBackgroundRefresh(); }
    return status;
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
    return SubmitAsync(ClientOpType::DELETE, keys, config_.timeoutMs, taskId);
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

    auto firstIter = snapshot->transports.find(snapshot->asuIds.front());
    if (firstIter == snapshot->transports.end()) {
        auto status = Status::Error(StatusCode::NOT_FOUND, "first asu transport not found");
        needRefresh |= IsRefreshNeeded(status);
        return WithContext(status, "asuIndex=0 asuId=" + std::to_string(snapshot->asuIds.front()));
    }

    auto status = firstIter->second->RegisterRegions(regions, registeredRegions);
    if (!status.ok()) {
        needRefresh |= IsRefreshNeeded(status);
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
            needRefresh |= IsRefreshNeeded(status);
            finalStatus = WithContext(PartialFailed("one or more asu region bindings failed"),
                                      "asuIndex=" + std::to_string(asuIndex) +
                                          " asuId=" + std::to_string(snapshot->asuIds[asuIndex]));
            continue;
        }

        status = iter->second->BindRegisteredRegions(registeredRegions);
        if (!status.ok() && finalStatus.ok()) {
            needRefresh |= IsRefreshNeeded(status);
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

    auto ctx = std::make_unique<ClientTask>();
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
                                  std::uint64_t timeoutMs, TaskId& taskId)
{
    auto snapshot = GetSnapshot();
    if (!snapshot || !snapshot->router || snapshot->transports.empty()) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    if (opType != ClientOpType::QUERY && opType != ClientOpType::DELETE) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "keys submit only supports query/delete");
    }

    auto ctx = std::make_unique<ClientTask>();
    ctx->opType = opType;
    ctx->viewSnapshot = snapshot;
    ctx->keys = keys;
    ctx->entryStatus.assign(keys.size(), Status::OK());
    ctx->queryResult.exists.assign(opType == ClientOpType::QUERY ? keys.size() : 0, 0);
    ctx->timeoutMs = timeoutMs;

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

void AsuClientImpl::WorkerLoop()
{
    while (true) {
        ClientTaskPtr ctx;
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
        auto status = taskManager_.Process(ctx);
        if (IsRefreshNeeded(status)) { RequestBackgroundRefresh(); }
    }
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
            needRefresh |= IsRefreshNeeded(status);
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
                UC_WARN("Background view refresh failed: code={} message={}",
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
