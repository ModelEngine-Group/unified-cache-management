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
#include <functional>
#include <iterator>
#include <thread>
#include <utility>
#include "device.h"
#include "kv_metrics/metrics.h"
#include "kv_types.h"
#include "logger.h"
#include "parser_common.h"
#include "router/config.h"
#include "router/router.h"
#include "trans_types.h"
#include "transport/asu_transport_pool.h"
#include "utils/config_utils.h"
#include "view/m800_view_master_server.h"
#include "view/m800_view_service_factory.h"
#include "view/m800_view_slave_server.h"
#include "view/threaded_refresh_scheduler.h"

namespace kv {

namespace {

constexpr std::uint64_t kDefaultRefreshIntervalMs = 30000;
constexpr std::uint64_t kMaxRefreshIntervalMs =
    static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());

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

Status ParseRefreshInterval(const KvClientConfig& config, std::uint64_t& refreshIntervalMs)
{
    refreshIntervalMs = kDefaultRefreshIntervalMs;
    const auto iter = config.attrs.find("view.refresh_interval_ms");
    if (iter == config.attrs.end()) { return Status::OK(); }

    const auto value = TrimConfigValue(iter->second);
    if (value.empty() || value.find('\0') != std::string::npos || value.front() == '-') {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "view.refresh_interval_ms must be a valid interval in milliseconds");
    }

    constexpr auto invalid = std::numeric_limits<std::uint64_t>::max();
    const auto parsed = ParseConfigUint64(value, invalid);
    if (parsed == invalid || parsed == 0 || parsed > kMaxRefreshIntervalMs) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "view.refresh_interval_ms must be between 1 and " +
                                 std::to_string(kMaxRefreshIntervalMs));
    }
    refreshIntervalMs = parsed;
    return Status::OK();
}

std::uint64_t ViewsGlobalEpoch(const std::unordered_map<KvNsId, GlobalView>& views)
{
    std::uint64_t epoch{0};
    for (const auto& item : views) { epoch = std::max(epoch, item.second.viewEpoch); }
    return epoch;
}

Status ValidateViews(const std::unordered_map<KvNsId, GlobalView>& views)
{
    if (views.empty()) { return Status::Error(StatusCode::INVALID_ARGUMENT, "view map is empty"); }
    const auto expectedEpoch = views.begin()->second.viewEpoch;
    for (const auto& item : views) {
        if (item.second.viewEpoch != expectedEpoch) {
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "M800 view epochs must be identical across namespaces, nsId=" +
                                     std::to_string(item.first) +
                                     ", epoch=" + std::to_string(item.second.viewEpoch) +
                                     ", expectedEpoch=" + std::to_string(expectedEpoch));
        }
    }
    return Status::OK();
}

Status StartM800ViewServer(const std::shared_ptr<ViewServer>& viewServer)
{
    auto master = std::dynamic_pointer_cast<M800ViewMasterServer>(viewServer);
    if (master != nullptr) { return master->Start(); }
    auto slave = std::dynamic_pointer_cast<M800ViewSlaveServer>(viewServer);
    if (slave != nullptr) { return slave->Start(); }
    return Status::Error(StatusCode::INVALID_ARGUMENT,
                         "view.provider=m800 requires an M800 view service");
}

void StopM800ViewServer(const std::shared_ptr<ViewServer>& viewServer)
{
    if (viewServer == nullptr) { return; }
    auto master = std::dynamic_pointer_cast<M800ViewMasterServer>(viewServer);
    if (master != nullptr) {
        master->Stop();
        return;
    }
    auto slave = std::dynamic_pointer_cast<M800ViewSlaveServer>(viewServer);
    if (slave != nullptr) { slave->Stop(); }
}

}  // namespace

constexpr std::uint32_t kMaxShutdownDrainAttempts = 64;

KvClientImpl::KvClientImpl(TransportFactory transportFactory, ViewServerFactory viewServerFactory,
                           TransProviderFactory transProviderFactory)
    : transportFactory_(std::move(transportFactory)),
      transProviderFactory_(std::move(transProviderFactory)),
      viewServerFactory_(std::move(viewServerFactory))
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
    if (config.transportConfigs.empty()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT,
                             "at least one transport config is required");
    }
    const auto namespaceAttr = config.transportConfigs.front().attrs.find("kv_ns_id");
    const auto configuredNamespaceId = namespaceAttr == config.transportConfigs.front().attrs.end()
                                           ? 0U
                                           : ParseConfigUint32(namespaceAttr->second, 0);

    config_ = config;
    keyResolver_ = CreateDefaultKeyResolver(config.keyResolution, configuredNamespaceId);
    std::uint64_t refreshIntervalMs{0};
    bool multiNamespaceView{false};
    if (HasViewProvider(config)) {
        if (!IsM800ViewProvider(config)) {
            const auto iter = config.attrs.find("view.provider");
            return Status::Error(StatusCode::INVALID_ARGUMENT,
                                 "view.provider must be m800, value=" +
                                     (iter == config.attrs.end() ? std::string{} : iter->second));
        }
        M800ViewServerConfig m800Config;
        const auto m800Status = M800ViewServerConfig::Build(config, m800Config);
        if (!m800Status.ok()) { return m800Status; }
        const auto intervalStatus = ParseRefreshInterval(config, refreshIntervalMs);
        if (!intervalStatus.ok()) { return intervalStatus; }
        m800Config.refreshIntervalMs = refreshIntervalMs;
        multiNamespaceView = true;
    } else {
        const auto intervalStatus = ParseRefreshInterval(config, refreshIntervalMs);
        if (!intervalStatus.ok()) { return intervalStatus; }
    }
    viewServer_ = viewServerFactory_(config);
    if (viewServer_ == nullptr) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "view server factory returned null");
    }
    multiNamespaceView_ = multiNamespaceView;
    transportConfigs_.clear();
    for (const auto& transportConfig : config.transportConfigs) {
        transportConfigs_[transportConfig.nodeId] = transportConfig;
    }
    const auto queueDepth =
        std::max<std::size_t>(2, static_cast<std::size_t>(config.maxInflightTasks));

    if (multiNamespaceView_) {
        auto status = StartM800ViewServer(viewServer_);
        if (!status.ok()) {
            viewServer_.reset();
            return status;
        }
    }

    std::shared_ptr<TransProvider> memoryProvider;
    auto status = transProviderFactory_(config.transportConfigs.front(), memoryProvider);
    if (!status.ok()) {
        StopM800ViewServer(viewServer_);
        viewServer_.reset();
        return WithContext(status, "create business-memory provider failed");
    }
    if (!memoryProvider) {
        StopM800ViewServer(viewServer_);
        viewServer_.reset();
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "business-memory provider factory returned null");
    }
    {
        std::lock_guard<std::mutex> memoryLock{memoryMu_};
        memoryProvider_ = std::move(memoryProvider);
        providerMemoryStates_.push_back({memoryProvider_, {}});
    }
    transportPool_ = std::make_unique<AsuTransportPool>(
        [this](const TransportConfig& config, std::shared_ptr<AsuTransport>& transport) {
            return CreateTransport(config, transport);
        });
    const auto cleanupInitializationResources = [this](const char* context) {
        StopM800ViewServer(viewServer_);
        viewServer_.reset();
        const auto transportShutdownStatus = ShutdownTransportPool();
        if (!transportShutdownStatus.ok()) {
            KV_WARN("{} transport cleanup failed: code={} message={}", context,
                    static_cast<int>(transportShutdownStatus.code),
                    transportShutdownStatus.message);
        }
        std::lock_guard<std::mutex> memoryLock{memoryMu_};
        providerMemoryStates_.clear();
        memoryProvider_.reset();
    };

    std::shared_ptr<ViewSnapshot> nextSnapshot;
    ViewSnapshotMap nextSnapshots;
    if (multiNamespaceView_) {
        std::unordered_map<KvNsId, GlobalView> views;
        auto status = viewServer_->GetAllViews(views);
        if (!status.ok()) {
            cleanupInitializationResources("M800 init");
            return status;
        }
        status = ValidateViews(views);
        if (!status.ok()) {
            cleanupInitializationResources("M800 init");
            return status;
        }
        status = BuildSnapshots(views, nextSnapshots);
        if (!status.ok()) {
            cleanupInitializationResources("M800 init");
            return status;
        }
    } else {
        GlobalView view;
        auto status = viewServer_->GetGlobalView(view);
        if (!status.ok()) {
            cleanupInitializationResources("Legacy init");
            return status;
        }
        status = BuildSnapshot(view, nextSnapshot);
        if (!status.ok()) {
            cleanupInitializationResources("Legacy init");
            return status;
        }
        nextSnapshots.emplace(kDefaultKvNsId, nextSnapshot);
    }

    taskQueue_.Setup(queueDepth + 1);
    stopWorker_.store(false, std::memory_order_release);
    worker_ = std::thread(&KvClientImpl::WorkerLoop, this);
    shutdownRequested_ = false;

    Status schedulerStatus;
    auto refreshScheduler = std::make_unique<ThreadedRefreshScheduler>();
    {
        std::lock_guard<std::mutex> lock{mutex_};
        schedulerStatus =
            refreshScheduler->Start(std::chrono::milliseconds{refreshIntervalMs}, [this] {
                const auto status = RefreshView();
                if (!status.ok()) {
                    KV_WARN("Background view refresh failed: code={} message={}",
                            static_cast<int>(status.code), status.message);
                }
                return status;
            });
        if (schedulerStatus.ok()) {
            refreshScheduler_ = std::move(refreshScheduler);
            snapshots_ = std::move(nextSnapshots);
            // kDefaultKvNsId is reserved for the legacy single-view path. M800
            // namespace ids are derived from Ceph pool names, so they normally
            // do not collide with the default namespace. Legacy snapshot APIs
            // therefore remain unavailable in M800 mode until namespace-aware
            // IO integration is delivered; use GetSnapshotByKvnsId() instead.
            snapshot_ = multiNamespaceView_ ? (snapshots_.find(kDefaultKvNsId) == snapshots_.end()
                                                   ? nullptr
                                                   : snapshots_.at(kDefaultKvNsId))
                                            : nextSnapshot;
            initialized_ = true;
        }
    }
    if (!schedulerStatus.ok()) {
        const auto failureStatus = schedulerStatus;
        const auto shutdownStatus = Shutdown();
        (void)shutdownStatus;
        return failureStatus;
    }
    const auto& resolution = config_.keyResolution;
    const char* engine = resolution.engine == InferenceEngine::VLLM_ASCEND ? "vllm_ascend"
                         : resolution.engine == InferenceEngine::SGLANG    ? "sglang"
                                                                           : "generic";
    KV_INFO("KVClient initialized: engine={}", engine);
    return Status::OK();
}

Status KvClientImpl::Shutdown()
{
    std::uint64_t waitTimeoutMs = 0;
    std::unique_ptr<RefreshScheduler> refreshScheduler;
    std::shared_ptr<ViewServer> viewServer;
    std::unique_ptr<AsuTransportPool> transportPool;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        shutdownRequested_ = true;
        initialized_ = false;
        waitTimeoutMs = config_.defaultWaitTimeoutMs;
        refreshScheduler = std::move(refreshScheduler_);
    }
    if (refreshScheduler != nullptr) { refreshScheduler->Stop(); }
    {
        std::lock_guard<std::mutex> lock{producerMu_};
        stopWorker_.store(true, std::memory_order_release);
    }
    workerCv_.notify_one();
    if (worker_.joinable()) { worker_.join(); }

    std::vector<std::shared_ptr<ViewSnapshot>> activeSnapshots;
    std::vector<ProviderMemoryState> providerMemoryStates;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (snapshot_ != nullptr) { activeSnapshots.emplace_back(std::move(snapshot_)); }
        snapshot_.reset();
        for (auto& item : snapshots_) {
            if (item.second != nullptr) { activeSnapshots.emplace_back(std::move(item.second)); }
        }
        snapshots_.clear();
        config_ = KvClientConfig{};
        viewServer = std::move(viewServer_);
        viewServer_.reset();
        transportPool = std::move(transportPool_);
        transportPool_.reset();
        multiNamespaceView_ = false;
        transportConfigs_.clear();
    }
    StopM800ViewServer(viewServer);
    {
        std::lock_guard<std::mutex> lock{memoryMu_};
        providerMemoryStates = std::move(providerMemoryStates_);
        memoryProvider_.reset();
        registeredRegions_.clear();
    }

    Status finalStatus = Status::OK();
    auto drainStatus = taskManager_.Drain(waitTimeoutMs);
    if (!drainStatus.ok()) { finalStatus = drainStatus; }
    if (transportPool != nullptr) {
        auto status = transportPool->ShutdownAll();
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

Status KvClientImpl::QueryAsync(const std::vector<CacheKey>& keys, TaskId& taskId,
                                const IoOptions& options)
{
    auto status = SubmitAsync(AsuOpType::QUERY, keys, taskId, options);
    RecordSubmit(AsuOpType::QUERY, keys.size(), status);
    if (IsRefreshNeeded(status)) { RequestBackgroundRefresh(); }
    return status;
}

Status KvClientImpl::LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId,
                               const IoOptions& options)
{
    const auto useBatchIo =
        entries.size() > 1 || (entries.size() == 1 && config_.useBatchIoForSingleKey);
    const auto opType = useBatchIo ? AsuOpType::BATCH_LOAD : AsuOpType::LOAD;
    auto status = SubmitAsync(opType, entries, taskId, options);
    RecordSubmit(opType, entries.size(), status);
    return status;
}

Status KvClientImpl::StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId,
                                const IoOptions& options)
{
    const auto useBatchIo =
        entries.size() > 1 || (entries.size() == 1 && config_.useBatchIoForSingleKey);
    const auto opType = useBatchIo ? AsuOpType::BATCH_STORE : AsuOpType::STORE;
    auto status = SubmitAsync(opType, entries, taskId, options);
    RecordSubmit(opType, entries.size(), status);
    return status;
}

Status KvClientImpl::DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId,
                                 const IoOptions& options)
{
    auto status = SubmitAsync(AsuOpType::DELETE, keys, taskId, options);
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
        RequestBackgroundRefresh();
    }
    return status;
}

Status KvClientImpl::RegisterRegions(const std::vector<MemoryRegion>& regions,
                                     std::vector<RegisteredMemory>& registeredRegions)
{
    bool needRefresh = false;
    auto status = RegisterRegionsOnce(regions, registeredRegions, needRefresh);
    if (needRefresh) { RequestBackgroundRefresh(); }
    return status;
}

Status KvClientImpl::RegisterRegionsOnce(const std::vector<MemoryRegion>& regions,
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

Status KvClientImpl::SubmitAsync(AsuOpType opType, const std::vector<KVBuffer>& entries,
                                 TaskId& taskId, const IoOptions& options)
{
    const auto taskStart = std::chrono::steady_clock::now();
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
    ctx->submittedAt = taskStart;
    ctx->opType = opType;
    ctx->prerequisite = options.prerequisite;
    ctx->viewSnapshot = snapshot;
    ctx->entries.reserve(entries.size());
    for (const auto& entry : entries) {
        ResolvedKVBuffer resolvedKVBuffer;
        auto status = keyResolver_->Resolve(entry.key, options.kvns, resolvedKVBuffer.key);
        if (!status.ok()) {
            taskId = kInvalidTaskId;
            return status;
        }
        resolvedKVBuffer.buffer = entry.buffer;
        resolvedKVBuffer.offset = entry.offset;
        ctx->entries.push_back(std::move(resolvedKVBuffer));
    }
    if (kv::logger::isEnabledFor(kv::logger::Level::DEBUG)) {
        std::string resolvedKeys;
        for (std::size_t index = 0; index < entries.size(); ++index) {
            resolvedKeys += "<" + std::string(CacheKeyView(entries[index].key)) + " -> " +
                            CacheKeyToHex(ctx->entries[index].key) +
                            " gnsid=" + std::to_string(ctx->entries[index].key.gnsid) + "> ";
        }
        KV_DEBUG("KV client submit entries: ns=\"{}\" entries={} resolved=[{}]", options.kvns,
                 entries.size(), resolvedKeys);
    }
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
                                 TaskId& taskId, const IoOptions& options)
{
    const auto taskStart = std::chrono::steady_clock::now();
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
    ctx->submittedAt = taskStart;
    ctx->opType = opType;
    ctx->viewSnapshot = snapshot;
    ctx->prerequisite = options.prerequisite;
    ctx->keys.reserve(keys.size());
    for (const auto& key : keys) {
        ResolvedCacheKey resolvedKey;
        auto status = keyResolver_->Resolve(key, options.kvns, resolvedKey);
        if (!status.ok()) {
            taskId = kInvalidTaskId;
            return status;
        }
        ctx->keys.push_back(std::move(resolvedKey));
    }
    if (kv::logger::isEnabledFor(kv::logger::Level::DEBUG)) {
        std::string resolvedKeys;
        for (std::size_t index = 0; index < keys.size(); ++index) {
            resolvedKeys += "<" + std::string(CacheKeyView(keys[index])) + " -> " +
                            CacheKeyToHex(ctx->keys[index]) +
                            " gnsid=" + std::to_string(ctx->keys[index].gnsid) + "> ";
        }
        KV_DEBUG("KV client submit keys: ns=\"{}\" keys={} resolved=[{}]", options.kvns,
                 keys.size(), resolvedKeys);
    }
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
        if (IsRefreshNeeded(status)) { RequestBackgroundRefresh(); }
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

Status KvClientImpl::BuildSnapshots(const std::unordered_map<KvNsId, GlobalView>& views,
                                    ViewSnapshotMap& snapshots)
{
    snapshots.clear();
    if (views.empty()) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "view universe is empty");
    }

    std::vector<KvNsId> nsIds;
    nsIds.reserve(views.size());
    for (const auto& item : views) { nsIds.emplace_back(item.first); }
    std::sort(nsIds.begin(), nsIds.end());

    for (const auto nsId : nsIds) {
        const auto& view = views.at(nsId);
        // Namespace snapshots remain independent routing units, while their
        // transports are shared through the global per-node transport pool.
        std::shared_ptr<ViewSnapshot> nextSnapshot;
        auto status = BuildSnapshotByKvnsId(nsId, view, nextSnapshot);
        if (!status.ok()) { return WithContext(status, "nsId=" + std::to_string(nsId)); }
        snapshots.emplace(nsId, std::move(nextSnapshot));
    }
    return Status::OK();
}

Status KvClientImpl::BuildSnapshot(const GlobalView& view, std::shared_ptr<ViewSnapshot>& snapshot)
{
    return BuildSnapshotByKvnsId(kDefaultKvNsId, view, snapshot);
}

Status KvClientImpl::BuildSnapshotByKvnsId(KvNsId nsId, const GlobalView& view,
                                           std::shared_ptr<ViewSnapshot>& snapshot)
{
    auto nextSnapshot = std::make_shared<ViewSnapshot>();
    auto asuIds = GetSortedAsuIds(view);
    nextSnapshot->nsId = nsId;
    nextSnapshot->view = view;

    for (std::size_t asuIndex = 0; asuIndex < asuIds.size(); ++asuIndex) {
        auto nodeId = asuIds[asuIndex];
        std::shared_ptr<AsuTransport> transport;
        auto viewIter = view.asuMap.find(nodeId);
        auto asuInfo = viewIter == view.asuMap.end() ? NodeInfo{} : viewIter->second;
        TransportConfig transportConfig;
        auto status = MakeTransportConfig(nodeId, asuInfo, transportConfig);
        if (!status.ok()) {
            return WithContext(status, "asuIndex=" + std::to_string(asuIndex) +
                                           " nodeId=" + std::to_string(nodeId));
        }

        status = transportPool_->GetOrCreate(transportConfig, transport);
        if (status.code == StatusCode::ENDPOINT_CHANGED) {
            status = transportPool_->Replace(transportConfig, transport);
        }
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

Status KvClientImpl::MakeTransportConfig(NodeId nodeId, const NodeInfo& asuInfo,
                                         TransportConfig& config)
{
    std::lock_guard<std::mutex> lock{mutex_};
    auto configIter = transportConfigs_.find(nodeId);
    if (configIter == transportConfigs_.end()) {
        return Status::Error(StatusCode::NOT_FOUND,
                             "transport config not found, nodeId=" + std::to_string(nodeId));
    }
    config = configIter->second;
    ApplyAsuInfoToTransportConfig(asuInfo, config);
    return Status::OK();
}

Status KvClientImpl::CreateTransport(const TransportConfig& config,
                                     std::shared_ptr<AsuTransport>& transport)
{
    SharedProviderMode sharedProviderMode = SharedProviderMode::INDEPENDENT;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        sharedProviderMode = config_.sharedProviderMode;
    }

    std::lock_guard<std::mutex> memoryLock{memoryMu_};
    std::shared_ptr<TransProvider> transProvider;
    if (sharedProviderMode == SharedProviderMode::SHARED) { transProvider = memoryProvider_; }
    const bool createdProvider = !transProvider;
    if (createdProvider) {
        auto status = transProviderFactory_(config, transProvider);
        if (!status.ok()) {
            return WithContext(status, "create transport provider failed, nodeId=" +
                                           std::to_string(config.nodeId));
        }
        if (!transProvider) {
            return Status::Error(StatusCode::INTERNAL_ERROR,
                                 "transport provider factory returned null, nodeId=" +
                                     std::to_string(config.nodeId));
        }
    }

    auto nextTransport = transportFactory_();
    if (!nextTransport) {
        return Status::Error(
            StatusCode::INTERNAL_ERROR,
            "transport factory returned null, nodeId=" + std::to_string(config.nodeId));
    }

    auto status = nextTransport->Init(config, transProvider);
    if (!status.ok()) {
        return WithContext(status,
                           "init transport failed, nodeId=" + std::to_string(config.nodeId));
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
            return WithContext(
                status, "bind provider memory failed, nodeId=" + std::to_string(config.nodeId));
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

Status KvClientImpl::RefreshView()
{
    std::shared_ptr<ViewServer> viewServer;
    std::shared_ptr<ViewSnapshot> oldSnapshot;
    ViewSnapshotMap oldSnapshots;
    bool multiNamespaceView{false};
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_ && !shutdownRequested_) { return NotInitialized(); }
        viewServer = viewServer_;
        multiNamespaceView = multiNamespaceView_;
        oldSnapshots = snapshots_;
        oldSnapshot = snapshot_;
    }
    if (viewServer == nullptr) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "view server is not initialized");
    }

    if (multiNamespaceView) {
        std::unordered_map<KvNsId, GlobalView> nextViews;
        nextViews.reserve(oldSnapshots.size());
        for (const auto& item : oldSnapshots) {
            if (item.second != nullptr) { nextViews.emplace(item.first, item.second->view); }
        }
        auto status = viewServer->GetAllViews(nextViews);
        if (!status.ok()) { return status; }
        status = ValidateViews(nextViews);
        if (!status.ok()) { return status; }

        std::uint64_t currentEpoch{0};
        for (const auto& item : oldSnapshots) {
            if (item.second != nullptr) {
                currentEpoch = std::max(currentEpoch, item.second->view.viewEpoch);
            }
        }
        const auto nextEpoch = ViewsGlobalEpoch(nextViews);
        if (!oldSnapshots.empty() && nextEpoch < currentEpoch) {
            return Status::Error(
                StatusCode::INVALID_ARGUMENT,
                "fetched view map epoch is stale, epoch=" + std::to_string(nextEpoch) +
                    ", currentEpoch=" + std::to_string(currentEpoch));
        }
        if (!oldSnapshots.empty() && nextEpoch == currentEpoch && currentEpoch != 0) {
            return Status::OK();
        }

        ViewSnapshotMap nextSnapshots;
        status = BuildSnapshots(nextViews, nextSnapshots);
        if (!status.ok()) { return status; }

        {
            std::lock_guard<std::mutex> lock{mutex_};
            if (!initialized_ && !shutdownRequested_) { return NotInitialized(); }
            if (snapshots_ != oldSnapshots) {
                return Status::Error(StatusCode::RESOURCE_BUSY,
                                     "multi-namespace snapshots changed during refresh");
            }
            snapshots_ = std::move(nextSnapshots);
            const auto defaultSnapshot = snapshots_.find(kDefaultKvNsId);
            snapshot_ = defaultSnapshot == snapshots_.end() ? nullptr : defaultSnapshot->second;
        }
        return Status::OK();
    }

    GlobalView view;
    auto status = viewServer->GetGlobalView(view);
    if (!status.ok()) { return status; }

    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_ && !shutdownRequested_) { return NotInitialized(); }
        if (oldSnapshot != nullptr && !viewServer->ShouldPublishView(oldSnapshot->view, view)) {
            return Status::OK();
        }
    }

    std::shared_ptr<ViewSnapshot> nextSnapshot;
    status = BuildSnapshot(view, nextSnapshot);
    if (!status.ok()) { return status; }

    ViewSnapshotMap nextSnapshots;
    nextSnapshots.emplace(kDefaultKvNsId, nextSnapshot);

    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_ && !shutdownRequested_) { return NotInitialized(); }
        if (oldSnapshot != nullptr && !viewServer->ShouldPublishView(oldSnapshot->view, view)) {
            return Status::OK();
        }
        snapshot_ = nextSnapshot;
        snapshots_ = std::move(nextSnapshots);
    }

    return Status::OK();
}

void KvClientImpl::RequestBackgroundRefresh()
{
    RefreshScheduler* scheduler = nullptr;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        if (!initialized_) { return; }
        scheduler = refreshScheduler_.get();
        if (scheduler == nullptr) { return; }
        // Stop cannot remove the scheduler while mutex_ is held. Triggering
        // under mutex_ guarantees the IO-failure refresh is accepted before
        // Shutdown starts draining the scheduler.
        scheduler->TriggerByIoFailure(
            Status::Error(StatusCode::IO_ERROR, "view refresh requested"));
    }
}

std::shared_ptr<ViewSnapshot> KvClientImpl::GetSnapshot() const
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (!initialized_) { return nullptr; }
    return snapshot_;
}

std::shared_ptr<ViewSnapshot> KvClientImpl::GetSnapshotByKvnsId(KvNsId nsId) const
{
    std::lock_guard<std::mutex> lock{mutex_};
    if (!initialized_) { return nullptr; }
    const auto iter = snapshots_.find(nsId);
    return iter == snapshots_.end() ? nullptr : iter->second;
}

Status KvClientImpl::ShutdownTransportPool()
{
    std::unique_ptr<AsuTransportPool> transportPool;
    {
        std::lock_guard<std::mutex> lock{mutex_};
        transportPool = std::move(transportPool_);
        transportPool_.reset();
    }
    if (transportPool == nullptr) { return Status::OK(); }
    return transportPool->ShutdownAll();
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
