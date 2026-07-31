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
#include "drampool_server.h"
#include <chrono>
#include <exception>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include "channels/tcp/tcp_message_channel.h"
#include "core/transport_manager.h"
#include "logger/logger.h"
#include "metadata.h"
#include "pool/buffer_pool.h"
#include "task_worker.h"
#include "trans/device.h"

namespace UC::DramPool {
namespace {

template <typename Callback>
class ScopeExit final {
public:
    explicit ScopeExit(Callback callback) : callback_(std::move(callback)) {}
    ~ScopeExit()
    {
        if (active_) { callback_(); }
    }

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

    ScopeExit(ScopeExit&& other) noexcept(std::is_nothrow_move_constructible_v<Callback>)
        : callback_(std::move(other.callback_)), active_(std::exchange(other.active_, false))
    {
    }

    void Release() noexcept { active_ = false; }

private:
    Callback callback_;
    bool active_{true};
};

template <typename Callback>
auto MakeScopeExit(Callback&& callback)
{
    return ScopeExit<std::decay_t<Callback>>(std::forward<Callback>(callback));
}

}  // namespace

DramPoolServer::DramPoolServer() = default;

DramPoolServer::~DramPoolServer() { Stop(); }

Status DramPoolServer::Init()
{
    if (state_ != ServerState::New) {
        return Status::InvalidParam("DramPoolServer cannot be initialized in its current state");
    }

    // Build runtime dependencies without starting transport services or listeners.
    auto rollback = MakeScopeExit([this]() { ResetInitializedComponents(); });
    try {
        if (auto status = InitializeDeviceRuntime(); status.Failure()) { return status; }
        if (auto status = InitMemoryPool(); status.Failure()) { return status; }
        if (auto status = InitFlagBufferPool(); status.Failure()) { return status; }
        if (auto status = InitMetadata(); status.Failure()) { return status; }
        if (auto status = InitProtocol(); status.Failure()) { return status; }
        if (auto status = InitQueues(); status.Failure()) { return status; }
        if (auto status = InitTransportManager(); status.Failure()) { return status; }
        if (auto status = CreateRuntimeContext(); status.Failure()) { return status; }
    } catch (const std::exception& error) {
        return Status::Error(std::string{"DramPoolServer initialization failed: "} + error.what());
    }

    state_ = ServerState::Initialized;
    rollback.Release();
    return Status::OK();
}

Status DramPoolServer::Start()
{
    if (state_ != ServerState::Initialized) {
        return Status::InvalidParam("DramPoolServer is not ready to start");
    }

    auto rollback = MakeScopeExit([this]() { Stop(); });
    try {
        if (auto status = StartCompletionPoller(); status.Failure()) { return status; }
        if (auto status = StartTaskWorker(); status.Failure()) { return status; }
        if (auto status = StartGCThread(); status.Failure()) { return status; }
        if (auto status = StartRequestReceiver(); status.Failure()) { return status; }
        // Start transport listeners only after every request consumer is ready.
        if (auto status = StartTransportService(); status.Failure()) { return status; }
        if (auto status = StartTcpMessageChannel(); status.Failure()) { return status; }
        state_ = ServerState::Running;
    } catch (const std::exception& error) {
        return Status::Error(std::string{"DramPoolServer start failed: "} + error.what());
    }
    rollback.Release();
    return Status::OK();
}

void DramPoolServer::Stop()
{
    if (state_ == ServerState::New || state_ == ServerState::Stopped) { return; }
    // Close ingress, stop its receiver, then let TaskWorker drain accepted tasks.
    StopRequestReceiver();
    StopTaskWorker();
    StopCompletionPoller();
    StopGCThread();
    StopTransportService();
    // Object destruction is shared with Init() rollback and happens only after
    // every active thread and transport service has stopped.
    ResetInitializedComponents();
    state_ = ServerState::Stopped;
}

Status DramPoolServer::InitializeDeviceRuntime()
{
    if (g_config.transportDeviceIds.empty()) {
        return Status::InvalidParam("transport.device_ids must not be empty");
    }
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<std::size_t> distribution(0,
                                                            g_config.transportDeviceIds.size() - 1);
    const auto deviceId = g_config.transportDeviceIds[distribution(generator)];

    auto status = device_.Init(&deviceRuntimeOwned_);
    if (status.Failure()) {
        return Status::Error("device runtime initialization failed: " + status.ToString());
    }
    status = device_.Setup(deviceId);
    if (status.Success()) {
        deviceId_ = deviceId;
        return Status::OK();
    }

    if (deviceRuntimeOwned_) {
        (void)device_.Finalize();
        deviceRuntimeOwned_ = false;
    }
    return Status::Error("device runtime setup failed: " + status.ToString());
}

Status DramPoolServer::InitMemoryPool()
{
    std::vector<std::pair<std::size_t, std::size_t>> slots;
    slots.reserve(g_config.poolBlockSizes.size());
    for (std::size_t index = 0; index < g_config.poolBlockSizes.size(); ++index) {
        slots.emplace_back(static_cast<std::size_t>(g_config.poolBlockSizes[index]),
                           static_cast<std::size_t>(g_config.poolSlotCounts[index]));
    }
    try {
        bufferManager_ = std::make_unique<BufferManager>(slots);
    } catch (const std::exception& error) {
        return Status::Error(std::string{"DramPool buffer manager init failed: "} + error.what());
    }
    return Status::OK();
}

Status DramPoolServer::InitFlagBufferPool()
{
    auto flagBufferPool = std::make_unique<UC::BufferPool>();
    const auto status = flagBufferPool->Init(
        "drampool_flag_buffer_pool", UC::BufferPool::MemoryType::HOST,
        static_cast<std::size_t>(g_config.flagBufferSlotSizeBytes),
        static_cast<std::size_t>(g_config.flagBufferSlotCount), false, kFlagBufferSlotAlignment);
    if (status.Failure()) {
        return Status::Error("DramPool flag buffer pool init failed: " + status.ToString());
    }
    flagBufferPool_ = std::move(flagBufferPool);
    return Status::OK();
}

Status DramPoolServer::InitMetadata()
{
    // Metadata eviction settings come from the process-wide runtime configuration.
    const UC::DramPool::MetadataConfig config{
        g_config.metadataPeriodicEvictionPolicy,
        g_config.metadataDeepEvictionPolicy,
        std::chrono::milliseconds(g_config.metadataLeaseTimeMs),
        g_config.metadataDefaultEvictRatio,
    };
    metadataManager_ = std::make_unique<UC::DramPool::MetadataManager>(config, *bufferManager_);
    return Status::OK();
}

Status DramPoolServer::InitProtocol()
{
    protocolManager_ = std::make_unique<ProtocolManager>();
    return Status::OK();
}

Status DramPoolServer::InitQueues()
{
    requestQueue_.Setup(g_config.requestQueueDepth);
    completionQueue_.Setup(g_config.completionQueueDepth);
    return Status::OK();
}

Status DramPoolServer::InitTransportManager()
{
    const auto localEndpoint = g_config.twoSidedToOneSided.find(g_config.addr.ToString());
    if (localEndpoint == g_config.twoSidedToOneSided.end()) {
        return Status::InvalidParam("local static transport endpoint is not configured");
    }
    transportManager_ = std::make_unique<transport::TransportManager>(localEndpoint->second);
    tcpMessageChannel_ = std::make_unique<transport::TcpMessageChannel>();
    return Status::OK();
}

Status DramPoolServer::StartTransportService()
{
    if (!transportManager_) {
        return Status::InvalidParam("DramPool transport manager is not initialized");
    }
    const auto localControlId = g_config.addr.ToString();
    const auto localEndpoint = g_config.twoSidedToOneSided.find(localControlId);
    if (localEndpoint == g_config.twoSidedToOneSided.end()) {
        return Status::InvalidParam("local static transport endpoint is not configured");
    }
    transport::HixlInitAttrs attrs;
    // DramPool actively connects to DramStore. Client HIXL instances use a
    // host-only engine ID and therefore do not open an HIXL listening port.
    transport::Endpoint managerEndpoint;
    if (auto status = ParseDramPoolEndpoint("transport local one_sided", localEndpoint->second,
                                            managerEndpoint);
        status.Failure()) {
        return status;
    }
    attrs.ip = managerEndpoint.host;
    attrs.role = transport::HixlRole::Client;
    attrs.instances.reserve(g_config.transportDeviceIds.size());
    for (const auto deviceId : g_config.transportDeviceIds) {
        transport::HixlInitAttrs::Instance instance;
        instance.port = -1;
        instance.device_id = deviceId;
        attrs.instances.push_back(std::move(instance));
    }
    attrs.connect_timeout_ms = static_cast<std::int32_t>(g_config.opTimeoutMs);
    attrs.transfer_timeout_ms = static_cast<std::int32_t>(g_config.opTimeoutMs);
    auto status = transportManager_->InstallTransport(transport::TransportProtocol::Hixl, attrs);
    if (status.Failure()) { return status; }
    if (auto registerStatus = RegisterBufferPools(); registerStatus.Failure()) {
        return registerStatus;
    }
    // Accept metadata exchanges initiated by DramStore. DramPool peer routing remains
    // static and does not actively call ExchangeMetadata().
    status = transportManager_->Init();
    if (status.Failure()) { return status; }
    return Status::OK();
}

Status DramPoolServer::RegisterBufferPools()
{
    if (!bufferManager_ || !flagBufferPool_) {
        return Status::InvalidParam("DramPool buffer pools are not initialized");
    }
    const auto& regions = bufferManager_->MemoryRegions();
    for (const auto& memory : regions) {
        transport::MemoryHandle handle = transport::kInvalidMemoryHandle;
        const auto status = transportManager_->RegisterMemory(memory, handle);
        if (status.Failure()) { return status; }
    }

    transport::MemoryRegion flagBufferRegion;
    flagBufferRegion.addr = flagBufferPool_->GetLocalAddr();
    flagBufferRegion.length = flagBufferPool_->GetTotalSize();
    flagBufferRegion.type = transport::MemoryType::Host;
    transport::MemoryHandle handle = transport::kInvalidMemoryHandle;
    const auto flagStatus = transportManager_->RegisterMemory(flagBufferRegion, handle);
    if (flagStatus.Failure()) { return flagStatus; }
    return Status::OK();
}

Status DramPoolServer::StartTcpMessageChannel()
{
    if (!tcpMessageChannel_) {
        return Status::InvalidParam("DramPool TCP message channel is not initialized");
    }

    const auto channelStatus = tcpMessageChannel_->Init(g_config.addr);
    if (channelStatus.Failure()) { return channelStatus; }
    {
        std::lock_guard<std::mutex> waitGuard(requestReceiverWaitMutex_);
        tcpMessageChannelReady_ = true;
    }
    requestReceiverWaitCv_.notify_one();
    return Status::OK();
}

Status DramPoolServer::CreateRuntimeContext()
{
    if (!flagBufferPool_ || !metadataManager_ || !protocolManager_ || !transportManager_) {
        return Status::InvalidParam("DramPool runtime dependencies are not initialized");
    }
    try {
        runtime_ = std::make_unique<DramPoolRuntime>(*metadataManager_, *flagBufferPool_,
                                                     *transportManager_, *protocolManager_,
                                                     requestQueue_, completionQueue_);
    } catch (const std::exception& error) {
        return Status::Error(std::string{"failed to create DramPool runtime: "} + error.what());
    }
    return Status::OK();
}

Status DramPoolServer::StartCompletionPoller()
{
    if (!runtime_) { return Status::InvalidParam("DramPool runtime is not initialized"); }
    try {
        completionPoller_ = std::make_unique<CompletionPoller>(*runtime_);
        completionPollerStop_.store(false, std::memory_order_release);
        completionPollerThread_ = std::thread(&DramPoolServer::CompletionPollerLoop, this);
    } catch (const std::exception& e) {
        completionPollerStop_.store(true, std::memory_order_release);
        completionPoller_.reset();
        return Status::Error(std::string{"failed to start CompletionPoller: "} + e.what());
    }
    return Status::OK();
}

Status DramPoolServer::StartTaskWorker()
{
    if (!runtime_) { return Status::InvalidParam("DramPool runtime is not initialized"); }
    try {
        taskWorker_ = std::make_unique<TaskWorker>(*runtime_);
        taskWorkerStop_.store(false, std::memory_order_release);
        taskWorkerThread_ = std::thread(&DramPoolServer::TaskWorkerLoop, this);
    } catch (const std::exception& e) {
        taskWorkerStop_.store(true, std::memory_order_release);
        taskWorker_.reset();
        return Status::Error(std::string{"failed to start TaskWorker: "} + e.what());
    }
    return Status::OK();
}

Status DramPoolServer::StartRequestReceiver()
{
    if (!runtime_ || !tcpMessageChannel_) {
        return Status::InvalidParam("DramPool RequestReceiver dependencies are not initialized");
    }
    try {
        {
            std::lock_guard<std::mutex> waitGuard(requestReceiverWaitMutex_);
            requestReceiverStop_.store(false, std::memory_order_release);
        }
        requestReceiverThread_ = std::thread(&DramPoolServer::RequestReceiveLoop, this);
    } catch (const std::exception& e) {
        requestReceiverStop_.store(true, std::memory_order_release);
        return Status::Error(std::string{"failed to start RequestReceiver: "} + e.what());
    }
    return Status::OK();
}

Status DramPoolServer::StartGCThread()
{
    if (!g_config.gcEnabled) { return Status::OK(); }
    try {
        gcThreadStop_.store(false, std::memory_order_release);
        gcThread_ = std::thread(&DramPoolServer::GCThreadLoop, this);
    } catch (const std::exception& e) {
        gcThreadStop_.store(true, std::memory_order_release);
        return Status::Error(std::string{"failed to start GCThread: "} + e.what());
    }
    return Status::OK();
}

void DramPoolServer::StopTcpMessageChannel()
{
    {
        std::lock_guard<std::mutex> waitGuard(requestReceiverWaitMutex_);
        tcpMessageChannelReady_ = false;
    }
    if (tcpMessageChannel_) {
        const auto status = tcpMessageChannel_->Shutdown();
        if (status.Failure()) {
            UC_ERROR_UNLIMITED("DramPool TCP message channel shutdown failed");
        }
    }
}

void DramPoolServer::StopRequestReceiver()
{
    {
        std::lock_guard<std::mutex> waitGuard(requestReceiverWaitMutex_);
        requestReceiverStop_.store(true, std::memory_order_release);
    }
    requestReceiverWaitCv_.notify_one();
    StopTcpMessageChannel();
    if (requestReceiverThread_.joinable()) { requestReceiverThread_.join(); }
}

void DramPoolServer::StopTaskWorker()
{
    taskWorkerStop_.store(true, std::memory_order_release);
    if (taskWorkerThread_.joinable()) { taskWorkerThread_.join(); }
}

void DramPoolServer::StopCompletionPoller()
{
    completionPollerStop_.store(true, std::memory_order_release);
    if (completionPollerThread_.joinable()) { completionPollerThread_.join(); }
}

void DramPoolServer::StopGCThread()
{
    gcThreadStop_.store(true, std::memory_order_release);
    stopWaitCv_.notify_one();
    if (gcThread_.joinable()) { gcThread_.join(); }
}

void DramPoolServer::StopTransportService()
{
    if (transportManager_) {
        const auto status = transportManager_->Shutdown();
        if (status.Failure()) { UC_ERROR_UNLIMITED("DramPool TransportManager shutdown failed"); }
    }
}

void DramPoolServer::TaskWorkerLoop()
{
    UC_INFO_UNLIMITED("DramPool TaskWorker started");
    taskWorker_->Run(taskWorkerStop_);
    UC_INFO_UNLIMITED("DramPool TaskWorker stopped");
}

void DramPoolServer::RequestReceiveLoop()
{
    UC_INFO_UNLIMITED("DramPool RequestReceiver started, addr={}", g_config.addr.ToString());
    if (!WaitForChannelReady()) {
        UC_INFO_UNLIMITED("DramPool RequestReceiver stopped before TCP channel became ready");
        return;
    }

    const auto idleWait = std::chrono::microseconds(g_config.requestReceiverIdleWaitUs);
    while (!requestReceiverStop_.load(std::memory_order_acquire)) {
        transport::Endpoint controlPeer;
        transport::Metadata received;
        const auto receiveStatus = tcpMessageChannel_->Receive(controlPeer, received);
        if (requestReceiverStop_.load(std::memory_order_acquire)) { break; }
        if (receiveStatus.Failure()) {
            UC_ERROR("RequestReceiver TCP message channel stopped unexpectedly");
            break;
        }

        RequestPtr request;
        const auto unpackStatus =
            runtime_->protocol.UnpackRequest(received.data(), received.size(), request);
        if (unpackStatus.Failure()) {
            UC_WARN("RequestReceiver rejected KV request from {}: {}", controlPeer.ToString(),
                    unpackStatus);
            continue;
        }

        const auto controlPeerId = controlPeer.ToString();
        const auto peerIt = g_config.twoSidedToOneSided.find(controlPeerId);
        if (peerIt == g_config.twoSidedToOneSided.end()) {
            UC_WARN("RequestReceiver rejected unconfigured control peer {}", controlPeerId);
            continue;
        }

        auto task = std::make_unique<RequestTask>();
        task->request = std::move(request);
        task->peer_one_sided_id = peerIt->second;
        // This bounded handoff keeps transport I/O separate from potentially slow request handling.
        bool queueFullLogged = false;
        while (!requestReceiverStop_.load(std::memory_order_acquire)) {
            if (requestQueue_.TryPush(std::move(task))) { break; }
            if (!queueFullLogged) {
                UC_WARN("RequestReceiver queue is full, depth={}, retry_wait_us={}",
                        g_config.requestQueueDepth, g_config.requestReceiverIdleWaitUs);
                queueFullLogged = true;
            }
            std::this_thread::sleep_for(idleWait);
        }
    }
    UC_INFO_UNLIMITED("DramPool RequestReceiver stopped");
}

bool DramPoolServer::WaitForChannelReady()
{
    std::unique_lock<std::mutex> waitLock(requestReceiverWaitMutex_);
    requestReceiverWaitCv_.wait(waitLock, [this]() {
        return tcpMessageChannelReady_ || requestReceiverStop_.load(std::memory_order_acquire);
    });
    return tcpMessageChannelReady_ && !requestReceiverStop_.load(std::memory_order_acquire);
}

void DramPoolServer::CompletionPollerLoop()
{
    UC_INFO_UNLIMITED("DramPool CompletionPoller started");
    completionPoller_->Run(completionPollerStop_);
    UC_INFO_UNLIMITED("DramPool CompletionPoller stopped");
}

void DramPoolServer::GCThreadLoop()
{
    UC_INFO_UNLIMITED("DramPool GCThread started, interval_ms={}", g_config.gcIntervalMs);
    const auto interval = std::chrono::milliseconds(g_config.gcIntervalMs);
    const auto stopRequested = [this]() { return gcThreadStop_.load(std::memory_order_acquire); };
    while (true) {
        std::unique_lock<std::mutex> waitLock(stopWaitMutex_);
        if (stopWaitCv_.wait_for(waitLock, interval, stopRequested)) { break; }
        metadataManager_->PerformEvict();
    }
    UC_INFO_UNLIMITED("DramPool GCThread stopped");
}

void DramPoolServer::ResetInitializedComponents()
{
    // Dependents must be destroyed before the non-owning runtime context and
    // the components referenced by that context.
    completionPoller_.reset();
    taskWorker_.reset();
    runtime_.reset();
    protocolManager_.reset();
    metadataManager_.reset();
    tcpMessageChannel_.reset();
    flagBufferPool_.reset();
    bufferManager_.reset();
    transportManager_.reset();
    if (deviceRuntimeOwned_) {
        if (deviceId_) { (void)device_.Reset(*deviceId_); }
        (void)device_.Finalize();
        deviceRuntimeOwned_ = false;
    }
    deviceId_.reset();
}

}  // namespace UC::DramPool
