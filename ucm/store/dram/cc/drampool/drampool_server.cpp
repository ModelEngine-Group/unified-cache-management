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
#include <acl/acl.h>
#include <chrono>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include "buffer_manager.h"
#include "core/transport_manager.h"
#include "logger/logger.h"
#include "metadata.h"
#include "task_worker.h"
#include "two_sided/tcp/tcp_message_channel.h"

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
        if (auto status = InitializeAclRuntime(); status.Failure()) { return status; }
        if (auto status = InitMemoryPool(); status.Failure()) { return status; }
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
    MarkInflightTransportsFailed();
    StopCompletionPoller();
    StopGCThread();
    StopTransportService();
    // Object destruction is shared with Init() rollback and happens only after
    // every active thread and transport service has stopped.
    ResetInitializedComponents();
    state_ = ServerState::Stopped;
}

Status DramPoolServer::InitializeAclRuntime()
{
    const auto initStatus = aclInit(nullptr);
    if (initStatus == ACL_SUCCESS) {
        aclRuntimeOwned_ = true;
    } else if (initStatus != ACL_ERROR_REPEAT_INITIALIZE) {
        return Status::Error("aclInit failed: " + std::to_string(initStatus));
    }

    const auto setDeviceStatus = aclrtSetDevice(g_config.transportDeviceId);
    if (setDeviceStatus == ACL_SUCCESS) { return Status::OK(); }

    if (aclRuntimeOwned_) {
        (void)aclFinalize();
        aclRuntimeOwned_ = false;
    }
    return Status::Error("aclrtSetDevice failed: " + std::to_string(setDeviceStatus));
}

Status DramPoolServer::InitMemoryPool()
{
    constexpr char kBufferManagerNamePrefix[] = "drampool-kvcache-";
    for (std::size_t index = 0; index < g_config.poolBlockSizes.size(); ++index) {
        auto bufferManager = std::make_unique<UC::ASU::BufferManager>();
        const auto status = bufferManager->Init(
            kBufferManagerNamePrefix + std::to_string(index), UC::ASU::MemoryType::HOST,
            static_cast<std::size_t>(g_config.poolBlockSizes[index]),
            static_cast<std::size_t>(g_config.poolSlotCounts[index]), nullptr);
        if (!status.ok()) { return Status::Error("BufferManager::Init failed: " + status.message); }
        bufferManagers_.push_back(std::move(bufferManager));
    }
    return Status::OK();
}

Status DramPoolServer::InitMetadata()
{
    // Metadata eviction settings come from the process-wide runtime configuration.
    const UC::DramPool::MetadataConfig config{
        UC::DramPool::EvictionPolicyType::TTL,
        UC::DramPool::EvictionPolicyType::POSITION,
        std::chrono::milliseconds(g_config.metadataLeaseTimeMs),
        g_config.metadataDefaultEvictRatio,
        std::chrono::milliseconds(g_config.metadataEvictPeriodMs),
    };
    metadataManager_ = std::make_unique<UC::DramPool::MetadataManager>(config);
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
    if (bufferManagers_.empty()) { return Status::InvalidParam("memory pool is not initialized"); }
    const auto localControlId = g_config.addr.ToString();
    const auto localEndpoint = g_config.twoSidedToOneSided.find(localControlId);
    if (localEndpoint == g_config.twoSidedToOneSided.end()) {
        return Status::InvalidParam("local static transport endpoint is not configured");
    }
    transport::HixlInitAttrs attrs;
    // A -1 port lets HIXL allocate its internal endpoint without occupying the
    // TransportManager metadata listener port.
    transport::Endpoint managerEndpoint;
    if (auto status = ParseDramPoolEndpoint("transport local one_sided", localEndpoint->second,
                                            managerEndpoint);
        status.Failure()) {
        return status;
    }
    const bool isIpv6 = managerEndpoint.host.find(':') != std::string::npos;
    attrs.local_engine = (isIpv6 ? "[" + managerEndpoint.host + "]" : managerEndpoint.host) + ":-1";
    attrs.device_id = g_config.transportDeviceId;
    attrs.connect_timeout_ms = static_cast<std::int32_t>(g_config.opTimeoutMs);
    attrs.transfer_timeout_ms = static_cast<std::int32_t>(g_config.opTimeoutMs);
    auto status = transportManager_->InstallTransport(transport::TransportProtocol::Hixl, attrs);
    if (status != transport::Status::Ok) {
        return ToUcStatus(status, "TransportManager::InstallTransport");
    }
    // Accept metadata exchanges initiated by DramStore. DramPool peer routing remains
    // static and does not actively call ExchangeMetadata().
    status = transportManager_->Init();
    if (status != transport::Status::Ok) { return ToUcStatus(status, "TransportManager::Init"); }
    return Status::OK();
}

Status DramPoolServer::StartTcpMessageChannel()
{
    if (!tcpMessageChannel_) {
        return Status::InvalidParam("DramPool TCP message channel is not initialized");
    }

    const auto channelStatus = tcpMessageChannel_->Init(g_config.addr);
    if (channelStatus != transport::Status::Ok) {
        return ToUcStatus(channelStatus, "TcpMessageChannel::Init");
    }
    {
        std::lock_guard<std::mutex> waitGuard(requestReceiverWaitMutex_);
        tcpMessageChannelReady_ = true;
    }
    requestReceiverWaitCv_.notify_one();
    return Status::OK();
}

Status DramPoolServer::CreateRuntimeContext()
{
    if (!metadataManager_ || !protocolManager_ || !transportManager_) {
        return Status::InvalidParam("DramPool runtime dependencies are not initialized");
    }
    if (bufferManagers_.empty()) {
        return Status::InvalidParam("DramPool buffer managers are not initialized");
    }
    try {
        runtime_ = std::make_unique<DramPoolRuntime>(*metadataManager_, bufferManagers_,
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
        if (status != transport::Status::Ok) {
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

void DramPoolServer::MarkInflightTransportsFailed()
{
    if (completionPoller_) { completionPoller_->RequestDrainAllAsFailed(); }
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
        if (status != transport::Status::Ok) {
            UC_ERROR_UNLIMITED("DramPool TransportManager shutdown failed");
        }
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
        if (receiveStatus != transport::Status::Ok) {
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
        waitLock.unlock();
        // TODO: Run periodic eviction after MetadataManager exposes a production API
        // that returns each victim Entry (including its buffer slot). GC must release
        // the BufferManager slot before removing the metadata entry; the current
        // metadata eviction path only returns keys and does not release DRAM buffers.
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
    bufferManagers_.clear();
    transportManager_.reset();
    if (aclRuntimeOwned_) {
        (void)aclrtResetDevice(g_config.transportDeviceId);
        (void)aclFinalize();
        aclRuntimeOwned_ = false;
    }
}

}  // namespace UC::DramPool
