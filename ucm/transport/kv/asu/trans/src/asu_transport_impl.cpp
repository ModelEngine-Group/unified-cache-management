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
#include "asu_transport_impl.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#ifdef UCM_ASU_ENABLE_AICPU_PROVIDER
#include "aicpu_trans_provider.h"
#endif
#ifdef UCM_ASU_ENABLE_AIV_PROVIDER
#include "aiv_trans_provider.h"
#endif
#include "asu_transport/asu_transport.h"
#include "connection_manager.h"
#ifdef UCM_ASU_ENABLE_FAKE_PROVIDER
#include "fake_trans_provider.h"
#endif
#include "logger.h"
#include "transport_config_parser.h"

namespace UC::ASU {

namespace {

std::chrono::steady_clock::time_point TaskDeadline(std::uint64_t timeoutMs)
{
    if (timeoutMs == 0) { return std::chrono::steady_clock::time_point::max(); }
    return std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
}

}  // namespace

AsuTransportImpl::AsuTransportImpl() = default;

AsuTransportImpl::~AsuTransportImpl() { Shutdown(); }

Status AsuTransportImpl::Init(const std::string& configPath)
{
    TransportConfig config;
    auto status = LoadTransportConfig(configPath, config);
    if (!status.ok()) { return status; }
    return Init(config);
}

Status AsuTransportImpl::Init(const TransportConfig& config)
{
    UC_DEBUG("AsuTransportImpl::Init start");
    if (worker_.joinable()) {
        UC_DEBUG("AsuTransportImpl::Init already initialized");
        return Status::OK();
    }
    config_ = config;
    if (config_.maxErrorCount == 0) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "maxErrorCount must be greater than 0");
    }

    if (!transProvider_) {
        switch (config_.providerType) {
            case TransProviderType::AICPU:
#ifdef UCM_ASU_ENABLE_AICPU_PROVIDER
                transProvider_ = std::make_unique<AICPUTransProvider>();
                break;
#else
                return Status::Error(
                    StatusCode::UNSUPPORTED,
                    "AICPU trans provider is not built; enable BUILD_UCM_ASU_PROVIDER_AICPU");
#endif
            case TransProviderType::FAKE:
#ifdef UCM_ASU_ENABLE_FAKE_PROVIDER
                transProvider_ =
                    std::make_unique<FakeTransProvider>(MakeFakeTransProviderConfig(config_));
                break;
#else
                return Status::Error(
                    StatusCode::UNSUPPORTED,
                    "FAKE trans provider is not built; enable BUILD_UCM_ASU_PROVIDER_FAKE");
#endif
            case TransProviderType::AIV:
#ifdef UCM_ASU_ENABLE_AIV_PROVIDER
                transProvider_ = std::make_unique<AIVTransProviderAdapter>(config_.deviceId);
                break;
#else
                return Status::Error(
                    StatusCode::UNSUPPORTED,
                    "AIV trans provider is not built; enable BUILD_UCM_ASU_PROVIDER_AIV and set "
                    "ASU_AIV_PROVIDER_ROOT");
#endif
            case TransProviderType::UNSUPPORTED:
                return Status::Error(StatusCode::UNSUPPORTED,
                                     "ASU trans provider backend is not supported");
        }
    }
    if (!transProvider_) {
        UC_ERROR("AsuTransportImpl::Init: TransProvider is null");
        return Status::Error(StatusCode::NOT_INITIALIZED, "TransProvider is null");
    }

    std::string localIp;
    auto it = config_.attrs.find("localIp");
    if (it != config_.attrs.end()) { localIp = it->second; }

    std::uint32_t timeout = 5000;
    auto tit = config_.attrs.find("timeout");
    if (tit != config_.attrs.end()) {
        timeout = static_cast<std::uint32_t>(std::stoul(tit->second));
    }

    connManager_.reset();
    connManager_ = std::make_unique<ConnectionManager>(*transProvider_, localIp, timeout,
                                                       config_.maxErrorCount);

    std::uint32_t qp_num = config_.queryQpNum + config_.loadQpNum + config_.storeQpNum;
    UC_DEBUG("AsuTransportImpl::Init endpoints={} qp_num={}", config_.endpoints.size(), qp_num);
    for (const auto& ep : config_.endpoints) {
        auto s = connManager_->AddGroup(ep, qp_num);
        if (!s.ok()) {
            UC_DEBUG("AsuTransportImpl::Init AddGroup FAILED: {}", s.message);
            const auto shutdownStatus = Shutdown();
            if (!shutdownStatus.ok()) {
                UC_WARN(
                    "AsuTransportImpl::Init cleanup failed after AddGroup failure: code={} "
                    "message={}",
                    static_cast<int>(shutdownStatus.code), shutdownStatus.message);
            }
            return s;
        }
    }

    const auto serverCapabilities = connManager_->GetServerCapabilities();
    for (std::size_t index = 0; index < serverCapabilities.size(); ++index) {
        const auto& capabilities = serverCapabilities[index];
        UC_INFO(
            "AsuTransportImpl::Init ServerKvCapabilities group={} queueNum={} ioQueueDepth={} "
            "ioQueueKeyConcurrency={} connectionKeyConcurrency={} singleValueMaxBytes={} "
            "batchValueMaxBytes={} batchStoreKeys={} batchLoadKeys={} deleteKeys={} "
            "queryKeys={} keyLength={} kvCapabilities={}",
            index, capabilities.queueNum, capabilities.ioQueueDepth,
            capabilities.ioQueueKeyConcurrency, capabilities.connectionKeyConcurrency,
            capabilities.singleValueMaxBytes, capabilities.batchValueMaxBytes,
            capabilities.batchStoreKeys, capabilities.batchLoadKeys, capabilities.deleteKeys,
            capabilities.queryKeys, capabilities.keyLength, capabilities.kvCapabilities);
        auto applyCapLimit = [](auto& cfgField, auto capValue) {
            if (capValue != 0) {
                cfgField = std::min(cfgField, static_cast<std::size_t>(capValue));
            }
        };
        applyCapLimit(config_.asuBatchStoreIoNum, capabilities.batchStoreKeys);
        applyCapLimit(config_.asuBatchLoadIoNum, capabilities.batchLoadKeys);
        applyCapLimit(config_.asuDeleteIoNum, capabilities.deleteKeys);
        applyCapLimit(config_.asuQueryIoNum, capabilities.queryKeys);
    }
    UC_INFO("AsuTransportImpl::Init effective batch limits: store={} load={} delete={} query={}",
            config_.asuBatchStoreIoNum, config_.asuBatchLoadIoNum, config_.asuDeleteIoNum,
            config_.asuQueryIoNum);

    ioScheduler_ = IoScheduler(config_);
    connManager_->StartRecoverLoop();

    auto status = sendBufferManager_.Init("asu send buffer", MemoryType::HOST_PINNED,
                                          config_.sendBufferSlotSize, config_.sendBufferSlotNum,
                                          transProvider_.get());
    if (!status.ok()) {
        const auto shutdownStatus = Shutdown();
        if (!shutdownStatus.ok()) {
            UC_WARN(
                "AsuTransportImpl::Init cleanup failed after send buffer initialization "
                "failure: code={} message={}",
                static_cast<int>(shutdownStatus.code), shutdownStatus.message);
        }
        return status;
    }

    status = flagBufferManager_.Init("asu flag buffer", MemoryType::HOST_PINNED,
                                     config_.flagBufferSlotSize, config_.flagBufferSlotNum,
                                     transProvider_.get());
    if (!status.ok()) {
        const auto shutdownStatus = Shutdown();
        if (!shutdownStatus.ok()) {
            UC_WARN(
                "AsuTransportImpl::Init cleanup failed after flag buffer initialization "
                "failure: code={} message={}",
                static_cast<int>(shutdownStatus.code), shutdownStatus.message);
        }
        return status;
    }
    protocolManager_ = std::make_unique<ProtocolManager>();
    taskExecutor_ = std::make_unique<TransportTaskExecutor>(
        config_, ioScheduler_, transProvider_, sendBufferManager_, flagBufferManager_,
        protocolManager_, connManager_, nextRequestCid_, registeredRegionsMu_, registeredRegions_);
    auto queueDepth = std::max<std::size_t>(2, static_cast<std::size_t>(config_.maxInflightTasks));
    executeQueue_.Setup(queueDepth + 1);
    stopWorker_.store(false, std::memory_order_release);
    stopCompletionWorker_.store(false, std::memory_order_release);
    worker_ = std::thread(&AsuTransportImpl::WorkerLoop, this);
    completionWorker_ = std::thread(&AsuTransportImpl::CompletionLoop, this);
    UC_DEBUG("AsuTransportImpl::Init OK: queueDepth={}", queueDepth);
    return Status::OK();
}

Status AsuTransportImpl::Shutdown()
{
    Status finalStatus = Status::OK();
    {
        std::lock_guard<std::mutex> lock(producerMu_);
        stopWorker_.store(true, std::memory_order_release);
    }

    if (worker_.joinable()) { worker_.join(); }

    if (completionWorker_.joinable() && config_.timeoutMs != 0) {
        bool hasInflightTask = false;
        for (const auto& ctx : taskManager_.GetAll()) {
            if (ctx != nullptr &&
                ctx->state.load(std::memory_order_acquire) == TransportTaskState::INFLIGHT) {
                hasInflightTask = true;
                break;
            }
        }
        if (hasInflightTask) {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.timeoutMs));
        }
    }

    stopCompletionWorker_.store(true, std::memory_order_release);
    if (completionWorker_.joinable()) { completionWorker_.join(); }

    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx == nullptr) { continue; }
        const auto canceledStatus =
            Status::Error(StatusCode::CANCELED, "transport shutdown canceled task");
        if (!taskExecutor_->Cancel(ctx, canceledStatus)) { continue; }
        taskManager_.NotifyCompletion(ctx);
    }
    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx != nullptr) { (void)taskManager_.Remove(ctx->taskId); }
    }
    taskExecutor_.reset();
    {
        std::lock_guard<std::mutex> lock(registeredRegionsMu_);
        if (ownsRegisteredRegionHandles_) {
            std::vector<MRHandle> handles;
            handles.reserve(registeredRegions_.size());
            for (const auto& item : registeredRegions_) { handles.push_back(item.first); }
            if (!handles.empty() && transProvider_) {
                const auto status = UnregisterOwnedRegionHandles(handles);
                if (!status.ok()) {
                    UC_ERROR(
                        "Transport shutdown ignored memory unregister failure: code={} "
                        "message={}",
                        static_cast<int>(status.code), status.message);
                }
            }
        }
        registeredRegions_.clear();
        ownsRegisteredRegionHandles_ = false;
    }
    flagBufferManager_.Shutdown();
    sendBufferManager_.Shutdown();

    if (connManager_) {
        auto status = connManager_->Shutdown();
        if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
        connManager_.reset();
    }

    UC_DEBUG("AsuTransportImpl::Shutdown OK");
    return finalStatus;
}

Status AsuTransportImpl::CheckHealth()
{
    if (!worker_.joinable() || !completionWorker_.joinable()) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "transport worker is not running");
    }
    return Status::OK();
}

Status AsuTransportImpl::Submit(const TransportTaskPtr& task)
{
    if (!task) { return Status::Error(StatusCode::INVALID_ARGUMENT, "transport task is null"); }
    const auto entryCount = IsEntryBatchOp(task->opType) ? task->entries.size() : task->keys.size();
    task->entryStatus.assign(entryCount, Status::OK());
    task->deadline = TaskDeadline(config_.timeoutMs);
    return SubmitTask(task);
}

Status AsuTransportImpl::Cancel(TaskId taskId)
{
    auto ctx = taskManager_.Get(taskId);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found"); }

    const auto canceledStatus = Status::Error(StatusCode::CANCELED, "transport task canceled");
    if (!taskExecutor_->Cancel(ctx, canceledStatus)) { return Status::OK(); }
    taskManager_.NotifyCompletion(ctx);
    return Status::OK();
}

Status AsuTransportImpl::RegisterRegions(const std::vector<MemoryRegion>& regions,
                                         std::vector<RegisteredMemory>& registeredRegions)
{
    registeredRegions.clear();
    registeredRegions.reserve(regions.size());
    if (regions.empty()) { return Status::OK(); }

    std::lock_guard<std::mutex> lock(registeredRegionsMu_);
    std::vector<TransProvider::RegisterMemoryDesc> registerDescs;
    registerDescs.reserve(regions.size());
    for (const auto& region : regions) {
        const auto memType = region.memoryType == MemoryType::ASCEND_DEVICE
                                 ? TransProvider::MemType::MEM_DEVICE
                                 : TransProvider::MemType::MEM_HOST;
        registerDescs.push_back({memType, static_cast<std::uintptr_t>(region.addr),
                                 static_cast<std::size_t>(region.size)});
    }

    std::vector<MRHandle> mrHandles;

    auto status = transProvider_->RegisterMemory(registerDescs, mrHandles);
    if (!status.ok()) {
        const auto cleanupStatus = UnregisterOwnedRegionHandles(mrHandles);
        return Status::Error(StatusCode::PARTIAL_FAILED,
                             cleanupStatus.ok()
                                 ? "one or more memory regions failed to register"
                                 : "memory region registration failed and cleanup was incomplete");
    }
    if (mrHandles.size() != regions.size()) {
        status = Status::Error(StatusCode::INTERNAL_ERROR,
                               "register result count does not match region count");
        const auto cleanupStatus = UnregisterOwnedRegionHandles(mrHandles);
        return Status::Error(StatusCode::PARTIAL_FAILED,
                             cleanupStatus.ok()
                                 ? status.message
                                 : "register result count mismatch and cleanup was incomplete");
    }

    std::vector<std::uint32_t> tokenIds(regions.size());
    for (std::size_t index = 0; index < mrHandles.size(); ++index) {
        status = transProvider_->GetMemTokenId(mrHandles[index], tokenIds[index]);
        if (status.ok()) { continue; }

        const auto cleanupStatus = UnregisterOwnedRegionHandles(mrHandles);
        return Status::Error(StatusCode::PARTIAL_FAILED,
                             cleanupStatus.ok()
                                 ? "one or more memory region token lookups failed"
                                 : "memory region token lookup failed and cleanup was incomplete");
    }

    for (std::size_t index = 0; index < regions.size(); ++index) {
        RegisteredMemory registeredMemory;
        registeredMemory.region = regions[index];
        registeredMemory.handle = mrHandles[index];
        registeredMemory.tokenId = tokenIds[index];
        registeredRegions_[mrHandles[index]] = registeredMemory;
        registeredRegions.emplace_back(registeredMemory);
    }
    ownsRegisteredRegionHandles_ = true;
    return Status::OK();
}

Status AsuTransportImpl::BindRegisteredRegions(const std::vector<RegisteredMemory>& regions)
{
    std::lock_guard<std::mutex> lock(registeredRegionsMu_);
    for (const auto& region : regions) { registeredRegions_[region.handle] = region; }
    return Status::OK();
}

Status AsuTransportImpl::UnregisterRegions(const std::vector<MRHandle>& handles)
{
    std::lock_guard<std::mutex> lock(registeredRegionsMu_);
    if (!ownsRegisteredRegionHandles_) { return Status::OK(); }
    std::vector<MRHandle> ownedHandles;
    ownedHandles.reserve(handles.size());
    for (auto handle : handles) {
        if (handle == kInvalidMRHandle) { continue; }
        if (registeredRegions_.find(handle) == registeredRegions_.end()) { continue; }
        ownedHandles.push_back(handle);
    }
    return UnregisterOwnedRegionHandles(ownedHandles);
}

Status AsuTransportImpl::UnregisterOwnedRegionHandles(const std::vector<MRHandle>& handles)
{
    if (handles.empty()) { return Status::OK(); }

    std::vector<TransProvider::UnregisterMemoryDesc> unregisterDescs;
    unregisterDescs.reserve(handles.size());
    for (const auto handle : handles) {
        unregisterDescs.push_back(TransProvider::UnregisterMemoryDesc{handle});
    }

    const auto statuses = transProvider_->UnregisterMemory(unregisterDescs);
    Status failure = statuses.size() == handles.size()
                         ? Status::OK()
                         : Status::Error(StatusCode::INTERNAL_ERROR,
                                         "unregister result count does not match handle count");
    for (std::size_t index = 0; index < handles.size(); ++index) {
        if (index < statuses.size() && statuses[index].ok()) {
            registeredRegions_.erase(handles[index]);
        } else if (failure.ok()) {
            failure = index < statuses.size()
                          ? statuses[index]
                          : Status::Error(StatusCode::INTERNAL_ERROR,
                                          "unregister result count does not match handle count");
        }
    }
    ownsRegisteredRegionHandles_ = !registeredRegions_.empty();
    return failure;
}

Status AsuTransportImpl::SubmitTask(const TransportTaskPtr& task)
{
    std::lock_guard<std::mutex> lock(producerMu_);
    if (stopWorker_.load(std::memory_order_acquire) || !worker_.joinable()) {
        task->taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "transport is not accepting tasks");
    }

    TaskId taskId = kInvalidTaskId;
    auto status = taskManager_.Submit(task, taskId);
    if (!status.ok()) { return status; }

    auto submittedTask = taskManager_.Get(taskId);
    if (!submittedTask) {
        task->taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "transport task disappeared after submit");
    }

    if (!executeQueue_.TryPush(std::move(submittedTask))) {
        taskManager_.Remove(taskId);
        task->taskId = kInvalidTaskId;
        return Status::Error(StatusCode::RESOURCE_BUSY, "transport task queue is full");
    }
    return Status::OK();
}

void AsuTransportImpl::WorkerLoop()
{
    executeQueue_.ConsumerLoop(stopWorker_, [this](TransportTaskPtr task) {
        if (!task) { return; }
        if (taskExecutor_->Execute(task)) { taskManager_.NotifyCompletion(task); }
    });
}

void AsuTransportImpl::CompletionLoop()
{
    while (!stopCompletionWorker_.load(std::memory_order_acquire)) {
        for (const auto& ctx : taskManager_.GetAll()) {
            if (taskExecutor_->Poll(ctx)) { taskManager_.NotifyCompletion(ctx); }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void AsuTransportImpl::SetTransProvider(std::unique_ptr<TransProvider> provider)
{
    transProvider_ = std::move(provider);
}

std::unique_ptr<AsuTransport> CreateAsuTransport() { return std::make_unique<AsuTransportImpl>(); }

extern "C" std::unique_ptr<AsuTransport> UcmAsuCreateAsuTransport() { return CreateAsuTransport(); }

}  // namespace UC::ASU
