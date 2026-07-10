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
#include <acl/acl.h>
#include <algorithm>
#include <array>
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
#include "asu_response_status.h"
#include "asu_transport/asu_transport.h"
#include "connection_internal.h"
#include "connection_manager.h"
#ifdef UCM_ASU_ENABLE_FAKE_PROVIDER
#include "fake_trans_provider.h"
#endif
#include "logger.h"
#include "transport_config_parser.h"

namespace UC::ASU {

namespace {

constexpr std::size_t kFlagBufferHeaderCopySize = kCqeDwordCount * sizeof(std::uint32_t);

Status CopyDeviceToHost(const ScatterGatherEntry& sge, void* host, std::size_t size)
{
    if (size > sge.length) {
        return Status::Error(StatusCode::INVALID_ARGUMENT, "copy size exceeds buffer length");
    }
    const auto ret = aclrtMemcpy(host, size, reinterpret_cast<void*>(sge.device_addr), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return Status::Error(StatusCode::INTERNAL_ERROR,
                             "copy device memory to host failed ret=" + std::to_string(ret));
    }
    return Status::OK();
}

}  // namespace

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
    ioScheduler_ = IoScheduler(config_);

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
                transProvider_ = std::make_unique<AIVTransProviderAdapter>(
                    config_.endpoints.empty() ? -1 : config_.endpoints.front().deviceId);
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
    connManager_ = std::make_unique<ConnectionManager>(*transProvider_, localIp, timeout);

    std::uint32_t qp_num = config_.queryQpNum + config_.loadQpNum + config_.storeQpNum;
    UC_DEBUG("AsuTransportImpl::Init endpoints={} qp_num={}", config_.endpoints.size(), qp_num);
    for (const auto& ep : config_.endpoints) {
        auto s = connManager_->AddGroup(ep, qp_num);
        if (!s.ok()) {
            UC_DEBUG("AsuTransportImpl::Init AddGroup FAILED: {}", s.message);
            (void)Shutdown();
            return s;
        }
    }

    connManager_->StartRecoverLoop();

    auto status = sendBufferManager_.Init("asu send buffer", MemoryType::HOST_PINNED,
                                          config_.sendBufferSlotSize, config_.sendBufferSlotNum,
                                          transProvider_.get());
    if (!status.ok()) {
        (void)Shutdown();
        return status;
    }

    status = flagBufferManager_.Init("asu flag buffer", MemoryType::HOST_PINNED,
                                     config_.flagBufferSlotSize, config_.flagBufferSlotNum,
                                     transProvider_.get());
    if (!status.ok()) {
        (void)Shutdown();
        return status;
    }
    protocolManager_ = std::make_unique<ProtocolManager>();

    auto queueDepth = std::max<std::size_t>(2, static_cast<std::size_t>(config_.maxInflightTasks));
    executeQueue_.Setup(queueDepth + 1);
    stop_.store(false, std::memory_order_release);
    worker_ = std::thread(&AsuTransportImpl::WorkerLoop, this);
    completionWorker_ = std::thread(&AsuTransportImpl::CompletionLoop, this);
    UC_DEBUG("AsuTransportImpl::Init OK: queueDepth={}", queueDepth);
    return Status::OK();
}

Status AsuTransportImpl::Shutdown()
{
    Status finalStatus = Status::OK();
    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx == nullptr) { continue; }
        std::lock_guard<std::mutex> lock(ctx->waitMu);
        if (ctx->Done()) { continue; }
        ctx->finalStatus = Status::Error(StatusCode::CANCELED, "transport shutdown canceled task");
        ReleaseAllSubBatchResources(ctx->subBatchContexts);
        ctx->state.store(TransportTaskState::CANCELED, std::memory_order_release);
        ctx->cv.notify_all();
    }

    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
        UC_DEBUG("AsuTransportImpl::Shutdown stopping worker thread");
        worker_.join();
    }
    if (completionWorker_.joinable()) { completionWorker_.join(); }
    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx != nullptr) { (void)taskManager_.Remove(ctx->taskId); }
    }
    {
        std::lock_guard<std::mutex> lock(registeredRegionsMu_);
        std::vector<TransProvider::UnregisterMemoryDesc> descs;
        descs.reserve(ownedRegisteredRegionHandles_.size());
        for (auto handle : ownedRegisteredRegionHandles_) {
            descs.push_back(TransProvider::UnregisterMemoryDesc{handle});
        }
        if (!descs.empty() && transProvider_) {
            const auto statuses = transProvider_->UnregisterMemory(descs);
            for (const auto& status : statuses) {
                if (!status.ok() && finalStatus.ok()) { finalStatus = status; }
            }
        }
        registeredRegions_.clear();
        ownedRegisteredRegionHandles_.clear();
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

Status AsuTransportImpl::Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                               QueryResult& result)
{
    TaskId taskId{kInvalidTaskId};
    auto status = QueryAsync(keys, options, taskId);
    if (!status.ok()) { return status; }

    TaskResult taskResult;
    const auto timeoutMs = options.timeoutMs == 0 ? config_.queryTimeoutMs : options.timeoutMs;
    status = Wait(taskId, timeoutMs, taskResult);
    if (!status.ok()) { return status; }
    if (taskResult.queryResult.has_value()) { result = *taskResult.queryResult; }
    return taskResult.status;
}

Status AsuTransportImpl::QueryAsync(const std::vector<CacheKey>& keys, const QueryOptions& options,
                                    TaskId& taskId)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->opType = TransportOpType::QUERY;
    ctx->keys = BatchView<CacheKey>{keys.data(), keys.size()};
    ctx->queryOptions = options;
    ctx->entryStatus.assign(keys.size(), Status::OK());
    return SubmitAsync(std::move(ctx), taskId);
}

Status AsuTransportImpl::LoadAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->opType = TransportOpType::BATCH_LOAD;
    ctx->entries = BatchView<KVBuffer>{entries.data(), entries.size()};
    ctx->entryStatus.assign(entries.size(), Status::OK());
    return SubmitAsync(std::move(ctx), taskId);
}

Status AsuTransportImpl::StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->opType = TransportOpType::BATCH_STORE;
    ctx->entries = BatchView<KVBuffer>{entries.data(), entries.size()};
    ctx->entryStatus.assign(entries.size(), Status::OK());
    return SubmitAsync(std::move(ctx), taskId);
}

Status AsuTransportImpl::DeleteAsync(const std::vector<CacheKey>& keys, TaskId& taskId)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->opType = TransportOpType::DELETE;
    ctx->keys = BatchView<CacheKey>{keys.data(), keys.size()};
    ctx->entryStatus.assign(keys.size(), Status::OK());
    return SubmitAsync(std::move(ctx), taskId);
}

Status AsuTransportImpl::Cancel(TaskId taskId)
{
    auto ctx = taskManager_.Get(taskId);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found"); }

    std::lock_guard<std::mutex> lock(ctx->waitMu);
    if (ctx->Done()) { return Status::OK(); }
    ctx->finalStatus = Status::Error(StatusCode::CANCELED, "transport task canceled");
    ReleaseAllSubBatchResources(ctx->subBatchContexts);
    ctx->state.store(TransportTaskState::CANCELED, std::memory_order_release);
    ctx->cv.notify_all();
    return Status::OK();
}

Status AsuTransportImpl::Check(TaskId taskId, TaskResult& result)
{
    auto ctx = taskManager_.Get(taskId);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found"); }

    std::lock_guard<std::mutex> lock(ctx->waitMu);
    BuildResult(*ctx, result);
    if (!ctx->Done()) {
        result.status = Status::Error(StatusCode::IN_PROGRESS, "transport task in progress");
    }
    return Status::OK();
}

Status AsuTransportImpl::Wait(TaskId taskId, std::uint64_t timeoutMs, TaskResult& result)
{
    auto ctx = taskManager_.Get(taskId);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found"); }

    std::unique_lock<std::mutex> lock(ctx->waitMu);
    const bool done = timeoutMs == 0 ? (ctx->cv.wait(lock, [ctx] { return ctx->Done(); }), true)
                                     : ctx->cv.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                                                        [ctx] { return ctx->Done(); });
    BuildResult(*ctx, result);
    if (!done) {
        result.status = Status::Error(StatusCode::TIMEOUT, "transport task wait timeout");
        return result.status;
    }
    lock.unlock();
    taskManager_.Remove(taskId);
    return Status::OK();
}

Status AsuTransportImpl::RegisterRegions(const std::vector<MemoryRegion>& regions,
                                         std::vector<RegisterResult>& results)
{
    results.clear();
    results.reserve(regions.size());

    std::lock_guard<std::mutex> lock(registeredRegionsMu_);
    bool hasFailure = false;
    std::vector<MRHandle> registeredHandles;
    std::vector<TransProvider::UnregisterMemoryDesc> registeredDescs;
    registeredHandles.reserve(regions.size());
    registeredDescs.reserve(regions.size());
    for (const auto& region : regions) {
        const auto memType = region.memoryType == MemoryType::ASCEND_DEVICE
                                 ? TransProvider::MemType::MEM_DEVICE
                                 : TransProvider::MemType::MEM_HOST;
        std::vector<TransProvider::RegisterMemoryDesc> descs{
            {memType, static_cast<std::uintptr_t>(region.addr),
             static_cast<std::size_t>(region.size)}
        };
        std::vector<MRHandle> mrHandles;
        auto status = transProvider_->RegisterMemory(descs, mrHandles);
        if (!status.ok() || mrHandles.empty()) {
            hasFailure = true;
            results.emplace_back(RegisterResult{
                status.ok() ? Status::Error(StatusCode::INTERNAL_ERROR,
                                            "transport register memory returned no handle")
                            : status,
                kInvalidMRHandle});
            break;
        }

        auto handle = mrHandles[0];
        uint32_t tokenId{0};
        status = transProvider_->GetMemTokenId(mrHandles[0], tokenId);
        if (!status.ok()) {
            hasFailure = true;
            (void)transProvider_->UnregisterMemory(
                {TransProvider::UnregisterMemoryDesc{mrHandles[0]}});
            results.emplace_back(RegisterResult{status, kInvalidMRHandle});
            break;
        }

        RegisteredMemory regMem;
        regMem.region = region;
        regMem.handle = handle;
        regMem.tokenId = tokenId;  // Only UB is supported for the current version.
        registeredRegions_[handle] = regMem;
        ownedRegisteredRegionHandles_.insert(handle);
        registeredHandles.emplace_back(handle);
        registeredDescs.push_back(TransProvider::UnregisterMemoryDesc{mrHandles[0]});
        results.emplace_back(RegisterResult{Status::OK(), handle, tokenId});
    }
    if (hasFailure) {
        if (!registeredDescs.empty()) { (void)transProvider_->UnregisterMemory(registeredDescs); }
        for (auto handle : registeredHandles) {
            registeredRegions_.erase(handle);
            ownedRegisteredRegionHandles_.erase(handle);
        }
        return Status::Error(StatusCode::PARTIAL_FAILED,
                             "one or more memory regions failed to register");
    }
    return Status::OK();
}

Status AsuTransportImpl::BindRegisteredRegions(const std::vector<RegisteredMemory>& regions,
                                               std::vector<RegisterResult>& results)
{
    results.clear();
    results.reserve(regions.size());

    std::lock_guard<std::mutex> lock(registeredRegionsMu_);
    for (const auto& region : regions) {
        registeredRegions_[region.handle] = region;
        results.emplace_back(RegisterResult{Status::OK(), region.handle, region.tokenId});
    }
    return Status::OK();
}

Status AsuTransportImpl::UnregisterRegions(const std::vector<MRHandle>& handles)
{
    std::lock_guard<std::mutex> lock(registeredRegionsMu_);
    std::vector<TransProvider::UnregisterMemoryDesc> descs;
    descs.reserve(handles.size());
    for (auto handle : handles) {
        if (handle == kInvalidMRHandle) { continue; }
        if (ownedRegisteredRegionHandles_.find(handle) == ownedRegisteredRegionHandles_.end()) {
            continue;
        }
        descs.push_back(TransProvider::UnregisterMemoryDesc{handle});
    }

    if (!descs.empty()) {
        const auto statuses = transProvider_->UnregisterMemory(descs);
        for (const auto& status : statuses) {
            if (!status.ok()) { return status; }
        }
    }

    for (auto handle : handles) { registeredRegions_.erase(handle); }
    for (auto handle : handles) { ownedRegisteredRegionHandles_.erase(handle); }
    return Status::OK();
}

std::uint16_t AsuTransportImpl::AllocateRequestCid()
{
    auto requestCid = nextRequestCid_.fetch_add(1, std::memory_order_relaxed);
    if (requestCid == 0) { requestCid = nextRequestCid_.fetch_add(1, std::memory_order_relaxed); }
    return requestCid;
}

Status AsuTransportImpl::SubmitAsync(std::unique_ptr<TransportTaskContext> ctx, TaskId& taskId)
{
    if (!worker_.joinable()) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "transport worker is not running");
    }

    auto status = taskManager_.Submit(std::move(ctx), taskId);
    if (!status.ok()) { return status; }

    auto rawCtx = taskManager_.Get(taskId);
    if (!rawCtx) {
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "transport task disappeared after submit");
    }

    std::lock_guard<std::mutex> lock(producerMu_);
    if (!executeQueue_.TryPush(std::move(rawCtx))) {
        taskManager_.Remove(taskId);
        taskId = kInvalidTaskId;
        return Status::Error(StatusCode::RESOURCE_BUSY, "transport task queue is full");
    }
    return Status::OK();
}

void AsuTransportImpl::WorkerLoop()
{
    executeQueue_.ConsumerLoop(stop_, [this](TransportTaskContextPtr ctx) {
        if (!ctx) { return; }
        ProcessTask(ctx);
    });
}

void AsuTransportImpl::CompletionLoop()
{
    while (!stop_.load(std::memory_order_acquire)) {
        for (const auto& ctx : taskManager_.GetAll()) { PollTaskCompletions(ctx); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

Status AsuTransportImpl::AssignSubBatchConnections(
    std::vector<TransportSubBatchContext>& subBatchContexts)
{
    Status status = Status::OK();
    for (auto& subBatchContext : subBatchContexts) {
        if (!subBatchContext.status.ok()) { continue; }

        auto channel = connManager_->SelectConnection();
        if (!channel) {
            const auto subBatchStatus =
                Status::Error(StatusCode::CONNECTION_ERROR, "no available connection channel");
            std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                      subBatchStatus);
            subBatchContext.state = TransportSubBatchState::COMPLETED;
            subBatchContext.status = subBatchStatus;
            if (status.ok()) { status = subBatchStatus; }
            continue;
        }

        subBatchContext.channel = channel;
    }
    return status;
}

void AsuTransportImpl::ProcessTask(const TransportTaskContextPtr& ctx)
{
    TransportTaskState expected = TransportTaskState::PENDING;
    if (!ctx->state.compare_exchange_strong(expected, TransportTaskState::INFLIGHT,
                                            std::memory_order_acq_rel)) {
        if (ctx->state.load(std::memory_order_acquire) == TransportTaskState::CANCELED) {
            ctx->cv.notify_all();
        }
        return;
    }

    std::vector<TransportSubBatchContext> subBatchContexts;
    SubmitTaskRequests(*ctx, subBatchContexts);

    const bool hasSubBatches = !subBatchContexts.empty();
    if (hasSubBatches) {
        AssignSubBatchConnections(subBatchContexts);

        std::vector<TransProvider::SendIoBatch> ioBatches;
        std::vector<std::size_t> subBatchIndexes;
        BuildSubBatchSendBuffers(subBatchContexts, ioBatches, subBatchIndexes);
        SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes);
    }

    std::lock_guard<std::mutex> lock(ctx->waitMu);
    if (ctx->state.load(std::memory_order_acquire) == TransportTaskState::CANCELED) {
        UC_DEBUG("AsuTransportImpl::ProcessTask canceled during process task_id={} sub_batches={}",
                 ctx->taskId, subBatchContexts.size());
        ReleaseAllSubBatchResources(subBatchContexts);
        ctx->cv.notify_all();
        return;
    }

    if (hasSubBatches) { ctx->subBatchContexts = std::move(subBatchContexts); }
    ctx->InitializeTerminalSubBatchCount();
    ctx->TryFinalizeFromSubBatches();
    UC_DEBUG(
        "AsuTransportImpl::ProcessTask submitted task_id={} op_type={} entries={} keys={} "
        "sub_batches={} done={} code={} message={}",
        ctx->taskId, static_cast<int>(ctx->opType), ctx->entries.size, ctx->keys.size,
        ctx->subBatchContexts.size(), ctx->Done(), static_cast<int>(ctx->finalStatus.code),
        ctx->finalStatus.message);

    for (auto& subBatchContext : ctx->subBatchContexts) {
        if (subBatchContext.status.ok()) { continue; }
        ReleaseSubBatchResources(subBatchContext);
    }

    if (ctx->Done()) { ctx->cv.notify_all(); }
}

void AsuTransportImpl::PollTaskCompletions(const TransportTaskContextPtr& ctx)
{
    if (!ctx) { return; }

    std::lock_guard<std::mutex> lock(ctx->waitMu);
    if (ctx->state.load(std::memory_order_acquire) != TransportTaskState::INFLIGHT) { return; }
    if (ctx->subBatchContexts.empty()) { return; }

    for (auto& subBatchContext : ctx->subBatchContexts) {
        if (subBatchContext.state != TransportSubBatchState::PENDING) { continue; }

        auto completeWithError = [this, &ctx, &subBatchContext](const Status& status) {
            std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                      status);
            CompleteSubBatch(*ctx, subBatchContext, status);
        };

        std::uint16_t completedCid = 0;
        const void* responseData = nullptr;
        std::array<std::uint8_t, kFlagBufferHeaderCopySize> flagHeader{};
        std::vector<std::uint8_t> flagBuffer;
        if (subBatchContext.flagBuffer.memory_type == MemoryType::ASCEND_DEVICE) {
            auto status =
                CopyDeviceToHost(subBatchContext.flagBuffer, flagHeader.data(), flagHeader.size());
            if (!status.ok()) {
                // Without a readable header, this sub-batch cannot be polled or unpacked.
                UC_ERROR("Copy flag buffer header from device failed cid={} code={} message={}",
                         subBatchContext.cid, static_cast<int>(status.code), status.message);
                completeWithError(status);
                continue;
            }
            responseData = flagHeader.data();
        } else {
            responseData = reinterpret_cast<void*>(subBatchContext.flagBuffer.local_addr);
        }

        if (const auto status = protocolManager_->PollResponseCid(responseData, completedCid);
            !status.ok()) {
            continue;
        }
        if (completedCid == 0 || completedCid != subBatchContext.cid) { continue; }

        if (subBatchContext.flagBuffer.memory_type == MemoryType::ASCEND_DEVICE) {
            // The header matched; copy the full CQE before unpacking entry status.
            flagBuffer.resize(subBatchContext.flagBuffer.length);
            auto status =
                CopyDeviceToHost(subBatchContext.flagBuffer, flagBuffer.data(), flagBuffer.size());
            if (!status.ok()) {
                // The matched CQE cannot be decoded without the complete flag buffer.
                UC_ERROR("Copy flag buffer from device failed cid={} code={} message={}",
                         subBatchContext.cid, static_cast<int>(status.code), status.message);
                completeWithError(status);
                continue;
            }
            responseData = flagBuffer.data();
        }

        KvResponse response;
        const auto batchNumber = static_cast<std::uint16_t>(subBatchContext.entryStatus.size());
        if (const auto status = protocolManager_->UnpackResponse(
                responseData, ToKvOpcode(subBatchContext.opType), batchNumber, response);
            !status.ok()) {
            completeWithError(status);
            continue;
        }

        subBatchContext.status = KvResponseStatusToSubBatchStatus(response.status);
        FillEntryStatusFromCqeResult(response, subBatchContext);

        const bool queryResultBufferStatus =
            subBatchContext.opType == TransportOpType::QUERY &&
            subBatchContext.status.code == StatusCode::ASU_CQE_CHECK_RESULT_BUFFER;
        const auto status = subBatchContext.status.ok() || queryResultBufferStatus
                                ? Status::OK()
                                : subBatchContext.status;
        if (status.code == StatusCode::ASU_CQE_INTERNAL_ERROR ||
            status.code == StatusCode::ASU_CQE_IO_TIMEOUT) {
            connManager_->ReportFailure(subBatchContext.channel);
        }
        CompleteSubBatch(*ctx, subBatchContext, status);
    }
    ctx->TryFinalizeFromSubBatches();
    if (ctx->Done()) { ctx->cv.notify_all(); }
}

void AsuTransportImpl::BuildResult(const TransportTaskContext& ctx, TaskResult& result)
{
    result.status = ctx.finalStatus;
    result.entryStatus = ctx.entryStatus;
    if (!ctx.subBatchContexts.empty()) {
        std::size_t resultIndex = 0;
        for (const auto& subBatchContext : ctx.subBatchContexts) {
            for (const auto& status : subBatchContext.entryStatus) {
                if (resultIndex >= result.entryStatus.size()) { break; }
                result.entryStatus[resultIndex++] = status;
            }
        }
    }

    result.queryResult.reset();
    if (ctx.opType == TransportOpType::QUERY) {
        result.queryResult = BuildQueryResultFromEntryStatus(result.entryStatus);
    }
}

void AsuTransportImpl::SetTransProvider(std::unique_ptr<TransProvider> provider)
{
    transProvider_ = std::move(provider);
}

std::unique_ptr<AsuTransport> CreateAsuTransport() { return std::make_unique<AsuTransportImpl>(); }

extern "C" std::unique_ptr<AsuTransport> UcmAsuCreateAsuTransport() { return CreateAsuTransport(); }

}  // namespace UC::ASU
