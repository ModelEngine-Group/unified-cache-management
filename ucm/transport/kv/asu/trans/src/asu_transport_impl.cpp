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
#include <memory>
#include <thread>
#include <utility>
#include "aicpu_trans_provider.h"
#include "asu_response_status.h"
#include "asu_transport/asu_transport.h"
#include "connection_internal.h"
#include "connection_manager.h"
#include "logger.h"
#include "transport_config_parser.h"

namespace UC::ASU {

namespace {

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
                transProvider_ = std::make_unique<AICPUTransProvider>();
                break;
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
            (void)connManager_->Shutdown();
            return s;
        }
    }

    connManager_->StartRecoverLoop();

    auto status = ValidateSqeRequestAttrs();
    if (!status.ok()) { return status; }

    status = sendBufferManager_.Init("asu send buffer", MemoryType::HOST,
                                     config_.sendBufferSlotSize, config_.sendBufferSlotNum);
    if (!status.ok()) { return status; }

    status = flagBufferManager_.Init("asu flag buffer", MemoryType::HOST,
                                     config_.flagBufferSlotSize, config_.flagBufferSlotNum);
    if (!status.ok()) { return status; }
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
    taskManager_.CancelAll();

    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
        UC_DEBUG("AsuTransportImpl::Shutdown stopping worker thread");
        worker_.join();
    }
    if (completionWorker_.joinable()) { completionWorker_.join(); }
    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx != nullptr) { (void)taskManager_.Remove(ctx->taskId); }
    }
    if (connManager_) {
        connManager_->Shutdown();
        connManager_.reset();
    }

    UC_DEBUG("AsuTransportImpl::Shutdown OK");
    return Status::OK();
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
    for (const auto& region : regions) {
        auto handle = nextMrHandle_.fetch_add(1, std::memory_order_relaxed);
        if (handle == kInvalidMRHandle) {
            handle = nextMrHandle_.fetch_add(1, std::memory_order_relaxed);
        }
        registeredRegions_[handle] = region;
        results.emplace_back(RegisterResult{Status::OK(), handle});
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
        registeredRegions_[region.handle] = region.region;
        results.emplace_back(RegisterResult{Status::OK(), region.handle});
    }
    return Status::OK();
}

Status AsuTransportImpl::UnregisterRegions(const std::vector<MRHandle>& handles)
{
    std::lock_guard<std::mutex> lock(registeredRegionsMu_);
    for (auto handle : handles) { registeredRegions_.erase(handle); }
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
    Status finalStatus = Status::OK();
    for (auto& subBatchContext : subBatchContexts) {
        if (!subBatchContext.status.ok()) { continue; }

        auto channel = connManager_->SelectConnection();
        if (!channel) {
            const auto status =
                Status::Error(StatusCode::CONNECTION_ERROR, "no available connection channel");
            std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                      status);
            subBatchContext.state = TransportSubBatchState::COMPLETED;
            subBatchContext.status = status;
            if (finalStatus.ok()) { finalStatus = status; }
            continue;
        }

        subBatchContext.channel = channel;
    }
    return finalStatus;
}

void AsuTransportImpl::ProcessTask(const TransportTaskContextPtr& ctx)
{
    UC_DEBUG("AsuTransportImpl::ProcessTask start task_id={} op_type={} entries={} keys={}",
             ctx->taskId, static_cast<int>(ctx->opType), ctx->entries.size, ctx->keys.size);
    TransportTaskState expected = TransportTaskState::PENDING;
    if (!ctx->state.compare_exchange_strong(expected, TransportTaskState::INFLIGHT,
                                            std::memory_order_acq_rel)) {
        UC_DEBUG("AsuTransportImpl::ProcessTask skip task_id={} current_state={}", ctx->taskId,
                 static_cast<int>(ctx->state.load(std::memory_order_acquire)));
        if (ctx->state.load(std::memory_order_acquire) == TransportTaskState::CANCELED) {
            ctx->cv.notify_all();
        }
        return;
    }

    std::vector<TransportSubBatchContext> subBatchContexts;
    auto finalStatus = SubmitTaskRequests(*ctx, subBatchContexts);
    UC_DEBUG(
        "AsuTransportImpl::ProcessTask submit done task_id={} sub_batches={} code={} "
        "message={}",
        ctx->taskId, subBatchContexts.size(), static_cast<int>(finalStatus.code),
        finalStatus.message);
    if (!finalStatus.ok()) {
        UC_ERROR("AsuTransportImpl::ProcessTask submit failed task_id={} code={} message={}",
                 ctx->taskId, static_cast<int>(finalStatus.code), finalStatus.message);
    }

    auto connectionStatus = AssignSubBatchConnections(subBatchContexts);
    UC_DEBUG(
        "AsuTransportImpl::ProcessTask assign connections done task_id={} code={} "
        "message={}",
        ctx->taskId, static_cast<int>(connectionStatus.code), connectionStatus.message);
    if (!connectionStatus.ok()) {
        UC_ERROR(
            "AsuTransportImpl::ProcessTask assign connections failed task_id={} code={} "
            "message={}",
            ctx->taskId, static_cast<int>(connectionStatus.code), connectionStatus.message);
        if (finalStatus.ok()) { finalStatus = connectionStatus; }
    }

    std::vector<TransProvider::SendIoBatch> ioBatches;
    std::vector<std::size_t> subBatchIndexes;
    auto buildStatus = BuildSubBatchSendBuffers(subBatchContexts, ioBatches, subBatchIndexes);
    UC_DEBUG(
        "AsuTransportImpl::ProcessTask build send buffers done task_id={} io_batches={} "
        "send_indexes={} code={} message={}",
        ctx->taskId, ioBatches.size(), subBatchIndexes.size(), static_cast<int>(buildStatus.code),
        buildStatus.message);
    if (!buildStatus.ok()) {
        UC_ERROR(
            "AsuTransportImpl::ProcessTask build send buffers failed task_id={} code={} "
            "message={}",
            ctx->taskId, static_cast<int>(buildStatus.code), buildStatus.message);
        if (finalStatus.ok()) { finalStatus = buildStatus; }
    }

    auto sendStatus = SendSubBatchBuffers(subBatchContexts, ioBatches, subBatchIndexes);
    UC_DEBUG("AsuTransportImpl::ProcessTask send done task_id={} code={} message={}", ctx->taskId,
             static_cast<int>(sendStatus.code), sendStatus.message);
    if (!sendStatus.ok()) {
        UC_ERROR("AsuTransportImpl::ProcessTask send failed task_id={} code={} message={}",
                 ctx->taskId, static_cast<int>(sendStatus.code), sendStatus.message);
        if (finalStatus.ok()) { finalStatus = sendStatus; }
    }

    std::lock_guard<std::mutex> lock(ctx->waitMu);
    if (ctx->state.load(std::memory_order_acquire) == TransportTaskState::CANCELED) {
        UC_DEBUG("AsuTransportImpl::ProcessTask canceled after send task_id={} sub_batches={}",
                 ctx->taskId, subBatchContexts.size());
        ReleaseAllSubBatchResources(subBatchContexts);
        ctx->cv.notify_all();
        return;
    }
    if (!subBatchContexts.empty()) { ctx->subBatchContexts = std::move(subBatchContexts); }
    ctx->finalStatus = finalStatus;
    ctx->InitializeTerminalSubBatchCount();
    ctx->TryFinalizeFromSubBatches();
    UC_DEBUG(
        "AsuTransportImpl::ProcessTask context updated task_id={} stored_sub_batches={} "
        "done={} final_code={} message={}",
        ctx->taskId, ctx->subBatchContexts.size(), ctx->Done(),
        static_cast<int>(ctx->finalStatus.code), ctx->finalStatus.message);

    for (auto& subBatchContext : ctx->subBatchContexts) {
        if (subBatchContext.status.ok()) { continue; }
        UC_DEBUG(
            "AsuTransportImpl::ProcessTask release failed sub-batch task_id={} cid={} "
            "code={} message={}",
            ctx->taskId, subBatchContext.cid, static_cast<int>(subBatchContext.status.code),
            subBatchContext.status.message);
        ReleaseSubBatchResources(subBatchContext);
    }
    if (ctx->Done()) {
        UC_DEBUG("AsuTransportImpl::ProcessTask notify done task_id={}", ctx->taskId);
        ctx->cv.notify_all();
    }
}

void AsuTransportImpl::PollTaskCompletions(const TransportTaskContextPtr& ctx)
{
    if (!ctx || ctx->state.load(std::memory_order_acquire) != TransportTaskState::INFLIGHT) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->waitMu);
    if (ctx->subBatchContexts.empty()) { return; }

    for (auto& subBatchContext : ctx->subBatchContexts) {
        if (subBatchContext.state != TransportSubBatchState::PENDING) { continue; }

        std::uint16_t completedCid = 0;
        auto cidStatus = protocolManager_->PollResponseCid(
            reinterpret_cast<void*>(subBatchContext.flagBuffer.addr), completedCid);
        if (!cidStatus.ok()) { continue; }
        if (completedCid == 0 || completedCid != subBatchContext.cid) { continue; }

        KvResponse response;
        const auto batchNumber = static_cast<std::uint16_t>(subBatchContext.entryStatus.size());
        const auto unpackStatus = protocolManager_->UnpackResponse(
            reinterpret_cast<void*>(subBatchContext.flagBuffer.addr),
            ToKvOpcode(subBatchContext.opType), batchNumber, response);
        if (!unpackStatus.ok()) {
            std::fill(subBatchContext.entryStatus.begin(), subBatchContext.entryStatus.end(),
                      unpackStatus);
            CompleteSubBatch(*ctx, subBatchContext, unpackStatus);
            continue;
        }

        subBatchContext.status = KvResponseStatusToSubBatchStatus(response.status);
        FillEntryStatusFromCqeResult(response, subBatchContext);

        if (subBatchContext.status.ok()) {
            CompleteSubBatch(*ctx, subBatchContext, Status::OK());
            continue;
        }

        if (subBatchContext.status.code == StatusCode::ASU_CQE_INTERNAL_ERROR ||
            subBatchContext.status.code == StatusCode::ASU_CQE_IO_TIMEOUT) {
            connManager_->ReportFailure(subBatchContext.channel);
        }
        CompleteSubBatch(*ctx, subBatchContext, subBatchContext.status);
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

}  // namespace UC::ASU
