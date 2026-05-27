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
#include "aicpu_trans_provider.h"
#include "asu_transport/asu_transport.h"
#include "asu_transport/types.h"
#include "completion_poller.h"
#include "connection_internal.h"
#include "logger.h"
#include "transport_config_parser.h"

namespace UC::ASU {

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

    transProvider_ = std::make_unique<AICPUTransProvider>();

    connManager_ = std::make_unique<ConnectionManager>(
        [this](const AsuEndpoint& ep, std::uint32_t num) -> std::vector<ConnectionHandle> {
            std::string localIp;
            auto it = config_.attrs.find("localIp");
            if (it != config_.attrs.end()) { localIp = it->second; }

            std::uint32_t timeout = 5000;
            auto tit = config_.attrs.find("timeout");
            if (tit != config_.attrs.end()) {
                timeout = static_cast<std::uint32_t>(std::stoul(tit->second));
            }

            std::vector<TransProvider::ConnectionHandle> handles;
            auto status = transProvider_->CreateConnection(
                localIp, ep.ip, ep.port, num, timeout, handles);

            if (!status.ok()) {
                UC_ERROR("CreateConnection failed: {}", status.message);
                return {};
            }

            std::vector<ConnectionHandle> result;
            result.reserve(handles.size());
            for (auto& handle : handles) {
                result.push_back(handle);
            }
            return result;
        });

    std::uint32_t qp_num = config_.queryQpNum + config_.loadQpNum + config_.storeQpNum;
    UC_DEBUG("AsuTransportImpl::Init endpoints={} qp_num={}", config_.endpoints.size(), qp_num);
    for (const auto& ep : config_.endpoints) {
        auto s = connManager_->AddGroup(ep, qp_num);
        if (!s.ok()) {
            UC_DEBUG("AsuTransportImpl::Init AddGroup FAILED: {}", s.message);
            return s;
        }
    }

    connManager_->StartRecoverLoop();

    // 初始化 flagBuffer 内存池
    size_t poolSize = config_.maxInflightTasks;
    flagBufferPool_.resize(poolSize, 0);
    freeFlagSlots_.reserve(poolSize);
    for (size_t i = 0; i < poolSize; ++i) {
        freeFlagSlots_.push_back(i);
    }

    // 注册 flagBuffer 内存到 NPU（使用第一个连接）
    if (!config_.endpoints.empty()) {
        auto firstConn = connManager_->SelectConnection();
        if (firstConn) {
            TransProvider::RegisterMemoryDesc memDesc;
            memDesc.memoryType = TransProvider::MemType::MEM_HOST;
            memDesc.addr = reinterpret_cast<uintptr_t>(flagBufferPool_.data());
            memDesc.size = poolSize * sizeof(uint32_t);

            std::vector<TransProvider::MemHandle> memHandles;
            auto regStatus = transProvider_->RegisterMemory(
                firstConn->GetLink(),
                {memDesc},
                memHandles);

            if (regStatus.ok() && !memHandles.empty()) {
                flagBufferMemHandle_ = memHandles[0];
                UC_DEBUG("AsuTransportImpl::Init flagBuffer registered, size={}", poolSize);
            } else {
                UC_WARN("AsuTransportImpl::Init flagBuffer registration failed: {}", regStatus.message);
            }
            firstConn->ReleaseInflight();
        }
    }

    auto pollerCapacity = std::max<std::size_t>(
        static_cast<std::size_t>(config_.maxInflightTasks), std::size_t{256});
    completionPoller_.Start(pollerCapacity,
        [this](const std::shared_ptr<ConnectionChannel>& ch) {
            connManager_->ReportFailure(ch);
        },
        [this](volatile uint32_t* flagBuffer) {
            // 计算 slot 索引
            size_t slot = flagBuffer - flagBufferPool_.data();
            std::lock_guard<std::mutex> lock(flagBufferMu_);
            freeFlagSlots_.push_back(slot);
        });

    auto queueDepth = std::max<std::size_t>(2, static_cast<std::size_t>(config_.maxInflightTasks));
    executeQueue_.Setup(queueDepth + 1);
    stop_.store(false, std::memory_order_release);
    worker_ = std::thread(&AsuTransportImpl::WorkerLoop, this);
    UC_DEBUG("AsuTransportImpl::Init OK: queueDepth={}", queueDepth);
    return Status::OK();
}

Status AsuTransportImpl::Shutdown()
{
    if (!worker_.joinable()) { return Status::OK(); }

    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx == nullptr || ctx->Done()) { continue; }
        std::lock_guard<std::mutex> lock(ctx->waitMu);
        ctx->finalStatus =
            Status::Error(StatusCode::CANCELED, "transport task canceled by shutdown");
        ctx->state.store(TransportTaskState::CANCELED, std::memory_order_release);
        ctx->cv.notify_all();
    }

    stop_.store(true, std::memory_order_release);
    UC_DEBUG("AsuTransportImpl::Shutdown stopping worker thread");
    if (worker_.joinable()) { worker_.join(); }
    completionPoller_.Stop();
    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx != nullptr) { (void)taskManager_.Remove(ctx->taskId); }
    }
    if (connManager_) {
        connManager_->Shutdown();
        connManager_.reset();
    }

    // 释放 flagBuffer 内存
    if (flagBufferMemHandle_ && transProvider_ && !config_.endpoints.empty()) {
        auto firstConn = connManager_ ? connManager_->SelectConnection() : nullptr;
        if (firstConn) {
            TransProvider::UnregisterMemoryDesc unregDesc;
            unregDesc.connectionHandle = firstConn->GetLink();
            unregDesc.memoryHandle = flagBufferMemHandle_;
            transProvider_->UnregisterMemory({unregDesc});
            firstConn->ReleaseInflight();
        }
        flagBufferMemHandle_ = nullptr;
    }
    flagBufferPool_.clear();
    freeFlagSlots_.clear();

    UC_DEBUG("AsuTransportImpl::Shutdown OK");
    return Status::OK();
}

Status AsuTransportImpl::CheckHealth()
{
    if (!worker_.joinable()) {
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
    ctx->opType = TransportOpType::LOAD;
    ctx->entries = BatchView<KVBuffer>{entries.data(), entries.size()};
    ctx->entryStatus.assign(entries.size(), Status::OK());
    return SubmitAsync(std::move(ctx), taskId);
}

Status AsuTransportImpl::StoreAsync(const std::vector<KVBuffer>& entries, TaskId& taskId)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->opType = TransportOpType::STORE;
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

Status AsuTransportImpl::SubmitAsync(std::unique_ptr<TransportTaskContext> ctx, TaskId& taskId)
{
    // 分配 flagBuffer slot
    size_t flagSlot = SIZE_MAX;
    {
        std::lock_guard<std::mutex> lock(flagBufferMu_);
        if (freeFlagSlots_.empty()) {
            return Status::Error(StatusCode::RESOURCE_BUSY, "no available flagBuffer slot");
        }
        flagSlot = freeFlagSlots_.back();
        freeFlagSlots_.pop_back();
    }

    // 初始化 flagBuffer
    flagBufferPool_[flagSlot] = 0;
    ctx->flagBuffer = &flagBufferPool_[flagSlot];

    auto status = taskManager_.Submit(std::move(ctx), taskId);
    if (!status.ok()) {
        // 释放 flagBuffer slot
        std::lock_guard<std::mutex> lock(flagBufferMu_);
        freeFlagSlots_.push_back(flagSlot);
        return status;
    }

    auto rawCtx = taskManager_.Get(taskId);
    if (!rawCtx) {
        taskId = kInvalidTaskId;
        // 释放 flagBuffer slot
        std::lock_guard<std::mutex> lock(flagBufferMu_);
        freeFlagSlots_.push_back(flagSlot);
        return Status::Error(StatusCode::INTERNAL_ERROR, "transport task disappeared after submit");
    }

    std::lock_guard<std::mutex> lock(producerMu_);
    if (!executeQueue_.TryPush(std::move(rawCtx))) {
        taskManager_.Remove(taskId);
        taskId = kInvalidTaskId;
        // 释放 flagBuffer slot
        std::lock_guard<std::mutex> flagLock(flagBufferMu_);
        freeFlagSlots_.push_back(flagSlot);
        return Status::Error(StatusCode::RESOURCE_BUSY, "transport task queue is full");
    }
    UC_DEBUG("AsuTransportImpl::SubmitAsync OK: taskId={}, flagSlot={}", taskId, flagSlot);
    return Status::OK();
}

void AsuTransportImpl::WorkerLoop()
{
    executeQueue_.ConsumerLoop(stop_, [this](TransportTaskContextPtr ctx) {
        if (!ctx) { return; }
        CompleteTask(ctx);
    });
    UC_DEBUG("AsuTransportImpl::WorkerLoop stopped");
}

void AsuTransportImpl::CompleteTask(const TransportTaskContextPtr& ctx)
{
    static constexpr int kMaxRetryAttempts = 2;
    int retries = kMaxRetryAttempts;
    auto channel = connManager_->SelectConnection();

    while (retries-- > 0 && channel) {
        if (ctx->state.load(std::memory_order_acquire) != TransportTaskState::PENDING) {
            channel->ReleaseInflight();
            return;
        }

        // Send via TransProvider
        auto handle = channel->GetLink();
        if (!handle || !ctx->flagBuffer) {
            channel->ReleaseInflight();
            connManager_->ReportFailure(channel);
            channel = connManager_->SelectConnection();
            continue;
        }

        *ctx->flagBuffer = 0;

        TransProvider::SendIoBatch batch;
        batch.connectionHandle = handle;
        batch.sendBuffer = nullptr;
        batch.flagBuffer = const_cast<uint32_t*>(ctx->flagBuffer);

        auto results = transProvider_->Send({batch}, 0, 0);
        if (results.empty() || !results[0].ok()) {
            channel->ReleaseInflight();
            connManager_->ReportFailure(channel);
            channel = connManager_->SelectConnection();
            continue;
        }

        TransportTaskState expected = TransportTaskState::PENDING;
        if (!ctx->state.compare_exchange_strong(expected, TransportTaskState::INFLIGHT,
                                                std::memory_order_acq_rel)) {
            channel->ReleaseInflight();
            return;
        }

        std::uint64_t timeoutMs = GetTimeoutMs(ctx->opType);

        auto deadlineMs = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count()) +
            timeoutMs;

        PendingRequest req;
        req.ctx = ctx;
        req.channel = channel;
        req.flagBuffer = ctx->flagBuffer;
        req.deadlineMs = deadlineMs;
        completionPoller_.SubmitPending(std::move(req));
        return;
    }

    if (channel) { channel->ReleaseInflight(); }

    // When failure occurs, the waiting thread will be awakened; if successful, it will be placed in the pending queue
    std::lock_guard<std::mutex> lock(ctx->waitMu);
    ctx->finalStatus = Status::Error(StatusCode::NO_ACTIVE_CONNECTION, "no available channel");
    ctx->state.store(TransportTaskState::FAILED, std::memory_order_release);
    ctx->cv.notify_all();
}

void AsuTransportImpl::BuildResult(const TransportTaskContext& ctx, TaskResult& result)
{
    result.status = ctx.finalStatus;
    result.entryStatus = ctx.entryStatus;
    result.queryResult.reset();
    if (ctx.opType == TransportOpType::QUERY) { result.queryResult = ctx.queryResult; }
}

std::uint64_t AsuTransportImpl::GetTimeoutMs(TransportOpType opType) const
{
    switch (opType) {
    case TransportOpType::QUERY:
        return config_.queryTimeoutMs;
    case TransportOpType::LOAD:
        return config_.loadTimeoutMs;
    case TransportOpType::STORE:
    case TransportOpType::BATCH_STORE:
    case TransportOpType::DELETE:
    case TransportOpType::KEEP_ALIVE:
    default:
        return config_.storeTimeoutMs;
    }
}

std::unique_ptr<AsuTransport> CreateAsuTransport() { return std::make_unique<AsuTransportImpl>(); }

}  // namespace UC::ASU