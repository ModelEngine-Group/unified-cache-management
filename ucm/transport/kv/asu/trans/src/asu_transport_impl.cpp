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
#include "asu_transport/asu_transport.h"
#include "asu_transport/types.h"
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

    connManager_.SetConnectionOps(
        [this](const AsuEndpoint& ep, std::uint32_t num) { return StubCreateConnection(ep, num); },
        [this](const std::vector<ConnectionHandle>& handles) {
            return StubDeleteConnections(handles);
        });

    std::uint32_t qp_num = config_.queryQpNum + config_.loadQpNum + config_.storeQpNum;
    UC_DEBUG("AsuTransportImpl::Init endpoints={} qp_num={}", config_.endpoints.size(), qp_num);
    for (const auto& ep : config_.endpoints) {
        auto s = connManager_.AddGroup(ep, qp_num);
        if (!s.ok()) {
            UC_DEBUG("AsuTransportImpl::Init AddGroup FAILED: {}", s.message);
            return s;
        }
    }

    connManager_.StartRecoverLoop();

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
    for (const auto& ctx : taskManager_.GetAll()) {
        if (ctx != nullptr) { (void)taskManager_.Remove(ctx->taskId); }
    }
    connManager_.Shutdown();
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

Status AsuTransportImpl::StubCheck(
    TaskId task_id, TaskResult& result)  // Stub for testing, remove after real implementation
{
    auto ctx = taskManager_.Get(task_id);
    if (!ctx) {
        UC_DEBUG("AsuTransportImpl::StubCheck task_id={} NOT FOUND", task_id);
        return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found");
    }
    std::unique_lock<std::mutex> lock(ctx->waitMu);
    if (!ctx->StubDone()) {
        UC_DEBUG("AsuTransportImpl::StubCheck task_id={} IN_PROGRESS", task_id);
        result.status = Status::Error(StatusCode::IN_PROGRESS, "transport task in progress");
        return Status::OK();
    }
    UC_DEBUG("AsuTransportImpl::StubCheck task_id={} DONE", task_id);
    BuildResult(*ctx, result);
    return Status::OK();
}

Status AsuTransportImpl::StubWait(
    TaskId task_id, std::uint64_t timeout_ms,
    TaskResult& result)  // Stub for testing, remove after real implementation
{
    UC_DEBUG("AsuTransportImpl::StubWait task_id={} timeout_ms={}", task_id, timeout_ms);
    auto ctx = taskManager_.Get(task_id);
    if (!ctx) {
        UC_DEBUG("AsuTransportImpl::StubWait task_id={} NOT FOUND", task_id);
        return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found");
    }

    std::unique_lock<std::mutex> lock(ctx->waitMu);
    const bool done = timeout_ms == 0
                          ? (ctx->cv.wait(lock, [ctx] { return ctx->StubDone(); }), true)
                          : ctx->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                             [ctx] { return ctx->StubDone(); });
    if (!done) {
        UC_DEBUG("AsuTransportImpl::StubWait task_id={} TIMEOUT", task_id);
        auto prev_state =
            ctx->state.exchange(TransportTaskState::FAILED, std::memory_order_acq_rel);
        if (prev_state == TransportTaskState::INFLIGHT) {
            auto* channel = ctx->channel.load(std::memory_order_acquire);
            if (channel) {
                UC_DEBUG(
                    "AsuTransportImpl::StubWait timeout: prev_state=INFLIGHT, inflight-1 on "
                    "ch_id={}",
                    channel->GetChannelId());
                channel->ReleaseInflight();
            }
            BuildResult(*ctx, result);
            result.status =
                Status::Error(StatusCode::RESULT_TIMEOUT, "transport task result timeout");
            lock.unlock();
            taskManager_.Remove(task_id);
            return result.status;
        }
        UC_DEBUG(
            "AsuTransportImpl::StubWait timeout: prev_state=PENDING, submit timeout (CompleteTask "
            "will CAS undo)");
        BuildResult(*ctx, result);
        result.status = Status::Error(StatusCode::SUBMIT_TIMEOUT, "transport task submit timeout");
        lock.unlock();
        taskManager_.Remove(task_id);
        return result.status;
    }
    UC_DEBUG("AsuTransportImpl::StubWait task_id={} DONE", task_id);
    BuildResult(*ctx, result);
    lock.unlock();
    taskManager_.Remove(task_id);
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
    UC_DEBUG("AsuTransportImpl::SubmitAsync OK: taskId={}", taskId);
    return Status::OK();
}

void AsuTransportImpl::WorkerLoop()
{
    executeQueue_.ConsumerLoop(stop_, [this](TransportTaskContextPtr ctx) {
        if (!ctx) { return; }
#ifdef ASU_BUILD_TESTS
        if (useStubCompleteTask_.load(std::memory_order_acquire)) {
            StubCompleteTask(ctx);
            return;
        }
#endif
        CompleteTask(ctx);
    });
    UC_DEBUG("AsuTransportImpl::WorkerLoop stopped");
}

void AsuTransportImpl::CompleteTask(const TransportTaskContextPtr& ctx)
{
    // TODO: do REAL work here
    TransportTaskState expected = TransportTaskState::PENDING;
    if (!ctx->state.compare_exchange_strong(expected, TransportTaskState::INFLIGHT,
                                            std::memory_order_acq_rel)) {
        if (ctx->state.load(std::memory_order_acquire) == TransportTaskState::CANCELED) {
            ctx->cv.notify_all();
        }
        return;
    }

    std::lock_guard<std::mutex> lock(ctx->waitMu);
    if (ctx->state.load(std::memory_order_acquire) == TransportTaskState::CANCELED) {
        ctx->cv.notify_all();
        return;
    }
    if (ctx->state.load(std::memory_order_acquire) == TransportTaskState::CANCELED) {
        ctx->cv.notify_all();
        return;
    }
    if (ctx->opType == TransportOpType::QUERY) {
        ctx->queryResult.exists.assign(ctx->keys.size, 0);
        ctx->queryResult.prefixHitKeys = 0;
    }
    ctx->finalStatus = Status::OK();
    ctx->state.store(TransportTaskState::COMPLETED, std::memory_order_release);
    ctx->cv.notify_all();
}

void AsuTransportImpl::StubCompleteTask(const TransportTaskContextPtr& ctx)
{
    static constexpr int kMaxRetryAttempts = 2;
    int retries = kMaxRetryAttempts;
    ConnectionChannel* channel = connManager_.SelectConnection();
    Status s;

    while (retries-- > 0 && channel) {
        if (ctx->state.load(std::memory_order_acquire) != TransportTaskState::PENDING) {
            UC_DEBUG(
                "AsuTransportImpl::CompleteTask task_id={} state!=PENDING (Wait timeout/Cancel), "
                "undo inflight on ch_id={}",
                ctx->taskId, channel->GetChannelId());
            channel->ReleaseInflight();
            return;
        }
        ctx->channel.store(channel, std::memory_order_release);
        s = StubSend(channel, ctx.get());
        if (s.ok()) {
            TransportTaskState expected = TransportTaskState::PENDING;
            if (!ctx->state.compare_exchange_strong(expected, TransportTaskState::INFLIGHT,
                                                    std::memory_order_acq_rel)) {
                UC_DEBUG(
                    "AsuTransportImpl::CompleteTask task_id={} CAS PENDING->INFLIGHT failed "
                    "(state={}), undo inflight on ch_id={}",
                    ctx->taskId, static_cast<int>(expected), channel->GetChannelId());
                channel->ReleaseInflight();
                return;
            }
            UC_DEBUG("AsuTransportImpl::CompleteTask task_id={} Send OK + INFLIGHT on ch_id={}",
                     ctx->taskId, channel->GetChannelId());
            ctx->finalStatus = Status::OK();
            ctx->cv.notify_all();
            return;
        }
        UC_DEBUG(
            "AsuTransportImpl::CompleteTask task_id={} Send FAILED on ch_id={} retries_left={}",
            ctx->taskId, channel->GetChannelId(), retries);
        channel->ReleaseInflight();
        connManager_.ReportFailure(channel);
        channel = connManager_.SelectConnection();
    }

    if (channel) { channel->ReleaseInflight(); }

    UC_DEBUG("AsuTransportImpl::CompleteTask task_id={} no available channel, state->FAILED",
             ctx->taskId);
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

Status AsuTransportImpl::StubSend(ConnectionChannel* channel, TransportTaskContext* ctx)
{
    UC_DEBUG("AsuTransportImpl::StubSend ch_id={} state={}", channel->GetChannelId(),
             static_cast<int>(channel->GetState()));
    ctx->flagbufferStatus.store(1, std::memory_order_release);
    UC_DEBUG("AsuTransportImpl::StubSend ch_id={} stub: flagbuffer->1, notify_all",
             channel->GetChannelId());
    return Status::OK();
}

std::vector<ConnectionHandle> AsuTransportImpl::StubCreateConnection(const AsuEndpoint& endpoint,
                                                                     std::uint32_t qp_num)
{
    (void)endpoint;
    return std::vector<ConnectionHandle>(qp_num, nullptr);
}

std::vector<Status> AsuTransportImpl::StubDeleteConnections(
    const std::vector<ConnectionHandle>& handles)
{
    return std::vector<Status>(handles.size(), Status::OK());
}

std::unique_ptr<AsuTransport> CreateAsuTransport() { return std::make_unique<AsuTransportImpl>(); }

}  // namespace UC::ASU