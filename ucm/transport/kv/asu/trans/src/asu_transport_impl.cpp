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

    connManager_.PrepareForInit();

    std::uint32_t qp_num = config_.queryQpNum + config_.loadQpNum + config_.storeQpNum;
    UC_DEBUG("AsuTransportImpl::Init endpoints={} qp_num={}", config_.endpoints.size(), qp_num);
    for (const auto& ep : config_.endpoints) {
        auto s = connManager_.AddGroup(ep, qp_num);
        if (!s.ok()) {
            UC_DEBUG("AsuTransportImpl::Init AddGroup FAILED: {}", s.message);
            (void)connManager_.Shutdown();
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
    taskManager_.CancelAll();

    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
        UC_DEBUG("AsuTransportImpl::Shutdown stopping worker thread");
        worker_.join();
    }
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
    UC_DEBUG("AsuTransportImpl::SubmitAsync OK: taskId={}", taskId);
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

void AsuTransportImpl::BuildResult(const TransportTaskContext& ctx, TaskResult& result)
{
    result.status = ctx.finalStatus;
    result.entryStatus = ctx.entryStatus;
    result.queryResult.reset();
    if (ctx.opType == TransportOpType::QUERY) { result.queryResult = ctx.queryResult; }
}

std::unique_ptr<AsuTransport> CreateAsuTransport() { return std::make_unique<AsuTransportImpl>(); }

}  // namespace UC::ASU
