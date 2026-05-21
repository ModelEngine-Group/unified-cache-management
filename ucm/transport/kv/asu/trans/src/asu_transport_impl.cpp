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

namespace UC::ASU {

AsuTransportImpl::~AsuTransportImpl() { Shutdown(); }

Status AsuTransportImpl::Init(const TransportConfig& config)
{
    if (worker_.joinable()) { return Status::OK(); }

    config_ = config;
    auto queue_depth =
        std::max<std::size_t>(2, static_cast<std::size_t>(config_.max_inflight_tasks));
    execute_queue_.Setup(queue_depth + 1);
    stop_.store(false, std::memory_order_release);
    worker_ = std::thread(&AsuTransportImpl::WorkerLoop, this);
    return Status::OK();
}

Status AsuTransportImpl::Shutdown()
{
    if (!worker_.joinable()) { return Status::OK(); }

    // TODO: Drain task queue and fail all pending tasks

    stop_.store(true, std::memory_order_release);
    if (worker_.joinable()) { worker_.join(); }
    return Status::OK();
}

Status AsuTransportImpl::CheckHealth()
{
    // TODO: real health check
    return Status::OK();
}

Status AsuTransportImpl::Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                               QueryResult& result)
{
    TaskId task_id{kInvalidTaskId};
    auto status = QueryAsync(keys, options, task_id);
    if (!status.ok()) { return status; }

    TaskResult task_result;
    const auto timeout_ms = options.timeout_ms == 0 ? config_.query_timeout_ms : options.timeout_ms;
    status = Wait(task_id, timeout_ms, task_result);
    if (!status.ok()) { return status; }
    if (task_result.query_result.has_value()) { result = *task_result.query_result; }
    return task_result.status;
}

Status AsuTransportImpl::QueryAsync(const std::vector<CacheKey>& keys, const QueryOptions& options,
                                    TaskId& task_id)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->op_type = TransportOpType::QUERY;
    ctx->keys = BatchView<CacheKey>{keys.data(), keys.size()};
    ctx->query_options = options;
    ctx->entry_status.assign(keys.size(), Status::OK());
    return SubmitAsync(std::move(ctx), task_id);
}

Status AsuTransportImpl::LoadAsync(const std::vector<KVBuffer>& entries, TaskId& task_id)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->op_type = TransportOpType::LOAD;
    ctx->entries = BatchView<KVBuffer>{entries.data(), entries.size()};
    ctx->entry_status.assign(entries.size(), Status::OK());
    return SubmitAsync(std::move(ctx), task_id);
}

Status AsuTransportImpl::StoreAsync(const std::vector<KVBuffer>& entries, TaskId& task_id)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->op_type = TransportOpType::STORE;
    ctx->entries = BatchView<KVBuffer>{entries.data(), entries.size()};
    ctx->entry_status.assign(entries.size(), Status::OK());
    return SubmitAsync(std::move(ctx), task_id);
}

Status AsuTransportImpl::DeleteAsync(const std::vector<CacheKey>& keys, TaskId& task_id)
{
    auto ctx = std::make_unique<TransportTaskContext>();
    ctx->op_type = TransportOpType::DELETE;
    ctx->keys = BatchView<CacheKey>{keys.data(), keys.size()};
    ctx->entry_status.assign(keys.size(), Status::OK());
    return SubmitAsync(std::move(ctx), task_id);
}

Status AsuTransportImpl::Cancel(TaskId task_id)
{
    return Status::Error(StatusCode::INTERNAL_ERROR, "cancel is not supported now");
}

Status AsuTransportImpl::Check(TaskId task_id, TaskResult& result)
{
    auto ctx = task_manager_.Get(task_id);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found"); }

    std::lock_guard<std::mutex> lock(ctx->wait_mu);
    BuildResult(*ctx, result);
    if (!ctx->Done()) {
        result.status = Status::Error(StatusCode::IN_PROGRESS, "transport task in progress");
    }
    return Status::OK();
}

Status AsuTransportImpl::Wait(TaskId task_id, std::uint64_t timeout_ms, TaskResult& result)
{
    auto ctx = task_manager_.Get(task_id);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "transport task not found"); }

    std::unique_lock<std::mutex> lock(ctx->wait_mu);
    const bool done = timeout_ms == 0
                          ? (ctx->cv.wait(lock, [ctx] { return ctx->Done(); }), true)
                          : ctx->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                             [ctx] { return ctx->Done(); });
    BuildResult(*ctx, result);
    if (!done) {
        result.status = Status::Error(StatusCode::TIMEOUT, "transport task wait timeout");
        return result.status;
    }
    lock.unlock();
    task_manager_.Remove(task_id);
    return Status::OK();
}

Status AsuTransportImpl::RegisterRegions(const std::vector<MemoryRegion>& regions,
                                         std::vector<RegisterResult>& results)
{
    results.clear();
    results.assign(regions.size(), RegisterResult{Status::OK(), kInvalidMRHandle});
    // TODO:
    return Status::OK();
}

Status AsuTransportImpl::BindRegisteredRegions(const std::vector<RegisteredMemory>& regions,
                                               std::vector<RegisterResult>& results)
{
    results.clear();
    results.assign(regions.size(), RegisterResult{Status::OK(), kInvalidMRHandle});
    // TODO:
    return Status::OK();
}

Status AsuTransportImpl::UnregisterRegions(const std::vector<MRHandle>& handles)
{
    // TODO:
    return Status::OK();
}

Status AsuTransportImpl::SubmitAsync(std::unique_ptr<TransportTaskContext> ctx, TaskId& task_id)
{
    auto status = task_manager_.Submit(std::move(ctx), task_id);
    if (!status.ok()) { return status; }

    auto raw_ctx = task_manager_.Get(task_id);
    if (!raw_ctx) {
        task_id = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "transport task disappeared after submit");
    }

    std::lock_guard<std::mutex> lock(producer_mu_);
    if (!execute_queue_.TryPush(std::move(raw_ctx))) {
        task_manager_.Remove(task_id);
        task_id = kInvalidTaskId;
        return Status::Error(StatusCode::RESOURCE_BUSY, "transport task queue is full");
    }
    return Status::OK();
}

void AsuTransportImpl::WorkerLoop()
{
    execute_queue_.ConsumerLoop(stop_, [this](TransportTaskContextPtr ctx) {
        if (!ctx) { return; }
        CompleteTask(ctx);
    });
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

    std::lock_guard<std::mutex> lock(ctx->wait_mu);
    if (ctx->op_type == TransportOpType::QUERY) {
        ctx->query_result.exists.assign(ctx->keys.size, 0);
        ctx->query_result.prefix_hit_keys = 0;
    }
    ctx->final_status = Status::OK();
    ctx->state.store(TransportTaskState::COMPLETED, std::memory_order_release);
    ctx->cv.notify_all();
}

void AsuTransportImpl::BuildResult(const TransportTaskContext& ctx, TaskResult& result)
{
    result.status = ctx.final_status;
    result.entry_status = ctx.entry_status;
    result.query_result.reset();
    if (ctx.op_type == TransportOpType::QUERY) { result.query_result = ctx.query_result; }
}

std::unique_ptr<AsuTransport> CreateAsuTransport() { return std::make_unique<AsuTransportImpl>(); }

}  // namespace UC::ASU
