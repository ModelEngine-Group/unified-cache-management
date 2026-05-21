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
#include "asu_client_impl.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <thread>
#include "asu_transport/types.h"

namespace UC::ASU {

AsuClientImpl::AsuClientImpl(TransportFactory factory) : transport_factory_(std::move(factory))
{
    if (!transport_factory_) { transport_factory_ = CreateAsuTransport; }
}

AsuClientImpl::~AsuClientImpl() { Shutdown(); }

AsuId AsuClientImpl::Router::Pick(const CacheKey& key) const
{
    if (asu_ids.empty()) { return 0; }
    const auto index = std::hash<CacheKey>{}(key) % asu_ids.size();
    return asu_ids[index];
}

Status AsuClientImpl::Init(const AsuClientConfig& config)
{
    if (view_) { return Status::OK(); }

    auto view = std::make_shared<ViewSnapshot>();
    view->router = std::make_shared<Router>();
    AsuId next_generated_asu_id = 1;
    for (const auto& transport_config : config.transport_configs) {
        auto asu_id = transport_config.asu_id;
        if (asu_id == 0) {
            while (view->transports.find(next_generated_asu_id) != view->transports.end()) {
                ++next_generated_asu_id;
            }
            asu_id = next_generated_asu_id++;
        } else if (view->transports.find(asu_id) != view->transports.end()) {
            return Status::Error(StatusCode::INVALID_ARGUMENT, "duplicate ASU transport id");
        }

        auto transport = transport_factory_();
        if (!transport) {
            return Status::Error(StatusCode::INTERNAL_ERROR, "transport factory returned null");
        }

        auto status = transport->Init(transport_config);
        if (!status.ok()) { return status; }

        view->router->asu_ids.push_back(asu_id);
        view->transports.emplace(asu_id, std::shared_ptr<AsuTransport>(std::move(transport)));
    }
    config_ = config;
    view_ = std::move(view);
    return Status::OK();
}

Status AsuClientImpl::Shutdown()
{
    auto view = view_;
    if (view) {
        for (auto& pair : view->transports) { pair.second->Shutdown(); }
    }
    view_.reset();
    return Status::OK();
}

Status AsuClientImpl::Query(const std::vector<CacheKey>& keys, const QueryOptions& options,
                            QueryResult& result)
{
    auto view = view_;
    if (!view || !view->router || view->transports.empty()) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    result.exists.assign(keys.size(), 0);
    result.prefix_hit_keys = 0;

    return Status::OK();
}

Status AsuClientImpl::LoadAsync(const std::vector<KVBuffer>& entries, TaskId& task_id)
{
    return SubmitAsync(ClientOpType::LOAD, entries, task_id);
}

Status AsuClientImpl::StoreAsync(const std::vector<KVBuffer>& entries, TaskId& task_id)
{
    return SubmitAsync(ClientOpType::STORE, entries, task_id);
}

Status AsuClientImpl::DeleteAsync(const std::vector<CacheKey>& keys, TaskId& task_id)
{
    (void)keys;
    task_id = kInvalidTaskId;
    return Status::Error(StatusCode::UNSUPPORTED, "client delete async is not supported now");
}

Status AsuClientImpl::Check(TaskId task_id, TaskResult& result)
{
    auto ctx = task_manager_.Get(task_id);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "client task not found"); }

    PollTask(ctx);
    return BuildResult(ctx, result);
}

Status AsuClientImpl::Wait(TaskId task_id, std::uint64_t timeout_ms, TaskResult& result)
{
    auto ctx = task_manager_.Get(task_id);
    if (!ctx) { return Status::Error(StatusCode::TASK_NOT_FOUND, "client task not found"); }

    const auto wait_ms = timeout_ms == 0 ? config_.default_wait_timeout_ms : timeout_ms;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms);

    while (!ctx->Done()) {
        if (PollTask(ctx)) { break; }
        if (wait_ms != 0 && std::chrono::steady_clock::now() >= deadline) {
            BuildResult(ctx, result);
            result.status = Status::Error(StatusCode::TIMEOUT, "client task wait timeout");
            return result.status;
        }
        // TODO: maybe no busy wait
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    auto status = BuildResult(ctx, result);
    if (!status.ok()) { return status; }
    task_manager_.Remove(task_id);
    return Status::OK();
}

Status AsuClientImpl::RegisterRegions(const std::vector<MemoryRegion>& regions,
                                      std::vector<RegisterResult>& results)
{
    auto view = view_;
    if (!view || view->transports.empty()) {
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    results.assign(regions.size(), RegisterResult{Status::OK(), kInvalidMRHandle});
    // TODO: register or bind
    return Status::OK();
}

Status AsuClientImpl::UnregisterRegions(const std::vector<MRHandle>& handles)
{
    auto view = view_;
    if (!view || view->transports.empty()) { return Status::OK(); }
    // TODO: unregister
    return Status::OK();
}

Status AsuClientImpl::SubmitAsync(ClientOpType op_type, const std::vector<KVBuffer>& entries,
                                  TaskId& task_id)
{
    auto view = view_;
    if (!view || !view->router || view->transports.empty()) {
        task_id = kInvalidTaskId;
        return Status::Error(StatusCode::NOT_INITIALIZED, "client has no ASU transports");
    }

    auto ctx = std::make_unique<ClientTaskContext>();
    ctx->op_type = op_type;
    const auto count = entries.size();
    ctx->entry_status.assign(count, Status::OK());

    std::unordered_map<AsuId, std::size_t> group_index;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& key = entries[i].key;
        const auto asu_id = view->router->Pick(key);
        auto iter = group_index.find(asu_id);
        if (iter == group_index.end()) {
            iter = group_index.emplace(asu_id, ctx->sub_tasks.size()).first;
            ClientSubTask sub_task;
            sub_task.asu_id = asu_id;
            ctx->sub_tasks.push_back(std::move(sub_task));
        }
        auto& sub_task = ctx->sub_tasks[iter->second];
        sub_task.entries.push_back(entries[i]);
        sub_task.original_indices.push_back(i);
    }

    auto status = task_manager_.Submit(std::move(ctx), task_id);
    if (!status.ok()) { return status; }

    auto raw_ctx = task_manager_.Get(task_id);
    if (!raw_ctx) {
        task_id = kInvalidTaskId;
        return Status::Error(StatusCode::INTERNAL_ERROR, "client task disappeared after submit");
    }

    status = DispatchTask(raw_ctx);
    if (!status.ok()) {
        task_manager_.Remove(task_id);
        task_id = kInvalidTaskId;
        return status;
    }

    raw_ctx->state.store(ClientTaskState::INFLIGHT, std::memory_order_release);
    return Status::OK();
}

Status AsuClientImpl::DispatchTask(const ClientTaskContextPtr& ctx)
{
    auto view = view_;
    if (!view) { return Status::Error(StatusCode::NOT_INITIALIZED, "client view is not ready"); }

    for (auto& sub_task : ctx->sub_tasks) {
        auto trans_iter = view->transports.find(sub_task.asu_id);
        if (trans_iter == view->transports.end()) {
            return Status::Error(StatusCode::NOT_FOUND, "routed ASU transport not found");
        }

        Status status;
        if (ctx->op_type == ClientOpType::LOAD) {
            status = trans_iter->second->LoadAsync(sub_task.entries, sub_task.trans_task_id);
        } else if (ctx->op_type == ClientOpType::STORE) {
            status = trans_iter->second->StoreAsync(sub_task.entries, sub_task.trans_task_id);
        } else {
            status = trans_iter->second->DeleteAsync(sub_task.keys, sub_task.trans_task_id);
        }
        // TODO: deal with partial dispatch failure
        if (!status.ok()) { return status; }
    }
    return Status::OK();
}

bool AsuClientImpl::PollTask(const ClientTaskContextPtr& ctx)
{
    auto view = view_;
    if (!ctx || ctx->Done()) { return true; }
    if (!view || ctx->state.load(std::memory_order_acquire) != ClientTaskState::INFLIGHT) {
        return false;
    }

    bool all_done = true;
    bool any_failed = false;
    for (auto& sub_task : ctx->sub_tasks) {
        auto trans_iter = view->transports.find(sub_task.asu_id);
        if (trans_iter == view->transports.end()) {
            any_failed = true;
            continue;
        }

        TaskResult sub_result;
        auto status = trans_iter->second->Check(sub_task.trans_task_id, sub_result);
        if (!status.ok()) {
            any_failed = true;
            continue;
        }
        if (sub_result.status.code == StatusCode::IN_PROGRESS) {
            all_done = false;
            continue;
        }
        if (!sub_result.status.ok()) { any_failed = true; }

        const auto& original_indices = sub_task.original_indices;
        for (std::size_t i = 0; i < original_indices.size() && i < sub_result.entry_status.size();
             ++i) {
            ctx->entry_status[original_indices[i]] = sub_result.entry_status[i];
        }
    }

    if (all_done) {
        ctx->final_status =
            any_failed ? Status::Error(StatusCode::PARTIAL_FAILED, "client task partially failed")
                       : Status::OK();
        ctx->state.store(any_failed ? ClientTaskState::FAILED : ClientTaskState::COMPLETED,
                         std::memory_order_release);
        return true;
    }
    return false;
}

Status AsuClientImpl::BuildResult(const ClientTaskContextPtr& ctx, TaskResult& result)
{
    result.status = ctx->Done() ? ctx->final_status
                                : Status::Error(StatusCode::IN_PROGRESS, "client task in progress");
    result.entry_status = ctx->entry_status;
    result.query_result.reset();
    return Status::OK();
}

std::unique_ptr<AsuClient> CreateAsuClient(TransportFactory factory)
{
    return std::make_unique<AsuClientImpl>(std::move(factory));
}

}  // namespace UC::ASU
