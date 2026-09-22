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
#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <utility>
#include <vector>
#include "cache_buffer.h"
#include "copy_stream.h"
#include "ctrl_layout.h"
#include "global_config.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "status/status.h"
#include "template/spsc_ring_queue.h"
#include "thread/cpu_affinity.h"
#include "time/now_time.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::Cache2 {

/**
 * @brief Dump pipeline: device KV shards -> host cache buffer -> backend store.
 *
 * Two-stage design on SPSC queues (single producer: TransManager dispatch):
 *  - dispatcher_: pop task -> wait prerequisite event -> pin buffer slots ->
 *    D2H gather -> mark slots ready -> submit to backend -> signal waiter.
 *    A task is complete for the caller once its data is in the cache buffer
 *    and the backend write has been submitted.
 *  - dumper_: wait for each backend write, then release the pinned slots.
 * Shards whose slot is owned by a concurrent task are skipped (dedup).
 */
template <typename BufferT = Buffer, typename StreamT = CopyStream>
class DumpQ {
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using Handle = typename BufferT::Handle;
    using SlotState = CtrlLayout::SlotMeta::State;
    struct DumpCtx {
        Detail::TaskHandle taskHandle{};
        Detail::TaskHandle backendTaskHandle{};
        std::vector<Handle> bufferHandles{};
    };

    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    BufferT* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    size_t streamNumber_{1};
    std::vector<size_t> tensorSizes_{};
    SpscRingQueue<TaskPair> waiting_{};
    SpscRingQueue<DumpCtx> dumping_{};
    std::thread dispatcher_{};
    std::thread dumper_{};

public:
    DumpQ() = default;
    ~DumpQ() { Close(); }
    DumpQ(const DumpQ&) = delete;
    DumpQ& operator=(const DumpQ&) = delete;

    Status Setup(const Config& config, TaskIdSet* failureSet, BufferT* buffer)
    {
        failureSet_ = failureSet;
        buffer_ = buffer;
        backend_ = config.storeBackend;
        deviceId_ = config.deviceId;
        streamNumber_ = config.streamNumber;
        tensorSizes_ = config.tensorSizes;
        waiting_.Setup(config.waitingQueueDepth);
        if (backend_ != nullptr) {
            dumping_.Setup(config.runningQueueDepth);
            dumper_ = std::thread{&DumpQ::BackendWaitStage, this};
        }
        std::promise<Status> started;
        auto fut = started.get_future();
        dispatcher_ = std::thread{&DumpQ::DispatchStage, this, std::ref(started)};
        return fut.get();
    }
    void Submit(TaskPtr task, WaiterPtr waiter)
    {
        waiter->Up();
        if (waiting_.TryPush({task, waiter})) { return; }
        UC_ERROR("Waiting queue full, submit dump task({}) failed.", task->id);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_full_total"), 1.0);
        failureSet_->Insert(task->id);
        waiter->Done();
    }
    void Close()
    {
        if (stop_.exchange(true)) { return; }
        if (dispatcher_.joinable()) { dispatcher_.join(); }
        DrainWaiting();
        if (dumper_.joinable()) { dumper_.join(); }
        DrainDumping();
    }

private:
    void DispatchStage(std::promise<Status>& started)
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_c2_dump_d2h");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set dump d2h thread name.", nameStatus);
        }
        StreamT stream;
        auto s = stream.Setup(deviceId_, streamNumber_);
        started.set_value(s);
        if (s.Failure()) [[unlikely]] { return; }
        waiting_.ConsumerLoop(stop_, &DumpQ::DispatchOneTask, this, stream);
    }
    void DispatchOneTask(StreamT& stream, TaskPair&& pair)
    {
        auto& [task, waiter] = pair;
        auto queueWaitMs = (NowTime::Now() - waiter->startTp) * 1e3;
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_wait_duration_ms"), queueWaitMs);
        if (!failureSet_->Contains(task->id)) {
            auto s = DumpOneTask(stream, task);
            if (s.Failure()) [[unlikely]] { failureSet_->Insert(task->id); }
        }
        waiter->Done();
    }
    Status DumpOneTask(StreamT& stream, const TaskPtr& task)
    {
        auto s = WaitPrerequisite(stream, task);
        if (s.Failure()) [[unlikely]] { return s; }
        DumpCtx ctx;
        ctx.taskHandle = task->id;
        ctx.bufferHandles.reserve(task->desc.size());
        Detail::TaskDesc backendDesc;
        backendDesc.brief = "Cache2Backend";
        size_t copiedShards = 0;
        s = GatherShards(stream, task, ctx, backendDesc, copiedShards);
        if (s.Failure()) [[unlikely]] { return s; }
        if (ctx.bufferHandles.empty()) { return Status::OK(); }
        s = CompleteD2h(stream, task, ctx, copiedShards);
        if (s.Failure()) [[unlikely]] { return s; }
        if (backend_ == nullptr || backendDesc.empty()) { return Status::OK(); }
        return SubmitBackendDump(task, ctx, std::move(backendDesc));
    }
    Status WaitPrerequisite(StreamT& stream, const TaskPtr& task)
    {
        if (task->desc.prerequisiteHandle == 0) { return Status::OK(); }
        auto s = stream.WaitEvent(Trans::Event{task->desc.prerequisiteHandle});
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait prerequisite event for dump task({}).", s, task->id);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
        }
        return s;
    }
    Status GatherShards(StreamT& stream, const TaskPtr& task, DumpCtx& ctx,
                        Detail::TaskDesc& backendDesc, size_t& copiedShards)
    {
        const auto startTp = NowTime::Now();
        if (backend_ != nullptr) { backendDesc.reserve(task->desc.size()); }
        for (auto& shard : task->desc) {
            auto handle = buffer_->Get(shard.owner, shard.index);
            if (!handle.Owner()) { continue; }
            ctx.bufferHandles.push_back(std::move(handle));
            auto& pinned = ctx.bufferHandles.back();
            const auto hostAccessible = pinned.HostAccessible();
            if (pinned.GetState() != SlotState::Ready) {
                auto s = hostAccessible
                             ? stream.DeviceToHostGatherAsync(shard.addrs.data(), pinned.Data(),
                                                              tensorSizes_)
                             : stream.DeviceToDeviceGatherAsync(shard.addrs.data(),
                                                                pinned.DeviceData(), tensorSizes_);
                if (s.Failure()) [[unlikely]] {
                    UC_ERROR("Failed({}) to copy shard for dump task({}).", s, task->id);
                    Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
                    FailOwnedHandles(ctx);
                    return s;
                }
                ++copiedShards;
            }
            if (backend_ != nullptr && hostAccessible) {
                backendDesc.push_back(Detail::Shard{shard.owner, shard.index, {pinned.Data()}});
            }
        }
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_mkbuf_duration_ms"),
                             (NowTime::Now() - startTp) * 1e3);
        return Status::OK();
    }
    Status CompleteD2h(StreamT& stream, const TaskPtr& task, DumpCtx& ctx, size_t copiedShards)
    {
        if (copiedShards > 0) {
            const auto startTp = NowTime::Now();
            auto s = stream.Synchronize();
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to sync stream for dump task({}).", s, task->id);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
                FailOwnedHandles(ctx);
                return s;
            }
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_duration_ms"),
                                 (NowTime::Now() - startTp) * 1e3);
        }
        for (auto& handle : ctx.bufferHandles) { handle.MarkReady(); }
        return Status::OK();
    }
    Status SubmitBackendDump(const TaskPtr& task, DumpCtx& ctx, Detail::TaskDesc&& backendDesc)
    {
        const auto startTp = NowTime::Now();
        const auto nBackendShards = backendDesc.size();
        auto res = backend_->Dump(std::move(backendDesc));
        if (!res) [[unlikely]] {
            auto error = res.Error();
            if (error != Status::StoreUnhealthy()) {
                UC_ERROR("Failed({}) to submit dump task({}) to backend.", error, task->id);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_submit_errors_total"),
                                     1.0);
            }
            return error;
        }
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_shards_total"),
                             static_cast<double>(nBackendShards));
        ctx.backendTaskHandle = res.Value();
        dumping_.Push(std::move(ctx));
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_submit_duration_ms"),
                             (NowTime::Now() - startTp) * 1e3);
        return Status::OK();
    }
    void FailOwnedHandles(DumpCtx& ctx)
    {
        for (auto& handle : ctx.bufferHandles) { handle.MarkFailed(); }
    }
    void BackendWaitStage()
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_c2_dump_wait");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set dump wait thread name.", nameStatus);
        }
        dumping_.ConsumerLoop(stop_, [this](DumpCtx&& ctx) {
            if (ctx.backendTaskHandle == 0) { return; }
            const auto startTp = NowTime::Now();
            auto s = backend_->Wait(ctx.backendTaskHandle);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_wait_duration_ms"),
                                 (NowTime::Now() - startTp) * 1e3);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to wait backend({}) for dump task({}).", s,
                         ctx.backendTaskHandle, ctx.taskHandle);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_wait_errors_total"),
                                     1.0);
                return;
            }
        });
    }
    void DrainWaiting()
    {
        TaskPair pair;
        while (waiting_.TryPop(pair)) {
            UC_WARN("Cache2 dump task({}) discarded on close.", pair.first->id);
            failureSet_->Insert(pair.first->id);
            pair.second->Done();
        }
    }
    void DrainDumping()
    {
        if (backend_ == nullptr) { return; }
        DumpCtx ctx;
        while (dumping_.TryPop(ctx)) {
            if (ctx.backendTaskHandle == 0) { continue; }
            auto s = backend_->Wait(ctx.backendTaskHandle);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to wait backend({}) for dump task({}).", s,
                         ctx.backendTaskHandle, ctx.taskHandle);
            }
        }
    }
};

}  // namespace UC::Cache2
