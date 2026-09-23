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
#include <optional>
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
 * @brief Load pipeline: host cache buffer (filled from backend) -> device.
 *
 * Two-stage design on SPSC queues (single producer: TransManager dispatch):
 *  - dispatcher_: pop task -> pin buffer slots (cache hits skip the backend)
 *    -> submit backend loads for cold owned slots -> push shard tasks in a
 *    rank-interleaved order -> prealloc follow-up shard slots.
 *  - transfer_: per shard, wait until the slot is ready (backend wait for
 *    owned slots, state polling for others) -> H2D scatter. Only the last
 *    shard of a task carries the waiter and performs the single stream
 *    synchronize covering all pending H2D copies of the task.
 */
template <typename BufferT = Buffer, typename StreamT = CopyStream>
class LoadQ {
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using Handle = typename BufferT::Handle;
    using SlotState = CtrlLayout::SlotMeta::State;
    struct ShardTask {
        TaskPtr task;
        Detail::Shard* shard{nullptr};
        std::optional<Handle> bufferHandle;
        Detail::TaskHandle backendTaskHandle{0};
        WaiterPtr waiter;
        bool fromCache{false};
    };

    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    BufferT* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    size_t streamNumber_{1};
    size_t localRankSize_{1};
    size_t nShardPerBlock_{0};
    std::vector<size_t> tensorSizes_{};
    SpscRingQueue<TaskPair> waiting_{};
    SpscRingQueue<ShardTask> running_{};
    std::thread dispatcher_{};
    std::thread transfer_{};
    std::vector<ShardTask> holder_{};

public:
    LoadQ() = default;
    ~LoadQ() { Close(); }
    LoadQ(const LoadQ&) = delete;
    LoadQ& operator=(const LoadQ&) = delete;

    Status Setup(const Config& config, TaskIdSet* failureSet, BufferT* buffer)
    {
        failureSet_ = failureSet;
        buffer_ = buffer;
        backend_ = config.storeBackend;
        deviceId_ = config.deviceId;
        streamNumber_ = config.streamNumber;
        localRankSize_ = config.localRankSize;
        nShardPerBlock_ = config.blockSize / config.shardSize;
        tensorSizes_ = config.tensorSizes;
        waiting_.Setup(config.waitingQueueDepth);
        running_.Setup(config.runningQueueDepth);
        dispatcher_ = std::thread{&LoadQ::DispatchStage, this};
        std::promise<Status> started;
        auto fut = started.get_future();
        transfer_ = std::thread{&LoadQ::TransferStage, this, std::ref(started)};
        return fut.get();
    }
    void Submit(TaskPtr task, WaiterPtr waiter)
    {
        waiter->Up();
        if (waiting_.TryPush({task, waiter})) { return; }
        UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_full_total"), 1.0);
        failureSet_->Insert(task->id);
        waiter->Done();
    }
    void Close()
    {
        if (stop_.exchange(true)) { return; }
        if (dispatcher_.joinable()) { dispatcher_.join(); }
        DrainWaiting();
        if (transfer_.joinable()) { transfer_.join(); }
        DrainRunning();
    }

private:
    void DispatchStage()
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_c2_load_disp");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set load dispatch thread name.", nameStatus);
        }
        waiting_.ConsumerLoop(stop_, &LoadQ::DispatchOneTask, this);
    }
    void DispatchOneTask(TaskPair&& pair)
    {
        auto& [task, waiter] = pair;
        auto queueWaitMs = (NowTime::Now() - waiter->startTp) * 1e3;
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"), queueWaitMs);
        if (failureSet_->Contains(task->id)) {
            waiter->Done();
            return;
        }
        DispatchShards(task, waiter);
    }
    void DispatchShards(const TaskPtr& task, const WaiterPtr& waiter)
    {
        const auto startTp = NowTime::Now();
        const auto nShard = task->desc.size();
        const auto indexes = RearrangeIndex(nShard, deviceId_, localRankSize_);
        size_t backendSubmitCount = 0;
        size_t waitShardCount = 0;
        for (size_t i = 0; i < nShard; ++i) {
            auto& shard = task->desc[indexes[i]];
            ShardTask shardTask;
            auto handle = buffer_->Get(shard.owner, shard.index, true);
            shardTask.fromCache = handle.GetState() == SlotState::Ready;
            if (!shardTask.fromCache) { ++waitShardCount; }
            if (handle.Owner() && !shardTask.fromCache) {
                auto res = SubmitBackendLoad(task, shard, handle);
                if (!res) [[unlikely]] {
                    handle.MarkFailed();
                    RecordLoadSourceShards(i + 1, waitShardCount);
                    RecordFailedShards(nShard - i);
                    FailTask(task, waiter);
                    return;
                }
                shardTask.backendTaskHandle = res.Value();
                ++backendSubmitCount;
            }
            shardTask.task = task;
            shardTask.shard = &shard;
            shardTask.bufferHandle = std::move(handle);
            shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
            running_.Push(std::move(shardTask));
        }
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_submit_duration_ms"),
                             (NowTime::Now() - startTp) * 1e3);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
        RecordLoadSourceShards(nShard, waitShardCount);
        PreallocNextShards(task, indexes);
    }
    Expected<Detail::TaskHandle> SubmitBackendLoad(const TaskPtr& task, const Detail::Shard& shard,
                                                   Handle& handle)
    {
        if (!handle.HostAccessible()) {
            UC_ERROR("Load task({}) shard miss on host-inaccessible slot.", task->id);
            return Status::Error("load on host-inaccessible slot");
        }
        if (backend_ == nullptr) {
            UC_ERROR("Load task({}) shard miss without backend.", task->id);
            return Status::Error("load without backend");
        }
        Detail::TaskDesc backendTask;
        backendTask.brief = "Backend2Cache";
        backendTask.push_back(Detail::Shard{shard.owner, shard.index, {handle.Data()}});
        auto res = backend_->Load(std::move(backendTask));
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit load task({}) to backend.", res.Error(), task->id);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_submit_errors_total"), 1.0);
        }
        return res;
    }
    void PreallocNextShards(const TaskPtr& task, const std::vector<size_t>& indexes)
    {
        for (auto i : indexes) {
            auto& shard = task->desc[i];
            if (shard.index + 1 != nShardPerBlock_) {
                buffer_->Prealloc(shard.owner, shard.index + 1, true);
            }
        }
    }
    static std::vector<size_t> RearrangeIndex(size_t n, size_t iProc, size_t nProc)
    {
        std::vector<size_t> order;
        order.reserve(n);
        for (size_t r = 0; r < nProc; ++r) {
            size_t slice = (iProc + r) % nProc;
            for (size_t j = 0;; ++j) {
                size_t i = slice + j * nProc;
                if (i >= n) { break; }
                order.push_back(i);
            }
        }
        return order;
    }
    void TransferStage(std::promise<Status>& started)
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_c2_load_h2d");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set load h2d thread name.", nameStatus);
        }
        StreamT stream;
        auto s = stream.Setup(deviceId_, streamNumber_);
        started.set_value(s);
        if (s.Failure()) [[unlikely]] { return; }
        running_.ConsumerLoop(stop_, &LoadQ::TransferOneTask, this, stream);
    }
    void TransferOneTask(StreamT& stream, ShardTask&& task)
    {
        if (failureSet_->Contains(task.task->id)) {
            DiscardFailedShard(task);
            FinishShard(stream, std::move(task), Status::Error());
            return;
        }
        auto s = WaitShardReady(task);
        if (s.Success()) { s = ScatterShard(stream, task); }
        if (s.Failure()) [[unlikely]] {
            RecordShardResults(holder_, &task, false);
            FinishShard(stream, std::move(task), s);
            return;
        }
        if (!task.waiter) {
            holder_.push_back(std::move(task));
            return;
        }
        s = SynchronizeH2d(stream, task);
        RecordShardResults(holder_, &task, s.Success());
        holder_.clear();
        FinishShard(stream, std::move(task), s);
    }
    void DiscardFailedShard(ShardTask& task)
    {
        if (task.bufferHandle->GetState() != SlotState::Ready) { task.bufferHandle->MarkFailed(); }
        RecordFailedShards(1);
    }
    Status WaitShardReady(ShardTask& task)
    {
        if (task.backendTaskHandle != 0) {
            const auto waitStartTp = NowTime::Now();
            auto s = backend_->Wait(task.backendTaskHandle);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 (NowTime::Now() - waitStartTp) * 1e3);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to wait backend({}) for load task({}).", s,
                         task.backendTaskHandle, task.task->id);
                Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_wait_errors_total"),
                                     1.0);
                task.bufferHandle->MarkFailed();
                return s;
            }
            task.bufferHandle->MarkReady();
            return Status::OK();
        }
        for (;;) {
            auto state = task.bufferHandle->GetState();
            if (state == SlotState::Ready) { return Status::OK(); }
            if (state == SlotState::Failed) { return Status::Error(); }
            if (failureSet_->Contains(task.task->id)) { return Status::Error(); }
            std::this_thread::yield();
        }
    }
    Status ScatterShard(StreamT& stream, ShardTask& task)
    {
        const auto startTp = NowTime::Now();
        auto s = task.bufferHandle->HostAccessible()
                     ? stream.HostToDeviceScatterAsync(task.bufferHandle->Data(),
                                                       task.shard->addrs.data(), tensorSizes_)
                     : stream.DeviceToDeviceScatterAsync(task.bufferHandle->DeviceData(),
                                                         task.shard->addrs.data(), tensorSizes_);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                             (NowTime::Now() - startTp) * 1e3);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to copy shard for load task({}).", s, task.task->id);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
        }
        return s;
    }
    Status SynchronizeH2d(StreamT& stream, const ShardTask& task)
    {
        const auto startTp = NowTime::Now();
        auto s = stream.Synchronize();
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_sync_ms"),
                             (NowTime::Now() - startTp) * 1e3);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync stream for load task({}).", s, task.task->id);
            Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
        }
        return s;
    }
    void FinishShard(StreamT& stream, ShardTask&& task, const Status& s)
    {
        if (s.Failure()) [[unlikely]] {
            failureSet_->Insert(task.task->id);
            ReleaseHolder(stream);
        }
        if (task.waiter) { task.waiter->Done(); }
    }
    void ReleaseHolder(StreamT& stream)
    {
        if (holder_.empty()) { return; }
        stream.Synchronize();
        holder_.clear();
    }
    void RecordShardResults(const std::vector<ShardTask>& tasks, const ShardTask* extra,
                            bool success) const
    {
        size_t cache = 0;
        size_t posix = 0;
        for (auto& t : tasks) { t.fromCache ? ++cache : ++posix; }
        if (extra != nullptr) { extra->fromCache ? ++cache : ++posix; }
        if (!success) {
            RecordFailedShards(cache + posix);
            return;
        }
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_success_shards_total"),
                             static_cast<double>(cache));
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_posix_load_success_shards_total"),
                             static_cast<double>(posix));
    }
    void RecordLoadSourceShards(size_t total, size_t wait) const
    {
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_shards_total"),
                             static_cast<double>(total));
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_wait_shards_total"),
                             static_cast<double>(wait));
    }
    void RecordFailedShards(size_t count) const
    {
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_failed_shards_total"),
                             static_cast<double>(count));
    }
    void FailTask(const TaskPtr& task, const WaiterPtr& waiter)
    {
        failureSet_->Insert(task->id);
        waiter->Done();
    }
    void DrainWaiting()
    {
        TaskPair pair;
        while (waiting_.TryPop(pair)) {
            UC_WARN("Cache2 load task({}) discarded on close.", pair.first->id);
            failureSet_->Insert(pair.first->id);
            pair.second->Done();
        }
    }
    void DrainRunning()
    {
        ShardTask shardTask;
        while (running_.TryPop(shardTask)) {
            UC_WARN("Cache2 load task({}) shard discarded on close.", shardTask.task->id);
            failureSet_->Insert(shardTask.task->id);
            if (shardTask.waiter) { shardTask.waiter->Done(); }
        }
    }
};

}  // namespace UC::Cache2
