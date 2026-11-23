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
#include <cstddef>
#include <cstring>
#include <future>
#include <memory>
#include <numeric>
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
 *    owned slots, state polling for others). HBM then does an H2D scatter;
 *    the last shard (or a marker after a dispatch failure) carries the waiter.
 *    Pending H2D copies are synchronized before completion. DDR skips the
 *    device copy: Load memcpy's the whole shard, LoadOnToken splices the
 *    selected tensor ranges into the caller buffer.
 */
template <typename BufferT = Buffer, typename StreamT = CopyStream>
class LoadQ {
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using Handle = typename BufferT::Handle;
    using SlotState = CtrlLayout::SlotMeta::State;
    struct ShardTask {
        TaskPtr task;
        Detail::Shard* shard{nullptr};
        size_t filterIndex{0};
        std::optional<Handle> bufferHandle;
        Detail::TaskHandle backendTaskHandle{0};
        WaiterPtr waiter;
        bool fromCache{false};
        bool firstShard{false};
    };

    alignas(64) std::atomic_bool stop_{false};
    std::atomic_bool transferStop_{false};
    TaskIdSet* failureSet_{nullptr};
    BufferT* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    size_t streamNumber_{1};
    bool sdmaDirect_{false};
    bool ioAggregation_{false};
    bool useGdr_{false};
    size_t localRankSize_{1};
    size_t nShardPerBlock_{0};
    size_t shardSize_{0};
    size_t slotWaitTimeoutMs_{Config::kDefaultTimeoutMs};
    std::vector<size_t> tensorSizes_{};
    std::vector<ssize_t> cpuAffinityCores_{};
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
        streamNumber_ = config.EffectiveStreamNumber();
        sdmaDirect_ = config.sdmaDirect;
        ioAggregation_ = config.ioAggregation;
        useGdr_ = config.useGdr;
        localRankSize_ = config.localRankSize;
        nShardPerBlock_ = config.blockSize / config.shardSize;
        shardSize_ = config.shardSize;
        // Even with an unlimited task Wait(), another rank may never publish its slot.
        slotWaitTimeoutMs_ = config.timeoutMs == 0 ? Config::kDefaultTimeoutMs : config.timeoutMs;
        tensorSizes_ = config.tensorSizes;
        cpuAffinityCores_ = config.cpuAffinityCores;
        waiting_.Setup(config.waitingQueueDepth);
        running_.Setup(config.runningQueueDepth);
        std::promise<Status> started;
        auto fut = started.get_future();
        transfer_ = std::thread{&LoadQ::TransferStage, this, std::ref(started)};
        auto status = fut.get();
        if (status.Success()) { dispatcher_ = std::thread{&LoadQ::DispatchStage, this}; }
        return status;
    }
    void Submit(TaskPtr task, WaiterPtr waiter)
    {
        waiter->Up();
        if (waiting_.TryPush({task, waiter})) { return; }
        UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_full_total"), 1.0);
        RecordFailedShards(task->desc.size());
        failureSet_->Insert(task->id);
        waiter->Done();
    }
    void Close()
    {
        if (stop_.exchange(true)) { return; }
        // Keep consuming while the dispatcher publishes its final shard or failure marker.
        if (dispatcher_.joinable()) { dispatcher_.join(); }
        DrainWaiting();
        transferStop_.store(true, std::memory_order_release);
        if (transfer_.joinable()) { transfer_.join(); }
    }

private:
    void DispatchStage()
    {
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_disp");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set load dispatch thread name.", nameStatus);
        }
        BindCpuAffinity();
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
        if (task->type == Task::Type::LOAD_ON_TOKEN) {
            auto s = ValidateTokenFilter(task);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("LoadOnToken task({}) invalid filter: {}.", task->id, s);
                RecordFailedShards(nShard);
                task->Fail(s);
                FailTask(task, waiter);
                return;
            }
        }
        std::vector<size_t> indexes(nShard);
        if (task->desc.distributeBackendLoad) {
            indexes = RearrangeIndex(nShard, deviceId_, localRankSize_);
        } else {
            std::iota(indexes.begin(), indexes.end(), 0);
        }
        auto status = Status::OK();
        size_t backendSubmitCount = 0;
        size_t acquiredShardCount = 0;
        size_t waitShardCount = 0;
        size_t i = 0;
        for (; i < nShard; ++i) {
            const auto originalIndex = indexes[i];
            auto& shard = task->desc[originalIndex];
            const auto preferredRank =
                task->desc.distributeBackendLoad ? originalIndex % localRankSize_ : kInvalid;
            auto acquired = AcquireShardSlot(task, shard, preferredRank);
            if (!acquired) {
                status = acquired.Error();
                if (status == Status::Timeout()) { InvalidatePrefix(shard); }
                break;
            }
            ++acquiredShardCount;
            auto handle = std::move(acquired).Value();
            const auto state = handle.GetState();
            ShardTask shardTask;
            shardTask.fromCache = state == SlotState::Ready;
            if (!shardTask.fromCache) { ++waitShardCount; }
            if (state == SlotState::Failed) {
                status = Status::Error();
                break;
            }
            if (handle.Owner() && !shardTask.fromCache) {
                auto res = SubmitBackendLoad(task, shard, handle);
                if (!res) [[unlikely]] {
                    FailSlot(shard, handle);
                    status = res.Error();
                    break;
                }
                shardTask.backendTaskHandle = res.Value();
                ++backendSubmitCount;
            }
            shardTask.task = task;
            shardTask.shard = &shard;
            if (task->type == Task::Type::LOAD_ON_TOKEN) { shardTask.filterIndex = originalIndex; }
            shardTask.bufferHandle = std::move(handle);
            shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
            shardTask.firstShard = i == 0;
            running_.Push(std::move(shardTask));
        }
        RecordLoadSourceShards(acquiredShardCount, waitShardCount);
        if (status.Failure()) [[unlikely]] {
            RecordFailedShards(nShard - i);
            task->Fail(status);
            failureSet_->Insert(task->id);
            if (task->desc.distributeBackendLoad && !stop_.load(std::memory_order_relaxed)) {
                PublishFailedSlots(task, indexes, i + 1);
            }
            if (i == 0) {
                waiter->Done();
                return;
            }
            // The transfer thread owns any queued backend reads and pending copies.
            ShardTask end;
            end.task = task;
            end.waiter = waiter;
            running_.Push(std::move(end));
            return;
        }
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_submit_duration_ms"),
                             (NowTime::Now() - startTp) * 1e3);
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
        PreallocNextShards(task, indexes);
    }
    Expected<Handle> AcquireShardSlot(const TaskPtr& task, const Detail::Shard& shard,
                                      size_t preferredRank)
    {
        auto handle = buffer_->TryGet(shard.owner, shard.index, true, preferredRank);
        if (handle) { return std::move(*handle); }
        const auto deadline = NowTime::Now() + slotWaitTimeoutMs_ * 1e-3;
        for (;;) {
            auto status = SlotWaitStatus(task);
            if (status.Failure()) { return status; }
            if (NowTime::Now() >= deadline) {
                UC_ERROR("Load task({}) timed out acquiring shard({}) slot, preferred rank({}).",
                         task->id, shard.index, preferredRank);
                return Status::Timeout();
            }
            std::this_thread::yield();
            handle = buffer_->TryGet(shard.owner, shard.index, true, preferredRank);
            if (handle) { return std::move(*handle); }
        }
    }
    void PublishFailedSlots(const TaskPtr& task, const std::vector<size_t>& indexes, size_t start)
    {
        // Local shards come first. Visit the remaining local slots once; any that cannot be
        // acquired have the peer's timed acquisition as a fallback.
        const auto localRank = static_cast<size_t>(deviceId_) % localRankSize_;
        for (size_t i = start; i < indexes.size(); ++i) {
            const auto originalIndex = indexes[i];
            const auto preferredRank = originalIndex % localRankSize_;
            if (preferredRank != localRank) { break; }
            const auto& shard = task->desc[originalIndex];
            auto handle = buffer_->TryGet(shard.owner, shard.index, true, preferredRank);
            if (handle && handle->Owner()) { FailSlot(shard, *handle); }
        }
    }
    void InvalidatePrefix(const Detail::Shard& shard)
    {
        if (shard.index != 0) { buffer_->InvalidateReady(shard.owner, 0); }
    }
    void FailSlot(const Detail::Shard& shard, Handle& handle)
    {
        handle.MarkFailed();
        InvalidatePrefix(shard);
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
        for (const auto i : indexes) {
            auto& shard = task->desc[i];
            if (shard.index + 1 != nShardPerBlock_) {
                if (task->desc.distributeBackendLoad) {
                    buffer_->Prealloc(shard.owner, shard.index + 1, true, i % localRankSize_);
                } else {
                    buffer_->Prealloc(shard.owner, shard.index + 1, true);
                }
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
        auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_h2d");
        if (nameStatus.Failure()) {
            UC_WARN("Failed({}) to set load h2d thread name.", nameStatus);
        }
        StreamT stream;
        auto s = Status::OK();
        if (ioAggregation_) {
            s = stream.SetupIoAggregation(deviceId_, useGdr_);
        } else if (sdmaDirect_) {
            s = stream.SetupSdmaDirect(deviceId_, streamNumber_, useGdr_);
        } else {
            s = stream.Setup(deviceId_, streamNumber_, useGdr_);
        }
        started.set_value(s);
        if (s.Failure()) [[unlikely]] { return; }
        BindCpuAffinity();
        running_.ConsumerLoop(transferStop_, &LoadQ::TransferOneTask, this, stream);
        ReleaseHolder(stream);
        DrainRunning();
    }
    Status ValidateTokenFilter(const TaskPtr& task) const
    {
        const auto nShard = task->desc.size();
        if (task->filter.size() != nShard) {
            return Status::InvalidParam("filter size({}) != shards({})", task->filter.size(),
                                        nShard);
        }
        for (size_t i = 0; i < nShard; ++i) {
            const auto& tokenFilter = task->filter[i];
            if (tokenFilter.ranges.size() != tensorSizes_.size()) {
                return Status::InvalidParam("shard({}) tensor filter size({}) != tensorSizes({})",
                                            task->desc[i].index, tokenFilter.ranges.size(),
                                            tensorSizes_.size());
            }
            for (size_t t = 0; t < tokenFilter.ranges.size(); ++t) {
                const auto tensorSize = tensorSizes_[t];
                for (const auto& range : tokenFilter.ranges[t]) {
                    if (range.length == 0) {
                        return Status::InvalidParam("shard({}) tensor({}) has empty token range",
                                                    task->desc[i].index, t);
                    }
                    if (range.offset >= tensorSize || range.length > tensorSize - range.offset) {
                        return Status::InvalidParam(
                            "shard({}) tensor({}) range({},{}) exceeds size({})",
                            task->desc[i].index, t, range.offset, range.length, tensorSize);
                    }
                }
            }
        }
        return Status::OK();
    }
    void TransferOneTask(StreamT& stream, ShardTask&& task)
    {
        if (!task.bufferHandle) {
            ReleaseHolder(stream);
            task.waiter->Done();
            return;
        }
        if (task.task->desc.memoryType == Detail::TaskDesc::MemoryType::DDR) {
            TransferDdr(std::move(task));
            return;
        }
        if (stop_.load(std::memory_order_relaxed) || failureSet_->Contains(task.task->id)) {
            DiscardFailedShard(task);
            FinishShard(stream, std::move(task), Status::Error());
            return;
        }
        auto s = WaitShardReady(task);
        if (s.Success() && task.firstShard && task.task->desc.prerequisiteHandle != 0) {
            // CopyStream applies the dependency to every rotating copy stream.
            s = stream.WaitEvent(Trans::Event{task.task->desc.prerequisiteHandle});
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to wait prerequisite event for load task({}).", s,
                         task.task->id);
            }
        }
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
    void TransferDdr(ShardTask&& task)
    {
        if (stop_.load(std::memory_order_relaxed) || failureSet_->Contains(task.task->id)) {
            DiscardFailedShard(task);
            FinishDdrShard(std::move(task), Status::Error());
            return;
        }
        auto s = WaitShardReady(task);
        if (s.Success()) { s = CopyDdrShard(task); }
        RecordShardResults({}, &task, s.Success());
        FinishDdrShard(std::move(task), s);
    }
    Status CopyDdrShard(ShardTask& task)
    {
        if (!task.bufferHandle->HostAccessible()) {
            UC_ERROR("DDR load task({}) shard is not host-accessible.", task.task->id);
            return Status::Error("ddr load on host-inaccessible slot");
        }
        auto* src = task.bufferHandle->Data();
        auto* dst = task.shard->addrs.empty() ? nullptr : task.shard->addrs.front();
        if (src == nullptr || dst == nullptr || task.shard->addrs.size() != 1) [[unlikely]] {
            UC_ERROR("DDR load shard({}) requires one host address.", task.shard->index);
            return Status::InvalidParam();
        }
        if (task.task->type == Task::Type::LOAD_ON_TOKEN) {
            return SpliceTokenRanges(src, static_cast<std::byte*>(dst), task);
        }
        std::memcpy(dst, src, shardSize_);
        return Status::OK();
    }
    Status SpliceTokenRanges(void* src, std::byte* dst, const ShardTask& task) const
    {
        const auto& tokenFilter = task.task->filter[task.filterIndex];
        auto* bytes = static_cast<const std::byte*>(src);
        size_t tensorBase = 0;
        for (size_t t = 0; t < tokenFilter.ranges.size(); ++t) {
            for (const auto& range : tokenFilter.ranges[t]) {
                std::memcpy(dst + range.dstOffset, bytes + tensorBase + range.offset, range.length);
            }
            tensorBase += tensorSizes_[t];
        }
        return Status::OK();
    }
    void FinishDdrShard(ShardTask&& task, const Status& s)
    {
        if (s.Failure()) [[unlikely]] {
            task.task->Fail(s);
            failureSet_->Insert(task.task->id);
        }
        if (task.waiter) { task.waiter->Done(); }
    }
    void DiscardFailedShard(ShardTask& task)
    {
        // An accepted read can still write into the slot after this task fails.
        if (task.backendTaskHandle != 0) { WaitShardReady(task); }
        RecordFailedShards(1);
    }
    Status SlotWaitStatus(const TaskPtr& task) const
    {
        if (stop_.load(std::memory_order_relaxed)) { return Status::Error(); }
        if (failureSet_->Contains(task->id)) { return task->FailureStatus(); }
        return Status::OK();
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
                FailSlot(*task.shard, *task.bufferHandle);
                return s;
            }
            task.bufferHandle->MarkReady();
            return SlotWaitStatus(task.task);
        }
        for (;;) {
            auto state = task.bufferHandle->GetState();
            if (state == SlotState::Ready) { return Status::OK(); }
            if (state == SlotState::Failed) { return Status::Error(); }
            auto status = SlotWaitStatus(task.task);
            if (status.Failure()) { return status; }
            std::this_thread::yield();
        }
    }
    Status ScatterShard(StreamT& stream, ShardTask& task)
    {
        const auto startTp = NowTime::Now();
        /* SDMA Direct feeds device descriptors, so the host side has to be the
         * device-visible alias rather than the CPU address. */
        void* hostSrc =
            sdmaDirect_ ? task.bufferHandle->HostMappedData() : task.bufferHandle->Data();
        auto s =
            task.bufferHandle->HostAccessible()
                ? stream.HostToDeviceScatterAsync(hostSrc, task.shard->addrs.data(), tensorSizes_)
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
            task.task->Fail(s);
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
    void BindCpuAffinity() const
    {
        if (cpuAffinityCores_.empty()) { return; }
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set cache load affinity.", s); }
    }
    void DrainRunning()
    {
        ShardTask shardTask;
        while (running_.TryPop(shardTask)) {
            UC_WARN("Cache2 load task({}) shard discarded on close.", shardTask.task->id);
            failureSet_->Insert(shardTask.task->id);
            if (shardTask.bufferHandle) { DiscardFailedShard(shardTask); }
            if (shardTask.waiter) { shardTask.waiter->Done(); }
        }
    }
};

}  // namespace UC::Cache2
