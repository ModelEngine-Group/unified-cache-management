/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
#include "dump_queue.h"
#include <algorithm>
#include <atomic>
#include <list>
#include <memory>
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

DumpQueue::~DumpQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (useHostBuffer_) { hostCopyExecutor_.Synchronize(); }
    backendStop_.store(true);
    if (dumper_.joinable()) { dumper_.join(); }
}

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    streamNumber_ = config.EffectiveStreamNumber();
    useGdr_ = config.useGdr;
    cacheIOAggregation_ = config.cacheIOAggregation;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    useHostBuffer_ = config.cacheUseHostBuffer;
    cpuAffinityCores_ = config.cpuAffinityCores;
    waiting_.Setup(config.waitingQueueDepth);
    dumping_.Setup(config.runningQueueDepth);
    if (useHostBuffer_) {
        auto s = hostCopyExecutor_.Setup(config.h2hWorkerNumber, config.h2hQueueDepth,
                                         cpuAffinityCores_);
        if (s.Failure()) { return s; }
    }
    dumper_ = std::thread{&DumpQueue::BackendDumpStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    dispatcher_ = std::thread{&DumpQueue::DispatchStage, this, std::ref(started)};
    return fut.get();
}

void DumpQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit dump task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_full_total"), 1.0);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::DispatchStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_dump_disp");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM dump dispatcher name.", nameStatus);
    }
    CopyStream stream;
    auto s = Status::OK();
    if (useHostBuffer_) {
        s = Status::OK();
    } else if (cacheIOAggregation_) {
        s = stream.SetupIoAggregation(deviceId_, useGdr_);
    } else if (cacheSdmaDirect_) {
        s = stream.SetupSdmaDirect(deviceId_, useGdr_);
    } else {
        s = stream.Setup(deviceId_, streamNumber_, useGdr_);
    }
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &DumpQueue::DispatchOneTask, this, stream);
}

void DumpQueue::DispatchOneTask(CopyStream& stream, TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    auto wait = NowTime::Now() - waiter->startTp;
    UC_DEBUG("Cache task({}) start running, wait {:.3f}ms.", task->id, wait * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_wait_duration_ms"), wait * 1e3);
    if (useHostBuffer_) {
        if (!failureSet_->Contains(task->id)) {
            auto s = DispatchH2HDump(task, waiter);
            if (s.Success()) { return; }
            task->Fail(s);
            failureSet_->Insert(task->id);
        }
        waiter->Done();
        return;
    }
    if (!failureSet_->Contains(task->id)) {
        auto s = DumpOneTask(stream, task);
        if (s.Failure()) [[unlikely]] {
            if (s == Status::StoreUnhealthy()) { task->Fail(s); }
            failureSet_->Insert(task->id);
        }
    }
    waiter->Done();
}

Status DumpQueue::DumpOneTask(CopyStream& stream, TaskPtr task)
{
    auto dumpStartTp = NowTime::Now();
    Detail::TaskDesc backendTaskDesc;
    backendTaskDesc.brief = "Cache2Backend";
    const auto nShard = task->desc.size();
    UC_DEBUG("Try to dump ({}) shards.", nShard);
    DumpCtx dumpCtx;
    dumpCtx.taskHandle = task->id;
    std::shared_ptr<std::atomic<double>> eventReadyTp;
    if (task->desc.prerequisiteHandle != 0) {
        auto s = stream.WaitEvent(Trans::Event{task->desc.prerequisiteHandle});
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait prerequisite event for dump task({}).", s, task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
            return s;
        }
        eventReadyTp = std::make_shared<std::atomic<double>>(0.0);
        auto cbStatus = stream.AppendCallback([eventReadyTp](bool) {
            eventReadyTp->store(NowTime::Now(), std::memory_order_release);
        });
        if (cbStatus.Failure()) [[unlikely]] { eventReadyTp.reset(); }
    }
    size_t copiedShards = 0;
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        auto handle = buffer_->Get(shard.owner, shard.index);
        if (!handle.Owner()) { continue; }
        if (!handle.Ready()) {
            auto* host = cacheSdmaDirect_ ? handle.DeviceData() : handle.Data();
            auto s = DeviceToHostAsync(stream, shard.addrs.data(), host);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do D2H for task({}).", s, task->id);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
                return s;
            }
            copiedShards++;
        }
        backendTaskDesc.push_back(Detail::Shard{shard.owner, shard.index, {handle.Data()}});
        dumpCtx.bufferHandles.push_back(std::move(handle));
    }
    auto tpMakeBuffer = NowTime::Now();
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_shards_total"),
                             static_cast<double>(nShard));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_shards_total"),
                             static_cast<double>(backendTaskDesc.size()));
    if (backendTaskDesc.empty()) { return Status::OK(); }
    auto tpSyncStart = NowTime::Now();
    auto s = stream.Synchronize();
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to sync on stream for task({}).", s, task->id);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
        return s;
    }
    auto tpSyncStream = NowTime::Now();
    auto tpBackendSubmitStart = NowTime::Now();
    for (auto& handle : dumpCtx.bufferHandles) { handle.MarkReady(); }
    auto res = backend_->Dump(std::move(backendTaskDesc));
    if (!res) [[unlikely]] {
        auto error = res.Error();
        if (error != Status::StoreUnhealthy()) {
            UC_ERROR("Failed({}) to submit dump task({}) to backend.", error, task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_submit_errors_total"),
                                     1.0);
        }
        return error;
    }
    dumpCtx.backendTaskHandle = res.Value();
    dumping_.Push(std::move(dumpCtx));
    auto tpEnd = NowTime::Now();
    auto prereqWaitMs = 0.0;
    auto d2hMs = std::max(0.0, tpSyncStream - tpSyncStart) * 1e3;
    if (eventReadyTp) {
        auto ready = eventReadyTp->load(std::memory_order_acquire);
        if (ready > 0.0) {
            prereqWaitMs = std::max(0.0, ready - dumpStartTp) * 1e3;
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_prereq_wait_ms"), prereqWaitMs);
        }
    }
    if (copiedShards > 0 && d2hMs > 0.0) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_duration_ms"), d2hMs);
    }
    UC_DEBUG("Cache task({}) mk_buf={:.3f}ms, prereq={:.3f}ms, d2h={:.3f}ms, back={:.3f}ms.",
             task->id, (tpMakeBuffer - dumpStartTp) * 1e3, prereqWaitMs, d2hMs,
             (tpEnd - tpBackendSubmitStart) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_mkbuf_duration_ms"),
                             (tpMakeBuffer - dumpStartTp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_submit_duration_ms"),
                             (tpEnd - tpBackendSubmitStart) * 1e3);
    return Status::OK();
}

Status DumpQueue::DeviceToHostAsync(CopyStream& stream, void** device, void* host)
{
    return stream.DeviceToHostAsync(device, host, tensorSizes_);
}

Status DumpQueue::DispatchH2HDump(TaskPtr task, WaiterPtr waiter)
{
    if (task->desc.prerequisiteHandle != 0) {
        return Status::InvalidParam(
            "host-to-host dump does not accept a device prerequisite event");
    }

    auto context = std::make_shared<H2HDumpContext>();
    context->task = task;
    context->waiter = waiter;
    context->backendTaskDesc.brief = "Cache2Backend";
    context->bufferHandles.reserve(task->desc.size());
    struct PendingCopy {
        size_t shardIndex;
        size_t handleIndex;
    };
    std::vector<PendingCopy> pendingCopies;

    for (size_t shardIndex = 0; shardIndex < task->desc.size(); ++shardIndex) {
        const auto& shard = task->desc[shardIndex];
        auto handle = buffer_->Get(shard.owner, shard.index);
        if (!handle.Owner()) { continue; }
        const auto handleIndex = context->bufferHandles.size();
        const auto ready = handle.Ready();
        context->backendTaskDesc.push_back(
            Detail::Shard{shard.owner, shard.index, {handle.Data()}});
        context->bufferHandles.push_back(std::move(handle));
        if (!ready) { pendingCopies.push_back({shardIndex, handleIndex}); }
    }

    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_shards_total"),
                             static_cast<double>(task->desc.size()));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_shards_total"),
                             static_cast<double>(context->backendTaskDesc.size()));
    if (context->backendTaskDesc.empty()) {
        waiter->Done();
        return Status::OK();
    }
    if (pendingCopies.empty()) {
        return hostCopyExecutor_.PostCompletion(
            [this, context](const Trans::HostCopyExecutor::Result&) {
                CompleteH2HDump(context);
            });
    }
    auto reservationResult = hostCopyExecutor_.Reserve(pendingCopies.size());
    if (!reservationResult) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2h_dump_queue_full_total"), 1.0);
        auto s = Status::Error("CacheStore H2H dump queue full");
        for (const auto& copy : pendingCopies) {
            context->bufferHandles[copy.handleIndex].MarkFailed(s);
        }
        return s;
    }
    auto reservation = std::move(reservationResult).Value();
    context->pending.store(pendingCopies.size(), std::memory_order_release);
    std::list<Trans::HostCopyExecutor::Job> jobs;
    for (const auto& copy : pendingCopies) {
        auto& shard = context->task->desc[copy.shardIndex];
        Trans::HostCopyExecutor::Job job;
        job.direction = Trans::HostCopyExecutor::Direction::GATHER;
        job.contiguous = context->bufferHandles[copy.handleIndex].Data();
        job.segments = MakeH2HSegments(shard);
        job.prerequisite = [this, context, shardIndex = copy.shardIndex] {
            const auto& currentShard = context->task->desc[shardIndex];
            if (currentShard.addrs.size() != tensorSizes_.size()) {
                return Status::InvalidParam("invalid host addr number({}, expect {})",
                                            currentShard.addrs.size(), tensorSizes_.size());
            }
            return Status::OK();
        };
        job.completion = [this, context, handleIndex = copy.handleIndex](
                             const Trans::HostCopyExecutor::Result& result) {
            CompleteH2HCopy(context, handleIndex, result);
        };
        jobs.push_back(std::move(job));
    }
    auto submitStatus = reservation.Submit(jobs);
    if (submitStatus.Failure()) {
        for (const auto& copy : pendingCopies) {
            context->bufferHandles[copy.handleIndex].MarkFailed(submitStatus);
        }
        return submitStatus;
    }
    return Status::OK();
}

void DumpQueue::CompleteH2HCopy(const H2HDumpContextPtr& context, size_t handleIndex,
                                const Trans::HostCopyExecutor::Result& result)
{
    auto& handle = context->bufferHandles[handleIndex];
    auto s = result.status;
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to do H2H gather for task({}).", s, context->task->id);
        handle.MarkFailed(s);
        context->task->Fail(s);
        context->failed.store(true, std::memory_order_release);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2h_dump_errors_total"), 1.0);
    } else {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2h_dump_bytes_total"),
                                 static_cast<double>(result.bytes));
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2h_dump_duration_ms"),
                                 result.durationMs);
    }
    if (context->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        CompleteH2HDump(context);
    }
}

void DumpQueue::CompleteH2HDump(H2HDumpContextPtr context)
{
    if (context->failed.load(std::memory_order_acquire)) {
        auto error = context->task->FailureStatus();
        for (auto& handle : context->bufferHandles) { handle.MarkFailed(error); }
        failureSet_->Insert(context->task->id);
        context->waiter->Done();
        return;
    }
    for (auto& handle : context->bufferHandles) { handle.MarkReady(); }
    auto res = backend_->Dump(std::move(context->backendTaskDesc));
    if (!res) [[unlikely]] {
        auto error = res.Error();
        context->task->Fail(error);
        failureSet_->Insert(context->task->id);
        UC_ERROR("Failed({}) to submit H2H dump task({}) to backend.", error,
                 context->task->id);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_submit_errors_total"),
                                 1.0);
        context->waiter->Done();
        return;
    }
    DumpCtx dumpCtx;
    dumpCtx.taskHandle = context->task->id;
    dumpCtx.backendTaskHandle = res.Value();
    dumpCtx.bufferHandles = std::move(context->bufferHandles);
    dumping_.Push(std::move(dumpCtx));
    context->waiter->Done();
}

std::vector<Trans::HostCopyExecutor::Segment> DumpQueue::MakeH2HSegments(
    const Detail::Shard& shard) const
{
    std::vector<Trans::HostCopyExecutor::Segment> segments;
    if (shard.addrs.size() != tensorSizes_.size()) { return segments; }
    segments.reserve(tensorSizes_.size());
    for (size_t i = 0; i < tensorSizes_.size(); ++i) {
        segments.push_back({shard.addrs[i], tensorSizes_[i]});
    }
    return segments;
}

void DumpQueue::BackendDumpStage()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_dump_back");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM dump backend name.", nameStatus); }
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    dumping_.ConsumerLoop(backendStop_, [this](auto&& task) {
        if (task.backendTaskHandle > finishedBackendTaskHandle_) {
            auto tpWait = NowTime::Now();
            auto s = backend_->Wait(task.backendTaskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_wait_duration_ms"),
                                     (NowTime::Now() - tpWait) * 1e3);
            finishedBackendTaskHandle_ = task.backendTaskHandle;
            if (s.Failure()) {
                UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                         task.taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_wait_errors_total"),
                                         1.0);
                return;
            }
        }
    });
}

}  // namespace UC::CacheStore
