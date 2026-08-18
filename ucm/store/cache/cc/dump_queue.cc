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
#include <memory>
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

DumpQueue::~DumpQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
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
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    cpuAffinityCores_ = config.cpuAffinityCores;
    waiting_.Setup(config.waitingQueueDepth);
    dumping_.Setup(config.runningQueueDepth);
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
    auto s = cacheSdmaDirect_ ? stream.SetupSdmaDirect(deviceId_, useGdr_)
                              : stream.Setup(deviceId_, streamNumber_, useGdr_);
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
        auto s = stream.WaitEvent(reinterpret_cast<void*>(task->desc.prerequisiteHandle));
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

void DumpQueue::BackendDumpStage()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_dump_back");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM dump backend name.", nameStatus); }
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    dumping_.ConsumerLoop(stop_, [this](auto&& task) {
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
