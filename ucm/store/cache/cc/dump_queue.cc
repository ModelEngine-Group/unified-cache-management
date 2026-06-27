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
#include <cstddef>
#include <cstring>
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
    shardBytes_ = 0;
    for (const auto size : tensorSizes_) { shardBytes_ += size; }
    streamNumber_ = config.streamNumber;
    useGdr_ = config.useGdr;
    useHostBuffer_ = config.useHostBuffer;
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
    CopyStream stream;
    auto s = useHostBuffer_ ? Status::OK() : stream.Setup(deviceId_, streamNumber_, useGdr_);
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
        if (s.Failure()) [[unlikely]] { failureSet_->Insert(task->id); }
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
    if (useHostBuffer_ && task->desc.prerequisiteHandle != 0) {
        return Status::InvalidParam("prerequisite handle is unsupported for host buffer dump");
    }
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
        if (useHostBuffer_) {
            auto s = ValidateHostAddrs(shard);
            if (s.Failure()) [[unlikely]] { return s; }
        }
        auto handle = buffer_->Get(shard.owner, shard.index);
        if (!handle.Owner()) { continue; }
        if (useHostBuffer_ || !handle.Ready()) {
            auto s = useHostBuffer_
                         ? HostToHostGather(shard.addrs, handle.Data())
                         : DeviceToHostGatherAsync(stream.NextStream(), shard.addrs.data(),
                                                   handle.Data());
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to gather {} buffers for task({}).", s,
                         useHostBuffer_ ? "host" : "device", task->id);
                if (!useHostBuffer_) {
                    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
                }
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
    if (!useHostBuffer_) {
        auto s = stream.Synchronize();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
            return s;
        }
    }
    auto tpSyncStream = NowTime::Now();
    auto tpBackendSubmitStart = NowTime::Now();
    for (auto& handle : dumpCtx.bufferHandles) { handle.MarkReady(); }
    auto res = backend_->Dump(std::move(backendTaskDesc));
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit dump task({}) to backend.", res.Error(), task->id);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_submit_errors_total"), 1.0);
        return res.Error();
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
        auto copiedBytes = static_cast<double>(copiedShards) * static_cast<double>(shardBytes_);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_duration_ms"), d2hMs);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_bandwidth_gbps"),
                                 copiedBytes / (d2hMs * 1e-3) / 1e9);
    }
    UC_DEBUG("Cache task({}) mk_buf={:.3f}ms, prereq={:.3f}ms, d2h={:.3f}ms, back={:.3f}ms.",
             task->id, (tpMakeBuffer - dumpStartTp) * 1e3, prereqWaitMs, d2hMs,
             (tpEnd - tpBackendSubmitStart) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_mkbuf_duration_ms"),
                             (tpMakeBuffer - dumpStartTp) * 1e3);
    if (!useHostBuffer_) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_duration_ms"),
                                 (tpSyncStream - tpMakeBuffer) * 1e3);
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_submit_duration_ms"),
                             (tpEnd - tpBackendSubmitStart) * 1e3);
    return Status::OK();
}

Status DumpQueue::DeviceToHostGatherAsync(std::shared_ptr<Trans::Stream> stream, void** device,
                                          void* host)
{
    const auto number = tensorSizes_.size();
    for (size_t i = 0, offset = 0; i < number; i++) {
        auto pDevice = device[i];
        auto pHost = (void*)(((int8_t*)host) + offset);
        auto size = tensorSizes_[i];
        auto s = stream->DeviceToHostAsync(pDevice, pHost, size);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do D2H({}) batch({}/{}) async.", s, size, i, number);
            return s;
        }
        offset += size;
    }
    return Status::OK();
}

Status DumpQueue::HostToHostGather(const std::vector<void*>& sources, void* destination)
{
    auto s = ValidateHostAddrs(Detail::Shard{{}, 0, sources});
    if (s.Failure()) [[unlikely]] { return s; }
    if (!destination) { return Status::InvalidParam("null host destination"); }
    auto* destinationBytes = static_cast<std::byte*>(destination);
    size_t offset = 0;
    for (size_t i = 0; i < tensorSizes_.size(); ++i) {
        std::memcpy(destinationBytes + offset, sources[i], tensorSizes_[i]);
        offset += tensorSizes_[i];
    }
    return Status::OK();
}

Status DumpQueue::ValidateHostAddrs(const Detail::Shard& shard) const
{
    if (shard.addrs.size() != tensorSizes_.size()) {
        return Status::InvalidParam("invalid host buffer number({}), expected({})",
                                    shard.addrs.size(), tensorSizes_.size());
    }
    for (size_t i = 0; i < shard.addrs.size(); ++i) {
        if (!shard.addrs[i]) { return Status::InvalidParam("null host source({})", i); }
    }
    return Status::OK();
}

void DumpQueue::BackendDumpStage()
{
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
