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
#include "logger/logger.h"
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
    streamNumber_ = config.streamNumber;
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
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::DispatchStage(std::promise<Status>& started)
{
    CopyStream stream;
    auto s = stream.Setup(deviceId_, streamNumber_);
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
    if (!cpuAffinityCores_.empty() && s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    waiting_.ConsumerLoop(stop_, &DumpQueue::DispatchOneTask, this, stream);
}

void DumpQueue::DispatchOneTask(CopyStream& stream, TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    auto wait = NowTime::Now() - waiter->startTp;
    UC_DEBUG("Cache task({}) start running, wait {:.3f}ms.", task->id, wait * 1e3);
    if (!failureSet_->Contains(task->id)) {
        auto s = DumpOneTask(stream, task);
        if (s.Failure()) [[unlikely]] { failureSet_->Insert(task->id); }
    }
    waiter->Done();
}

Status DumpQueue::DumpOneTask(CopyStream& stream, TaskPtr task)
{
    auto tp = NowTime::Now();
    Detail::TaskDesc backendTaskDesc;
    backendTaskDesc.brief = "Cache2Backend";
    const auto nShard = task->desc.size();
    UC_DEBUG("Try to dump ({}) shards.", nShard);
    DumpCtx dumpCtx;
    dumpCtx.taskHandle = task->id;
    if (task->desc.prerequisiteHandle != 0) {
        auto s = stream.WaitEvent(reinterpret_cast<void*>(task->desc.prerequisiteHandle));
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait prerequisite event for dump task({}).", s, task->id);
            return s;
        }
    }
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        auto handle = buffer_->Get(shard.owner, shard.index);
        if (!handle.Owner()) { continue; }
        if (!handle.Ready()) {
            auto s =
                DeviceToHostGatherAsync(stream.NextStream(), shard.addrs.data(), handle.Data());
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do D2H batch async for task({}).", s, task->id);
                return s;
            }
        }
        backendTaskDesc.push_back(Detail::Shard{shard.owner, shard.index, {handle.Data()}});
        dumpCtx.bufferHandles.push_back(std::move(handle));
    }
    auto tpMakeBuffer = NowTime::Now();
    if (backendTaskDesc.empty()) { return Status::OK(); }
    auto s = stream.Synchronize();
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to sync on stream for task({}).", s, task->id);
        return s;
    }
    auto tpSyncStream = NowTime::Now();
    for (auto& handle : dumpCtx.bufferHandles) { handle.MarkReady(); }
    auto res = backend_->Dump(std::move(backendTaskDesc));
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit dump task({}) to backend.", res.Error(), task->id);
        return res.Error();
    }
    dumpCtx.backendTaskHandle = res.Value();
    dumping_.Push(std::move(dumpCtx));
    auto tpEnd = NowTime::Now();
    UC_DEBUG("Cache task({}) mk_buf={:.3f}ms, sync={:.3f}ms, back={:.3f}ms.", task->id,
             (tpMakeBuffer - tp) * 1e3, (tpSyncStream - tpMakeBuffer) * 1e3,
             (tpEnd - tpSyncStream) * 1e3);
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

void DumpQueue::BackendDumpStage()
{
    auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
    if (!cpuAffinityCores_.empty() && s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    dumping_.ConsumerLoop(stop_, [this](auto&& task) {
        if (task.backendTaskHandle > finishedBackendTaskHandle_) {
            auto s = backend_->Wait(task.backendTaskHandle);
            finishedBackendTaskHandle_ = task.backendTaskHandle;
            if (s.Failure()) {
                UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                         task.taskHandle);
                return;
            }
        }
    });
}

}  // namespace UC::CacheStore
