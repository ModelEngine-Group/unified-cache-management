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
#include <acl/acl_rt.h>
#include <chrono>
#include <numeric>
#include "logger/logger.h"
#include "metrics_api.h"
#include "replica.h"
#include "thread/cpu_affinity.h"

namespace UC::MooncakeStore {

DumpQueue::~DumpQueue() { Close(); }

void DumpQueue::Close()
{
    if (stop_.exchange(true)) { return; }
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (dumper_.joinable()) { dumper_.join(); }

    TaskPair pair;
    while (waiting_.TryPop(pair)) {
        if (pair.second) { pair.second->Done(); }
    }
    DumpCtx ctx;
    while (dumping_.TryPop(ctx)) {}
}

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet,
                        std::shared_ptr<mooncake::RealClient> realClient, StoreV1* backend,
                        HostBufferPool* bufPool)
{
    failureSet_ = failureSet;
    realClient_ = std::move(realClient);
    backend_ = backend;
    bufPool_ = bufPool;
    replicaNum_ = config.replicaNum;
    withSoftPin_ = config.withSoftPin;
    tensorSizes_ = config.tensorSizeList;
    deviceId_ = config.deviceId;
    streamNumber_ = config.streamNumber;
    cpuAffinityCores_ = config.cpuAffinityCores;

    waiting_.Setup(config.dumpQueueDepth);
    dumping_.Setup(config.dumpQueueDepth);

    dumper_ = std::thread{&DumpQueue::BackendDumpStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    dispatcher_ = std::thread{&DumpQueue::DispatchStage, this, std::ref(started)};
    return fut.get();
}

void DumpQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    if (waiting_.TryPush({task, waiter})) { return; }
    UC_ERROR("Waiting queue full, submit dump task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_queue_full_total"), 1.0);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::DispatchStage(std::promise<Status>& started)
{
    CopyStream stream;
    auto s = stream.Setup(deviceId_, streamNumber_);
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
    UC_DEBUG("Mooncake task({}) start running, wait {:.3f}ms.", task->id, wait * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_queue_wait_duration_ms"), wait * 1e3);
    if (!failureSet_->Contains(task->id)) {
        auto s = DumpOneTask(stream, task);
        if (s.Failure()) [[unlikely]] { failureSet_->Insert(task->id); }
    }
    waiter->Done();
}

Status DumpQueue::WaitPrerequisite(TaskPtr task)
{
    if (task->prerequisiteHandle == 0) { return Status::OK(); }
    auto tp = NowTime::Now();
    auto event = reinterpret_cast<aclrtEvent>(task->prerequisiteHandle);
    auto ret = aclrtSynchronizeEvent(event);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_prereq_wait_ms"),
                             (NowTime::Now() - tp) * 1e3);
    if (ret != 0) {
        UC_ERROR("aclrtSynchronizeEvent failed, ret={}, task={}", ret, task->id);
        return Status::Error("aclrtSynchronizeEvent failed");
    }
    return Status::OK();
}

Status DumpQueue::PrepareBackendDump(CopyStream& stream, TaskPtr task, DumpCtx& dumpCtx,
                                     Detail::TaskDesc& backendTaskDesc)
{
    backendTaskDesc.brief = "HostBuf2Backend";
    dumpCtx.taskHandle = task->id;
    for (auto& shard : task->shards) {
        auto buf = bufPool_->AcquireWithTimeout(std::chrono::milliseconds(3000));
        if (!buf) {
            UC_WARN("Host buffer pool exhausted for key={}, skip backend archive", shard.key);
            break;
        }

        auto s = DeviceToHostGatherAsync(stream.NextStream(), shard.addrs.data(), buf.get());
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do D2H batch async for task({}).", s, task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_d2h_errors_total"), 1.0);
            return s;
        }

        dumpCtx.hostBufs.push_back(std::move(buf));
        backendTaskDesc.push_back(
            Detail::Shard{shard.owner, shard.index, {dumpCtx.hostBufs.back().get()}});
    }
    return Status::OK();
}

Status DumpQueue::PutToMooncake(TaskPtr task, const std::vector<std::string>& keys,
                                const std::vector<std::vector<void*>>& allBuffers,
                                const std::vector<std::vector<size_t>>& allSizes)
{
    mooncake::ReplicateConfig cfg;
    cfg.replica_num = replicaNum_;
    cfg.with_soft_pin = withSoftPin_;
    auto tp = NowTime::Now();
    auto putResults = realClient_->batch_put_from_multi_buffers(keys, allBuffers, allSizes, cfg);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_put_duration_ms"),
                             (NowTime::Now() - tp) * 1e3);

    for (size_t i = 0; i < putResults.size(); ++i) {
        if (putResults[i] < 0) {
            UC_ERROR("Mooncake batch_put failed: task={}, key={}, rc={}", task->id, keys[i],
                     putResults[i]);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_put_errors_total"), 1.0);
            return Status::Error("batch_put failed");
        }
    }
    return Status::OK();
}

Status DumpQueue::SubmitBackendDump(CopyStream& stream, TaskPtr task, DumpCtx&& dumpCtx,
                                    Detail::TaskDesc&& backendTaskDesc)
{
    const auto copiedShards = dumpCtx.hostBufs.size();
    const auto backendShards = backendTaskDesc.size();
    auto tpSyncStart = NowTime::Now();
    auto s = stream.Synchronize();
    auto tpSyncEnd = NowTime::Now();
    RecordD2hMetrics(copiedShards, (tpSyncEnd - tpSyncStart) * 1e3);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to sync on stream for task({}).", s, task->id);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_d2h_errors_total"), 1.0);
        return s;
    }

    auto tpBackendSubmit = NowTime::Now();
    auto res = backend_->Dump(std::move(backendTaskDesc));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_backend_submit_duration_ms"),
                             (NowTime::Now() - tpBackendSubmit) * 1e3);
    if (!res) [[unlikely]] {
        UC_ERROR("Failed({}) to submit dump task({}) to backend.", res.Error(), task->id);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_backend_dump_submit_errors_total"),
                                 1.0);
        return res.Error();
    }
    dumpCtx.backendTaskHandle = res.Value();
    dumping_.Push(std::move(dumpCtx));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_backend_shards_total"),
                             static_cast<double>(backendShards));
    return Status::OK();
}

Status DumpQueue::DumpOneTask(CopyStream& stream, TaskPtr task)
{
    auto tp = NowTime::Now();
    UC_DEBUG("Try to dump ({}) shards.", task->shards.size());

    std::vector<std::string> keys;
    std::vector<std::vector<void*>> allBuffers;
    std::vector<std::vector<size_t>> allSizes;
    keys.reserve(task->shards.size());
    allBuffers.reserve(task->shards.size());
    allSizes.reserve(task->shards.size());
    for (auto& s : task->shards) {
        keys.push_back(s.key);
        allBuffers.push_back(s.addrs);
        allSizes.push_back(s.sizes);
    }

    auto tpExists = NowTime::Now();
    auto existsResult = realClient_->batchIsExist(keys);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_exists_duration_ms"),
                             (NowTime::Now() - tpExists) * 1e3);
    std::vector<std::string> missingKeys;
    std::vector<std::vector<void*>> missingBuffers;
    std::vector<std::vector<size_t>> missingSizes;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (existsResult[i] != 1) {
            missingKeys.push_back(keys[i]);
            missingBuffers.push_back(allBuffers[i]);
            missingSizes.push_back(allSizes[i]);
        }
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_existing_shards_total"),
                             static_cast<double>(keys.size() - missingKeys.size()));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_missing_shards_total"),
                             static_cast<double>(missingKeys.size()));

    UC_DEBUG("Mooncake task({}) exists check: {}/{} keys missing.", task->id, missingKeys.size(),
             keys.size());

    Detail::TaskDesc backendTaskDesc;
    DumpCtx dumpCtx;
    if (backend_) {
        if (task->prerequisiteHandle != 0) {
            auto s = stream.WaitEvent(reinterpret_cast<void*>(task->prerequisiteHandle));
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to set stream wait event for task({}).", s, task->id);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_d2h_errors_total"), 1.0);
                return s;
            }
        }
        auto s = PrepareBackendDump(stream, task, dumpCtx, backendTaskDesc);
        if (s.Failure()) [[unlikely]] { return s; }
    }

    if (!missingKeys.empty()) {
        auto s = WaitPrerequisite(task);
        if (s.Failure()) [[unlikely]] { return s; }
        s = PutToMooncake(task, missingKeys, missingBuffers, missingSizes);
        if (s.Failure()) [[unlikely]] { return s; }
    }

    if (!dumpCtx.hostBufs.empty()) {
        auto s = SubmitBackendDump(stream, task, std::move(dumpCtx), std::move(backendTaskDesc));
        if (s.Failure()) [[unlikely]] { return s; }
    }

    UC_DEBUG("Mooncake task({}) dump done, cost={:.3f}ms.", task->id, (NowTime::Now() - tp) * 1e3);
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
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_dump_backend_wait_duration_ms"),
                                     (NowTime::Now() - tpWait) * 1e3);
            finishedBackendTaskHandle_ = task.backendTaskHandle;
            if (s.Failure()) {
                UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                         task.taskHandle);
                UC::Metrics::UpdateStats(
                    NAME_TO_METRIC_ID("mooncake_backend_dump_wait_errors_total"), 1.0);
                return;
            }
        }
    });
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

size_t DumpQueue::BlockBytes() const
{
    return std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), size_t{0});
}

void DumpQueue::RecordD2hMetrics(size_t copiedShards, double d2hMs) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_d2h_duration_ms"), d2hMs);
    auto copiedBytes = static_cast<double>(copiedShards) * static_cast<double>(BlockBytes());
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_d2h_bytes_total"), copiedBytes);
}

}  // namespace UC::MooncakeStore
