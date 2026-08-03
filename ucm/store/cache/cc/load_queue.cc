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
#include "load_queue.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

LoadQueue::~LoadQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    streamNumber_ = config.streamNumber;
    useGdr_ = config.useGdr;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    sdmaDirectLaunchGranularity_ = config.sdmaDirectLaunchGranularity;
    cpuAffinityCores_ = config.cpuAffinityCores;
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    transfer_ = std::thread{&LoadQueue::TransferStage, this, std::ref(started)};
    return fut.get();
}

void LoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_full_total"), 1.0);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_disp");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM load dispatcher name.", nameStatus);
    }
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
}

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    auto tp = waiter->startTp;
    auto tpWait = NowTime::Now();
    const auto nShard = task->desc.size();
    const auto taskLaunch = UseSdmaDirectTaskLaunch();
    size_t backendSubmitCount = 0;
    std::vector<ShardTask> readyTasks;
    std::vector<ShardTask> pendingTasks;
    if (taskLaunch) {
        readyTasks.reserve(nShard);
        pendingTasks.reserve(nShard);
    }
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        ShardTask shardTask;
        shardTask.bufferHandle = buffer_->Get(shard.owner, shard.index, true, true);
        shardTask.backendTaskHandle = 0;
        if (shardTask.bufferHandle.Owner() && !shardTask.bufferHandle.Ready()) {
            Detail::TaskDesc backendTask{
                Detail::Shard{shard.owner, shard.index, {shardTask.bufferHandle.Data()}}
            };
            backendTask.brief = "Backend2Cache";
            auto res = backend_->Load(std::move(backendTask));
            if (!res) [[unlikely]] {
                UC_ERROR("Failed({}) to submit load task({}) to backend.", res.Error(), task->id);
                UC::Metrics::UpdateStats(
                    NAME_TO_METRIC_ID("cache_backend_load_submit_errors_total"), 1.0);
                shardTask.bufferHandle.MarkFailed(res.Error());
                task->Fail(res.Error());
                failureSet_->Insert(task->id);
                waiter->Done();
                return;
            }
            shardTask.backendTaskHandle = res.Value();
            backendSubmitCount++;
        }
        shardTask.task = task;
        shardTask.shard = std::move(shard);
        if (taskLaunch) {
            if (shardTask.bufferHandle.Ready()) {
                readyTasks.push_back(std::move(shardTask));
            } else {
                pendingTasks.push_back(std::move(shardTask));
            }
            continue;
        }
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
        running_.Push(std::move(shardTask));
    }
    if (taskLaunch) {
        if (!readyTasks.empty()) {
            readyTasks.back().launchBoundary = true;
            if (pendingTasks.empty()) { readyTasks.back().waiter = waiter; }
        }
        if (!pendingTasks.empty()) {
            pendingTasks.back().launchBoundary = true;
            pendingTasks.back().waiter = waiter;
        }
        for (auto& shardTask : readyTasks) { running_.Push(std::move(shardTask)); }
        for (auto& shardTask : pendingTasks) { running_.Push(std::move(shardTask)); }
    }
    auto tpDispatch = NowTime::Now();
    UC_DEBUG("Cache task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
             (tpWait - tp) * 1e3, (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"),
                             (tpWait - tp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_submit_duration_ms"),
                             (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_shards_total"),
                             static_cast<double>(nShard));
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_xfer");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM load transfer name.", nameStatus); }
    CopyStream stream;
    auto s = cacheSdmaDirect_ ? stream.SetupSdmaDirect(deviceId_, useGdr_)
                              : stream.Setup(deviceId_, streamNumber_, useGdr_);
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, stream);
}

void LoadQueue::TransferOneTask(CopyStream& stream, ShardTask&& task)
{
    auto parentTask = task.task;
    const auto taskHandle = parentTask->id;
    if (failureSet_->Contains(taskHandle)) {
        if (task.waiter) {
            ClearSdmaDirectHolders();
            task.waiter->Done();
        }
        return;
    }
    auto s = Status::OK();
    auto waiter = task.waiter;
    do {
        auto tpBackendWait = NowTime::Now();
        s = WaitBackendTaskReady(task);
        if (s.Failure()) [[unlikely]] { break; }
        auto tpBackendReady = NowTime::Now();
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 (tpBackendReady - tpBackendWait) * 1e3);
        if (UseSdmaDirectTaskLaunch()) {
            const auto launchBoundary = task.launchBoundary;
            holder_.push_back(std::move(task));
            if (!launchBoundary) { return; }
            s = FlushSdmaDirectTaskBatch(stream);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to flush H2D task batch for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            if (waiter) { waiter->Done(); }
            return;
        }

        if (cacheSdmaDirect_) {
            s = HostToDeviceAsync(stream, task.bufferHandle.DeviceData(), task.shard.addrs.data());
            auto tpH2dSubmitted = NowTime::Now();
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                                     (tpH2dSubmitted - tpBackendReady) * 1e3);
            if (!task.waiter) {
                holder_.push_back(std::move(task));
                return;
            }
            auto tpH2dSyncStart = NowTime::Now();
            s = stream.Synchronize();
            auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
            RecordH2dSyncMetrics(h2dSyncMs);
            holder_.clear();
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            break;
        }

        s = HostToDeviceAsync(stream, task.bufferHandle.Data(), task.shard.addrs.data());
        auto tpH2dSubmitted = NowTime::Now();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                                 (tpH2dSubmitted - tpBackendReady) * 1e3);
        if (!task.waiter) {
            holder_.push_back(std::move(task));
            return;
        }
        auto tpH2dSyncStart = NowTime::Now();
        s = stream.Synchronize();
        auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
        RecordH2dSyncMetrics(h2dSyncMs);
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
    } while (0);
    if (s.Failure()) [[unlikely]] {
        parentTask->Fail(s);
        failureSet_->Insert(taskHandle);
    }
    if (UseSdmaDirectTaskLaunch()) { ClearSdmaDirectHolders(); }
    if (waiter) { waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.backendTaskHandle != 0) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_wait_errors_total"),
                                     1.0);
            task.bufferHandle.MarkFailed(s);
            return s;
        }
        task.bufferHandle.MarkReady();
        return Status::OK();
    }
    for (;;) {
        auto state = task.bufferHandle.GetState();
        if (state == TransBuffer::State::READY) { return Status::OK(); }
        if (state == TransBuffer::State::FAILED) { return task.bufferHandle.FailureStatus(); }
        if (failureSet_->Contains(task.task->id)) { return task.task->FailureStatus(); }
        std::this_thread::yield();
    }
}

Status LoadQueue::HostToDeviceAsync(CopyStream& stream, void* host, void** device)
{
    return stream.HostToDeviceAsync(host, device, tensorSizes_);
}

Status LoadQueue::HostToDeviceTaskAsync(CopyStream& stream, std::vector<ShardTask>& tasks)
{
    std::vector<void*> hosts;
    std::vector<void**> devices;
    hosts.reserve(tasks.size());
    devices.reserve(tasks.size());
    for (auto& task : tasks) {
        hosts.push_back(cacheSdmaDirect_ ? task.bufferHandle.DeviceData()
                                         : task.bufferHandle.Data());
        devices.push_back(task.shard.addrs.data());
    }
    return stream.HostToDeviceAsync(hosts, devices, tensorSizes_);
}

Status LoadQueue::FlushSdmaDirectTaskBatch(CopyStream& stream)
{
    if (holder_.empty()) { return Status::OK(); }
    auto tpH2dSubmitStart = NowTime::Now();
    auto s = HostToDeviceTaskAsync(stream, holder_);
    auto tpH2dSubmitted = NowTime::Now();
    if (s.Failure()) [[unlikely]] {
        ClearSdmaDirectHolders();
        return s;
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                             (tpH2dSubmitted - tpH2dSubmitStart) * 1e3);
    auto tpH2dSyncStart = NowTime::Now();
    s = stream.Synchronize();
    auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
    RecordH2dSyncMetrics(h2dSyncMs);
    ClearSdmaDirectHolders();
    return s;
}

void LoadQueue::RecordH2dSyncMetrics(double h2dSyncMs) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_sync_ms"), h2dSyncMs);
}

void LoadQueue::ClearSdmaDirectHolders() noexcept { holder_.clear(); }

bool LoadQueue::UseSdmaDirectTaskLaunch() const noexcept
{
    return cacheSdmaDirect_ && sdmaDirectLaunchGranularity_ == kSdmaDirectLaunchTask;
}

}  // namespace UC::CacheStore
