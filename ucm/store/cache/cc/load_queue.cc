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
#include "trans/device.h"

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
    backend_ = static_cast<StoreV1*>((void*)config.storeBackend);
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    transfer_ = std::thread{&LoadQueue::TransferStage, this, config.deviceId, config.tensorSize,
                            std::ref(started)};
    return fut.get();
}

void LoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage() { waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this); }

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    auto wait = NowTime::Now() - waiter->startTp;
    UC_DEBUG("Cache task({}) start running, wait {:.3f}ms.", task->id, wait * 1e3);
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    Detail::TaskDesc backendTaskDesc;
    backendTaskDesc.brief = "Backend2Cache";
    const auto nShard = task->desc.size();
    std::vector<size_t> backendTaskIndex;
    backendTaskIndex.reserve(nShard);
    std::vector<ShardTask> shardTasks(nShard);
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        auto& shardTask = shardTasks[i];
        shardTask.bufferHandle = buffer_->Get(shard.owner, shard.index);
        shardTask.backendTaskHandle = 0;
        if (shardTask.bufferHandle.Owner() && !shardTask.bufferHandle.Ready()) {
            backendTaskDesc.push_back(
                Detail::Shard{shard.owner, shard.index, {shardTask.bufferHandle.Data()}});
            backendTaskIndex.emplace_back(i);
        }
        shardTask.taskHandle = task->id;
        shardTask.shard = std::move(shard);
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
    }
    if (!backendTaskDesc.empty()) {
        auto res = backend_->Load(std::move(backendTaskDesc));
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit load task to backend.", res.Error());
            failureSet_->Insert(task->id);
            waiter->Done();
            return;
        }
        for (const auto& i : backendTaskIndex) { shardTasks[i].backendTaskHandle = res.Value(); }
    }
    for (size_t i = 0; i < nShard; i++) { running_.Push(std::move(shardTasks[i])); }
}

void LoadQueue::TransferStage(int32_t deviceId, size_t tensorSize, std::promise<Status>& started)
{
    Trans::Device device;
    auto s = device.Setup(deviceId);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
        started.set_value(s);
        return;
    }
    auto stream = device.MakeStream();
    if (!stream) [[unlikely]] {
        UC_ERROR("Failed to make stream on device({}).", deviceId);
        started.set_value(Status::Error());
        return;
    }
    started.set_value(Status::OK());
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, stream.get(), tensorSize);
}

void LoadQueue::TransferOneTask(Trans::Stream* stream, size_t tensorSize, ShardTask&& task)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.waiter) { task.waiter->Done(); }
        return;
    }
    auto s = Status::OK();
    do {
        s = WaitBackendTaskReady(task);
        if (s.Failure()) [[unlikely]] { break; }
        s = stream->HostToDeviceAsync(task.bufferHandle.Data(), task.shard.addrs.data(), tensorSize,
                                      task.shard.addrs.size());
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D({}) batch({}) async.", s, tensorSize,
                     task.shard.addrs.size());
            break;
        }
        if (!task.waiter) { return; }
        s = stream->Synchronized();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream.", s);
            break;
        }
    } while (0);
    if (s.Failure()) [[unlikely]] { failureSet_->Insert(task.taskHandle); }
    if (task.waiter) { task.waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.bufferHandle.Ready()) { return Status::OK(); }
    if (!task.bufferHandle.Owner()) {
        for (;;) {
            if (failureSet_->Contains(task.taskHandle)) { return Status::Error(); }
            if (task.bufferHandle.Ready()) { return Status::OK(); }
            std::this_thread::yield();
        }
    }
    if (task.backendTaskHandle > finishedBackendTaskHandle_) {
        auto s = backend_->Wait(task.backendTaskHandle);
        finishedBackendTaskHandle_ = task.backendTaskHandle;
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend task({}).", s, task.backendTaskHandle);
            return s;
        }
    }
    task.bufferHandle.MarkReady();
    return Status::OK();
}

}  // namespace UC::CacheStore
