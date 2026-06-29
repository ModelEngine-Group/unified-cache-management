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
#include <chrono>
#include "logger/logger.h"
#include "share_load_queue.h"
#include "thread/cpu_affinity.h"

namespace UC::MooncakeStore {

LoadQueue::~LoadQueue() { Close(); }

void LoadQueue::Close()
{
    if (stop_.exchange(true)) { return; }
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }

    TaskPair pair;
    while (waiting_.TryPop(pair)) {
        if (pair.second) { pair.second->Done(); }
    }
    ShardTask task;
    while (running_.TryPop(task)) {
        if (task.waiter) { task.waiter->Done(); }
    }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet,
                        std::shared_ptr<mooncake::RealClient> realClient, StoreV1* backend,
                        HostBufferPool* bufPool, ShareLoadQueue* shareLoadQ)
{
    failureSet_ = failureSet;
    realClient_ = std::move(realClient);
    backend_ = backend;
    bufPool_ = bufPool;
    shareLoadQ_ = shareLoadQ;
    tensorSizes_ = config.tensorSizeList;
    deviceId_ = config.deviceId;
    streamNumber_ = config.streamNumber;
    cpuAffinityCores_ = config.cpuAffinityCores;

    waiting_.Setup(config.loadQueueDepth);
    running_.Setup(config.loadQueueDepth);
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
    if (waiting_.TryPush({task, waiter})) { return; }
    UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
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
    const auto nShard = task->shards.size();

    auto results = TryMooncakeLoad(task);

    size_t missCount = 0;
    for (size_t i = 0; i < nShard; i++) {
        if (results[i] < 0) { ++missCount; }
    }

    if (missCount == 0) {
        waiter->Done();
        UC_DEBUG("Mooncake task({}) all hit({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
                 (tpWait - tp) * 1e3, (NowTime::Now() - tpWait) * 1e3);
        return;
    }

    if (!backend_) {
        UC_WARN("Mooncake miss({}/{}) with no backend, task={}, will recompute.", missCount, nShard,
                task->id);
        waiter->Done();
        return;
    }

    if (!SubmitMissShards(task, waiter, results, missCount)) { return; }

    UC_DEBUG("Mooncake task({}) dispatch shards({}, miss={}), wait={:.3f}ms, cost={:.3f}ms.",
             task->id, nShard, missCount, (tpWait - tp) * 1e3, (NowTime::Now() - tpWait) * 1e3);
}

std::vector<int> LoadQueue::TryMooncakeLoad(TaskPtr task)
{
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
    return realClient_->batch_get_into_multi_buffers(keys, allBuffers, allSizes, false);
}

bool LoadQueue::SubmitMissShards(TaskPtr task, WaiterPtr waiter, const std::vector<int>& results,
                                 size_t missCount)
{
    // MLA shared path: delegate miss shards to ShareLoadQueue
    if (shareLoadQ_) {
        auto missTask = std::make_shared<TransTask>();
        missTask->id = task->id;
        missTask->type = TaskType::LOAD;
        missTask->brief = task->brief;
        for (size_t i = 0; i < task->shards.size(); i++) {
            if (results[i] >= 0) { continue; }
            missTask->shards.push_back(task->shards[i]);
        }
        shareLoadQ_->Submit(missTask, waiter);
        return true;
    }

    size_t pushed = 0;
    for (size_t i = 0; i < task->shards.size(); i++) {
        if (results[i] >= 0) { continue; }

        auto& shard = task->shards[i];
        auto buf = bufPool_->AcquireWithTimeout(std::chrono::milliseconds(3000));
        if (!buf) {
            UC_ERROR("Host buffer pool exhausted for key={}", shard.key);
            failureSet_->Insert(task->id);
            waiter->Done();
            return false;
        }

        Detail::TaskDesc backendTask;
        backendTask.brief = "Backend2Host";
        backendTask.push_back(Detail::Shard{shard.owner, shard.index, {buf.get()}});

        auto res = backend_->Load(std::move(backendTask));
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit load task({}) to backend.", res.Error(), task->id);
            failureSet_->Insert(task->id);
            waiter->Done();
            return false;
        }

        ShardTask shardTask;
        shardTask.taskHandle = task->id;
        shardTask.backendTaskHandle = res.Value();
        shardTask.hostBuf = std::move(buf);
        shardTask.shard = std::move(shard);
        ++pushed;
        shardTask.waiter = (pushed == missCount) ? waiter : nullptr;
        running_.Push(std::move(shardTask));
    }
    return true;
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    CopyStream stream;
    auto s = stream.Setup(deviceId_, streamNumber_);
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
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.waiter) { task.waiter->Done(); }
        return;
    }

    auto s = Status::OK();
    do {
        s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.taskHandle);
            break;
        }

        s = HostToDeviceScatterAsync(stream.NextStream(), task.hostBuf.get(),
                                     task.shard.addrs.data());
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D batch async for task({}).", s, task.taskHandle);
            break;
        }

        if (!task.waiter) {
            holder_.push_back(std::move(task));
            return;
        }

        s = stream.Synchronize();
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, task.taskHandle);
            break;
        }
    } while (0);

    if (s.Failure()) [[unlikely]] { failureSet_->Insert(task.taskHandle); }
    if (task.waiter) { task.waiter->Done(); }
}

Status LoadQueue::HostToDeviceScatterAsync(std::shared_ptr<Trans::Stream> stream, void* host,
                                           void** device)
{
    const auto number = tensorSizes_.size();
    for (size_t i = 0, offset = 0; i < number; i++) {
        auto pHost = (void*)(((int8_t*)host) + offset);
        auto pDevice = device[i];
        auto size = tensorSizes_[i];
        auto s = stream->HostToDeviceAsync(pHost, pDevice, size);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D({}) batch({}/{}) async.", s, size, i, number);
            return s;
        }
        offset += size;
    }
    return Status::OK();
}

}  // namespace UC::MooncakeStore
