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
#include "share_load_queue.h"
#include <numeric>
#include "logger/logger.h"
#include "metrics_api.h"
#include "time/now_time.h"
#include "trans/device.h"

namespace UC::MooncakeStore {

ShareLoadQueue::~ShareLoadQueue() { Close(); }

void ShareLoadQueue::Close()
{
    {
        std::lock_guard<std::mutex> lg(mutex_);
        stop_ = true;
        cv_.notify_all();
    }
    for (auto& w : threads_) {
        if (w.joinable()) { w.join(); }
    }
}

Status ShareLoadQueue::Setup(const Config& config, TaskIdSet* failureSet, StoreV1* backend)
{
    failureSet_ = failureSet;
    backend_ = backend;
    deviceId_ = config.deviceId;
    tensorSizes_.assign(config.tensorSizeList.begin(), config.tensorSizeList.end());
    streamNumber_ = config.streamNumber;

    size_t blockSize = 0;
    for (auto sz : config.tensorSizeList) { blockSize += sz; }
    auto s = buffer_.Setup(blockSize, config.shareBufferNumber, config.uniqueId);
    if (s.Failure()) {
        UC_ERROR("Failed({}) to setup share buffer.", s);
        return s;
    }

    std::list<std::promise<Status>> start(streamNumber_);
    std::list<std::future<Status>> fut;
    for (auto& p : start) {
        fut.push_back(p.get_future());
        threads_.emplace_back([&] { WorkerLoop(p); });
    }
    Status status = Status::OK();
    for (auto& f : fut) {
        if (status.Failure()) { break; }
        status = f.get();
    }
    if (status.Success()) {
        UC_INFO("ShareLoadQueue setup ok, streamNumber={}, bufferNumber={}, blockSize={}.",
                streamNumber_, config.shareBufferNumber, blockSize);
    }
    return status;
}

void ShareLoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }

    const auto nShard = task->shards.size();
    std::list<BlockTask> blkTasks;

    // Use a shared counter to track completion of all shards.
    // When all shards are done, call waiter->Done() exactly once.
    auto remaining = std::make_shared<std::atomic<size_t>>(nShard);

    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->shards[i];
        auto reader = buffer_.MakeReader(shard.key);
        if (!reader) [[unlikely]] {
            UC_ERROR("Failed to make reader for key={}", shard.key);
            failureSet_->Insert(task->id);
            waiter->Done();
            return;
        }

        BlockTask blockTask;
        blockTask.taskHandle = task->id;
        blockTask.reader = std::move(reader);
        blockTask.shard = shard;
        blockTask.backendTaskHandle = 0;
        blockTask.done = [remaining, waiter](bool) {
            if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) { waiter->Done(); }
        };
        blkTasks.push_back(std::move(blockTask));
    }

    std::lock_guard<std::mutex> lg(mutex_);
    waitList_.splice(waitList_.end(), blkTasks);
    cv_.notify_all();
}

void ShareLoadQueue::WorkerLoop(std::promise<Status>& status)
{
    Trans::Device device;
    auto s = device.Setup(deviceId_);
    if (s.Failure()) {
        UC_ERROR("Failed({}) to set context on device({}).", s.ToString(), deviceId_);
        status.set_value(Status::Error());
        return;
    }
    auto stream = device.MakeStream();
    if (!stream) {
        UC_ERROR("Failed to create stream on device({}).", deviceId_);
        status.set_value(Status::Error());
        return;
    }
    status.set_value(Status::OK());
    while (!stop_) { Worker(*stream); }
}

void ShareLoadQueue::Worker(Trans::Stream& stream)
{
    std::unique_lock<std::mutex> ul{mutex_};
    if (loadList_.empty() && waitList_.empty()) {
        cv_.wait(ul, [this] { return stop_ || !loadList_.empty() || !waitList_.empty(); });
    }
    if (stop_) { return; }

    for (auto iter = loadList_.begin(); iter != loadList_.end(); iter++) {
        // For tasks with pending backend load, check if backend is done
        if (iter->backendTaskHandle != 0) {
            auto task = std::move(*iter);
            loadList_.erase(iter);
            ul.unlock();
            HandleBackendComplete(task, stream);
            return;
        }
        auto s = iter->reader->Ready4Read();
        if (s != Status::Retry()) {
            auto task = std::move(*iter);
            loadList_.erase(iter);
            ul.unlock();
            HandleReadyTask(s, task, stream);
            return;
        }
    }
    if (loadList_.size() >= streamNumber_) { return; }
    if (waitList_.empty()) { return; }
    auto task = std::move(waitList_.front());
    waitList_.pop_front();
    ul.unlock();
    HandleLoadTask(task, stream);
}

void ShareLoadQueue::HandleLoadTask(BlockTask& task, Trans::Stream& stream)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.done) { task.done(false); }
        return;
    }

    auto s = task.reader->Ready4Read();
    if (s == Status::Retry()) {
        std::lock_guard<std::mutex> lg{mutex_};
        loadList_.push_back(std::move(task));
        cv_.notify_one();
        return;
    }
    if (s == Status::OK()) {
        HandleReadyTask(s, task, stream);
        return;
    }
    if (s == Status::DuplicateKey()) {
        // This process is the owner — load data from backend into shared buffer
        if (!backend_) {
            UC_WARN("ShareLoadQueue: no backend for key={}, mark failed.", task.shard.key);
            task.reader->MarkFailed();
            failureSet_->Insert(task.taskHandle);
            if (task.done) { task.done(false); }
            return;
        }

        Detail::TaskDesc backendTask;
        backendTask.brief = "Backend2ShareBuf";
        backendTask.push_back(
            Detail::Shard{task.shard.owner, task.shard.index, {task.reader->GetData()}});

        auto tpSubmit = NowTime::Now();
        auto res = backend_->Load(std::move(backendTask));
        if (!res) [[unlikely]] {
            UC_ERROR("Failed({}) to submit backend load for key={}.", res.Error(), task.shard.key);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_backend_load_submit_errors_total"),
                                     1.0);
            task.reader->MarkFailed();
            failureSet_->Insert(task.taskHandle);
            if (task.done) { task.done(false); }
            return;
        }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_load_backend_submit_duration_ms"),
                                 (NowTime::Now() - tpSubmit) * 1e3);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_load_backend_shards_total"), 1.0);

        task.backendTaskHandle = res.Value();
        // Put into load list to wait for backend completion
        std::lock_guard<std::mutex> lg{mutex_};
        loadList_.push_back(std::move(task));
        cv_.notify_one();
        return;
    }
    // Error or other failure
    task.reader->MarkFailed();
    failureSet_->Insert(task.taskHandle);
    if (task.done) { task.done(false); }
}

void ShareLoadQueue::HandleReadyTask(Status s, BlockTask& task, Trans::Stream& stream)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.done) { task.done(false); }
        return;
    }

    if (s.Failure()) {
        failureSet_->Insert(task.taskHandle);
        if (task.done) { task.done(false); }
        return;
    }

    // H2D scatter from shared buffer to device
    auto host = task.reader->GetData();
    auto hs = HostToDeviceScatterAsync(stream, host, task.shard.addrs.data());
    if (hs.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to H2D for key={}.", hs, task.shard.key);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_h2d_errors_total"), 1.0);
        failureSet_->Insert(task.taskHandle);
        if (task.done) { task.done(false); }
        return;
    }

    auto h2dSyncStart = NowTime::Now();
    auto ss = stream.Synchronized();
    auto h2dMs = (NowTime::Now() - h2dSyncStart) * 1e3;
    RecordH2dMetrics(h2dMs);
    if (ss.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to sync stream for key={}.", ss, task.shard.key);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_h2d_errors_total"), 1.0);
        failureSet_->Insert(task.taskHandle);
        if (task.done) { task.done(false); }
        return;
    }

    if (task.done) { task.done(true); }
}

void ShareLoadQueue::HandleBackendComplete(BlockTask& task, Trans::Stream& stream)
{
    if (failureSet_->Contains(task.taskHandle)) {
        task.reader->MarkFailed();
        if (task.done) { task.done(false); }
        return;
    }

    auto tpWait = NowTime::Now();
    auto ws = backend_->Wait(task.backendTaskHandle);
    auto tpReady = NowTime::Now();
    if (ws.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to wait backend for key={}.", ws, task.shard.key);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_backend_load_wait_errors_total"), 1.0);
        task.reader->MarkFailed();
        failureSet_->Insert(task.taskHandle);
        if (task.done) { task.done(false); }
        return;
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_backend_load_wait_duration_ms"),
                             (tpReady - tpWait) * 1e3);
    task.reader->MarkLoaded();
    task.backendTaskHandle = 0;

    // Now do H2D
    auto host = task.reader->GetData();
    auto hs = HostToDeviceScatterAsync(stream, host, task.shard.addrs.data());
    if (hs.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to H2D for key={}.", hs, task.shard.key);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_h2d_errors_total"), 1.0);
        failureSet_->Insert(task.taskHandle);
        if (task.done) { task.done(false); }
        return;
    }

    auto h2dSyncStart = NowTime::Now();
    auto ss = stream.Synchronized();
    auto h2dMs = (NowTime::Now() - h2dSyncStart) * 1e3;
    RecordH2dMetrics(h2dMs);
    if (ss.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to sync stream for key={}.", ss, task.shard.key);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_h2d_errors_total"), 1.0);
        failureSet_->Insert(task.taskHandle);
        if (task.done) { task.done(false); }
        return;
    }

    if (task.done) { task.done(true); }
}

Status ShareLoadQueue::HostToDeviceScatterAsync(Trans::Stream& stream, void* host, void** device)
{
    const auto number = tensorSizes_.size();
    for (size_t i = 0, offset = 0; i < number; i++) {
        auto pHost = (void*)(((int8_t*)host) + offset);
        auto pDevice = device[i];
        auto size = tensorSizes_[i];
        auto s = stream.HostToDeviceAsync(pHost, pDevice, size);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D({}) batch({}/{}) async.", s, size, i, number);
            return s;
        }
        offset += size;
    }
    return Status::OK();
}

size_t ShareLoadQueue::BlockBytes() const
{
    return std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), size_t{0});
}

void ShareLoadQueue::RecordH2dMetrics(double h2dMs) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_h2d_duration_ms"), h2dMs);
    auto bytes = static_cast<double>(BlockBytes());
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("mooncake_h2d_bytes_total"), bytes);
}

}  // namespace UC::MooncakeStore
