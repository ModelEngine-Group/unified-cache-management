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
#include "load_queue.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <fmt/format.h>
#include <future>
#include "logger/logger.h"
#include "metrics_api.h"
#ifdef __linux__
#include "thread/cpu_affinity.h"
#endif
#include "yuanrong_helper.h"

namespace UC::YuanRongStore {

void LoadQueue::LoadStats::Record() const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("yuanrong_load_success_shards_total"),
                             static_cast<double>(yuanrongSuccess));
    UC::Metrics::UpdateStats(
        NAME_TO_METRIC_ID("yuanrong_lookup_miss_posix_load_success_shards_total"),
        static_cast<double>(lookupMissPosixSuccess));
    UC::Metrics::UpdateStats(
        NAME_TO_METRIC_ID("yuanrong_load_fallback_posix_load_success_shards_total"),
        static_cast<double>(loadFallbackPosixSuccess));
    const auto succeeded = yuanrongSuccess + lookupMissPosixSuccess + loadFallbackPosixSuccess;
    const auto failed = requested > succeeded ? requested - succeeded : 0;
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("yuanrong_load_failed_shards_total"),
                             static_cast<double>(failed));
}

LoadQueue::~LoadQueue() { Close(); }

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet,
                        std::shared_ptr<datasystem::HeteroClient> heteroClient,
                        std::shared_ptr<datasystem::KVClient> kvClient)
{
    config_ = config;
    failureSet_ = failureSet;
    heteroClient_ = std::move(heteroClient);
    kvClient_ = std::move(kvClient);
    backend_ = config.storeBackend;
    waiting_.Setup(config.waitingQueueDepth);
    running_.reserve(config.loadWorkerCount);
    for (size_t i = 0; i < config.loadWorkerCount; ++i) {
        auto queue = std::make_unique<SpscRingQueue<TaskPair>>();
        queue->Setup(config.waitingQueueDepth);
        running_.push_back(std::move(queue));
    }
    Status status = Status::OK();
    if (backend_ != nullptr) {
        status =
            hostBufferPool_.Setup(config.deviceId, static_cast<uint32_t>(config.hostBufferCount),
                                  config.objectSize, config.ioDirect);
        if (status.Failure()) { return status; }
        status = backfillQueue_.Setup(config, kvClient_);
        if (status.Failure()) { return status; }
    }
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    std::vector<std::promise<Status>> started(config.loadWorkerCount);
    std::vector<std::future<Status>> futures;
    futures.reserve(config.loadWorkerCount);
    workers_.reserve(config.loadWorkerCount);
    for (size_t i = 0; i < config.loadWorkerCount; ++i) {
        futures.push_back(started[i].get_future());
        workers_.emplace_back(&LoadQueue::WorkerStage, this, i, std::move(started[i]));
    }
    for (auto& future : futures) {
        status = future.get();
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

void LoadQueue::Close()
{
    if (stop_.exchange(true)) { return; }
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    for (auto& worker : workers_) {
        if (worker.joinable()) { worker.join(); }
    }
    backfillQueue_.Close();
    TaskPair pair;
    while (waiting_.TryPop(pair)) {
        failureSet_->Insert(pair.first->id);
        pair.second->Done();
    }
    for (auto& queue : running_) {
        while (queue->TryPop(pair)) {
            failureSet_->Insert(pair.first->id);
            pair.second->Done();
        }
    }
}

void LoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    std::lock_guard<std::mutex> lock(submitMutex_);
    if (waiting_.TryPush({task, waiter})) { return; }
    UC_ERROR("YuanRong load queue full, task({}) rejected.", task->id);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
#ifdef __linux__
    if (!config_.cpuAffinityCores.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(config_.cpuAffinityCores);
        if (s.Failure()) { UC_WARN("Failed({}) to set YuanRong load affinity.", s); }
    }
#endif
    waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
}

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    if (running_.empty()) {
        failureSet_->Insert(pair.first->id);
        pair.second->Done();
        return;
    }
    const auto workerCount = running_.size();
    for (size_t i = 0; !stop_.load(std::memory_order_acquire); ++i) {
        auto index = (nextWorker_ + i) % workerCount;
        if (running_[index]->TryPush(std::move(pair))) {
            nextWorker_ = (index + 1) % workerCount;
            return;
        }
        std::this_thread::yield();
    }
    failureSet_->Insert(pair.first->id);
    pair.second->Done();
}

void LoadQueue::WorkerStage(size_t workerIndex, std::promise<Status> started)
{
    CopyStream stream;
    auto status = stream.Setup(config_.deviceId, config_.h2dStreamCount);
    started.set_value(status);
    if (status.Failure()) { return; }
#ifdef __linux__
    if (!config_.cpuAffinityCores.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(config_.cpuAffinityCores);
        if (s.Failure()) { UC_WARN("Failed({}) to set YuanRong load affinity.", s); }
    }
#endif
    running_[workerIndex]->ConsumerLoop(stop_, &LoadQueue::RunOne, this, stream);
}

void LoadQueue::RunOne(CopyStream& stream, TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (!failureSet_->Contains(task->id)) {
        LoadStats stats{task->desc.size()};
        auto s = LoadOne(stream, task, stats);
        stats.Record();
        if (s.Failure()) {
            UC_ERROR("YuanRong load task({}) failed: {}.", task->id, s);
            failureSet_->Insert(task->id);
        }
    }
    waiter->Done();
}

Status LoadQueue::LoadOne(CopyStream& stream, TaskPtr task, LoadStats& stats)
{
    auto taskStart = NowTime::Now();
    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto s = BuildKeysAndBlobs(config_, task->desc, keys, blobLists);
    if (s.Failure()) { return s; }
    if (keys.empty()) { return Status::OK(); }

    if (backend_ == nullptr) {
        return LoadThenRecover(stream, task, keys, blobLists, taskStart, stats);
    }

    auto existStart = NowTime::Now();
    std::vector<bool> exists;
    auto existStatus = heteroClient_->Exist(keys, exists);
    auto existenceEnd = NowTime::Now();
    if (existStatus.IsError() || exists.size() != keys.size()) {
        UC_WARN(
            "YuanRong load task({}) Exist unusable (result={}/{}), first_key={}, "
            "exist_cost={:.3f}ms, status={}; falling back to MGetH2D-first load.",
            task->id, exists.size(), keys.size(), keys.empty() ? "" : keys.front(),
            (existenceEnd - existStart) * 1e3, existStatus.ToString());
        return LoadThenRecover(stream, task, keys, blobLists, taskStart, stats);
    }

    std::vector<size_t> hitIndexes;
    std::vector<size_t> missIndexes;
    SplitByExistence(exists, hitIndexes, missIndexes);
    UC_DEBUG(
        "YuanRong load task({}) Exist keys={}, hit={}, miss={}, exist cost={:.3f}ms, status={}.",
        task->id, keys.size(), hitIndexes.size(), missIndexes.size(),
        (existenceEnd - existStart) * 1e3, existStatus.ToString());

    if (hitIndexes.empty()) {
        return RecoverFromBackend(stream, task, keys, blobLists, missIndexes,
                                  RecoverySource::LOOKUP_MISS, stats);
    }
    if (missIndexes.empty()) {
        return LoadThenRecover(stream, task, keys, blobLists, taskStart, stats);
    }

    std::vector<std::string> hitKeys;
    std::vector<datasystem::DeviceBlobList> hitBlobLists;
    hitKeys.reserve(hitIndexes.size());
    hitBlobLists.reserve(hitIndexes.size());
    for (auto index : hitIndexes) {
        hitKeys.push_back(keys[index]);
        hitBlobLists.push_back(blobLists[index]);
    }

    std::shared_future<datasystem::AsyncResult> getFuture;
    auto getStart = NowTime::Now();
    try {
        getFuture = heteroClient_->AsyncMGetH2D(hitKeys, hitBlobLists, 0);
    } catch (const std::exception& e) {
        UC_WARN("Failed({}) to start YuanRong AsyncMGetH2D; using sync mode.", e.what());
    }

    auto backendStart = NowTime::Now();
    auto backendStatus = RecoverFromBackend(stream, task, keys, blobLists, missIndexes,
                                            RecoverySource::LOOKUP_MISS, stats);
    auto backendEnd = NowTime::Now();

    std::vector<std::string> failedHitKeys;
    bool getFailed = false;
    std::string getStatusText;
    auto getWaitStart = NowTime::Now();
    if (getFuture.valid()) {
        try {
            auto result = getFuture.get();
            failedHitKeys = std::move(result.failedList);
            getFailed = result.status.IsError();
            getStatusText = result.status.ToString();
        } catch (const std::exception& e) {
            getFailed = true;
            getStatusText = fmt::format("AsyncMGetH2D future failed: {}", e.what());
        }
    } else {
        constexpr int32_t mgetTimeoutMs = 0;
        auto status = heteroClient_->MGetH2D(hitKeys, hitBlobLists, failedHitKeys, mgetTimeoutMs);
        getFailed = status.IsError();
        getStatusText = status.ToString();
    }
    auto getEnd = NowTime::Now();
    auto failedHitSubsetIndexes = FilterH2dFailedIndexes(hitKeys, failedHitKeys, getFailed);
    std::vector<size_t> h2dMissIndexes;
    h2dMissIndexes.reserve(failedHitSubsetIndexes.size());
    for (auto index : failedHitSubsetIndexes) { h2dMissIndexes.push_back(hitIndexes[index]); }
    stats.yuanrongSuccess += hitIndexes.size() - h2dMissIndexes.size();
    auto firstFailure = std::move(backendStatus);

    if (!h2dMissIndexes.empty()) {
        auto status = RecoverFromBackend(stream, task, keys, blobLists, h2dMissIndexes,
                                         RecoverySource::LOAD_FALLBACK, stats);
        if (firstFailure.Success() && status.Failure()) { firstFailure = std::move(status); }
    }
    UC_DEBUG(
        "YuanRong load task({}) finished, mode=parallel, h2d_keys={}, h2d_mb={:.3f}, "
        "posix_keys={}, posix_mb={:.3f}, h2d_fallback_keys={}, h2d_window={:.3f}ms, "
        "join_wait={:.3f}ms, posix_load={:.3f}ms, total={:.3f}ms, h2d_status={}, "
        "status={}.",
        task->id, hitIndexes.size(),
        static_cast<double>(config_.objectSize) * hitIndexes.size() / (1024.0 * 1024.0),
        missIndexes.size(),
        static_cast<double>(config_.objectSize) * missIndexes.size() / (1024.0 * 1024.0),
        h2dMissIndexes.size(), (getEnd - getStart) * 1e3, (getEnd - getWaitStart) * 1e3,
        (backendEnd - backendStart) * 1e3, (NowTime::Now() - taskStart) * 1e3, getStatusText,
        firstFailure.ToString());
    return firstFailure;
}

Status LoadQueue::LoadThenRecover(CopyStream& stream, TaskPtr task,
                                  const std::vector<std::string>& keys,
                                  const std::vector<datasystem::DeviceBlobList>& blobLists,
                                  double taskStart, LoadStats& stats)
{
    std::vector<std::string> failedKeys;
    constexpr int32_t mgetTimeoutMs = 0;
    auto getStart = NowTime::Now();
    auto rc = heteroClient_->MGetH2D(keys, blobLists, failedKeys, mgetTimeoutMs);
    auto getEnd = NowTime::Now();
    auto missIndexes = FilterH2dFailedIndexes(keys, failedKeys, rc.IsError());
    stats.yuanrongSuccess += keys.size() - missIndexes.size();
    const auto totalMb = static_cast<double>(config_.objectSize) * keys.size() / (1024.0 * 1024.0);
    if (!missIndexes.empty() && backend_ == nullptr) {
        return Status::Error(
            fmt::format("YuanRong MGetH2D miss({}/{}) and no backend is "
                        "configured, first miss key({}): {}",
                        missIndexes.size(), keys.size(), keys[missIndexes.front()], rc.ToString()));
    }

    auto loadEnd = getEnd;
    auto recoverStatus = Status::OK();
    double recoverCostMs = 0;
    if (!missIndexes.empty()) {
        auto recoverStart = NowTime::Now();
        recoverStatus = RecoverFromBackend(stream, task, keys, blobLists, missIndexes,
                                           RecoverySource::LOAD_FALLBACK, stats);
        loadEnd = NowTime::Now();
        recoverCostMs = (loadEnd - recoverStart) * 1e3;
    }
    UC_DEBUG(
        "YuanRong load task({}) finished, mode=mget_first, h2d_keys={}, h2d_mb={:.3f}, "
        "posix_keys={}, h2d_load={:.3f}ms, posix_load={:.3f}ms, total={:.3f}ms, "
        "h2d_status={}, status={}.",
        task->id, keys.size(), totalMb, missIndexes.size(), (getEnd - getStart) * 1e3,
        recoverCostMs, (loadEnd - taskStart) * 1e3, rc.ToString(), recoverStatus.ToString());
    return recoverStatus;
}

Status LoadQueue::RecoverFromBackend(CopyStream& stream, TaskPtr task,
                                     const std::vector<std::string>& keys,
                                     const std::vector<datasystem::DeviceBlobList>& blobLists,
                                     const std::vector<size_t>& missIndexes, RecoverySource source,
                                     LoadStats& stats)
{
    auto ranges = RecoveryBatchRanges(missIndexes.size(), config_.recoveryBatchSize);
    if (ranges.empty()) { return Status::OK(); }

    auto prepare = [this, task, &keys, &missIndexes](const auto& range) {
        return PrepareHostBatch(task, keys, missIndexes, range.first, range.second);
    };

    auto current = prepare(ranges.front());
    Status firstFailure = Status::OK();
    for (size_t i = 0; i < ranges.size(); ++i) {
        std::future<HostBatch> next;
        bool prepareNextSynchronously = false;
        if (i + 1 < ranges.size()) {
            auto range = ranges[i + 1];
            try {
                next = std::async(std::launch::async, [prepare, range] { return prepare(range); });
            } catch (const std::exception& e) {
                UC_WARN("Failed({}) to start YuanRong host load lookahead, using synchronous mode.",
                        e.what());
                prepareNextSynchronously = true;
            }
        }

        UC_DEBUG("YuanRong host load task({}) finalizing batch({}/{},{} blocks).", task->id, i + 1,
                 ranges.size(), current.indexes.size());
        auto status = FinalizeHostBatch(stream, blobLists, current);
        if (status.Success()) {
            auto& success = source == RecoverySource::LOOKUP_MISS ? stats.lookupMissPosixSuccess
                                                                   : stats.loadFallbackPosixSuccess;
            success += current.indexes.size();
        }
        if (firstFailure.Success() && status.Failure()) { firstFailure = std::move(status); }

        if (next.valid()) {
            try {
                current = next.get();
            } catch (const std::exception& e) {
                if (firstFailure.Success()) {
                    firstFailure = Status::Error(
                        fmt::format("YuanRong host load lookahead failed: {}", e.what()));
                }
                break;
            }
        } else if (prepareNextSynchronously) {
            current = prepare(ranges[i + 1]);
        }
    }
    return firstFailure;
}

LoadQueue::HostBatch LoadQueue::PrepareHostBatch(TaskPtr task, const std::vector<std::string>& keys,
                                                 const std::vector<size_t>& missIndexes,
                                                 size_t begin, size_t end)
{
    HostBatch batch;
    batch.indexes.reserve(end - begin);
    batch.keys.reserve(end - begin);
    batch.hostBuffers.reserve(end - begin);
    Detail::TaskDesc backendTask;
    backendTask.brief = "Posix2Host";
    auto prepareStart = NowTime::Now();
    for (size_t i = begin; i < end; ++i) {
        const auto index = missIndexes[i];
        auto host = hostBufferPool_.Acquire(std::chrono::milliseconds(config_.timeoutMs));
        if (!host) {
            batch.status = Status::Timeout();
            return batch;
        }
        const auto& sourceShard = task->desc[index];
        backendTask.push_back(Detail::Shard{sourceShard.owner, sourceShard.index, {host.get()}});
        batch.indexes.push_back(index);
        batch.keys.push_back(keys[index]);
        batch.hostBuffers.push_back(std::move(host));
    }

    auto backendResult = backend_->Load(std::move(backendTask));
    auto submitEnd = NowTime::Now();
    if (!backendResult) {
        batch.status = Status::Error(
            fmt::format("failed to submit Posix host load: {}", backendResult.Error().ToString()));
        return batch;
    }
    batch.backendTaskHandle = backendResult.Value();
    UC_DEBUG(
        "YuanRong host load task({}) prepared batch({} blocks), submit backend task({}), "
        "cost {:.3f}ms.",
        task->id, batch.indexes.size(), batch.backendTaskHandle, (submitEnd - prepareStart) * 1e3);
    return batch;
}

Status LoadQueue::FinalizeHostBatch(CopyStream& stream,
                                    const std::vector<datasystem::DeviceBlobList>& blobLists,
                                    HostBatch& batch)
{
    if (batch.status.Failure()) { return batch.status; }
    auto waitStart = NowTime::Now();
    auto waitStatus = backend_->Wait(batch.backendTaskHandle);
    if (waitStatus.Failure()) {
        return Status::Error(
            fmt::format("failed to wait Posix host load: {}", waitStatus.ToString()));
    }
    auto h2dStart = NowTime::Now();

    Status copyStatus = Status::OK();
    for (size_t i = 0; i < batch.indexes.size(); ++i) {
        copyStatus = HostToDeviceScatterAsync(stream.NextStream(), batch.hostBuffers[i].get(),
                                              blobLists[batch.indexes[i]]);
        if (copyStatus.Failure()) { break; }
    }
    auto syncStatus = stream.Synchronize();
    auto h2dEnd = NowTime::Now();
    if (copyStatus.Failure()) { return copyStatus; }
    if (syncStatus.Failure()) { return syncStatus; }

    UC_DEBUG("YuanRong host batch({}) Posix wait {:.3f}ms, direct H2D {:.3f}ms.",
             batch.indexes.size(), (h2dStart - waitStart) * 1e3, (h2dEnd - h2dStart) * 1e3);

    backfillQueue_.Submit(BackfillTask{std::move(batch.keys), std::move(batch.hostBuffers)});
    UC_DEBUG("YuanRong host batch({}) async backfill submitted.", batch.indexes.size());
    return Status::OK();
}

Status LoadQueue::HostToDeviceScatterAsync(std::shared_ptr<Trans::Stream> stream, void* host,
                                           const datasystem::DeviceBlobList& blobList)
{
    if (!stream || host == nullptr) { return Status::InvalidParam("invalid direct H2D buffer"); }
    size_t offset = 0;
    for (const auto& blob : blobList.blobs) {
        if (blob.pointer == nullptr || offset + blob.size > config_.objectSize) {
            return Status::InvalidParam("invalid YuanRong direct H2D blob layout");
        }
        auto status = stream->HostToDeviceAsync(static_cast<uint8_t*>(host) + offset, blob.pointer,
                                                blob.size);
        if (status.Failure()) { return status; }
        offset += blob.size;
    }
    return offset == config_.objectSize ? Status::OK()
                                        : Status::InvalidParam("incomplete direct H2D blob layout");
}

}  // namespace UC::YuanRongStore
