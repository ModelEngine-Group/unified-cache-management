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
#include "dump_queue.h"
#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <fmt/format.h>
#include "logger/logger.h"
#ifdef __linux__
#include "thread/cpu_affinity.h"
#endif
#include "time/now_time.h"
#include "trans/device.h"
#include "yuanrong_helper.h"

namespace UC::YuanRongStore {

DumpQueue::~DumpQueue() { Close(); }

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet,
                        std::shared_ptr<datasystem::HeteroClient> heteroClient,
                        std::shared_ptr<datasystem::KVClient> kvClient)
{
    config_ = config;
    failureSet_ = failureSet;
    heteroClient_ = std::move(heteroClient);
    kvClient_ = std::move(kvClient);
    backend_ = config.storeBackend;
    ready_.Setup(config.waitingQueueDepth);
    persistence_.Setup(kPersistenceQueueDepth);
    closing_.store(false, std::memory_order_release);
    stopD2H_.store(false, std::memory_order_release);
    stopPersistence_.store(false, std::memory_order_release);
    pendingCount_.store(0, std::memory_order_release);

    Trans::Device validationDevice;
    auto validationStatus = validationDevice.Setup(config_.deviceId);
    if (validationStatus.Failure()) { return validationStatus; }

    try {
        std::promise<Status> d2hStarted;
        auto d2hStatus = d2hStarted.get_future();
        d2hWorker_ = std::thread{&DumpQueue::D2HStage, this, std::ref(d2hStarted)};
        auto status = d2hStatus.get();
        if (status.Failure()) {
            d2hWorker_.join();
            return status;
        }
        if (backend_ != nullptr) {
            persistenceWorker_ = std::thread{&DumpQueue::PersistenceStage, this};
        }

        prerequisitePool_ =
            std::make_unique<ThreadPool<DumpTaskContext, std::shared_ptr<WorkerContext>>>();
        auto prerequisiteStarted =
            prerequisitePool_
                ->SetWorkerInitFn([this](auto& worker) {
                    try {
                        worker = std::make_shared<WorkerContext>();
                        worker->status = worker->device.Setup(config_.deviceId);
                    } catch (const std::exception& e) {
                        worker.reset();
                        UC_ERROR("Failed({}) to initialize YuanRong prerequisite worker.",
                                 e.what());
                    }
                    return true;
                })
                .SetWorkerFn(
                    [this](auto& context, const auto& worker) { RunPrerequisite(context, worker); })
                .SetNWorker(config_.dumpPrerequisiteWorkerCount)
                .SetCpuAffinity(config_.cpuAffinityCores)
                .Run();
        if (!prerequisiteStarted) {
            prerequisitePool_.reset();
            stopD2H_.store(true, std::memory_order_release);
            d2hWorker_.join();
            stopPersistence_.store(true, std::memory_order_release);
            if (persistenceWorker_.joinable()) { persistenceWorker_.join(); }
            return Status::Error("failed to start YuanRong prerequisite worker pool");
        }
    } catch (const std::exception& e) {
        prerequisitePool_.reset();
        stopD2H_.store(true, std::memory_order_release);
        if (d2hWorker_.joinable()) { d2hWorker_.join(); }
        stopPersistence_.store(true, std::memory_order_release);
        if (persistenceWorker_.joinable()) { persistenceWorker_.join(); }
        return Status::Error(fmt::format("failed to start YuanRong dump pipeline: {}", e.what()));
    }
    return Status::OK();
}

void DumpQueue::Close()
{
    {
        std::lock_guard<std::mutex> lock(submitMutex_);
        if (closing_.exchange(true)) { return; }
    }

    auto cancel = [this](DumpTaskContext& context) {
        Finish(context, Status::Error("YuanRong dump pipeline is closing"));
    };
    auto all = [](const DumpTaskContext&) { return true; };

    if (prerequisitePool_) {
        prerequisitePool_->TraverseWaitQueue(all, cancel, {});
        // Running prerequisite tasks are allowed to hand off to D2H before this returns.
        prerequisitePool_.reset();
    }
    stopD2H_.store(true, std::memory_order_release);
    if (d2hWorker_.joinable()) { d2hWorker_.join(); }
    DumpTaskContext readyContext;
    while (ready_.TryPop(readyContext)) { cancel(readyContext); }

    stopPersistence_.store(true, std::memory_order_release);
    if (persistenceWorker_.joinable()) { persistenceWorker_.join(); }
}

void DumpQueue::D2HStage(std::promise<Status>& started)
{
    Trans::Device device;
    auto status = device.Setup(config_.deviceId);
    started.set_value(status);
    if (status.Failure()) { return; }
#ifdef __linux__
    if (!config_.cpuAffinityCores.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(config_.cpuAffinityCores);
        if (s.Failure()) { UC_WARN("Failed({}) to set YuanRong D2H affinity.", s); }
    }
#endif
    ready_.ConsumerLoop(stopD2H_, &DumpQueue::RunD2H, this);
}

void DumpQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    {
        std::lock_guard<std::mutex> lock(submitMutex_);
        if (!closing_.load(std::memory_order_acquire) && prerequisitePool_ &&
            pendingCount_.load(std::memory_order_relaxed) < config_.waitingQueueDepth) {
            pendingCount_.fetch_add(1, std::memory_order_relaxed);
            prerequisitePool_->Push(DumpTaskContext{task, waiter, NowTime::Now(), 0.0, 0.0});
            return;
        }
    }
    UC_ERROR("YuanRong dump pipeline unavailable or full, task({}) rejected, pending={}/{}.",
             task->id, pendingCount_.load(std::memory_order_relaxed), config_.waitingQueueDepth);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::PersistenceStage()
{
#ifdef __linux__
    if (!config_.cpuAffinityCores.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(config_.cpuAffinityCores);
        if (s.Failure()) { UC_WARN("Failed({}) to set YuanRong persistence affinity.", s); }
    }
#endif
    size_t inflightBytes = 0;
    std::list<PersistenceContext> inflight;
    while (true) {
        PersistenceTask task;
        const bool hasTask = persistence_.TryPop(task);
        if (hasTask) { Persist(task, inflightBytes, inflight); }
        ReclaimCompletedInflight(inflightBytes, inflight);

        // The producer has already stopped before stopPersistence_ is set, so an
        // empty pop here means the persistence queue has been fully drained.
        if (stopPersistence_.load(std::memory_order_acquire) && !hasTask && inflight.empty()) {
            break;
        }
        if (!hasTask) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }
}

void DumpQueue::RunPrerequisite(DumpTaskContext& context,
                                const std::shared_ptr<WorkerContext>& worker)
{
    if (!worker) {
        Finish(context, Status::OutOfMemory());
        return;
    }
    if (worker->status.Failure()) {
        Finish(context, worker->status);
        return;
    }
    if (failureSet_->Contains(context.task->id)) {
        Finish(context, Status::Error("YuanRong dump task was already marked failed"));
        return;
    }

    context.prerequisiteStart = NowTime::Now();
    auto status = Status::OK();
    if (context.task->desc.prerequisiteHandle != 0) {
        auto event = reinterpret_cast<aclrtEvent>(context.task->desc.prerequisiteHandle);
        UC_DEBUG("YuanRong dump task({}) waiting for prerequisite event({}).", context.task->id,
                 context.task->desc.prerequisiteHandle);
        auto ret = aclrtSynchronizeEvent(event);
        if (ret != ACL_SUCCESS) {
            status = Status{ret, fmt::format("aclrtSynchronizeEvent failed, ret={}", ret)};
        } else {
            UC_DEBUG("YuanRong dump task({}) prerequisite event({}) completed.", context.task->id,
                     context.task->desc.prerequisiteHandle);
        }
    }
    context.prerequisiteEnd = NowTime::Now();
    if (status.Failure()) {
        Finish(context, status);
        return;
    }
    std::lock_guard<std::mutex> lock(readySubmitMutex_);
    ready_.Push(std::move(context));
}

void DumpQueue::RunD2H(DumpTaskContext&& context)
{
    auto status = Status::OK();
    if (!failureSet_->Contains(context.task->id)) {
        const auto prerequisiteQueueWaitMs = (context.prerequisiteStart - context.submitTime) * 1e3;
        const auto prerequisiteMs = (context.prerequisiteEnd - context.prerequisiteStart) * 1e3;
        const auto d2hQueueWaitMs = (NowTime::Now() - context.prerequisiteEnd) * 1e3;
        status = DumpReadyTask(context.task, prerequisiteQueueWaitMs, prerequisiteMs,
                               d2hQueueWaitMs, context.submitTime);
    }
    Finish(context, status);
}

void DumpQueue::Finish(DumpTaskContext& context, const Status& status)
{
    if (!context.waiter) { return; }
    if (status.Failure()) {
        UC_ERROR("YuanRong dump task({}) failed: {}.", context.task->id, status);
        failureSet_->Insert(context.task->id);
    }
    auto previous = pendingCount_.fetch_sub(1, std::memory_order_relaxed);
    if (previous == 0) {
        pendingCount_.store(0, std::memory_order_relaxed);
        UC_ERROR("YuanRong dump pending task counter underflow for task({}).", context.task->id);
    }
    auto waiter = std::move(context.waiter);
    waiter->Done();
}

Status DumpQueue::DumpReadyTask(TaskPtr task, double prerequisiteQueueWaitMs, double prerequisiteMs,
                                double d2hQueueWaitMs, double pipelineStart)
{
    std::vector<std::string> keys;
    std::vector<datasystem::DeviceBlobList> blobLists;
    auto s = BuildKeysAndBlobs(config_, task->desc, keys, blobLists);
    if (s.Failure()) { return s; }
    if (keys.empty()) { return Status::OK(); }
    DeduplicateYuanRongObjects(keys, blobLists, &task->desc);
    const auto totalMb = static_cast<double>(config_.objectSize) * keys.size() / (1024.0 * 1024.0);

    datasystem::SetParam setParam;
    setParam.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    setParam.existence = datasystem::ExistenceOpt::NONE;
    setParam.cacheType = datasystem::CacheType::MEMORY;
    std::vector<std::string> localSetKeys;
    auto d2hStart = NowTime::Now();
    auto dumpStatus = heteroClient_->MSetD2H(keys, blobLists, setParam, &localSetKeys);
    auto publishEnd = NowTime::Now();
    UC_DEBUG(
        "YuanRong dump task({}) MSetD2H keys={}, local_set={}, total_mb={:.3f}, "
        "prereq_queue_wait={:.3f}ms, prereq_event_wait={:.3f}ms, ready_queue_wait={:.3f}ms, "
        "d2h_transfer={:.3f}ms, status={}.",
        task->id, keys.size(), localSetKeys.size(), totalMb, prerequisiteQueueWaitMs,
        prerequisiteMs, d2hQueueWaitMs, (publishEnd - d2hStart) * 1e3, dumpStatus.ToString());
    if (localSetKeys.empty()) {
        if (dumpStatus.IsError()) {
            return Status::Error(
                fmt::format("YuanRong MSetD2H failed without any confirmed local key: {}",
                            dumpStatus.ToString()));
        }
        UC_DEBUG("YuanRong dump task({}) has no newly published local key; skipping Posix.",
                 task->id);
        return Status::OK();
    }

    s = FilterKeysByLocalSetKeys(localSetKeys, keys, task->desc);
    if (s.Failure()) { return s; }
    if (dumpStatus.IsError()) {
        UC_WARN(
            "YuanRong dump task({}) MSetD2H partially failed, success local set keys: {}, failed "
            "reason: {}",
            task->id, keys.size(), dumpStatus.ToString());
    }
    if (backend_ == nullptr) {
        UC_DEBUG("YuanRong dump task({}) finished without backend, local_set={}, total={:.3f}ms.",
                 task->id, keys.size(), (NowTime::Now() - pipelineStart) * 1e3);
        return Status::OK();
    }

    PersistenceTask persistenceTask{task->id, std::move(keys), task->desc, NowTime::Now()};
    const auto persistenceKeys = persistenceTask.keys.size();
    if (!persistence_.TryPush(std::move(persistenceTask))) {
        UC_WARN(
            "YuanRong dump task({}) skipping Posix persistence for {} keys: "
            "background queue is full.",
            task->id, persistenceKeys);
    } else {
        UC_DEBUG(
            "YuanRong dump task({}) queued Posix persistence for {} keys, foreground "
            "total={:.3f}ms.",
            task->id, persistenceKeys, (NowTime::Now() - pipelineStart) * 1e3);
    }
    return Status::OK();
}

void DumpQueue::Persist(const PersistenceTask& task, size_t& inflightBytes,
                        std::list<PersistenceContext>& inflight)
{
    const size_t maxInflightBytes = config_.posixMaxInflightGb << 30;
    for (size_t begin = 0; begin < task.keys.size(); begin += config_.posixDumpBatchSize) {
        const auto end = std::min(begin + config_.posixDumpBatchSize, task.keys.size());
        const size_t batchBytes = config_.objectSize * (end - begin);
        while (inflightBytes > maxInflightBytes - batchBytes) {
            ReclaimCompletedInflight(inflightBytes, inflight);
            if (inflightBytes <= maxInflightBytes - batchBytes) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto status = PersistBatch(task, begin, end, inflightBytes, inflight);
        if (status.Failure()) {
            UC_WARN("Background Posix persistence for YuanRong task({}) skipped batch [{},{}): {}.",
                    task.ownerTaskId, begin, end, status);
        }
        ReclaimCompletedInflight(inflightBytes, inflight);
    }
}

Status DumpQueue::PersistBatch(const PersistenceTask& task, size_t begin, size_t end,
                               size_t& inflightBytes, std::list<PersistenceContext>& inflight)
{
    if (begin >= end || end > task.keys.size() || task.desc.size() != task.keys.size()) {
        return Status::InvalidParam("invalid YuanRong Posix persistence batch");
    }
    const size_t batchBytes = config_.objectSize * (end - begin);
    const size_t maxInflightBytes = config_.posixMaxInflightGb << 30;
    if (batchBytes > maxInflightBytes || inflightBytes > maxInflightBytes - batchBytes) {
        return Status::Error(fmt::format(
            "inflight byte limit exceeded, batch={} bytes, inflight={} bytes, limit={} bytes",
            batchBytes, inflightBytes, maxInflightBytes));
    }
    inflightBytes += batchBytes;
    auto releaseReservedBytes = [&inflightBytes](size_t bytes) {
        inflightBytes = inflightBytes >= bytes ? inflightBytes - bytes : 0;
    };

    std::vector<std::string> keys(task.keys.begin() + begin, task.keys.begin() + end);
    auto persistenceStart = NowTime::Now();
    std::vector<datasystem::MetaInfo> metaInfos;
    std::vector<std::string> metaFailedKeys;
    auto metaStart = NowTime::Now();
    auto metaStatus = heteroClient_->GetMetaInfo(keys, false, metaInfos, metaFailedKeys);
    auto metaEnd = NowTime::Now();
    if (metaStatus.IsError() || metaInfos.size() != keys.size() || !metaFailedKeys.empty()) {
        releaseReservedBytes(batchBytes);
        return Status::Error(fmt::format(
            "GetMetaInfo failed: {}, failed keys({}), meta count({}), key count({})",
            metaStatus.ToString(), metaFailedKeys.size(), metaInfos.size(), keys.size()));
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        auto status = ValidateYuanRongBlobSizes(keys[i], metaInfos[i], config_.tensorSizes);
        if (status.Failure()) {
            releaseReservedBytes(batchBytes);
            return status;
        }
    }

    std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>> buffers;
    auto kvGetStart = NowTime::Now();
    auto getStatus = kvClient_->Get(keys, buffers, 0);
    auto kvGetEnd = NowTime::Now();
    if (getStatus.IsError() || buffers.size() != keys.size()) {
        releaseReservedBytes(batchBytes);
        return Status::Error(fmt::format("YuanRong Get failed: {}", getStatus.ToString()));
    }

    std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>> lockedBuffers;
    lockedBuffers.reserve(buffers.size());
    Detail::TaskDesc backendTask;
    backendTask.brief = "YuanRong2Posix";
    backendTask.reserve(buffers.size());
    size_t unavailableBufferCount = 0;
    std::string firstUnavailableBufferKey;
    for (size_t i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers[i];
        if (!buffer) {
            if (unavailableBufferCount == 0) { firstUnavailableBufferKey = keys[i]; }
            ++unavailableBufferCount;
            continue;
        }
        auto latchStatus = buffer->RLatch();
        if (latchStatus.IsError()) {
            if (unavailableBufferCount == 0) { firstUnavailableBufferKey = keys[i]; }
            ++unavailableBufferCount;
            UC_WARN("Failed to latch YuanRong buffer for key({}), skipping it: {}.", keys[i],
                    latchStatus.ToString());
            continue;
        }
        const auto* bufferAddress = buffer->ImmutableData();
        const void* payloadAddress = nullptr;
        auto payloadStatus =
            GetYuanRongPayloadAddress(keys[i], bufferAddress, buffer->GetSize(), metaInfos[i],
                                      config_.tensorSizes, config_.memoryAlignment, payloadAddress);
        if (payloadStatus.Failure()) {
            (void)buffer->UnRLatch();
            ReleaseBuffers(lockedBuffers);
            releaseReservedBytes(batchBytes);
            return payloadStatus;
        }
        if (config_.ioDirect) {
            auto directStatus = ValidateYuanRongDirectIoPayload(
                keys[i], payloadAddress, config_.objectSize, config_.memoryAlignment);
            if (directStatus.Failure()) {
                (void)buffer->UnRLatch();
                ReleaseBuffers(lockedBuffers);
                releaseReservedBytes(batchBytes);
                return directStatus;
            }
        }
        const auto& source = task.desc[begin + i];
        backendTask.push_back(
            Detail::Shard{source.owner, source.index, {const_cast<void*>(payloadAddress)}});
        lockedBuffers.push_back(std::move(buffer));
    }
    if (unavailableBufferCount != 0) {
        UC_WARN(
            "Background YuanRong Get returned unavailable buffers({}/{}), first unavailable "
            "key({}); persisting readable buffers only.",
            unavailableBufferCount, buffers.size(), firstUnavailableBufferKey);
    }
    if (backendTask.empty()) {
        releaseReservedBytes(batchBytes);
        return Status::Error("YuanRong Get returned no readable buffer for Posix dump");
    }

    const size_t persistedBytes = config_.objectSize * backendTask.size();
    releaseReservedBytes(batchBytes - persistedBytes);
    const auto persistedKeys = backendTask.size();
    auto backendResult = backend_->Dump(std::move(backendTask));
    auto backendSubmitEnd = NowTime::Now();
    if (!backendResult) {
        ReleaseBuffers(lockedBuffers);
        releaseReservedBytes(persistedBytes);
        return Status::Error(
            fmt::format("failed to submit Posix dump: {}", backendResult.Error().ToString()));
    }
    inflight.push_back(PersistenceContext{task.ownerTaskId, backendResult.Value(), persistedBytes,
                                          std::move(lockedBuffers)});
    UC_DEBUG(
        "Background Posix persistence for YuanRong task({}) submitted keys={}, total_mb={:.3f}, "
        "queue_wait={:.3f}ms, get_meta={:.3f}ms, kv_get={:.3f}ms, prepare_submit={:.3f}ms, "
        "inflight_mb={:.3f}.",
        task.ownerTaskId, persistedKeys, static_cast<double>(persistedBytes) / (1024.0 * 1024.0),
        (persistenceStart - task.enqueueTime) * 1e3, (metaEnd - metaStart) * 1e3,
        (kvGetEnd - kvGetStart) * 1e3, (backendSubmitEnd - kvGetEnd) * 1e3,
        static_cast<double>(inflightBytes) / (1024.0 * 1024.0));
    return Status::OK();
}

void DumpQueue::ReclaimCompletedInflight(size_t& inflightBytes,
                                         std::list<PersistenceContext>& inflight)
{
    for (auto it = inflight.begin(); it != inflight.end();) {
        auto result = backend_->Check(it->backendTaskId);
        if (result && !result.Value()) {
            ++it;
            continue;
        }

        auto status = backend_->Wait(it->backendTaskId);
        if (!result && status.Success()) {
            status = Status::Error(
                fmt::format("Posix Check failed before Wait: {}", result.Error().ToString()));
        }
        if (status.Failure()) {
            UC_WARN("Background Posix dump({}) for YuanRong task({}) failed: {}.",
                    it->backendTaskId, it->ownerTaskId, status);
        } else {
            UC_DEBUG("Background Posix dump({}) for YuanRong task({}) finished.", it->backendTaskId,
                     it->ownerTaskId);
        }
        ReleasePersistenceContext(*it, inflightBytes);
        it = inflight.erase(it);
    }
}

void DumpQueue::ReleasePersistenceContext(PersistenceContext& context, size_t& inflightBytes)
{
    ReleaseBuffers(context.buffers);
    if (inflightBytes < context.bytes) {
        UC_ERROR(
            "YuanRong Posix persistence inflight byte counter underflow, inflight={}, "
            "release={}.",
            inflightBytes, context.bytes);
        inflightBytes = 0;
    } else {
        inflightBytes -= context.bytes;
    }
}

void DumpQueue::ReleaseBuffers(
    std::vector<datasystem::Optional<datasystem::ReadOnlyBuffer>>& buffers)
{
    for (auto& buffer : buffers) {
        if (!buffer) { continue; }
        auto status = buffer->UnRLatch();
        if (status.IsError()) {
            UC_WARN("Failed to unlock YuanRong read buffer: {}.", status.ToString());
        }
    }
    buffers.clear();
}

}  // namespace UC::YuanRongStore
