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
 */
#include "delegator_executor.h"
#include <acl/acl.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <exception>
#include <functional>
#include <limits>
#include <new>
#include <system_error>
#include "logger/logger.h"
#include "time/now_time.h"

namespace UC::Delegator {
namespace {

constexpr std::size_t kBufferAlignment = 16 * 1024;

struct TransferTiming {
    double backendStartTp{0.0};
    double d2dStartTp{0.0};
};

const char* OperationName(Operation operation) noexcept
{
    switch (operation) {
        case Operation::LOAD: return "LOAD";
        case Operation::DUMP: return "DUMP";
        default: return "UNKNOWN";
    }
}

Status ValidateCurrentDevice(std::int32_t expectedDeviceId)
{
    aclrtContext context = nullptr;
    auto ret = aclrtGetCurrentContext(&context);
    if (ret != ACL_SUCCESS || context == nullptr) {
        return Status::Error("delegator executor requires a current ACL context");
    }

    std::int32_t currentDeviceId = -1;
    ret = aclrtGetDevice(&currentDeviceId);
    if (ret != ACL_SUCCESS) { return Status::Error("aclrtGetDevice failed"); }
    if (currentDeviceId != expectedDeviceId) {
        return Status::InvalidParam("current ACL context device does not match device_id");
    }
    return Status::OK();
}

Status BindDevice(std::int32_t deviceId)
{
    const auto ret = aclrtSetDevice(deviceId);
    return ret == ACL_SUCCESS ? Status::OK() : Status::Error("aclrtSetDevice failed");
}

}  // namespace

Expected<std::unique_ptr<Executor>> Executor::Create(std::shared_ptr<StoreV1> backend,
                                                     std::vector<std::size_t> tensorSizes,
                                                     std::int32_t deviceId, std::size_t slotNum,
                                                     std::size_t streamNumber,
                                                     const Detail::Dictionary& backendConfig)
{
    if (backend == nullptr || deviceId < 0 || slotNum == 0 || streamNumber == 0 ||
        tensorSizes.empty()) {
        return Status::InvalidParam("invalid delegator executor config");
    }

    std::size_t payloadSize = 0;
    for (const auto size : tensorSizes) {
        if (size == 0 || size > std::numeric_limits<std::size_t>::max() - payloadSize) {
            return Status::InvalidParam("invalid delegator tensor sizes");
        }
        payloadSize += size;
    }

    auto status = ValidateCurrentDevice(deviceId);
    if (status.Failure()) { return status; }

    auto executor = std::unique_ptr<Executor>{new (std::nothrow) Executor(
        std::move(backend), std::move(tensorSizes), deviceId, slotNum, streamNumber)};
    if (!executor) { return Status::OutOfMemory(); }

    try {
        status = executor->Start(payloadSize, slotNum, backendConfig);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    } catch (const std::system_error& error) {
        return Status::OsApiError(error.what());
    }
    if (status.Failure()) { return status; }

    return executor;
}

Executor::Executor(std::shared_ptr<StoreV1> backend, std::vector<std::size_t> tensorSizes,
                   std::int32_t deviceId, std::size_t slotNum, std::size_t streamNumber)
    : backend_{std::move(backend)},
      tensorSizes_{std::move(tensorSizes)},
      deviceId_{deviceId},
      streamNumber_{std::min(streamNumber, slotNum)}
{
}

Status Executor::Start(std::size_t payloadSize, std::size_t slotNum,
                       const Detail::Dictionary& backendConfig)
{
    auto status = bufferPool_.Init("delegator_buffer_pool", BufferPool::MemoryType::Device,
                                   payloadSize, slotNum, false, kBufferAlignment);
    if (status.Failure()) { return status; }
    slotNum_ = slotNum;
    const auto address = reinterpret_cast<std::uintptr_t>(bufferPool_.GetDeviceAddr());
    const auto size = bufferPool_.GetTotalSize();
    if (address > static_cast<std::uintptr_t>(std::numeric_limits<ssize_t>::max()) ||
        size > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        return Status::InvalidParam("delegator buffer range cannot be represented in config");
    }
    auto config = backendConfig;
    config.Set("gpu_kv_buffer_addrs", std::vector<ssize_t>{static_cast<ssize_t>(address)});
    config.Set("gpu_kv_buffer_sizes", std::vector<ssize_t>{static_cast<ssize_t>(size)});
    status = backend_->Setup(config);
    if (status.Failure()) { return status; }
    availableSlots_ = slotNum;

    std::promise<Status> dumpStarted;
    std::promise<Status> loadStarted;
    auto dumpStartedFuture = dumpStarted.get_future();
    auto loadStartedFuture = loadStarted.get_future();
    try {
        dumpThread_ = std::thread(&Executor::DumpLoop, this, std::ref(dumpStarted));
        loadThread_ = std::thread(&Executor::LoadLoop, this, std::ref(loadStarted));
    } catch (...) {
        Shutdown();
        return Status::Error();
    }

    const auto dumpStatus = dumpStartedFuture.get();
    const auto loadStatus = loadStartedFuture.get();
    if (dumpStatus.Failure() || loadStatus.Failure()) {
        const auto status = dumpStatus.Failure() ? dumpStatus : loadStatus;
        Shutdown();
        return status;
    }
    return Status::OK();
}

Executor::~Executor() { Shutdown(); }

Expected<Detail::TaskDesc> Executor::MakeBackendTask(const TransferGroup& group) const
{
    Detail::TaskDesc task;
    task.brief = group.task->desc.brief;
    try {
        task.reserve(group.shards.size());
        for (const auto& shard : group.shards) {
            const auto& source = group.task->desc[shard.shardIndex];
            Detail::Shard backendShard;
            backendShard.owner = source.owner;
            backendShard.index = source.index;
            backendShard.addrs.reserve(tensorSizes_.size());

            auto* address = static_cast<std::byte*>(shard.slot.deviceAddr);
            std::size_t offset = 0;
            for (const auto size : tensorSizes_) {
                backendShard.addrs.push_back(address + offset);
                offset += size;
            }
            task.push_back(std::move(backendShard));
        }
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }
    return task;
}

Status Executor::ValidateTask(const Detail::TaskDesc& task, Operation operation) const
{
    if (operation != Operation::LOAD && operation != Operation::DUMP) {
        return Status::InvalidParam("invalid delegator operation");
    }
    if (task.empty() || tensorSizes_.empty()) {
        return Status::InvalidParam("empty delegator task");
    }
    // Verify that the submitted task matches the tensor layout configured for this executor.
    for (std::size_t shardIndex = 0; shardIndex < task.size(); ++shardIndex) {
        const auto& addrs = task[shardIndex].addrs;
        if (addrs.size() != tensorSizes_.size()) {
            return Status::InvalidParam("invalid delegator shard({}) address count", shardIndex);
        }
        for (const auto* addr : addrs) {
            if (addr == nullptr) {
                return Status::InvalidParam("null address in delegator shard({})", shardIndex);
            }
        }
    }
    return Status::OK();
}

Status Executor::GatherAsync(const TransferGroup& group, CopyStream& streams)
{
    for (const auto& shard : group.shards) {
        const auto& desc = group.task->desc[shard.shardIndex];
        const auto stream = streams.NextStream();
        auto* destination = static_cast<std::byte*>(shard.slot.deviceAddr);
        std::size_t offset = 0;
        for (std::size_t index = 0; index < desc.addrs.size(); ++index) {
            const auto status = streams.DeviceToDeviceAsync(stream, destination + offset,
                                                            shard.slot.length - offset,
                                                            desc.addrs[index], tensorSizes_[index]);
            if (status.Failure()) { return status; }
            offset += tensorSizes_[index];
        }
    }
    return Status::OK();
}

Status Executor::ScatterAsync(const TransferGroup& group, CopyStream& streams)
{
    for (const auto& shard : group.shards) {
        const auto& desc = group.task->desc[shard.shardIndex];
        const auto stream = streams.NextStream();
        const auto* source = static_cast<const std::byte*>(shard.slot.deviceAddr);
        std::size_t offset = 0;
        for (std::size_t index = 0; index < desc.addrs.size(); ++index) {
            const auto status =
                streams.DeviceToDeviceAsync(stream, desc.addrs[index], tensorSizes_[index],
                                            source + offset, tensorSizes_[index]);
            if (status.Failure()) { return status; }
            offset += tensorSizes_[index];
        }
    }
    return Status::OK();
}

std::string Executor::DescribeShards(const TransferGroup& group) const
{
    std::string result;
    for (const auto& shard : group.shards) {
        const auto& desc = group.task->desc[shard.shardIndex];
        if (!result.empty()) { result += ", "; }
        result += fmt::format("{{owner={:02x},index={},slot={}}}", fmt::join(desc.owner, ""),
                              desc.index, shard.slot.slotIndex);
    }
    return result;
}

void Executor::LogTaskCompletion(const TaskContext& task) const
{
    const auto* operation = OperationName(task.operation);
    if (task.error) {
        UC_ERROR("Delegator {} task({},{}) failed, shards={}, status={}.", operation, task.id,
                 task.desc.brief, task.desc.size(), *task.error);
        return;
    }
    UC_INFO("Delegator {} task({},{}) completed, shards={}.", operation, task.id, task.desc.brief,
            task.desc.size());
}

void Executor::CheckSchedulerInvariantsLocked() const
{
    const bool valid =
        availableSlots_ + inFlightLoadShardNum_ + inFlightDumpShardNum_ == slotNum_ &&
        outstandingShardNum_ == queuedLoadShardNum_ + queuedDumpShardNum_ + inFlightLoadShardNum_ +
                                    inFlightDumpShardNum_ &&
        loadQueue_.size() == queuedLoadShardNum_ && dumpQueue_.size() == queuedDumpShardNum_;
    if (valid) { return; }

    UC_ERROR(
        "Delegator scheduler invariant violated: slots={}/{}, outstanding={}, queued(load/dump)="
        "{}/{}, in_flight(load/dump)={}/{}, queue(load/dump)={}/{}.",
        availableSlots_, slotNum_, outstandingShardNum_, queuedLoadShardNum_, queuedDumpShardNum_,
        inFlightLoadShardNum_, inFlightDumpShardNum_, loadQueue_.size(), dumpQueue_.size());
    assert(valid);
}

Expected<Detail::TaskHandle> Executor::Submit(Detail::TaskDesc task, Operation operation)
{
    auto status = ValidateTask(task, operation);
    if (status.Failure()) { return status; }

    std::shared_ptr<TaskContext> taskContext;
    try {
        taskContext = std::make_shared<TaskContext>();
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }
    taskContext->desc = std::move(task);
    taskContext->operation = operation;
    taskContext->remaining = taskContext->desc.size();

    const auto handle = taskContext->id;
    {
        // Keep the whole publish atomic against Shutdown.
        std::lock_guard<std::mutex> schedLock(schedMutex_);
        if (shutdownStarted_) { return Status::Error(); }
        CheckSchedulerInvariantsLocked();
        const auto count = taskContext->desc.size();
        if (count > std::numeric_limits<std::size_t>::max() - outstandingShardNum_) {
            return Status::OutOfMemory();
        }

        auto& queue = operation == Operation::LOAD ? loadQueue_ : dumpQueue_;
        if (count > queue.max_size() - queue.size()) { return Status::OutOfMemory(); }

        const auto previousQueueSize = queue.size();
        const auto rollbackPublishedShards = [&queue, previousQueueSize]() noexcept {
            while (queue.size() != previousQueueSize) { queue.pop_back(); }
        };
        try {
            for (std::size_t shardIndex = 0; shardIndex < count; ++shardIndex) {
                queue.push_back(QueuedShardContext{taskContext, shardIndex});
            }
            {
                std::lock_guard<std::shared_mutex> tasksLock(taskMapMutex_);
                if (!tasks_.emplace(handle, taskContext).second) {
                    rollbackPublishedShards();
                    return Status::DuplicateKey();
                }
            }
        } catch (const std::bad_alloc&) {
            rollbackPublishedShards();
            return Status::OutOfMemory();
        }

        outstandingShardNum_ += count;
        auto& queuedNum = operation == Operation::LOAD ? queuedLoadShardNum_ : queuedDumpShardNum_;
        queuedNum += count;
        CheckSchedulerInvariantsLocked();
    }
    UC_INFO("Delegator {} task({},{}) submitted, shards={}.", OperationName(operation), handle,
            taskContext->desc.brief, taskContext->desc.size());
    slotsReady_.notify_all();
    return Detail::TaskHandle{handle};
}

Expected<bool> Executor::Check(Detail::TaskHandle task)
{
    std::shared_ptr<TaskContext> taskContext;
    {
        std::shared_lock<std::shared_mutex> lock(taskMapMutex_);
        const auto iter = tasks_.find(task);
        if (iter == tasks_.end()) { return Status::NotFound(); }
        taskContext = iter->second;
    }

    std::lock_guard<std::mutex> lock(taskContext->stateMutex);
    return bool{taskContext->remaining == 0};
}

Status Executor::Wait(Detail::TaskHandle task)
{
    std::shared_ptr<TaskContext> taskContext;
    {
        std::lock_guard<std::shared_mutex> lock(taskMapMutex_);
        const auto iter = tasks_.find(task);
        if (iter == tasks_.end()) { return Status::NotFound(); }

        // Wait claims the handle before blocking; task handles have single-consumer semantics.
        taskContext = iter->second;
        tasks_.erase(iter);
    }

    std::unique_lock<std::mutex> lock(taskContext->stateMutex);
    taskContext->completed.wait(lock, [&taskContext]() { return taskContext->remaining == 0; });
    return taskContext->error.value_or(Status::OK());
}

Expected<Executor::TransferBatch> Executor::AcquireBatch(Operation operation)
{
    assert(operation == Operation::LOAD || operation == Operation::DUMP);
    auto& queue = operation == Operation::LOAD ? loadQueue_ : dumpQueue_;
    TransferBatch batch;
    std::vector<QueuedShardContext> reservedShards;
    const auto batchCapacity = slotNum_;
    try {
        batch.groups.reserve(batchCapacity);
        reservedShards.reserve(batchCapacity);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }

    const auto canReserve = [this, operation]() {
        if (availableSlots_ == 0) { return false; }
        // LOAD is admitted whenever a queued shard and a slot are available.
        if (operation == Operation::LOAD) { return queuedLoadShardNum_ != 0; }
        // DUMP is admitted only when no LOAD shard is queued or in flight.
        return queuedDumpShardNum_ != 0 && queuedLoadShardNum_ == 0 && inFlightLoadShardNum_ == 0;
    };

    // Keep trying until a non-empty batch can be returned or shutdown begins.
    for (;;) {
        // Phase 1: reserve shards under the scheduler lock and complete cancellations outside it.
        reservedShards.clear();
        while (reservedShards.size() < batchCapacity) {
            std::optional<QueuedShardContext> cancelledShard;
            {
                std::unique_lock<std::mutex> lock(schedMutex_);

                // Wait only for the first shard; do not wait to fill a partial batch.
                if (reservedShards.empty()) {
                    slotsReady_.wait(
                        lock, [this, &canReserve]() { return shutdownStarted_ || canReserve(); });
                }
                if (shutdownStarted_) {
                    if (reservedShards.empty()) { return Status::Error(); }
                    break;
                }

                auto& queuedNum =
                    operation == Operation::LOAD ? queuedLoadShardNum_ : queuedDumpShardNum_;
                auto& inFlightNum =
                    operation == Operation::LOAD ? inFlightLoadShardNum_ : inFlightDumpShardNum_;
                while (reservedShards.size() < batchCapacity && canReserve()) {
                    auto shard = std::move(queue.front());
                    queue.pop_front();
                    --queuedNum;

                    if (shard.task->failed.load(std::memory_order_acquire)) {
                        --outstandingShardNum_;
                        cancelledShard = std::move(shard);
                        break;
                    }

                    ++inFlightNum;
                    --availableSlots_;
                    reservedShards.push_back(std::move(shard));
                }
                CheckSchedulerInvariantsLocked();
            }

            if (cancelledShard) {
                CompleteTaskShards(cancelledShard->task, 1, Status::OK());
                slotsReady_.notify_all();
                continue;
            }
            break;
        }

        // Phase 2: create groups and preallocate their shard lists, rolling back on OOM.
        batch.groups.clear();
        try {
            std::size_t groupBegin = 0;
            while (groupBegin < reservedShards.size()) {
                std::size_t groupEnd = groupBegin + 1;
                while (groupEnd < reservedShards.size() &&
                       reservedShards[groupEnd].task.get() ==
                           reservedShards[groupBegin].task.get()) {
                    ++groupEnd;
                }

                TransferGroup group;
                group.task = reservedShards[groupBegin].task;
                group.shards.reserve(groupEnd - groupBegin);
                batch.groups.push_back(std::move(group));
                groupBegin = groupEnd;
            }
        } catch (const std::bad_alloc&) {
            batch.groups.clear();
            for (const auto& reservedShard : reservedShards) {
                DiscardShard(reservedShard, Status::OutOfMemory(), ShardStage::IN_FLIGHT);
            }
            continue;
        }

        // Phase 3: allocate BufferPool slots and materialize the reserved shards into groups.
        std::size_t reservedIndex = 0;
        for (auto& group : batch.groups) {
            while (reservedIndex < reservedShards.size() &&
                   reservedShards[reservedIndex].task.get() == group.task.get()) {
                auto& reservedShard = reservedShards[reservedIndex++];
                if (reservedShard.task->failed.load(std::memory_order_acquire)) {
                    DiscardShard(reservedShard, Status::OK(), ShardStage::IN_FLIGHT);
                    continue;
                }

                BufferPool::Slot slot;
                const auto status = bufferPool_.Allocate(slot);
                if (status.Failure()) {
                    DiscardShard(reservedShard, status, ShardStage::IN_FLIGHT);
                    continue;
                }

                InFlightShardContext inFlightShard;
                inFlightShard.shardIndex = reservedShard.shardIndex;
                inFlightShard.slot = std::move(slot);
                group.shards.push_back(std::move(inFlightShard));
            }
        }
        assert(reservedIndex == reservedShards.size());

        batch.groups.erase(
            std::remove_if(batch.groups.begin(), batch.groups.end(),
                           [](const TransferGroup& group) { return group.shards.empty(); }),
            batch.groups.end());
        if (!batch.groups.empty()) { return batch; }
    }
}

void Executor::ReleaseBatch(TransferBatch& batch)
{
    assert(!batch.groups.empty());
    assert(batch.groups.front().task);

    const auto operation = batch.groups.front().task->operation;
    assert(operation == Operation::LOAD || operation == Operation::DUMP);

    std::size_t count = 0;
    for (auto& group : batch.groups) {
        for (const auto& shard : group.shards) {
            const auto freeStatus = bufferPool_.Free(shard.slot.slotIndex);
            if (!group.error && freeStatus.Failure()) { group.error = freeStatus; }
        }
        count += group.shards.size();
    }

    // Publish task state before making the released slots available. A worker woken by the slot
    // update must observe task->failed before acquiring later shards of the same task.
    // Phase 1: publish each transfer group's result and completion independently.
    for (const auto& group : batch.groups) {
        CompleteTaskShards(group.task, group.shards.size(), group.error.value_or(Status::OK()));
    }

    // Phase 2: publish the released slots only after task completion state is visible.
    {
        std::lock_guard<std::mutex> lock(schedMutex_);
        auto& inFlightNum =
            operation == Operation::LOAD ? inFlightLoadShardNum_ : inFlightDumpShardNum_;
        assert(inFlightNum >= count);
        assert(outstandingShardNum_ >= count);
        inFlightNum -= count;
        availableSlots_ += count;
        outstandingShardNum_ -= count;
        CheckSchedulerInvariantsLocked();
    }
    slotsReady_.notify_all();
}

void Executor::CompleteTaskShards(const std::shared_ptr<TaskContext>& task, std::size_t count,
                                  const Status& status)
{
    bool completed = false;
    {
        std::lock_guard<std::mutex> lock(task->stateMutex);
        assert(task->remaining >= count);
        task->remaining -= count;
        if (!task->error && status.Failure()) {
            task->error = status;
            task->failed.store(true, std::memory_order_release);
        }
        completed = task->remaining == 0;
        if (completed) { LogTaskCompletion(*task); }
    }
    if (completed) { task->completed.notify_all(); }
}

void Executor::DiscardShard(const QueuedShardContext& shard, const Status& status, ShardStage stage)
{
    CompleteTaskShards(shard.task, 1, status);
    {
        std::lock_guard<std::mutex> lock(schedMutex_);
        auto& queuedNum =
            shard.task->operation == Operation::LOAD ? queuedLoadShardNum_ : queuedDumpShardNum_;
        auto& inFlightNum = shard.task->operation == Operation::LOAD ? inFlightLoadShardNum_
                                                                     : inFlightDumpShardNum_;
        assert(outstandingShardNum_ != 0);
        if (stage == ShardStage::QUEUED) {
            assert(queuedNum != 0);
            --queuedNum;
        } else {
            assert(inFlightNum != 0);
            --inFlightNum;
            ++availableSlots_;
        }
        --outstandingShardNum_;
        CheckSchedulerInvariantsLocked();
    }

    slotsReady_.notify_all();
}

void Executor::DrainQueue(std::deque<QueuedShardContext>& queue)
{
    const auto status = Status::Error();
    while (!queue.empty()) {
        auto shard = std::move(queue.front());
        queue.pop_front();
        DiscardShard(shard, status, ShardStage::QUEUED);
    }
}

void Executor::DumpLoop(std::promise<Status>& started)
{
    // Init
    CopyStream streams;
    auto status = BindDevice(deviceId_);
    if (status.Success()) { status = streams.Setup(deviceId_, streamNumber_); }
    started.set_value(status);
    if (status.Failure()) { return; }

    for (;;) {
        auto batchResult = AcquireBatch(Operation::DUMP);
        if (!batchResult) { break; }
        auto batch = std::move(batchResult).Value();
        std::unordered_map<Detail::TaskHandle, TransferTiming> timings;
        timings.reserve(batch.groups.size());

        for (auto& group : batch.groups) {
            if (Logger::isEnabledFor(Logger::Level::DEBUG)) {
                UC_DEBUG("Delegator DUMP task({},{}) processing KVCache shards=[{}].",
                         group.task->id, group.task->desc.brief, DescribeShards(group));
            }
            auto& timing = timings[group.task->id];
            // Record the start time for D2D duration calculation.
            timing.d2dStartTp = NowTime::Now();
            const auto gatherStatus = GatherAsync(group, streams);
            if (gatherStatus.Failure()) {
                group.error = gatherStatus;
                UC_ERROR("Delegator DUMP task({},{}) stage=gather failed, status={}.",
                         group.task->id, group.task->desc.brief, gatherStatus);
            }
        }

        const auto syncStatus = streams.SynchronizeAll();
        // Record the end time for D2D duration calculation.
        const auto d2dEndTp = NowTime::Now();
        if (syncStatus.Failure()) {
            for (auto& group : batch.groups) {
                if (!group.error) {
                    group.error = syncStatus;
                    UC_ERROR("Delegator DUMP task({},{}) stage=gather_sync failed, status={}.",
                             group.task->id, group.task->desc.brief, syncStatus);
                }
            }
        } else {
            for (const auto& group : batch.groups) {
                if (!group.error) {
                    const auto& timing = timings.at(group.task->id);
                    UC_DEBUG(
                        "Delegator DUMP task({},{}) stage=gather_complete, shards={}, "
                        "d2d_duration={:.3f}ms.",
                        group.task->id, group.task->desc.brief, group.shards.size(),
                        (d2dEndTp - timing.d2dStartTp) * 1e3);
                }
            }
        }

        for (auto& group : batch.groups) {
            if (group.error) { continue; }

            auto backendTask = MakeBackendTask(group);
            if (!backendTask) {
                group.error = backendTask.Error();
                UC_ERROR("Delegator DUMP task({},{}) stage=backend_task_build failed, status={}.",
                         group.task->id, group.task->desc.brief, *group.error);
                continue;
            }
            auto& timing = timings.at(group.task->id);
            // Record the start time for backend duration calculation.
            timing.backendStartTp = NowTime::Now();
            auto submitted = backend_->Dump(std::move(backendTask).Value());
            if (submitted) {
                group.transferTask = std::move(submitted).Value();
                group.transferPending = true;
            } else {
                group.error = submitted.Error();
                UC_ERROR("Delegator DUMP task({},{}) stage=backend_submit failed, status={}.",
                         group.task->id, group.task->desc.brief, *group.error);
            }
        }

        for (auto& group : batch.groups) {
            if (!group.transferPending) { continue; }
            const auto waitStatus = backend_->Wait(group.transferTask);
            // Record the end time for backend duration calculation.
            const auto backendEndTp = NowTime::Now();
            const auto& timing = timings.at(group.task->id);
            if (waitStatus.Failure()) {
                group.error = waitStatus;
                UC_ERROR(
                    "Delegator DUMP task({},{}) stage=backend_wait failed, backend_task={}, "
                    "status={}.",
                    group.task->id, group.task->desc.brief, group.transferTask, waitStatus);
            } else {
                UC_DEBUG(
                    "Delegator DUMP task({},{}) stage=backend_complete, backend_task={}, "
                    "shards={}, backend_duration={:.3f}ms.",
                    group.task->id, group.task->desc.brief, group.transferTask, group.shards.size(),
                    (backendEndTp - timing.backendStartTp) * 1e3);
            }
            group.transferPending = false;
        }

        ReleaseBatch(batch);
    }
}

void Executor::LoadLoop(std::promise<Status>& started)
{
    // Init
    CopyStream streams;
    auto status = BindDevice(deviceId_);
    if (status.Success()) { status = streams.Setup(deviceId_, streamNumber_); }
    started.set_value(status);
    if (status.Failure()) { return; }

    for (;;) {
        auto batchResult = AcquireBatch(Operation::LOAD);
        if (!batchResult) { break; }
        auto batch = std::move(batchResult).Value();
        std::unordered_map<Detail::TaskHandle, TransferTiming> timings;
        timings.reserve(batch.groups.size());

        std::size_t pendingGroupCount = 0;
        for (auto& group : batch.groups) {
            if (Logger::isEnabledFor(Logger::Level::DEBUG)) {
                UC_DEBUG("Delegator LOAD task({},{}) processing KVCache shards=[{}].",
                         group.task->id, group.task->desc.brief, DescribeShards(group));
            }
            auto& timing = timings[group.task->id];
            auto backendTask = MakeBackendTask(group);
            if (!backendTask) {
                group.error = backendTask.Error();
                UC_ERROR("Delegator LOAD task({},{}) stage=backend_task_build failed, status={}.",
                         group.task->id, group.task->desc.brief, *group.error);
                continue;
            }
            // Record the start time for backend duration calculation.
            timing.backendStartTp = NowTime::Now();
            auto submitted = backend_->Load(std::move(backendTask).Value());
            if (submitted) {
                group.transferTask = std::move(submitted).Value();
                group.transferPending = true;
                ++pendingGroupCount;
            } else {
                group.error = submitted.Error();
                UC_ERROR("Delegator LOAD task({},{}) stage=backend_submit failed, status={}.",
                         group.task->id, group.task->desc.brief, *group.error);
            }
        }

        while (pendingGroupCount != 0) {
            for (auto& group : batch.groups) {
                if (!group.transferPending) { continue; }

                auto completed = backend_->Check(group.transferTask);
                if (!completed.Value()) { continue; }

                // Check(true) only means that the backend task has completed. The final
                // success/failure status is retrieved by Wait().
                auto waitStatus = backend_->Wait(group.transferTask);
                auto& timing = timings.at(group.task->id);
                if (waitStatus.Failure()) {
                    group.error = waitStatus;
                    UC_ERROR(
                        "Delegator LOAD task({},{}) stage=backend_wait failed, "
                        "backend_task={}, status={}.",
                        group.task->id, group.task->desc.brief, group.transferTask, waitStatus);
                } else {
                    // Calculate backend duration on completion.
                    UC_DEBUG(
                        "Delegator LOAD task({},{}) stage=backend_complete, "
                        "backend_task={}, shards={}, backend_duration={:.3f}ms.",
                        group.task->id, group.task->desc.brief, group.transferTask,
                        group.shards.size(), (NowTime::Now() - timing.backendStartTp) * 1e3);
                    // Record the start time for D2D duration calculation.
                    timing.d2dStartTp = NowTime::Now();
                    const auto scatterStatus = ScatterAsync(group, streams);
                    if (scatterStatus.Failure()) {
                        group.error = scatterStatus;
                        UC_ERROR("Delegator LOAD task({},{}) stage=scatter failed, status={}.",
                                 group.task->id, group.task->desc.brief, scatterStatus);
                    }
                }

                group.transferPending = false;
                --pendingGroupCount;
            }
        }
        const auto syncStatus = streams.SynchronizeAll();
        // Record the end time for D2D duration calculation.
        const auto d2dEndTp = NowTime::Now();
        if (syncStatus.Failure()) {
            for (auto& group : batch.groups) {
                if (!group.error) {
                    group.error = syncStatus;
                    UC_ERROR("Delegator LOAD task({},{}) stage=scatter_sync failed, status={}.",
                             group.task->id, group.task->desc.brief, syncStatus);
                }
            }
        } else {
            for (const auto& group : batch.groups) {
                if (!group.error) {
                    const auto& timing = timings.at(group.task->id);
                    UC_DEBUG(
                        "Delegator LOAD task({},{}) stage=scatter_complete, shards={}, "
                        "d2d_duration={:.3f}ms.",
                        group.task->id, group.task->desc.brief, group.shards.size(),
                        (d2dEndTp - timing.d2dStartTp) * 1e3);
                }
            }
        }
        ReleaseBatch(batch);
    }
}

void Executor::Shutdown()
{
    {
        std::unique_lock<std::mutex> lock(schedMutex_);
        if (shutdownStarted_) {
            shutdownCompleted_.wait(lock, [this]() { return shutdownComplete_; });
            return;
        }
        shutdownStarted_ = true;
        slotsReady_.notify_all();
    }

    if (dumpThread_.joinable()) { dumpThread_.join(); }
    if (loadThread_.joinable()) { loadThread_.join(); }

    // Workers are gone; we are now the sole consumer of both queues.
    DrainQueue(dumpQueue_);
    DrainQueue(loadQueue_);
    if (bufferPool_.IsInitialized()) {
        // Assumption: Executor is created, used, and destroyed under the same ACL device context.
        bufferPool_.Reset();
    }

    {
        std::lock_guard<std::mutex> lock(schedMutex_);
        shutdownComplete_ = true;
    }
    shutdownCompleted_.notify_all();
}

}  // namespace UC::Delegator
