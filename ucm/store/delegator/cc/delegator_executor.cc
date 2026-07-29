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

namespace UC::Delegator {
namespace {

constexpr std::size_t kBufferAlignment = 16 * 1024;

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

Expected<std::unique_ptr<Executor>> Executor::Create(
    std::unique_ptr<TransferEndpoint> endpoint, std::vector<std::size_t> tensor_sizes,
    std::int32_t device_id, std::size_t slot_num, std::size_t stream_number)
{
    if (endpoint == nullptr || device_id < 0 || slot_num == 0 || stream_number == 0 ||
        tensor_sizes.empty()) {
        return Status::InvalidParam("invalid delegator executor config");
    }

    std::size_t payloadSize = 0;
    for (const auto size : tensor_sizes) {
        if (size == 0 || size > std::numeric_limits<std::size_t>::max() - payloadSize) {
            return Status::InvalidParam("invalid delegator tensor sizes");
        }
        payloadSize += size;
    }

    auto status = ValidateCurrentDevice(device_id);
    if (status.Failure()) { return status; }

    auto executor = std::unique_ptr<Executor>{new (std::nothrow) Executor(
        std::move(endpoint), std::move(tensor_sizes), device_id, slot_num, stream_number)};
    if (!executor) { return Status::OutOfMemory(); }

    try {
        status = executor->Start(payloadSize, slot_num);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    } catch (const std::system_error& error) {
        return Status::OsApiError(error.what());
    }
    if (status.Failure()) { return status; }

    return executor;
}

Executor::Executor(std::unique_ptr<TransferEndpoint> endpoint,
                   std::vector<std::size_t> tensor_sizes, std::int32_t device_id,
                   std::size_t slot_num, std::size_t stream_number)
    : endpoint_{std::move(endpoint)},
      tensor_sizes_{std::move(tensor_sizes)},
      device_id_{device_id},
      stream_number_{std::min(stream_number, slot_num)}
{
}

Status Executor::Start(std::size_t payload_size, std::size_t slot_num)
{
    auto status = buffer_pool_.Init("delegator_buffer_pool", BufferPool::MemoryType::ASCEND_DEVICE,
                                    payload_size, slot_num, false, kBufferAlignment);
    if (status.Failure()) { return status; }
    slot_num_ = slot_num;
    status = endpoint_->SetupTransferRegion(
        TransferRegion{buffer_pool_.GetLocalAddr(), buffer_pool_.GetDeviceAddr(),
                       buffer_pool_.GetTotalSize(), buffer_pool_.GetMemoryType(), device_id_});
    if (status.Failure()) { return status; }
    transfer_region_ready_ = true;
    available_slots_ = slot_num;

    std::promise<Status> dumpStarted;
    std::promise<Status> loadStarted;
    auto dumpStartedFuture = dumpStarted.get_future();
    auto loadStartedFuture = loadStarted.get_future();
    try {
        dump_thread_ = std::thread(&Executor::DumpLoop, this, std::ref(dumpStarted));
        load_thread_ = std::thread(&Executor::LoadLoop, this, std::ref(loadStarted));
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

Status Executor::ValidateTask(const Detail::TaskDesc& task, Operation operation) const
{
    if (operation != Operation::LOAD && operation != Operation::DUMP) {
        return Status::InvalidParam("invalid delegator operation");
    }
    if (task.empty() || tensor_sizes_.empty()) {
        return Status::InvalidParam("empty delegator task");
    }
    // Verify that the submitted task matches the tensor layout configured for this executor.
    for (std::size_t shardIndex = 0; shardIndex < task.size(); ++shardIndex) {
        const auto& addrs = task[shardIndex].addrs;
        if (addrs.size() != tensor_sizes_.size()) {
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
        const auto& desc = group.task->desc[shard.shard_index];
        const auto stream = streams.NextStream();
        auto* destination = static_cast<std::byte*>(shard.slot.device_addr);
        std::size_t offset = 0;
        for (std::size_t index = 0; index < desc.addrs.size(); ++index) {
            const auto status =
                streams.DeviceToDeviceAsync(stream, destination + offset,
                                            shard.slot.length - offset, desc.addrs[index],
                                            tensor_sizes_[index]);
            if (status.Failure()) { return status; }
            offset += tensor_sizes_[index];
        }
    }
    return Status::OK();
}

Status Executor::ScatterAsync(const TransferGroup& group, CopyStream& streams)
{
    for (const auto& shard : group.shards) {
        const auto& desc = group.task->desc[shard.shard_index];
        const auto stream = streams.NextStream();
        const auto* source = static_cast<const std::byte*>(shard.slot.device_addr);
        std::size_t offset = 0;
        for (std::size_t index = 0; index < desc.addrs.size(); ++index) {
            const auto status =
                streams.DeviceToDeviceAsync(stream, desc.addrs[index], tensor_sizes_[index],
                                            source + offset, tensor_sizes_[index]);
            if (status.Failure()) { return status; }
            offset += tensor_sizes_[index];
        }
    }
    return Status::OK();
}

void Executor::AssertSchedulerInvariantsLocked() const
{
    assert(available_slots_ + in_flight_load_shards_ + in_flight_dump_shards_ == slot_num_);
    assert(outstanding_shards_ == queued_load_shards_ + queued_dump_shards_ +
                                      in_flight_load_shards_ + in_flight_dump_shards_);
    assert(load_queue_.size() == queued_load_shards_);
    assert(dump_queue_.size() == queued_dump_shards_);
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
        std::lock_guard<std::mutex> schedLock(sched_mutex_);
        if (shutdown_started_) { return Status::Error(); }
        AssertSchedulerInvariantsLocked();
        const auto count = taskContext->desc.size();
        if (count > std::numeric_limits<std::size_t>::max() - outstanding_shards_) {
            return Status::OutOfMemory();
        }

        auto& queue = operation == Operation::LOAD ? load_queue_ : dump_queue_;
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
                std::lock_guard<std::mutex> tasksLock(tasks_mutex_);
                if (!tasks_.emplace(handle, taskContext).second) {
                    rollbackPublishedShards();
                    return Status::DuplicateKey();
                }
            }
        } catch (const std::bad_alloc&) {
            rollbackPublishedShards();
            return Status::OutOfMemory();
        }

        outstanding_shards_ += count;
        auto& queued =
            operation == Operation::LOAD ? queued_load_shards_ : queued_dump_shards_;
        queued += count;
        AssertSchedulerInvariantsLocked();
    }
    slots_ready_.notify_all();
    return Detail::TaskHandle{handle};
}

Expected<bool> Executor::Check(Detail::TaskHandle task)
{
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    const auto iter = tasks_.find(task);
    if (iter == tasks_.end()) { return Status::NotFound(); }
    return bool{iter->second->remaining == 0};
}

Status Executor::Wait(Detail::TaskHandle task)
{
    std::unique_lock<std::mutex> lock(tasks_mutex_);
    const auto iter = tasks_.find(task);
    if (iter == tasks_.end()) { return Status::NotFound(); }

    const auto taskContext = iter->second;
    taskContext->completed.wait(lock, [&taskContext]() { return taskContext->remaining == 0; });
    const auto status = taskContext->error.value_or(Status::OK());
    tasks_.erase(task);
    return status;
}

Expected<Executor::TransferBatch> Executor::AcquireBatch(Operation operation)
{
    assert(operation == Operation::LOAD || operation == Operation::DUMP);
    auto& queue = operation == Operation::LOAD ? load_queue_ : dump_queue_;
    TransferBatch batch;
    std::vector<QueuedShardContext> reservedShards;
    const auto batchCapacity = slot_num_;
    try {
        batch.groups.reserve(batchCapacity);
        reservedShards.reserve(batchCapacity);
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory();
    }

    const auto canReserve = [this, operation]() {
        if (available_slots_ == 0) { return false; }
        // LOAD is admitted whenever a queued shard and a slot are available.
        if (operation == Operation::LOAD) { return queued_load_shards_ != 0; }
        // DUMP is admitted only when no LOAD shard is queued or in flight.
        return queued_dump_shards_ != 0 && queued_load_shards_ == 0 &&
               in_flight_load_shards_ == 0;
    };

    // Keep trying until a non-empty batch can be returned or shutdown begins.
    for (;;) {
        // Phase 1: reserve shards under the scheduler lock and complete cancellations outside it.
        reservedShards.clear();
        while (reservedShards.size() < batchCapacity) {
            std::optional<QueuedShardContext> cancelledShard;
            {
                std::unique_lock<std::mutex> lock(sched_mutex_);

                // Wait only for the first shard; do not wait to fill a partial batch.
                if (reservedShards.empty()) {
                    slots_ready_.wait(lock, [this, &canReserve]() {
                        return shutdown_started_ || canReserve();
                    });
                }
                if (shutdown_started_) {
                    if (reservedShards.empty()) { return Status::Error(); }
                    break;
                }
                if (!canReserve()) { break; }

                auto& queued =
                    operation == Operation::LOAD ? queued_load_shards_ : queued_dump_shards_;
                auto& inFlight = operation == Operation::LOAD ? in_flight_load_shards_
                                                              : in_flight_dump_shards_;
                while (reservedShards.size() < batchCapacity && canReserve()) {
                    assert(!queue.empty());
                    if (queue.empty()) { std::terminate(); }

                    auto shard = std::move(queue.front());
                    queue.pop_front();
                    assert(shard.task);
                    if (!shard.task) { std::terminate(); }
                    assert(queued != 0);
                    --queued;

                    if (shard.task->failed.load(std::memory_order_acquire)) {
                        assert(outstanding_shards_ != 0);
                        --outstanding_shards_;
                        cancelledShard = std::move(shard);
                        break;
                    }

                    ++inFlight;
                    --available_slots_;
                    reservedShards.push_back(std::move(shard));
                }
                AssertSchedulerInvariantsLocked();
            }

            if (cancelledShard) {
                RecordShardCompletion(*cancelledShard, Status::OK());
                slots_ready_.notify_all();
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
                const auto status = buffer_pool_.Allocate(slot);
                if (status.Failure()) {
                    DiscardShard(reservedShard, status, ShardStage::IN_FLIGHT);
                    continue;
                }

                InFlightShardContext inFlightShard;
                inFlightShard.shard_index = reservedShard.shard_index;
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
    if (batch.groups.empty()) { std::terminate(); }

    assert(batch.groups.front().task);
    if (!batch.groups.front().task) { std::terminate(); }
    const auto operation = batch.groups.front().task->operation;
    assert(operation == Operation::LOAD || operation == Operation::DUMP);
    if (operation != Operation::LOAD && operation != Operation::DUMP) { std::terminate(); }

    std::size_t count = 0;
    for (auto& group : batch.groups) {
        const bool validGroup =
            group.task && !group.shards.empty() && group.task->operation == operation;
        assert(validGroup);
        if (!validGroup) { std::terminate(); }

        for (const auto& shard : group.shards) {
            const auto freeStatus = buffer_pool_.Free(shard.slot.slot_index);
            if (!group.error && freeStatus.Failure()) { group.error = freeStatus; }
        }
        count += group.shards.size();
    }

    // Publish task state before making the released slots available. A worker woken by the slot
    // update must observe task->failed before acquiring later shards of the same task.
    // Phase 1: publish each transfer group's result and completion independently.
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        for (const auto& group : batch.groups) {
            assert(group.task->remaining >= group.shards.size());
            group.task->remaining -= group.shards.size();
            if (!group.task->error && group.error) {
                group.task->error = *group.error;
                group.task->failed.store(true, std::memory_order_release);
            }
            if (group.task->remaining == 0) { group.task->completed.notify_all(); }
        }
    }

    // Phase 2: publish the released slots only after task completion state is visible.
    {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        auto& inFlight = operation == Operation::LOAD ? in_flight_load_shards_
                                                     : in_flight_dump_shards_;
        assert(inFlight >= count);
        assert(outstanding_shards_ >= count);
        inFlight -= count;
        available_slots_ += count;
        outstanding_shards_ -= count;
        AssertSchedulerInvariantsLocked();
    }
    slots_ready_.notify_all();
}

void Executor::RecordShardCompletion(const QueuedShardContext& shard, const Status& status)
{
    bool completed = false;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        if (!shard.task->error && status.Failure()) {
            shard.task->error = status;
            shard.task->failed.store(true, std::memory_order_release);
        }
        if (shard.task->remaining > 0) { --shard.task->remaining; }
        completed = shard.task->remaining == 0;
    }
    if (completed) { shard.task->completed.notify_all(); }
}

void Executor::DiscardShard(const QueuedShardContext& shard, const Status& status,
                            ShardStage stage)
{
    RecordShardCompletion(shard, status);
    {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        auto& queued = shard.task->operation == Operation::LOAD ? queued_load_shards_
                                                               : queued_dump_shards_;
        auto& inFlight = shard.task->operation == Operation::LOAD ? in_flight_load_shards_
                                                                 : in_flight_dump_shards_;
        assert(outstanding_shards_ != 0);
        if (stage == ShardStage::QUEUED) {
            assert(queued != 0);
            --queued;
        } else {
            assert(inFlight != 0);
            --inFlight;
            ++available_slots_;
        }
        --outstanding_shards_;
        AssertSchedulerInvariantsLocked();
    }

    slots_ready_.notify_all();
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
    auto status = BindDevice(device_id_);
    if (status.Success()) { status = streams.Setup(device_id_, stream_number_); }
    started.set_value(status);
    if (status.Failure()) { return; }

    for (;;) {
        auto batchResult = AcquireBatch(Operation::DUMP);
        if (!batchResult) { break; }
        auto batch = std::move(batchResult).Value();

        std::vector<TransferBuffer> buffers;
        for (auto& group : batch.groups) {
            const auto gatherStatus = GatherAsync(group, streams);
            if (gatherStatus.Failure()) { group.error = gatherStatus; }
        }

        const auto syncStatus = streams.SynchronizeAll();
        if (syncStatus.Failure()) {
            for (auto& group : batch.groups) {
                if (!group.error) { group.error = syncStatus; }
            }
        }

        for (auto& group : batch.groups) {
            if (group.error) { continue; }

            try {
                buffers.clear();
                for (const auto& shard : group.shards) {
                    const auto& desc = group.task->desc[shard.shard_index];
                    buffers.push_back(TransferBuffer{desc.owner, desc.index, shard.slot});
                }
                auto submitted = endpoint_->SubmitDump(buffers);
                if (submitted) {
                    group.transfer_task = std::move(submitted).Value();
                    group.transfer_pending = true;
                } else {
                    group.error = submitted.Error();
                }
            } catch (const std::bad_alloc&) {
                group.error = Status::OutOfMemory();
            }
        }

        for (auto& group : batch.groups) {
            if (!group.transfer_pending) { continue; }
            const auto waitStatus = endpoint_->Wait(group.transfer_task);
            if (waitStatus.Failure()) { group.error = waitStatus; }
            group.transfer_pending = false;
        }

        ReleaseBatch(batch);
    }
}

void Executor::LoadLoop(std::promise<Status>& started)
{
    // Init
    CopyStream streams;
    auto status = BindDevice(device_id_);
    if (status.Success()) { status = streams.Setup(device_id_, stream_number_); }
    started.set_value(status);
    if (status.Failure()) { return; }

    for (;;) {
        auto batchResult = AcquireBatch(Operation::LOAD);
        if (!batchResult) { break; }
        auto batch = std::move(batchResult).Value();

        std::vector<TransferBuffer> buffers;
        std::size_t pendingGroupCount = 0;
        for (auto& group : batch.groups) {
            try {
                buffers.clear();
                for (const auto& shard : group.shards) {
                    const auto& desc = group.task->desc[shard.shard_index];
                    buffers.push_back(TransferBuffer{desc.owner, desc.index, shard.slot});
                }
                auto submitted = endpoint_->SubmitLoad(buffers);
                if (submitted) {
                    group.transfer_task = std::move(submitted).Value();
                    group.transfer_pending = true;
                    ++pendingGroupCount;
                } else {
                    group.error = submitted.Error();
                }
            } catch (const std::bad_alloc&) {
                group.error = Status::OutOfMemory();
            }
        }

        while (pendingGroupCount != 0) {
            for (auto& group : batch.groups) {
                if (!group.transfer_pending) { continue; }

                auto completed = endpoint_->Check(group.transfer_task);
                if (!completed) {
                    group.error = completed.Error();
                } else if (!completed.Value()) {
                    continue;
                } else {
                    const auto scatterStatus = ScatterAsync(group, streams);
                    if (scatterStatus.Failure()) { group.error = scatterStatus; }
                }
                group.transfer_pending = false;
                --pendingGroupCount;
            }
        }
        const auto syncStatus = streams.SynchronizeAll();
        if (syncStatus.Failure()) {
            for (auto& group : batch.groups) {
                if (!group.error) { group.error = syncStatus; }
            }
        }
        ReleaseBatch(batch);
    }
}

void Executor::Shutdown()
{
    {
        std::unique_lock<std::mutex> lock(sched_mutex_);
        if (shutdown_started_) {
            shutdown_completed_.wait(lock, [this]() { return shutdown_complete_; });
            return;
        }
        shutdown_started_ = true;
        slots_ready_.notify_all();
    }

    if (dump_thread_.joinable()) { dump_thread_.join(); }
    if (load_thread_.joinable()) { load_thread_.join(); }

    // Workers are gone; we are now the sole consumer of both queues.
    DrainQueue(dump_queue_);
    DrainQueue(load_queue_);
    if (transfer_region_ready_) {
        endpoint_->ResetTransferRegion();
        transfer_region_ready_ = false;
    }
    if (buffer_pool_.IsInitialized()) {
        // Assumption: Executor is created, used, and destroyed under the same ACL device context.
        buffer_pool_.Reset();
    }

    {
        std::lock_guard<std::mutex> lock(sched_mutex_);
        shutdown_complete_ = true;
    }
    shutdown_completed_.notify_all();
}

}  // namespace UC::Delegator
