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
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "delegator_copy_stream.h"
#include "detail/type/types.h"
#include "pool/buffer_pool.h"
#include "ucmstore_v1.h"

namespace UC::Delegator {

enum class Operation {
    LOAD,
    DUMP,
};

class Executor {
public:
    static constexpr std::size_t kDefaultStreamNumber = 4;

    static Expected<std::unique_ptr<Executor>> Create(
        std::shared_ptr<StoreV1> backend, std::vector<std::size_t> tensor_sizes,
        std::int32_t device_id, std::size_t slot_num,
        std::size_t stream_number = kDefaultStreamNumber);
    ~Executor();

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    Expected<Detail::TaskHandle> Submit(Detail::TaskDesc task, Operation operation);
    Expected<bool> Check(Detail::TaskHandle task);
    Status Wait(Detail::TaskHandle task);
    void Shutdown();

private:
    Executor(std::shared_ptr<StoreV1> backend, std::vector<std::size_t> tensor_sizes,
             std::int32_t device_id, std::size_t slot_num, std::size_t stream_number);

    struct TaskContext {
        Detail::TaskHandle id{NextId()};
        Detail::TaskDesc desc;
        Operation operation{Operation::LOAD};

        std::mutex stateMutex;  // Protects remaining and error.
        std::size_t remaining{0};
        std::optional<Status> error;

        std::atomic<bool> failed{false};
        std::condition_variable completed;

        static Detail::TaskHandle NextId() noexcept
        {
            static std::atomic<Detail::TaskHandle> id{1};
            auto next = id.fetch_add(1, std::memory_order_relaxed);
            if (next == 0) { next = id.fetch_add(1, std::memory_order_relaxed); }
            return next;
        }
    };

    enum class ShardStage {
        QUEUED,
        IN_FLIGHT,
    };

    struct QueuedShardContext {
        std::shared_ptr<TaskContext> task;
        std::size_t shard_index{0};
    };

    // Always owned by a TransferGroup, which records the shard's parent task.
    struct InFlightShardContext {
        std::size_t shard_index{0};
        BufferPool::Slot slot;
    };

    struct TransferGroup {
        std::shared_ptr<TaskContext> task;
        std::vector<InFlightShardContext> shards;
        Detail::TaskHandle transfer_task{0};
        bool transfer_pending{false};
        std::optional<Status> error;
    };

    struct TransferBatch {
        std::vector<TransferGroup> groups;
    };

    Status ValidateTask(const Detail::TaskDesc& task, Operation operation) const;
    Status Start(std::size_t payload_size, std::size_t slot_num);
    Expected<Detail::TaskDesc> MakeBackendTask(const TransferGroup& group) const;

    Status GatherAsync(const TransferGroup& group, CopyStream& streams);
    Status ScatterAsync(const TransferGroup& group, CopyStream& streams);
    std::string DescribeShards(const TransferGroup& group) const;
    void LogTaskCompletion(const TaskContext& task) const;

    Expected<TransferBatch> AcquireBatch(Operation operation);
    void ReleaseBatch(TransferBatch& batch);

    void CompleteTaskShards(const std::shared_ptr<TaskContext>& task, std::size_t count,
                            const Status& status);
    void DiscardShard(const QueuedShardContext& shard, const Status& status, ShardStage stage);
    // Called only after all worker threads have stopped, when Shutdown is the sole queue consumer.
    void DrainQueue(std::deque<QueuedShardContext>& queue);
    void AssertSchedulerInvariantsLocked() const;

    void DumpLoop(std::promise<Status>& started);
    void LoadLoop(std::promise<Status>& started);

    BufferPool buffer_pool_;
    std::shared_ptr<StoreV1> backend_;
    std::vector<std::size_t> tensor_sizes_;
    std::int32_t device_id_{-1};
    std::size_t stream_number_{0};

    std::shared_mutex taskMapMutex_;  // Protects tasks_ only.
    std::mutex sched_mutex_;
    std::condition_variable slots_ready_;
    std::condition_variable shutdown_completed_;

    std::deque<QueuedShardContext> dump_queue_;
    std::deque<QueuedShardContext> load_queue_;
    std::unordered_map<Detail::TaskHandle, std::shared_ptr<TaskContext>> tasks_;

    std::size_t slot_num_{0};
    std::size_t available_slots_{0};
    std::size_t outstanding_shards_{0};
    std::size_t queued_load_shards_{0};
    std::size_t queued_dump_shards_{0};
    std::size_t in_flight_load_shards_{0};
    std::size_t in_flight_dump_shards_{0};

    bool shutdown_started_{false};
    bool shutdown_complete_{false};
    std::thread dump_thread_;
    std::thread load_thread_;
};

}  // namespace UC::Delegator
