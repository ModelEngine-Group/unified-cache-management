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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_LOAD_QUEUE_H
#define UNIFIEDCACHE_CACHE_STORE_CC_LOAD_QUEUE_H

#include <atomic>
#include <future>
#include <list>
#include <thread>
#include <vector>
#include "copy_stream.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"
#include "trans_buffer.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::CacheStore {

class LoadQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    struct H2HLoadContext {
        TaskPtr task;
        WaiterPtr waiter;
        std::atomic<size_t> pending{0};
        std::atomic<bool> failed{false};
    };
    using H2HLoadContextPtr = std::shared_ptr<H2HLoadContext>;
    struct ShardTask {
        TaskPtr task;
        Detail::Shard shard;
        TransBuffer::Handle bufferHandle;
        Detail::TaskHandle backendTaskHandle{0};
        WaiterPtr waiter;
        H2HLoadContextPtr h2hContext;
        bool fromPosix{false};
    };

private:
    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    TransBuffer* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    std::vector<size_t> tensorSizes_{};
    size_t nShardPerBlock_{0};
    size_t streamNumber_{1};
    bool useGdr_{false};
    bool cacheIOAggregation_{false};
    bool cacheSdmaDirect_{false};
    bool useHostBuffer_{false};
    size_t h2hQueueDepth_{0};
    std::atomic<size_t> h2hOutstanding_{0};
    std::vector<ssize_t> cpuAffinityCores_{};
    size_t localRankSize_{};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<ShardTask> running_;
    std::thread dispatcher_;
    std::thread transfer_;
    std::vector<ShardTask> holder_;
    ThreadPool<ShardTask> h2hCopyPool_;

public:
    ~LoadQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    void DispatchOneH2HTask(TaskPair&& pair);
    void TransferStage(std::promise<Status>& started);
    void TransferOneTask(CopyStream& stream, ShardTask&& task);
    Status WaitBackendTaskReady(ShardTask& task);
    Status HostToDeviceAsync(CopyStream& stream, void* host, void** device);
    Status HostToHostScatter(void* source, const Detail::Shard& shard) const;
    void H2HLoadWorker(ShardTask& task);
    void FinishH2HLoadShard(const H2HLoadContextPtr& context, bool success);
    bool TryReserveH2HJobs(size_t number);
    void RecordShardResults(const std::vector<ShardTask>& tasks, const ShardTask* extra,
                            bool success) const;
    void RecordFailedShards(size_t count) const;
    void RecordH2dSyncMetrics(double h2dSyncMs) const;
};

}  // namespace UC::CacheStore

#endif
