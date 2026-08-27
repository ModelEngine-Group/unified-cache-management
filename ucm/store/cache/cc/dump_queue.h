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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_DUMP_QUEUE_H
#define UNIFIEDCACHE_CACHE_STORE_CC_DUMP_QUEUE_H

#include <atomic>
#include <future>
#include <list>
#include <string>
#include <thread>
#include "copy_stream.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"
#include "trans_buffer.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::CacheStore {

class DumpQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    struct DumpCtx {
        Detail::TaskHandle taskHandle;
        Detail::TaskHandle backendTaskHandle;
        std::vector<TransBuffer::Handle> bufferHandles;
    };
    struct H2HDumpContext {
        TaskPtr task;
        WaiterPtr waiter;
        std::atomic<size_t> pending{0};
        std::atomic<bool> failed{false};
        Detail::TaskDesc backendTaskDesc;
        std::vector<TransBuffer::Handle> bufferHandles;
    };
    using H2HDumpContextPtr = std::shared_ptr<H2HDumpContext>;
    struct H2HDumpJob {
        H2HDumpContextPtr context;
        size_t shardIndex{0};
        size_t handleIndex{0};
    };

private:
    alignas(64) std::atomic_bool stop_{false};
    Detail::TaskHandle finishedBackendTaskHandle_{0};
    TaskIdSet* failureSet_{nullptr};
    TransBuffer* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    int32_t deviceId_{-1};
    std::vector<size_t> tensorSizes_{};
    size_t streamNumber_{1};
    bool useGdr_{false};
    bool cacheIOAggregation_{false};
    bool cacheSdmaDirect_{false};
    bool useHostBuffer_{false};
    size_t h2hQueueDepth_{0};
    std::atomic<size_t> h2hOutstanding_{0};
    std::vector<ssize_t> cpuAffinityCores_{};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<DumpCtx> dumping_;
    std::thread dispatcher_;
    std::thread dumper_;
    ThreadPool<H2HDumpContextPtr> h2hCompletionPool_;
    ThreadPool<H2HDumpJob> h2hCopyPool_;

public:
    ~DumpQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage(std::promise<Status>& started);
    void DispatchOneTask(CopyStream& stream, TaskPair&& pair);
    Status DumpOneTask(CopyStream& stream, TaskPtr task);
    Status DeviceToHostAsync(CopyStream& stream, void** device, void* host);
    Status DispatchH2HDump(TaskPtr task, WaiterPtr waiter);
    void H2HDumpWorker(H2HDumpJob& job);
    void CompleteH2HDump(H2HDumpContextPtr context);
    Status HostToHostGather(const Detail::Shard& shard, void* destination) const;
    bool TryReserveH2HJobs(size_t number);
    void BackendDumpStage();
};

}  // namespace UC::CacheStore

#endif
