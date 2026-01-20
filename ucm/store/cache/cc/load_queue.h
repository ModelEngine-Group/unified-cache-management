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

#include <future>
#include <thread>
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans/stream.h"
#include "trans_buffer.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::CacheStore {

class LoadQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    struct ShardTask {
        Detail::TaskHandle taskHandle;
        Detail::Shard shard;
        TransBuffer::Handle bufferHandle;
        Detail::TaskHandle backendTaskHandle;
        WaiterPtr waiter;
    };

private:
    alignas(64) std::atomic_bool stop_{false};
    Detail::TaskHandle finishedBackendTaskHandle_{0};
    TaskIdSet* failureSet_{nullptr};
    TransBuffer* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<ShardTask> running_;
    std::thread dispatcher_;
    std::thread transfer_;
    std::vector<ShardTask> holder_;

public:
    ~LoadQueue();
    Status Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer);
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    void TransferStage(int32_t deviceId, size_t tensorSize, std::promise<Status>& started);
    void TransferOneTask(Trans::Stream* stream, size_t tensorSize, ShardTask&& task);
    Status WaitBackendTaskReady(ShardTask& task);
};

}  // namespace UC::CacheStore

#endif
