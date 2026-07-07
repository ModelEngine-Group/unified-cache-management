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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_LOAD_QUEUE_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_LOAD_QUEUE_H

#include <atomic>
#include <future>
#include <memory>
#include <thread>
#include <vector>
#include "copy_stream.h"
#include "global_config.h"
#include "host_buffer_pool.h"
#include "real_client.h"
#include "template/hashset.h"
#include "template/spsc_ring_queue.h"
#include "thread/latch.h"
#include "trans_task.h"
#include "type/types.h"
#include "ucmstore_v1.h"

namespace UC::MooncakeStore {

class ShareLoadQueue;

class LoadQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;

    struct ShardTask {
        Detail::TaskHandle taskHandle;
        Detail::TaskHandle backendTaskHandle{0};
        TransShard shard;
        HostBufferPool::Handle hostBuf;
        WaiterPtr waiter;
    };

public:
    ~LoadQueue();

    Status Setup(const Config& config, TaskIdSet* failureSet,
                 std::shared_ptr<mooncake::RealClient> realClient, StoreV1* backend,
                 HostBufferPool* bufPool, ShareLoadQueue* shareLoadQ = nullptr);
    void Close();
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void DispatchStage();
    void DispatchOneTask(TaskPair&& pair);
    std::vector<int> TryMooncakeLoad(TaskPtr task);
    bool SubmitMissShards(TaskPtr task, WaiterPtr waiter, const std::vector<int>& results,
                          size_t missCount);
    void TransferStage(std::promise<Status>& started);
    void TransferOneTask(CopyStream& stream, ShardTask&& task);

    Status HostToDeviceScatterAsync(std::shared_ptr<Trans::Stream> stream, void* host,
                                    void** device);

private:
    alignas(64) std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    std::shared_ptr<mooncake::RealClient> realClient_;
    StoreV1* backend_{nullptr};
    HostBufferPool* bufPool_{nullptr};
    ShareLoadQueue* shareLoadQ_{nullptr};
    std::vector<size_t> tensorSizes_;
    int32_t deviceId_{-1};
    size_t streamNumber_{1};
    std::vector<ssize_t> cpuAffinityCores_;

    SpscRingQueue<TaskPair> waiting_;
    SpscRingQueue<ShardTask> running_;
    std::vector<ShardTask> holder_;

    std::thread dispatcher_;
    std::thread transfer_;
};

}  // namespace UC::MooncakeStore

#endif
