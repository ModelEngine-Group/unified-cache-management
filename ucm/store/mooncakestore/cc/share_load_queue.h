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
#ifndef UNIFIEDCACHE_MOONCAKE_STORE_CC_SHARE_LOAD_QUEUE_H
#define UNIFIEDCACHE_MOONCAKE_STORE_CC_SHARE_LOAD_QUEUE_H

#include <atomic>
#include <condition_variable>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "global_config.h"
#include "share_buffer.h"
#include "template/hashset.h"
#include "thread/latch.h"
#include "trans/stream.h"
#include "trans_task.h"
#include "type/types.h"
#include "ucmstore_v1.h"

namespace UC::MooncakeStore {

class ShareLoadQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using TaskIdSet = HashSet<Detail::TaskHandle>;

    struct BlockTask {
        Detail::TaskHandle taskHandle;
        std::shared_ptr<ShareBuffer::Reader> reader;
        TransShard shard;
        Detail::TaskHandle backendTaskHandle{0};
        std::function<void(bool)> done;
    };

public:
    ~ShareLoadQueue();

    Status Setup(const Config& config, TaskIdSet* failureSet, StoreV1* backend);
    void Close();
    void Submit(TaskPtr task, WaiterPtr waiter);

private:
    void WorkerLoop(std::promise<Status>& status);
    void Worker(Trans::Stream& stream);
    void HandleLoadTask(BlockTask& task, Trans::Stream& stream);
    void HandleReadyTask(Status s, BlockTask& task, Trans::Stream& stream);
    void HandleBackendComplete(BlockTask& task, Trans::Stream& stream);
    Status HostToDeviceScatterAsync(Trans::Stream& stream, void* host, void** device);
    size_t BlockBytes() const;
    void RecordH2dMetrics(double h2dMs) const;

private:
    std::atomic_bool stop_{false};
    TaskIdSet* failureSet_{nullptr};
    StoreV1* backend_{nullptr};
    ShareBuffer buffer_;
    int32_t deviceId_{-1};
    std::vector<size_t> tensorSizes_;
    size_t streamNumber_{1};

    std::mutex mutex_;
    std::condition_variable cv_;
    std::list<BlockTask> loadList_;
    std::list<BlockTask> waitList_;
    std::list<std::thread> threads_;
};

}  // namespace UC::MooncakeStore

#endif
