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
#ifndef UNIFIEDCACHE_TRANS_SHARE_QUEUE_H
#define UNIFIEDCACHE_TRANS_SHARE_QUEUE_H

#include <future>
#include <list>
#include <memory>
#include <thread>
#include "share_buffer.h"
#include "space/space_layout.h"
#include "task/task_set.h"
#include "task/task_waiter.h"
#include "trans/stream.h"
#include "trans_task.h"

namespace UC {

class TransShareQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<TaskWaiter>;
    struct BlockTask {
        std::shared_ptr<ShareBuffer::Reader> reader;
        size_t owner;
        std::vector<uintptr_t> shards;
        std::function<void(bool)> done;
    };
    int32_t deviceId_;
    size_t streamNumber_;
    size_t ioSize_;
    ShareBuffer buffer_;
    const SpaceLayout* layout_;
    TaskSet* failureSet_;
    std::atomic_bool stop_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::list<BlockTask> load_;
    std::list<BlockTask> wait_;
    std::list<std::thread> threads_;

public:
    ~TransShareQueue();
    Status Setup(const int32_t deviceId, const size_t streamNumber, const size_t blockSize,
                 const size_t ioSize, const bool ioDirect, const size_t bufferNumber,
                 const SpaceLayout* layout, TaskSet* failureSet, const std::string& uniqueId);
    void Dispatch(TaskPtr task, WaiterPtr waiter);

private:
    void WorkerLoop(std::promise<Status>& status);
    void Worker(Trans::Stream& stream);
    void HandleReadyTask(Status s, BlockTask& task, Trans::Stream& stream);
    void HandleLoadTask(BlockTask& task, Trans::Stream& stream);
};

}  // namespace UC

#endif
