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
#ifndef UNIFIEDCACHE_TRAN_QUEUE_H
#define UNIFIEDCACHE_TRAN_QUEUE_H

#include "space/space_layout.h"
#include "task/task_set.h"
#include "task/task_waiter.h"
#include "thread/thread_pool.h"
#include "trans/buffer.h"
#include "trans/stream.h"
#include "trans_task.h"

namespace UC {

class TransQueue {
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<TaskWaiter>;
    struct BlockTask {
        size_t owner;
        std::string block;
        TransTask::Type type;
        std::vector<uintptr_t> shards;
        std::shared_ptr<void> buffer;
        std::function<void(bool)> done;
    };
    void DeviceWorker(BlockTask&& task);
    void FileWorker(BlockTask&& task);

public:
    Status Setup(const int32_t deviceId, const size_t streamNumber, const size_t blockSize,
                 const size_t ioSize, const bool ioDirect, const size_t bufferNumber,
                 const SpaceLayout* layout, TaskSet* failureSet_, const bool scatterGatherEnable);
    void Dispatch(TaskPtr task, WaiterPtr waiter);
    void DispatchDump(TaskPtr task, WaiterPtr waiter);
    void DispatchSatterGatherDump(TaskPtr task, WaiterPtr waiter);

private:
    std::unique_ptr<Trans::Buffer> buffer_{nullptr};
    std::unique_ptr<Trans::Stream> stream_{nullptr};
    std::unique_ptr<Trans::Buffer> devBuffer_{nullptr};
    std::unique_ptr<Trans::Stream> smStream_{nullptr};
    const SpaceLayout* layout_;
    size_t ioSize_;
    bool ioDirect_;
    ThreadPool<BlockTask> devPool_;
    ThreadPool<BlockTask> filePool_;
    TaskSet* failureSet_;
    bool scatterGatherEnable_;
};

} // namespace UC

#endif
