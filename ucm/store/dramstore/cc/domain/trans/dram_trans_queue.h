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
#ifndef UNIFIEDCACHE_DRAM_TRANS_QUEUE_H
#define UNIFIEDCACHE_DRAM_TRANS_QUEUE_H

#include "device/idevice.h"
#include "status/status.h"
#include "task_queue.h"
#include "task_set.h"
#include "thread/thread_pool.h"
#include "memory/memory_pool.h"

namespace UC {

class DramTransQueue : public TaskQueue {
    using Device = std::unique_ptr<IDevice>;
    int32_t deviceId_{-1};
    TaskSet* failureSet_{nullptr};
    const MemoryPool* memPool_{nullptr};
    ThreadPool<Task::Shard, Device> backend_{}; 
    // 之后要搞IO聚合（即调用H2DBatch/D2HBatch这些接口）的话，这里ThreadPool里的第一个范型，大概就需要从Task::Shard变成std::list<Task::Shard>吧，以及重写一下Work, Done这两个函数。
    // 根据ThreadPool的第一个范型的类型，期望在调用ThreadPool::Push()时，
    // 调用到void Push(Task&& task)而不是void Push(std::list<Task>& tasks)这个方法，将一个std::list<Task::Shard>当成线程池任务队列中的一个任务（即改变任务的粒度）

public:
    Status Setup(const int32_t deviceId, 
                 TaskSet* failureSet, 
                 const MemoryPool* memPool, 
                 const size_t timeoutMs);
    void Push(std::list<Task::Shard>& shards) noexcept override;

private:
    bool Init(Device& device);
    void Exit(Device& device);
    void Work(Task::Shard& shard, const Device& device);
    void Done(Task::Shard& shard, const Device& device, const bool success);
    Status H2D(Task::Shard& shard, const Device& device);
    Status D2H(Task::Shard& shard, const Device& device);
};

} // namespace UC

#endif
