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
#ifndef UNIFIEDCACHE_POSIX_QUEUE_H
#define UNIFIEDCACHE_POSIX_QUEUE_H

#include "device/idevice.h"
#include "space/space_layout.h"
#include "status/status.h"
#include "task_queue.h"
#include "task_set.h"
#include "thread/thread_pool.h"

namespace UC {

class PosixQueue : public TaskQueue {
    using Device = std::unique_ptr<IDevice>;
    int32_t deviceId_{-1};
    size_t bufferSize_{0};
    size_t bufferNumber_{0};
    TaskSet* failureSet_{nullptr};
    const SpaceLayout* layout_{nullptr};
    bool useDirect_{false};
    ThreadPool<Task::Shard, Device> backend_{};

public:
    Status Setup(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber,
                 TaskSet* failureSet, const SpaceLayout* layout, const size_t timeoutMs, bool useDirect = false);
    void Push(std::list<Task::Shard>& shards) noexcept override;

private:
    bool Init(Device& device);
    void Exit(Device& device);
    void Work(Task::Shard&& shard, const Device& device);
    void Done(Task::Shard& shard, const Device& device, const bool success);
    Status D2S(Task::Shard& shard, const Device& device);
    Status S2D(Task::Shard& shard, const Device& device);
    Status H2S(Task::Shard& shard);
    Status S2H(Task::Shard& shard);
};

} // namespace UC

#endif
