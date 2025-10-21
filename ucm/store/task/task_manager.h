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
#ifndef UNIFIEDCACHE_TASK_MANAGER_H
#define UNIFIEDCACHE_TASK_MANAGER_H

#include <unordered_map>
#include "status/status.h"
#include "task_queue.h"
#include "task_set.h"

namespace UC {

class TaskManager {
    using TaskPtr = std::shared_ptr<Task>;
    using WaiterPtr = std::shared_ptr<TaskWaiter>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using QueuePtr = std::shared_ptr<TaskQueue>;

public:
    virtual ~TaskManager() = default;
    virtual Status Submit(Task&& task, size_t& taskId) noexcept;
    virtual Status Wait(const size_t taskId) noexcept;
    virtual Status Check(const size_t taskId, bool& finish) noexcept;

protected:
    std::mutex mutex_;
    std::unordered_map<size_t, TaskPair> tasks_;
    size_t qIndex_{0};
    std::vector<QueuePtr> queues_;
    size_t timeoutMs_{0};
    TaskSet failureSet_;
};

} // namespace UC

#endif
