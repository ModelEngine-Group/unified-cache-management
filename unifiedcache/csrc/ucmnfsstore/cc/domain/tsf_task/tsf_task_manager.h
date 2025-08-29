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
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR O THER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#ifndef UNIFIEDCACHE_TSF_TASK_MANAGER_H
#define UNIFIEDCACHE_TSF_TASK_MANAGER_H

#include <memory>
#include <unordered_map>
#include <vector>
#include "tsf_task_queue.h"

namespace UC {

class TsfTaskManager {
public:
    Status Setup(const int32_t deviceId, const size_t streamNumber, const size_t bufferSize, const size_t bufferNumber,
                 const size_t timeoutMs, const SpaceLayout* layout);
    Status Submit(std::list<TsfTask>& tasks, const size_t size, const size_t number, const std::string& brief,
                  size_t& taskId);
    Status Wait(const size_t taskId);

private:
    void Dispatch(std::list<TsfTask>& tasks, std::vector<std::list<TsfTask>>& targets, const size_t taskId,
                  std::shared_ptr<TsfTaskWaiter> waiter) const;

private:
    std::mutex _mutex;
    TsfTaskSet _failureSet;
    std::unordered_map<size_t, std::shared_ptr<TsfTaskWaiter>> _waiters;
    std::vector<std::unique_ptr<TsfTaskQueue>> _queues;
    size_t _qIdx{0};
    size_t _taskIdSeed{0};
    size_t _timeoutMs{0};
};

} // namespace UC

#endif
