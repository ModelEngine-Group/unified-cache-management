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
#include "tsf_task_manager.h"

namespace UCM {

Status TsfTaskManager::Setup(const int32_t deviceId, const size_t streamNumber, const size_t bufferSize,
                             const size_t bufferNumber)
{
    this->_queues.reserve(streamNumber);
    for (size_t i = 0; i < streamNumber; ++i) {
        auto& queue = this->_queues.emplace_back(std::make_unique<TsfTaskQueue>());
        auto status = queue->Setup(deviceId, bufferSize, bufferNumber);
        if (status.Failure()) { return status; }
    }
    return Status::OK();
}

Status TsfTaskManager::Submit(std::list<TsfTask>& tasks, const size_t size,
                              const size_t number, size_t& taskId)
{
    std::unique_lock<std::mutex> lk(this->_mutex);
    taskId = ++this->_taskIdSeed;
    auto [iter, success] = this->_waiters.emplace(taskId, std::make_shared<TsfTaskWaiter>(taskId, size, number));
    if (!success) { return Status::OutOfMemory(); }
    std::vector<std::list<TsfTask>> lists;
    this->Dispatch(tasks, lists, taskId, iter->second);
    for (size_t i = 0; i < lists.size(); i++) {
        if (lists[i].empty()) { continue; }
        this->_queues[this->_qIdx]->Push(lists[i]);
        this->_qIdx = (this->_qIdx + 1) % this->_queues.size();
    }
    return Status::OK();
}

Status TsfTaskManager::Wait(const size_t taskId)
{
    std::shared_ptr<TsfTaskWaiter> waiter = nullptr;
    {
        std::unique_lock<std::mutex> lk(this->_mutex);
        auto iter = this->_waiters.find(taskId);
        if (iter == this->_waiters.end()) { return Status::NotFound(); }
        waiter = iter->second;
        this->_waiters.erase(iter);
    }
    waiter->Wait();
    return Status::OK();
}

void TsfTaskManager::Dispatch(std::list<TsfTask>& tasks, std::vector<std::list<TsfTask>>& targets,
                              const size_t taskId, std::shared_ptr<TsfTaskWaiter> waiter) const
{
    auto qNumber = this->_queues.size();
    auto index = size_t(0);
    targets.resize(qNumber);
    auto it = tasks.begin();
    while (it != tasks.end()) {
        auto next = std::next(it);
        it->owner = taskId;
        it->waiter = waiter;
        auto& target = targets[index % qNumber];
        target.splice(target.end(), tasks, it);
        index++;
        it = next;
    }
}

} // namespace UCM
