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
#include "task_manager.h"

namespace UC {

Status TaskManager::Submit(Task&& task, size_t& taskId) noexcept
{
    taskId = task.Id();
    const auto taskStr = task.Str();
    TaskPtr taskPtr = nullptr;
    WaiterPtr waiterPtr = nullptr;
    try {
        taskPtr = std::make_shared<Task>(std::move(task));
        waiterPtr = std::make_shared<TaskWaiter>(0, task.StartTp());
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to submit task({}).", e.what(), taskStr);
        return Status::OutOfMemory();
    }
    std::lock_guard<std::mutex> lg(mutex_);
    const auto& [iter, success] =
        tasks_.emplace(taskId, std::make_pair(std::move(taskPtr), std::move(waiterPtr)));
    if (!success) {
        UC_ERROR("Failed to submit task({}).", taskStr);
        return Status::OutOfMemory();
    }
    auto shards = iter->second.first->Split(queues_.size(), iter->second.second);
    for (auto& shard : shards) {
        auto& q = queues_[qIndex_++];
        if (qIndex_ == queues_.size()) { qIndex_ = 0; }
        q->Push(shard);
    }
    return Status::OK();
}

Status TaskManager::Wait(const size_t taskId) noexcept
{
    TaskPtr task = nullptr;
    WaiterPtr waiter = nullptr;
    {
        std::lock_guard<std::mutex> lg(mutex_);
        auto iter = tasks_.find(taskId);
        if (iter == tasks_.end()) {
            UC_ERROR("Not found task by id({}).", taskId);
            return Status::NotFound();
        }
        task = iter->second.first;
        waiter = iter->second.second;
        tasks_.erase(iter);
    }
    if (!waiter->Wait(timeoutMs_)) {
        UC_ERROR("Task({}) timeout({}).", task->Str(), timeoutMs_);
        failureSet_.Insert(taskId);
        waiter->Wait();
    }
    auto failure = failureSet_.Contains(taskId);
    if (failure) {
        failureSet_.Remove(taskId);
        UC_ERROR("Task({}) failed.", task->Str());
        return Status::Error();
    }
    return Status::OK();
}

Status TaskManager::Check(const size_t taskId, bool& finish) noexcept
{
    std::lock_guard<std::mutex> lg(mutex_);
    auto iter = tasks_.find(taskId);
    if (iter == tasks_.end()) {
        UC_ERROR("Not found task by id({}).", taskId);
        return Status::NotFound();
    }
    finish = iter->second.second->Finish();
    return Status::OK();
}

} // namespace UC
