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
#include "trans_manager.h"
#include "logger/logger.h"

namespace UC {

Status TransManager::Setup(const size_t rankSize, const int32_t deviceId, const size_t streamNumber,
                           const size_t blockSize, const size_t ioSize, const bool ioDirect,
                           const size_t bufferNumber, const SpaceLayout* layout,
                           const size_t timeoutMs, const bool scatterGatherEnable,
                           const std::string& uniqueId)
{
    auto s = Status::OK();
    if (rankSize > 1) {
        s = this->shareQueue_.Setup(rankSize, deviceId, streamNumber, blockSize, ioSize, ioDirect,
                                    bufferNumber, layout, &this->failureSet_, uniqueId);
        if (s.Failure()) { return s; }
    }
    s = this->queue_.Setup(deviceId, streamNumber, blockSize, ioSize, ioDirect, bufferNumber,
                           layout, &this->failureSet_, scatterGatherEnable, timeoutMs);
    if (s.Failure()) { return s; }
    this->rankSize_ = rankSize;
    this->timeoutMs_ = timeoutMs;
    return Status::OK();
}

Status TransManager::Submit(TransTask task, size_t& taskId) noexcept
{
    taskId = task.id;
    const auto taskStr = task.Str();
    const auto blockNumber = task.GroupNumber();
    TaskPtr taskPtr = nullptr;
    WaiterPtr waiterPtr = nullptr;
    try {
        taskPtr = std::make_shared<TransTask>(std::move(task));
        waiterPtr = std::make_shared<TaskWaiter>(blockNumber, taskPtr->startTp);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to submit task({}).", e.what(), taskStr);
        return Status::OutOfMemory();
    }
    std::unique_lock<std::mutex> lg(mutex_);
    const auto& [iter, success] = tasks_.emplace(taskId, std::make_pair(taskPtr, waiterPtr));
    if (!success) {
        UC_ERROR("Failed to submit task({}).", taskStr);
        return Status::OutOfMemory();
    }
    lg.unlock();
    if (this->rankSize_ > 1 && iter->second.first->type == TransTask::Type::LOAD) {
        this->shareQueue_.Dispatch(iter->second.first, iter->second.second);
        return Status::OK();
    }
    this->queue_.Dispatch(iter->second.first, iter->second.second);
    return Status::OK();
}

Status TransManager::Wait(const size_t taskId) noexcept
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
        failureSet_.Remove(taskId);
        return Status::Timeout();
    }
    auto failure = failureSet_.Contains(taskId);
    if (failure) {
        failureSet_.Remove(taskId);
        UC_ERROR("Task({}) failed.", task->Str());
        return Status::Error();
    }
    return Status::OK();
}

Status TransManager::Check(const size_t taskId, bool& finish) noexcept
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

}  // namespace UC
