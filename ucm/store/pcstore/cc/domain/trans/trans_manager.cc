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

Status TransManager::Setup(const int32_t deviceId, const size_t streamNumber,
                           const size_t blockSize, const size_t ioSize, const size_t bufferNumber,
                           const SpaceLayout* layout, const size_t timeoutMs)
{
    this->device_ = DeviceFactory::Make(deviceId, blockSize, bufferNumber);
    if (!this->device_) { return Status::OutOfMemory(); }
    auto s = this->device_->Setup();
    if (s.Failure()) { return s; }
    auto success =
        this->devPool_.SetWorkerFn([this](auto& t, auto) { this->DeviceWorker(t); }).Run();
    if (!success) { return Status::Error(); }
    success = this->filePool_.SetWorkerFn([this](auto& t, auto) { this->FileWorker(t); })
                  .SetNWorker(streamNumber)
                  .Run();
    if (!success) { return Status::Error(); }
    this->layout_ = layout;
    this->ioSize_ = ioSize;
    this->timeoutMs_ = timeoutMs_;
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
        waiterPtr = std::make_shared<TaskWaiter>(blockNumber);
    } catch (const std::exception& e) {
        UC_ERROR("Failed({}) to submit task({}).", e.what(), taskStr);
        return Status::OutOfMemory();
    }
    std::lock_guard<std::mutex> lg(mutex_);
    const auto& [iter, success] = tasks_.emplace(taskId, std::make_pair(taskPtr, waiterPtr));
    if (!success) {
        UC_ERROR("Failed to submit task({}).", taskStr);
        return Status::OutOfMemory();
    }
    Dispatch(iter->second.first, iter->second.second);
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

void TransManager::DeviceWorker(BlockTask& task)
{
    if (this->failureSet_.Contains(task.owner)) {
        task.waiter->Done(nullptr);
        return;
    }
    if (task.type == TransTask::Type::DUMP) {
    } else {
    }
}

void TransManager::FileWorker(BlockTask& task)
{
    if (this->failureSet_.Contains(task.owner)) {
        task.waiter->Done(nullptr);
        return;
    }
    if (task.type == TransTask::Type::DUMP) {
    } else {
    }
}

void TransManager::Dispatch(TaskPtr task, WaiterPtr waiter)
{
    task->ForEachGroup([type = task->type, owner = task->id, waiter,
                        this](const std::string& block, std::vector<uintptr_t>& shards) {
        BlockTask blockTask;
        blockTask.owner = owner;
        blockTask.block = block;
        blockTask.type = type;
        std::swap(blockTask.shards, shards);
        blockTask.buffer = this->device_->GetBuffer(0); // fixme
        blockTask.waiter = waiter;
        if (type == TransTask::Type::DUMP) {
            this->devPool_.Push(std::move(blockTask));
        } else {
            this->filePool_.Push(std::move(blockTask));
        }
    });
}

} // namespace UC
