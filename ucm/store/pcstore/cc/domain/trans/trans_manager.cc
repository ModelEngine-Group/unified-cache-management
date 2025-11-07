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
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

Status TransManager::Setup(const int32_t deviceId, const size_t streamNumber,
                           const size_t blockSize, const size_t ioSize, const bool ioDirect,
                           const size_t bufferNumber, const SpaceLayout* layout,
                           const size_t timeoutMs)
{
    auto s = this->device_.Setup(deviceId, blockSize, bufferNumber);
    if (s.Failure()) { return s; }
    auto success =
        this->devPool_.SetWorkerFn([this](auto t, auto) { this->DeviceWorker(std::move(t)); })
            .Run();
    if (!success) { return Status::Error(); }
    success = this->filePool_.SetWorkerFn([this](auto t, auto) { this->FileWorker(std::move(t)); })
                  .SetNWorker(streamNumber)
                  .Run();
    if (!success) { return Status::Error(); }
    this->layout_ = layout;
    this->ioSize_ = ioSize;
    this->ioDirect_ = ioDirect;
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
        waiterPtr = std::make_shared<TaskWaiter>(blockNumber, taskPtr->startTp);
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

void TransManager::DeviceWorker(BlockTask&& task)
{
    if (this->failureSet_.Contains(task.owner)) {
        task.done(false);
        return;
    }
    auto number = task.shards.size();
    auto size = this->ioSize_;
    auto done = task.done;
    auto devPtrs = task.shards.data();
    auto hostPtr = (uintptr_t)task.buffer.get();
    auto s = Status::OK();
    if (task.type == TransTask::Type::LOAD) {
        s = this->device_.stream.H2DBatchSync(hostPtr, devPtrs, size, number);
    } else {
        s = this->device_.stream.D2HBatchSync(devPtrs, hostPtr, size, number);
        if (s.Success()) { this->filePool_.Push(std::move(task)); }
    }
    if (s.Failure()) { this->failureSet_.Insert(task.owner); }
    done(s.Success());
    return;
}

void TransManager::FileWorker(BlockTask&& task)
{
    if (this->failureSet_.Contains(task.owner)) {
        task.done(false);
        return;
    }
    auto hostPtr = (uintptr_t)task.buffer.get();
    auto length = this->ioSize_ * task.shards.size();
    if (task.type == TransTask::Type::DUMP) {
        const auto& path = this->layout_->DataFilePath(task.block, true);
        auto s = File::Write(path, 0, length, hostPtr, this->ioDirect_);
        this->layout_->Commit(task.block, s.Success());
        if (s.Failure()) { this->failureSet_.Insert(task.owner); }
        return;
    }
    const auto& path = this->layout_->DataFilePath(task.block, false);
    auto s = File::Read(path, 0, length, hostPtr, this->ioDirect_);
    if (s.Success()) {
        this->devPool_.Push(std::move(task));
        return;
    }
    this->failureSet_.Insert(task.owner);
    task.done(false);
}

void TransManager::Dispatch(TaskPtr task, WaiterPtr waiter)
{
    task->ForEachGroup(
        [task, waiter, this](const std::string& block, std::vector<uintptr_t>& shards) {
            BlockTask blockTask;
            blockTask.owner = task->id;
            blockTask.block = block;
            blockTask.type = task->type;
            auto bufferSize = this->ioSize_ * shards.size();
            std::swap(blockTask.shards, shards);
            blockTask.buffer = this->device_.buffer.GetBuffer(bufferSize);
            blockTask.done = [task, waiter, ioSize = this->ioSize_](bool success) {
                if (!success) {
                    waiter->Done(nullptr);
                } else {
                    waiter->Done([task, ioSize] { UC_DEBUG("{}", task->Epilog(ioSize)); });
                }
            };
            if (task->type == TransTask::Type::DUMP) {
                this->devPool_.Push(std::move(blockTask));
            } else {
                this->filePool_.Push(std::move(blockTask));
            }
        });
}

} // namespace UC
