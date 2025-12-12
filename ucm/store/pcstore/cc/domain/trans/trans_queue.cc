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
#include "trans_queue.h"
#include "file/file.h"
#include "logger/logger.h"
#include "trans/device.h"

namespace UC {

void TransQueue::DeviceWorker(BlockTask&& task)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }
    auto number = task.shards.size();
    auto size = this->ioSize_;
    auto done = task.done;
    auto devPtrs = (void**)task.shards.data();
    auto hostPtr = task.buffer.get();
    auto s = Status::OK();
    if (task.type == TransTask::Type::LOAD) {
        s = stream_->HostToDevice(hostPtr, devPtrs, size, number);
    } else {
        s = stream_->DeviceToHost(devPtrs, hostPtr, size, number);
        if (s.Success()) { this->filePool_.Push(std::move(task)); }
    }
    if (s.Failure()) { this->failureSet_->Insert(task.owner); }
    done(s.Success());
    return;
}

void TransQueue::FileWorker(BlockTask&& task)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }
    auto hostPtr = (uintptr_t)task.buffer.get();
    auto length = this->ioSize_ * task.shards.size();
    if (task.type == TransTask::Type::DUMP) {
        const auto& path = this->layout_->DataFilePath(task.block, true);
        auto s = File::Write(path, 0, length, hostPtr, this->ioDirect_, true);
        this->layout_->Commit(task.block, s.Success());
        return;
    }
    const auto& path = this->layout_->DataFilePath(task.block, false);
    auto s = File::Read(path, 0, length, hostPtr, this->ioDirect_);
    if (s.Success()) {
        this->devPool_.Push(std::move(task));
        return;
    }
    this->failureSet_->Insert(task.owner);
    task.done(false);
}

Status TransQueue::Setup(const int32_t deviceId, const size_t streamNumber, const size_t blockSize,
                         const size_t ioSize, const bool ioDirect, const size_t bufferNumber,
                         const SpaceLayout* layout, TaskSet* failureSet_,
                         const bool scatterGatherEnable)
{
    Trans::Device device;
    auto ts = device.Setup(deviceId);
    if (ts.Failure()) {
        UC_ERROR("Failed({}) to set context on device({}).", ts.ToString(), deviceId);
        return Status::Error();
    }
    buffer_ = device.MakeBuffer();
    stream_ = device.MakeStream();
    if (!buffer_ || !stream_) {
        UC_ERROR("Failed to make buffer and stream on device({}).", deviceId);
        return Status::Error();
    }
    if (scatterGatherEnable) {
        devBuffer_ = device.MakeBuffer();
        smStream_ = device.MakeSMStream();
        if (!devBuffer_ || !smStream_) {
            UC_ERROR("Failed to make devBuffer and smStream on device({}).", deviceId);
            return Status::Error();
        }
    }
    ts = buffer_->MakeHostBuffers(blockSize, bufferNumber);
    if (ts.Failure()) {
        UC_ERROR("Failed({}) to make host buffer({},{}).", ts.ToString(), blockSize, bufferNumber);
        return Status::Error();
    }
    auto success = this->devPool_
                       .SetWorkerInitFn([deviceId, &device](auto&) {
                           auto ts = device.Setup(deviceId);
                           return ts.Success();
                       })
                       .SetWorkerFn([this](auto t, auto) { this->DeviceWorker(std::move(t)); })
                       .Run();
    if (!success) { return Status::Error(); }
    success = this->filePool_.SetWorkerFn([this](auto t, auto) { this->FileWorker(std::move(t)); })
                  .SetNWorker(streamNumber)
                  .Run();
    if (!success) { return Status::Error(); }
    this->layout_ = layout;
    this->ioSize_ = ioSize;
    this->ioDirect_ = ioDirect;
    this->failureSet_ = failureSet_;
    this->scatterGatherEnable_ = scatterGatherEnable;
    return Status::OK();
}

void TransQueue::Dispatch(TaskPtr task, WaiterPtr waiter)
{
    if (task->type == TransTask::Type::DUMP) {
        if (this->scatterGatherEnable_) {
            this->DispatchSatterGatherDump(task, waiter);
        } else {
            this->DispatchDump(task, waiter);
        }
        return;
    }
    task->ForEachGroup(
        [task, waiter, this](const std::string& block, std::vector<uintptr_t>& shards) {
            BlockTask blockTask;
            blockTask.owner = task->id;
            blockTask.block = block;
            blockTask.type = task->type;
            auto bufferSize = this->ioSize_ * shards.size();
            std::swap(blockTask.shards, shards);
            blockTask.buffer = buffer_->GetHostBuffer(bufferSize);
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

void TransQueue::DispatchDump(TaskPtr task, WaiterPtr waiter)
{
    std::vector<BlockTask> blocks;
    blocks.reserve(task->GroupNumber());
    task->ForEachGroup(
        [task, &blocks, this](const std::string& block, std::vector<uintptr_t>& shards) {
            BlockTask blockTask;
            blockTask.owner = task->id;
            blockTask.block = block;
            blockTask.type = task->type;
            auto bufferSize = this->ioSize_ * shards.size();
            blockTask.buffer = buffer_->GetHostBuffer(bufferSize);
            std::swap(blockTask.shards, shards);
            auto device = (void**)blockTask.shards.data();
            auto host = blockTask.buffer.get();
            stream_->DeviceToHostAsync(device, host, this->ioSize_, blockTask.shards.size());
            blocks.push_back(std::move(blockTask));
        });
    auto s = stream_->Synchronized();
    if (s.Failure()) { this->failureSet_->Insert(task->id); }
    for (auto&& block : blocks) {
        if (s.Failure()) {
            waiter->Done(nullptr);
            return;
        }
        this->filePool_.Push(std::move(block));
        waiter->Done([task, ioSize = this->ioSize_] { UC_DEBUG("{}", task->Epilog(ioSize)); });
    }
}

void TransQueue::DispatchSatterGatherDump(TaskPtr task, WaiterPtr waiter)
{
    std::vector<BlockTask> blocks;
    blocks.reserve(task->GroupNumber());
    std::vector<std::shared_ptr<void>> addrs;
    addrs.reserve(task->GroupNumber());
    task->ForEachGroup(
        [task, &blocks, &addrs, this](const std::string& block, std::vector<uintptr_t>& shards) {
            BlockTask blockTask;
            blockTask.owner = task->id;
            blockTask.block = block;
            blockTask.type = task->type;
            auto number = shards.size();
            auto bufferSize = this->ioSize_ * number;
            blockTask.buffer = buffer_->GetHostBuffer(bufferSize);
            std::swap(blockTask.shards, shards);
            auto device = (void*)blockTask.shards.data();
            auto host = blockTask.buffer.get();
            auto devAddr = this->devBuffer_->MakeDeviceBuffer(sizeof(void*) * number);
            smStream_->HostToDeviceAsync(device, devAddr.get(), sizeof(void*) * number);
            smStream_->DeviceToHostAsync((void**)devAddr.get(), host, this->ioSize_, number);
            addrs.push_back(devAddr);
            blocks.push_back(std::move(blockTask));
        });
    auto s = smStream_->Synchronized();
    if (s.Failure()) { this->failureSet_->Insert(task->id); }
    for (auto&& block : blocks) {
        if (s.Failure()) {
            waiter->Done(nullptr);
            return;
        }
        this->filePool_.Push(std::move(block));
        waiter->Done([task, ioSize = this->ioSize_] { UC_DEBUG("{}", task->Epilog(ioSize)); });
    }
}

} // namespace UC
