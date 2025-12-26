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
#include "trans_share_queue.h"
#include "logger/logger.h"
#include "trans/device.h"

namespace UC {

TransShareQueue::~TransShareQueue()
{
    {
        std::lock_guard<std::mutex> lg(this->mutex_);
        this->stop_ = true;
        this->cv_.notify_all();
    }
    for (auto& w : this->threads_) {
        if (w.joinable()) { w.join(); }
    }
}

Status TransShareQueue::Setup(const size_t nSharer, const int32_t deviceId,
                              const size_t streamNumber, const size_t blockSize,
                              const size_t ioSize, const bool ioDirect, const size_t bufferNumber,
                              const SpaceLayout* layout, TaskSet* failureSet,
                              const std::string& uniqueId)
{
    this->deviceId_ = deviceId;
    this->streamNumber_ = streamNumber;
    this->ioSize_ = ioSize;
    this->layout_ = layout;
    this->failureSet_ = failureSet;
    auto status = this->buffer_.Setup(blockSize, bufferNumber, ioDirect, nSharer, uniqueId);
    if (status.Failure()) { return status; }
    std::list<std::promise<Status>> start(streamNumber);
    std::list<std::future<Status>> fut;
    for (auto& s : start) {
        fut.push_back(s.get_future());
        this->threads_.emplace_back([&] { this->WorkerLoop(s); });
    }
    for (auto& f : fut) {
        if (status.Failure()) { break; }
        status = f.get();
    }
    return status;
}

void TransShareQueue::Dispatch(TaskPtr task, WaiterPtr waiter)
{
    std::vector<BlockTask> tasks;
    task->ForEachGroup(
        [task, waiter, &tasks, this](const std::string& block, std::vector<uintptr_t>& shards) {
            BlockTask blockTask;
            blockTask.reader =
                this->buffer_.MakeReader(block, this->layout_->DataFilePath(block, false));
            blockTask.owner = task->id;
            std::swap(blockTask.shards, shards);
            blockTask.done = [task, waiter, ioSize = this->ioSize_](bool success) {
                if (!success) {
                    waiter->Done(nullptr);
                } else {
                    waiter->Done([task, ioSize] { UC_DEBUG("{}", task->Epilog(ioSize)); });
                }
            };
            tasks.push_back(std::move(blockTask));
        });

    {
        std::lock_guard<std::mutex> lg(this->mutex_);
        for (auto& task : tasks) { this->wait_.push_back(std::move(task)); }
    }

    this->cv_.notify_all();
}

void TransShareQueue::WorkerLoop(std::promise<Status>& status)
{
    Trans::Device device;
    auto s = device.Setup(deviceId_);
    if (s.Failure()) {
        UC_ERROR("Failed({}) to set context on device({}).", s.ToString(), deviceId_);
        status.set_value(Status::Error());
        return;
    }
    auto stream = device.MakeStream();
    if (!stream) {
        UC_ERROR("Failed to create stream on device({}).", deviceId_);
        status.set_value(Status::Error());
        return;
    }
    status.set_value(Status::OK());
    while (!stop_) { Worker(*stream); }
}

void TransShareQueue::Worker(Trans::Stream& stream)
{
    std::unique_lock<std::mutex> ul{this->mutex_};
    if (this->load_.empty() && this->wait_.empty()) {
        this->cv_.wait(
            ul, [this] { return this->stop_ || !this->load_.empty() || !this->wait_.empty(); });
    }
    if (this->stop_) { return; }
    for (auto iter = this->load_.begin(); iter != this->load_.end(); iter++) {
        auto s = iter->reader->Ready4Read();
        if (s != Status::Retry()) {
            auto task = std::move(*iter);
            this->load_.erase(iter);
            ul.unlock();
            this->HandleReadyTask(s, task, stream);
            return;
        }
    }
    if (this->load_.size() >= this->streamNumber_) { return; }
    if (this->wait_.empty()) { return; }
    auto task = std::move(this->wait_.front());
    this->wait_.pop_front();
    ul.unlock();
    this->HandleLoadTask(task, stream);
}

void TransShareQueue::HandleReadyTask(Status s, BlockTask& task, Trans::Stream& stream)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }
    if (s.Success()) {
        auto host = (void*)task.reader->GetData();
        auto device = (void**)task.shards.data();
        auto status = stream.HostToDeviceAsync(host, device, this->ioSize_, task.shards.size());
        if (status.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to copy data from host to device.", status.ToString());
            s = Status::Error();
        }
    }
    if (s.Failure()) { this->failureSet_->Insert(task.owner); }
    task.done(s.Success());
}

void TransShareQueue::HandleLoadTask(BlockTask& task, Trans::Stream& stream)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }
    auto s = task.reader->Ready4Read();
    if (s == Status::Retry()) {
        std::lock_guard<std::mutex> lg{this->mutex_};
        this->load_.push_back(task);
        this->cv_.notify_one();
        return;
    }
    this->HandleReadyTask(s, task, stream);
}

}  // namespace UC
