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
#include "trans_s2d_pool.h"
#include "device.h"
#include "logger/logger.h"

namespace UC {

TransS2DPool::~TransS2DPool()
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

Status TransS2DPool::Setup(const size_t nSharer, const int32_t deviceId, const size_t streamNumber,
                           const size_t blockSize, const size_t ioSize, const bool ioDirect,
                           const SpaceLayout* layout, TaskSet* failureSet)
{
    this->nSharer_ = nSharer;
    this->deviceId_ = deviceId;
    this->streamNumber_ = streamNumber;
    this->blockSize_ = blockSize;
    this->ioSize_ = ioSize;
    this->ioDirect_ = ioDirect;
    this->layout_ = layout;
    this->failureSet_ = failureSet;
    std::list<std::promise<Status>> start(streamNumber);
    std::list<std::future<Status>> fut;
    for (auto& s : start) {
        fut.push_back(s.get_future());
        this->threads_.emplace_back([&] { this->WorkerLoop(s); });
    }
    auto status = Status::OK();
    for (auto& f : fut) {
        if (status.Failure()) { break; }
        status = f.get();
    }
    return status;
}

void TransS2DPool::Dispatch(TaskPtr task, WaiterPtr waiter)
{
    std::lock_guard<std::mutex> lg(this->mutex_);
    task->ForEachGroup(
        [task, waiter, this](const std::string& block, std::vector<uintptr_t>& shards) {
            BlockTask blockTask{block, this->layout_->DataFilePath(block, false), this->blockSize_,
                                this->ioDirect_, this->nSharer_};
            blockTask.owner = task->id;
            std::swap(blockTask.shards, shards);
            blockTask.done = [task, waiter, ioSize = this->ioSize_](bool success) {
                if (!success) {
                    waiter->Done(nullptr);
                } else {
                    waiter->Done([task, ioSize] { UC_DEBUG("{}", task->Epilog(ioSize)); });
                }
            };
            this->wait_.push_back(std::move(blockTask));
        });
    this->cv_.notify_all();
}

void TransS2DPool::WorkerLoop(std::promise<Status>& status)
{
    auto s = Status::OK();
    Stream stream;
    do {
        if ((s = Device::Setup(this->deviceId_)).Failure()) { break; }
        if ((s = stream.Setup()).Failure()) { break; }
    } while (0);
    status.set_value(s);
    if (s.Failure()) { return; }
    for (;;) { this->Worker(stream); }
}

void TransS2DPool::Worker(Stream& stream)
{
    constexpr auto interval = std::chrono::milliseconds(10);
    std::unique_lock<std::mutex> ul{this->mutex_};
    this->cv_.wait_for(ul, interval, [this] {
        return this->stop_ || !this->load_.empty() || !this->wait_.empty();
    });
    if (this->stop_) { return; }
    auto iter = this->load_.begin();
    while (iter != this->load_.end()) {
        auto s = iter->reader.Ready4Read();
        if (s != Status::Retry()) {
            auto&& task = std::move(*iter);
            this->load_.erase(iter);
            ul.unlock();
            this->HandleReadyTask(s, std::move(task), stream);
            return;
        }
    }
    if (this->load_.size() >= this->streamNumber_) { return; }
    if (this->wait_.empty()) { return; }
    auto&& task = std::move(this->wait_.front());
    this->wait_.pop_front();
    ul.unlock();
    this->HandleLoadTask(std::move(task), stream);
}

void TransS2DPool::HandleReadyTask(Status s, BlockTask&& task, Stream& stream)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }
    if (s.Success()) {
        s = stream.H2DBatchSync(task.reader.GetData(), task.shards.data(), this->ioSize_,
                                task.shards.size());
    }
    if (s.Failure()) { this->failureSet_->Insert(task.owner); }
    task.done(s.Success());
}

void TransS2DPool::HandleLoadTask(BlockTask&& task, Stream& stream)
{
    if (this->failureSet_->Contains(task.owner)) {
        task.done(false);
        return;
    }
    auto s = task.reader.Ready4Read();
    if (s == Status::Retry()) {
        std::lock_guard<std::mutex> lg{this->mutex_};
        this->load_.push_back(std::move(task));
        return;
    }
    this->HandleReadyTask(s, std::move(task), stream);
}

} // namespace UC
