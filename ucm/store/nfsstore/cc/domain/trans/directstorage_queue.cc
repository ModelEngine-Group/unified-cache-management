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
#include "directstorage_queue.h"
#include "file/file.h"
#include "logger/logger.h"
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

namespace UC {

Status DirectStorageQueue::Setup(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber,
                         TaskSet* failureSet, const SpaceLayout* layout, const size_t timeoutMs, bool transferUseDirect)
{
    this->deviceId_ = deviceId;
    this->bufferSize_ = bufferSize;
    this->bufferNumber_ = bufferNumber;
    this->failureSet_ = failureSet;
    this->layout_ = layout;
    this->transferUseDirect_ = transferUseDirect;
    auto success =
        this->backend_.SetWorkerInitFn([this](auto& device) { return this->Init(device); })
            .SetWorkerFn([this](auto& shard, const auto& device) { this->Work(shard, device); })
            .SetWorkerExitFn([this](auto& device) { this->Exit(device); })
            .Run();
    return success ? Status::OK() : Status::Error();
}

void DirectStorageQueue::Push(std::list<Task::Shard>& shards) noexcept { this->backend_.Push(shards); }

bool DirectStorageQueue::Init(Device& device)
{
    if (this->deviceId_ < 0) { return true; }
    device = DeviceFactory::Make(this->deviceId_, this->bufferSize_, this->bufferNumber_);
    if (!device) { return false; }
    return device->Setup(this->transferUseDirect_).Success();
}

void DirectStorageQueue::Exit(Device& device) { device.reset(); }

void DirectStorageQueue::Work(Task::Shard& shard, const Device& device)
{
    if (this->failureSet_->Contains(shard.owner)) {
        this->Done(shard, device, true);
        return;
    }
    auto status = Status::OK();
    if (shard.type == Task::Type::DUMP) {
        status = this->D2S(shard, device);
    } else {
        status = this->S2D(shard, device);
    }
    this->Done(shard, device, status.Success());
}

void DirectStorageQueue::Done(Task::Shard& shard, const Device& device, const bool success)
{
    if (!success) { this->failureSet_->Insert(shard.owner); }
    if (!shard.done) { return; }
    if (device) {
        if (device->Synchronized().Failure()) { this->failureSet_->Insert(shard.owner); }
    }
    shard.done();
}

Status DirectStorageQueue::S2D(Task::Shard& shard, const Device& device) {
    auto path = this->layout_->DataFilePath(shard.block, false);
    return device->S2DSync(path, (void*)shard.address, shard.length, shard.offset, 0);
}

Status DirectStorageQueue::D2S(Task::Shard& shard, const Device& device) {
    auto path = this->layout_->DataFilePath(shard.block, true);
    return device->D2SSync(path, (void*)shard.address, shard.length, shard.offset, 0);
}

} // namespace UC
