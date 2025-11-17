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
#include "posix_queue.h"
#include "file/file.h"

namespace UC {

template <typename T>
bool IsAligned(const T value)
{
    static constexpr size_t alignment = 4096;
    static constexpr size_t alignMask = alignment - 1;
    return (value & alignMask) == 0;
}

Status PosixQueue::Setup(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber,
                         TaskSet* failureSet, const SpaceLayout* layout, const size_t timeoutMs, bool useDirect)
{
    this->deviceId_ = deviceId;
    this->bufferSize_ = bufferSize;
    this->bufferNumber_ = bufferNumber;
    this->failureSet_ = failureSet;
    this->layout_ = layout;
    this->useDirect_ = useDirect;
    auto success =
        this->backend_.SetWorkerInitFn([this](auto& device) { return this->Init(device); })
            .SetWorkerFn([this](auto shard, const auto& device) { this->Work(std::move(shard), device); })
            .SetWorkerExitFn([this](auto& device) { this->Exit(device); })
            .Run();
    return success ? Status::OK() : Status::Error();
}

void PosixQueue::Push(std::list<Task::Shard>& shards) noexcept { this->backend_.Push(shards); }

bool PosixQueue::Init(Device& device)
{
    if (this->deviceId_ < 0) { return true; }
    device = DeviceFactory::Make(this->deviceId_, this->bufferSize_, this->bufferNumber_);
    if (!device) { return false; }
    return device->Setup().Success();
}

void PosixQueue::Exit(Device& device) { device.reset(); }

void PosixQueue::Work(Task::Shard&& shard, const Device& device)
{
    if (this->failureSet_->Contains(shard.owner)) {
        this->Done(shard, device, true);
        return;
    }
    auto status = Status::OK();
    if (shard.location == Task::Location::DEVICE) {
        if (shard.type == Task::Type::DUMP) {
            status = this->D2S(shard, device);
        } else {
            status = this->S2D(shard, device);
        }
    } else {
        if (shard.type == Task::Type::DUMP) {
            status = this->H2S(shard);
        } else {
            status = this->S2H(shard);
        }
    }
    this->Done(shard, device, status.Success());
}

void PosixQueue::Done(Task::Shard& shard, const Device& device, const bool success)
{
    if (!success) { this->failureSet_->Insert(shard.owner); }
    if (!shard.done) { return; }
    if (device) {
        if (device->Synchronized().Failure()) { this->failureSet_->Insert(shard.owner); }
    }
    shard.done();
}

Status PosixQueue::D2S(Task::Shard& shard, const Device& device)
{
    shard.buffer = device->GetBuffer(shard.length);
    if (!shard.buffer) {
        UC_ERROR("Out of memory({}).", shard.length);
        return Status::OutOfMemory();
    }
    auto hub = shard.buffer.get();
    auto status = device->D2HSync((std::byte*)hub, (std::byte*)shard.address, shard.length);
    if (status.Failure()) { return status; }
    auto path = this->layout_->DataFilePath(shard.block, true);
    return File::Write(path, shard.offset, shard.length, (uintptr_t)hub, useDirect_);
}

Status PosixQueue::S2D(Task::Shard& shard, const Device& device)
{
    shard.buffer = device->GetBuffer(shard.length);
    if (!shard.buffer) {
        UC_ERROR("Out of memory({}).", shard.length);
        return Status::OutOfMemory();
    }
    auto hub = shard.buffer.get();
    auto path = this->layout_->DataFilePath(shard.block, false);
    auto status = File::Read(path, shard.offset, shard.length, (uintptr_t)hub, useDirect_);
    if (status.Failure()) { return status; }
    return device->H2DAsync((std::byte*)shard.address, (std::byte*)hub, shard.length);
}

Status PosixQueue::H2S(Task::Shard& shard)
{
    auto path = this->layout_->DataFilePath(shard.block, true);
    auto aligned = IsAligned(shard.offset) && IsAligned(shard.length) && IsAligned(shard.address);
    return File::Write(path, shard.offset, shard.length, shard.address, aligned);
}

Status PosixQueue::S2H(Task::Shard& shard)
{
    auto path = this->layout_->DataFilePath(shard.block, false);
    auto aligned = IsAligned(shard.offset) && IsAligned(shard.length) && IsAligned(shard.address);
    return File::Read(path, shard.offset, shard.length, shard.address, aligned);
}

} // namespace UC
