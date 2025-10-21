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

#include "dram_trans_queue.h"

namespace UC {

Status DramTransQueue::Setup(const int32_t deviceId, TaskSet* failureSet, 
                                const MemoryPool* memPool, const size_t timeoutMs) {
    // return Status::OK();
    this->deviceId_ = deviceId;
    this->failureSet_ = failureSet;
    this->memPool_ = memPool;
    auto success =
        this->backend_.SetWorkerInitFn([this](auto& device) { return this->Init(device); })
            .SetWorkerFn([this](auto& shard, const auto& device) { this->Work(shard, device); })
            .SetWorkerExitFn([this](auto& device) { this->Exit(device); })
            .Run();
    return success ? Status::OK() : Status::Error();
}

void DramTransQueue::Push(std::list<Task::Shard>& shards) noexcept {
    this->backend_.Push(shards);
}

bool DramTransQueue::Init(Device& device) {
    if (this->deviceId_ < 0) { return true; }
    device = DeviceFactory::Make(this->deviceId_, 0, 0);
    if (!device) {
        return false;
    }
    return device->Setup().Success();
}

void DramTransQueue::Exit(Device& device) {
    device.reset();
}

void DramTransQueue::Work(Task::Shard& shard, const Device& device) {
    if (this->failureSet_->Contains(shard.owner)) {
        this->Done(shard, device, true);
    }
    auto status = Status::OK();
    if (shard.type == Task::Type::DUMP) {
        // TODO: D2H
        status = this->D2H(shard, device);
    } else {
        // TODO: H2D
        status = this->H2D(shard, device);
    }
    this->Done(shard, device, status.Success());
}

Status DramTransQueue::H2D(Task::Shard& shard, const Device& device) {
    auto block_addr = this->memPool_->GetAddress(shard.block);
    if (!block_addr) {
        return Status::Error();
    }
    auto host_src = block_addr + shard.offset;
    return device->H2DAsync((std::byte*)shard.address, (std::byte*)host_src, shard.length);
}

Status DramTransQueue::D2H(Task::Shard& shard, const Device& device) {
    auto block_addr = this->memPool_->GetAddress(shard.block);
    if (!block_addr) {
        return Status::Error();
    }
    auto host_src = block_addr + shard.offset;
    return device->D2HAsync((std::byte*)host_src, (std::byte*)shard.address, shard.length);
}

void DramTransQueue::Done(Task::Shard& shard, const Device& device, const bool success) {
    if (!success) { this->failureSet_->Insert(shard.owner); }
    if (!shard.done) { return; }
    if (device) {
        if (device->Synchronized().Failure()) { this->failureSet_->Insert(shard.owner); }
    }
    shard.done();
}

} // namespace UC