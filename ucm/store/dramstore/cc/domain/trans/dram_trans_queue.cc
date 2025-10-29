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
            .SetWorkerFn([this](auto& shards, const auto& device) { this->Work(shards, device); })
            .SetWorkerExitFn([this](auto& device) { this->Exit(device); })
            .Run();
    return success ? Status::OK() : Status::Error();
}

void DramTransQueue::Push(std::list<Task::Shard>& shards) noexcept {
    this->backend_.Push(std::move(shards));
}

bool DramTransQueue::Init(Device& device) {
    if (this->deviceId_ < 0) { return true; }
    device = DeviceFactory::Make(this->deviceId_, 262144, 512);
    if (!device) {
        return false;
    }
    return device->Setup().Success();
}

void DramTransQueue::Exit(Device& device) {
    device.reset();
}

void DramTransQueue::Work(std::list<Task::Shard>& shards, const Device& device) {
    auto it = shards.begin();
    if (this->failureSet_->Contains(it->owner)) {
        this->Done(shards, device, true);
    }
    auto status = Status::OK();
    if (it->type == Task::Type::DUMP) {
        // TODO: D2H
        status = this->D2H(shards, device);
    } else {
        // TODO: H2D
        status = this->H2D(shards, device);
    }
    this->Done(shards, device, status.Success());
}

Status DramTransQueue::H2D(std::list<Task::Shard>& shards, const Device& device) {
    // TODO: 里面要重写
    size_t pool_offset = 0;
    std::vector<std::byte*> host_addrs(shards.size());
    std::vector<uintptr_t> device_addrs(shards.size());
    for (auto& shard : shards) {
        int shard_index = 0;
        bool found = this->memPool_->GetOffset(shard.block, &pool_offset);
        if (!found) {
            return Status::Error();
        }
        auto host_addr = this->memPool_->GetStartAddr().get() + pool_offset + shard.offset;       
        auto device_addr = shard.address;
        host_addrs[shard_index] = host_addr;
        device_addrs[shard_index] = device_addr;
        shard_index++;
    }
    // return device->H2DAsync((std::byte*)shard.address, (std::byte*)host_src, shard.length);
    auto it = shards.begin();
    return device->H2DBatchSync(reinterpret_cast<std::byte**>(device_addrs.data()), const_cast<const std::byte**>(host_addrs.data()), shards.size(), it->length * shards.size());
}

Status DramTransQueue::D2H(std::list<Task::Shard>& shards, const Device& device) {
    // TODO: 里面要重写
    size_t pool_offset = 0;
    std::vector<std::byte*> host_addrs(shards.size());
    std::vector<uintptr_t> device_addrs(shards.size());
    for (auto& shard : shards) {
        int shard_index = 0;
        bool found = this->memPool_->GetOffset(shard.block, &pool_offset);
        if (!found) {
            return Status::Error();
        }
        auto host_addr = this->memPool_->GetStartAddr().get() + pool_offset + shard.offset;       
        auto device_addr = shard.address;
        host_addrs[shard_index] = host_addr;
        device_addrs[shard_index] = device_addr;
        shard_index++;
    }
    // return device->D2HAsync((std::byte*)host_src, (std::byte*)shard.address, shard.length);
    auto it = shards.begin();
    return device->D2HBatchSync(reinterpret_cast<std::byte**>(host_addrs.data()), const_cast<const std::byte**>(device_addrs.data()), shards.size(), it->length * shards.size());
}

void DramTransQueue::Done(std::list<Task::Shard>& shards, const Device& device, const bool success) {
    auto it = shards.begin();
    if (!success) { this->failureSet_->Insert(it->owner); }
    for (auto& shard : shards) { 
        if (shard.done) {
            if (device) {
                if (device->Synchronized().Failure()) { this->failureSet_->Insert(shard.owner); }
            }
            shard.done();
        }
    }
}

} // namespace UC