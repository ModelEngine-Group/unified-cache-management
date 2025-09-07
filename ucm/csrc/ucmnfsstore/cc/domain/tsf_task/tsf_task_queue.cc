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
#include "tsf_task_queue.h"
#include <map>
#include "file/file.h"
#include "logger/logger.h"

namespace UC {

#define TO_GB(byte) ((byte) / (1ULL << 30))

Status TsfTaskQueue::Setup(const int32_t deviceId, const size_t streamNumber, const size_t bufferSize,
                           const size_t bufferNumber, TsfTaskSet* failureSet, const SpaceLayout* layout)
{
    this->_deviceId = deviceId;
    this->_bufferSize = bufferSize;
    this->_bufferNumber = bufferNumber;
    this->_failureSet = failureSet;
    this->_layout = layout;
    auto success = this->_front.Setup([this](TsfTask& task) { this->Dispatch(task); },
                                      [this] { return this->CreateDevice(); }, [this] { this->_device.reset(); });
    if (!success) { return Status::Error(); }
    success = this->_back.Setup([](auto& t) { t(); }, [] { return true; }, [] {}, streamNumber);
    return success ? Status::OK() : Status::Error();
}

void TsfTaskQueue::Push(TsfTask&& task) { this->_front.Push(std::move(task)); }

bool TsfTaskQueue::CreateDevice()
{
    if (this->_deviceId < 0) { return true; }
    this->_device = DeviceFactory::Make(this->_deviceId, this->_bufferSize, this->_bufferNumber);
    if (!this->_device) { return false; }
    return this->_device->Setup().Success();
}

void TsfTaskQueue::Dispatch(TsfTask& task)
{
    using Location = TsfTask::Location;
    using Type = TsfTask::Type;
    using Key = std::pair<Location, Type>;
    using Value = std::function<void(TsfTask&)>;
    std::map<Key, Value> runners = {
        {{Location::HOST, Type::LOAD},   [this](TsfTask& task) { this->S2H(task); }},
        {{Location::HOST, Type::DUMP},   [this](TsfTask& task) { this->H2S(task); }},
        {{Location::DEVICE, Type::LOAD}, [this](TsfTask& task) { this->S2D(task); }},
        {{Location::DEVICE, Type::DUMP}, [this](TsfTask& task) { this->D2S(task); }},
    };
    auto runner = runners.find({task.location, task.type});
    if (runner == runners.end()) {
        UC_ERROR("Unsupported task({},{})", fmt::underlying(task.location), fmt::underlying(task.type));
        this->_failureSet->Insert(task.id);
        task.waiter->Done();
        return;
    }
    if (!this->_failureSet->Contains(task.id)) { runner->second(task); }
    task.waiter->Done();
    return;
}

void TsfTaskQueue::S2H(TsfTask& task)
{
    auto scheTime = task.sw.elapsed().count();
    Latch latch{task.number};
    for (auto& shard : task.shards) {
        this->_back.Push([&latch, shard, this, owner = task.id, length = task.size] {
            if (!this->_failureSet->Contains(owner)) {
                if (this->Read(shard.blockId, shard.offset, length, shard.address).Failure()) {
                    this->_failureSet->Insert(owner);
                }
            }
            latch.Done();
        });
    }
    latch.Wait();
    if (!this->_failureSet->Contains(task.id)) {
        auto execTime = task.sw.elapsed().count() - scheTime;
        UC_INFO("Task({}) finished, elapsed={:.06f}s:{:.06f}s, bw={:.06f}GB/s.", task.Str(), scheTime, execTime,
                TO_GB(task.size * task.number / execTime));
    }
}

void TsfTaskQueue::H2S(TsfTask& task)
{
    auto scheTime = task.sw.elapsed().count();
    Latch latch{task.number};
    for (auto& shard : task.shards) {
        this->_back.Push([&latch, shard, this, owner = task.id, length = task.size] {
            if (!this->_failureSet->Contains(owner)) {
                if (this->Write(shard.blockId, shard.offset, length, shard.address).Failure()) {
                    this->_failureSet->Insert(owner);
                }
            }
            latch.Done();
        });
    }
    latch.Wait();
    if (!this->_failureSet->Contains(task.id)) {
        auto execTime = task.sw.elapsed().count() - scheTime;
        UC_INFO("Task({}) finished, elapsed={:.06f}s:{:.06f}s, bw={:.06f}GB/s.", task.Str(), scheTime, execTime,
                TO_GB(task.size * task.number / execTime));
    }
}

void TsfTaskQueue::S2D(TsfTask& task)
{
    auto scheTime = task.sw.elapsed().count();
    uintptr_t host[task.number] = {0};
    uintptr_t dev[task.number] = {0};
    if (this->AcquireBuffer(host, task.number).Failure()) {
        this->_failureSet->Insert(task.id);
        UC_ERROR("Task({}) failed.", task.Str());
        return;
    }
    Latch latch{task.number};
    for (auto& shard : task.shards) {
        dev[shard.index] = shard.address;
        this->_back.Push([&latch, shard, this, owner = task.id, length = task.size, &host] {
            if (!this->_failureSet->Contains(owner)) {
                if (this->Read(shard.blockId, shard.offset, length, host[shard.index]).Failure()) {
                    this->_failureSet->Insert(owner);
                }
            }
            latch.Done();
        });
    }
    latch.Wait();
    auto s2hTime = task.sw.elapsed().count() - scheTime;
    if (!this->_failureSet->Contains(task.id)) {
        auto status = this->_device->H2DBatch(host, dev, task.number, task.size);
        if (status.Failure()) {
            this->_failureSet->Insert(task.id);
            UC_ERROR("Task({}) failed({}).", task.Str(), status);
        } else {
            auto h2dTime = task.sw.elapsed().count() - s2hTime;
            auto total = task.size * task.number;
            UC_INFO("Task({}) finished, elapsed={:.06f}s:{:.06f}s:{:.06f}s, s2h={:.06f}GB/s, h2d={:.06f}GB/s, "
                    "s2d={:.06f}GB/s.",
                    task.Str(), scheTime, s2hTime, h2dTime, TO_GB(total / s2hTime), TO_GB(total / h2dTime),
                    TO_GB(total / (s2hTime + h2dTime)));
        }
    }
    this->ReleaseBuffer(host, task.number);
}

void TsfTaskQueue::D2S(TsfTask& task)
{
    auto scheTime = task.sw.elapsed().count();
    uintptr_t host[task.number] = {0};
    uintptr_t dev[task.number] = {0};
    for (auto& shard : task.shards) {
        auto idx = shard.index;
        auto ptr = this->_device->GetBuffer(idx);
        if (!ptr) {
            this->ReleaseBuffer(host, idx);
            this->_failureSet->Insert(task.id);
            UC_ERROR("Task({}) failed.", task.Str());
            return;
        }
        host[idx] = (uintptr_t)ptr;
        dev[idx] = shard.address;
    }
    auto status = this->_device->D2HBatch(dev, host, task.number, task.size);
    if (status.Failure()) {
        this->_failureSet->Insert(task.id);
        this->ReleaseBuffer(host, task.number);
        UC_ERROR("Task({}) failed({}).", task.Str(), status);
        return;
    }
    auto d2hTime = task.sw.elapsed().count() - scheTime;
    Latch latch{task.number};
    for (auto& shard : task.shards) {
        this->_back.Push([&latch, shard, this, owner = task.id, length = task.size, &host] {
            if (!this->_failureSet->Contains(owner)) {
                if (this->Write(shard.blockId, shard.offset, length, host[shard.index]).Failure()) {
                    this->_failureSet->Insert(owner);
                }
            }
            latch.Done();
        });
    }
    latch.Wait();
    if (!this->_failureSet->Contains(task.id)) {
        auto h2sTime = task.sw.elapsed().count() - d2hTime;
        auto total = task.size * task.number;
        UC_INFO(
            "Task({}) finished, elapsed={:.06f}s:{:.06f}s:{:.06f}s, d2h={:.06f}GB/s, h2s={:.06f}GB/s, d2s={:.06f}GB/s.",
            task.Str(), scheTime, d2hTime, h2sTime, TO_GB(total / d2hTime), TO_GB(total / h2sTime),
            TO_GB(total / (d2hTime + h2sTime)));
    }
    this->ReleaseBuffer(host, task.number);
}

Status TsfTaskQueue::Read(const std::string& blockId, const size_t offset, const size_t length, uintptr_t address)
{
    auto path = this->_layout->DataFilePath(blockId, false);
    return File::Read(path, offset, length, address);
}

Status TsfTaskQueue::Write(const std::string& blockId, const size_t offset, const size_t length,
                           const uintptr_t address)
{
    auto path = this->_layout->DataFilePath(blockId, true);
    return File::Write(path, offset, length, address);
}

Status TsfTaskQueue::AcquireBuffer(uintptr_t* buffer, const size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto ptr = this->_device->GetBuffer(i);
        if (!ptr) {
            this->ReleaseBuffer(buffer, i);
            return Status::OutOfMemory();
        }
        buffer[i] = (uintptr_t)ptr;
    }
    return Status::OK();
}

void TsfTaskQueue::ReleaseBuffer(uintptr_t* buffer, const size_t number)
{
    for (size_t i = 0; i < number; i++) { this->_device->PutBuffer(i, (void*)buffer[i]); }
}

} // namespace UC
