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
#include "fsstore.h"
#include <fmt/ranges.h>
#include "logger/logger.h"
#include "space/space_manager.h"
#include "tsf_task/tsf_task_manager.h"

namespace UC {

class FSStoreImpl : public FSStore {
public:
    int32_t Setup(const Config& config)
    {
        auto status = this->spaceMgr_.Setup(config.storageBackends, config.kvcacheBlockSize);
        if (status.Failure()) {
            UC_ERROR("Failed({}) to setup SpaceManager.", status);
            return status.Underlying();
        }
        if (config.transferEnable) {
            status =
                this->transMgr_.Setup(config.transferDeviceId, config.transferStreamNumber,
                                      config.transferIoSize, config.transferBufferNumber,
                                      config.transferTimeoutMs, this->spaceMgr_.GetSpaceLayout());
            if (status.Failure()) {
                UC_ERROR("Failed({}) to setup TsfTaskManager.", status);
                return status.Underlying();
            }
        }
        this->ShowConfig(config);
        return Status::OK().Underlying();
    }
    int32_t Alloc(const std::string& block) override
    {
        return this->spaceMgr_.NewBlock(block).Underlying();
    }
    bool Lookup(const std::string& block) override { return this->spaceMgr_.LookupBlock(block); }
    void Commit(const std::string& block, const bool success) override
    {
        this->spaceMgr_.CommitBlock(block, success);
    }
    std::list<int32_t> Alloc(const std::list<std::string>& blocks) override
    {
        std::list<int32_t> results;
        for (const auto& block : blocks) { results.emplace_back(this->Alloc(block)); }
        return results;
    }
    std::list<bool> Lookup(const std::list<std::string>& blocks) override
    {
        std::list<bool> founds;
        for (const auto& block : blocks) { founds.emplace_back(this->Lookup(block)); }
        return founds;
    }
    void Commit(const std::list<std::string>& blocks, const bool success) override
    {
        for (const auto& block : blocks) { this->Commit(block, success); }
    }
    size_t Submit(Task&& task) override
    {
        std::list<TsfTask> tasks;
        for (auto& shard : task.shards) {
            tasks.push_back(
                {task.type, task.location, shard.block, shard.offset, shard.address, task.size});
        }
        size_t taskId;
        return this->transMgr_
                       .Submit(tasks, task.number * task.size, task.number, task.brief, taskId)
                       .Success()
                   ? taskId
                   : CCStore::invalidTaskId;
    }
    int32_t Wait(const size_t task) override { return this->transMgr_.Wait(task).Underlying(); }
    int32_t Check(const size_t task, bool& finish) override
    {
        return this->transMgr_.Check(task, finish).Underlying();
    }

private:
    void ShowConfig(const Config& config)
    {
        std::string buildType = UCM_BUILD_TYPE;
        if (buildType.empty()) { buildType = "Release"; }
        UC_INFO("FSStore-{}({}).", UCM_COMMIT_ID, buildType);
        UC_INFO("Set UC::StorageBackends to {}.", config.storageBackends);
        UC_INFO("Set UC::BlockSize to {}.", config.kvcacheBlockSize);
        UC_INFO("Set UC::TransferEnable to {}.", config.transferEnable);
        UC_INFO("Set UC::DeviceId to {}.", config.transferDeviceId);
        UC_INFO("Set UC::StreamNumber to {}.", config.transferStreamNumber);
        UC_INFO("Set UC::IOSize to {}.", config.transferIoSize);
        UC_INFO("Set UC::BufferNumber to {}.", config.transferBufferNumber);
        UC_INFO("Set UC::TimeoutMs to {}.", config.transferTimeoutMs);
    }

private:
    SpaceManager spaceMgr_;
    TsfTaskManager transMgr_;
};

int32_t FSStore::Setup(const Config& config)
{
    auto impl = new (std::nothrow) FSStoreImpl();
    if (!impl) {
        UC_ERROR("Out of memory.");
        return Status::OutOfMemory().Underlying();
    }
    this->impl_ = impl;
    return impl->Setup(config);
}

} // namespace UC
