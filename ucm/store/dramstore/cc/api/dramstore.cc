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
#include "dramstore.h"
#include "logger/logger.h"
#include "status/status.h"
#include "trans/trans_manager.h"
#include "memory/memory_pool.h"

namespace UC {

class DRAMStoreImpl : public DRAMStore {
public:
    int32_t Setup(const size_t ioSize, const size_t capacity, const int32_t deviceId) {
        // 这里如何传入参数，待讨论
        int32_t capacity = 14400;
        int32_t blockSize = 144;
        this->memPool_ = std::make_unique<MemoryPool>(capacity, blockSize).release();
        int32_t deviceId = 1;
        int32_t streamNumber = 10;
        int32_t timeoutMs = 10000;
        auto status = this->transMgr_.Setup(deviceId, streamNumber, timeoutMs, this->memPool_);
        if (status.Failure()) {
            UC_ERROR("Failed({}) to setup TsfTaskManager.", status);
            return status.Underlying();
        }
        return Status::OK().Underlying();
    }
    int32_t Alloc(const std::string& block) override { return this->memPool_->NewBlock(block).Underlying(); }
    bool Lookup(const std::string& block) override { return this->memPool_->LookupBlock(block); }
    void Commit(const std::string& block, const bool success) override { this->memPool_->CommitBlock(block, success).Underlying(); }
    std::list<int32_t> Alloc(const std::list<std::string>& blocks) override
    {
        std::list<int32_t> results;
        for (const auto &block : blocks) {
            results.emplace_back(this->Alloc(block));
        }
        return results;
    }
    std::list<bool> Lookup(const std::list<std::string>& blocks) override
    {
        std::list<bool> founds;
        for (const auto &block : blocks) {
            founds.emplace_back(this->Lookup(block));
        }
        return founds;
    }
    void Commit(const std::list<std::string>& blocks, const bool success) override {
        for (const auto &block : blocks) {
            this->Commit(block, success);
        }
    }
    size_t Submit(Task&& task) override {
        auto taskId = Task::invalid;
        auto status = this->transMgr_.Submit(std::move(task), taskId);
        if (status.Failure()) { taskId = Task::invalid; }
        return taskId;    }

    int32_t Wait(const size_t task) override {
        return this->transMgr_.Wait(task).Underlying();
    }

    int32_t Check(const size_t task, bool& finish) override {
        return this->transMgr_.Check(task, finish).Underlying();
    }


private:

    TransManager transMgr_;
    MemoryPool* memPool_;

};

int32_t DRAMStore::Setup(const Config& config)
{
    auto impl = new (std::nothrow) DRAMStoreImpl();
    if (!impl) {
        UC_ERROR("Out of memory.");
        return Status::OutOfMemory().Underlying();
    }
    this->impl_ = impl;
    return impl->Setup(config.ioSize, config.capacity, config.deviceId);
}

} // namespace UC
