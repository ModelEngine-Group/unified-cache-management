/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>
#include "buffer_manager.h"
#include "global_config.h"
#include "logger/logger.h"
#include "trans_manager.h"
#include "trans_task.h"
#include "ucmstore_v1.h"

namespace UC::Cache2 {

class Store : public StoreV1 {
    BufferManager<> bufferMgr_;
    TransManager transMgr_;

public:
    Status Setup(const Detail::Dictionary& inConfig) override
    {
        auto config = Config::From(inConfig);
        auto s = config.Validate();
        if (s.Failure()) {
            UC_ERROR("Failed to check config: {}.", s);
            return s;
        }
        config.Show();
        if (s = bufferMgr_.Setup(config); s.Failure()) { return s; }
        if (config.deviceId >= 0) {
            if (s = transMgr_.Setup(config, bufferMgr_.GetTransBuffer()); s.Failure()) { return s; }
        }
        return Status::OK();
    }
    std::string Readme() const override { return "CacheStore"; }
    Expected<std::vector<uint8_t>> Lookup(const Detail::BlockId* blocks, size_t num) override
    {
        return Status::Unsupported();
    }
    Expected<ssize_t> LookupOnPrefix(const Detail::BlockId* blocks, size_t num) override
    {
        return bufferMgr_.LookupOnPrefix(blocks, num);
    }
    Expected<ssize_t> LookupOnReverse(const Detail::BlockId* blocks, size_t num) override
    {
        return bufferMgr_.LookupOnReverse(blocks, num);
    }
    void Prefetch(const Detail::BlockId* blocks, size_t num) override
    {
        bufferMgr_.Prefetch(blocks, num);
    }
    Expected<Detail::TaskHandle> Load(Detail::TaskDesc task) override
    {
        return transMgr_.Submit({Task::Type::LOAD, std::move(task)});
    }
    Expected<Detail::TaskHandle> Dump(Detail::TaskDesc task) override
    {
        return transMgr_.Submit({Task::Type::DUMP, std::move(task)});
    }
    Expected<bool> Check(Detail::TaskHandle taskId) override { return transMgr_.Check(taskId); }
    Status Wait(Detail::TaskHandle taskId) override { return transMgr_.Wait(taskId); }
};

}  // namespace UC::Cache2

extern "C" UC::StoreV1* MakeCache2Store() { return new UC::Cache2::Store(); }
