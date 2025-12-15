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
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_TRANS_MANAGER_H
#define UNIFIEDCACHE_CACHE_STORE_CC_TRANS_MANAGER_H

#include "dump_queue.h"
#include "load_queue.h"
#include "logger/logger.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::CacheStore {

class TransManager : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    size_t shardSize_;
    TransBuffer buffer_;
    LoadQueue loadQ_;
    DumpQueue dumpQ_;

public:
    Status Setup(const Config& config)
    {
        timeoutMs_ = config.transferTimeoutMs;
        shardSize_ = config.shardSize;
        auto s = buffer_.Setup(config);
        if (s.Failure()) [[unlikely]] { return s; }
        s = loadQ_.Setup(config, &failureSet_, &buffer_);
        if (s.Failure()) [[unlikely]] { return s; }
        return dumpQ_.Setup(config, &failureSet_, &buffer_);
    }

protected:
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        const auto id = t->id;
        const auto& brief = t->desc.brief;
        const auto num = t->desc.size();
        const auto size = shardSize_ * num;
        const auto tp = w->startTp;
        UC_DEBUG("Cache task({},{},{},{}) dispatching.", id, brief, num, size);
        w->SetEpilog([id, brief = std::move(brief), num, size, tp] {
            auto cost = NowTime::Now() - tp;
            UC_DEBUG("Cache task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                     cost * 1e3);
        });
        if (t->type == TransTask::Type::LOAD) {
            loadQ_.Submit(t, w);
        } else {
            dumpQ_.Submit(t, w);
        }
    }
};

}  // namespace UC::CacheStore

#endif
