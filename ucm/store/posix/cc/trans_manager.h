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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_TRANS_MANAGER_H
#define UNIFIEDCACHE_POSIX_STORE_CC_TRANS_MANAGER_H

#include "logger/logger.h"
#include "template/task_wrapper.h"
#include "trans_queue.h"
#if UCM_HAS_LIBURING
#include "async_trans_queue.h"
#endif

namespace UC::PosixStore {

class TransManager : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    TransQueue queue_;
#if UCM_HAS_LIBURING
    AsyncTransQueue asyncQueue_;
#endif
    bool useIoUring_{false};
    size_t shardSize_;

public:
    Status Setup(const Config& config, const SpaceLayout* layout)
    {
        timeoutMs_ = config.timeoutMs;
        shardSize_ = config.shardSize;
        useIoUring_ = config.useIoUring;
#if UCM_HAS_LIBURING
        if (useIoUring_) {
            return asyncQueue_.Setup(config, &failureSet_, layout);
        }
        return queue_.Setup(config, &failureSet_, layout);
#else
        if (useIoUring_) {
            return Status::Error(
                "posix_use_io_uring is enabled but liburing is unavailable at build time");
        }
        return queue_.Setup(config, &failureSet_, layout);
#endif
    }

protected:
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        const auto id = t->id;
        const auto& brief = t->desc.brief;
        const auto num = t->desc.size();
        const auto size = shardSize_ * num;
        const auto tp = w->startTp;
        UC_DEBUG("Posix task({},{},{},{}) dispatching.", id, brief, num, size);
        w->SetEpilog([id, brief = std::move(brief), num, size, tp] {
            auto cost = NowTime::Now() - tp;
            UC_DEBUG("Posix task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                     cost * 1e3);
        });
        if (useIoUring_) {
#if UCM_HAS_LIBURING
            asyncQueue_.Push(t, w);
#else
            queue_.Push(t, w);
#endif
        } else {
            queue_.Push(t, w);
        }
    }
};

}  // namespace UC::PosixStore

#endif
