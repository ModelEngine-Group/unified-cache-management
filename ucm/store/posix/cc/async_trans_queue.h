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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_ASYNC_TRANS_QUEUE_H
#define UNIFIEDCACHE_POSIX_STORE_CC_ASYNC_TRANS_QUEUE_H

#include "global_config.h"
#include "io_uring.h"
#include "space_layout.h"
#include "template/hashset.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"
#include "trans_task.h"

namespace UC::PosixStore {

class AsyncTransQueue {
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;
    using IoUringContextPtr = std::unique_ptr<IoUringContext>;

private:
    struct IoUnit {
        Detail::TaskHandle owner;
        std::vector<Detail::Shard> shards;
        std::shared_ptr<Latch> waiter;
    };
    TaskIdSet* failureSet_;
    const SpaceLayout* layout_;
    ThreadPool<IoUnit, IoUringContextPtr> loadIoUringPool_;
    ThreadPool<IoUnit, IoUringContextPtr> dumpIoUringPool_;
    size_t ioSize_;
    size_t shardSize_;
    size_t nShardPerBlock_;
    bool ioDirect_;

public:
    Status Setup(const Config& config, TaskIdSet* failureSet, const SpaceLayout* layout);
    void Push(TaskPtr task, WaiterPtr waiter);

private:
    void LoadWorkerAsync(IoUnit& ios, const IoUringContextPtr& ctx);
    void DumpWorkerAsync(IoUnit& ios, const IoUringContextPtr& ctx);
    Status H2SAsync(IoUnit& ios, const IoUringContextPtr& ctx);
    Status S2HAsync(IoUnit& ios, const IoUringContextPtr& ctx);
};

}  // namespace UC::PosixStore

#endif
