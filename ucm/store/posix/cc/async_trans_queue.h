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

#include <atomic>
#include <memory>
#include <mutex>
#include <sys/uio.h>
#include <thread>
#include <vector>
#include <cerrno>
#include "global_config.h"
#include "io_uring.h"
#include "posix_file.h"
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

private:
    struct IoUnit {
        Detail::TaskHandle owner;
        Detail::Shard shard;
        std::shared_ptr<Latch> waiter;
    };
    struct IoTask {
        Detail::TaskHandle owner;
        std::vector<Detail::Shard> shards;
        std::shared_ptr<Latch> waiter;
        bool firstIo{false};
    };
    struct IoUnitCtx {
        IoUnit unit;
        TaskIdSet* failureSet;
        const SpaceLayout* layout;
        size_t expectedBytes{0};
        bool isLastShardOfBlock{false};
        std::vector<struct iovec> iov;
        std::unique_ptr<PosixFile> file;
    };

    TaskIdSet* failureSet_;
    const SpaceLayout* layout_;
    IoUringContext loadRing_;
    IoUringContext dumpRing_;
    ThreadPool<IoTask, IoUringContext*> loadSubmitterPool_;
    ThreadPool<IoTask, IoUringContext*> dumpSubmitterPool_;
    std::mutex loadSqMtx_;
    std::mutex dumpSqMtx_;
    std::thread loadReaperThread_;
    std::thread dumpReaperThread_;
    std::atomic<bool> stop_{false};
    size_t ioSize_;
    size_t shardSize_;
    size_t nShardPerBlock_;
    size_t waitCqeTimeoutMs_{30000};
    bool ioDirect_;

public:
    Status Setup(const Config& config, TaskIdSet* failureSet, const SpaceLayout* layout);
    ~AsyncTransQueue();
    void Push(TaskPtr task, WaiterPtr waiter);

private:
    void LoadSubmitter(IoTask& task, IoUringContext* ring);
    void DumpSubmitter(IoTask& task, IoUringContext* ring);
    void SubmitTask(IoTask& task, IoUringContext* ring, bool isDump);
    void LoadReaperLoop();
    void DumpReaperLoop();
};

}  // namespace UC::PosixStore

#endif
