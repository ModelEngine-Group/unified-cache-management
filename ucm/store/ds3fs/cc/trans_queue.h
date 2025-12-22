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
#ifndef UNIFIEDCACHE_DS3FS_STORE_CC_TRANS_QUEUE_H
#define UNIFIEDCACHE_DS3FS_STORE_CC_TRANS_QUEUE_H

#include <hf3fs_usrbio.h>
#include <memory>
#include <mutex>
#include "global_config.h"
#include "space_layout.h"
#include "template/hashset.h"
#include "thread/latch.h"
#include "thread/thread_pool.h"
#include "trans_task.h"

namespace UC::Ds3fsStore {

class TransQueue {
    using TaskIdSet = HashSet<Detail::TaskHandle>;
    using TaskPtr = std::shared_ptr<TransTask>;
    using WaiterPtr = std::shared_ptr<Latch>;

private:
    struct IoUnit {
        Detail::TaskHandle owner;
        TransTask::Type type;
        Detail::Shard shard;
        std::shared_ptr<Latch> waiter;
        bool firstIo{false};
    };

    class IovGuard {
        struct hf3fs_iov iov_;
        bool valid_{false};

    public:
        IovGuard() = default;
        Status Create(const std::string& mountPoint, size_t size)
        {
            int res = hf3fs_iovcreate(&iov_, mountPoint.c_str(), size, 0, -1);
            if (res < 0) {
                return Status::OsApiError(fmt::format("Failed to create IOV: {}", res));
            }
            valid_ = true;
            return Status::OK();
        }
        ~IovGuard()
        {
            if (valid_) hf3fs_iovdestroy(&iov_);
        }
        struct hf3fs_iov* Get() { return &iov_; }
        void* Base() { return iov_.base; }
        IovGuard(const IovGuard&) = delete;
        IovGuard& operator=(const IovGuard&) = delete;
    };

    class IorGuard {
        struct hf3fs_ior ior_;
        bool valid_{false};

    public:
        IorGuard() = default;
        Status Create(const std::string& mountPoint, size_t entries, bool isRead, int depth)
        {
            int res = hf3fs_iorcreate4(&ior_, mountPoint.c_str(), entries, isRead, depth, 0, -1, 0);
            if (res < 0) {
                return Status::OsApiError(fmt::format("Failed to create IOR: {}", res));
            }
            valid_ = true;
            return Status::OK();
        }
        ~IorGuard()
        {
            if (valid_) hf3fs_iordestroy(&ior_);
        }
        struct hf3fs_ior* Get() { return &ior_; }
        IorGuard(const IorGuard&) = delete;
        IorGuard& operator=(const IorGuard&) = delete;
    };

    class FdGuard {
        int fd_{-1};

    public:
        FdGuard() = default;
        Status Register(int fd)
        {
            auto res = hf3fs_reg_fd(fd, 0);
            if (res > 0) {
                return Status::OsApiError(fmt::format("Failed to register fd({}): {}", fd, res));
            }
            fd_ = fd;
            return Status::OK();
        }
        ~FdGuard()
        {
            if (fd_ >= 0) hf3fs_dereg_fd(fd_);
        }
        FdGuard(const FdGuard&) = delete;
        FdGuard& operator=(const FdGuard&) = delete;
    };

    TaskIdSet* failureSet_;
    const SpaceLayout* layout_;
    ThreadPool<IoUnit> pool_;
    size_t ioSize_;
    size_t shardSize_;
    size_t nShardPerBlock_;
    bool ioDirect_;
    std::string mountPoint_;
    size_t iorEntries_;
    int iorDepth_;

public:
    Status Setup(const Config& config, TaskIdSet* failureSet, const SpaceLayout* layout);
    void Push(TaskPtr task, WaiterPtr waiter);

private:
    void Worker(IoUnit& ios);
    Status H2S(IoUnit& ios);
    Status S2H(IoUnit& ios);
    Status DoIo(IorGuard& ior, IovGuard& iov, bool isRead, int fd, size_t offset, size_t size);
};

}  // namespace UC::Ds3fsStore

#endif