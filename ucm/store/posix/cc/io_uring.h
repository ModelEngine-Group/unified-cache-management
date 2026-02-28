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
 */
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_IO_URING_H
#define UNIFIEDCACHE_POSIX_STORE_CC_IO_URING_H

#include <fcntl.h>
#include <liburing.h>
#include <cstdint>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "status/status.h"

namespace UC::PosixStore {

class IoUringTask {
public:
    int32_t fd{-1};
    void* addr{nullptr};
    size_t size{0};
    off64_t offset{0};
};

class IoUringContext {
public:
    IoUringContext()
    {
        ring_.ring_fd = -1;
    }
    ~IoUringContext() { Destroy(); }

    IoUringContext(const IoUringContext&) = delete;
    IoUringContext& operator=(const IoUringContext&) = delete;

    Status Init(int32_t ringEntries = 256);
    void Destroy();

    Status H2SBatch(std::vector<IoUringTask>& tasks);

    Status S2HBatch(std::vector<IoUringTask>& tasks);

    struct io_uring* Ring() { return &ring_; }
    struct io_uring_sqe* GetSqe() { return io_uring_get_sqe(&ring_); }
    int Submit() { return io_uring_submit(&ring_); }
    int WaitCqe(struct io_uring_cqe** cqe, size_t timeoutMs)
    {
        if (timeoutMs == 0) { return io_uring_wait_cqe(&ring_, cqe); }
        __kernel_timespec timeout{};
        timeout.tv_sec = timeoutMs / 1000;
        timeout.tv_nsec = static_cast<long>(timeoutMs % 1000) * 1000 * 1000;
        return io_uring_wait_cqe_timeout(&ring_, cqe, &timeout);
    }
    void CqeSeen(struct io_uring_cqe* cqe) { io_uring_cqe_seen(&ring_, cqe); }

private:
    struct io_uring ring_{};
    bool initialized_{false};
    size_t ringEntries_{0};
};

}  // namespace UC::PosixStore

#endif