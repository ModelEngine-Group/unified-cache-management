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

#include <liburing.h>
#include "status/status.h"

namespace UC::PosixStore {

class IoUringContext {
public:
    IoUringContext()
    {
        ring_.ring_fd = -1;
    }
    ~IoUringContext() { Destroy(); }

    IoUringContext(const IoUringContext&) = delete;
    IoUringContext& operator=(const IoUringContext&) = delete;

    Status Init(int32_t ringEntries = 1024);
    void Destroy();

    struct io_uring_sqe* GetSqe() { return io_uring_get_sqe(&ring_); }
    int Submit() { return io_uring_submit(&ring_); }
    unsigned PeekBatchCqe(struct io_uring_cqe** cqes, unsigned count)
    {
        return io_uring_peek_batch_cqe(&ring_, cqes, count);
    }
    int WaitCqe(struct io_uring_cqe** cqe, size_t timeoutMs)
    {
        if (timeoutMs == 0) { return io_uring_wait_cqe(&ring_, cqe); }
        __kernel_timespec timeout{};
        timeout.tv_sec = timeoutMs / 1000;
        timeout.tv_nsec = static_cast<long>(timeoutMs % 1000) * 1000 * 1000;
        return io_uring_wait_cqe_timeout(&ring_, cqe, &timeout);
    }
    void CqAdvance(unsigned nr) { io_uring_cq_advance(&ring_, nr); }
    void CqeSeen(struct io_uring_cqe* cqe) { io_uring_cqe_seen(&ring_, cqe); }

private:
    struct io_uring ring_{};
    bool initialized_{false};
};

}  // namespace UC::PosixStore

#endif