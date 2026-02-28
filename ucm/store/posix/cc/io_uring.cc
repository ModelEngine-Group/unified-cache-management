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
#include "io_uring.h"
#include <cerrno>
#include <cstring>
#include <fmt/core.h>

namespace UC::PosixStore {

Status IoUringContext::Init(int32_t ringEntries)
{
    if (initialized_) { return Status::OK(); }
    ringEntries_ = ringEntries;
    int ret = io_uring_queue_init(ringEntries_, &ring_, 0);
    if (ret < 0) {
        return Status::OsApiError(fmt::format("io_uring_queue_init failed: {}", strerror(-ret)));
    }
    initialized_ = true;
    return Status::OK();
}

void IoUringContext::Destroy()
{
    if (!initialized_) { return; }
    io_uring_queue_exit(&ring_);
    ring_ = {};
    ring_.ring_fd = -1;
    initialized_ = false;
}

static Status SubmitAndWaitBatch(struct io_uring* ring, size_t submitted)
{
    if (submitted == 0) { return Status::OK(); }
    int ret = io_uring_submit_and_wait(ring, submitted);
    if (ret < 0) {
        return Status::OsApiError(fmt::format("io_uring_submit_and_wait: {}", strerror(-ret)));
    }
    for (int j = 0; j < ret; ++j) {
        struct io_uring_cqe* cqe = nullptr;
        io_uring_wait_cqe(ring, &cqe);
        if (cqe->res < 0) {
            Status s = Status::OsApiError(std::to_string(-cqe->res));
            io_uring_cqe_seen(ring, cqe);
            return s;
        }
        io_uring_cqe_seen(ring, cqe);
    }
    return Status::OK();
}

Status IoUringContext::H2SBatch(std::vector<IoUringTask>& tasks)
{
    size_t idx = 0;
    while (idx < tasks.size()) {
        size_t submitted = 0;
        while (idx < tasks.size() && submitted < ringEntries_) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (!sqe) {
                auto s = SubmitAndWaitBatch(&ring_, submitted);
                if (s.Failure()) [[unlikely]] { return s; }
                submitted = 0;
                continue;
            }
            const auto& task = tasks[idx];
            io_uring_prep_write(sqe, task.fd, task.addr, task.size, task.offset);
            ++idx;
            ++submitted;
        }
        auto s = SubmitAndWaitBatch(&ring_, submitted);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

Status IoUringContext::S2HBatch(std::vector<IoUringTask>& tasks)
{
    size_t idx = 0;
    while (idx < tasks.size()) {
        size_t submitted = 0;
        while (idx < tasks.size() && submitted < ringEntries_) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (!sqe) {
                auto s = SubmitAndWaitBatch(&ring_, submitted);
                if (s.Failure()) [[unlikely]] { return s; }
                submitted = 0;
                continue;
            }
            const auto& task = tasks[idx];
            io_uring_prep_read(sqe, task.fd, task.addr, static_cast<size_t>(task.size),
                              task.offset);
            ++idx;
            ++submitted;
        }
        auto s = SubmitAndWaitBatch(&ring_, submitted);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

}  // namespace UC::PosixStore
