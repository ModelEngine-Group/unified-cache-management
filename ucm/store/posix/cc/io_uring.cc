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
#include <cstring>
#include <fmt/core.h>

namespace UC::PosixStore {

Status IoUringContext::Init(int32_t ringEntries)
{
    if (initialized_) { return Status::OK(); }
    int ret = io_uring_queue_init(ringEntries, &ring_, 0);
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

}  // namespace UC::PosixStore
