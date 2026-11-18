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
#pragma once

#include <chrono>
#include <functional>
#include "kv_types.h"

namespace kv {

using RefreshCallback = std::function<Status()>;

class RefreshScheduler {
public:
    virtual ~RefreshScheduler() = default;

    // Starts one resident refresh worker. The implementation is delivered by
    // the RefreshScheduler shard.
    //
    // interval: the fixed timer period between refreshes. It must be greater
    // than zero.
    // callback: the refresh operation invoked serially by the resident worker
    // for timer, IO-failure, and manual triggers. It must not be empty, and
    // its Status is returned to synchronous RefreshNow callers. Stop waits for
    // an in-progress callback to finish, so the callback must return within a
    // bounded time.
    virtual Status Start(std::chrono::milliseconds interval, RefreshCallback callback) = 0;
    // Stops accepting triggers, wakes the worker, joins it, and waits for the
    // refresh in progress to finish.
    virtual void Stop() = 0;

    virtual void TriggerByTimer() = 0;
    virtual void TriggerByIoFailure(const Status& status) = 0;
    // Synchronously waits for one accepted refresh.
    virtual Status RefreshNow() = 0;
};

}  // namespace kv
