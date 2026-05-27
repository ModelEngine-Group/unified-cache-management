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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include "connection_internal.h"
#include "template/spsc_ring_queue.h"
#include "transport_task_manager.h"

namespace UC::ASU {

struct PendingRequest {
    std::shared_ptr<TransportTaskContext> ctx;
    std::shared_ptr<ConnectionChannel> channel;
    volatile uint32_t* flagBuffer{nullptr};
    std::uint64_t deadlineMs{0};
};

class CompletionPoller {
public:
    using ReportFailureFunc = std::function<void(const std::shared_ptr<ConnectionChannel>&)>;
    using ReleaseFlagBufferFunc = std::function<void(volatile uint32_t*)>;

    CompletionPoller() = default;
    ~CompletionPoller();

    void Start(std::size_t queueCapacity, ReportFailureFunc reportFn, ReleaseFlagBufferFunc releaseFn);
    void Stop();

    void SubmitPending(PendingRequest req);

private:
    void PollLoop();
    void Finalize(PendingRequest& req, Status status);

    UC::SpscRingQueue<PendingRequest> pendingQueue_;
    ReportFailureFunc reportFailureFn_;
    ReleaseFlagBufferFunc releaseFlagBufferFn_;

    std::thread pollerThread_;
    std::atomic<bool> stop_{false};
};

}  // namespace UC::ASU
