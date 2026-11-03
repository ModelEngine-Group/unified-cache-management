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
#ifndef UNIFIEDCACHE_POSIX_STORE_CC_GC_EXECUTOR_H
#define UNIFIEDCACHE_POSIX_STORE_CC_GC_EXECUTOR_H

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include "logger/logger.h"
#include "status/status.h"
#include "thread/cpu_affinity.h"
#include "thread/thread_pool.h"

namespace UC::PosixStore {

// Runs GC lock and heartbeat calls of all backends with a time limit, so a stuck path
// cannot hang GC. A call that times out keeps running until it returns, so it must own
// everything it touches and be harmless when it finishes late.
class GcExecutor {
    enum State : int { kQueued, kRunning, kFinished, kAbandoned };
    struct Task {
        std::function<Status()> op;
        std::promise<Status> result;
        std::atomic<int> state{kQueued};
    };

public:
    Status Setup(size_t nWorker, size_t timeoutMs)
    {
        timeoutMs_ = timeoutMs;
        maxRunning_ = nWorker * 2;
        const auto success =
            pool_
                .SetWorkerInitFn([](auto&) {
                    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_posix_gcop");
                    if (nameStatus.Failure()) {
                        UC_WARN("Failed({}) to set posix GC executor worker name.", nameStatus);
                    }
                    return true;
                })
                .SetWorkerFn([running = running_](std::shared_ptr<Task>& task, auto&) {
                    auto queued = static_cast<int>(kQueued);
                    if (task->state.compare_exchange_strong(queued, kRunning)) {
                        running->fetch_add(1);
                        auto status = task->op();
                        running->fetch_sub(1);
                        auto expected = static_cast<int>(kRunning);
                        if (task->state.compare_exchange_strong(expected, kFinished)) {
                            task->result.set_value(std::move(status));
                        }
                    }
                })
                .SetWorkerTimeoutFn([](auto&, auto) {}, timeoutMs)
                .SetNWorker(nWorker)
                .Run();
        if (success) {
            return Status::OK();
        } else {
            return Status::Error("failed to run posix GC executor thread pool");
        }
    }

    Status Run(std::function<Status()> op)
    {
        if (running_->load() < static_cast<int64_t>(maxRunning_)) {
            auto task = std::make_shared<Task>();
            task->op = std::move(op);
            auto future = task->result.get_future();
            pool_.Push(std::shared_ptr<Task>{task});
            auto queued = static_cast<int>(kQueued);
            auto running = static_cast<int>(kRunning);
            if (future.wait_for(std::chrono::milliseconds(timeoutMs_)) ==
                std::future_status::ready) {
                return future.get();
            } else if (task->state.compare_exchange_strong(queued, kAbandoned)) {
                return Status::TimeoutInQueue();
            } else if (task->state.compare_exchange_strong(running, kAbandoned)) {
                return Status::Timeout();
            } else {
                return future.get();
            }
        } else {
            // Too many calls are stuck. This call never starts, so it does not count against
            // the backend it was meant for.
            return Status::TimeoutInQueue();
        }
    }

private:
    size_t timeoutMs_{0};
    size_t maxRunning_{0};
    std::shared_ptr<std::atomic<int64_t>> running_{std::make_shared<std::atomic<int64_t>>(0)};
    ThreadPool<std::shared_ptr<Task>> pool_;
};

}  // namespace UC::PosixStore

#endif
