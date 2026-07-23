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
 */
#ifndef UNIFIEDCACHE_STORE_DETAIL_HEALTH_CHECK_EXECUTOR_H
#define UNIFIEDCACHE_STORE_DETAIL_HEALTH_CHECK_EXECUTOR_H

#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "logger/logger.h"
#include "status/status.h"

namespace UC::Detail {

class HealthCheckExecutor {
    static constexpr size_t kMaxInFlight = 64;

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        Status status{Status::Error()};
        bool done{false};
    };
    struct Worker {
        std::shared_ptr<State> state;
        std::thread thread;
    };

public:
    explicit HealthCheckExecutor(std::chrono::milliseconds timeout) : timeout_(timeout) {}
    ~HealthCheckExecutor() { Stop(); }

    Status Run(std::function<Status()> check)
    {
        auto state = std::make_shared<State>();
        {
            std::lock_guard<std::mutex> runLock(runMutex_);
            ReapFinished();
            if (workers_.size() >= kMaxInFlight) {
                UC_WARN(
                    "Health check executor reached max threads, rejecting probe with Timeout, "
                    "in-flight={}.",
                    workers_.size());
                return Status::Timeout();
            }
            try {
                if (workers_.capacity() < kMaxInFlight) { workers_.reserve(kMaxInFlight); }
                workers_.push_back(Worker{
                    state, std::thread([state, check = std::move(check)]() mutable {
                        auto status = Status::Error();
                        try {
                            status = check();
                        } catch (const std::exception& e) {
                            status = Status::Error(e.what());
                        } catch (...) {
                            status = Status::Error("health check threw an unknown exception");
                        }
                        {
                            std::lock_guard<std::mutex> lock(state->mutex);
                            state->status = std::move(status);
                            state->done = true;
                        }
                        state->cv.notify_all();
                    })});
            } catch (const std::exception& e) {
                return Status::Error(e.what());
            }
        }

        std::unique_lock<std::mutex> stateLock(state->mutex);
        const auto finished = state->cv.wait_for(stateLock, timeout_, [&] { return state->done; });
        if (!finished) { return Status::Timeout(); }
        auto status = state->status;
        stateLock.unlock();
        {
            std::lock_guard<std::mutex> runLock(runMutex_);
            ReapFinished();
        }
        return status;
    }

    void Stop()
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        for (auto& worker : workers_) {
            if (worker.thread.joinable()) { worker.thread.join(); }
        }
        workers_.clear();
    }

private:
    void ReapFinished()
    {
        auto worker = workers_.begin();
        while (worker != workers_.end()) {
            bool done = false;
            {
                std::lock_guard<std::mutex> stateLock(worker->state->mutex);
                done = worker->state->done;
            }
            if (!done) {
                ++worker;
                continue;
            }
            if (worker->thread.joinable()) { worker->thread.join(); }
            worker = workers_.erase(worker);
        }
    }
    std::chrono::milliseconds timeout_;
    std::mutex runMutex_;
    std::vector<Worker> workers_;
};

}  // namespace UC::Detail

#endif
