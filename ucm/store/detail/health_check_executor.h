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
#include "thread/cpu_affinity.h"

namespace UC::Detail {

class HealthCheckExecutor {
    static constexpr size_t kMaxInFlight = 64;

    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        std::function<Status()> check;
        Status status{Status::Error()};
        bool done{false};
        bool stop{false};
        bool exited{false};
    };
    struct Worker {
        std::shared_ptr<State> state;
        std::thread thread;
    };

public:
    // A zero limit allows replacement workers while timed-out I/O is still running.
    explicit HealthCheckExecutor(std::chrono::milliseconds timeout,
                                 size_t maxInFlight = kMaxInFlight)
        : timeout_(timeout), maxInFlight_(maxInFlight)
    {
    }
    ~HealthCheckExecutor() { Stop(); }

    Status Run(std::function<Status()> check)
    {
        std::shared_ptr<State> state;
        {
            std::lock_guard<std::mutex> runLock(runMutex_);
            ReapFinished();
            state = std::move(current_);
            if (!state) {
                if (maxInFlight_ != 0 && workers_.size() >= maxInFlight_) {
                    UC_WARN(
                        "Health check executor reached max threads, rejecting probe with Timeout, "
                        "in-flight={}.",
                        workers_.size());
                    return Status::Timeout();
                }
                try {
                    workers_.reserve(workers_.size() + 1);
                    state = std::make_shared<State>();
                    // Linux threads inherit the creating monitor thread's CPU affinity.
                    workers_.push_back(Worker{
                        state, std::thread([state] {
                            auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_health_io");
                            if (nameStatus.Failure()) {
                                UC_WARN("Failed({}) to set UCM health I/O thread name.",
                                        nameStatus);
                            }
                            std::unique_lock<std::mutex> lock(state->mutex);
                            while (true) {
                                state->cv.wait(lock, [&] { return state->stop || state->check; });
                                if (state->stop) { break; }
                                auto work = std::move(state->check);
                                state->check = nullptr;
                                lock.unlock();
                                auto status = Status::Error();
                                try {
                                    status = work();
                                } catch (const std::exception& e) {
                                    status = Status::Error(e.what());
                                } catch (...) {
                                    status =
                                        Status::Error("health check threw an unknown exception");
                                }
                                lock.lock();
                                state->status = std::move(status);
                                state->done = true;
                                state->cv.notify_all();
                            }
                            state->exited = true;
                        })});
                } catch (const std::exception& e) {
                    return Status::Error(e.what());
                }
            }
            std::lock_guard<std::mutex> stateLock(state->mutex);
            state->check = std::move(check);
            state->done = false;
            state->cv.notify_all();
        }

        std::unique_lock<std::mutex> stateLock(state->mutex);
        const auto finished = state->cv.wait_for(stateLock, timeout_, [&] { return state->done; });
        if (!finished) {
            // Retire this worker; its late result must never satisfy a later probe.
            state->stop = true;
            state->cv.notify_all();
            return Status::Timeout();
        }
        auto status = state->status;
        stateLock.unlock();
        {
            std::lock_guard<std::mutex> runLock(runMutex_);
            std::lock_guard<std::mutex> lock(state->mutex);
            // Concurrent callers may create extra workers; keep only one idle worker.
            if (!state->stop && !current_) {
                current_ = state;
            } else {
                state->stop = true;
                state->cv.notify_all();
            }
        }
        return status;
    }

    void Stop()
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        current_.reset();
        for (auto& worker : workers_) {
            {
                std::lock_guard<std::mutex> stateLock(worker.state->mutex);
                worker.state->stop = true;
            }
            worker.state->cv.notify_all();
        }
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
            bool exited = false;
            {
                std::lock_guard<std::mutex> stateLock(worker->state->mutex);
                exited = worker->state->exited;
            }
            if (!exited) {
                ++worker;
                continue;
            }
            if (worker->thread.joinable()) { worker->thread.join(); }
            worker = workers_.erase(worker);
        }
    }
    std::chrono::milliseconds timeout_;
    size_t maxInFlight_;
    std::mutex runMutex_;
    std::shared_ptr<State> current_;
    std::vector<Worker> workers_;
};

}  // namespace UC::Detail

#endif
