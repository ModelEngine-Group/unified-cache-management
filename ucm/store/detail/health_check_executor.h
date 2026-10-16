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

    struct Worker {
        std::mutex mutex;
        std::condition_variable cv;
        std::function<Status()> check;
        Status status{Status::Error()};
        bool done{false};
        bool stop{false};
        bool exited{false};
        std::thread thread;

        Worker()
        {
            // Linux threads inherit the creating monitor thread's CPU affinity.
            thread = std::thread([this] {
                auto nameStatus =
                    CpuAffinity::SetCurrentThreadName("ucm_health_backend_check_worker");
                if (nameStatus.Failure()) {
                    UC_WARN("Failed({}) to set UCM health I/O thread name.", nameStatus);
                }
                std::unique_lock<std::mutex> lock(mutex);
                while (true) {
                    cv.wait(lock, [this] { return stop || check; });
                    if (!check) { break; }
                    auto work = std::move(check);
                    check = nullptr;
                    lock.unlock();
                    auto result = Status::Error();
                    try {
                        result = work();
                    } catch (const std::exception& e) {
                        result = Status::Error(e.what());
                    } catch (...) {
                        result = Status::Error("health check threw an unknown exception");
                    }
                    lock.lock();
                    status = std::move(result);
                    done = true;
                    cv.notify_all();
                }
                exited = true;
            });
        }
        ~Worker()
        {
            if (!thread.joinable()) { return; }
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = true;
            }
            cv.notify_all();
            thread.join();
        }
    };

public:
    explicit HealthCheckExecutor(std::chrono::milliseconds timeout,
                                 size_t maxInFlight = kMaxInFlight)
        : timeout_(timeout), maxInFlight_(maxInFlight)
    {
    }
    ~HealthCheckExecutor() { Stop(); }

    // Calls to Run on same executor must not overlap.
    Status Run(std::function<Status()> check)
    {
        std::lock_guard<std::mutex> runLock(runMutex_);
        auto it = cleanup_.begin();
        while (it != cleanup_.end()) {
            std::unique_lock<std::mutex> lock((*it)->mutex, std::try_to_lock);
            if (!lock.owns_lock() || !(*it)->exited) {
                ++it;
                continue;
            }
            lock.unlock();
            it = cleanup_.erase(it);
        }
        if (!current_worker_) {
            if (maxInFlight_ != 0 && cleanup_.size() >= maxInFlight_) {
                UC_WARN(
                    "Health check executor reached max threads, rejecting probe with "
                    "Timeout, in-flight={}.",
                    cleanup_.size());
                return Status::Timeout();
            }
            try {
                current_worker_ = std::make_unique<Worker>();
            } catch (const std::exception& e) {
                return Status::Error(e.what());
            }
        }

        auto& worker = *current_worker_;
        std::unique_lock<std::mutex> lock(worker.mutex);
        worker.check = std::move(check);
        worker.done = false;
        worker.cv.notify_all();
        const auto finished = worker.cv.wait_for(lock, timeout_, [&] { return worker.done; });
        if (!finished) {
            worker.stop = true;
            worker.cv.notify_all();
            cleanup_.push_back(std::move(current_worker_));
            return Status::Timeout();
        }
        return worker.status;
    }

    void Stop()
    {
        std::lock_guard<std::mutex> lock(runMutex_);
        current_worker_.reset();
        cleanup_.clear();
    }

private:
    std::chrono::milliseconds timeout_;
    size_t maxInFlight_;
    std::mutex runMutex_;
    std::unique_ptr<Worker> current_worker_;
    std::vector<std::unique_ptr<Worker>> cleanup_;
};

}  // namespace UC::Detail

#endif
