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
#ifndef UNIFIEDCACHE_INFRA_TIMER_H
#define UNIFIEDCACHE_INFRA_TIMER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace UC {

template <typename Callable>
class Timer {
public:
    Timer(const std::chrono::seconds& interval, Callable&& callable)
        : interval_(interval), callable_(callable), running_(false)
    {
    }
    ~Timer()
    {
        {
            std::lock_guard<std::mutex> lg(this->mutex_);
            this->running_ = false;
            this->cv_.notify_one();
        }
        if (this->thread_.joinable()) { this->thread_.join(); }
    }
    bool Start()
    {
        {
            std::lock_guard<std::mutex> lg(this->mutex_);
            if (this->running_) { return true; }
        }
        try {
            this->running_ = true;
            this->thread_ = std::thread(&Timer::Runner, this);
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    void Runner()
    {
        while (this->running_) {
            {
                std::unique_lock<std::mutex> lg(this->mutex_);
                this->cv_.wait_for(lg, this->interval_, [this] { return !this->running_; });
                if (!this->running_) { break; }
            }
            this->callable_();
        }
    }

private:
    std::chrono::seconds interval_;
    Callable callable_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_;
};

} // namespace UC

#endif
