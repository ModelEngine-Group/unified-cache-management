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
#ifndef UNIFIEDCACHE_TIMER_H
#define UNIFIEDCACHE_TIMER_H

#include <chrono>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "logger/logger.h"
#include "status/status.h"

namespace UC {

template <typename Callable>
class Timer {
public:
    Timer(const std::chrono::seconds& interval, Callable&& callable)
        : _interval(interval), _callable(callable), _running(false)
    {

    }
    ~Timer()
    {
        {
            std::lock_guard<std::mutex> lock(this->_mutex);
            this->_running = false;
        }
        this->_cv.notify_one();
        if (this->_thread.joinable()) {
            this->_thread.join();
        }
    }
    Status Start()
    {
        std::lock_guard<std::mutex> lock(this->_mutex);
        if (this->_running) {
            return Status::OK();
        }
        try {
            this->_thread = std::thread(&Timer::Runner, this);
            this->_running = true;
            return Status::OK();
        } catch (const std::exception& e) {
            UC_ERROR("Failed({}) to start timer thread.", e.what());
            return Status::OutOfMemory();
        }
    }

private:
    void Runner()
    {
        while (this->_running) {
            try {
                {
                    std::unique_lock<std::mutex> lock(this->_mutex);
                    this->_cv.wait_for(lock, this->_interval, [this] { return !this->_running; });
                }
                if (!this->_running) {
                    break;
                }
                this->_callable();
            } catch (const std::exception& e) {
                UC_WARN("Failed({}) to run timer.", e.what());
            }
        }
    }

private:
    std::chrono::seconds _interval;
    Callable _callable;
    std::thread _thread;
    std::mutex _mutex;
    std::condition_variable _cv;
    bool _running;
};

} // namespace UC

#endif
