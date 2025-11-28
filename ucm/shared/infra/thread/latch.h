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
#ifndef UNIFIEDCACHE_INFRA_LATCH_H
#define UNIFIEDCACHE_INFRA_LATCH_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>

namespace UC {

class Latch {
public:
    explicit Latch(const size_t expected = 0) : counter_{expected} {}
    void Up() { ++this->counter_; }
    void Done(std::function<void(void)> finish) noexcept
    {
        auto counter = this->counter_.load(std::memory_order_acquire);
        while (counter > 0) {
            auto desired = counter - 1;
            if (this->counter_.compare_exchange_weak(counter, desired, std::memory_order_acq_rel)) {
                if (desired == 0) {
                    if (finish) { finish(); }
                    std::lock_guard<std::mutex> lg(this->mutex_);
                    this->cv_.notify_all();
                }
                return;
            }
            counter = this->counter_.load(std::memory_order_acquire);
        }
    }
    void Wait()
    {
        std::unique_lock<std::mutex> lk(this->mutex_);
        if (this->counter_ == 0) { return; }
        this->cv_.wait(lk, [this] { return this->counter_ == 0; });
    }

protected:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<size_t> counter_;
};

} // namespace UC

#endif // UNIFIEDCACHE_INFRA_LATCH_H
