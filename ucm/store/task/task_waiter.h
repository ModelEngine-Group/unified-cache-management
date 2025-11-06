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
#ifndef UNIFIEDCACHE_ITASK_WAITER_H
#define UNIFIEDCACHE_ITASK_WAITER_H

#include <chrono>
#include "thread/latch.h"

namespace UC {

class TaskWaiter : public Latch {
protected:
    double startTp_;

public:
    TaskWaiter(const size_t expected, const double startTp) : Latch{expected}, startTp_{startTp} {}
    virtual ~TaskWaiter() = default;
    virtual void Set(const size_t expected) noexcept { this->counter_.store(expected); }
    using Latch::Wait;
    virtual bool Wait(const size_t timeoutMs) noexcept
    {
        if (timeoutMs == 0) {
            this->Wait();
            return true;
        }
        std::unique_lock<std::mutex> ul(this->mutex_);
        if (this->counter_ == 0) { return true; }
        auto elapsed = std::chrono::duration<double>(NowTp() - startTp_);
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        auto timeMs = std::chrono::milliseconds(timeoutMs);
        if (timeMs <= elapsedMs) { return false; }
        auto remainMs = timeMs - elapsedMs;
        return this->cv_.wait_for(ul, remainMs, [this] { return this->counter_ == 0; });
    }
    virtual bool Finish() noexcept { return this->counter_ == 0; }

private:
    static double NowTp() noexcept
    {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double>(now).count();
    }
};

} // namespace UC

#endif
