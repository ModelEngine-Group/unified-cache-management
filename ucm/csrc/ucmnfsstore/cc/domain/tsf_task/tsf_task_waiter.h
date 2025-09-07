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
#ifndef UNIFIEDCACHE_TSF_TASK_WAITER_H
#define UNIFIEDCACHE_TSF_TASK_WAITER_H

#include <spdlog/stopwatch.h>
#include "thread/latch.h"

namespace UC {

class TsfTaskWaiter : public Latch {
    spdlog::stopwatch _sw;

public:
    explicit TsfTaskWaiter(const size_t expected = 1) : Latch{expected} {}
    using Latch::Wait;
    bool Wait(const size_t timeoutMs)
    {
        if (timeoutMs == 0) {
            this->Wait();
            return true;
        }
        auto finish = false;
        {
            std::unique_lock<std::mutex> lk(this->_mutex);
            if (this->_counter == 0) { return true; }
            auto elapsed = (size_t)this->_sw.elapsed_ms().count();
            if (elapsed < timeoutMs) {
                finish = this->_cv.wait_for(lk, std::chrono::milliseconds(timeoutMs - elapsed),
                                            [this] { return this->_counter == 0; });
            }
        }
        return finish;
    }
    bool Finish() { return this->_counter == 0; }
};

} // namespace UC

#endif
