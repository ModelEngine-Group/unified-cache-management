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

#include "logger/logger.h"
#include "thread/latch.h"
#include "time/stopwatch.h"

namespace UC {

class TsfTaskWaiter : public Latch {
public:
    TsfTaskWaiter(const size_t id, const size_t size, const size_t number, const std::string& brief)
        : Latch{number}, id_{id}, size_{size}, number_{number}, brief_{brief}
    {
    }
    void Done()
    {
        if (Latch::Done() == 0) {
            auto elapsed = this->sw_.Elapsed().count();
            UC_DEBUG("Task({},{},{},{}) finished, elapsed={:.06f}s, bw={:.06f}GB/s.", this->id_,
                     this->brief_, this->number_, this->size_, elapsed,
                     this->size_ / elapsed / (1ULL << 30));
            this->Notify();
        }
    }
    using Latch::Wait;
    bool Wait(const size_t timeoutMs)
    {
        if (timeoutMs == 0) {
            this->Wait();
            return true;
        }
        auto finish = false;
        {
            std::unique_lock<std::mutex> lk(this->mutex_);
            if (this->counter_ == 0) { return true; }
            auto elapsed = (size_t)this->sw_.ElapsedMs().count();
            if (elapsed < timeoutMs) {
                finish = this->cv_.wait_for(lk, std::chrono::milliseconds(timeoutMs - elapsed),
                                            [this] { return this->counter_ == 0; });
            }
        }
        if (!finish) {
            UC_WARN("Task({},{},{},{}) timeout, elapsed={:.06f}s.", this->id_, this->brief_,
                    this->number_, this->size_, this->sw_.Elapsed().count());
        }
        return finish;
    }
    bool Finish() { return this->counter_ == 0; }

private:
    size_t id_;
    size_t size_;
    size_t number_;
    std::string brief_;
    StopWatch sw_;
};

} // namespace UC

#endif // UNIFIEDCACHE_TSF_TASK_WAITER_H
