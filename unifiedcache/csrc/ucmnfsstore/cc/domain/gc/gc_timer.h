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
#ifndef UNIFIEDCACHE_GC_TIMER_H
#define UNIFIEDCACHE_GC_TIMER_H

#include "gc_runner.h"
#include "template/timer.h"

namespace UC {

class GCTimer {
public:
    void Setup(const size_t interval, const uint64_t capacity, const float thresholdPercent, const float cleanupPercent)
    {
        this->_interval = std::chrono::seconds(interval);
        this->_threshold = capacity * (thresholdPercent / 100.0f);
        this->_percent = cleanupPercent / 100.0f;
    }
    Status Start()
    {
        try {
            this->_timer = std::make_unique<Timer<GCRunner>>(this->_interval, std::move(GCRunner(this->_threshold, this->_percent)));
        } catch (const std::exception& e) {
            UC_ERROR("Failed({}) to make GC Timer.", e.what());
            return Status::OutOfMemory();
        }
        return this->_timer->Start();
    }

private:
    std::chrono::seconds _interval;
    uint64_t _threshold;
    float _percent;
    std::unique_ptr<Timer<GCRunner>> _timer;
};

} // namespace UC
#endif