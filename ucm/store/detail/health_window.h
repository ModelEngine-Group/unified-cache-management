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
#ifndef UNIFIEDCACHE_STORE_DETAIL_HEALTH_WINDOW_H
#define UNIFIEDCACHE_STORE_DETAIL_HEALTH_WINDOW_H

#include <deque>
#include "detail/store_health_config.h"

namespace UC::Detail {

class HealthWindow {
public:
    explicit HealthWindow(const StoreHealthConfig& config = {}, bool healthy = true)
        : capacity_(config.healthWindowSize), threshold_(config.failureThreshold), healthy_(healthy)
    {
    }

    void Record(bool healthy)
    {
        if (results_.size() == capacity_) {
            if (!results_.front()) { --failures_; }
            results_.pop_front();
        }
        results_.push_back(healthy);
        if (!healthy) { ++failures_; }
        if (healthy_ && failures_ >= threshold_) {
            healthy_ = false;
        } else if (!healthy_ && results_.size() == capacity_ && failures_ == 0) {
            healthy_ = true;
        }
    }

    bool Healthy() const { return healthy_; }
    size_t FailureCount() const { return failures_; }
    size_t SampleCount() const { return results_.size(); }
    const std::deque<bool>& Results() const { return results_; }

private:
    size_t capacity_;
    size_t threshold_;
    bool healthy_;
    std::deque<bool> results_;
    size_t failures_{0};
};

}  // namespace UC::Detail

#endif
