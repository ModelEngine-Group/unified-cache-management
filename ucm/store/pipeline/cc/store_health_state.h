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
#ifndef UNIFIEDCACHE_PIPELINE_STORE_HEALTH_STATE_H
#define UNIFIEDCACHE_PIPELINE_STORE_HEALTH_STATE_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>
#include "store_health_config.h"

namespace UC::PipelineStore {

// I/O recording and passive queries are concurrent.
// The caller serializes probe updates and UpdatePassiveHealth transitions.
class StoreHealthState {
public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;

    struct PassiveWindowStats {
        size_t total{0};
        size_t failures{0};
    };

    explicit StoreHealthState(const StoreHealthConfig& config);
    bool RecordProbe(bool healthy, uint64_t generation, Time started, Time now);
    void RecordIo(bool healthy, uint64_t generation, Time now);
    bool PassiveThresholdExceeded(uint64_t generation, Time now) const;
    // Requires a positive threshold observation and the caller's transition lock.
    bool UpdatePassiveHealth(uint64_t generation, Time now);
    PassiveWindowStats GetPassiveWindowStats(Time now) const;

    bool Enabled() const { return enabled_.load(); }
    uint64_t Generation() const { return generation_.load(); }
    size_t FailureCount() const { return failureCount_; }
    size_t SampleCount() const { return probeResults_.size(); }
    std::chrono::milliseconds Cooldown() const { return cooldown_; }

private:
    class AtomicSecondCounter {
    public:
        // Ignores older or out-of-range seconds; counts saturate at UINT32_MAX.
        void Increment(std::chrono::seconds second);
        // Returns zero outside [now - window + 1s, now].
        uint32_t Count(std::chrono::seconds now, std::chrono::seconds window) const;
        // Concurrent increments may fall on either side of a reset.
        void Reset() { value_.store(0, std::memory_order_relaxed); }

    private:
        static constexpr uint64_t kCountMask = UINT32_MAX;
        // High 32 bits: monotonic second; low 32 bits: count.
        std::atomic<uint64_t> value_{0};
    };

    struct alignas(64) Bucket {
        AtomicSecondCounter total;
        AtomicSecondCounter failures;
    };

    bool ToHealthy(Time now);
    bool ToUnhealthy(Time now);

    StoreHealthConfig config_;
    std::atomic<bool> enabled_{true};
    std::atomic<uint64_t> generation_{0};
    // Avoid scanning the window when no failures are recent.
    AtomicSecondCounter recentFailures_;
    std::deque<bool> probeResults_;
    size_t failureCount_{0};
    std::vector<Bucket> buckets_;
    std::chrono::milliseconds cooldown_{0};
    Time recoverAfter_{};
    std::optional<Time> recoveredAt_;
};

}  // namespace UC::PipelineStore

#endif
