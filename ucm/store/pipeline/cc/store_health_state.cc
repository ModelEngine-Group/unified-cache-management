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
#include "store_health_state.h"
#include <algorithm>

namespace UC::PipelineStore {

void StoreHealthState::AtomicSecondCounter::Increment(std::chrono::seconds second)
{
    const auto stamp = static_cast<uint64_t>(second.count());
    if (stamp > kCountMask) { return; }
    auto value = value_.load(std::memory_order_relaxed);
    for (;;) {
        const auto previousSecond = value >> 32;
        if (previousSecond > stamp) { return; }
        const auto count = previousSecond == stamp ? value & kCountMask : 0;
        if (count == kCountMask) { return; }
        const auto next = (stamp << 32) | (count + 1);
        if (value_.compare_exchange_weak(value, next, std::memory_order_relaxed)) { return; }
    }
}

uint32_t StoreHealthState::AtomicSecondCounter::Count(std::chrono::seconds now,
                                                      std::chrono::seconds window) const
{
    const auto value = value_.load(std::memory_order_relaxed);
    const auto age = now.count() - static_cast<int64_t>(value >> 32);
    if (age < 0 || age >= window.count()) { return 0; }
    return static_cast<uint32_t>(value & kCountMask);
}

StoreHealthState::StoreHealthState(const StoreHealthConfig& config)
    : config_(config), buckets_(config.passiveWindow.count())
{
}

bool StoreHealthState::ToHealthy(Time now)
{
    ++generation_;
    // In-flight increments may overlap the reset; passive counts are approximate.
    for (auto& bucket : buckets_) {
        bucket.total.Reset();
        bucket.failures.Reset();
    }
    recentFailures_.Reset();
    recoveredAt_ = now;
    enabled_.store(true);
    return true;
}

bool StoreHealthState::ToUnhealthy(Time now)
{
    if (recoveredAt_ && now - *recoveredAt_ < config_.stableResetAfter) {
        const auto next = static_cast<long double>(cooldown_.count()) * config_.backoffFactor;
        cooldown_ = std::chrono::milliseconds{static_cast<int64_t>(
            std::min(next, static_cast<long double>(config_.maxCooldown.count())))};
    } else {
        cooldown_ = config_.initialCooldown;
    }
    recoverAfter_ = now + cooldown_;
    enabled_.store(false);
    ++generation_;
    probeResults_.clear();
    failureCount_ = 0;
    return true;
}

bool StoreHealthState::RecordProbe(bool healthy, uint64_t generation, Time started, Time now)
{
    if (generation != Generation()) { return false; }
    if (probeResults_.size() == config_.healthWindowSize) {
        if (!probeResults_.front()) { --failureCount_; }
        probeResults_.pop_front();
    }
    probeResults_.push_back(healthy);
    if (!healthy) { ++failureCount_; }

    if (Enabled() && failureCount_ >= config_.failureThreshold) { return ToUnhealthy(now); }
    if (!Enabled() && healthy && started >= recoverAfter_ &&
        probeResults_.size() == config_.healthWindowSize && failureCount_ == 0) {
        return ToHealthy(now);
    }
    return false;
}

void StoreHealthState::RecordIo(bool healthy, uint64_t generation, Time now)
{
    if (!config_.passiveEnabled || !Enabled() || generation != Generation()) { return; }
    const auto second = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
    auto& bucket = buckets_[second.count() % buckets_.size()];
    bucket.total.Increment(second);
    if (!healthy) {
        bucket.failures.Increment(second);
        recentFailures_.Increment(second);
    }
}

bool StoreHealthState::PassiveThresholdExceeded(uint64_t generation, Time now) const
{
    if (!config_.passiveEnabled || generation != Generation() || !Enabled()) { return false; }
    const auto second = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
    if (recentFailures_.Count(second, config_.passiveWindow) == 0) { return false; }
    return GetPassiveWindowStats(now).failures >= config_.passiveFailureThreshold;
}

bool StoreHealthState::UpdatePassiveHealth(uint64_t generation, Time now)
{
    if (generation != Generation() || !Enabled()) { return false; }
    return ToUnhealthy(now);
}

StoreHealthState::PassiveWindowStats StoreHealthState::GetPassiveWindowStats(Time now) const
{
    const auto second = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
    PassiveWindowStats stats;
    for (const auto& sample : buckets_) {
        const auto failures = sample.failures.Count(second, config_.passiveWindow);
        const auto total = sample.total.Count(second, config_.passiveWindow);
        stats.total += total;
        stats.failures += std::min(failures, total);
    }
    return stats;
}

}  // namespace UC::PipelineStore
