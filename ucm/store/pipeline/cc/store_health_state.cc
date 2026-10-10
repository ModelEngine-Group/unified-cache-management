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

StoreHealthState::StoreHealthState(const StoreHealthConfig& config) : config_(config) {}

std::chrono::milliseconds StoreHealthState::CooldownRemaining(Time now) const
{
    return !Enabled() && now < recoverAfter_
               ? std::chrono::ceil<std::chrono::milliseconds>(recoverAfter_ - now)
               : std::chrono::milliseconds{0};
}

bool StoreHealthState::ToHealthy(Time now)
{
    ++generation_;
    // In-flight increments may cross this baseline; passive counts are approximate.
    windowStartSuccesses_.store(ioSuccesses_.load(std::memory_order_relaxed));
    windowStartFailures_.store(ioFailures_.load(std::memory_order_relaxed));
    passiveSnapshots_.clear();
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

void StoreHealthState::RecordIo(bool healthy, uint64_t generation)
{
    if (!config_.passiveEnabled || !Enabled() || generation != Generation()) { return; }
    auto& counter = healthy ? ioSuccesses_ : ioFailures_;
    counter.fetch_add(1, std::memory_order_relaxed);
}

void StoreHealthState::SamplePassiveWindow(Time now)
{
    if (!config_.passiveEnabled) { return; }
    passiveSnapshots_.push_back({now, ioSuccesses_.load(std::memory_order_relaxed),
                                 ioFailures_.load(std::memory_order_relaxed)});
    const auto cutoff = now - config_.passiveWindow;
    while (!passiveSnapshots_.empty() && passiveSnapshots_.front().time <= cutoff) {
        const auto& snapshot = passiveSnapshots_.front();
        windowStartSuccesses_.store(snapshot.successes);
        windowStartFailures_.store(snapshot.failures);
        passiveSnapshots_.pop_front();
    }
}

bool StoreHealthState::PassiveThresholdExceeded(uint64_t generation) const
{
    if (!config_.passiveEnabled || generation != Generation() || !Enabled()) { return false; }
    const auto start = windowStartFailures_.load();
    return ioFailures_.load(std::memory_order_relaxed) - start >= config_.passiveFailureThreshold;
}

bool StoreHealthState::UpdatePassiveHealth(uint64_t generation, Time now)
{
    if (generation != Generation() || !Enabled()) { return false; }
    return ToUnhealthy(now);
}

StoreHealthState::PassiveWindowStats StoreHealthState::GetPassiveWindowStats() const
{
    // Read baselines first so a concurrent sample cannot cause unsigned underflow.
    const auto startSuccesses = windowStartSuccesses_.load();
    const auto startFailures = windowStartFailures_.load();
    const auto successes = ioSuccesses_.load(std::memory_order_relaxed) - startSuccesses;
    const auto failures = ioFailures_.load(std::memory_order_relaxed) - startFailures;
    return {successes + failures, failures};
}

}  // namespace UC::PipelineStore
