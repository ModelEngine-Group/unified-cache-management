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
#ifndef UNIFIEDCACHE_STORE_DETAIL_HEALTH_TRACKER_H
#define UNIFIEDCACHE_STORE_DETAIL_HEALTH_TRACKER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "detail/store_health_config.h"
#include "logger/logger.h"
#include "status/status.h"

namespace UC::Detail {

// Sliding window over the most recent probe results.
class HealthWindow {
public:
    explicit HealthWindow(size_t capacity) : capacity_(capacity) {}

    void Record(bool healthy)
    {
        if (results_.size() == capacity_) {
            if (!results_.front()) { --failures_; }
            results_.pop_front();
        }
        results_.push_back(healthy);
        if (!healthy) { ++failures_; }
    }

    void Clear()
    {
        results_.clear();
        failures_ = 0;
    }

    bool Full() const { return results_.size() == capacity_; }
    size_t FailureCount() const { return failures_; }
    size_t SampleCount() const { return results_.size(); }
    // Oldest result first.
    const std::deque<bool>& Results() const { return results_; }

private:
    size_t capacity_;
    std::deque<bool> results_;
    size_t failures_{0};
};

// Record and GetStats may run concurrently; the caller serializes UpdateWindowStart and Reset.
class PassiveIoWindow {
public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;

    struct Stats {
        uint64_t successes{0};
        uint64_t failures{0};
    };

    explicit PassiveIoWindow(std::chrono::milliseconds window) : window_{window} {}

    void Record(bool success)
    {
        auto& counter = success ? totalSuccess_ : totalFailure_;
        counter.fetch_add(1, std::memory_order_relaxed);
    }

    void UpdateWindowStart(Time now)
    {
        snapshots_.push_back({now, totalSuccess_.load(std::memory_order_relaxed),
                              totalFailure_.load(std::memory_order_relaxed)});
        const auto cutoff = now - window_;
        while (!snapshots_.empty() && snapshots_.front().time <= cutoff) {
            const auto& snapshot = snapshots_.front();
            windowStartSuccess_.store(snapshot.successes);
            windowFailure_.store(snapshot.failures);
            snapshots_.pop_front();
        }
    }

    Stats GetStats() const
    {
        // Read baselines first so a concurrent sample cannot cause unsigned underflow.
        const auto windowStartSuccess = windowStartSuccess_.load();
        const auto windowFailure = windowFailure_.load();
        return {totalSuccess_.load(std::memory_order_relaxed) - windowStartSuccess,
                totalFailure_.load(std::memory_order_relaxed) - windowFailure};
    }

    void Reset()
    {
        // In-flight increments may cross this baseline; counts are approximate at recovery.
        windowStartSuccess_.store(totalSuccess_.load(std::memory_order_relaxed));
        windowFailure_.store(totalFailure_.load(std::memory_order_relaxed));
        snapshots_.clear();
    }

private:
    struct Snapshot {
        Time time;
        uint64_t successes;
        uint64_t failures;
    };

    std::chrono::milliseconds window_;
    std::atomic<uint64_t> totalSuccess_{0};
    std::atomic<uint64_t> totalFailure_{0};
    std::atomic<uint64_t> windowStartSuccess_{0};
    std::atomic<uint64_t> windowFailure_{0};
    std::deque<Snapshot> snapshots_;
};

enum class HealthSource { ActiveProbe, PassiveIo };

struct HealthTransition {
    bool healthy;
    HealthSource source;
    std::chrono::milliseconds cooldown;
    uint64_t generation;
};

struct HealthSnapshot {
    bool healthy;
    uint64_t generation;
    size_t probeSamples;
    size_t probeFailures;
    std::chrono::milliseconds cooldown;
    std::chrono::milliseconds cooldownRemaining;
    PassiveIoWindow::Stats passive;
};

// These statuses describe the request rather than the backend.
inline bool ShouldRecordResult(const Status& status)
{
    return status != Status::NotFound() && status != Status::StoreUnhealthy() &&
           status != Status::InvalidParam() && status != Status::DuplicateKey() &&
           status != Status::Unsupported() && status != Status::TimeoutInQueue();
}

// Healthy, Generation and RecordIo are lock-free unless RecordIo trips the breaker;
// every other member takes the internal lock, so any thread may call any member.
class HealthTracker {
public:
    using Clock = std::chrono::steady_clock;
    using Time = Clock::time_point;

    // module tags every log line so store-level and backend-level trackers are distinguishable.
    // The config must already be validated.
    HealthTracker(std::string module, std::string name, const StoreHealthConfig& config)
        : module_{std::move(module)},
          name_{std::move(name)},
          config_{config},
          passiveWindow_{config.passiveWindow},
          probeWindow_{config.healthWindowSize}
    {
    }
    HealthTracker(const HealthTracker&) = delete;
    HealthTracker& operator=(const HealthTracker&) = delete;

    bool Healthy() const { return healthy_.load(); }
    // Captured before issuing a probe or I/O; results carrying an older generation are dropped.
    uint64_t Generation() const { return generation_.load(); }

    // Record Active probe results, return not null if health state has changed
    std::optional<HealthTransition> RecordProbe(uint64_t generation, const Status& result,
                                                Time now = Clock::now())
    {
        if (result.Failure()) {
            UC_WARN("[{}] Active store health check({}) failed({}).", module_, name_, result);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (generation != generation_.load()) { return std::nullopt; }
        probeWindow_.Record(result.Success());
        if (Healthy()) {
            if (probeWindow_.FailureCount() >= config_.failureThreshold) {
                return ToUnhealthy(HealthSource::ActiveProbe, now);
            } else {
                return std::nullopt;
            }
        } else if (probeWindow_.Full() && probeWindow_.FailureCount() == 0) {
            if (now >= recoverAfter_) {
                return ToHealthy(now);
            } else {
                UC_INFO(
                    "[{}] Store health breaker({}) has a healthy probe window; waiting for "
                    "cooldown, remaining_ms={}.",
                    module_, name_, CooldownRemaining(now).count());
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    }

    // return not null if transitioned to unhealthy
    std::optional<HealthTransition> RecordIo(uint64_t generation, const Status& result,
                                             Time now = Clock::now())
    {
        if (ShouldRecordResult(result) && Healthy() && generation == generation_.load()) {
            passiveWindow_.Record(result.Success());
        } else {
            return std::nullopt;
        }
        if (result.Failure() && PassiveThresholdExceeded()) {
            std::lock_guard<std::mutex> lock(mutex_);
            // Another I/O or probe may have changed the state since the lock-free check.
            if (Healthy() && generation == generation_.load()) {
                return ToUnhealthy(HealthSource::PassiveIo, now);
            } else {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    }

    // Advances the passive window baseline; callers invoke it periodically.
    void SamplePassiveWindow(Time now = Clock::now())
    {
        std::lock_guard<std::mutex> lock(mutex_);
        passiveWindow_.UpdateWindowStart(now);
        const auto stats = passiveWindow_.GetStats();
        if (Healthy() && stats.failures > 0) {
            UC_INFO(
                "[{}] Store passive health window({}): window_s={}, samples={}, failures={}, "
                "failure_threshold={}.",
                module_, name_, config_.passiveWindow.count(), stats.successes + stats.failures,
                stats.failures, config_.passiveFailureThreshold);
        }
    }

    HealthSnapshot Snapshot(Time now = Clock::now()) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return {Healthy(),
                generation_.load(),
                probeWindow_.SampleCount(),
                probeWindow_.FailureCount(),
                cooldown_,
                CooldownRemaining(now),
                passiveWindow_.GetStats()};
    }

private:
    bool PassiveThresholdExceeded() const
    {
        return passiveWindow_.GetStats().failures >= config_.passiveFailureThreshold;
    }

    // The members below require mutex_.
    std::chrono::milliseconds CooldownRemaining(Time now) const
    {
        return !Healthy() && now < recoverAfter_
                   ? std::chrono::ceil<std::chrono::milliseconds>(recoverAfter_ - now)
                   : std::chrono::milliseconds{0};
    }

    HealthTransition ToUnhealthy(HealthSource source, Time now)
    {
        if (recoveredAt_ && now - *recoveredAt_ < config_.stableResetAfter) {
            const auto next = static_cast<long double>(cooldown_.count()) * config_.backoffFactor;
            cooldown_ = std::chrono::milliseconds{static_cast<int64_t>(
                std::min(next, static_cast<long double>(config_.maxCooldown.count())))};
        } else {
            cooldown_ = config_.initialCooldown;
        }
        recoverAfter_ = now + cooldown_;
        healthy_.store(false);
        const auto generation = ++generation_;
        if (source == HealthSource::PassiveIo) {
            const auto stats = passiveWindow_.GetStats();
            UC_WARN_UNLIMITED(
                "[{}] Store health breaker({}) transitioned to UNHEALTHY, source=passive_io, "
                "cooldown_ms={}, generation={}, window_s={}, successes={}, failures={}, "
                "failure_threshold={}.",
                module_, name_, cooldown_.count(), generation, config_.passiveWindow.count(),
                stats.successes, stats.failures, config_.passiveFailureThreshold);
        } else {
            std::vector<const char*> results;
            for (const auto healthy : probeWindow_.Results()) {
                results.push_back(healthy ? "ok" : "fail");
            }
            UC_WARN_UNLIMITED(
                "[{}] Store health breaker({}) transitioned to UNHEALTHY, source=active_probe, "
                "cooldown_ms={}, generation={}, window_size={}, failures={}, "
                "failure_threshold={}, results=[{}].",
                module_, name_, cooldown_.count(), generation, config_.healthWindowSize,
                probeWindow_.FailureCount(), config_.failureThreshold, fmt::join(results, ","));
        }
        probeWindow_.Clear();
        return {false, source, cooldown_, generation};
    }

    HealthTransition ToHealthy(Time now)
    {
        const auto generation = ++generation_;
        passiveWindow_.Reset();
        recoveredAt_ = now;
        healthy_.store(true);
        UC_WARN_UNLIMITED(
            "[{}] Store health breaker({}) transitioned to HEALTHY, source=active_probe, "
            "cooldown_ms={}, generation={}.",
            module_, name_, cooldown_.count(), generation);
        return {true, HealthSource::ActiveProbe, cooldown_, generation};
    }

    const std::string module_;
    const std::string name_;
    const StoreHealthConfig config_;
    std::atomic<bool> healthy_{true};
    std::atomic<uint64_t> generation_{0};
    PassiveIoWindow passiveWindow_;
    mutable std::mutex mutex_;
    HealthWindow probeWindow_;
    std::chrono::milliseconds cooldown_{0};
    Time recoverAfter_{};
    std::optional<Time> recoveredAt_;
};

}  // namespace UC::Detail

#endif
