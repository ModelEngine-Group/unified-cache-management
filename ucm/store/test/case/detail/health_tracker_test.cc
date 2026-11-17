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
#include "detail/health_tracker.h"
#include <chrono>
#include <gtest/gtest.h>

namespace UC::Test {

using Detail::HealthSource;
using Detail::HealthTracker;
using Detail::StoreHealthConfig;
using namespace std::chrono_literals;

namespace {

StoreHealthConfig TestConfig()
{
    StoreHealthConfig config;
    config.healthWindowSize = 3;
    config.failureThreshold = 2;
    config.passiveWindow = 60s;
    config.passiveFailureThreshold = 2;
    config.initialCooldown = 100ms;
    config.maxCooldown = 1000ms;
    config.backoffFactor = 2.0;
    config.stableResetAfter = 10s;
    return config;
}

void ProbeOk(HealthTracker& tracker, HealthTracker::Time now, size_t times)
{
    for (size_t i = 0; i < times; ++i) {
        tracker.RecordProbe(tracker.Generation(), Status::OK(), now);
    }
}

void Trip(HealthTracker& tracker, HealthTracker::Time now)
{
    tracker.RecordProbe(tracker.Generation(), Status::Error(), now);
    EXPECT_TRUE(tracker.RecordProbe(tracker.Generation(), Status::Error(), now).has_value());
    EXPECT_FALSE(tracker.Healthy());
}

}  // namespace

TEST(UCHealthTrackerTest, ProbeFailuresTripAndStaleProbeIsDropped)
{
    const auto t0 = HealthTracker::Clock::now();
    HealthTracker tracker{"Test", "backend-0", TestConfig()};
    const auto stale = tracker.Generation();

    EXPECT_FALSE(tracker.RecordProbe(tracker.Generation(), Status::Error(), t0).has_value());
    const auto transition = tracker.RecordProbe(tracker.Generation(), Status::Timeout(), t0);
    ASSERT_TRUE(transition.has_value());
    EXPECT_FALSE(transition->healthy);
    EXPECT_EQ(transition->source, HealthSource::ActiveProbe);
    EXPECT_EQ(transition->cooldown, 100ms);
    EXPECT_EQ(transition->generation, 1);

    EXPECT_FALSE(tracker.RecordProbe(stale, Status::OK(), t0).has_value());
    const auto snapshot = tracker.Snapshot(t0);
    EXPECT_EQ(snapshot.probeSamples, 0);
    EXPECT_EQ(snapshot.cooldownRemaining, 100ms);
}

TEST(UCHealthTrackerTest, RecoversOnlyAfterCooldownWithFullSuccessWindow)
{
    const auto t0 = HealthTracker::Clock::now();
    HealthTracker tracker{"Test", "backend-0", TestConfig()};
    Trip(tracker, t0);

    ProbeOk(tracker, t0 + 50ms, 3);
    EXPECT_FALSE(tracker.Healthy());
    EXPECT_EQ(tracker.Snapshot(t0 + 50ms).probeSamples, 3);

    const auto transition = tracker.RecordProbe(tracker.Generation(), Status::OK(), t0 + 100ms);
    ASSERT_TRUE(transition.has_value());
    EXPECT_TRUE(transition->healthy);
    EXPECT_EQ(transition->generation, 2);
    EXPECT_TRUE(tracker.Healthy());
}

TEST(UCHealthTrackerTest, FailureInsideRecoveryWindowDelaysRecovery)
{
    const auto t0 = HealthTracker::Clock::now();
    HealthTracker tracker{"Test", "backend-0", TestConfig()};
    Trip(tracker, t0);
    const auto after = t0 + 200ms;

    ProbeOk(tracker, after, 2);
    tracker.RecordProbe(tracker.Generation(), Status::Error(), after);
    ProbeOk(tracker, after, 2);
    EXPECT_FALSE(tracker.Healthy());
    ProbeOk(tracker, after, 1);
    EXPECT_TRUE(tracker.Healthy());
}

TEST(UCHealthTrackerTest, CooldownBacksOffUntilStableThenResets)
{
    auto now = HealthTracker::Clock::now();
    HealthTracker tracker{"Test", "backend-0", TestConfig()};
    const std::chrono::milliseconds expected[] = {100ms, 200ms, 400ms, 800ms, 1000ms, 1000ms};
    for (const auto cooldown : expected) {
        Trip(tracker, now);
        EXPECT_EQ(tracker.Snapshot(now).cooldown, cooldown);
        now += cooldown;
        ProbeOk(tracker, now, 3);
        ASSERT_TRUE(tracker.Healthy());
    }

    now += 10s;
    Trip(tracker, now);
    EXPECT_EQ(tracker.Snapshot(now).cooldown, 100ms);
}

TEST(UCHealthTrackerTest, PassiveFailuresTripImmediatelyAtThreshold)
{
    const auto t0 = HealthTracker::Clock::now();
    HealthTracker tracker{"Test", "backend-0", TestConfig()};

    EXPECT_FALSE(tracker.RecordIo(tracker.Generation(), Status::OsApiError(), t0).has_value());
    EXPECT_FALSE(tracker.RecordIo(tracker.Generation(), Status::OK(), t0).has_value());
    const auto transition = tracker.RecordIo(tracker.Generation(), Status::Timeout(), t0);
    ASSERT_TRUE(transition.has_value());
    EXPECT_FALSE(transition->healthy);
    EXPECT_EQ(transition->source, HealthSource::PassiveIo);
    EXPECT_FALSE(tracker.Healthy());
}

TEST(UCHealthTrackerTest, PassiveIgnoresRequestErrorsAndStaleGenerations)
{
    const auto t0 = HealthTracker::Clock::now();
    HealthTracker tracker{"Test", "backend-0", TestConfig()};
    for (const auto& status :
         {Status::NotFound(), Status::InvalidParam(), Status::DuplicateKey(), Status::Unsupported(),
          Status::StoreUnhealthy(), Status::TimeoutInQueue()}) {
        EXPECT_FALSE(tracker.RecordIo(tracker.Generation(), status, t0).has_value());
    }
    EXPECT_EQ(tracker.Snapshot(t0).passive.failures, 0);

    const auto beforeTrip = tracker.Generation();
    Trip(tracker, t0);
    ProbeOk(tracker, t0 + 100ms, 3);
    ASSERT_TRUE(tracker.Healthy());

    tracker.RecordIo(beforeTrip, Status::OsApiError(), t0 + 100ms);
    tracker.RecordIo(beforeTrip, Status::OsApiError(), t0 + 100ms);
    EXPECT_TRUE(tracker.Healthy());
    EXPECT_EQ(tracker.Snapshot(t0 + 100ms).passive.failures, 0);
}

TEST(UCHealthTrackerTest, PassiveWindowEvictsOldFailures)
{
    const auto t0 = HealthTracker::Clock::now();
    const auto window = TestConfig().passiveWindow;
    HealthTracker tracker{"Test", "backend-0", TestConfig()};

    tracker.SamplePassiveWindow(t0);
    tracker.RecordIo(tracker.Generation(), Status::OsApiError(), t0);
    tracker.SamplePassiveWindow(t0 + window);
    tracker.SamplePassiveWindow(t0 + 2 * window);
    EXPECT_EQ(tracker.Snapshot(t0 + 2 * window).passive.failures, 0);

    EXPECT_FALSE(
        tracker.RecordIo(tracker.Generation(), Status::OsApiError(), t0 + 2 * window).has_value());
    EXPECT_TRUE(tracker.Healthy());
}

}  // namespace UC::Test
