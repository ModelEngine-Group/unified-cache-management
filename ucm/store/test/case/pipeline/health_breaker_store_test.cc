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
#include "health_breaker_store.h"
#include <array>
#include <condition_variable>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mutex>
#include <vector>
#include "detail/mock_store.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "store_health_config.h"

namespace UC::Test {

using PipelineStore::HealthBreakerConfig;
using PipelineStore::HealthBreakerStore;
using testing::Invoke;
using testing::Return;
using testing::StrictMock;

TEST(UCHealthBreakerStoreTest, StoreV1ProvidesHealthyDefault)
{
    Detail::MockStore store;
    auto& base = static_cast<StoreV1&>(store);
    EXPECT_TRUE(base.StoreV1::CheckHealth().Success());
}

namespace {

HealthBreakerConfig TestConfig()
{
    HealthBreakerConfig config;
    config.healthWindowSize = 5;
    config.failureThreshold = 3;
    config.healthCheckInterval = std::chrono::hours(1);
    return config;
}

void Trip(HealthBreakerStore& breaker, Detail::MockStore& store)
{
    EXPECT_CALL(store, CheckHealth()).Times(3).WillRepeatedly(Return(Status::Error()));
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    EXPECT_FALSE(breaker.Enabled());
}

}  // namespace

TEST(UCHealthBreakerStoreTest, TripsEarlyAndRecoversAfterFullSuccessWindow)
{
    StrictMock<Detail::MockStore> store;
    HealthBreakerStore breaker(&store, "cache-0", TestConfig());

    Trip(breaker, store);
    EXPECT_EQ(breaker.FailureCount(), 3);
    EXPECT_EQ(breaker.SampleCount(), 3);

    EXPECT_CALL(store, CheckHealth()).Times(5).WillRepeatedly(Return(Status::OK()));
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(breaker.CheckHealth().Success());
        EXPECT_FALSE(breaker.Enabled());
    }
    EXPECT_TRUE(breaker.CheckHealth().Success());
    EXPECT_TRUE(breaker.Enabled());
    EXPECT_EQ(breaker.FailureCount(), 0);
    EXPECT_EQ(breaker.SampleCount(), 5);
}

TEST(UCHealthBreakerStoreTest, LogsWindowOnEveryStateTransition)
{
    StrictMock<Detail::MockStore> store;
    auto config = TestConfig();
    config.healthWindowSize = 3;
    config.failureThreshold = 2;
    HealthBreakerStore breaker(&store, "cache-0", config);

    testing::internal::CaptureStdout();
    EXPECT_CALL(store, CheckHealth())
        .WillOnce(Return(Status::Error()))
        .WillOnce(Return(Status::Error()))
        .WillRepeatedly(Return(Status::OK()));
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    EXPECT_TRUE(breaker.CheckHealth().Success());
    EXPECT_TRUE(breaker.CheckHealth().Success());
    EXPECT_TRUE(breaker.CheckHealth().Success());
    UC::Logger::Flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto output = testing::internal::GetCapturedStdout();

    EXPECT_THAT(output, testing::HasSubstr("transitioned to UNHEALTHY"));
    EXPECT_THAT(output, testing::HasSubstr("window=[failure, failure]"));
    EXPECT_THAT(output, testing::HasSubstr("transitioned to HEALTHY"));
    EXPECT_THAT(output, testing::HasSubstr("window=[success, success, success]"));
}

TEST(UCHealthBreakerStoreTest, SlidingWindowEvictsOldFailure)
{
    StrictMock<Detail::MockStore> store;
    auto config = TestConfig();
    config.failureThreshold = 5;
    HealthBreakerStore breaker(&store, "cache-0", config);

    EXPECT_CALL(store, CheckHealth())
        .WillOnce(Return(Status::Error()))
        .WillRepeatedly(Return(Status::OK()));
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    for (size_t i = 0; i < 5; ++i) { EXPECT_TRUE(breaker.CheckHealth().Success()); }

    EXPECT_EQ(breaker.FailureCount(), 0);
    EXPECT_EQ(breaker.SampleCount(), 5);
    EXPECT_TRUE(breaker.Enabled());
}

TEST(UCHealthBreakerStoreTest, RecordsPosixProbeResultsAndEffectiveBreakerState)
{
    UC::Metrics::SetUp();
    UC::Metrics::CreateStats("posix_healthy_count_total", "counter");
    UC::Metrics::CreateStats("posix_unhealthy_count_total", "counter");
    UC::Metrics::CreateStats("posix_store_health", "gauge");
    UC::Metrics::GetAllStatsAndClear();

    StrictMock<Detail::MockStore> store;
    auto config = TestConfig();
    config.healthWindowSize = 1;
    config.failureThreshold = 1;
    HealthBreakerStore breaker(&store, "pipeline/0:PosixStore", config);

    EXPECT_CALL(store, CheckHealth())
        .WillOnce(Return(Status::Error()))
        .WillOnce(Return(Status::OK()));
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    auto failedStats = UC::Metrics::GetAllStatsAndClear();
    auto& failedCounters = std::get<0>(failedStats);
    auto& failedGauges = std::get<1>(failedStats);
    EXPECT_EQ(failedCounters.at("posix_unhealthy_count_total"), 1);
    EXPECT_EQ(failedCounters.count("posix_healthy_count_total"), 0);
    ASSERT_EQ(failedGauges.count("posix_store_health"), 1);
    EXPECT_EQ(failedGauges.at("posix_store_health"), 0);

    EXPECT_TRUE(breaker.CheckHealth().Success());
    auto healthyStats = UC::Metrics::GetAllStatsAndClear();
    auto& healthyCounters = std::get<0>(healthyStats);
    auto& healthyGauges = std::get<1>(healthyStats);
    EXPECT_EQ(healthyCounters.at("posix_healthy_count_total"), 1);
    EXPECT_EQ(healthyCounters.count("posix_unhealthy_count_total"), 0);
    EXPECT_EQ(healthyGauges.at("posix_store_health"), 1);
}

TEST(UCHealthBreakerStoreTest, AppliesTimeoutBeforeStart)
{
    StrictMock<Detail::MockStore> store;
    auto config = TestConfig();
    config.healthCheckTimeout = std::chrono::milliseconds(10);
    config.healthWindowSize = 1;
    config.failureThreshold = 1;
    HealthBreakerStore breaker(&store, "cache-0", config);

    EXPECT_CALL(store, CheckHealth()).WillOnce(Invoke([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return Status::OK();
    }));

    EXPECT_EQ(breaker.CheckHealth(), Status::Timeout());
    EXPECT_FALSE(breaker.Enabled());
}

TEST(UCHealthBreakerStoreTest, TimesOutSlowStoreCheck)
{
    StrictMock<Detail::MockStore> store;
    auto config = TestConfig();
    config.healthCheckTimeout = std::chrono::milliseconds(10);
    config.healthWindowSize = 1;
    config.failureThreshold = 1;
    HealthBreakerStore breaker(&store, "cache-0", config);

    EXPECT_CALL(store, CheckHealth()).WillOnce(Invoke([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return Status::OK();
    }));
    ASSERT_TRUE(breaker.Start().Success());

    EXPECT_EQ(breaker.CheckHealth(), Status::Timeout());
    EXPECT_FALSE(breaker.Enabled());
    breaker.Stop();
}

TEST(UCHealthBreakerStoreTest, ProbeIntervalIncludesHealthCheckTime)
{
    StrictMock<Detail::MockStore> store;
    auto config = TestConfig();
    config.healthCheckInterval = std::chrono::milliseconds(100);
    config.healthCheckTimeout = std::chrono::milliseconds(90);
    HealthBreakerStore breaker(&store, "cache-0", config);
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::chrono::steady_clock::time_point> starts;

    EXPECT_CALL(store, CheckHealth()).WillRepeatedly(Invoke([&] {
        {
            std::lock_guard<std::mutex> lock{mutex};
            starts.push_back(std::chrono::steady_clock::now());
        }
        cv.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        return Status::OK();
    }));
    ASSERT_TRUE(breaker.Start().Success());
    {
        std::unique_lock<std::mutex> lock{mutex};
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(1), [&] { return starts.size() >= 3; }));
    }
    breaker.Stop();

    ASSERT_GE(starts.size(), 3);
    EXPECT_LT(starts[2] - starts[0], std::chrono::milliseconds(300));
}

TEST(UCHealthBreakerStoreTest, UnhealthyOperationMatrix)
{
    StrictMock<Detail::MockStore> store;
    HealthBreakerStore breaker(&store, "posix-0", TestConfig());
    Trip(breaker, store);

    std::array<UC::Detail::BlockId, 2> blocks{};
    auto lookup = breaker.Lookup(blocks.data(), blocks.size());
    ASSERT_TRUE(lookup);
    EXPECT_EQ(lookup.Value(), (std::vector<uint8_t>{0, 0}));

    auto prefix = breaker.LookupOnPrefix(blocks.data(), blocks.size());
    ASSERT_TRUE(prefix);
    EXPECT_EQ(prefix.Value(), -1);

    breaker.Prefetch(blocks.data(), blocks.size());

    EXPECT_EQ(breaker.Load({}).Error(), Status::StoreUnhealthy());
    EXPECT_EQ(breaker.Dump({}).Error(), Status::StoreUnhealthy());

    EXPECT_CALL(store, Check(17)).WillOnce(Return(true));
    EXPECT_TRUE(breaker.Check(17).Value());
    EXPECT_CALL(store, Wait(18)).WillOnce(Return(Status::OK()));
    EXPECT_TRUE(breaker.Wait(18).Success());
}

TEST(UCHealthBreakerStoreTest, HealthyOperationsPassThroughWithoutChangingWindow)
{
    StrictMock<Detail::MockStore> store;
    HealthBreakerStore breaker(&store, "cache-0", TestConfig());
    std::array<UC::Detail::BlockId, 1> blocks{};

    EXPECT_CALL(store, Lookup(blocks.data(), blocks.size()))
        .WillOnce(Return(std::vector<uint8_t>{1}));
    EXPECT_EQ(breaker.Lookup(blocks.data(), blocks.size()).Value(), std::vector<uint8_t>{1});
    EXPECT_CALL(store, LookupOnPrefix(blocks.data(), blocks.size())).WillOnce(Return(0));
    EXPECT_EQ(breaker.LookupOnPrefix(blocks.data(), blocks.size()).Value(), 0);
    EXPECT_CALL(store, Prefetch(blocks.data(), blocks.size()));
    breaker.Prefetch(blocks.data(), blocks.size());
    EXPECT_CALL(store, Load(testing::_)).WillOnce(Return(21));
    EXPECT_EQ(breaker.Load({}).Value(), 21);
    EXPECT_CALL(store, Dump(testing::_)).WillOnce(Return(22));
    EXPECT_EQ(breaker.Dump({}).Value(), 22);
    EXPECT_TRUE(breaker.Enabled());
    EXPECT_EQ(breaker.SampleCount(), 0);
    EXPECT_EQ(breaker.FailureCount(), 0);
}

TEST(UCHealthBreakerStoreTest, ValidatesStoreHealthConfig)
{
    PipelineStore::StoreHealthConfig config;
    EXPECT_FALSE(config.enabled);
    EXPECT_TRUE(config.Validate().Success());

    config.enabled = true;
    EXPECT_TRUE(config.enabled);

    config.failureThreshold = config.healthWindowSize + 1;
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
    config.failureThreshold = config.healthWindowSize;
    config.healthCheckInterval = std::chrono::milliseconds(0);
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
    config.healthCheckInterval = std::chrono::milliseconds(10);
    config.healthCheckTimeout = config.healthCheckInterval;
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
}

}  // namespace UC::Test
