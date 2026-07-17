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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <array>
#include "breaker_store.h"
#include "detail/mock_store.h"
#include "store_health_config.h"

namespace UC::Test {

using PipelineStore::BreakerConfig;
using PipelineStore::BreakerStore;
using testing::Return;
using testing::StrictMock;

TEST(UCBreakerStoreTest, StoreV1ProvidesHealthyDefault)
{
    Detail::MockStore store;
    auto& base = static_cast<StoreV1&>(store);
    EXPECT_TRUE(base.StoreV1::CheckHealth().Success());
}

namespace {

BreakerConfig TestConfig()
{
    BreakerConfig config;
    config.healthWindowSize = 5;
    config.failureThreshold = 3;
    config.healthCheckInterval = std::chrono::hours(1);
    config.healthCheckTimeout = std::chrono::seconds(1);
    return config;
}

void Trip(BreakerStore& breaker, Detail::MockStore& store)
{
    EXPECT_CALL(store, CheckHealth()).Times(3).WillRepeatedly(Return(Status::Error()));
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    EXPECT_FALSE(breaker.Enabled());
}

}  // namespace

TEST(UCBreakerStoreTest, TripsEarlyAndRecoversAfterFullSuccessWindow)
{
    StrictMock<Detail::MockStore> store;
    BreakerStore breaker(&store, "cache-0", TestConfig());

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

TEST(UCBreakerStoreTest, SlidingWindowEvictsOldFailure)
{
    StrictMock<Detail::MockStore> store;
    auto config = TestConfig();
    config.failureThreshold = 5;
    BreakerStore breaker(&store, "cache-0", config);

    EXPECT_CALL(store, CheckHealth())
        .WillOnce(Return(Status::Error()))
        .WillRepeatedly(Return(Status::OK()));
    EXPECT_TRUE(breaker.CheckHealth().Failure());
    for (size_t i = 0; i < 5; ++i) { EXPECT_TRUE(breaker.CheckHealth().Success()); }

    EXPECT_EQ(breaker.FailureCount(), 0);
    EXPECT_EQ(breaker.SampleCount(), 5);
    EXPECT_TRUE(breaker.Enabled());
}

TEST(UCBreakerStoreTest, UnhealthyOperationMatrix)
{
    StrictMock<Detail::MockStore> store;
    BreakerStore breaker(&store, "posix-0", TestConfig());
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

TEST(UCBreakerStoreTest, HealthyOperationsPassThroughWithoutChangingWindow)
{
    StrictMock<Detail::MockStore> store;
    BreakerStore breaker(&store, "cache-0", TestConfig());
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

TEST(UCBreakerStoreTest, ValidatesStoreHealthConfig)
{
    PipelineStore::StoreHealthConfig config;
    EXPECT_TRUE(config.Validate().Success());

    config.failureThreshold = config.healthWindowSize + 1;
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
    config.failureThreshold = config.healthWindowSize;
    config.healthCheckTimeout = config.healthCheckInterval;
    EXPECT_EQ(config.Validate(), Status::InvalidParam());
}

}  // namespace UC::Test
