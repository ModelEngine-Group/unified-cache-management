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
 * */
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "routing/strategy.h"
#include "routing/strategy_registry.h"
#include "status/status.h"
#include "type/dictionary.h"
#include "type/request.h"

namespace UC::Proxy {
namespace {

std::unique_ptr<RoutingStrategy> CreateStrategy(const std::string& name)
{
    auto created = RoutingStrategyRegistry::Instance().Create(name);
    if (!created) { return nullptr; }
    return std::unique_ptr<RoutingStrategy>(*created);
}

Dictionary MakeConfig(const std::vector<std::int64_t>& thresholds)
{
    Dictionary config;
    config.SetNumbers("thresholds", thresholds);
    return config;
}

void ExpectSetupRejected(const std::string& name, const Dictionary& config, std::size_t branchCount)
{
    auto strategy = CreateStrategy(name);
    ASSERT_NE(strategy, nullptr);
    const Status result = strategy->Setup(config, branchCount);
    EXPECT_EQ(result, Status::InvalidParam());
}

TEST(RoutingStrategyRegistryTest, CreateUnknownNameFails)
{
    auto created = RoutingStrategyRegistry::Instance().Create("no-such-strategy");
    ASSERT_FALSE(created);
    EXPECT_EQ(created.Error(), Status::NotFound());
}

class LengthStrategyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        strategy_ = CreateStrategy("length");
        ASSERT_NE(strategy_, nullptr);
        Dictionary config;
        config.SetNumbers("thresholds", std::vector<std::int64_t>{1000, 100});
        ASSERT_TRUE(strategy_->Setup(config, 3).Success());
    }

    std::size_t Select(const std::optional<std::uint64_t> totalBlockCount)
    {
        RequestContext request;
        request.totalBlockCount = totalBlockCount;
        return strategy_->Select(request);
    }

    std::unique_ptr<RoutingStrategy> strategy_;
};

TEST_F(LengthStrategyTest, SelectRoutesByPredictedBlockCount)
{
    EXPECT_EQ(Select(1001), 0u);
    EXPECT_EQ(Select(1000), 1u);
    EXPECT_EQ(Select(101), 1u);
    EXPECT_EQ(Select(100), 2u);
    EXPECT_EQ(Select(0), 2u);
}

TEST_F(LengthStrategyTest, SelectFallsBackWithoutPrediction)
{
    EXPECT_EQ(Select(std::nullopt), 2u);
}

TEST_F(LengthStrategyTest, SetupRejectsMissingThresholds)
{
    ExpectSetupRejected("length", Dictionary{}, 3);
}

TEST_F(LengthStrategyTest, SetupRejectsEmptyThresholds)
{
    ExpectSetupRejected("length", MakeConfig({}), 3);
}

TEST_F(LengthStrategyTest, SetupRejectsNonNumericThresholds)
{
    Dictionary config;
    config.Set("thresholds", std::string("not-a-number-list"));
    ExpectSetupRejected("length", config, 3);
}

TEST_F(LengthStrategyTest, SetupRejectsNonDescendingThresholds)
{
    ExpectSetupRejected("length", MakeConfig({100, 1000}), 3);
    ExpectSetupRejected("length", MakeConfig({100, 100}), 3);
}

TEST_F(LengthStrategyTest, SetupRejectsNegativeThresholds)
{
    ExpectSetupRejected("length", MakeConfig({1000, -5}), 3);
}

TEST_F(LengthStrategyTest, SetupRejectsBranchCountMismatch)
{
    ExpectSetupRejected("length", MakeConfig({1000, 100}), 2);
    ExpectSetupRejected("length", MakeConfig({1000, 100}), 4);
}

class TtlStrategyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        strategy_ = CreateStrategy("ttl");
        ASSERT_NE(strategy_, nullptr);
        Dictionary config;
        config.SetNumbers("thresholds", std::vector<std::int64_t>{5000, 1000});
        ASSERT_TRUE(strategy_->Setup(config, 3).Success());
    }

    std::size_t Select(const std::optional<std::chrono::milliseconds> ttl)
    {
        RequestContext request;
        request.ttl = ttl;
        return strategy_->Select(request);
    }

    std::unique_ptr<RoutingStrategy> strategy_;
};

TEST_F(TtlStrategyTest, SelectRoutesByPredictedTtl)
{
    EXPECT_EQ(Select(std::chrono::milliseconds(6000)), 0u);
    EXPECT_EQ(Select(std::chrono::milliseconds(5000)), 1u);
    EXPECT_EQ(Select(std::chrono::milliseconds(1001)), 1u);
    EXPECT_EQ(Select(std::chrono::milliseconds(1000)), 2u);
    EXPECT_EQ(Select(std::chrono::milliseconds(0)), 2u);
}

TEST_F(TtlStrategyTest, SelectFallsBackWithoutPrediction) { EXPECT_EQ(Select(std::nullopt), 2u); }

TEST_F(TtlStrategyTest, SetupRejectsMissingThresholds)
{
    ExpectSetupRejected("ttl", Dictionary{}, 3);
}

TEST_F(TtlStrategyTest, SetupRejectsEmptyThresholds)
{
    ExpectSetupRejected("ttl", MakeConfig({}), 3);
}

TEST_F(TtlStrategyTest, SetupRejectsNonNumericThresholds)
{
    Dictionary config;
    config.Set("thresholds", std::string("not-a-number-list"));
    ExpectSetupRejected("ttl", config, 3);
}

TEST_F(TtlStrategyTest, SetupRejectsNonDescendingThresholds)
{
    ExpectSetupRejected("ttl", MakeConfig({100, 1000}), 3);
}

TEST_F(TtlStrategyTest, SetupRejectsBranchCountMismatch)
{
    ExpectSetupRejected("ttl", MakeConfig({5000, 1000}), 2);
    ExpectSetupRejected("ttl", MakeConfig({5000, 1000}), 4);
}

}  // namespace
}  // namespace UC::Proxy
