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
#include <gtest/gtest.h>
#include <unistd.h>
#include "stats_monitor_api.h"

using namespace UC::Metrics;

class UCStatsMonitorUT : public testing::Test {
protected:
    void SetUp() override
    {
        try {
            CreateStats("test_stats");
            CreateStats("stats1");
            CreateStats("stats2");
        } catch (const std::exception& e) {
            throw;
        }
    }
};

TEST_F(UCStatsMonitorUT, UpdateAndGetStats)
{
    std::string statsName = "test_stats";

    std::unordered_map<std::string, double> params;
    params["value1"] = 10.5;
    params["value2"] = 20.0;
    UpdateStats(statsName, params);

    StatsResult result = GetStats(statsName);
    ASSERT_EQ(result.data.size(), 2);
    ASSERT_EQ(result.data["value1"][0], 10.5);
    ASSERT_EQ(result.data["value2"][0], 20.0);

    params["value1"] = 30.5;
    UpdateStats(statsName, params);

    result = GetStats(statsName);
    ASSERT_EQ(result.data["value1"].size(), 2);
    ASSERT_EQ(result.data["value1"][1], 30.5);
    ASSERT_EQ(result.data["value2"].size(), 2);
    ASSERT_EQ(result.data["value2"][1], 20.0);

    StatsResult clearResult = GetStatsAndClear(statsName);
    ASSERT_EQ(clearResult.data.size(), 2);
    ASSERT_EQ(clearResult.data["value1"].size(), 2);
    ASSERT_EQ(clearResult.data["value2"].size(), 2);

    result = GetStats(statsName);
    EXPECT_TRUE(result.data.empty() ||
                (result.data["value1"].empty() && result.data["value2"].empty()));

    UpdateStats(statsName, params);
    ResetStats(statsName);
    result = GetStats(statsName);
    EXPECT_TRUE(result.data.empty() ||
                (result.data["value1"].empty() && result.data["value2"].empty()));
}

TEST_F(UCStatsMonitorUT, MultipleStatsAndResetAll)
{
    std::string stats1 = "stats1";
    std::string stats2 = "stats2";

    UpdateStats(stats1, {
                            {"a", 1.0},
                            {"b", 2.0}
    });
    UpdateStats(stats2, {
                            {"c", 3.0},
                            {"d", 4.0}
    });

    ASSERT_EQ(GetStats(stats1).data.size(), 2);
    ASSERT_EQ(GetStats(stats2).data.size(), 2);

    ResetAllStats();
    EXPECT_TRUE(GetStats(stats1).data.empty() ||
                (GetStats(stats1).data["a"].empty() && GetStats(stats1).data["b"].empty()));
    EXPECT_TRUE(GetStats(stats2).data.empty() ||
                (GetStats(stats2).data["c"].empty() && GetStats(stats2).data["d"].empty()));
}

TEST_F(UCStatsMonitorUT, MultipleStatsAndGetAll)
{
    std::string statsA = "stats1";
    std::string statsB = "stats2";

    UpdateStats(statsA, {
                            {"x", 100.0}
    });
    UpdateStats(statsB, {
                            {"y", 200.0}
    });

    StatsResult allStats = GetAllStatsAndClear();
    ASSERT_EQ(allStats.data.size(), 2);

    EXPECT_TRUE(!allStats.data.count(statsA) || !allStats.data.count(statsB));
}