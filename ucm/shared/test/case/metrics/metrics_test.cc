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
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <unistd.h>
#include "metrics_api.h"

using namespace UC::Metrics;

class UCMetricsUT : public testing::Test {
protected:
    void SetUp() override
    {
        try {
            UC::Metrics::SetUp(1000000);
            CreateStats("stats1", "counter");
            CreateStats("stats2", "gauge");
            CreateStats("stats3", "histogram");
        } catch (const std::exception& e) {
            throw;
        }
    }
};

TEST_F(UCMetricsUT, UpdateSingleStatAndGet)
{
    // Update stat
    UpdateStats("stats1", 1.0);
    UpdateStats("stats1", 3.0);

    // GetStatsAndClear
    auto stats = GetAllStatsAndClear();
    const auto& counter_iter = std::get<0>(stats);
    ASSERT_NE(counter_iter.find("stats1"), counter_iter.end());
    ASSERT_EQ(counter_iter.at("stats1"), 4.0);

    // Update Multiple Stats
    UpdateStats("stats1", 1.0);
    UpdateStats("stats2", 7.0);
    UpdateStats("stats2", 9.0);
    UpdateStats("stats3", 8.8);

    stats = GetAllStatsAndClear();
    const auto& counter_iter1 = std::get<0>(stats);
    const auto& gauge_iter = std::get<1>(stats);
    const auto& histogram_iter = std::get<2>(stats);

    ASSERT_NE(counter_iter1.find("stats1"), counter_iter1.end());
    ASSERT_EQ(counter_iter1.at("stats1"), 1.0);

    ASSERT_NE(gauge_iter.find("stats2"), gauge_iter.end());
    ASSERT_EQ(gauge_iter.at("stats2"), 9.0);

    ASSERT_NE(histogram_iter.find("stats3"), histogram_iter.end());
    ASSERT_EQ(histogram_iter.at("stats3")[0], 8.8);
}

TEST_F(UCMetricsUT, UpdateMultipleStatsAndGet)
{
    UpdateStats({
        {"stats1", 3.0},
        {"stats2", 4.0},
        {"stats3", 5.0},
    });
    UpdateStats({
        {"stats1", 3.3},
        {"stats2", 4.4},
        {"stats3", 5.5},
    });

    auto stats = GetAllStatsAndClear();
    const auto& counter_iter = std::get<0>(stats);
    const auto& gauge_iter = std::get<1>(stats);
    const auto& histogram_iter = std::get<2>(stats);

    ASSERT_NE(counter_iter.find("stats1"), counter_iter.end());
    ASSERT_EQ(counter_iter.at("stats1"), 6.3);

    ASSERT_NE(gauge_iter.find("stats2"), gauge_iter.end());
    ASSERT_EQ(gauge_iter.at("stats2"), 4.4);

    ASSERT_NE(histogram_iter.find("stats3"), histogram_iter.end());
    ASSERT_EQ(histogram_iter.at("stats3")[1], 5.5);
}