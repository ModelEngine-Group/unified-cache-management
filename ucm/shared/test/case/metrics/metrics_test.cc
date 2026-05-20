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
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <vector>
#include "metrics_api.h"

using namespace UC::Metrics;

namespace {
uint64_t HistogramCount(const HistogramStat& histogram)
{
    return std::accumulate(histogram.bucketCounts.begin(), histogram.bucketCounts.end(),
                           uint64_t{0});
}
}  // namespace

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
    ASSERT_EQ(HistogramCount(histogram_iter.at("stats3")), 1);
    ASSERT_EQ(histogram_iter.at("stats3").sum, 8.8);
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
    ASSERT_EQ(HistogramCount(histogram_iter.at("stats3")), 2);
    ASSERT_EQ(histogram_iter.at("stats3").sum, 10.5);
}

TEST_F(UCMetricsUT, UpdateCachedMetricAndGet)
{
    GetAllStatsAndClear();

    UpdateStats(NAME_TO_METRIC_ID("stats1"), 2.0);
    UpdateStats(NAME_TO_METRIC_ID("stats1"), 3.0);
    UpdateStats(NAME_TO_METRIC_ID("stats2"), 5.0);
    UpdateStats(NAME_TO_METRIC_ID("stats3"), 7.0);

    auto stats = GetAllStatsAndClear();
    const auto& counter_iter = std::get<0>(stats);
    const auto& gauge_iter = std::get<1>(stats);
    const auto& histogram_iter = std::get<2>(stats);

    ASSERT_NE(counter_iter.find("stats1"), counter_iter.end());
    ASSERT_EQ(counter_iter.at("stats1"), 5.0);

    ASSERT_NE(gauge_iter.find("stats2"), gauge_iter.end());
    ASSERT_EQ(gauge_iter.at("stats2"), 5.0);

    ASSERT_NE(histogram_iter.find("stats3"), histogram_iter.end());
    ASSERT_EQ(HistogramCount(histogram_iter.at("stats3")), 1);
    ASSERT_EQ(histogram_iter.at("stats3").sum, 7.0);
}

TEST_F(UCMetricsUT, CachedMetricRetriesAfterLateRegistration)
{
    GetAllStatsAndClear();

    CachedMetric metric{"late_epoch_stats"};
    UpdateStats(metric, 1.0);
    UpdateStats(metric, 3.0);

    auto missStats = GetAllStatsAndClear();
    const auto& miss_counter_iter = std::get<0>(missStats);

    ASSERT_EQ(miss_counter_iter.find("late_epoch_stats"), miss_counter_iter.end());
    ASSERT_EQ(metric.id.load(std::memory_order_acquire), INVALID_METRIC_ID);
    ASSERT_NE(metric.seenEpoch.load(std::memory_order_acquire), 0);

    CreateStats("late_epoch_stats", "counter");
    ASSERT_EQ(metric.id.load(std::memory_order_acquire), INVALID_METRIC_ID);

    UpdateStats(metric, 2.0);
    ASSERT_NE(metric.id.load(std::memory_order_acquire), INVALID_METRIC_ID);

    auto stats = GetAllStatsAndClear();
    const auto& counter_iter = std::get<0>(stats);

    ASSERT_NE(counter_iter.find("late_epoch_stats"), counter_iter.end());
    ASSERT_EQ(counter_iter.at("late_epoch_stats"), 2.0);
}

TEST_F(UCMetricsUT, CachedMetricDoesNotMissResolveAgainWithoutRegistration)
{
    GetAllStatsAndClear();

    CachedMetric metric{"missing_epoch_stats"};
    UpdateStats(metric, 1.0);
    auto seenEpoch = metric.seenEpoch.load(std::memory_order_acquire);
    UpdateStats(metric, 2.0);

    auto stats = GetAllStatsAndClear();
    const auto& counter_iter = std::get<0>(stats);

    ASSERT_EQ(counter_iter.find("missing_epoch_stats"), counter_iter.end());
    ASSERT_EQ(metric.id.load(std::memory_order_acquire), INVALID_METRIC_ID);
    ASSERT_EQ(metric.seenEpoch.load(std::memory_order_acquire), seenEpoch);

    CreateStats("missing_epoch_stats", "counter");
    UpdateStats(metric, 2.0);

    stats = GetAllStatsAndClear();
    const auto& counter_iter_after_register = std::get<0>(stats);

    ASSERT_NE(counter_iter_after_register.find("missing_epoch_stats"),
              counter_iter_after_register.end());
    ASSERT_EQ(counter_iter_after_register.at("missing_epoch_stats"), 2.0);
}

TEST_F(UCMetricsUT, HistogramAggregatesConfiguredBuckets)
{
    GetAllStatsAndClear();

    CreateStats("bucket_histogram_stats", "histogram", {1.0, 5.0});
    GetAllStatsAndClear();

    UpdateStats(NAME_TO_METRIC_ID("bucket_histogram_stats"), 0.5);
    UpdateStats(NAME_TO_METRIC_ID("bucket_histogram_stats"), 3.0);
    UpdateStats(NAME_TO_METRIC_ID("bucket_histogram_stats"), 10.0);

    auto stats = GetAllStatsAndClear();
    const auto& histogram_iter = std::get<2>(stats);

    ASSERT_NE(histogram_iter.find("bucket_histogram_stats"), histogram_iter.end());
    const auto& histogram = histogram_iter.at("bucket_histogram_stats");
    const auto& bucketCounts = histogram.bucketCounts;
    ASSERT_EQ(bucketCounts.size(), 3);
    ASSERT_EQ(bucketCounts[0], 1);
    ASSERT_EQ(bucketCounts[1], 1);
    ASSERT_EQ(bucketCounts[2], 1);
    ASSERT_EQ(histogram.sum, 13.5);

    stats = GetAllStatsAndClear();
    const auto& empty_histogram_iter = std::get<2>(stats);
    ASSERT_EQ(empty_histogram_iter.find("bucket_histogram_stats"), empty_histogram_iter.end());
}

TEST_F(UCMetricsUT, ConcurrentUpdateAndCollect)
{
    GetAllStatsAndClear();

    constexpr int numThreads = 8;
    constexpr int updatesPerThread = 2000;
    constexpr double expectedUpdates = numThreads * updatesPerThread;
    std::atomic<int> ready{0};
    std::atomic<int> finished{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            for (int j = 0; j < updatesPerThread; ++j) {
                UpdateStats("stats1", 1.0);
                UpdateStats("stats3", static_cast<double>(j));
            }
            finished.fetch_add(1, std::memory_order_release);
        });
    }

    while (ready.load(std::memory_order_acquire) < numThreads) { std::this_thread::yield(); }
    start.store(true, std::memory_order_release);

    double totalCounter = 0.0;
    size_t totalHistogram = 0;
    auto collect = [&] {
        auto stats = GetAllStatsAndClear();
        const auto& counters = std::get<0>(stats);
        const auto& histograms = std::get<2>(stats);
        auto counter = counters.find("stats1");
        if (counter != counters.end()) { totalCounter += counter->second; }
        auto histogram = histograms.find("stats3");
        if (histogram != histograms.end()) { totalHistogram += HistogramCount(histogram->second); }
    };

    while (finished.load(std::memory_order_acquire) < numThreads) {
        collect();
        std::this_thread::yield();
    }

    for (auto& thread : threads) { thread.join(); }
    collect();

    ASSERT_EQ(totalCounter, expectedUpdates);
    ASSERT_EQ(totalHistogram, static_cast<size_t>(expectedUpdates));
}
