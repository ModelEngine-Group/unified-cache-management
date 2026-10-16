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
#include <array>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include "metrics_api.h"

using namespace UC::Metrics;

TEST(UCMetricsStress, DISABLED_TenWritersOneCollector)
{
    constexpr int numWriters = 10;
    constexpr auto duration = std::chrono::seconds(3);
    const std::string metricName = "concurrent_stress_histogram";
    UC::Metrics::SetUp();
    CreateStats(metricName, "histogram", {0.5, 1.0, 2.0});
    GetAllStatsAndClear();
    GetAllStatsAndClear();

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    std::array<uint64_t, numWriters> writes{};
    std::vector<std::thread> writers;
    for (int i = 0; i < numWriters; ++i) {
        writers.emplace_back([&, i] {
            CachedMetric metric{metricName};
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            uint64_t count = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                for (int j = 0; j < 1024; ++j) {
                    UpdateStats(metric, 1.0);
                    ++count;
                }
            }
            writes[i] = count;
        });
    }

    uint64_t collections = 0;
    uint64_t totalCount = 0;
    double totalSum = 0.0;
    bool invalidSnapshot = false;
    std::chrono::steady_clock::time_point started;
    auto collect = [&] {
        auto stats = GetAllStatsAndClear();
        ++collections;
        const auto& histograms = std::get<2>(stats);
        auto it = histograms.find(metricName);
        if (it == histograms.end()) { return; }
        const auto& histogram = it->second;
        auto count = std::accumulate(histogram.bucketCounts.begin(), histogram.bucketCounts.end(),
                                     uint64_t{0});
        totalCount += count;
        totalSum += histogram.sum;
        bool valid = histogram.bucketCounts.size() == 4 && histogram.bucketCounts[1] == count &&
                     histogram.sum == static_cast<double>(count);
        if (!valid && !invalidSnapshot) {
            invalidSnapshot = true;
            double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::cout << "First invalid snapshot: elapsed_seconds=" << elapsed
                      << " collections=" << collections << " count=" << count
                      << " sum=" << histogram.sum << " buckets=" << histogram.bucketCounts.size()
                      << std::endl;
        }
    };

    std::thread collector([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
        while (!stop.load(std::memory_order_relaxed)) {
            collect();
            if (std::chrono::steady_clock::now() - started >= duration) {
                stop.store(true, std::memory_order_relaxed);
            }
        }
    });

    while (ready.load(std::memory_order_acquire) != numWriters + 1) { std::this_thread::yield(); }
    started = std::chrono::steady_clock::now();
    std::cout << "Stress started: writers=10 collectors=1 limit_seconds=3" << std::endl;
    start.store(true, std::memory_order_release);
    collector.join();
    for (auto& writer : writers) { writer.join(); }
    collect();
    collect();

    auto expectedCount = std::accumulate(writes.begin(), writes.end(), uint64_t{0});
    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "Stress finished: elapsed_seconds=" << elapsed << " writes=" << expectedCount
              << " collections=" << collections << " collected_count=" << totalCount
              << " collected_sum=" << totalSum << std::endl;
    EXPECT_FALSE(invalidSnapshot);
    EXPECT_GT(collections, 2);
    for (auto count : writes) { EXPECT_GT(count, 0); }
    EXPECT_EQ(totalCount, expectedCount);
    EXPECT_EQ(totalSum, static_cast<double>(expectedCount));
}
