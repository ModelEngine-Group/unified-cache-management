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
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <gtest/gtest.h>
#include "drampool_config.h"
#include "drampool_metrics.h"
#include "metrics_api.h"

namespace UC::DramPool {
namespace {

std::uint64_t HistogramCount(const Metrics::HistogramStat& histogram)
{
    std::uint64_t total = 0;
    for (const auto count : histogram.bucketCounts) { total += count; }
    return total;
}

// Every static name registered by SetupDrampoolMetrics(), grouped by metric
// type so the completeness probe below updates each name with matching
// semantics (metrics_design.md §3.2 and §8 case 1).
constexpr const char* kCounterNames[] = {
    kDumpRequestsTotal,
    kLoadRequestsTotal,
    kLookupRequestsTotal,
    kDumpNospaceFailuresTotal,
    kDumpFailedEntriesTotal,
    kLoadMissEntriesTotal,
    kLookupMissEntriesTotal,
    kTransferFailuresTotal,
    kResponseFailuresTotal,
    kSubmitFailuresTotal,
    kQueueRequestFullTotal,
    kQueueCompletionFullTotal,
    kQueueResponseBufferRetryTotal,
};

constexpr const char* kGaugeNames[] = {
    kMetadataEntryCount,
    kFlagPoolUsageRatio,
    kQueueRequestSize,
    kQueueCompletionInflight,
    kQueueCompletionSize,
};

constexpr const char* kHistogramNames[] = {
    kDumpPrepareDurationMs,
    kLoadPrepareDurationMs,
    kLookupScanDurationMs,
    kDumpTransferDurationMs,
    kLoadTransferDurationMs,
    kResponseRttMs,
    kMetadataStoreendDurationMs,
    kMetadataLoadendDurationMs,
    kMetadataEvictGcDurationMs,
    kQueueRequestEnqueueWaitMs,
    kDumpBatchTotalDurationMs,
    kLoadBatchTotalDurationMs,
    kLookupBatchTotalDurationMs,
};

DramPoolConfig g_savedConfig;

class UCDrampoolMetricsTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        // poolBlockSizes drives the dynamic per-slot-size gauge registration, so
        // it must be configured before the one-time SetupDrampoolMetrics() call.
        g_savedConfig = g_config;
        g_config.poolBlockSizes = {512, 4096};
        SetupDrampoolMetrics();
    }

    static void TearDownTestSuite() { g_config = std::move(g_savedConfig); }

    // GetAllStatsAndClear() is read-and-clear (C4): draining here gives every
    // test a clean baseline, so assertions below cover the increment only.
    void SetUp() override { Metrics::GetAllStatsAndClear(); }
};

TEST_F(UCDrampoolMetricsTest, SetupRegistersEveryStaticMetricName)
{
    for (const auto* name : kCounterNames) { Metrics::UpdateStats(name, 1); }
    for (const auto* name : kGaugeNames) { Metrics::UpdateStats(name, 0.5); }
    for (const auto* name : kHistogramNames) { Metrics::UpdateStats(name, 1.0); }

    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    const auto& gauges = std::get<1>(stats);
    const auto& histograms = std::get<2>(stats);
    for (const auto* name : kCounterNames) { EXPECT_EQ(counters.at(name), 1.0) << name; }
    for (const auto* name : kGaugeNames) { EXPECT_EQ(gauges.at(name), 0.5) << name; }
    for (const auto* name : kHistogramNames) {
        EXPECT_EQ(HistogramCount(histograms.at(name)), 1) << name;
    }
}

TEST_F(UCDrampoolMetricsTest, SetupRegistersBufferPoolUsageGaugePerSlotSize)
{
    EXPECT_EQ(BufferPoolUsageRatioName(4096), "drampool_buffer_pool_usage_ratio_4096");
    Metrics::UpdateStats(BufferPoolUsageRatioName(512), 0.25);
    Metrics::UpdateStats(BufferPoolUsageRatioName(4096), 0.25);
    // Slot sizes absent from g_config.poolBlockSizes are never registered (C2).
    Metrics::UpdateStats(BufferPoolUsageRatioName(8192), 0.9);

    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& gauges = std::get<1>(stats);
    EXPECT_EQ(gauges.at(BufferPoolUsageRatioName(512)), 0.25);
    EXPECT_EQ(gauges.at(BufferPoolUsageRatioName(4096)), 0.25);
    EXPECT_EQ(gauges.count(BufferPoolUsageRatioName(8192)), 0);
}

TEST_F(UCDrampoolMetricsTest, UnregisteredNamesAreSilentlyDropped)
{
    const std::string unregistered = "drampool_unregistered_metric";
    Metrics::UpdateStats(unregistered, 1.0);

    const auto stats = Metrics::GetAllStatsAndClear();
    EXPECT_TRUE(std::get<0>(stats).empty());
    EXPECT_TRUE(std::get<1>(stats).empty());
    EXPECT_TRUE(std::get<2>(stats).empty());
}

TEST_F(UCDrampoolMetricsTest, CounterUpdatesAccumulate)
{
    // Aggregated-count call sites guard count != 0 locally before updating.
    Metrics::UpdateStats(kDumpRequestsTotal, 3);
    Metrics::UpdateStats(kDumpRequestsTotal, 4);
    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.size(), std::size_t{1});
    EXPECT_EQ(counters.at(kDumpRequestsTotal), 7);
}

TEST_F(UCDrampoolMetricsTest, GaugeOverwritesWithLatestValue)
{
    // Mirrors the queue-length gauges maintained by task_worker and
    // completion_poller from the atomic queue length accountings.
    g_requestQueueLen.store(11);
    Metrics::UpdateStats(kQueueRequestSize, static_cast<double>(g_requestQueueLen.load()));
    g_requestQueueLen.store(3);
    Metrics::UpdateStats(kQueueRequestSize, static_cast<double>(g_requestQueueLen.load()));

    g_completionQueueLen.store(7);
    Metrics::UpdateStats(kQueueCompletionSize, static_cast<double>(g_completionQueueLen.load()));

    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& gauges = std::get<1>(stats);
    EXPECT_EQ(gauges.at(kQueueRequestSize), 3);
    EXPECT_EQ(gauges.at(kQueueCompletionSize), 7);
}

TEST_F(UCDrampoolMetricsTest, HistogramObservationsAccumulate)
{
    Metrics::UpdateStats(kDumpPrepareDurationMs, 1.5);
    Metrics::UpdateStats(kDumpPrepareDurationMs, 2.5);

    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& histogram = std::get<2>(stats).at(kDumpPrepareDurationMs);
    EXPECT_EQ(HistogramCount(histogram), 2);
    EXPECT_DOUBLE_EQ(histogram.sum, 4.0);
    // SetupDrampoolMetrics registers histograms without explicit buckets, so
    // every observation falls into the single catch-all (infinity) bucket.
    ASSERT_EQ(histogram.bucketCounts.size(), std::size_t{1});
    EXPECT_EQ(histogram.bucketCounts[0], 2);
}

TEST_F(UCDrampoolMetricsTest, ScopedTimerRecordsDurationOnScopeExit)
{
    {
        ScopedTimer timer(NAME_TO_METRIC_ID(kLoadPrepareDurationMs));
    }

    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& histogram = std::get<2>(stats).at(kLoadPrepareDurationMs);
    EXPECT_EQ(HistogramCount(histogram), 1);
    EXPECT_GE(histogram.sum, 0.0);
}

TEST_F(UCDrampoolMetricsTest, ScopedTimerDisarmSkipsObservation)
{
    {
        ScopedTimer timer(NAME_TO_METRIC_ID(kLoadPrepareDurationMs));
        timer.Disarm();
    }

    const auto stats = Metrics::GetAllStatsAndClear();
    EXPECT_TRUE(std::get<2>(stats).empty());
}

TEST_F(UCDrampoolMetricsTest, GetAllStatsAndClearReturnsIncrementAndResets)
{
    Metrics::UpdateStats(kLookupRequestsTotal, 2);
    Metrics::UpdateStats(kMetadataEntryCount, 5);
    Metrics::UpdateStats(kResponseRttMs, 1.0);

    const auto first = Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(first).at(kLookupRequestsTotal), 2);
    EXPECT_EQ(std::get<1>(first).at(kMetadataEntryCount), 5);
    EXPECT_EQ(HistogramCount(std::get<2>(first).at(kResponseRttMs)), 1);

    const auto second = Metrics::GetAllStatsAndClear();
    EXPECT_TRUE(std::get<0>(second).empty());
    EXPECT_TRUE(std::get<1>(second).empty());
    EXPECT_TRUE(std::get<2>(second).empty());
}

TEST_F(UCDrampoolMetricsTest, UpdatesFromMultipleThreadsAggregate)
{
    // C5: the first UpdateStats on a thread registers its thread-local buffer
    // and GetAllStatsAndClear() folds every registered buffer into the total.
    constexpr std::uint64_t kUpdatesPerThread = 100;
    const auto worker = [] {
        for (std::uint64_t index = 0; index < kUpdatesPerThread; ++index) {
            Metrics::UpdateStats(kDumpRequestsTotal, 1);
        }
    };
    std::thread first(worker);
    std::thread second(worker);
    first.join();
    second.join();

    const auto stats = Metrics::GetAllStatsAndClear();
    EXPECT_EQ(std::get<0>(stats).at(kDumpRequestsTotal), kUpdatesPerThread * 2);
}

TEST_F(UCDrampoolMetricsTest, SetupDrampoolMetricsIsIdempotent)
{
    EXPECT_NO_THROW(SetupDrampoolMetrics());
    Metrics::UpdateStats(kDumpRequestsTotal, 2);

    const auto stats = Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.size(), std::size_t{1});
    EXPECT_EQ(counters.at(kDumpRequestsTotal), 2);
}

}  // namespace
}  // namespace UC::DramPool
