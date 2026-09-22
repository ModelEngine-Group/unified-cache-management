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
#include <functional>
#include <gtest/gtest.h>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>
#include "cache2/cc/trans_manager.h"
#include "detail/types_helper.h"
#include "metrics_api.h"

namespace {

using UC::Cache2::Task;
using UC::Cache2::TaskIdSet;
using UC::Cache2::TaskPtr;
using UC::Cache2::WaiterPtr;
using UC::Test::Detail::TypesHelper;

template <typename Tag>
class FakeQueue {
public:
    static inline TaskIdSet* lastFailureSet{nullptr};
    static inline UC::Cache2::Buffer* lastBuffer{nullptr};
    static inline std::vector<std::pair<TaskPtr, WaiterPtr>> submitted;
    static inline std::function<void(const TaskPtr&, const WaiterPtr&)> action;

    UC::Status Setup(const UC::Cache2::Config&, TaskIdSet* failureSet, UC::Cache2::Buffer* buffer)
    {
        lastFailureSet = failureSet;
        lastBuffer = buffer;
        return UC::Status::OK();
    }
    void Submit(TaskPtr task, WaiterPtr waiter)
    {
        submitted.emplace_back(std::move(task), std::move(waiter));
        if (action) { action(submitted.back().first, submitted.back().second); }
    }
    static void Reset()
    {
        lastFailureSet = nullptr;
        lastBuffer = nullptr;
        submitted.clear();
        action = nullptr;
    }
};

struct LoadTag {};
struct DumpTag {};
using FakeLoadQueue = FakeQueue<LoadTag>;
using FakeDumpQueue = FakeQueue<DumpTag>;

using TestedManager = UC::Cache2::TransManager<FakeLoadQueue, FakeDumpQueue>;

UC::Detail::TaskDesc MakeDesc(size_t shards)
{
    UC::Detail::TaskDesc desc;
    desc.brief = "test";
    for (size_t i = 0; i < shards; ++i) {
        desc.push_back({TypesHelper::MakeBlockIdRandomly(), i, {}});
    }
    return desc;
}

std::uint64_t HistogramCount(const UC::Metrics::HistogramStat& histogram)
{
    return std::accumulate(histogram.bucketCounts.begin(), histogram.bucketCounts.end(),
                           std::uint64_t{0});
}

class UCCache2TransManagerTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        UC::Metrics::SetUp();
        UC::Metrics::CreateStats("cache_load_duration_ms", "histogram", {0.1, 1, 100, 500, 5000});
        UC::Metrics::CreateStats("cache_dump_duration_ms", "histogram", {0.1, 1, 100, 500, 5000});
        UC::Metrics::CreateStats("cache_load_bandwidth_gbps", "histogram", {1, 10, 100, 1000});
        UC::Metrics::CreateStats("cache_dump_bandwidth_gbps", "histogram", {1, 10, 100, 1000});
        UC::Metrics::CreateStats("cache_load_bytes_total", "counter");
        UC::Metrics::CreateStats("cache_dump_bytes_total", "counter");
        UC::Metrics::CreateStats("cache_dump_shards_total", "counter");
    }
    void SetUp() override
    {
        FakeLoadQueue::Reset();
        FakeDumpQueue::Reset();
        UC::Metrics::GetAllStatsAndClear();
        UC::Cache2::Config config;
        config.timeoutMs = 30000;
        config.shardSize = 4096;
        ASSERT_TRUE(mgr_.Setup(config, &buffer_).Success());
    }

    UC::Cache2::Buffer buffer_;
    TestedManager mgr_;
};

TEST_F(UCCache2TransManagerTest, SetupWiresQueuesAndFailureSet)
{
    EXPECT_NE(FakeLoadQueue::lastFailureSet, nullptr);
    EXPECT_EQ(FakeLoadQueue::lastFailureSet, FakeDumpQueue::lastFailureSet);
    EXPECT_EQ(FakeLoadQueue::lastBuffer, &buffer_);
    EXPECT_EQ(FakeDumpQueue::lastBuffer, &buffer_);
}

TEST_F(UCCache2TransManagerTest, LoadTaskIsRoutedToLoadQueue)
{
    const auto block = TypesHelper::MakeBlockId("a1b2c3d4e5f6789012345678901234ab");
    UC::Detail::TaskDesc desc;
    desc.brief = "load-test";
    desc.push_back({block, 0, {}});
    auto handle = mgr_.Submit({Task::Type::LOAD, std::move(desc)});
    ASSERT_TRUE(handle.HasValue());
    ASSERT_EQ(FakeLoadQueue::submitted.size(), 1);
    ASSERT_EQ(FakeDumpQueue::submitted.size(), 0);
    const auto& task = FakeLoadQueue::submitted.front().first;
    EXPECT_EQ(task->id, handle.Value());
    EXPECT_EQ(task->type, Task::Type::LOAD);
    ASSERT_EQ(task->desc.size(), 1);
    EXPECT_EQ(task->desc.brief, "load-test");
    EXPECT_EQ(task->desc[0].owner, block);
    EXPECT_EQ(task->desc[0].index, 0);
}

TEST_F(UCCache2TransManagerTest, DumpTaskIsRoutedToDumpQueue)
{
    auto handle = mgr_.Submit({Task::Type::DUMP, MakeDesc(1)});
    ASSERT_TRUE(handle.HasValue());
    ASSERT_EQ(FakeDumpQueue::submitted.size(), 1);
    ASSERT_EQ(FakeLoadQueue::submitted.size(), 0);
    EXPECT_EQ(FakeDumpQueue::submitted.front().first->type, Task::Type::DUMP);
}

TEST_F(UCCache2TransManagerTest, WaitSucceedsAfterQueueCompletes)
{
    FakeLoadQueue::action = [](const TaskPtr&, const WaiterPtr& waiter) { waiter->Up(); };
    auto handle = mgr_.Submit({Task::Type::LOAD, MakeDesc(1)});
    ASSERT_TRUE(handle.HasValue());
    EXPECT_FALSE(mgr_.Check(handle.Value()).Value());
    for (auto& [task, waiter] : FakeLoadQueue::submitted) { waiter->Done(); }
    EXPECT_TRUE(mgr_.Check(handle.Value()).Value());
    EXPECT_EQ(mgr_.Wait(handle.Value()), UC::Status::OK());
}

TEST_F(UCCache2TransManagerTest, WaitReturnsErrorWhenQueueMarksFailure)
{
    FakeLoadQueue::action = [](const TaskPtr& task, const WaiterPtr& waiter) {
        FakeLoadQueue::lastFailureSet->Insert(task->id);
        waiter->Up();
        waiter->Done();
    };
    auto handle = mgr_.Submit({Task::Type::LOAD, MakeDesc(1)});
    ASSERT_TRUE(handle.HasValue());
    EXPECT_EQ(mgr_.Wait(handle.Value()), UC::Status::Error());
}

TEST_F(UCCache2TransManagerTest, LoadEpilogMetrics)
{
    constexpr size_t shards = 3;
    FakeLoadQueue::action = [](const TaskPtr&, const WaiterPtr& waiter) {
        waiter->Up();
        waiter->Done();
    };
    auto handle = mgr_.Submit({Task::Type::LOAD, MakeDesc(shards)});
    ASSERT_TRUE(handle.HasValue());
    EXPECT_EQ(mgr_.Wait(handle.Value()), UC::Status::OK());
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    const auto& histograms = std::get<2>(stats);
    EXPECT_EQ(HistogramCount(histograms.at("cache_load_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_load_bandwidth_gbps")), 1);
    EXPECT_EQ(counters.at("cache_load_bytes_total"), 4096 * shards);
    EXPECT_EQ(histograms.count("cache_dump_duration_ms"), 0);
    EXPECT_EQ(histograms.count("cache_dump_bandwidth_gbps"), 0);
    EXPECT_EQ(counters.count("cache_dump_bytes_total"), 0);
    EXPECT_EQ(counters.count("cache_dump_shards_total"), 0);
}

TEST_F(UCCache2TransManagerTest, DumpEpilogMetrics)
{
    constexpr size_t shards = 2;
    FakeDumpQueue::action = [](const TaskPtr&, const WaiterPtr& waiter) {
        waiter->Up();
        waiter->Done();
    };
    auto handle = mgr_.Submit({Task::Type::DUMP, MakeDesc(shards)});
    ASSERT_TRUE(handle.HasValue());
    EXPECT_EQ(mgr_.Wait(handle.Value()), UC::Status::OK());
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    const auto& histograms = std::get<2>(stats);
    EXPECT_EQ(HistogramCount(histograms.at("cache_dump_duration_ms")), 1);
    EXPECT_EQ(HistogramCount(histograms.at("cache_dump_bandwidth_gbps")), 1);
    EXPECT_EQ(counters.at("cache_dump_shards_total"), shards);
    EXPECT_EQ(counters.at("cache_dump_bytes_total"), 4096 * shards);
    EXPECT_EQ(histograms.count("cache_load_duration_ms"), 0);
    EXPECT_EQ(histograms.count("cache_load_bandwidth_gbps"), 0);
    EXPECT_EQ(counters.count("cache_load_bytes_total"), 0);
}

TEST_F(UCCache2TransManagerTest, WaitTimesOutWhenQueueStalls)
{
    TestedManager mgr;
    UC::Cache2::Config config;
    config.timeoutMs = 50;
    config.shardSize = 4096;
    ASSERT_TRUE(mgr.Setup(config, nullptr).Success());
    FakeLoadQueue::action = [](const TaskPtr&, const WaiterPtr& waiter) { waiter->Up(); };
    auto handle = mgr.Submit({Task::Type::LOAD, MakeDesc(1)});
    ASSERT_TRUE(handle.HasValue());
    std::thread completer([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        for (auto& [task, waiter] : FakeLoadQueue::submitted) { waiter->Done(); }
    });
    EXPECT_EQ(mgr.Wait(handle.Value()), UC::Status::Timeout());
    completer.join();
}

}  // namespace
