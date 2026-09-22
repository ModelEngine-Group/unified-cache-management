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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include <sys/types.h>
#include <vector>
#include "cache2/cc/buffer_manager.h"
#include "detail/mock_store.h"
#include "detail/types_helper.h"
#include "metrics_api.h"

namespace {

using UC::Test::Detail::MockStore;
using UC::Test::Detail::TypesHelper;

class FakeBuffer {
public:
    UC::Status Setup(const UC::Cache2::Config&) { return UC::Status::OK(); }
    bool Exist(const UC::Detail::BlockId& blockId, size_t) const
    {
        return hits_.count(blockId) > 0;
    }
    void Touch(const UC::Detail::BlockId* blocks, size_t num)
    {
        touched_.assign(blocks, blocks + num);
    }
    void SetHits(const std::vector<UC::Detail::BlockId>& blocks)
    {
        hits_.clear();
        hits_.insert(blocks.begin(), blocks.end());
    }
    const std::vector<UC::Detail::BlockId>& Touched() const { return touched_; }

private:
    std::set<UC::Detail::BlockId> hits_;
    std::vector<UC::Detail::BlockId> touched_;
};

using TestedManager = UC::Cache2::BufferManager<FakeBuffer>;

std::vector<UC::Detail::BlockId> MakeBlocks(size_t num)
{
    std::vector<UC::Detail::BlockId> blocks(num);
    for (auto& block : blocks) { block = TypesHelper::MakeBlockIdRandomly(); }
    return blocks;
}

std::uint64_t HistogramCount(const UC::Metrics::HistogramStat& histogram)
{
    return std::accumulate(histogram.bucketCounts.begin(), histogram.bucketCounts.end(),
                           std::uint64_t{0});
}

class UCCache2BufferManagerTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        UC::Metrics::SetUp();
        UC::Metrics::CreateStats("cache_lookup_duration_ms", "histogram", {0.1, 1, 100, 5000});
        UC::Metrics::CreateStats("cache_lookup_backend_duration_ms", "histogram",
                                 {0.1, 1, 100, 5000});
        UC::Metrics::CreateStats("cache_lookup_hit_blocks_total", "counter");
        UC::Metrics::CreateStats("cache_lookup_miss_blocks_total", "counter");
    }
    void SetUp() override
    {
        UC::Metrics::GetAllStatsAndClear();
        UC::Cache2::Config config;
        config.storeBackend = &backend_;
        ASSERT_TRUE(mgr_.Setup(config).Success());
    }

    MockStore backend_;
    TestedManager mgr_;
};

TEST_F(UCCache2BufferManagerTest, LookupOnPrefixAllBufferHitsSkipsBackend)
{
    auto blocks = MakeBlocks(3);
    mgr_.GetTransBuffer()->SetHits(blocks);
    EXPECT_CALL(backend_, LookupOnPrefix).Times(0);
    EXPECT_EQ(mgr_.LookupOnPrefix(blocks.data(), blocks.size()).Value(), 2);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    const auto& histograms = std::get<2>(stats);
    EXPECT_EQ(counters.at("cache_lookup_hit_blocks_total"), 3);
    EXPECT_EQ(counters.at("cache_lookup_miss_blocks_total"), 0);
    EXPECT_EQ(HistogramCount(histograms.at("cache_lookup_duration_ms")), 1);
    EXPECT_EQ(histograms.count("cache_lookup_backend_duration_ms"), 0);
}

TEST_F(UCCache2BufferManagerTest, LookupOnPrefixMapsBackendResultThroughMissIndices)
{
    auto blocks = MakeBlocks(5);
    mgr_.GetTransBuffer()->SetHits({blocks[1], blocks[2]});
    EXPECT_CALL(backend_, LookupOnPrefix)
        .WillOnce(testing::Invoke([&](const UC::Detail::BlockId* missBlocks, size_t num) {
            EXPECT_EQ(num, 3);
            EXPECT_THAT((std::vector<UC::Detail::BlockId>(missBlocks, missBlocks + num)),
                        testing::ElementsAre(blocks[0], blocks[3], blocks[4]));
            return static_cast<ssize_t>(0);
        }));
    EXPECT_EQ(mgr_.LookupOnPrefix(blocks.data(), blocks.size()).Value(), 2);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_lookup_hit_blocks_total"), 2);
    EXPECT_EQ(counters.at("cache_lookup_miss_blocks_total"), 3);
}

TEST_F(UCCache2BufferManagerTest, LookupOnPrefixBackendFindsAllMisses)
{
    auto blocks = MakeBlocks(3);
    mgr_.GetTransBuffer()->SetHits({blocks[1]});
    EXPECT_CALL(backend_, LookupOnPrefix).WillOnce(testing::Invoke([](auto, size_t num) {
        return static_cast<ssize_t>(num) - 1;
    }));
    EXPECT_EQ(mgr_.LookupOnPrefix(blocks.data(), blocks.size()).Value(), 2);
}

TEST_F(UCCache2BufferManagerTest, LookupOnPrefixBackendFindsNone)
{
    auto blocks = MakeBlocks(3);
    mgr_.GetTransBuffer()->SetHits({blocks[0]});
    EXPECT_CALL(backend_, LookupOnPrefix).WillOnce(testing::Return(-1));
    EXPECT_EQ(mgr_.LookupOnPrefix(blocks.data(), blocks.size()).Value(), 0);
}

TEST_F(UCCache2BufferManagerTest, LookupOnReverseTailBufferHitSkipsBackend)
{
    auto blocks = MakeBlocks(3);
    mgr_.GetTransBuffer()->SetHits({blocks[2]});
    EXPECT_CALL(backend_, LookupOnReverse).Times(0);
    EXPECT_EQ(mgr_.LookupOnReverse(blocks.data(), blocks.size()).Value(), 2);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_lookup_hit_blocks_total"), 3);
    EXPECT_EQ(counters.at("cache_lookup_miss_blocks_total"), 0);
}

TEST_F(UCCache2BufferManagerTest, LookupOnReverseQueriesBackendOnlyForTailMisses)
{
    auto blocks = MakeBlocks(5);
    mgr_.GetTransBuffer()->SetHits({blocks[1]});
    EXPECT_CALL(backend_, LookupOnReverse)
        .WillOnce(testing::Invoke([&](const UC::Detail::BlockId* missBlocks, size_t num) {
            EXPECT_EQ(num, 3);
            EXPECT_THAT((std::vector<UC::Detail::BlockId>(missBlocks, missBlocks + num)),
                        testing::ElementsAre(blocks[2], blocks[3], blocks[4]));
            return static_cast<ssize_t>(2);
        }));
    EXPECT_EQ(mgr_.LookupOnReverse(blocks.data(), blocks.size()).Value(), 4);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    const auto& histograms = std::get<2>(stats);
    EXPECT_EQ(counters.at("cache_lookup_hit_blocks_total"), 2);
    EXPECT_EQ(counters.at("cache_lookup_miss_blocks_total"), 3);
    EXPECT_EQ(HistogramCount(histograms.at("cache_lookup_backend_duration_ms")), 1);
}

TEST_F(UCCache2BufferManagerTest, LookupOnReverseBackendFindsNone)
{
    auto blocks = MakeBlocks(3);
    mgr_.GetTransBuffer()->SetHits({blocks[0]});
    EXPECT_CALL(backend_, LookupOnReverse).WillOnce(testing::Return(-1));
    EXPECT_EQ(mgr_.LookupOnReverse(blocks.data(), blocks.size()).Value(), 0);
}

TEST_F(UCCache2BufferManagerTest, LookupOnReverseWithoutBufferHitsPassesAllBlocks)
{
    auto blocks = MakeBlocks(5);
    EXPECT_CALL(backend_, LookupOnReverse).WillOnce(testing::Return(2));
    EXPECT_EQ(mgr_.LookupOnReverse(blocks.data(), blocks.size()).Value(), 2);
    auto stats = UC::Metrics::GetAllStatsAndClear();
    const auto& counters = std::get<0>(stats);
    EXPECT_EQ(counters.at("cache_lookup_hit_blocks_total"), 0);
    EXPECT_EQ(counters.at("cache_lookup_miss_blocks_total"), 5);
}

TEST_F(UCCache2BufferManagerTest, LookupPropagatesBackendError)
{
    auto blocks = MakeBlocks(2);
    auto error = UC::Expected<ssize_t>(UC::Status::Error("backend failure"));
    EXPECT_CALL(backend_, LookupOnPrefix).WillOnce(testing::Return(error));
    EXPECT_CALL(backend_, LookupOnReverse).WillOnce(testing::Return(error));
    auto prefixRes = mgr_.LookupOnPrefix(blocks.data(), blocks.size());
    ASSERT_FALSE(prefixRes);
    EXPECT_TRUE(prefixRes.Error().Failure());
    auto reverseRes = mgr_.LookupOnReverse(blocks.data(), blocks.size());
    ASSERT_FALSE(reverseRes);
    EXPECT_TRUE(reverseRes.Error().Failure());
}

TEST_F(UCCache2BufferManagerTest, LookupWithoutBackendUsesBufferOnly)
{
    TestedManager mgr;
    UC::Cache2::Config config;
    ASSERT_TRUE(mgr.Setup(config).Success());
    auto blocks = MakeBlocks(3);
    mgr.GetTransBuffer()->SetHits({blocks[0], blocks[1]});
    EXPECT_EQ(mgr.LookupOnPrefix(blocks.data(), blocks.size()).Value(), 1);
    EXPECT_EQ(mgr.LookupOnReverse(blocks.data(), blocks.size()).Value(), 1);
    mgr.GetTransBuffer()->SetHits({});
    EXPECT_EQ(mgr.LookupOnPrefix(blocks.data(), blocks.size()).Value(), -1);
    EXPECT_EQ(mgr.LookupOnReverse(blocks.data(), blocks.size()).Value(), -1);
}

TEST_F(UCCache2BufferManagerTest, PrefetchForwardsToBufferAndBackend)
{
    auto blocks = MakeBlocks(2);
    EXPECT_CALL(backend_, Prefetch(blocks.data(), blocks.size())).Times(1);
    mgr_.Prefetch(blocks.data(), blocks.size());
    EXPECT_EQ(mgr_.GetTransBuffer()->Touched(), blocks);

    TestedManager mgr;
    UC::Cache2::Config config;
    ASSERT_TRUE(mgr.Setup(config).Success());
    mgr.Prefetch(blocks.data(), blocks.size());
    EXPECT_EQ(mgr.GetTransBuffer()->Touched(), blocks);
}

TEST_F(UCCache2BufferManagerTest, LookupWithZeroBlocks)
{
    EXPECT_CALL(backend_, LookupOnPrefix).Times(0);
    EXPECT_CALL(backend_, LookupOnReverse).Times(0);
    EXPECT_EQ(mgr_.LookupOnPrefix(nullptr, 0).Value(), -1);
    EXPECT_EQ(mgr_.LookupOnReverse(nullptr, 0).Value(), -1);
}

}  // namespace
