/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache/v2/numa/numa_policy.h"
#include <gtest/gtest.h>
#include <vector>

namespace UC::Cache2::Numa {
namespace {

TEST(NumaPolicyTest, SlicesSixteenRanksAcrossEightNodes)
{
    const std::vector<size_t> expected{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
    for (size_t rank = 0; rank < expected.size(); ++rank) {
        auto index = RankSliceIndex(rank, 16, 8);
        ASSERT_TRUE(index) << index.Error().ToString();
        EXPECT_EQ(expected[rank], index.Value()) << "rank=" << rank;
    }
}

TEST(NumaPolicyTest, MapsDp2Tp8WhenPlacementCountCoversAllLocalWorkers)
{
    constexpr size_t dpSize = 2;
    constexpr size_t tpSize = 8;
    constexpr size_t nodeCount = 8;
    constexpr size_t localWorkerCount = dpSize * tpSize;

    for (size_t dpRank = 0; dpRank < dpSize; ++dpRank) {
        for (size_t tpRank = 0; tpRank < tpSize; ++tpRank) {
            const size_t fallbackRank = dpRank * tpSize + tpRank;
            const size_t expectedNode = dpRank * (nodeCount / dpSize) + tpRank / 2;
            auto index = RankSliceIndex(fallbackRank, localWorkerCount, nodeCount);
            ASSERT_TRUE(index) << "dp=" << dpRank << " tp=" << tpRank << " "
                               << index.Error().ToString();
            EXPECT_EQ(expectedNode, index.Value()) << "dp=" << dpRank << " tp=" << tpRank;
        }
    }
}

TEST(NumaPolicyTest, SeparateTp8DomainsCannotUseGlobalDpFallbackRank)
{
    constexpr size_t tpSize = 8;
    constexpr size_t nodeCount = 8;

    /* DP0's global fallback ranks happen to fit an eight-rank Cache domain,
     * but they spread one worker per node rather than two workers per node. */
    for (size_t tpRank = 0; tpRank < tpSize; ++tpRank) {
        auto index = RankSliceIndex(tpRank, tpSize, nodeCount);
        ASSERT_TRUE(index) << index.Error().ToString();
        EXPECT_EQ(tpRank, index.Value()) << "tp=" << tpRank;
    }

    /* DP1 carries machine-global fallback ranks 8..15.  Those ranks are out
     * of range when the Cache domain itself advertises only eight ranks. */
    for (size_t tpRank = 0; tpRank < tpSize; ++tpRank) {
        const size_t fallbackRank = tpSize + tpRank;
        EXPECT_FALSE(RankSliceIndex(fallbackRank, tpSize, nodeCount)) << "tp=" << tpRank;
    }
}

TEST(NumaPolicyTest, DistributesUnevenRankCountContiguously)
{
    const std::vector<size_t> expected{0, 0, 0, 0, 1, 1, 1, 2, 2, 2};
    for (size_t rank = 0; rank < expected.size(); ++rank) {
        auto index = RankSliceIndex(rank, 10, 3);
        ASSERT_TRUE(index) << index.Error().ToString();
        EXPECT_EQ(expected[rank], index.Value()) << "rank=" << rank;
    }
}

TEST(NumaPolicyTest, RejectsInvalidGeometry)
{
    EXPECT_FALSE(RankSliceIndex(0, 0, 8));
    EXPECT_FALSE(RankSliceIndex(0, 16, 0));
    EXPECT_FALSE(RankSliceIndex(16, 16, 8));
}

}  // namespace
}  // namespace UC::Cache2::Numa
