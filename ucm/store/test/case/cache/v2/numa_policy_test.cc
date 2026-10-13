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
