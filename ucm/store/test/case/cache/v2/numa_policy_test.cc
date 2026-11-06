/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache/v2/numa/numa_policy.h"
#include <cerrno>
#include <gtest/gtest.h>
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include "ipc/mem_fd.h"

namespace UC::Cache2::Numa {
namespace {

class ThreadPolicyTest : public ::testing::Test {
protected:
    // The kernel rejects a userspace mask larger than one 4KiB page.
    static constexpr size_t kMaskBits = 32768;
    int originalMode_{};
    std::vector<unsigned long> originalMask_ =
        std::vector<unsigned long>(kMaskBits / (sizeof(unsigned long) * 8), 0);
    Plan plan_;

    void SetUp() override
    {
        if (::syscall(SYS_get_mempolicy, &originalMode_, originalMask_.data(), kMaskBits, nullptr,
                      0UL) != 0) {
            if (errno == EPERM) { GTEST_SKIP() << "NUMA policy syscalls are restricted"; }
            FAIL() << "get_mempolicy failed: errno=" << errno;
        }
        auto plan = Resolve(std::nullopt, 0, 1);
        ASSERT_TRUE(plan) << plan.Error().ToString();
        plan_ = plan.Value();
    }

    void ExpectPolicy(int expectedMode, const std::vector<unsigned long>& expectedMask)
    {
        int mode = -1;
        std::vector<unsigned long> mask(kMaskBits / (sizeof(unsigned long) * 8), 0);
        ASSERT_EQ(0, ::syscall(SYS_get_mempolicy, &mode, mask.data(), kMaskBits, nullptr, 0UL));
        EXPECT_EQ(expectedMode, mode);
        EXPECT_EQ(expectedMask, mask);
    }
};

TEST_F(ThreadPolicyTest, ExplicitRestorePreservesOriginalPolicy)
{
    ScopedThreadPolicy policy;
    auto status = policy.Bind(plan_);
    ASSERT_TRUE(status.Success()) << status.ToString();
    std::vector<unsigned long> mask(originalMask_.size(), 0);
    mask[plan_.node / (sizeof(unsigned long) * 8)] = 1UL
                                                     << (plan_.node % (sizeof(unsigned long) * 8));
    ExpectPolicy(MPOL_BIND | MPOL_F_STATIC_NODES, mask);
    EXPECT_TRUE(policy.Bind(plan_).Failure());
    ASSERT_TRUE(policy.Restore().Success());
    ExpectPolicy(originalMode_, originalMask_);
    EXPECT_TRUE(policy.Restore().Success());
}

TEST_F(ThreadPolicyTest, ScopedExitRestoresNonDefaultPolicy)
{
    ScopedThreadPolicy outer;
    ASSERT_TRUE(outer.Bind(plan_).Success());
    int boundMode = -1;
    std::vector<unsigned long> boundMask(originalMask_.size(), 0);
    ASSERT_EQ(0,
              ::syscall(SYS_get_mempolicy, &boundMode, boundMask.data(), kMaskBits, nullptr, 0UL));
    {
        ScopedThreadPolicy inner;
        ASSERT_TRUE(inner.Bind(plan_).Success());
    }
    ExpectPolicy(boundMode, boundMask);
    ASSERT_TRUE(outer.Restore().Success());
    ExpectPolicy(originalMode_, originalMask_);
}

TEST_F(ThreadPolicyTest, PreallocationFailureRestoresOriginalPolicy)
{
    const auto pageSize = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    const int fd = ::memfd_create("ucm_policy_failure", MFD_ALLOW_SEALING | MFD_CLOEXEC);
    ASSERT_GE(fd, 0);
    MemFd mem;
    ASSERT_TRUE(mem.Adopt(fd, pageSize * 2).Success());
    ASSERT_EQ(0, ::ftruncate(fd, static_cast<off_t>(pageSize)));
    ASSERT_EQ(0, ::fcntl(fd, F_ADD_SEALS, F_SEAL_GROW));
    ScopedThreadPolicy policy;
    ASSERT_TRUE(policy.Bind(plan_).Success());
    EXPECT_TRUE(mem.Preallocate().Failure());
    ASSERT_TRUE(policy.Restore().Success());
    ExpectPolicy(originalMode_, originalMask_);
}

TEST_F(ThreadPolicyTest, InvalidBindingLeavesPolicyUnchanged)
{
    ScopedThreadPolicy policy;
    EXPECT_TRUE(policy.Bind(Plan{}).Failure());
    ExpectPolicy(originalMode_, originalMask_);
}

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
